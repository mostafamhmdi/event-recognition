"""
event_pipeline_automation.py
=============================
Turns the manual workflow -

    run main.py for Telegram
    run main.py for Twitter/X
    every so often, run candidates_aggregator.py

- into a self-scheduling, crash-safe, restart-safe automation wrapper.

This script does NOT modify main.py or candidates_aggregator.py in any
way. It only calls them exactly the way you would from the command line
(as subprocesses), and adds the bookkeeping (which calendar day is next,
have we run today yet, has the 3-day counter for the aggregator come
around) that turns "run this by hand" into "runs unattended forever".

--------------------------------------------------------------------------
WHAT HAPPENS ON EACH SCHEDULED TICK
--------------------------------------------------------------------------
For each platform (Telegram, Twitter/X) independently, this finds the
next Jalali calendar day that platform hasn't processed yet and runs:

    python main.py --db-name <...> --table-name <...> --date-col <...>
                    --start-date <D> --end-date <D+1> ...

Once BOTH platforms have successfully finished the SAME calendar day,
that counts as one completed "pipeline day". Every AGGREGATOR_EVERY_N_DAYS
pipeline days (3 by default), it runs:

    python candidates_aggregator.py

with no date arguments at all - that script finds its own scan window
from the watermark already stored in cluster_stories (see that file's
own docstring/_get_scan_start_date), so it never needs to be told a date
range from the outside.

If a platform's run for a given day fails, that platform's watermark is
NOT advanced, so the same day is retried automatically on the next tick
(or immediately, on the next catch-up iteration) - nothing is silently
skipped, and nothing is double-submitted (main.py has no idempotency
guard of its own, so this wrapper's own watermark is what guarantees
"exactly once per calendar day per platform").

--------------------------------------------------------------------------
BOOTSTRAPPING - "WHERE DO WE START FROM?"
--------------------------------------------------------------------------
Each platform has its own START_DATE_JALALI constant below (e.g. "from
yesterday" or "from last Shahrivar" - any Jalali date works identically).
That constant is read EXACTLY ONCE: the very first time this script runs
and finds no state file yet. From that point on, the on-disk state file
(STATE_FILE) is the only source of truth for "what's the next day to
process" - the constant is never consulted again, so it's safe to leave
it in the code forever without it causing reprocessing.

Whatever START_DATE_JALALI you pick, the very same code path handles it:
on the first invocation, this walks forward one Jalali day at a time from
START_DATE_JALALI up to yesterday (today is deliberately never processed -
today's data is still arriving), running main.py for each day for each
platform, saving progress to disk after every single successful day. If
that backlog is one day ("start from yesterday") this finishes in one
short run. If it's several months ("start from last Shahrivar"), the
first run will simply take a long time (each day reloads the GPU models,
by main.py's own design) - but because progress is saved incrementally,
you can safely stop it (Ctrl+C / SIGTERM) and restart later; it resumes
exactly where it left off instead of starting over. Once caught up, every
later tick only ever has (at most) one new day to do, so steady-state
behaviour is "process yesterday, once a day".

--------------------------------------------------------------------------
RUNNING IT
--------------------------------------------------------------------------
Two ways to use this, pick whichever fits your ops setup - both are safe
to use together (see the lock-file note below):

  A) Cron-friendly, one tick then exit:
       0 3 * * *  cd /path/to/pipeline && /usr/bin/python3 event_pipeline_automation.py --once >> cron.log 2>&1

  B) Self-contained daemon (no cron needed), keep it running e.g. under
     systemd or `nohup ... &`:
       python3 event_pipeline_automation.py --loop

     In this mode the script schedules its own next run at DAILY_RUN_HOUR
     each day (persisted on disk, so a host reboot doesn't lose the
     schedule) and sleeps in between, waking up periodically to check.

Either way, `--status` prints the current state (next date per platform,
pipeline-day counter, last run times) without doing anything.

A simple PID lock file (LOCK_FILE) prevents two invocations from running
at the same time (e.g. cron firing while a long catch-up run from --loop
is still in progress) - a second invocation just logs and exits instead
of racing the first one's state file.
"""

import os
import sys
import json
import time
import signal
import argparse
import subprocess
from datetime import datetime, timedelta

import jdatetime
import psutil


PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_SCRIPT = os.path.join(PIPELINE_DIR, "main.py")
AGGREGATOR_SCRIPT = os.path.join(PIPELINE_DIR, "candidates_aggregator.py")
PYTHON_BIN = sys.executable  # same interpreter/venv this wrapper is running under

STATE_FILE = os.path.join(PIPELINE_DIR, "automation_state.json")
LOCK_FILE = os.path.join(PIPELINE_DIR, "automation.lock")
LOG_DIR = os.path.join(PIPELINE_DIR, "automation_logs")


PLATFORMS = [
    {
        "name": "telegram",
        "db_name": "telegram",
        "table_name": "posts",
        "text_col": "txtContent",
        "date_col": "shdate",
        "source_name": "telegram",
        "start_date_jalali": "1404-05-01",  
        "extra_args": [],
    },
    {
        "name": "twitter",
        "db_name": "x",
        "table_name": "tweets_2",
        "text_col": "txtContent",
        "date_col": "shdate",
        "source_name": "x",
        "start_date_jalali": "1404-05-01",  
        "extra_args": [],
    },
]

# Args identical for every platform's main.py invocation. Leave empty to
# just use main.py's own defaults for everything else (qwen model path,
# location classifier path, qwen sample size, etc.)
COMMON_MAIN_ARGS = []

# How often (in completed "pipeline days" - see module docstring) to run
# candidates_aggregator.py.
AGGREGATOR_EVERY_N_DAYS = 3

# Never process "today" - only up to (today - PROCESS_UP_TO_DAYS_AGO).
# 1 means "always stop at yesterday", which is the standard assumption
# for a source table that's still receiving today's messages.
PROCESS_UP_TO_DAYS_AGO = 1

# Retries for a single day's main.py call before giving up for this tick
# (transient DB/network hiccups, not a real fix for a structural problem).
MAX_ATTEMPTS_PER_DAY = 2
RETRY_SLEEP_SECONDS = 30

# Safety cap on how many days one single invocation will catch up in a
# row before returning control (state is saved after every day either
# way, so this only affects how "chunky" a catch-up run is, not
# correctness). None = no cap, catch up fully in one go.
MAX_DAYS_PER_INVOCATION = None

# --loop mode only: local clock hour (0-23) each day's tick is scheduled for.
DAILY_RUN_HOUR = 3
DAILY_RUN_MINUTE = 0
# How often --loop wakes up to check whether it's time yet.
LOOP_CHECK_INTERVAL_SECONDS = 5 * 60


# ==========================================================================
# logging
# ==========================================================================

os.makedirs(LOG_DIR, exist_ok=True)


def log(msg: str):
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Automation] {msg}"
    print(line, flush=True)


# ==========================================================================
# state file (the wrapper's own watermark - separate from, and in addition
# to, candidates_aggregator.py's own internal DB watermark)
# ==========================================================================

def _default_state():
    return {
        "platforms": {
            p["name"]: {"next_date_jalali": p["start_date_jalali"], "last_success_at": None}
            for p in PLATFORMS
        },
        "pipeline_day_counter": 0,
        "last_aggregator_run_at": None,
        "next_run_at": None,  # only meaningful in --loop mode
    }


def load_state():
    if not os.path.exists(STATE_FILE):
        state = _default_state()
        save_state(state)
        log(f"No state file found - bootstrapped a new one at {STATE_FILE} "
            f"using each platform's start_date_jalali.")
        return state

    with open(STATE_FILE, "r", encoding="utf-8") as f:
        state = json.load(f)

    # If a platform was added to PLATFORMS after the state file already
    # existed, seed it instead of crashing.
    for p in PLATFORMS:
        state["platforms"].setdefault(
            p["name"], {"next_date_jalali": p["start_date_jalali"], "last_success_at": None}
        )
    return state


def save_state(state: dict):
    # Write to a temp file then atomically replace, so a crash mid-write
    # can never leave a half-written/corrupt state file behind.
    tmp_path = STATE_FILE + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, STATE_FILE)


# ==========================================================================
# lock file (prevent two overlapping invocations)
# ==========================================================================

def acquire_lock():
    if os.path.exists(LOCK_FILE):
        try:
            with open(LOCK_FILE, "r") as f:
                old_pid = int(f.read().strip())
        except (ValueError, OSError):
            old_pid = None

        if old_pid and psutil.pid_exists(old_pid):
            log(f"Another instance appears to be running (pid {old_pid}) - skipping this run.")
            return False
        else:
            log("Found a stale lock file (owning process is gone) - removing it.")
            os.remove(LOCK_FILE)

    with open(LOCK_FILE, "w") as f:
        f.write(str(os.getpid()))
    return True


def release_lock():
    if os.path.exists(LOCK_FILE):
        os.remove(LOCK_FILE)


# ==========================================================================
# Jalali date helpers
# ==========================================================================

def jalali_str_to_date(s: str) -> jdatetime.date:
    y, m, d = map(int, s.split("-"))
    return jdatetime.date(y, m, d)


def jalali_date_to_str(d: jdatetime.date) -> str:
    return f"{d.year:04d}-{d.month:02d}-{d.day:02d}"


def jalali_add_days(s: str, n: int) -> str:
    return jalali_date_to_str(jalali_str_to_date(s) + timedelta(days=n))


def latest_processable_jalali_day() -> str:
    return jalali_date_to_str(jdatetime.date.today() - timedelta(days=PROCESS_UP_TO_DAYS_AGO))


# ==========================================================================
# subprocess runners
# ==========================================================================

def _run_subprocess(cmd: list, log_path: str) -> bool:
    log(f"RUN: {' '.join(cmd)}")
    log("     (outputting directly to command line)")
    
    # ???? ??? ???? ????? ????????? ?? ???? ???????
    # ?? ??? ?????????? stdout ? stderr? ???????? ?? ???? ????? ?? ??????? ??? ???????
    result = subprocess.run(
        cmd, cwd=PIPELINE_DIR
    )
    
    ok = result.returncode == 0
    log(f"{'OK' if ok else 'FAILED'} (exit code {result.returncode})")
    return ok


def run_main_for_platform(platform: dict, date_jalali: str) -> bool:
    end_date = jalali_add_days(date_jalali, 1)
    cmd = [
        PYTHON_BIN, MAIN_SCRIPT,
        "--db-name", platform["db_name"],
        "--table-name", platform["table_name"],
        "--text-col", platform["text_col"],
        "--date-col", platform["date_col"],
        "--source-name", platform["source_name"],
        "--start-date", date_jalali,
        "--end-date", end_date,
        *platform.get("extra_args", []),
        *COMMON_MAIN_ARGS,
    ]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOG_DIR, f"main_{platform['name']}_{date_jalali}_{ts}.log")

    for attempt in range(1, MAX_ATTEMPTS_PER_DAY + 1):
        log(f"[{platform['name']}] day {date_jalali} - attempt {attempt}/{MAX_ATTEMPTS_PER_DAY}")
        if _run_subprocess(cmd, log_path):
            return True
        if attempt < MAX_ATTEMPTS_PER_DAY:
            log(f"[{platform['name']}] retrying in {RETRY_SLEEP_SECONDS}s...")
            time.sleep(RETRY_SLEEP_SECONDS)

    log(f"[{platform['name']}] day {date_jalali} FAILED after "
        f"{MAX_ATTEMPTS_PER_DAY} attempt(s) - watermark NOT advanced, will retry next tick.")
    return False


def run_aggregator() -> bool:
    cmd = [PYTHON_BIN, AGGREGATOR_SCRIPT]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOG_DIR, f"aggregator_{ts}.log")
    return _run_subprocess(cmd, log_path)


# ==========================================================================
# core logic: catch up one calendar day at a time, across both platforms,
# in lockstep - see module docstring for why this is done day-by-day
# rather than as one big date range.
# ==========================================================================

def process_one_day(state: dict, date_jalali: str) -> bool:
    """Runs whichever platforms still need `date_jalali`. Returns True only
    if, by the end of this call, every platform's watermark is past this
    date (i.e. the day is now fully done, whether just now or already
    before)."""
    all_done = True
    for platform in PLATFORMS:
        p_state = state["platforms"][platform["name"]]
        if p_state["next_date_jalali"] != date_jalali:
            # this platform is already past this date (or, in principle,
            # unexpectedly behind it - the outer loop always requests the
            # minimum next_date across platforms, so "behind" can't happen)
            continue

        ok = run_main_for_platform(platform, date_jalali)
        if ok:
            p_state["next_date_jalali"] = jalali_add_days(date_jalali, 1)
            p_state["last_success_at"] = datetime.now().isoformat()
            save_state(state)
        else:
            all_done = False

    return all_done


def catch_up(state: dict):
    latest_ok_day = latest_processable_jalali_day()
    days_done_this_invocation = 0

    while True:
        pending = [
            state["platforms"][p["name"]]["next_date_jalali"]
            for p in PLATFORMS
            if state["platforms"][p["name"]]["next_date_jalali"] <= latest_ok_day
        ]
        if not pending:
            log(f"All platforms are caught up (nothing newer than {latest_ok_day} to process yet).")
            break

        if MAX_DAYS_PER_INVOCATION is not None and days_done_this_invocation >= MAX_DAYS_PER_INVOCATION:
            log(f"Hit MAX_DAYS_PER_INVOCATION ({MAX_DAYS_PER_INVOCATION}) - "
                f"stopping here, will continue the remaining backlog next tick.")
            break

        current_date = min(pending)
        log(f"--- processing pipeline day {current_date} ---")
        day_complete = process_one_day(state, current_date)

        if not day_complete:
            log(f"Day {current_date} did not complete for every platform - "
                f"stopping catch-up here so nothing gets out of sync. "
                f"Will retry automatically on the next tick.")
            break

        days_done_this_invocation += 1
        state["pipeline_day_counter"] += 1
        save_state(state)
        log(f"Pipeline day {current_date} complete for all platforms "
            f"({state['pipeline_day_counter']}/{AGGREGATOR_EVERY_N_DAYS} "
            f"toward the next aggregator run).")

        if state["pipeline_day_counter"] >= AGGREGATOR_EVERY_N_DAYS:
            log(f"Reached {AGGREGATOR_EVERY_N_DAYS} pipeline day(s) - running candidates_aggregator.py")
            if run_aggregator():
                state["pipeline_day_counter"] = 0
                state["last_aggregator_run_at"] = datetime.now().isoformat()
            else:
                log("Aggregator run FAILED - counter left as-is, will retry on the next tick "
                    "(new candidate days will keep accumulating in the meantime, which is fine - "
                    "the aggregator's own watermark just picks up everything since its last "
                    "successful run whenever it does succeed).")
            save_state(state)


# ==========================================================================
# entry points
# ==========================================================================

def print_status(state: dict):
    print(json.dumps(state, ensure_ascii=False, indent=2))
    print(f"\nLatest day eligible for processing right now: {latest_processable_jalali_day()}")


def run_once():
    if not acquire_lock():
        return
    try:
        state = load_state()
        catch_up(state)
    finally:
        release_lock()


def run_loop():
    log(f"Starting in daemon/loop mode. Daily tick scheduled for "
        f"{DAILY_RUN_HOUR:02d}:{DAILY_RUN_MINUTE:02d} local time.")

    stop = {"flag": False}

    def _handle_signal(signum, _frame):
        log(f"Received signal {signum} - will stop after the current check.")
        stop["flag"] = True

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    state = load_state()

    def _next_scheduled_from(base: datetime) -> datetime:
        candidate = base.replace(hour=DAILY_RUN_HOUR, minute=DAILY_RUN_MINUTE, second=0, microsecond=0)
        if candidate <= base:
            candidate += timedelta(days=1)
        return candidate

    if not state.get("next_run_at"):
        # first ever start in loop mode - run once immediately (there's
        # presumably backlog, per the bootstrap discussion), then settle
        # into the daily schedule from here on.
        next_run_at = datetime.now()
    else:
        next_run_at = datetime.fromisoformat(state["next_run_at"])

    while not stop["flag"]:
        now = datetime.now()
        if now >= next_run_at:
            if acquire_lock():
                try:
                    state = load_state()
                    catch_up(state)
                except Exception:
                    import traceback
                    log("UNEXPECTED ERROR during this tick:\n" + traceback.format_exc())
                finally:
                    release_lock()
            else:
                log("Skipped this tick - lock held by another instance.")

            # Schedule the NEXT tick from the scheduled time (not "now"),
            # so a slow catch-up run doesn't push future ticks later and
            # later - it always lands back on DAILY_RUN_HOUR.
            next_run_at = _next_scheduled_from(next_run_at)
            state["next_run_at"] = next_run_at.isoformat()
            save_state(state)
            log(f"Next tick scheduled for {next_run_at}")

        time.sleep(LOOP_CHECK_INTERVAL_SECONDS)

    log("Stopped.")


def main():
    parser = argparse.ArgumentParser(
        description="Automation wrapper for main.py (daily, per platform) and "
                    "candidates_aggregator.py (every N pipeline days)."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--once", action="store_true",
                       help="Do one catch-up pass (process every pending day, "
                            "trigger the aggregator if due) then exit. Cron-friendly.")
    mode.add_argument("--loop", action="store_true",
                       help="Run forever, ticking once a day at DAILY_RUN_HOUR local time.")
    mode.add_argument("--status", action="store_true",
                       help="Print current state and exit without doing anything.")
    args = parser.parse_args()

    if args.status:
        print_status(load_state())
    elif args.once:
        run_once()
    elif args.loop:
        run_loop()


if __name__ == "__main__":
    main()
