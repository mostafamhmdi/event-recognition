
import os
import re
import random
from collections import Counter
from datetime import datetime

import pandas as pd
from clickhouse_connect import get_client



def _as_datetime(value):
    """Accepts a pandas Timestamp / python datetime / date-like and
    returns a plain python datetime, which clickhouse_connect expects
    for DateTime columns."""
    return pd.Timestamp(value).to_pydatetime()


def _sample(messages, limit=10):
    """ClickHouse's JSON column type only accepts an object at the root
    (not a bare array) - hence the {"messages": [...]} wrapper instead of
    just a list."""
    return {"messages": list((messages or [])[:limit])}


def _text_field_or_empty(cand, key):
    """`keywords`/`sentiment_polarity` are non-Nullable String columns,
    so a real None crashes the whole insert. If candidate_extractor.py
    starts producing cand['keywords'] / cand['sentiment_polarity']
    itself, this picks it up automatically; until then it falls back to
    '' (the closest thing to "no value" this column type allows)."""
    value = cand.get(key)
    return value if value else ""


class ClickHouseResultsWriter:
    """
    Two public methods, one per table/task:
      - save_candidate_clusters(...)  -> INSERT into candidate_clusters (Phase 3)
      - save_detected_events(...)     -> INSERT into detected_events (Phase 4)
                                          + UPDATE candidate_clusters.validation_status

    Connection settings default to the same host/port/user/pass used
    elsewhere in this project (see DataLoader); override via env vars or
    constructor args if the results tables live in a different database.
    """

    def __init__(self, db_name="raya_sepehr_analytical", source="telegram",
                 candidate_table="candidate_clusters",
                 event_table="detected_events"):
        self.db_name = db_name
        self.source = source
        self.candidate_table = candidate_table
        self.event_table = event_table
        self._client = None
        self._next_candidate_id = None
        self._next_event_id = None
        self._alter_update_disabled = False

    # ---- connection ----

    def _get_client(self):
        if self._client is None:
            self._client = get_client(
                host=os.getenv("CH_HOST", '172.20.70.191'),
                port=int(os.getenv("CH_PORT", 8123)),
                database=self.db_name,
                username=os.getenv("CH_USER", 'labafi'),
                password=os.getenv("CH_PASS", 'l@b@fi@1234')
            )
        return self._client

    def close(self):
        if self._client is not None:
            self._client.close()
            self._client = None

    # ---- id allocation ----
    # No SELECT max(id) anymore - it broke the run when the table wasn't
    # reachable/didn't exist yet (Code 60 UNKNOWN_TABLE), and the `id`
    # column has no autoincrement to fall back on either way. Instead,
    # each run picks its own random starting id (once, the first time
    # each table is written to) and just increments from there for every
    # row in that same run - so ids never collide with each other WITHIN
    # a run, but two different runs could in principle still land on the
    # same id range. That's an acceptable tradeoff per your call; if you
    # ever need a hard uniqueness guarantee, this is the spot to swap
    # back to a real SELECT max(id)+1 (or a UUID/different key strategy).
    _ID_RANGE = (1_000, 2_000_000_000)

    def _next_id_for(self, table, cache_attr):
        cached = getattr(self, cache_attr)
        if cached is not None:
            return cached
        next_id = random.randint(*self._ID_RANGE)
        setattr(self, cache_attr, next_id)
        print(f"[ClickHouse] {table}: starting this run's ids at {next_id} "
              f"(random start, no DB round-trip).")
        return next_id

    # ---- Task 1 / Phase 3: candidate_clusters ----

    def save_candidate_clusters(self, day_candidates, execution_time):
        """
        Inserts one row per Phase-3 candidate into candidate_clusters.
        Mutates each dict in `day_candidates` in place, adding '_db_id'
        (the assigned integer id) - save_detected_events() relies on that
        later for the same candidates.
        """
        if not day_candidates:
            return

        exec_dt = _as_datetime(execution_time)
        next_id = self._next_id_for(self.candidate_table, '_next_candidate_id')

        rows = []
        for cand in day_candidates:
            assigned_id = next_id
            next_id += 1
            cand['_db_id'] = assigned_id

            reasons = cand.get('reasons', []) or []
            messages = cand.get('messages', []) or []

            rows.append([
                assigned_id,
                ', '.join(cand.get('keywords', [])), 
                1 if 'Dominant' in reasons else 0,
                int(cand.get('size', 0) or 0),
                1 if 'Spike' in reasons else 0,
                1 if 'Cohesive_Large' in reasons else 0,
                _text_field_or_empty(cand, 'sentiment_polarity'),
                exec_dt,
                'pending',
                _sample(messages),
            ])

        self._next_candidate_id = next_id

        try:
            client = self._get_client()
            client.insert(
                self.candidate_table,
                rows,
                column_names=[
                    'id', 'keywords', 'is_large', 'cluster_size', 'has_burst',
                    'is_cohesive', 'sentiment_polarity', 'execution_time',
                    'validation_status', 'sample_messages',
                ],
            )
            print(f"[ClickHouse] Inserted {len(rows)} row(s) into "
                  f"{self.candidate_table}")
        except Exception as e:
            print(f"[ClickHouse] ERROR inserting into {self.candidate_table}: {e}")

    # ---- Task 2 / Phase 4: detected_events (+ status update) ----

    def save_detected_events(self, day_candidates, day_verified_events,
                              execution_time, verification_attempted=True):
        """
        Inserts one row per Phase-4 confirmed event into detected_events,
        using each candidate's '_db_id' (set by save_candidate_clusters)
        as the FK `cluster_id`. Then updates validation_status back on
        candidate_clusters for every candidate from the same day - see
        the module docstring for exactly which rows get which status.

        `verification_attempted` should be False for a day where Phase 4
        didn't run at all (e.g. the Qwen verifier failed to load) so
        untouched candidates are left 'pending' instead of wrongly
        marked 'rejected'.
        """
        exec_dt = _as_datetime(execution_time)

        # ---- 2a. insert confirmed events ----
        rows = []
        if day_verified_events:
            next_id = self._next_id_for(self.event_table, '_next_event_id')
            for event in day_verified_events:
                cluster_db_id = event.get('_db_id')
                if cluster_db_id is None:
                    print(f"[ClickHouse] WARNING: a verified event has no "
                          f"'_db_id' (save_candidate_clusters wasn't called "
                          f"for it first) - skipping this row in "
                          f"{self.event_table}.")
                    continue

                assigned_id = next_id
                next_id += 1
                messages = event.get('messages', []) or []

                rows.append([
                    assigned_id,
                    int(cluster_db_id),
                    event.get('event_title', 'N/A'),
                    event.get('event_summary', 'N/A'),
                    exec_dt,
                    self.source,
                    int(event.get('size', 0) or 0),
                    _sample(messages),
                ])

            self._next_event_id = next_id

            if rows:
                try:
                    client = self._get_client()
                    client.insert(
                        self.event_table,
                        rows,
                        column_names=[
                            'id', 'cluster_id', 'title', 'summary',
                            'execution_time', 'source', 'total_messages',
                            'sample_messages',
                        ],
                    )
                    print(f"[ClickHouse] Inserted {len(rows)} row(s) into "
                          f"{self.event_table}")
                except Exception as e:
                    print(f"[ClickHouse] ERROR inserting into "
                          f"{self.event_table}: {e}")

        # ---- 2b. update validation_status on candidate_clusters ----
        if not verification_attempted or not day_candidates:
            return
        if getattr(self, '_alter_update_disabled', False):
            return

        confirmed_ids = {
            event['_db_id'] for event in day_verified_events
            if event.get('_db_id') is not None
        }
        # Only mark candidates 'rejected' that Phase 4 actually had a shot
        # at - i.e. it had messages (maain.py's own Phase 4 loop already
        # skips messageless candidates before calling verify_candidates).
        rejected_ids = {
            cand['_db_id'] for cand in day_candidates
            if cand.get('_db_id') is not None
            and cand['_db_id'] not in confirmed_ids
            and cand.get('messages')
        }

        if not confirmed_ids and not rejected_ids:
            return

        try:
            client = self._get_client()
            if confirmed_ids:
                id_list = ",".join(str(i) for i in confirmed_ids)
                client.command(
                    f"ALTER TABLE {self.candidate_table} "
                    f"UPDATE validation_status = 'confirmed' "
                    f"WHERE id IN ({id_list})"
                )
            if rejected_ids:
                id_list = ",".join(str(i) for i in rejected_ids)
                client.command(
                    f"ALTER TABLE {self.candidate_table} "
                    f"UPDATE validation_status = 'rejected' "
                    f"WHERE id IN ({id_list})"
                )
            print(f"[ClickHouse] Updated validation_status for "
                  f"{len(confirmed_ids)} confirmed / {len(rejected_ids)} "
                  f"rejected row(s) in {self.candidate_table} "
                  f"(note: ALTER TABLE UPDATE in ClickHouse is an async "
                  f"mutation - it may take a moment to become visible).")
        except Exception as e:
            if "ACCESS_DENIED" in str(e) or "Not enough privileges" in str(e):
                self._alter_update_disabled = True
                print(f"[ClickHouse] ERROR updating validation_status on "
                      f"{self.candidate_table}: {e}\n"
                      f"[ClickHouse] This DB user lacks the ALTER UPDATE "
                      f"grant on validation_status - disabling status "
                      f"updates for the rest of this run so the error "
                      f"doesn't repeat every day. Ask a ClickHouse admin "
                      f"to run:\n"
                      f"  GRANT ALTER UPDATE(validation_status) ON "
                      f"{self.db_name}.{self.candidate_table} TO <your_user>;")
            else:
                print(f"[ClickHouse] ERROR updating validation_status on "
                      f"{self.candidate_table}: {e}")