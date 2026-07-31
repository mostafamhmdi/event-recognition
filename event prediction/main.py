   

## python3 main.py --table-name "posts" --start-date "2025-08-23" --end-date "2025-08-25"

"""
main.py
=======
Orchestrates the end-to-end event-prediction pipeline:

    Stage 1 - DataLoader            : fetch raw messages from ClickHouse, one calendar day at a time
    Stage 2 - TemporalExtractor +
              LocationExtractor     : extract/normalize the "future date" and the location(s)
                                       mentioned in each message
    Stage 3 - EventClusterer        : cluster messages that share the same future date AND a
                                       common location (messages missing either are left unclustered)
    Stage 4 - ClusterPostProcessor  : semantic post-processing - drop noise/duplicates from each
                                       cluster and merge near-duplicate clusters (embedding-based)
    Stage 5 - QwenEventVerifier     : LLM verification of each surviving cluster

Every stage's output is written to a CSV file (a plain text format) under
<output-dir>/<YYYY-MM-DD>/ so each step can be inspected/audited independently
and the pipeline can be resumed/debugged day by day.
"""

import os
import sys
import gc
import ast
import json
import uuid
import argparse
import importlib
import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

try:
    import torch
except ImportError:
    torch = None

from data_loader import DataLoader
from temporal_extractor import process_temporal_data
from location_extractor import LocationExtractor
from event_clusterer import process_clusters
from event_verifier import QwenEventVerifier
from data_writer import VerifiedEventsWriter

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def setup_logging(log_dir: str) -> logging.Logger:
    """Configure a logger that writes timestamped, English log lines to both
    the console and a log file, from the very first line of execution."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"pipeline_{datetime.now():%Y%m%d_%H%M%S}.log")

    logger = logging.getLogger("event_pipeline")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not logger.handlers:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(fmt)
        logger.addHandler(stream_handler)

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger


# --------------------------------------------------------------------------- #
# Model load / unload helpers
# --------------------------------------------------------------------------- #
def _empty_cuda_cache():
    """Best-effort release of cached (but unused) CUDA memory back to the
    driver. Safe to call even if torch/CUDA aren't available."""
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def unload_object(obj, logger: logging.Logger, label: str):
    """Best-effort GPU/CPU memory release for a loaded model wrapper."""
    if obj is None:
        return
    heavy_attr_names = (
        "model", "tokenizer", "pipeline", "ner_pipeline", "nlp",
        "encoder", "embedder", "base_model", "net",
    )
    for attr in heavy_attr_names:
        if hasattr(obj, attr):
            try:
                delattr(obj, attr)
            except Exception:
                pass
    del obj
    gc.collect()
    _empty_cuda_cache()
    logger.info(f"{label} unloaded, GPU/CPU memory released.")


def load_ner(args, logger: logging.Logger):
    """Loads the GPU-resident location NER model needed for Stage 2."""
    logger.info("Loading location NER model (Stage 2) ...")
    location_extractor = LocationExtractor(
        model_path=args.location_model_path,
        device=0,
    )
    logger.info("Location NER model loaded successfully.")
    return location_extractor


def unload_ner(location_extractor, logger: logging.Logger):
    """Unloads the location NER model freeing its GPU memory."""
    unload_object(location_extractor, logger, "Location NER model")


def load_embedding(logger: logging.Logger):
    """Loads the embedding/similarity model used inside post_cluster.py for Stage 4."""
    logger.info("Loading embedding/similarity model (Stage 4) ...")
    for mod_name in ("post_cluster", "utils"):
        sys.modules.pop(mod_name, None)
    post_cluster_module = importlib.import_module("post_cluster")
    logger.info("Embedding/similarity model loaded successfully.")
    return post_cluster_module


def unload_embedding(post_cluster_module, logger: logging.Logger):
    """Unloads the embedding/similarity model freeing its GPU memory."""
    del post_cluster_module
    for mod_name in ("post_cluster", "utils"):
        sys.modules.pop(mod_name, None)
    gc.collect()
    _empty_cuda_cache()
    logger.info("Embedding/similarity model unloaded, GPU memory released.")


def load_llm(args, logger: logging.Logger) -> QwenEventVerifier:
    """Loads the Qwen LLM verifier (Stage 5) for a single day."""
    logger.info("Loading Qwen LLM verifier (Stage 5) ...")
    qwen_verifier = QwenEventVerifier(
        model_path=args.qwen_model_path,
        sample_size=args.qwen_sample_size,
        max_new_tokens=args.qwen_max_new_tokens,
        max_input_tokens=args.qwen_max_input_tokens,
    )
    logger.info("Qwen LLM verifier loaded successfully.")
    return qwen_verifier


def unload_llm(qwen_verifier, logger: logging.Logger):
    """Unloads the Qwen LLM verifier, freeing its GPU memory."""
    unload_object(qwen_verifier, logger, "Qwen LLM verifier")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def daterange(start: str, end: str):
    """Yield (day_start, day_end) 'YYYY-MM-DD' string pairs, one per calendar
    day, covering the half-open interval [start, end)."""
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    if end_dt <= start_dt:
        raise ValueError(f"end_date ({end}) must be after start_date ({start})")

    current = start_dt
    while current < end_dt:
        nxt = current + timedelta(days=1)
        yield current.strftime("%Y-%m-%d"), nxt.strftime("%Y-%m-%d")
        current = nxt


def save_stage_output(df: pd.DataFrame, out_dir: str, stage_name: str, logger: logging.Logger) -> str:
    """Persist a stage's dataframe to a CSV (plain text) file and log where it went."""
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{stage_name}.csv")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info(f"Saved stage output -> {out_path} ({len(df)} row(s))")
    return out_path


EVENT_COLUMNS = [
    "id", "title", "summary", "predicted_time", "predicted_location",
    "execution_time", "source", "sample_messages", "created_at",
]


def make_empty_events_df() -> pd.DataFrame:
    return pd.DataFrame(columns=EVENT_COLUMNS)


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #
def parse_args():
    parser = argparse.ArgumentParser(description="Day-by-day event prediction pipeline")

    parser.add_argument("--db-name", default="telegram", help="ClickHouse database name")
    parser.add_argument("--table-name",default="raya_sepehr_analytical", required=True, help="ClickHouse table name")
    parser.add_argument("--text-col", default="txtContent", help="Name of the message-text column")
    parser.add_argument("--date-col", default="date", help="Name of the message post-date column")

    parser.add_argument("--start-date", required=True, help="Start date, inclusive (YYYY-MM-DD)")
    parser.add_argument("--end-date", required=True, help="End date, exclusive (YYYY-MM-DD)")

    parser.add_argument("--location-model-path", default="../../models/arman NER",
                         help="Local path to the location NER model used by LocationExtractor")
    
    parser.add_argument("--output-dir", default="./pipeline_output",
                         help="Root directory where per-day stage outputs and logs are written")

    parser.add_argument("--source-name", default="telegram",
                         help="Value stored in the 'source' column of every verified event "
                              "(the social network the messages came from)")

    parser.add_argument("--qwen-model-path", default="../../models/qwen/",
                         help="Local path to the 4-bit Qwen model used by QwenEventVerifier "
                              "to verify each cluster and produce title/summary")
    parser.add_argument("--qwen-sample-size", type=int, default=10,
                         help="How many messages to randomly sample from each cluster and "
                              "send to the LLM for verification")
    parser.add_argument("--qwen-max-new-tokens", type=int, default=256,
                         help="max_new_tokens passed to QwenEventVerifier")
    parser.add_argument("--qwen-max-input-tokens", type=int, default=4800,
                         help="max_input_tokens passed to QwenEventVerifier")
    parser.add_argument("--skip-llm-verification", action="store_true",
                         help="Skip Stage 5 (Qwen LLM verification) entirely - useful for "
                              "debugging stages 1-4 without paying the cost of loading the LLM")

    return parser.parse_args()


def _most_common(series: Optional[pd.Series]):
    if series is None:
        return None
    vals = series.dropna()
    if vals.empty:
        return None
    mode = vals.mode()
    return mode.iloc[0] if not mode.empty else vals.iloc[0]


def _first_location(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return str(value[0]) if value else None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = ast.literal_eval(stripped)
                if isinstance(parsed, (list, tuple)):
                    return str(parsed[0]) if parsed else None
            except (ValueError, SyntaxError):
                pass
        return stripped
    return str(value)


def _most_common_location(series: Optional[pd.Series]):
    if series is None:
        return None
    firsts = [loc for loc in (_first_location(v) for v in series.dropna()) if loc]
    if not firsts:
        return None
    return pd.Series(firsts).mode().iloc[0]


def build_candidates_from_df(df: pd.DataFrame, text_col: str) -> list:
    candidates = []

    if "cluster_id" not in df.columns:
        return candidates

    clustered = df[df["cluster_id"].notna()]

    for cluster_id, group in clustered.groupby("cluster_id"):

        # فقط خوشه‌هایی که بیشتر از 25 پیام دارند
        if len(group) <= 25:
            continue

        # نمونه‌گیری تصادفی 15 پیام
        sampled_group = group.sample(n=15, random_state=42)

        messages = (
            sampled_group[text_col]
            .dropna()
            .astype(str)
            .tolist()
        )

        if not messages:
            continue

        candidates.append({
            "cluster_id": cluster_id,
            "messages": messages,
            # این دو ویژگی همچنان از کل خوشه محاسبه می‌شوند
            "predicted_time": _most_common(group.get("future_date")),
            "predicted_location": _most_common_location(group.get("location_entities")),
        })

    return candidates


def build_event_records(verified_events: list, execution_time: str, source: str) -> pd.DataFrame:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = []
    for ev in verified_events:
        rows.append({
            "id": str(uuid.uuid4()),
            "title": ev.get("event_title"),
            "summary": ev.get("event_summary"),
            "predicted_time": ev.get("predicted_time"),
            "predicted_location": ev.get("predicted_location"),
            "execution_time": execution_time,
            "source": source,
            "sample_messages": json.dumps(ev.get("sample_messages", []), ensure_ascii=False),
            "created_at": now,
        })

    return pd.DataFrame(rows, columns=EVENT_COLUMNS)


# --------------------------------------------------------------------------- #
# Per-day pipeline - Stages 1-4
# --------------------------------------------------------------------------- #
def run_pre_llm_stages(
    day_start: str,
    day_end: str,
    day_index: int,
    total_days: int,
    loader: DataLoader,
    args,
    text_col: str,
    date_col: str,
    output_dir: str,
    logger: logging.Logger,
) -> Optional[pd.DataFrame]:
    """Runs Stages 1-4 for a single calendar day: data loading, temporal
    extraction, NER-based location/event extraction, clustering, and
    embedding-based post-processing.
    """
    logger.info("-" * 70)
    logger.info(f"[Day {day_index}/{total_days}] Processing {day_start}")
    day_out_dir = os.path.join(output_dir, day_start)

    # --------------------------------------------------------------- #
    # Stage 1: fetch this day's messages from ClickHouse
    # --------------------------------------------------------------- #
    try:
        day_df = loader.load_and_prepare(
            text_col=text_col,
            date_col=date_col,
            start_date=day_start,
            end_date=day_end,
        )
    except Exception:
        logger.exception(f"[Day {day_index}/{total_days}] Failed to load data for {day_start}. Skipping this day.")
        return None

    if day_df is None or day_df.empty:
        logger.warning(f"[Day {day_index}/{total_days}] No messages found for {day_start}. Skipping this day.")
        return None

    day_df = day_df.reset_index(drop=True)
    logger.info(f"[Day {day_index}/{total_days}] Stage 1/4 (Data Loading) complete: {len(day_df)} row(s) fetched.")
    save_stage_output(day_df, day_out_dir, "01_raw_messages", logger)

    # --------------------------------------------------------------- #
    # Stage 2a: temporal extraction (future date)
    # --------------------------------------------------------------- #
    logger.info(f"[Day {day_index}/{total_days}] Stage 2/4 (Temporal Extraction) starting ...")
    day_df = process_temporal_data(day_df, text_col=text_col, date_col=date_col)
    future_mask = day_df["future_date"].notna()
    n_future = int(future_mask.sum())
    logger.info(
        f"[Day {day_index}/{total_days}] Temporal extraction complete: "
        f"{n_future}/{len(day_df)} row(s) mention a future date."
    )

    # --------------------------------------------------------------- #
    # Stage 2b: location extraction (GPU/NER, the expensive part).
    # --------------------------------------------------------------- #
    day_df["location_entities"] = None
    day_df["event_entities"] = None
    if n_future > 0:
        logger.info(
            f"[Day {day_index}/{total_days}] Stage 2/4 (Location + Event Extraction) starting on "
            f"{n_future} future-dated row(s) only (skipping the other "
            f"{len(day_df) - n_future} row(s))..."
        )
        future_subset = day_df.loc[future_mask].copy()
        
        # Load NER only when needed for Stage 2b
        location_extractor = load_ner(args, logger)
        try:
            future_subset = location_extractor.extract_entities(future_subset, text_col=text_col)
        finally:
            # Unload NER right after Stage 2b is complete
            unload_ner(location_extractor, logger)

        day_df.loc[future_subset.index, "location_entities"] = future_subset["location_entities"]
        day_df.loc[future_subset.index, "event_entities"] = future_subset["event_entities"]

        n_loc = int(day_df["location_entities"].notna().sum())
        n_event = int(day_df["event_entities"].notna().sum())
        n_ready = int((day_df["location_entities"].notna() & day_df["event_entities"].notna()).sum())
        logger.info(
            f"[Day {day_index}/{total_days}] Stage 2/4 complete: {n_loc} row(s) with a location, "
            f"{n_event} row(s) with an event, {n_ready} row(s) with BOTH "
            f"(these are the ones eligible for clustering in Stage 3)."
        )
    else:
        logger.info(
            f"[Day {day_index}/{total_days}] No rows with a future date today - "
            f"skipping location/event extraction entirely."
        )
    save_stage_output(day_df, day_out_dir, "02_temporal_location_extracted", logger)

    # --------------------------------------------------------------- #
    # Stage 3: event clustering.
    # --------------------------------------------------------------- #
    logger.info(f"[Day {day_index}/{total_days}] Stage 3/4 (Event Clustering) starting ...")
    day_df = process_clusters(day_df, text_col=text_col)
    n_clustered = day_df["cluster_id"].notna().sum()
    logger.info(f"[Day {day_index}/{total_days}] Stage 3/4 complete: {n_clustered} row(s) assigned to a cluster.")
    save_stage_output(day_df, day_out_dir, "03_event_clustered", logger)

    # --------------------------------------------------------------- #
    # Stage 4: semantic post-processing (denoise / dedupe / merge).
    # --------------------------------------------------------------- #
    logger.info(f"[Day {day_index}/{total_days}] Stage 4/5 (Cluster Post-Processing) starting ...")
    
    # Load Embedding model right before Stage 4
    post_cluster_module = load_embedding(logger)
    try:
        day_df = post_cluster_module.post_process_clusters(day_df, text_col=text_col)
    finally:
        # Unload Embedding model immediately after post_processing
        unload_embedding(post_cluster_module, logger)
        
    n_final = day_df["cluster_id"].notna().sum()
    logger.info(f"[Day {day_index}/{total_days}] Stage 4/5 complete: {n_final} row(s) remain clustered.")
    save_stage_output(day_df, day_out_dir, "04_post_processed", logger)

    return day_df


# --------------------------------------------------------------------------- #
# Per-day pipeline - Stage 5 (needs the LLM loaded)
# --------------------------------------------------------------------------- #
def run_llm_stage(
    day_df: pd.DataFrame,
    day_start: str,
    day_index: int,
    total_days: int,
    qwen_verifier: QwenEventVerifier,
    events_writer: VerifiedEventsWriter,
    text_col: str,
    output_dir: str,
    source_name: str,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Runs Stage 5 (LLM verification) on a single day's Stage-4 output."""
    day_out_dir = os.path.join(output_dir, day_start)
    events_df = make_empty_events_df()

    candidates = build_candidates_from_df(day_df, text_col=text_col)
    logger.info(f"[Day {day_index}/{total_days}] Stage 5/5 (LLM Verification) starting on "
                 f"{len(candidates)} cluster(s) ...")
    if candidates:
        verified_events = qwen_verifier.verify_candidates(candidates)
        events_writer.save_events(verified_events, execution_time=day_start)
        events_df = build_event_records(
            verified_events, execution_time=day_start, source=source_name,
        )
        logger.info(f"[Day {day_index}/{total_days}] Stage 5/5 complete: "
                     f"{len(events_df)}/{len(candidates)} cluster(s) confirmed as events.")
        save_stage_output(events_df, day_out_dir, "05_verified_events", logger)
    else:
        logger.info(f"[Day {day_index}/{total_days}] Stage 5/5: no clusters to verify.")

    return events_df


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    logger = setup_logging(os.path.join(args.output_dir, "logs"))

    logger.info("=" * 70)
    logger.info("Event prediction pipeline started")
    logger.info(f"Database         : {args.db_name}")
    logger.info(f"Table            : {args.table_name}")
    logger.info(f"Text column      : {args.text_col}")
    logger.info(f"Date column      : {args.date_col}")
    logger.info(f"Date range       : {args.start_date} -> {args.end_date} (exclusive)")
    logger.info(f"Location model   : {args.location_model_path}")
    logger.info(f"Output directory : {args.output_dir}")
    if args.skip_llm_verification:
        logger.info("Stage 5 (LLM Verification) disabled via --skip-llm-verification.")
    logger.info("Model lifecycle  : per-stage - NER is loaded specifically for Stage 2, unloaded, "
                "then Embedding is loaded for Stage 4, unloaded. Finally Qwen is loaded for Stage 5 "
                "and unloaded.")
    logger.info("=" * 70)

    loader = DataLoader(db_name=args.db_name, table_name=args.table_name)
    day_pairs = list(daterange(args.start_date, args.end_date))
    total_days = len(day_pairs)
    logger.info(f"Pipeline will process {total_days} day(s).")

    events_writer = VerifiedEventsWriter(
        db_name="raya_sepehr_analytical",
        source=args.source_name,
        event_table="predicted_events",
    )
    
    all_days_final = []
    all_events_final = []

    for day_index, (day_start, day_end) in enumerate(day_pairs, start=1):
        # ----------------------------------------------------------- #
        # Step A: run Stages 1-4.
        # NER and Embedding are loaded and unloaded internally within
        # run_pre_llm_stages exactly when they are needed.
        # ----------------------------------------------------------- #
        try:
            day_df = run_pre_llm_stages(
                day_start=day_start,
                day_end=day_end,
                day_index=day_index,
                total_days=total_days,
                loader=loader,
                args=args,
                text_col=args.text_col,
                date_col=args.date_col,
                output_dir=args.output_dir,
                logger=logger,
            )
        except Exception:
            logger.exception(
                f"[Day {day_index}/{total_days}] Failed during Stages 1-4. "
                f"Skipping {day_start} entirely."
            )
            continue

        if day_df is not None:
            all_days_final.append(day_df)
        
        # ----------------------------------------------------------- #
        # Step B: LLM stage.
        # ----------------------------------------------------------- #
        events_df = make_empty_events_df()
        if day_df is None:
            logger.info(f"[Day {day_index}/{total_days}] Nothing to verify today - Stage 5 skipped.")
        elif args.skip_llm_verification:
            logger.info(f"[Day {day_index}/{total_days}] Stage 5 (LLM Verification) skipped "
                         f"(--skip-llm-verification).")
        else:
            qwen_verifier = None
            try:
                qwen_verifier = load_llm(args, logger)
            except Exception:
                logger.exception(
                    f"[Day {day_index}/{total_days}] Failed to load the Qwen LLM verifier. "
                    f"Continuing WITHOUT Stage 5 for today (clusters were still produced "
                    f"through Stage 4, just not verified/turned into event records)."
                )

            if qwen_verifier is not None:
                try:
                    events_df = run_llm_stage(
                        day_df=day_df,
                        day_start=day_start,
                        day_index=day_index,
                        total_days=total_days,
                        qwen_verifier=qwen_verifier,
                        events_writer=events_writer,
                        text_col=args.text_col,
                        output_dir=args.output_dir,
                        source_name=args.source_name,
                        logger=logger,
                    )
                finally:
                    unload_llm(qwen_verifier, logger)

        if events_df is not None and not events_df.empty:
            all_events_final.append(events_df)

        logger.info(f"[Day {day_index}/{total_days}] Finished processing {day_start}.")

    # ------------------------------------------------------------------- #
    # Combine every processed day into one master file for convenience
    # ------------------------------------------------------------------- #
    logger.info("=" * 70)
    if all_days_final:
        combined_df = pd.concat(all_days_final, ignore_index=True)
        save_stage_output(combined_df, args.output_dir, "00_all_days_combined", logger)
        logger.info(f"Pipeline finished successfully. Total rows processed across all days: {len(combined_df)}.")
    else:
        logger.warning("Pipeline finished, but no day produced any data.")

    if all_events_final:
        combined_events_df = pd.concat(all_events_final, ignore_index=True)
        save_stage_output(combined_events_df, args.output_dir, "00_all_events_combined", logger)
        logger.info(f"Total verified events across all days (DB-ready): {len(combined_events_df)}.")
    else:
        logger.warning("No verified events were produced across any day.")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()