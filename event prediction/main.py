# """
# main.py
# =======
# Orchestrates the end-to-end event-prediction pipeline:

#     Stage 1 - DataLoader            : fetch raw messages from ClickHouse, one calendar day at a time
#     Stage 2 - TemporalExtractor +
#               LocationExtractor     : extract/normalize the "future date" and the location(s)
#                                        mentioned in each message
#     Stage 3 - EventClusterer        : cluster messages that share the same future date AND a
#                                        common location (messages missing either are left unclustered)
#     Stage 4 - ClusterPostProcessor  : semantic post-processing - drop noise/duplicates from each
#                                        cluster and merge near-duplicate clusters (embedding-based)

# Every stage's output is written to a CSV file (a plain text format) under
# <output-dir>/<YYYY-MM-DD>/ so each step can be inspected/audited independently
# and the pipeline can be resumed/debugged day by day.

# All local models (location NER model, embedding model used inside post_cluster.py's
# `utils.similarity_model`) are expected to already be available at the paths configured
# below / passed on the command line - this script does not download anything.
# """

# import os
# import sys
# import argparse
# import logging
# from datetime import datetime, timedelta
# from typing import Optional

# import pandas as pd

# from data_loader import DataLoader
# from temporal_extractor import process_temporal_data
# from location_extractor import LocationExtractor
# from event_clusterer import process_clusters
# from post_cluster import post_process_clusters


# # --------------------------------------------------------------------------- #
# # Logging
# # --------------------------------------------------------------------------- #
# def setup_logging(log_dir: str) -> logging.Logger:
#     """Configure a logger that writes timestamped, English log lines to both
#     the console and a log file, from the very first line of execution."""
#     os.makedirs(log_dir, exist_ok=True)
#     log_file = os.path.join(log_dir, f"pipeline_{datetime.now():%Y%m%d_%H%M%S}.log")

#     logger = logging.getLogger("event_pipeline")
#     logger.setLevel(logging.INFO)
#     logger.propagate = False

#     fmt = logging.Formatter(
#         fmt="%(asctime)s | %(levelname)-8s | %(message)s",
#         datefmt="%Y-%m-%d %H:%M:%S",
#     )

#     if not logger.handlers:
#         stream_handler = logging.StreamHandler(sys.stdout)
#         stream_handler.setFormatter(fmt)
#         logger.addHandler(stream_handler)

#         file_handler = logging.FileHandler(log_file, encoding="utf-8")
#         file_handler.setFormatter(fmt)
#         logger.addHandler(file_handler)

#     return logger


# # --------------------------------------------------------------------------- #
# # Helpers
# # --------------------------------------------------------------------------- #
# def daterange(start: str, end: str):
#     """Yield (day_start, day_end) 'YYYY-MM-DD' string pairs, one per calendar
#     day, covering the half-open interval [start, end)."""
#     start_dt = datetime.strptime(start, "%Y-%m-%d")
#     end_dt = datetime.strptime(end, "%Y-%m-%d")
#     if end_dt <= start_dt:
#         raise ValueError(f"end_date ({end}) must be after start_date ({start})")

#     current = start_dt
#     while current < end_dt:
#         nxt = current + timedelta(days=1)
#         yield current.strftime("%Y-%m-%d"), nxt.strftime("%Y-%m-%d")
#         current = nxt


# def save_stage_output(df: pd.DataFrame, out_dir: str, stage_name: str, logger: logging.Logger) -> str:
#     """Persist a stage's dataframe to a CSV (plain text) file and log where it went."""
#     os.makedirs(out_dir, exist_ok=True)
#     out_path = os.path.join(out_dir, f"{stage_name}.csv")
#     df.to_csv(out_path, index=False, encoding="utf-8-sig")
#     logger.info(f"Saved stage output -> {out_path} ({len(df)} row(s))")
#     return out_path


# # --------------------------------------------------------------------------- #
# # Argument parsing
# # --------------------------------------------------------------------------- #
# def parse_args():
#     parser = argparse.ArgumentParser(description="Day-by-day event prediction pipeline")

#     parser.add_argument("--db-name", default="telegram", help="ClickHouse database name")
#     parser.add_argument("--table-name", required=True, help="ClickHouse table name")
#     parser.add_argument("--text-col", default="txtContent", help="Name of the message-text column")
#     parser.add_argument("--date-col", default="date", help="Name of the message post-date column")

#     parser.add_argument("--start-date", required=True, help="Start date, inclusive (YYYY-MM-DD)")
#     parser.add_argument("--end-date", required=True, help="End date, exclusive (YYYY-MM-DD)")

#     parser.add_argument("--location-model-path", default="../../models/arman NER",
#                          help="Local path to the location NER model used by LocationExtractor")
#     parser.add_argument("--device", type=int, default=0,
#                          help="Device index for the NER pipeline (0 = first GPU, -1 = CPU)")

#     parser.add_argument("--output-dir", default="./pipeline_output",
#                          help="Root directory where per-day stage outputs and logs are written")

#     return parser.parse_args()


# # --------------------------------------------------------------------------- #
# # Per-day pipeline
# # --------------------------------------------------------------------------- #
# def process_single_day(
#     day_start: str,
#     day_end: str,
#     day_index: int,
#     total_days: int,
#     loader: DataLoader,
#     location_extractor: LocationExtractor,
#     text_col: str,
#     date_col: str,
#     output_dir: str,
#     logger: logging.Logger,
# ) -> Optional[pd.DataFrame]:
#     """Run all 4 stages for a single calendar day and return the final dataframe
#     (or None if there was nothing to process / an error occurred)."""

#     logger.info("-" * 70)
#     logger.info(f"[Day {day_index}/{total_days}] Processing {day_start}")
#     day_out_dir = os.path.join(output_dir, day_start)

#     # --------------------------------------------------------------- #
#     # Stage 1: fetch this day's messages from ClickHouse
#     # --------------------------------------------------------------- #
#     try:
#         day_df = loader.load_and_prepare(
#             text_col=text_col,
#             date_col=date_col,
#             start_date=day_start,
#             end_date=day_end,
#         )
#     except Exception:
#         logger.exception(f"[Day {day_index}/{total_days}] Failed to load data for {day_start}. Skipping this day.")
#         return None

#     if day_df is None or day_df.empty:
#         logger.warning(f"[Day {day_index}/{total_days}] No messages found for {day_start}. Skipping this day.")
#         return None

#     day_df = day_df.reset_index(drop=True)
#     logger.info(f"[Day {day_index}/{total_days}] Stage 1/4 (Data Loading) complete: {len(day_df)} row(s) fetched.")
#     save_stage_output(day_df, day_out_dir, "01_raw_messages", logger)

#     # --------------------------------------------------------------- #
#     # Stage 2: temporal extraction (future date) + location extraction
#     # --------------------------------------------------------------- #
#     logger.info(f"[Day {day_index}/{total_days}] Stage 2/4 (Temporal + Location Extraction) starting ...")
#     day_df = process_temporal_data(day_df, text_col=text_col, date_col=date_col)
#     day_df = location_extractor.extract_locations(day_df, text_col=text_col)
#     n_future = day_df["future_date"].notna().sum()
#     n_loc = day_df["location_entities"].notna().sum()
#     logger.info(
#         f"[Day {day_index}/{total_days}] Stage 2/4 complete: "
#         f"{n_future} row(s) with a future date, {n_loc} row(s) with a location."
#     )
#     save_stage_output(day_df, day_out_dir, "02_temporal_location_extracted", logger)

#     # --------------------------------------------------------------- #
#     # Stage 3: event clustering. Only messages that have BOTH a future
#     # date and a location are actually clustered - that filtering is
#     # done internally by EventClusterer.
#     # --------------------------------------------------------------- #
#     logger.info(f"[Day {day_index}/{total_days}] Stage 3/4 (Event Clustering) starting ...")
#     day_df = process_clusters(day_df, text_col=text_col)
#     n_clustered = day_df["cluster_id"].notna().sum()
#     logger.info(f"[Day {day_index}/{total_days}] Stage 3/4 complete: {n_clustered} row(s) assigned to a cluster.")
#     save_stage_output(day_df, day_out_dir, "03_event_clustered", logger)

#     # --------------------------------------------------------------- #
#     # Stage 4: semantic post-processing (denoise / dedupe / merge)
#     # --------------------------------------------------------------- #
#     logger.info(f"[Day {day_index}/{total_days}] Stage 4/4 (Cluster Post-Processing) starting ...")
#     day_df = post_process_clusters(day_df, text_col=text_col)
#     n_final = day_df["cluster_id"].notna().sum()
#     logger.info(f"[Day {day_index}/{total_days}] Stage 4/4 complete: {n_final} row(s) remain clustered.")
#     save_stage_output(day_df, day_out_dir, "04_post_processed", logger)

#     logger.info(f"[Day {day_index}/{total_days}] Finished processing {day_start}.")
#     return day_df


# # --------------------------------------------------------------------------- #
# # Main
# # --------------------------------------------------------------------------- #
# def main():
#     args = parse_args()
#     logger = setup_logging(os.path.join(args.output_dir, "logs"))

#     logger.info("=" * 70)
#     logger.info("Event prediction pipeline started")
#     logger.info(f"Database         : {args.db_name}")
#     logger.info(f"Table            : {args.table_name}")
#     logger.info(f"Text column      : {args.text_col}")
#     logger.info(f"Date column      : {args.date_col}")
#     logger.info(f"Date range       : {args.start_date} -> {args.end_date} (exclusive)")
#     logger.info(f"Location model   : {args.location_model_path}")
#     logger.info(f"Device           : {args.device}")
#     logger.info(f"Output directory : {args.output_dir}")
#     logger.info("=" * 70)

#     # ------------------------------------------------------------------- #
#     # Instantiate the heavy / stateful components ONCE. Loading the NER
#     # model is expensive, so it must not be reloaded for every single day.
#     # ------------------------------------------------------------------- #
#     DB_NAME = "telegram"
#     TABLE_NAME = "posts"
#     # loader = DataLoader(db_name=args.db_name, table_name=args.table_name)
#     loader = DataLoader(db_name=DB_NAME, table_name=TABLE_NAME)
#     logger.info("Loading location NER model ...")
#     try:
#         location_extractor = LocationExtractor(
#             model_path=args.location_model_path,
#             device=args.device,
#         )
#     except Exception:
#         logger.exception("Failed to load the location NER model. Aborting pipeline.")
#         sys.exit(1)
#     logger.info("Location NER model loaded successfully.")

#     day_pairs = list(daterange(args.start_date, args.end_date))
#     total_days = len(day_pairs)
#     logger.info(f"Pipeline will process {total_days} day(s).")

#     all_days_final = []
#     for day_index, (day_start, day_end) in enumerate(day_pairs, start=1):
#         result_df = process_single_day(
#             day_start=day_start,
#             day_end=day_end,
#             day_index=day_index,
#             total_days=total_days,
#             loader=loader,
#             location_extractor=location_extractor,
#             text_col=args.text_col,
#             date_col=args.date_col,
#             output_dir=args.output_dir,
#             logger=logger,
#         )
#         if result_df is not None:
#             all_days_final.append(result_df)

#     # ------------------------------------------------------------------- #
#     # Combine every processed day into one master file for convenience
#     # ------------------------------------------------------------------- #
#     logger.info("=" * 70)
#     if all_days_final:
#         combined_df = pd.concat(all_days_final, ignore_index=True)
#         save_stage_output(combined_df, args.output_dir, "00_all_days_combined", logger)
#         logger.info(f"Pipeline finished successfully. Total rows processed across all days: {len(combined_df)}.")
#     else:
#         logger.warning("Pipeline finished, but no day produced any data.")
#     logger.info("=" * 70)


# if __name__ == "__main__":
#     main()
    
    
    
    
    
    
## python3 main.py --table-name "posts" --start-date "2025-08-24" --end-date "2025-08-24"



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

Every stage's output is written to a CSV file (a plain text format) under
<output-dir>/<YYYY-MM-DD>/ so each step can be inspected/audited independently
and the pipeline can be resumed/debugged day by day.

All local models (location NER model, embedding model used inside post_cluster.py's
`utils.similarity_model`) are expected to already be available at the paths configured
below / passed on the command line - this script does not download anything.
"""

import os
import sys
import argparse
import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from data_loader import DataLoader
from temporal_extractor import process_temporal_data
from location_extractor import LocationExtractor
from event_clusterer import process_clusters
from post_cluster import post_process_clusters


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


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #
def parse_args():
    parser = argparse.ArgumentParser(description="Day-by-day event prediction pipeline")

    parser.add_argument("--db-name", default="telegram", help="ClickHouse database name")
    parser.add_argument("--table-name", required=True, help="ClickHouse table name")
    parser.add_argument("--text-col", default="txtContent", help="Name of the message-text column")
    parser.add_argument("--date-col", default="date", help="Name of the message post-date column")

    parser.add_argument("--start-date", required=True, help="Start date, inclusive (YYYY-MM-DD)")
    parser.add_argument("--end-date", required=True, help="End date, exclusive (YYYY-MM-DD)")

    parser.add_argument("--location-model-path", default="../../models/arman NER",
                         help="Local path to the location NER model used by LocationExtractor")
    parser.add_argument("--device", type=int, default=0,
                         help="Device index for the NER pipeline (0 = first GPU, -1 = CPU)")

    parser.add_argument("--output-dir", default="./pipeline_output",
                         help="Root directory where per-day stage outputs and logs are written")

    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Per-day pipeline
# --------------------------------------------------------------------------- #
def process_single_day(
    day_start: str,
    day_end: str,
    day_index: int,
    total_days: int,
    loader: DataLoader,
    location_extractor: LocationExtractor,
    text_col: str,
    date_col: str,
    output_dir: str,
    logger: logging.Logger,
) -> Optional[pd.DataFrame]:
    """Run all 4 stages for a single calendar day and return the final dataframe
    (or None if there was nothing to process / an error occurred)."""

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
    # Stage 2a: temporal extraction (future date) - cheap, regex-based,
    # runs on every row.
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
    # EventClusterer only ever clusters rows that have BOTH a future
    # date AND a location, so there is no point running NER on rows
    # that already failed the future-date check - this is the main
    # lever for cutting the per-day runtime down.
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
        future_subset = location_extractor.extract_entities(future_subset, text_col=text_col)
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
    # Stage 3: event clustering. Only messages that have BOTH a future
    # date and a location are actually clustered - that filtering is
    # done internally by EventClusterer.
    # --------------------------------------------------------------- #
    logger.info(f"[Day {day_index}/{total_days}] Stage 3/4 (Event Clustering) starting ...")
    day_df = process_clusters(day_df, text_col=text_col)
    n_clustered = day_df["cluster_id"].notna().sum()
    logger.info(f"[Day {day_index}/{total_days}] Stage 3/4 complete: {n_clustered} row(s) assigned to a cluster.")
    save_stage_output(day_df, day_out_dir, "03_event_clustered", logger)

    # --------------------------------------------------------------- #
    # Stage 4: semantic post-processing (denoise / dedupe / merge)
    # --------------------------------------------------------------- #
    logger.info(f"[Day {day_index}/{total_days}] Stage 4/4 (Cluster Post-Processing) starting ...")
    day_df = post_process_clusters(day_df, text_col=text_col)
    n_final = day_df["cluster_id"].notna().sum()
    logger.info(f"[Day {day_index}/{total_days}] Stage 4/4 complete: {n_final} row(s) remain clustered.")
    save_stage_output(day_df, day_out_dir, "04_post_processed", logger)

    logger.info(f"[Day {day_index}/{total_days}] Finished processing {day_start}.")
    return day_df


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
    logger.info(f"Device           : {args.device}")
    logger.info(f"Output directory : {args.output_dir}")
    logger.info("=" * 70)

    # ------------------------------------------------------------------- #
    # Instantiate the heavy / stateful components ONCE. Loading the NER
    # model is expensive, so it must not be reloaded for every single day.
    # ------------------------------------------------------------------- #
    DB_NAME = "telegram"
    TABLE_NAME = "posts"
    # loader = DataLoader(db_name=args.db_name, table_name=args.table_name)
    loader = DataLoader(db_name=DB_NAME, table_name=TABLE_NAME)

    logger.info("Loading location NER model ...")
    try:
        location_extractor = LocationExtractor(
            model_path=args.location_model_path,
            device=args.device,
        )
    except Exception:
        logger.exception("Failed to load the location NER model. Aborting pipeline.")
        sys.exit(1)
    logger.info("Location NER model loaded successfully.")

    day_pairs = list(daterange(args.start_date, args.end_date))
    total_days = len(day_pairs)
    logger.info(f"Pipeline will process {total_days} day(s).")

    all_days_final = []
    for day_index, (day_start, day_end) in enumerate(day_pairs, start=1):
        result_df = process_single_day(
            day_start=day_start,
            day_end=day_end,
            day_index=day_index,
            total_days=total_days,
            loader=loader,
            location_extractor=location_extractor,
            text_col=args.text_col,
            date_col=args.date_col,
            output_dir=args.output_dir,
            logger=logger,
        )
        if result_df is not None:
            all_days_final.append(result_df)

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
    logger.info("=" * 70)


if __name__ == "__main__":
    main()