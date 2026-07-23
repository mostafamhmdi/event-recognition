import os
import gc
import json
import pickle
import re
import csv
import time
from datetime import datetime

import numpy as np
import pandas as pd
import jdatetime
import torch

from data_loader import DataLoader
from classifier import TookaClassifier, print_resource_usage
from clustering import SocialMediaEpsilonClustering
from candidate_extractor import CandidateExtractor
from ev import QwenEventVerifier
from utils import similarity_model
from preprocessing import dynamic_preprocess, PreprocessingOptions  # not wired in yet - kept for a future phase


def now_str():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def append_phase1_report_txt(daily_item, filename="report_phase1_classification.txt"):
    with open(filename, "a", encoding="utf-8") as f:
        g_date = daily_item["date"].strftime('%Y-%m-%d')
        sh_date = jdatetime.datetime.fromgregorian(datetime=daily_item["date"]).strftime("%Y/%m/%d")
        f.write(f"Date (Jalali): {sh_date} | (Gregorian): {g_date}\n")
        f.write(f"Total raw messages: {daily_item['raw_count']} | Filtered out: {daily_item['filtered_count']}\n")
        f.write("-" * 70 + "\n")
        for i, (msg, ts) in enumerate(zip(daily_item["valid_clean_msgs"], daily_item["valid_timestamps"]), 1):
            ts_str = ts.strftime('%H:%M:%S') if hasattr(ts, 'strftime') else str(ts)
            f.write(f" {i}. [{ts_str}] {msg}\n")
        f.write("\n" + "=" * 70 + "\n\n")


def parse_phase1_report(filename="report_phase1_classification.txt"):
    
    if not os.path.exists(filename):
        raise FileNotFoundError(
            f"'{filename}' not found. Run phase 1 for real at least once first "
            f"(set USE_PRECOMPUTED_PHASE1 = False), or check the path."
        )

    with open(filename, "r", encoding="utf-8") as f:
        content = f.read()

    blocks = [b.strip() for b in content.split("=" * 70) if b.strip()]

    # accepts both "Passed filter:" (older wording, kept for files already
    # generated) and "Filtered out:" (the corrected wording above) - the
    # number itself was always the filtered/rejected count either way.
    header_re = re.compile(
        r"Date \(Jalali\):\s*(?P<shamsi>\S+)\s*\|\s*\(Gregorian\):\s*(?P<gregorian>\d{4}-\d{2}-\d{2})\s*\n"
        r"Total raw messages:\s*(?P<raw>\d+)\s*\|\s*(?:Passed filter|Filtered out):\s*(?P<filtered>\d+)"
    )
    msg_re = re.compile(r"^\s*\d+\.\s*\[(?P<time>\d{2}:\d{2}:\d{2})\]\s*(?P<text>.*)$")

    days = []
    for block in blocks:
        header_match = header_re.search(block)
        if not header_match:
            continue

        gregorian_str = header_match.group("gregorian")
        date = pd.to_datetime(gregorian_str)
        raw_count = int(header_match.group("raw"))

        valid_clean_msgs = []
        valid_timestamps = []
        for line in block.splitlines():
            m = msg_re.match(line)
            if m:
                ts = pd.to_datetime(f"{gregorian_str} {m.group('time')}")
                valid_clean_msgs.append(m.group("text"))
                valid_timestamps.append(ts)

        # derived from the actually-parsed lines rather than trusted from
        # the header number, so it can't drift out of sync with the data
        filtered_count = raw_count - len(valid_clean_msgs)

        days.append({
            "date": date,
            "raw_count": raw_count,
            "filtered_count": filtered_count,
            "valid_clean_msgs": valid_clean_msgs,
            "valid_timestamps": valid_timestamps,
        })

    days.sort(key=lambda d: d["date"])
    return days


def append_phase2_report_txt(daily_data, filename="report_phase2_clustering.txt"):
    with open(filename, "a", encoding="utf-8") as f:
        f.write(f"Date (Jalali): {daily_data['date_shamsi']} | "
                f"(Gregorian): {daily_data['date_gregorian'].strftime('%Y-%m-%d')}\n")
        f.write(f"Clusters found (with more than 1 message): {len(daily_data['communities'])}\n")
        f.write("-" * 70 + "\n")
        valid_msgs = daily_data["valid_texts"]
        valid_timestamps = daily_data["valid_timestamps"]
        for cluster_idx, comm_indices in enumerate(daily_data["communities"], 1):
            f.write(f"\n Cluster #{cluster_idx} ({len(comm_indices)} message(s)):\n")
            for idx in comm_indices:
                msg = valid_msgs[idx]
                ts = valid_timestamps[idx]
                ts_str = ts.strftime('%H:%M:%S') if hasattr(ts, 'strftime') else str(ts)
                f.write(f"    - [{ts_str}] {msg}\n")
        f.write("\n" + "=" * 70 + "\n\n")


def append_phase3_report_txt(day_candidates, date_str, filename="report_phase3_candidates.txt"):
    with open(filename, "a", encoding="utf-8") as f:
        f.write(f"Candidates for date: {date_str} (count: {len(day_candidates)})\n")
        f.write("-" * 70 + "\n")
        for i, cand in enumerate(day_candidates, 1):
            f.write(f" Candidate #{i}:\n")
            f.write(f"   Cluster size: {cand.get('size', 'N/A')}\n")
            f.write(f"   Share of day's messages: {cand.get('daily_ratio_pct', 'N/A')}%\n")
            f.write(f"   Reasons: {cand.get('reasons', [])}\n")
            f.write(f"   Metrics: {cand.get('metrics', {})}\n")
            f.write("   Sample messages:\n")
            messages = cand.get('messages', [])
            for msg_text in messages[:10]:
                f.write(f"      - {msg_text}\n")
            if len(messages) > 10:
                f.write(f"      ... ({len(messages) - 10} more message(s))\n")
            f.write("\n")
        f.write("=" * 70 + "\n\n")


def append_phase4_report_csv(day_verified_events, date_str, filename="report_phase4_verified_events.csv"):
    
    file_is_empty = (not os.path.exists(filename)) or os.path.getsize(filename) == 0
    with open(filename, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        if file_is_empty:
            writer.writerow([
                "date_shamsi", "event_index", "cluster_size", "daily_ratio_pct",
                "title", "summary", "sample_messages"
            ])
        for i, event in enumerate(day_verified_events, 1):
            messages = event.get('messages', [])
            sample_messages = " | ".join(messages[:10])
            if len(messages) > 10:
                sample_messages += f" | ... ({len(messages) - 10} more message(s))"
            writer.writerow([
                date_str,
                i,
                event.get('size', 'N/A'),
                event.get('daily_ratio_pct', 'N/A'),
                event.get('event_title', 'N/A'),
                event.get('event_summary', 'N/A'),
                sample_messages,
            ])


def load_light_models(use_precomputed_phase1, classifier_path,
                       clustering_threshold, clustering_max_block_bytes):
    
    classifier = None
    if not use_precomputed_phase1:
        print(f"[Main] Loading classifier model at {now_str()}...")
        classifier = TookaClassifier(model_path=classifier_path)

    print(f"[Main] Loading embedding model at {now_str()}...")
    similarity_model.reload_model()

    clusterer = SocialMediaEpsilonClustering(
        threshold=clustering_threshold,
        max_block_bytes=clustering_max_block_bytes
    )
    extractor = CandidateExtractor()

    warmup_emb = np.asarray(similarity_model.encode_texts(["warmup"]))
    embedding_dim = warmup_emb.shape[-1]
    print(f"[Main] Embedding dimension detected: {embedding_dim}")

    return classifier, clusterer, extractor, embedding_dim


def unload_light_models(classifier, clusterer, extractor):
    
    if classifier is not None:
        del classifier

    if similarity_model.model is not None:
        del similarity_model.model
        similarity_model.model = None

    if clusterer is not None:
        del clusterer
    if extractor is not None:
        del extractor

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return None, None, None  # classifier, clusterer, extractor


def load_qwen_verifier(model_path, sample_size, max_new_tokens, max_input_tokens,
                        max_consecutive_oom_before_abort):
    
    print(f"\n[Main] Loading Qwen event verifier at {now_str()}...")
    print_resource_usage("Before loading Qwen verifier")
    try:
        verifier = QwenEventVerifier(
            model_path=model_path,
            sample_size=sample_size,
            max_new_tokens=max_new_tokens,
            max_input_tokens=max_input_tokens,
            max_consecutive_oom_before_abort=max_consecutive_oom_before_abort
        )
        print_resource_usage("After loading Qwen verifier")
        return verifier
    except Exception as e:
        print(f"[Main] ERROR: could not load the Qwen verifier ({e}).")
        print("[Main] Phase 4 will be skipped for this day.")
        return None


def unload_qwen_verifier(verifier):
    """Fully releases Qwen's VRAM so the light models can be reloaded for
    the next day."""
    if verifier is not None:
        del verifier
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return None


def main():
    DB_NAME = "telegram"
    TABLE_NAME = "posts"
    CLASSIFIER_PATH = "../../models/tooka_bert_classifer/tooka_fine_funed_1"

    # ---- Phase 1 skip switch ----
    # Classification is the slowest stage. If you already have
    # report_phase1_classification.txt from a previous real run and just
    # want to iterate on phases 2-4, set this to True: phase 1 (and the
    # classifier model itself) is skipped entirely and its output is
    # parsed straight from that file instead. Set back to False to run
    # phase 1 for real (e.g. on new/different data).
    USE_PRECOMPUTED_PHASE1 = True
    PHASE1_REPORT_PATH = "report_phase1_classification.txt"

    CLASSIFIER_BATCH_SIZE = 128
    EMBEDDING_BATCH_SIZE = 128
    CLUSTERING_THRESHOLD = 0.55

    CLUSTERING_MAX_BLOCK_BYTES = 512 * 1024 * 1024

    QWEN_MODEL_PATH = "../../models/qwen/"
    QWEN_SAMPLE_SIZE = 20
    QWEN_MAX_NEW_TOKENS = 256
    QWEN_MAX_INPUT_TOKENS = 4600
    # How many candidates in a row have to completely fail (exhaust their
    # own retries with no valid JSON) before ev.py's verify_candidates
    # gives up early on the rest of that day - see verify_candidates'
    # docstring in ev.py. Matches ev.py's own default (5); listed here
    # explicitly so it's one obvious place to tune without touching ev.py.
    QWEN_MAX_CONSECUTIVE_OOM_BEFORE_ABORT = 5

    script_start = time.time()
    print(f"[Main] Script started at {now_str()}")
    print(f"[Main] USE_PRECOMPUTED_PHASE1 = {USE_PRECOMPUTED_PHASE1}")
    print_resource_usage("Startup - before loading anything")

    report_files = ["report_phase2_clustering.txt", "report_phase3_candidates.txt",
                     "report_phase4_verified_events.csv"]
    if not USE_PRECOMPUTED_PHASE1:
        report_files.append(PHASE1_REPORT_PATH)
    for file in report_files:
        open(file, "w", encoding="utf-8").close()

    # ---- Build the list of per-day inputs to phase 2 ----
    grouped_data = None
    days_data = None

    if USE_PRECOMPUTED_PHASE1:
        print(f"\n[Main] Skipping classification - loading its output from "
              f"'{PHASE1_REPORT_PATH}'...")
        days_data = parse_phase1_report(PHASE1_REPORT_PATH)
        print(f"[Main] Loaded {len(days_data)} day(s) from the phase 1 report.")
    else:
        loader = DataLoader(db_name=DB_NAME, table_name=TABLE_NAME)
        df = loader.load_and_prepare(text_col='txtContent', date_col='date')

        print("\n[Main] Applying date range filter...")
        mask_range1 = (df['date'] >= '2025-08-23') & (df['date'] <= '2025-08-24 23:59:59')
        mask_range2 = (df['date'] > '2025-08-24') & (df['date'] <= '2025-08-25 23:59:59')
        df = df[mask_range1 | mask_range2].copy()
        print(f"[Main] Rows remaining after filter: {len(df)}")

        grouped_data = df.groupby(pd.Grouper(key='date', freq='1D'))

    # ---- Models are now loaded/unloaded per day, per phase group ----
    # 8GB VRAM isn't enough to hold the embedder and Qwen at the same time,
    # so nothing is preloaded here. Each day in the loop below: load the
    # light models (classifier/embedder/clusterer/extractor) -> run phases
    # 1-3 -> fully release them -> load Qwen -> run phase 4 -> fully release
    # Qwen -> move to the next day. This means every model gets reloaded
    # once per day instead of once per run, but loading is fast and with
    # 8GB VRAM there isn't really another option.
    classifier = None
    clusterer = None
    extractor = None
    verifier = None
    embedder = similarity_model  # module-level singleton from utils.py

    # utils.py loads the embedding model automatically at import time
    # (module-level `similarity_model = SimilarityModel()`), so drop it
    # here to start from the same clean, nothing-loaded state that every
    # day's iteration below starts from.
    if embedder.model is not None:
        del embedder.model
        embedder.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    all_daily_results_for_json = []
    all_final_candidates = []
    all_verified_events = []

    print("\n" + "=" * 60)
    print(f"[Main] STARTING day-by-day processing loop at {now_str()}")
    print("=" * 60)

    total_raw = 0
    total_filtered = 0
    days_processed = 0

    day_iterator = days_data if USE_PRECOMPUTED_PHASE1 else grouped_data

    for day_item in day_iterator:
        if USE_PRECOMPUTED_PHASE1:
            date = day_item["date"]
            raw_count = day_item["raw_count"]
            filtered_count = day_item["filtered_count"]
            valid_clean_msgs = day_item["valid_clean_msgs"]
            valid_timestamps = day_item["valid_timestamps"]
        else:
            date, group = day_item
            if group.empty:
                continue
            raw_count = len(group)

        day_start = time.time()
        j_date = jdatetime.datetime.fromgregorian(datetime=date).strftime("%Y/%m/%d")
        print(f"\n[Main] --- Processing day: {j_date} ({date.strftime('%Y-%m-%d')}) | "
              f"{raw_count} raw messages | started {now_str()} ---")

        # ---- Load the light models (classifier/embedder/clusterer/extractor) for this day ----
        classifier, clusterer, extractor, EMBEDDING_DIM = load_light_models(
            USE_PRECOMPUTED_PHASE1, CLASSIFIER_PATH,
            CLUSTERING_THRESHOLD, CLUSTERING_MAX_BLOCK_BYTES
        )
        ZERO_VECTOR = [0.0] * EMBEDDING_DIM

        # ---- Phase 1: classification (or load from the precomputed report) ----
        if USE_PRECOMPUTED_PHASE1:
            print(f"[Main] Phase 1 SKIPPED (loaded from '{PHASE1_REPORT_PATH}') | "
                  f"{raw_count} raw -> {len(valid_clean_msgs)} kept, {filtered_count} filtered out")
        else:
            raw_messages = group['txtContent'].tolist()
            raw_timestamps = group['date'].tolist()

            results, filtered_count = classifier.predict_and_filter(
                raw_messages, raw_timestamps, batch_size=CLASSIFIER_BATCH_SIZE
            )
            valid_clean_msgs = [item["clean_text"] for item in results if item["passed_filter"]]
            valid_timestamps = [item["timestamp"] for item in results if item["passed_filter"]]

            append_phase1_report_txt({
                "date": date, "raw_count": raw_count, "filtered_count": filtered_count,
                "valid_clean_msgs": valid_clean_msgs, "valid_timestamps": valid_timestamps
            })

            phase1_elapsed = time.time() - day_start
            print(f"[Main] Phase 1 (classification) done in {phase1_elapsed:.2f}s | "
                  f"{raw_count} raw -> {len(valid_clean_msgs)} kept, {filtered_count} filtered out")
            del results

        total_raw += raw_count
        total_filtered += filtered_count
        days_processed += 1

        # ---- Phase 2: clustering ----
        phase2_start = time.time()
        embeddings = []
        valid_clusters = []

        if len(valid_clean_msgs) > 1:
            print(f"[Main] Generating embeddings for {len(valid_clean_msgs)} messages "
                  f"(batch_size={EMBEDDING_BATCH_SIZE})...")
            for i in range(0, len(valid_clean_msgs), EMBEDDING_BATCH_SIZE):
                sub_batch = valid_clean_msgs[i: i + EMBEDDING_BATCH_SIZE]
                try:
                    embs = embedder.encode_texts(sub_batch)
                    if hasattr(embs, 'cpu'):
                        embs = embs.cpu().numpy()
                    embeddings.extend(np.asarray(embs).tolist())
                except Exception as e:
                    print(f"[Main] ERROR generating embeddings for a batch: {e}. "
                          f"Filling with zero vectors so indices stay aligned.")
                    embeddings.extend([ZERO_VECTOR] * len(sub_batch))

            communities = clusterer.fit_predict(np.array(embeddings, dtype=np.float32))
            valid_clusters = [list(comm) for comm in communities if len(comm) > 1]

        phase2_elapsed = time.time() - phase2_start
        print(f"[Main] Phase 2 (clustering) done in {phase2_elapsed:.2f}s | "
              f"{len(valid_clusters)} cluster(s) with >1 message")

        embeddings_array = np.array(embeddings, dtype=np.float32) if len(embeddings) > 0 else None
        daily_full_data = {
            "date": date,
            "date_shamsi": j_date,
            "date_gregorian": date,
            "raw_count": raw_count,
            "filtered_count": filtered_count,
            "valid_texts": valid_clean_msgs,
            "valid_timestamps": valid_timestamps,
            "embeddings": embeddings_array,
            "communities": valid_clusters,
        }
        append_phase2_report_txt(daily_full_data)

        # ---- Phase 3: candidate extraction ----
        phase3_start = time.time()
        day_candidates = extractor.process_daily_results([daily_full_data])
        all_final_candidates.extend(day_candidates)
        append_phase3_report_txt(day_candidates, j_date)
        phase3_elapsed = time.time() - phase3_start
        print(f"[Main] Phase 3 (candidate extraction) done in {phase3_elapsed:.2f}s | "
              f"{len(day_candidates)} candidate(s)")

        # ---- Release the light models before loading Qwen ----
        # Classifier + embedder + Qwen together don't fit in 8GB VRAM, so
        # the light models have to be fully out of memory before Qwen loads.
        classifier, clusterer, extractor = unload_light_models(classifier, clusterer, extractor)

        # ---- Phase 4: LLM event verification - now runs right here, per day ----
        verifier = load_qwen_verifier(
            QWEN_MODEL_PATH, QWEN_SAMPLE_SIZE, QWEN_MAX_NEW_TOKENS, QWEN_MAX_INPUT_TOKENS,
            QWEN_MAX_CONSECUTIVE_OOM_BEFORE_ABORT
        )
        phase4_start = time.time()
        day_verified_events = []

        # ev.py's verify_candidates reads candidate['cluster_id'] and
        # candidate['messages'] directly (not .get(...)) - if
        # CandidateExtractor ever changes its output schema, or a candidate
        # is somehow missing messages, that would raise a KeyError partway
        # through the day's candidates and (per the "don't touch ev.py"
        # instruction) that's not something to patch inside ev.py itself.
        # Normalizing here instead guarantees every candidate handed to it
        # has what it needs, so phase 4 degrades gracefully (skips a
        # malformed candidate) instead of raising and losing every already
        # -verified candidate for the day.
        verifiable_candidates = []
        for i, cand in enumerate(day_candidates, 1):
            if not cand.get('messages'):
                print(f"[Main] WARNING: candidate #{i} for {j_date} has no messages - "
                      f"skipping it in Phase 4.")
                continue
            cand.setdefault('cluster_id', f"{j_date}-{i}")
            verifiable_candidates.append(cand)

        if verifiable_candidates and verifier is not None:
            try:
                day_verified_events = verifier.verify_candidates(verifiable_candidates)
            except Exception as e:
                print(f"[Main] ERROR during Phase 4 verification for {j_date} ({e}). "
                      f"Continuing with the next day.")
        all_verified_events.extend(day_verified_events)
        append_phase4_report_csv(day_verified_events, j_date)

        # ---- Release Qwen; the next day's iteration reloads the light models fresh ----
        verifier = unload_qwen_verifier(verifier)

        phase4_elapsed = time.time() - phase4_start
        print(f"[Main] Phase 4 (event verification) done in {phase4_elapsed:.2f}s | "
              f"{len(day_verified_events)}/{len(day_candidates)} confirmed as events")

        # Keep a lightweight summary (without the real embedding vectors)
        # for the final clusters.json, then drop the heavy embeddings array
        # for this day now that phases 3-4 are done using it.
        light_day_data = {k: v for k, v in daily_full_data.items() if k != 'embeddings'}
        light_day_data['embeddings'] = "not retained in clusters.json (used transiently in phase 2/3)"
        all_daily_results_for_json.append(light_day_data)

        del embeddings, embeddings_array, daily_full_data
        gc.collect()

        day_elapsed = time.time() - day_start
        print(f"[Main] Day {j_date} fully done in {day_elapsed:.2f}s (finished {now_str()})")

    # ---- Safety net: everything should already be unloaded from inside the ----
    # ---- loop, but this covers the empty-loop / early-exit-before-unload case ----
    print(f"\n[Main] All days processed at {now_str()}. Making sure VRAM is clear...")
    classifier, clusterer, extractor = unload_light_models(classifier, clusterer, extractor)
    verifier = unload_qwen_verifier(verifier)
    print("[Main] Models released.")
    print_resource_usage("After releasing all models")

    print("\n" + "=" * 60)
    print(f"[Main] Loop finished - {days_processed} day(s) | {total_raw} raw messages | "
          f"{total_raw - total_filtered} kept | {total_filtered} filtered out | "
          f"{len(all_final_candidates)} candidate(s) | {len(all_verified_events)} verified event(s)")
    print("=" * 60)

    # -------------------------------------
    # Save final JSON output files
    # -------------------------------------
    print(f"\n[Main] Saving final output files at {now_str()}...")

    with open("clusters.json", "w", encoding="utf-8") as f:
        json.dump(all_daily_results_for_json, f, ensure_ascii=False, indent=4, default=str)

    with open("db_candidates_output.json", "w", encoding="utf-8") as f:
        json.dump(all_final_candidates, f, ensure_ascii=False, indent=4, default=str)

    with open("db_verified_events_output.json", "w", encoding="utf-8") as f:
        json.dump(all_verified_events, f, ensure_ascii=False, indent=4, default=str)

    total_elapsed = time.time() - script_start
    print(f"\n[Main] Script finished successfully at {now_str()} - total time "
          f"{total_elapsed:.2f}s ({total_elapsed / 60:.1f} min)")
    print_resource_usage("End of script")


if __name__ == "__main__":
    main()