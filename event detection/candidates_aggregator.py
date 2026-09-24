

import os
import json
import random
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from clickhouse_connect import get_client


def _as_datetime(value):
    return pd.Timestamp(value).to_pydatetime()


def _split_keywords(keywords_str):
    """candidate_clusters.keywords is stored as a comma-joined string
    (', '.join(...) in data_writer.py), not an array - split it back out."""
    if not keywords_str:
        return []
    return [k.strip() for k in keywords_str.split(',') if k.strip()]


def _reasons_from_flags(cand_row):
    reasons = []
    if cand_row.get('is_large'):
        reasons.append('Dominant')
    if cand_row.get('has_burst'):
        reasons.append('Spike')
    if cand_row.get('is_cohesive'):
        reasons.append('Cohesive_Large')
    return reasons


def _growth_rate(daily_sizes: dict):
    """Slope of a linear fit over the story's daily total size. 0.0 if
    there's only one distinct day (a slope needs at least two points)."""
    if len(daily_sizes) < 2:
        return 0.0
    dates = sorted(daily_sizes.keys())
    x = np.array([(d - dates[0]).days for d in dates], dtype=np.float64)
    y = np.array([daily_sizes[d] for d in dates], dtype=np.float64)
    slope = float(np.polyfit(x, y, 1)[0])
    return slope


def _avg_pairwise_similarity(member_vecs):
    """True all-pairs average cosine similarity between member centroids.
    Story membership counts are small (tens, not thousands), so the O(n^2)
    similarity matrix is cheap - no need to approximate it."""
    if len(member_vecs) < 2:
        return 1.0
    matrix = np.vstack(member_vecs)
    sim_matrix = cosine_similarity(matrix)
    n = len(member_vecs)
    triu = np.triu_indices(n, k=1)
    return float(np.mean(sim_matrix[triu]))


def _emotion_volatility(emotion_counter: Counter):
    """Normalized entropy over the emotion distribution (against the fixed
    7-label emotion space, same convention as the emotional_cohesion idea
    discussed earlier) - 0 means every linked cluster shared the same
    dominant emotion, closer to 1 means the story's emotional reaction kept
    shifting from day to day."""
    if not emotion_counter:
        return 0.0
    counts = np.array(list(emotion_counter.values()), dtype=np.float64)
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum() / np.log(7))

def _importance_score(total_messages, days_active, avg_pairwise_similarity,
                       confirmed_count, rejected_count,
                       v_max=50_000, d_max=14, w_volume=0.7, w_persist=0.3):
    """0-100 importance/priority score for a story, combining reach,
    persistence, topical cohesion and LLM validation confidence.
    Emotion fields are intentionally excluded (too many NaNs to be a
    reliable signal - see the discussion this was derived from).

    - volume/persistence are log-compressed against v_max/d_max so a
      handful of huge stories don't flatten everything else; v_max/d_max
      are saturation references, not hard caps, and are worth
      recalibrating periodically (e.g. off the p99 of total_messages)
      as real traffic volumes drift.
    - validation uses Laplace/Beta(1,1) smoothing over confirmed vs
      rejected counts rather than the raw confirmation_ratio field,
      because confirmation_ratio is 0.0 both when a story was fully
      *rejected* and when it simply has no reviews yet - those are very
      different situations and smoothing tells them apart (unreviewed
      settles near 0.5, not 0.0).
    - avg_pairwise_similarity acts as a quality gate: a large story
      whose linked clusters drifted apart topically is more likely a bad
      cross-day merge than a genuinely important event, so low cohesion
      pulls the score down multiplicatively rather than just averaging in.
    """
    n_volume = min(1.0, np.log1p(total_messages) / np.log1p(v_max))
    n_persist = min(1.0, np.log1p(days_active) / np.log1p(d_max))
    magnitude = w_volume * n_volume + w_persist * n_persist

    validation = (confirmed_count + 1) / (confirmed_count + rejected_count + 2)
    cohesion = max(0.0, min(1.0, avg_pairwise_similarity))

    return float(100.0 * magnitude * validation * cohesion)


_EMOTION_LABELS  = {
        # 0 - neutral / خنثی
        'خنثی': 0, 'neutral': 0,
        # 1 - happiness / شادی
        'شادی': 1, 'خوشحالی': 1, 'happy': 1, 'happiness': 1, 'joy': 1,
        # 2 - sadness / غم
        'غم': 2, 'غمگین': 2, 'ناراحتی': 2, 'sad': 2, 'sadness': 2, 'grief': 2,
        # 3 - anger / عصبانیت
        'عصبانیت': 3, 'خشم': 3, 'anger': 3, 'angry': 3,
        # 4 - fear / ترس
        'ترس': 4, 'fear': 4, 'afraid': 4, 'scared': 4,
        # 5 - hate / نفرت
        'نفرت': 5, 'انزجار': 5, 'hate': 5, 'hatred': 5, 'disgust': 5,
        # 6 - surprise / تعجب
        'تعجب': 6, 'شگفتی': 6, 'surprise': 6, 'surprised': 6,
    }
class ClusterStoryBuilder:
    """
    Links candidate_clusters rows into cross-day "story" threads by
    centroid similarity, and writes aggregated stats per story into
    cluster_stories. See the module docstring for the full design and the
    one-time CREATE TABLE / ALTER TABLE this depends on.
    """

    _ID_RANGE = (1_000, 2_000_000_000)

    def __init__(self, db_name="raya_sepehr_analytical",
                 candidate_table="candidate_clusters",
                 story_table="story_clusters",
                 similarity_threshold=0.65):
        self.db_name = db_name
        self.candidate_table = candidate_table
        self.story_table = story_table
        # Slightly looser than CandidateExtractor's merge_threshold (0.75) -
        # a topic drifting slightly over several days is still the same
        # story, whereas merge_threshold is about near-duplicates *within*
        # a single day.
        self.similarity_threshold = similarity_threshold
        self._client = None
        self._candidate_cols = None

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

    # ---- watermark ----

    def _get_scan_start_date(self, client):
        """Where this run should start reading from candidate_clusters.
        See the WATERMARK section of the module docstring for the full
        reasoning. Returns None only if candidate_clusters has no usable
        rows at all yet."""
        try:
            result = client.query(f"""
                SELECT max(c.execution_time)
                FROM {self.candidate_table} AS c
                WHERE c.id IN (
                    SELECT arrayJoin(linked_cluster_ids) FROM {self.story_table} FINAL
                )
            """)
            max_linked = result.result_rows[0][0] if result.result_rows else None
        except Exception:
            # cluster_stories probably doesn't exist yet - treat as "nothing
            # linked so far", handled by the fallback below.
            max_linked = None

        if max_linked:
            return _as_datetime(max_linked).date()

        # Nothing has ever been linked (first run, or an empty stories
        # table) - start from the very first candidate that exists.
        result = client.query(f"""
            SELECT min(execution_time) FROM {self.candidate_table}
            WHERE length(centroid_embedding) > 0
        """)
        min_date = result.result_rows[0][0] if result.result_rows else None
        return _as_datetime(min_date).date() if min_date else None

    # ---- reading candidate_clusters ----

    def _candidate_columns(self, client):
        """Cached per instance. Adds 'dominant_emotion' only if that column
        actually exists yet - see the module docstring."""
        if self._candidate_cols is not None:
            return self._candidate_cols

        base_cols = ['id', 'keywords', 'cluster_size', 'is_large', 'has_burst',
                     'is_cohesive', 'execution_time', 'validation_status',
                     'centroid_embedding']
        try:
            client.query(f"SELECT dominant_emotion FROM {self.candidate_table} LIMIT 0")
            base_cols.append('dominant_emotion')
        except Exception:
            print(f"[StoryAggregator] '{self.candidate_table}' has no 'dominant_emotion' "
                  f"column yet - dominant_emotion_overall/emotion_volatility will be "
                  f"written empty/0 until that field is added.")

        self._candidate_cols = base_cols
        return base_cols

    def _fetch_already_linked_ids(self, client):
        try:
            result = client.query(f"SELECT linked_cluster_ids FROM {self.story_table} FINAL")
        except Exception as e:
            print(f"[StoryAggregator] could not read {self.story_table} - has it been "
                  f"created yet? (see the CREATE TABLE statement in this file's "
                  f"docstring): {e}")
            return set()
        seen = set()
        for (ids,) in result.result_rows:
            seen.update(ids)
        return seen

    def _fetch_new_candidates(self, client, start_date, already_linked_ids):
        """Every not-yet-linked candidate from `start_date` (the watermark)
        onward - no fixed lookback window, so a gap of any size gets
        picked up in one pass."""
        cols = self._candidate_columns(client)
        query = f"""
            SELECT {', '.join(cols)}
            FROM {self.candidate_table}
            WHERE execution_time >= toDate('{start_date.isoformat()}')
              AND length(centroid_embedding) > 0
            ORDER BY execution_time ASC
        """
        result = client.query(query)
        rows = [dict(zip(cols, row)) for row in result.result_rows]
        return [r for r in rows if r['id'] not in already_linked_ids]

    def _fetch_candidates_by_ids(self, client, ids):
        if not ids:
            return []
        cols = self._candidate_columns(client)
        id_list = ",".join(str(int(i)) for i in ids)
        query = f"SELECT {', '.join(cols)} FROM {self.candidate_table} WHERE id IN ({id_list})"
        result = client.query(query)
        return [dict(zip(cols, row)) for row in result.result_rows]

    # ---- reading cluster_stories (linking targets only) ----

    def _fetch_open_stories(self, client, scan_start):
        """Stories still eligible to receive a new member going into
        `scan_start` - ones that picked up a member on the day right
        before it, or that a previous partial run already advanced up to
        `scan_start` itself. Anything quieter than that is not fetched at
        all - per the CLOSING RULE, a story that already missed a day is
        closed for good and is never offered as a target again."""
        cutoff = scan_start - timedelta(days=1)
        try:
            query = f"""
                SELECT story_id, centroid_embedding, linked_cluster_ids, last_seen_date
                FROM {self.story_table} FINAL
                WHERE last_seen_date >= toDate('{cutoff.isoformat()}')
            """
            result = client.query(query)
        except Exception:
            return []
        cols = ['story_id', 'centroid_embedding', 'linked_cluster_ids', 'last_seen_date']
        return [dict(zip(cols, row)) for row in result.result_rows]

    def _alloc_story_id(self, used_ids):
        candidate_id = random.randint(*self._ID_RANGE)
        while candidate_id in used_ids:
            candidate_id = random.randint(*self._ID_RANGE)
        return candidate_id

    # ---- linking ----

    def _run_linking_simulation(self, new_candidates, open_stories, scan_start, scan_end):
        """
        Walks day by day from `scan_start` to `scan_end` (inclusive),
        replaying story membership chronologically instead of linking
        everything in one flat pass. Two things this makes correct at the
        same time:

          1. CLOSING RULE - a story is only a valid linking target as long
             as it keeps gaining a member every day it's alive. Simulating
             day by day is what makes "missed a day" well-defined even
             when a single run has to catch up on several days at once
             (see the WATERMARK section of the module docstring).
          2. Re-touching a day that was already fully linked by a previous
             run (so it has zero *new* candidates now) must NOT look like
             "the story went quiet that day" - each story's last_seen_date
             is tracked and only compared against, never blindly reset by
             an empty day.

        Returns {story_id: set(all linked candidate ids)} for every story
        that gained at least one new member this run (brand-new stories
        included). A story that only got closed, with no new members,
        isn't included - there's nothing new to persist for it.
        """
        candidates_by_date = defaultdict(list)
        for cand in new_candidates:
            cand_vec = np.array(cand['centroid_embedding'], dtype=np.float32)
            if cand_vec.size == 0 or not np.any(cand_vec):
                print(f"[StoryAggregator] candidate id={cand['id']} has an empty "
                      f"centroid_embedding - skipping.")
                continue
            d = _as_datetime(cand['execution_time']).date()
            candidates_by_date[d].append((cand, cand_vec))

        story_centroids = {
            s['story_id']: np.array(s['centroid_embedding'], dtype=np.float32)
            for s in open_stories
        }
        story_members = {
            s['story_id']: set(s['linked_cluster_ids']) for s in open_stories
        }
        # last day each story actually gained a member - the only thing
        # that decides whether it's still open.
        last_seen = {s['story_id']: s['last_seen_date'] for s in open_stories}
        used_ids = set(story_centroids.keys())
        touched = {}

        day = scan_start
        while day <= scan_end:
            for cand, cand_vec in candidates_by_date.get(day, []):
                best_id, best_sim = None, -1.0
                for sid, vec in story_centroids.items():
                    if sid not in last_seen:
                        continue  # already closed earlier in this same run
                    sim = float(cosine_similarity(cand_vec.reshape(1, -1), vec.reshape(1, -1))[0, 0])
                    if sim > best_sim:
                        best_sim, best_id = sim, sid

                if best_id is not None and best_sim >= self.similarity_threshold:
                    story_members[best_id].add(cand['id'])
                    last_seen[best_id] = day
                    touched[best_id] = story_members[best_id]
                else:
                    new_id = self._alloc_story_id(used_ids)
                    used_ids.add(new_id)
                    story_members[new_id] = {cand['id']}
                    story_centroids[new_id] = cand_vec
                    last_seen[new_id] = day
                    touched[new_id] = story_members[new_id]

            # CLOSING RULE: anything that didn't reach `day` is done.
            for sid in list(last_seen.keys()):
                if last_seen[sid] < day:
                    del last_seen[sid]

            day += timedelta(days=1)

        return touched

    # ---- full recompute of one story's aggregates ----

    def _build_story_record(self, story_id, member_rows, exec_dt):
        member_rows = sorted(member_rows, key=lambda r: r['execution_time'])
        vecs = [np.array(r['centroid_embedding'], dtype=np.float32) for r in member_rows]
        vecs = [v for v in vecs if v.size > 0]
        if not vecs:
            print(f"[StoryAggregator] story {story_id}: no members with a usable "
                  f"centroid - skipping.")
            return None

        dates = [_as_datetime(r['execution_time']).date() for r in member_rows]

        daily_sizes = defaultdict(int)
        keyword_counter = Counter()
        reasons_counter = Counter()
        emotion_counter = Counter()
        confirmed = rejected = pending = 0

        for r, d in zip(member_rows, dates):
            daily_sizes[d] += int(r.get('cluster_size', 0) or 0)
            keyword_counter.update(_split_keywords(r.get('keywords', '')))
            reasons_counter.update(_reasons_from_flags(r))
            if r.get('dominant_emotion') is not None:
                emotion_counter.update([r['dominant_emotion']])
            status = r.get('validation_status')
            if status == 'confirmed':
                confirmed += 1
            elif status == 'rejected':
                rejected += 1
            else:
                pending += 1

        centroid_embedding = np.mean(vecs, axis=0)
        first_centroid = vecs[0]
        drift = 1.0 - float(cosine_similarity(
            first_centroid.reshape(1, -1), centroid_embedding.reshape(1, -1))[0, 0])

        peak_date, peak_size = max(daily_sizes.items(), key=lambda kv: kv[1])

        total_messages = int(sum(daily_sizes.values()))
        avg_pairwise_sim = _avg_pairwise_similarity(vecs)
        importance_score = _importance_score(
            total_messages=total_messages,
            days_active=len(set(dates)),
            avg_pairwise_similarity=avg_pairwise_sim,
            confirmed_count=confirmed,
            rejected_count=rejected,
        )

        return {
            'story_id': story_id,
            'representative_keywords': [k for k, _ in keyword_counter.most_common(15)],
            'centroid_embedding': centroid_embedding.tolist(),
            'linked_cluster_ids': [int(r['id']) for r in member_rows],
            'first_seen_date': dates[0],
            'last_seen_date': dates[-1],
            'days_active': len(set(dates)),
            'total_messages': total_messages,
            'peak_date': peak_date,
            'peak_size': int(peak_size),
            'growth_rate': _growth_rate(daily_sizes),
            'avg_pairwise_similarity': avg_pairwise_sim,
            'centroid_drift': drift,
            'confirmed_count': confirmed,
            'rejected_count': rejected,
            'pending_count': pending,
            'confirmation_ratio': (confirmed / (confirmed + rejected)) if (confirmed + rejected) > 0 else 0.0,
            'importance_score': importance_score,
            'dominant_emotion_overall': _EMOTION_LABELS.get(emotion_counter.most_common(1)[0][0], '') if emotion_counter else '',
            'emotion_volatility': _emotion_volatility(emotion_counter),
            'reasons_frequency': dict(reasons_counter),
            'last_updated': exec_dt,
        }

    # ---- writing ----

    def _save_stories(self, client, records):
        if not records:
            return
        column_names = [
            'story_id', 'representative_keywords', 'centroid_embedding',
            'linked_cluster_ids', 'first_seen_date', 'last_seen_date',
            'days_active', 'total_messages', 'peak_date', 'peak_size',
            'growth_rate', 'avg_pairwise_similarity', 'centroid_drift',
            'confirmed_count', 'rejected_count', 'pending_count',
            'confirmation_ratio', 'importance_score', 'dominant_emotion_overall',
            'emotion_volatility', 'reasons_frequency', 'last_updated',
        ]
        rows = []
        for r in records:
            rows.append([
                r['story_id'], r['representative_keywords'], r['centroid_embedding'],
                r['linked_cluster_ids'], r['first_seen_date'], r['last_seen_date'],
                int(r['days_active']), int(r['total_messages']), r['peak_date'],
                int(r['peak_size']), float(r['growth_rate']),
                float(r['avg_pairwise_similarity']), float(r['centroid_drift']),
                int(r['confirmed_count']), int(r['rejected_count']), int(r['pending_count']),
                float(r['confirmation_ratio']), float(r['importance_score']),
                r['dominant_emotion_overall'] or '',
                float(r['emotion_volatility']),
                json.dumps(r['reasons_frequency'], ensure_ascii=False),
                r['last_updated'],
            ])
        try:
            client.insert(self.story_table, rows, column_names=column_names)
            print(f"[StoryAggregator] inserted/updated {len(rows)} row(s) in {self.story_table}")
        except Exception as e:
            print(f"[StoryAggregator] ERROR inserting into {self.story_table}: {e}")

    # ---- entry point ----

    def run(self, execution_time=None):
        """execution_time: optional datetime to treat as "now" (defaults to
        the real current time) - mainly useful for backfilling/testing
        against a specific date. It sets the *end* of the scan window; the
        start is always the watermark (see _get_scan_start_date)."""
        exec_dt = _as_datetime(execution_time) if execution_time else datetime.now()
        client = self._get_client()

        scan_start = self._get_scan_start_date(client)
        if scan_start is None:
            print(f"[StoryAggregator] '{self.candidate_table}' has no usable rows yet - "
                  f"nothing to do.")
            return
        if scan_start > exec_dt.date():
            print(f"[StoryAggregator] watermark ({scan_start}) is after the requested "
                  f"execution date ({exec_dt.date()}) - nothing to do.")
            return

        already_linked = self._fetch_already_linked_ids(client)
        new_candidates = self._fetch_new_candidates(client, scan_start, already_linked)
        if not new_candidates:
            print(f"[StoryAggregator] no new candidate_clusters found since {scan_start} - "
                  f"nothing to do.")
            return

        open_stories = self._fetch_open_stories(client, scan_start)
        touched = self._run_linking_simulation(new_candidates, open_stories, scan_start, exec_dt.date())
        if not touched:
            print("[StoryAggregator] nothing linked this run.")
            return

        updated_records = []
        for story_id, member_ids in touched.items():
            member_rows = self._fetch_candidates_by_ids(client, member_ids)
            record = self._build_story_record(story_id, member_rows, exec_dt)
            if record is not None:
                updated_records.append(record)

        self._save_stories(client, updated_records)
        print(f"[StoryAggregator] done: {len(new_candidates)} new candidate(s) linked "
              f"into {len(updated_records)} story(ies) this run "
              f"(scanned {scan_start} -> {exec_dt.date()}).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Link candidate_clusters into cross-day story threads and "
                    "write aggregated stats into cluster_stories. Safe to run "
                    "manually any time, or wire into a daily cron job. The scan "
                    "range is derived automatically from what's already linked "
                    "(see the WATERMARK section in this file's docstring) - "
                    "there is no lookback window to configure."
    )
    parser.add_argument("--similarity-threshold", type=float, default=0.65)
    args = parser.parse_args()

    builder = ClusterStoryBuilder(
        similarity_threshold=args.similarity_threshold,
    )
    try:
        builder.run()
    finally:
        builder.close()