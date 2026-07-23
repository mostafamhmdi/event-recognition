# import time
# import numpy as np
# import pandas as pd
# import networkx as nx
# from sklearn.metrics.pairwise import cosine_similarity


# class CandidateExtractor:
#     """
#     Extracts "candidate" clusters from a day's clustering results. A cluster
#     only needs to satisfy ONE of three independent rules to be selected:
#       - Dominant: unusually large compared to the other clusters that day
#         (an IQR-based outlier on cluster size)
#       - Spike: its messages are concentrated in a short burst of hours
#       - Cohesive_Large: a large enough share of the day's messages AND
#         tightly clustered around its own centroid
#     """

#     def __init__(self, merge_threshold=0.75, spike_min_peak=3, spike_z_threshold=2.2):
#         self.merge_threshold = merge_threshold
#         self.spike_min_peak = spike_min_peak
#         self.spike_z_threshold = spike_z_threshold

#     @staticmethod
#     def compute_cluster_cohesion(cluster_indices, embeddings):
#         cluster_embs = embeddings[list(cluster_indices)]
#         if len(cluster_embs) <= 1:
#             return 0.0, 0.0, 0.0
#         centroid = np.mean(cluster_embs, axis=0, keepdims=True)
#         sims = cosine_similarity(cluster_embs, centroid).flatten()
#         median_sim = float(np.median(sims))
#         p25_sim = float(np.percentile(sims, 25))
#         cohesion = (0.7 * median_sim + 0.3 * p25_sim)
#         return cohesion, median_sim, p25_sim

#     @staticmethod
#     def compute_cluster_coverage(cluster_indices, embeddings, threshold=0.75):
#         cluster_embs = embeddings[list(cluster_indices)]
#         if len(cluster_embs) <= 1:
#             return 0.0
#         centroid = np.mean(cluster_embs, axis=0, keepdims=True)
#         sims = cosine_similarity(cluster_embs, centroid).flatten()
#         coverage = np.mean(sims >= threshold)
#         return float(coverage)

#     def compute_hourly_spike(self, timestamps):
#         if not timestamps:
#             return 0.0, 0, False

#         counts = [0] * 24
#         for t in timestamps:
#             hour = pd.to_datetime(t).hour
#             counts[hour] += 1

#         peak = max(counts)
#         total_msgs = sum(counts)
#         mean_c = np.mean(counts)
#         std_c = np.std(counts)

#         if peak < self.spike_min_peak:
#             return 0.0, peak, False

#         epsilon = 1.0
#         z_score = (peak - mean_c) / (std_c + epsilon)
#         concentration = peak / total_msgs
#         is_spike = (z_score >= 2.0) and (concentration >= 0.20)
#         return float(z_score), peak, is_spike

#     def _merge_similar_clusters(self, valid_clusters, embeddings):
#         """
#         Merges clusters whose centroids are near-duplicates of each other
#         (cosine similarity >= merge_threshold). Vectorized with numpy
#         instead of a Python double loop over cluster pairs - same
#         threshold and same decision logic, just faster and safer if a
#         day ever produces a lot of clusters.
#         """
#         if len(valid_clusters) <= 1:
#             return valid_clusters

#         centroids = [np.mean(embeddings[c], axis=0) for c in valid_clusters]
#         centroids_matrix = np.vstack(centroids)
#         sim_matrix = cosine_similarity(centroids_matrix)
#         np.fill_diagonal(sim_matrix, 0.0)

#         gi, gj = np.where(sim_matrix >= self.merge_threshold)
#         keep = gi < gj
#         gi, gj = gi[keep], gj[keep]

#         g_merge = nx.Graph()
#         g_merge.add_nodes_from(range(len(valid_clusters)))
#         g_merge.add_edges_from(zip(gi.tolist(), gj.tolist()))

#         components = list(nx.connected_components(g_merge))
#         final_clusters = []
#         for comp in components:
#             merged_indices = []
#             for idx in comp:
#                 merged_indices.extend(valid_clusters[idx])
#             final_clusters.append(list(set(merged_indices)))

#         return final_clusters

#     def process_daily_results(self, all_daily_results: list) -> list:
#         """
#         Takes a list of daily clustering results and extracts candidate
#         clusters. Returns a list of dicts, ready to be stored/serialized.
#         Each item in `all_daily_results` is expected to have:
#           date_gregorian, date_shamsi, valid_texts, valid_timestamps,
#           embeddings (numpy array or None), communities (list of index
#           collections, may include singletons - they're filtered here).
#         """
#         t_start = time.time()
#         final_candidates = []
#         global_cluster_id = 1  # unique within this call; cluster_id also embeds the date

#         for daily_data in all_daily_results:
#             date_gregorian = daily_data['date_gregorian']
#             date_shamsi = daily_data['date_shamsi']
#             texts = daily_data['valid_texts']
#             timestamps = daily_data.get('valid_timestamps', [])
#             embeddings = daily_data['embeddings']
#             clusters = daily_data['communities']

#             valid_clusters = [list(c) for c in clusters if len(c) > 1]
#             if embeddings is None or len(valid_clusters) == 0:
#                 continue

#             # ---- Step 1: merge near-duplicate clusters ----
#             final_clusters = self._merge_similar_clusters(valid_clusters, embeddings)

#             # ---- Step 2: outlier threshold for "dominant" clusters ----
#             total_daily_msgs = len(texts)
#             sizes = [len(c) for c in final_clusters]
#             if len(sizes) > 1:
#                 q1 = np.percentile(sizes, 25)
#                 q3 = np.percentile(sizes, 75)
#                 iqr = q3 - q1
#                 upper_fence = q3 + 1.5 * iqr
#             else:
#                 upper_fence = float('inf')

#             # ---- Step 3: evaluate rules and extract final candidates ----
#             for community in final_clusters:
#                 n = len(community)
#                 cluster_times = [timestamps[i] for i in community] if timestamps else []
#                 cluster_ratio_pct = (n / total_daily_msgs) * 100

#                 z_score, peak_count, _ = self.compute_hourly_spike(cluster_times)
#                 cohesion, median_sim, p25_sim = self.compute_cluster_cohesion(community, embeddings)
#                 coverage = self.compute_cluster_coverage(community, embeddings, threshold=0.75)

#                 rule1_dominant = n > upper_fence
#                 rule2_spike = z_score >= self.spike_z_threshold
#                 rule3_cohesive = (cluster_ratio_pct > 5.0) and (median_sim > 0.75) and (coverage > 0.55)

#                 if rule1_dominant or rule2_spike or rule3_cohesive:
#                     passed_reasons = []
#                     if rule1_dominant:
#                         passed_reasons.append("Dominant")
#                     if rule2_spike:
#                         passed_reasons.append("Spike")
#                     if rule3_cohesive:
#                         passed_reasons.append("Cohesive_Large")

#                     cluster_messages = [texts[i] for i in community]

#                     candidate_record = {
#                         "cluster_id": f"CL_{date_gregorian.strftime('%Y%m%d')}_{global_cluster_id}",
#                         "date_shamsi": date_shamsi,
#                         "date_gregorian": date_gregorian,
#                         "size": n,
#                         "daily_ratio_pct": float(f"{cluster_ratio_pct:.2f}"),
#                         "reasons": passed_reasons,
#                         "metrics": {
#                             "z_score": float(f"{z_score:.2f}"),
#                             "median_sim": float(f"{median_sim:.3f}"),
#                             "coverage": float(f"{coverage:.3f}")
#                         },
#                         "messages": cluster_messages,
#                         "timestamps": cluster_times
#                     }
#                     final_candidates.append(candidate_record)
#                     global_cluster_id += 1

#         elapsed = time.time() - t_start
#         print(f"[CandidateExtractor] process_daily_results done in {elapsed:.2f}s | "
#               f"{len(final_candidates)} candidate(s) extracted")

#         return final_candidates



import time
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity


class CandidateExtractor:
    """
    Extracts candidate clusters from a day's clustering results. A cluster
    only needs to satisfy ONE of three independent rules to be selected:
      - Dominant: unusually large compared to the other clusters that day
        (an IQR-based outlier on cluster size)
      - Spike: its messages are concentrated in a short burst of hours
      - Cohesive_Large: a large enough share of the day's messages AND
        tightly clustered around its own centroid
    """

    def __init__(self, merge_threshold=0.75, spike_min_peak=3, spike_z_threshold=2.2,
                 max_representative_messages=35):
        self.merge_threshold = merge_threshold
        self.spike_min_peak = spike_min_peak
        self.spike_z_threshold = spike_z_threshold
        # Cap on how many messages get stored per candidate. Instead of the
        # full cluster, we keep the messages closest (by cosine similarity)
        # to the cluster's centroid - i.e. the ones that best represent
        # what the cluster is actually about.
        self.max_representative_messages = max_representative_messages

    @staticmethod
    def compute_cluster_cohesion(cluster_indices, embeddings):
        cluster_embs = embeddings[list(cluster_indices)]
        if len(cluster_embs) <= 1:
            return 0.0, 0.0, 0.0
        centroid = np.mean(cluster_embs, axis=0, keepdims=True)
        sims = cosine_similarity(cluster_embs, centroid).flatten()
        median_sim = float(np.median(sims))
        p25_sim = float(np.percentile(sims, 25))
        cohesion = (0.7 * median_sim + 0.3 * p25_sim)
        return cohesion, median_sim, p25_sim

    @staticmethod
    def compute_cluster_coverage(cluster_indices, embeddings, threshold=0.75):
        cluster_embs = embeddings[list(cluster_indices)]
        if len(cluster_embs) <= 1:
            return 0.0
        centroid = np.mean(cluster_embs, axis=0, keepdims=True)
        sims = cosine_similarity(cluster_embs, centroid).flatten()
        coverage = np.mean(sims >= threshold)
        return float(coverage)

    def compute_hourly_spike(self, timestamps):
        if not timestamps:
            return 0.0, 0, False

        counts = [0] * 24
        for t in timestamps:
            hour = pd.to_datetime(t).hour
            counts[hour] += 1

        peak = max(counts)
        total_msgs = sum(counts)
        mean_c = np.mean(counts)
        std_c = np.std(counts)

        if peak < self.spike_min_peak:
            return 0.0, peak, False

        epsilon = 1.0
        z_score = (peak - mean_c) / (std_c + epsilon)
        concentration = peak / total_msgs
        is_spike = (z_score >= 2.0) and (concentration >= 0.20)
        return float(z_score), peak, is_spike

    def _merge_similar_clusters(self, valid_clusters, embeddings):
        """
        Merges clusters whose centroids are near-duplicates of each other
        (cosine similarity >= merge_threshold). Vectorized with numpy
        instead of a Python double loop over cluster pairs - same
        threshold and same decision logic, just faster and safer if a
        day ever produces a lot of clusters.
        """
        if len(valid_clusters) <= 1:
            return valid_clusters

        centroids = [np.mean(embeddings[c], axis=0) for c in valid_clusters]
        centroids_matrix = np.vstack(centroids)
        sim_matrix = cosine_similarity(centroids_matrix)
        np.fill_diagonal(sim_matrix, 0.0)

        gi, gj = np.where(sim_matrix >= self.merge_threshold)
        keep = gi < gj
        gi, gj = gi[keep], gj[keep]

        g_merge = nx.Graph()
        g_merge.add_nodes_from(range(len(valid_clusters)))
        g_merge.add_edges_from(zip(gi.tolist(), gj.tolist()))

        components = list(nx.connected_components(g_merge))
        final_clusters = []
        for comp in components:
            merged_indices = []
            for idx in comp:
                merged_indices.extend(valid_clusters[idx])
            final_clusters.append(list(set(merged_indices)))

        return final_clusters

    def _select_representative_messages(self, community, texts, timestamps, embeddings):
        """
        Picks up to `self.max_representative_messages` messages from the
        cluster that are semantically closest to the cluster's centroid
        (mean embedding of every message in the cluster) - i.e. the
        messages that best represent the cluster's actual meaning, ranked
        by cosine similarity to that centroid, highest first.

        If the cluster already has `max_representative_messages` messages
        or fewer, all of them are kept as-is (nothing to trim). Returns the
        selected texts and their matching timestamps (same order, so the
        two lists stay aligned index-for-index).
        """
        community = list(community)

        if len(community) <= self.max_representative_messages:
            selected = community
        else:
            cluster_embs = embeddings[community]
            centroid = np.mean(cluster_embs, axis=0, keepdims=True)
            sims = cosine_similarity(cluster_embs, centroid).flatten()
            # indices into `community`, most similar to the centroid first
            order = np.argsort(-sims)[:self.max_representative_messages]
            selected = [community[i] for i in order]

        selected_texts = [texts[i] for i in selected]
        selected_timestamps = [timestamps[i] for i in selected] if timestamps else []
        return selected_texts, selected_timestamps

    def process_daily_results(self, all_daily_results: list) -> list:
        """
        Takes a list of daily clustering results and extracts candidate
        clusters. Returns a list of dicts, ready to be stored/serialized.
        Each item in `all_daily_results` is expected to have:
          date_gregorian, date_shamsi, valid_texts, valid_timestamps,
          embeddings (numpy array or None), communities (list of index
          collections, may include singletons - they're filtered here).
        """
        t_start = time.time()
        final_candidates = []
        global_cluster_id = 1  # unique within this call; cluster_id also embeds the date

        for daily_data in all_daily_results:
            date_gregorian = daily_data['date_gregorian']
            date_shamsi = daily_data['date_shamsi']
            texts = daily_data['valid_texts']
            timestamps = daily_data.get('valid_timestamps', [])
            embeddings = daily_data['embeddings']
            clusters = daily_data['communities']

            valid_clusters = [list(c) for c in clusters if len(c) > 1]
            if embeddings is None or len(valid_clusters) == 0:
                continue

            # ---- Step 1: merge near-duplicate clusters ----
            final_clusters = self._merge_similar_clusters(valid_clusters, embeddings)

            # ---- Step 2: outlier threshold for "dominant" clusters ----
            total_daily_msgs = len(texts)
            sizes = [len(c) for c in final_clusters]
            if len(sizes) > 1:
                q1 = np.percentile(sizes, 25)
                q3 = np.percentile(sizes, 75)
                iqr = q3 - q1
                upper_fence = q3 + 1.5 * iqr
            else:
                upper_fence = float('inf')

            # ---- Step 3: evaluate rules and extract final candidates ----
            for community in final_clusters:
                n = len(community)
                cluster_times = [timestamps[i] for i in community] if timestamps else []
                cluster_ratio_pct = (n / total_daily_msgs) * 100

                z_score, peak_count, _ = self.compute_hourly_spike(cluster_times)
                cohesion, median_sim, p25_sim = self.compute_cluster_cohesion(community, embeddings)
                coverage = self.compute_cluster_coverage(community, embeddings, threshold=0.75)

                rule1_dominant = n > upper_fence
                rule2_spike = z_score >= self.spike_z_threshold
                rule3_cohesive = (cluster_ratio_pct > 5.0) and (median_sim > 0.75) and (coverage > 0.55)

                if (rule1_dominant or rule2_spike or rule3_cohesive) and cluster_ratio_pct > 0.7:
                    passed_reasons = []
                    if rule1_dominant:
                        passed_reasons.append("Dominant")
                    if rule2_spike:
                        passed_reasons.append("Spike")
                    if rule3_cohesive:
                        passed_reasons.append("Cohesive_Large")

                    # Store only up to max_representative_messages messages -
                    # the ones semantically closest to the cluster's centroid
                    # - instead of dumping the whole cluster. `size` and the
                    # metrics above still reflect the full cluster; only what
                    # gets saved in "messages"/"timestamps" is trimmed.
                    representative_messages, representative_timestamps = \
                        self._select_representative_messages(community, texts, timestamps, embeddings)

                    candidate_record = {
                        "cluster_id": f"CL_{date_gregorian.strftime('%Y%m%d')}_{global_cluster_id}",
                        "date_shamsi": date_shamsi,
                        "date_gregorian": date_gregorian,
                        "size": n,
                        "daily_ratio_pct": float(f"{cluster_ratio_pct:.2f}"),
                        "reasons": passed_reasons,
                        "metrics": {
                            "z_score": float(f"{z_score:.2f}"),
                            "median_sim": float(f"{median_sim:.3f}"),
                            "coverage": float(f"{coverage:.3f}")
                        },
                        "messages": representative_messages,
                        "timestamps": representative_timestamps
                    }
                    final_candidates.append(candidate_record)
                    global_cluster_id += 1

        elapsed = time.time() - t_start
        print(f"[CandidateExtractor] process_daily_results done in {elapsed:.2f}s | "
              f"{len(final_candidates)} candidate(s) extracted")

        return final_candidates