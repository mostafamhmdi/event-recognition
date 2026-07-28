
import time
import numpy as np
import igraph as ig
import leidenalg
from sklearn.metrics.pairwise import cosine_similarity


class SocialMediaEpsilonClustering:
    """
    Groups embeddings into clusters ("communities") in two steps:
      1. Build an epsilon-similarity graph: an edge between i and j exists
         whenever cosine_similarity(i, j) >= threshold.
      2. Run the Leiden algorithm (igraph/leidenalg) on that graph.
    """

    def __init__(self, threshold: float = 0.55, max_block_bytes: int = 512 * 1024 * 1024):
        """
        max_block_bytes: memory budget (in bytes) for one similarity block.
        A full N x N similarity matrix needs N*N*4 bytes (float32), which
        for a busy day (tens of thousands of messages) can be several GB
        and cause an out-of-memory kill. Instead of ever materializing the
        whole N x N matrix, we compute it in row-blocks of size B, where
        B is chosen so that one B x N block stays under this budget. Each
        block is thresholded and discarded immediately. This produces the
        exact same edges/weights as computing the full matrix at once —
        it's the same formula, just computed in pieces — so clustering
        results are unaffected; only peak memory changes.
        """
        self.threshold = threshold
        self.max_block_bytes = max_block_bytes

    def _build_edges(self, embeddings: np.ndarray):
        embeddings = np.asarray(embeddings, dtype=np.float32)
        n = len(embeddings)

        bytes_per_row = max(n * 4, 1)  # one row of the block against all N, float32
        block_size = max(1, min(n, self.max_block_bytes // bytes_per_row))
        n_blocks = (n + block_size - 1) // block_size

        full_matrix_gb = (n * n * 4) / (1024 ** 3)
        block_gb = (block_size * n * 4) / (1024 ** 3)
        print(f"[Clustering] {n} embeddings | full matrix would be ~{full_matrix_gb:.2f} GB | "
              f"processing in {n_blocks} block(s) of {block_size} rows (~{block_gb:.2f} GB/block)")

        t0 = time.time()
        i_chunks, j_chunks, w_chunks = [], [], []
        total_edges = 0

        for block_idx, start in enumerate(range(0, n, block_size), start=1):
            end = min(start + block_size, n)
            block = embeddings[start:end]

            sim_block = cosine_similarity(block, embeddings)  # (b, n), not (n, n)

            local_i, j_idx = np.where(sim_block >= self.threshold)
            global_i = local_i + start

            # keep each unordered pair once and drop self-pairs (i == j)
            keep = global_i < j_idx
            gi = global_i[keep]
            gj = j_idx[keep]
            gw = sim_block[local_i[keep], j_idx[keep]]

            i_chunks.append(gi)
            j_chunks.append(gj)
            w_chunks.append(gw)
            total_edges += len(gi)

            if block_idx % max(1, n_blocks // 10) == 0 or block_idx == n_blocks:
                elapsed = time.time() - t0
                print(f"[Clustering] block {block_idx}/{n_blocks} | "
                      f"rows {start}-{end} | elapsed: {elapsed:.1f}s | "
                      f"edges so far: {total_edges}")

        i_idx = np.concatenate(i_chunks) if i_chunks else np.array([], dtype=int)
        j_idx = np.concatenate(j_chunks) if j_chunks else np.array([], dtype=int)
        weights = np.concatenate(w_chunks) if w_chunks else np.array([], dtype=np.float32)

        t1 = time.time()
        print(f"[Clustering] Similarity + thresholding done in {t1 - t0:.2f}s | "
              f"{len(i_idx)} edge(s) above threshold {self.threshold}")

        return n, i_idx, j_idx, weights

    def extract_clusters_leiden(self, n_nodes: int, i_idx, j_idx, weights):
        t0 = time.time()
        ig_g = ig.Graph(n=n_nodes)

        if len(i_idx) > 0:
            ig_g.add_edges(list(zip(i_idx.tolist(), j_idx.tolist())))
            ig_g.es['weight'] = weights.tolist()

        partition = leidenalg.find_partition(
            ig_g,
            leidenalg.ModularityVertexPartition,
            weights='weight' if len(i_idx) > 0 else None
        )
        communities = [set(comm) for comm in partition]
        t1 = time.time()

        multi_count = sum(1 for c in communities if len(c) > 1)
        print(f"[Clustering] Leiden partitioning done in {t1 - t0:.2f}s | "
              f"{len(communities)} community(ies) total, {multi_count} with >1 message")

        return communities

    def fit_predict(self, embeddings):
        """
        Takes an array of embeddings directly (no text / re-encoding needed)
        and returns a list of communities, each a set of row indices into
        `embeddings`.
        """
        t_start = time.time()
        print(f"[Clustering] fit_predict STARTED — {len(embeddings)} embeddings, "
              f"threshold={self.threshold}")

        n_nodes, i_idx, j_idx, weights = self._build_edges(embeddings)
        communities = self.extract_clusters_leiden(n_nodes, i_idx, j_idx, weights)

        print(f"[Clustering] fit_predict FINISHED in {time.time() - t_start:.2f}s")
        return communities