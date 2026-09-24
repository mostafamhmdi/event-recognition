import time
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import Counter, defaultdict

class CandidateExtractor:
    """
    Extracts candidate clusters from a day's clustering results. A cluster
    only needs to satisfy ONE of four independent rules to be selected:
      - Dominant: unusually large compared to the other clusters that day
        (an IQR-based outlier on cluster size)
      - Spike: its messages are concentrated in a short burst of hours
      - Cohesive_Large: a large enough share of the day's messages AND
        tightly clustered around its own centroid
      - Emotional_Reaction: the cluster's messages show a cohesive, strong,
        non-neutral emotional reaction (requires per-message sentiment +
        emotion; only evaluated when >50% of the cluster's messages have
        both a valid sentiment value and a valid emotion label - see
        compute_cluster_emotion)
    """

    # نگاشت برچسب‌های خام emotion (فارسی/انگلیسی) به کد عددی استاندارد 0 تا 6.
    # تعریف کدها: 0=خنثی، 1=شادی، 2=غم، 3=عصبانیت/خشم، 4=ترس، 5=نفرت، 6=تعجب
    # کلیدها باید lowercase باشند چون _normalize_emotion قبل از جست‌وجو
    # .lower().strip() انجام می‌دهد. این لیست بر اساس برچسب‌های رایج نوشته شده؛
    # اگر دیتابیس برچسب/مترادف دیگری هم دارد باید همین‌جا اضافه شود.
    EMOTION_MAP = {
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

    def __init__(self, merge_threshold=0.75, spike_min_peak=3, spike_z_threshold=2.2,
                 max_representative_messages=35, emotion_dominant_share_threshold=0.5,
                 emotional_charge_threshold=0.5,sentiment_positive_threshold=0.5, sentiment_negative_threshold=-0.5,
                 importance_weight_dominant=0.30, importance_weight_spike=0.25,
                 importance_weight_cohesive=0.20, importance_weight_emotional=0.25,
                 importance_dominant_ratio_saturation=15.0, importance_spike_z_saturation=4.0,
                 importance_breadth_bonus_per_extra_reason=5.0):
        self.merge_threshold = merge_threshold
        self.spike_min_peak = spike_min_peak
        self.spike_z_threshold = spike_z_threshold
        # آستانه‌های رول چهارم (Emotional_Reaction) - نگاه کن به process_daily_results
        self.emotion_dominant_share_threshold = emotion_dominant_share_threshold
        self.emotional_charge_threshold = emotional_charge_threshold
        self.sentiment_positive_threshold = sentiment_positive_threshold
        self.sentiment_negative_threshold = sentiment_negative_threshold
        # وزن هر یک از چهار رول در امتیاز نهایی اهمیت خوشه (مجموع = 1.0) و
        # ثابت‌های اشباع (saturation) برای نگاه کن به _compute_importance_score
        self.importance_weight_dominant = importance_weight_dominant
        self.importance_weight_spike = importance_weight_spike
        self.importance_weight_cohesive = importance_weight_cohesive
        self.importance_weight_emotional = importance_weight_emotional
        self.importance_dominant_ratio_saturation = importance_dominant_ratio_saturation
        self.importance_spike_z_saturation = importance_spike_z_saturation
        self.importance_breadth_bonus_per_extra_reason = importance_breadth_bonus_per_extra_reason
        # Cap on how many messages get stored per candidate. Instead of the
        # full cluster, we keep the messages closest (by cosine similarity)
        # to the cluster's centroid - i.e. the ones that best represent
        # what the cluster is actually about.
        self.max_representative_messages = max_representative_messages
        self.stopwords = {
            'و','در','به','از','که','این','آن','برای','با','را','می','است','شد',
            'شود','کرد','کرده','ها','های','یک','تا','اما','اگر','یا','هم','بر',
            'نه','دیگر','روی','پس','قبل','بعد','هر','چه','چرا','چطور'
        }

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
    
    @classmethod
    def _normalize_emotion(cls, raw_label):
        """لیبل خام را گرفته و عدد 0 تا 6 برمی‌گرداند. در صورت خالی یا نامعتبر بودن، -1 برمی‌گرداند."""
        # بررسی مقادیر None یا NaN
        if raw_label is None or pd.isna(raw_label):
            return -1

        # اگر مقدار از قبل به‌صورت کد عددی 0 تا 6 ذخیره شده باشد (نه برچسب متنی)
        if isinstance(raw_label, (int, float, np.integer, np.floating)) and not isinstance(raw_label, bool):
            try:
                code = int(raw_label)
            except (ValueError, TypeError):
                return -1
            return code if 0 <= code <= 6 else -1

        if not isinstance(raw_label, str):
            return -1
            
        lbl = raw_label.lower().strip()
        # بررسی رشته‌های خالی یا کلمه 'nan'
        if not lbl or lbl == 'nan':
            return -1
            
        # اگر لیبل در دیکشنری ما نبود هم نامعتبر (-1) در نظر می‌گیریم
        return cls.EMOTION_MAP.get(lbl, -1)
    
    @staticmethod
    def _normalize_sentiment(raw_val):
        """مقدار سنتیمنت را گرفته و تبدیل به float می‌کند. اگر نامعتبر بود np.nan برمی‌گرداند."""
        if raw_val is None or pd.isna(raw_val):
            return np.nan
        try:
            val = float(raw_val)
            if np.isnan(val):
                return np.nan
            return val
        except (ValueError, TypeError):
            return np.nan
    @staticmethod
    def compute_cluster_emotion(sentiment_scores, emotion_labels):
        s = np.array(sentiment_scores, dtype=np.float32)
        e = np.array(emotion_labels, dtype=np.int32)
        n_total = len(e)
        
        invalid_result = {
            "is_valid": False, "sentiment_mean": 0.0, "sentiment_std": 0.0,
            "intensity_mean": 0.0, "dominant_emotion": 0, "dominant_emotion_share": 0.0,
            "high_arousal_share": 0.0, "emotional_cohesion": 0.0, "emotional_charge": 0.0
        }

        if n_total == 0:
            return invalid_result

        # --- ترکیب دو فیلتر: پیام‌هایی که یا احساس ندارند یا سنتیمنت ندارند ---
        # np.isnan(s) برای پیدا کردن NaN در سنتیمنت 
        # (e == -1) برای پیدا کردن احساسات نامعتبر
        missing_mask = np.isnan(s) | (e == -1)
        missing_ratio = np.sum(missing_mask) / n_total

        # بررسی شرط ۵۰ درصد
        if missing_ratio >= 0.5:
            return invalid_result

        # حذف پیام‌های ناقص و نگه داشتن پیام‌های کامل
        valid_s = s[~missing_mask]
        valid_e = e[~missing_mask]
        n_valid = len(valid_e)

        if n_valid == 0:
            return invalid_result

        # حالا با خیال راحت محاسبات رو روی آرایه‌های تمیز (valid_s و valid_e) انجام می‌دیم
        sentiment_mean = float(np.mean(valid_s))
        sentiment_std  = float(np.std(valid_s))
        intensity_mean = float(np.mean(np.abs(valid_s)))

        emotion_dist = np.bincount(valid_e, minlength=7) / n_valid
        dominant_emotion = int(np.argmax(emotion_dist))
        dominant_emotion_share = float(emotion_dist[dominant_emotion])
        high_arousal_share = float(emotion_dist[[3, 4, 5, 6]].sum()) 

        p = emotion_dist[emotion_dist > 0]
        entropy = -(p * np.log(p)).sum() / np.log(7)
        emotional_cohesion = 1.0 - entropy

        emotional_charge = 0.5 * intensity_mean + 0.5 * high_arousal_share

        return {
            "is_valid": True,
            "sentiment_mean": sentiment_mean,
            "sentiment_std": sentiment_std,
            "intensity_mean": intensity_mean,
            "dominant_emotion": dominant_emotion,
            "dominant_emotion_share": dominant_emotion_share,
            "high_arousal_share": high_arousal_share,
            "emotional_cohesion": emotional_cohesion,
            "emotional_charge": emotional_charge
        }
    def _normalize_text(self, text):
        text = str(text).replace('ي', 'ی').replace('ك', 'ک')
        text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
        text = re.sub(r'[@#]\S+', ' ', text)
        text = re.sub(r'[^\w\sآ-ی]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    def _extract_candidate_terms(self, texts, min_freq=2):
        counter = Counter()

        for text in texts:
            text = self._normalize_text(text)
            tokens = [
                t for t in text.split()
                if len(t) > 2 and t not in self.stopwords
            ]

            # unigram
            counter.update(tokens)

            # bigram
            bigrams = [' '.join(tokens[i:i+2]) for i in range(len(tokens)-1)]
            counter.update(bigrams)
        return {term: freq for term, freq in counter.items() if freq >= min_freq}

    def extract_cluster_keywords(self, community, texts, embeddings,
                             top_k=10, min_freq=2):

        community = list(community)
        cluster_texts = [texts[i] for i in community]
        cluster_embs = embeddings[community]

        # centroid خوشه
        centroid = np.mean(cluster_embs, axis=0, keepdims=True)

        # کاندیدها
        term_freqs = self._extract_candidate_terms(cluster_texts, min_freq=min_freq)

        if not term_freqs:
            return []

        # term -> embedding میانگین پیام‌هایی که term را دارند
        term_vectors = defaultdict(list)

        normalized_texts = [self._normalize_text(t) for t in cluster_texts]

        for local_idx, text in enumerate(normalized_texts):
            for term in term_freqs.keys():
                if term in text:
                    term_vectors[term].append(cluster_embs[local_idx])

        scored = []

        for term, freq in term_freqs.items():
            vecs = term_vectors.get(term, [])
            if not vecs:
                continue

            term_emb = np.mean(np.vstack(vecs), axis=0, keepdims=True)
            sim = cosine_similarity(term_emb, centroid)[0, 0]

            # ترکیب شباهت معنایی و فراوانی
            score = 0.7 * sim + 0.3 * np.log1p(freq)

            scored.append((term, score, freq, sim))

        scored.sort(key=lambda x: x[1], reverse=True)

        return [t[0] for t in scored[:top_k]]
        
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

    def _compute_importance_score(self, rule1_dominant, rule2_spike, rule3_cohesive, rule4_emotional,
                                   cluster_ratio_pct, z_score, median_sim, coverage, emotion_result):
        """
        امتیاز اهمیت خوشه (0 تا 100). فقط از مقادیری استفاده می‌کند که در
        process_daily_results از قبل محاسبه شده‌اند - هیچ محاسبه‌ی سنگین
        جدیدی اضافه نمی‌کند.

        منطق: هر یک از چهار رول (Dominant/Spike/Cohesive_Large/Emotional_Reaction)
        فقط وقتی در امتیاز سهیم می‌شود که واقعاً برقرار باشد (rule=True)؛ سهم آن
        رول هم صرفاً «صفر یا یک» نیست بلکه یک قدرت پیوسته‌ی 0 تا 1 دارد که نشان
        می‌دهد آن رول چقدر قوی برقرار شده (مثلا یک خوشه‌ی Dominant که 40 درصد
        پیام‌های روز را دارد باید امتیاز بیشتری از خوشه‌ای بگیرد که فقط کمی از
        آستانه رد شده). این قدرت‌ها با وزن هر رول (self.importance_weight_*)
        جمع می‌شوند تا امتیاز پایه به دست بیاید، سپس یک پاداش کوچک برای
        «تعداد رول‌های برقرارشده» اضافه می‌شود، چون خوشه‌ای که هم‌زمان از چند
        منظر مستقل برجسته باشد معمولا مهم‌تر از خوشه‌ای است که فقط یک رول را
        رد کرده.
        """
        # قدرت رول اول (Dominant): سهم خوشه از پیام‌های روز، با اشباع در
        # importance_dominant_ratio_saturation درصد (پیش‌فرض 15%)
        strength_dominant = 0.0
        if rule1_dominant:
            strength_dominant = min(cluster_ratio_pct / self.importance_dominant_ratio_saturation, 1.0)

        # قدرت رول دوم (Spike): z-score تمرکز ساعتی، با اشباع در
        # importance_spike_z_saturation (پیش‌فرض 4.0)
        strength_spike = 0.0
        if rule2_spike:
            strength_spike = min(z_score / self.importance_spike_z_saturation, 1.0)

        # قدرت رول سوم (Cohesive_Large): میانگین شباهت میانه و پوشش خوشه
        # (هر دو از قبل بین 0 تا 1 هستند)
        strength_cohesive = 0.0
        if rule3_cohesive:
            strength_cohesive = max(0.0, min((median_sim + coverage) / 2.0, 1.0))

        # قدرت رول چهارم (Emotional_Reaction): میانگین سهم احساس غالب و
        # شدت/برانگیختگی احساسی خوشه (هر دو از قبل بین 0 تا 1 هستند)
        strength_emotional = 0.0
        if rule4_emotional and emotion_result is not None:
            strength_emotional = max(0.0, min(
                (emotion_result["dominant_emotion_share"] + emotion_result["emotional_charge"]) / 2.0, 1.0
            ))

        weighted_sum = (
            self.importance_weight_dominant * strength_dominant
            + self.importance_weight_spike * strength_spike
            + self.importance_weight_cohesive * strength_cohesive
            + self.importance_weight_emotional * strength_emotional
        )

        num_reasons_fired = sum([rule1_dominant, rule2_spike, rule3_cohesive, rule4_emotional])
        breadth_bonus = max(0, num_reasons_fired - 1) * self.importance_breadth_bonus_per_extra_reason

        importance_score = (weighted_sum * 100.0) + breadth_bonus
        return float(round(min(importance_score, 100.0), 2))

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

            # ---- sentiment/emotion خام هر پیام (اختیاری) ----
            # اگر daily_data این دو کلید را نداشته باشد یا طولشان با texts
            # نخواند (مثلاً هنوز در main.py/DataLoader وایر نشده)، رول چهارم
            # برای این روز به‌سادگی غیرفعال می‌ماند و سه رول قبلی دست‌نخورده
            # کار می‌کنند - هیچ خطایی رخ نمی‌دهد.
            raw_sentiments = daily_data.get('valid_sentiments', [])
            raw_emotions = daily_data.get('valid_emotions', [])
            has_emotion_data = (
                len(raw_sentiments) == len(texts)
                and len(raw_emotions) == len(texts)
                and len(texts) > 0
            )
            if has_emotion_data:
                normalized_sentiments = [self._normalize_sentiment(v) for v in raw_sentiments]
                normalized_emotions = [self._normalize_emotion(v) for v in raw_emotions]

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

                # compute_cluster_emotion خودش شرط «بالای 50٪ پیام‌های خوشه باید
                # هم sentiment و هم emotion معتبر داشته باشند» را چک می‌کند؛ اگر
                # این شرط برقرار نباشد is_valid=False برمی‌گرداند و رول چهارم
                # پایین به‌سادگی رد می‌شود (نه فعال، نه خطا).
                if has_emotion_data:
                    cluster_sentiments = [normalized_sentiments[i] for i in community]
                    cluster_emotions = [normalized_emotions[i] for i in community]
                    emotion_result = self.compute_cluster_emotion(cluster_sentiments, cluster_emotions)
                else:
                    emotion_result = None

                # Cluster-center vector for this candidate. Computed here rather
                # than passed through unchanged from phase 2, because `community`
                # is a *post-merge* set of indices - `_merge_similar_clusters`
                # can combine several of phase 2's original clusters into one,
                # so the centroid has to reflect the final merged membership.
                # `embeddings` for the whole day is already loaded in memory at
                # this point, so this is just a cheap mean over vectors we
                # already have, not a new heavy computation.
                centroid_embedding = np.mean(embeddings[community], axis=0).astype(np.float32)

                rule1_dominant = n > upper_fence
                rule2_spike = z_score >= self.spike_z_threshold
                rule3_cohesive = (cluster_ratio_pct > 5.0) and (median_sim > 0.75) and (coverage > 0.55)

                # ---- رول چهارم: Emotional_Reaction ----
                # فقط وقتی «معتبر» است که (طبق compute_cluster_emotion) بالای ۵۰٪
                # پیام‌های خوشه هم sentiment معتبر و هم emotion معتبر داشته باشند؛
                # در غیر این صورت رول چهارم رد می‌شود و اصلا وارد محاسبه نمی‌شود.
                # شرط: اکثریت پیام‌های خوشه (dominant_emotion_share) حول یک احساس
                # غیرخنثی مشخص جمع شده باشند (اجماع احساسی)، و آن واکنش از نظر
                # شدت/برانگیختگی هم قوی باشد (emotional_charge = ترکیب میانگین
                # قدرمطلق سنتیمنت و سهم احساسات پرتحرک مثل خشم/ترس/نفرت/تعجب).
                rule4_emotional = bool(
                    emotion_result is not None
                    and emotion_result["is_valid"]
                    and emotion_result["dominant_emotion"] != 0  # خنثی را حساب نکن
                    and emotion_result["dominant_emotion_share"] >= self.emotion_dominant_share_threshold
                    and emotion_result["emotional_charge"] >= self.emotional_charge_threshold
                )
                sentiment_label = None
                dominant_emotion_out = None
                if emotion_result is not None and emotion_result["is_valid"]:
                    sm = emotion_result["sentiment_mean"]
                    if sm >= self.sentiment_positive_threshold:
                        sentiment_label = "positive"
                    elif sm <= self.sentiment_negative_threshold:
                        sentiment_label = "negative"
                    else:
                        sentiment_label = "neutral"
                    dominant_emotion_out = emotion_result["dominant_emotion"]

                importance_score = self._compute_importance_score(
                    rule1_dominant, rule2_spike, rule3_cohesive, rule4_emotional,
                    cluster_ratio_pct, z_score, median_sim, coverage, emotion_result,
                )

                if (rule1_dominant or rule2_spike or rule3_cohesive or rule4_emotional) and cluster_ratio_pct > 0.7:
                    passed_reasons = []
                    if rule1_dominant:
                        passed_reasons.append("Dominant")
                    if rule2_spike:
                        passed_reasons.append("Spike")
                    if rule3_cohesive:
                        passed_reasons.append("Cohesive_Large")
                    if rule4_emotional:
                        passed_reasons.append("Emotional_Reaction")

                    # Store only up to max_representative_messages messages -
                    # the ones semantically closest to the cluster's centroid
                    # - instead of dumping the whole cluster. `size` and the
                    # metrics above still reflect the full cluster; only what
                    # gets saved in "messages"/"timestamps" is trimmed.
                    representative_messages, representative_timestamps = \
                        self._select_representative_messages(community, texts, timestamps, embeddings)
                    keywords = self.extract_cluster_keywords(
                        community=community,
                        texts=texts,
                        embeddings=embeddings,
                        top_k=10,
                        min_freq=2
                    )
                    candidate_record = {
                        "cluster_id": f"CL_{date_gregorian.strftime('%Y%m%d')}_{global_cluster_id}",
                        "date_shamsi": date_shamsi,
                        "date_gregorian": date_gregorian,
                        "size": n,
                        "centroid_embedding": centroid_embedding.tolist(),
                        "keywords":keywords,
                        "daily_ratio_pct": float(f"{cluster_ratio_pct:.2f}"),
                        "reasons": passed_reasons,
                        "metrics": {
                            "z_score": float(f"{z_score:.2f}"),
                            "median_sim": float(f"{median_sim:.3f}"),
                            "coverage": float(f"{coverage:.3f}")
                        },
                        # None وقتی کمتر از ۵۰٪ پیام‌های خوشه sentiment/emotion
                        # معتبر داشته باشند (یا اصلا داده‌ای در دسترس نباشد) -
                        # یعنی همان «کلا نان» که خواسته شده بود.
                        "sentiment_label": sentiment_label,
                        "dominant_emotion": dominant_emotion_out,
                        "importance_score": importance_score,
                        "messages": representative_messages,
                        "timestamps": representative_timestamps
                    }
                    final_candidates.append(candidate_record)
                    global_cluster_id += 1

        elapsed = time.time() - t_start
        print(f"[CandidateExtractor] process_daily_results done in {elapsed:.2f}s | "
              f"{len(final_candidates)} candidate(s) extracted")

        return final_candidates