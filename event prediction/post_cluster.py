# import numpy as np
# import pandas as pd
# import networkx as nx
# from sklearn.metrics.pairwise import cosine_similarity
# # ایمپورت مدل امبدینگ دقیقاً طبق ساختار شما
# from utils import similarity_model 
# from preprocessing import dynamic_preprocess, PreprocessingOptions  # not wired in yet — kept for a future phase


# class ClusterPostProcessor:
#     def __init__(self, 
#                  text_col: str = '0', 
#                  min_sim_threshold: float = 0.50,  # حداقل شباهت به مدوید (کمتر از این = نویز)
#                  max_sim_threshold: float = 0.98,  # حداکثر شباهت به مدوید (بیشتر از این = کپی/تکراری)
#                  merge_threshold: float = 0.75,    # حداقل شباهت برای ادغام دو خوشه با یکدیگر
#                  gpu_batch_size: int = 128):
        
#         self.text_col = text_col
#         self.min_sim = min_sim_threshold
#         self.max_sim = max_sim_threshold
#         self.merge_threshold = merge_threshold
#         self.gpu_batch_size = gpu_batch_size
#         self.embedder = similarity_model

#     def _generate_embeddings(self, texts: list):
#         """
#         تولید بردار با استفاده از کد سفارشی شما
#         """
#         print(f"🧠 در حال تولید امبدینگ برای {len(texts)} پیام...")
#         all_embeddings = []
        
#         # استخراج ابعاد از روی یک کلمه تستی (طبق منطق خودتان)
#         warmup_emb = self.embedder.encode_texts(["warmup"])
#         if hasattr(warmup_emb, 'cpu'): warmup_emb = warmup_emb.cpu().numpy()
#         emb_dim = warmup_emb.shape[1] if len(warmup_emb.shape) > 1 else warmup_emb.shape[0]
#         zero_vector = [0.0] * emb_dim

#         for i in range(0, len(texts), self.gpu_batch_size):
#             sub_batch = texts[i : i + self.gpu_batch_size]
#             try:
#                 embs = self.embedder.encode_texts(sub_batch)
#                 if hasattr(embs, 'cpu'): embs = embs.cpu().numpy()
#                 all_embeddings.extend(embs.tolist())
#             except Exception as e:
#                 print(f"❌ خطای GPU در بچ‌سایز: {e}")
#                 for _ in range(len(sub_batch)):
#                     all_embeddings.append(zero_vector)
                    
#         return np.array(all_embeddings)

#     def process(self, df: pd.DataFrame) -> pd.DataFrame:
#         working_df = df.copy()
        
#         # فقط روی داده‌هایی کار می‌کنیم که در مرحله قبل خوشه گرفته‌اند
#         clustered_mask = working_df['cluster_id'].notna()
#         target_indices = working_df[clustered_mask].index.tolist()
        
#         if not target_indices:
#             print("⚠️ خوشه‌ای برای پس‌پردازش یافت نشد.")
#             return working_df

#         # ۱. تولید بردارها فقط برای پیام‌های خوشه‌بندی شده
#         texts_to_embed = working_df.loc[target_indices, self.text_col].astype(str).tolist()
#         embeddings_matrix = self._generate_embeddings(texts_to_embed)
        
#         # ذخیره بردارها در یک دیکشنری برای دسترسی سریع (Index -> Vector)
#         emb_dict = {idx: embeddings_matrix[i] for i, idx in enumerate(target_indices)}

#         print("🧹 در حال پاکسازی درون‌خوشه‌ای (یافتن مدوید و حذف نویز)...")
#         cluster_groups = working_df[clustered_mask].groupby('cluster_id')
        
#         valid_clusters = {} # ساختار ذخیره‌سازی: {cluster_id: {'medoid_idx': int, 'members': list}}
#         rows_to_drop = []   # ردیف‌های اسپم و تکراری که باید کلاً حذف شوند
        
#         # ۲. پردازش هر خوشه (Intra-cluster)
#         for cluster_id, group in cluster_groups:
#             indices = group.index.tolist()
            
#             if len(indices) == 1:
#                 valid_clusters[cluster_id] = {'medoid_idx': indices[0], 'members': indices}
#                 continue
                
#             # استخراج بردارهای این خوشه
#             group_embs = np.array([emb_dict[idx] for idx in indices])
            
#             # محاسبه ماتریس شباهت بین تمام اعضای خوشه
#             sim_matrix = cosine_similarity(group_embs)
            
#             # یافتن مدوید (ایندکسی که بیشترین مجموع شباهت را با بقیه دارد)
#             medoid_relative_idx = np.argmax(sim_matrix.sum(axis=1))
#             medoid_idx = indices[medoid_relative_idx]
#             medoid_similarities = sim_matrix[medoid_relative_idx]
            
#             clean_members = []
#             for i, idx in enumerate(indices):
#                 sim = medoid_similarities[i]
                
#                 if idx == medoid_idx:
#                     clean_members.append(idx)
#                 elif sim < self.min_sim:
#                     # نویز: از خوشه خارج می‌شود اما در دیتافریم می‌ماند
#                     working_df.at[idx, 'cluster_id'] = None
#                 elif sim > self.max_sim:
#                     # کپی دقیق: برای جلوگیری از دیتای تکراری، علامت‌گذاری برای حذف
#                     rows_to_drop.append(idx)
#                 else:
#                     # عضو معتبر
#                     clean_members.append(idx)
                    
#             if clean_members:
#                 valid_clusters[cluster_id] = {'medoid_idx': medoid_idx, 'members': clean_members}

#         # حذف پیام‌های کپی و کاملاً تکراری از کل دیتافریم
#         if rows_to_drop:
#             working_df = working_df.drop(index=rows_to_drop)

#         print("🔗 در حال ادغام خوشه‌های مشابه (Inter-cluster)...")
#         # ۳. ادغام خوشه‌ها بر اساس تاریخ مشترک
#         # استخراج خوشه‌های معتبر و دسته‌بندی آن‌ها بر اساس تاریخ آینده
#         date_to_clusters = {}
#         for c_id, c_data in valid_clusters.items():
#             f_date = working_df.at[c_data['medoid_idx'], 'future_date']
#             if f_date not in date_to_clusters:
#                 date_to_clusters[f_date] = []
#             date_to_clusters[f_date].append(c_id)

#         final_cluster_mapping = {} # مپ کردن آیدی‌های قدیم به آیدی جدید ادغام شده
        
#         for f_date, c_ids in date_to_clusters.items():
#             if len(c_ids) < 2:
#                 for cid in c_ids:
#                     final_cluster_mapping[cid] = cid
#                 continue
                
#             # گراف برای اتصال خوشه‌های مشابه
#             G = nx.Graph()
#             G.add_nodes_from(c_ids)
            
#             medoid_embs = [emb_dict[valid_clusters[cid]['medoid_idx']] for cid in c_ids]
#             sim_matrix = cosine_similarity(medoid_embs)
            
#             for i in range(len(c_ids)):
#                 for j in range(i + 1, len(c_ids)):
#                     if sim_matrix[i, j] >= self.merge_threshold:
#                         G.add_edge(c_ids[i], c_ids[j])
                        
#             merged_components = list(nx.connected_components(G))
            
#             for comp in merged_components:
#                 comp = list(comp)
#                 # اولین آیدی به عنوان نماینده و اسم خوشه جدید انتخاب می‌شود
#                 primary_id = comp[0]
#                 for cid in comp:
#                     final_cluster_mapping[cid] = primary_id

#         # ۴. بروزرسانی نام خوشه‌ها در دیتافریم نهایی
#         for old_cid, new_cid in final_cluster_mapping.items():
#             if old_cid != new_cid:
#                 # جایگزینی نام خوشه ادغام شده
#                 mask = working_df['cluster_id'] == old_cid
#                 working_df.loc[mask, 'cluster_id'] = new_cid

#         print("✅ پس‌پردازش معنایی با موفقیت به پایان رسید.")
#         return working_df

# # تابع واسط برای استفاده در main
# def post_process_clusters(df: pd.DataFrame, text_col: str = '0') -> pd.DataFrame:
#     processor = ClusterPostProcessor(text_col=text_col)
#     return processor.process(df)

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
# ایمپورت مدل امبدینگ دقیقاً طبق ساختار شما
from utils import similarity_model
from preprocessing import dynamic_preprocess, PreprocessingOptions


class ClusterPostProcessor:
    def __init__(self, 
                 text_col: str = 'txtContent', 
                 min_sim_threshold: float = 0.50,  # حداقل شباهت به مدوید (کمتر از این = نویز)
                 max_sim_threshold: float = 0.98,  # حداکثر شباهت به مدوید (بیشتر از این = کپی/تکراری)
                 merge_threshold: float = 0.75,    # حداقل شباهت برای ادغام دو خوشه با یکدیگر
                 gpu_batch_size: int = 128):
        
        self.text_col = text_col
        self.min_sim = min_sim_threshold
        self.max_sim = max_sim_threshold
        self.merge_threshold = merge_threshold
        self.gpu_batch_size = gpu_batch_size
        self.embedder = similarity_model

    def _generate_embeddings(self, texts: list):
        """
        تولید بردار با استفاده از کد سفارشی شما
        """
        print(f"🧠 در حال تولید امبدینگ برای {len(texts)} پیام...")
        all_embeddings = []
        
        # استخراج ابعاد از روی یک کلمه تستی (طبق منطق خودتان)
        warmup_emb = self.embedder.encode_texts(["warmup"])
        if hasattr(warmup_emb, 'cpu'): warmup_emb = warmup_emb.cpu().numpy()
        emb_dim = warmup_emb.shape[1] if len(warmup_emb.shape) > 1 else warmup_emb.shape[0]
        zero_vector = [0.0] * emb_dim

        for i in range(0, len(texts), self.gpu_batch_size):
            sub_batch = texts[i : i + self.gpu_batch_size]
            try:
                embs = self.embedder.encode_texts(sub_batch)
                if hasattr(embs, 'cpu'): embs = embs.cpu().numpy()
                all_embeddings.extend(embs.tolist())
            except Exception as e:
                print(f"❌ خطای GPU در بچ‌سایز: {e}")
                for _ in range(len(sub_batch)):
                    all_embeddings.append(zero_vector)
                    
        return np.array(all_embeddings)

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        working_df = df.copy()
        
        # فقط روی داده‌هایی کار می‌کنیم که در مرحله قبل خوشه گرفته‌اند
        clustered_mask = working_df['cluster_id'].notna()
        target_indices = working_df[clustered_mask].index.tolist()
        
        if not target_indices:
            print("⚠️ خوشه‌ای برای پس‌پردازش یافت نشد.")
            return working_df

        # ۱. تولید بردارها فقط برای پیام‌های خوشه‌بندی شده
        texts_to_embed = working_df.loc[target_indices, self.text_col].astype(str).tolist()
        embeddings_matrix = self._generate_embeddings(texts_to_embed)
        
        # ذخیره بردارها در یک دیکشنری برای دسترسی سریع (Index -> Vector)
        emb_dict = {idx: embeddings_matrix[i] for i, idx in enumerate(target_indices)}

        print("🧹 در حال پاکسازی درون‌خوشه‌ای (یافتن مدوید و حذف نویز)...")
        cluster_groups = working_df[clustered_mask].groupby('cluster_id')
        
        valid_clusters = {} # ساختار ذخیره‌سازی: {cluster_id: {'medoid_idx': int, 'members': list}}
        rows_to_drop = []   # ردیف‌های اسپم و تکراری که باید کلاً حذف شوند
        
        # ۲. پردازش هر خوشه (Intra-cluster)
        for cluster_id, group in cluster_groups:
            indices = group.index.tolist()
            
            if len(indices) == 1:
                valid_clusters[cluster_id] = {'medoid_idx': indices[0], 'members': indices}
                continue
                
            # استخراج بردارهای این خوشه
            group_embs = np.array([emb_dict[idx] for idx in indices])
            
            # محاسبه ماتریس شباهت بین تمام اعضای خوشه
            sim_matrix = cosine_similarity(group_embs)
            
            # یافتن مدوید (ایندکسی که بیشترین مجموع شباهت را با بقیه دارد)
            medoid_relative_idx = np.argmax(sim_matrix.sum(axis=1))
            medoid_idx = indices[medoid_relative_idx]
            medoid_similarities = sim_matrix[medoid_relative_idx]
            
            clean_members = []
            for i, idx in enumerate(indices):
                sim = medoid_similarities[i]
                
                if idx == medoid_idx:
                    clean_members.append(idx)
                elif sim < self.min_sim:
                    # نویز: از خوشه خارج می‌شود اما در دیتافریم می‌ماند
                    working_df.at[idx, 'cluster_id'] = None
                elif sim > self.max_sim:
                    # کپی دقیق: برای جلوگیری از دیتای تکراری، علامت‌گذاری برای حذف
                    rows_to_drop.append(idx)
                else:
                    # عضو معتبر
                    clean_members.append(idx)
                    
            if clean_members:
                valid_clusters[cluster_id] = {'medoid_idx': medoid_idx, 'members': clean_members}

        # حذف پیام‌های کپی و کاملاً تکراری از کل دیتافریم
        if rows_to_drop:
            working_df = working_df.drop(index=rows_to_drop)

        print("🔗 در حال ادغام خوشه‌های مشابه (Inter-cluster)...")
        # ۳. ادغام خوشه‌ها بر اساس تاریخ مشترک
        # استخراج خوشه‌های معتبر و دسته‌بندی آن‌ها بر اساس تاریخ آینده
        date_to_clusters = {}
        for c_id, c_data in valid_clusters.items():
            f_date = working_df.at[c_data['medoid_idx'], 'future_date']
            if f_date not in date_to_clusters:
                date_to_clusters[f_date] = []
            date_to_clusters[f_date].append(c_id)

        final_cluster_mapping = {} # مپ کردن آیدی‌های قدیم به آیدی جدید ادغام شده
        
        for f_date, c_ids in date_to_clusters.items():
            if len(c_ids) < 2:
                for cid in c_ids:
                    final_cluster_mapping[cid] = cid
                continue
                
            # گراف برای اتصال خوشه‌های مشابه
            G = nx.Graph()
            G.add_nodes_from(c_ids)
            
            medoid_embs = [emb_dict[valid_clusters[cid]['medoid_idx']] for cid in c_ids]
            sim_matrix = cosine_similarity(medoid_embs)
            
            for i in range(len(c_ids)):
                for j in range(i + 1, len(c_ids)):
                    if sim_matrix[i, j] >= self.merge_threshold:
                        G.add_edge(c_ids[i], c_ids[j])
                        
            merged_components = list(nx.connected_components(G))
            
            for comp in merged_components:
                comp = list(comp)
                # اولین آیدی به عنوان نماینده و اسم خوشه جدید انتخاب می‌شود
                primary_id = comp[0]
                for cid in comp:
                    final_cluster_mapping[cid] = primary_id

        # ۴. بروزرسانی نام خوشه‌ها در دیتافریم نهایی
        for old_cid, new_cid in final_cluster_mapping.items():
            if old_cid != new_cid:
                # جایگزینی نام خوشه ادغام شده
                mask = working_df['cluster_id'] == old_cid
                working_df.loc[mask, 'cluster_id'] = new_cid

        print("✅ پس‌پردازش معنایی با موفقیت به پایان رسید.")
        return working_df

# تابع واسط برای استفاده در main
def post_process_clusters(df: pd.DataFrame, text_col: str = 'txtContent') -> pd.DataFrame:
    processor = ClusterPostProcessor(text_col=text_col)
    return processor.process(df)