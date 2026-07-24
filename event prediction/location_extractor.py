# import pandas as pd
# from tqdm.auto import tqdm
# from transformers import pipeline

# class LocationExtractor:
#     def __init__(self, model_path: str, device: int = 0, batch_size: int = 64, chunk_size: int = 200):
#         """
#         راه‌اندازی مدل NER مکانی.
#         برای اجرای روی CPU مقدار device را 1- قرار دهید.
#         """
#         print(f"🔄 در حال بارگذاری مدل NER از مسیر: {model_path} (Device: {device})")
#         self.ner_pipeline = pipeline(
#             "ner",
#             model=model_path,
#             tokenizer=model_path,  # برای مدل‌های لوکال، مسیر توکنایزر هم همان مسیر مدل است
#             aggregation_strategy="simple",
#             device=device
#         )

#         # برخی چک‌پوینت‌های لوکال 'model_max_length' درستی در تنظیمات توکنایزر ندارند،
#         # در نتیجه HuggingFace هنگام truncate کردن هشدار "Default to no truncation" می‌دهد
#         # و اصلاً هیچ‌چیز را truncate نمی‌کند. اگر یک chunk بیشتر از ظرفیت position embedding
#         # مدل (معمولاً ۵۱۲) توکن subword تولید کند، این باعث کرش می‌شود
#         # (RuntimeError: tensor size mismatch در embeddings += position_embeddings).
#         # اینجا صریحاً model_max_length را از خود کانفیگ مدل می‌گیریم تا truncation واقعاً
#         # فعال شود و دیگر کرش نکنیم.
#         model_max_len = getattr(self.ner_pipeline.model.config, "max_position_embeddings", 512)
#         self.ner_pipeline.tokenizer.model_max_length = model_max_len
#         print(f"ℹ️ سقف طول توکن برای truncation ایمن روی {model_max_len} تنظیم شد.")

#         self.batch_size = batch_size
#         # chunk_size بر اساس تعداد کلمه است، نه توکن. چون فارسی معمولاً ۲ تا ۲.۵ برابر
#         # subword تولید می‌کند، مقدار پیش‌فرض را کاهش دادیم تا کمتر به سقف truncation بخوریم
#         # و اطلاعات کمتری از انتهای متن‌های بلند از دست برود (truncation همچنان به‌عنوان
#         # شبکه‌ی ایمنی نهایی فعال است، نه راه‌حل اصلی).
#         self.chunk_size = chunk_size

#     def extract_locations(self, df: pd.DataFrame, text_col: str = 'txtContent') -> pd.DataFrame:
#         texts_chunks = []
#         chunk_to_row_mapping = [] 

#         print("✂️ در حال قطعه‌بندی متن‌های طولانی...")
#         for row_idx, text in enumerate(df[text_col]):
#             if pd.notna(text) and str(text).strip():
#                 words = str(text).strip().split()

#                 if not words:
#                     texts_chunks.append(" ")
#                     chunk_to_row_mapping.append(row_idx)
#                     continue

#                 for i in range(0, len(words), self.chunk_size):
#                     chunk = " ".join(words[i : i + self.chunk_size])
#                     texts_chunks.append(chunk)
#                     chunk_to_row_mapping.append(row_idx)
#             else:
#                 texts_chunks.append(" ")
#                 chunk_to_row_mapping.append(row_idx)

#         print(f"📦 تعداد کل تکه‌های تولید شده: {len(texts_chunks)}")
#         print("🌍 شروع استخراج موجودیت‌های مکانی (GPU)...")

#         all_extracted_entities = []

#         for i in tqdm(range(0, len(texts_chunks), self.batch_size), desc="Processing Batches"):
#             batch_texts = texts_chunks[i : i + self.batch_size]
#             # بدون پاس دادن batch_size اینجا، pipeline متن‌ها را یکی‌یکی روی GPU پردازش
#             # می‌کند حتی اگر لیستی از آن‌ها را با هم بدهیم؛ این پارامتر باعث batching واقعی
#             # (و در نتیجه سرعت بیشتر روی GPU) می‌شود.
#             batch_outputs = self.ner_pipeline(batch_texts, batch_size=self.batch_size)

#             if not isinstance(batch_outputs, list) or (len(batch_outputs) > 0 and isinstance(batch_outputs[0], dict)):
#                 batch_outputs = [batch_outputs]

#             for out in batch_outputs:
#                 locations = [
#                     ent['word'] for ent in out
#                     if 'loc' in ent['entity_group'].lower()
#                 ]
#                 all_extracted_entities.append(locations)

#         # تجمیع نتایج تکه‌ها
#         row_locations_map = {i: set() for i in range(len(df))}

#         for chunk_idx, loc_list in enumerate(all_extracted_entities):
#             original_row_idx = chunk_to_row_mapping[chunk_idx]
#             if loc_list:
#                 row_locations_map[original_row_idx].update(loc_list)

#         final_extracted_locations = []
#         for i in range(len(df)):
#             locs = row_locations_map.get(i, set())
#             if locs:
#                 final_extracted_locations.append("، ".join(sorted(list(locs))))
#             else:
#                 final_extracted_locations.append(None)

#         # اضافه کردن ستون به دیتافریم
#         df['location_entities'] = final_extracted_locations
#         print("✅ پردازش مکان‌ها با موفقیت به پایان رسید!")
        
#         return df

# # تابع واسط برای استفاده راحت‌تر در main
# def process_location_data(df: pd.DataFrame, model_path: str, text_col: str = 'txtContent', device: int = 0) -> pd.DataFrame:
#     extractor = LocationExtractor(model_path=model_path, device=device)
#     return extractor.extract_locations(df, text_col=text_col)

import pandas as pd
from tqdm.auto import tqdm
from transformers import pipeline

class LocationExtractor:
    def __init__(self, model_path: str, device: int = 0, batch_size: int = 64, chunk_size: int = 200):
        """
        راه‌اندازی مدل NER مکانی.
        برای اجرای روی CPU مقدار device را 1- قرار دهید.
        """
        print(f"🔄 در حال بارگذاری مدل NER از مسیر: {model_path} (Device: {device})")
        self.ner_pipeline = pipeline(
            "ner",
            model=model_path,
            tokenizer=model_path,  # برای مدل‌های لوکال، مسیر توکنایزر هم همان مسیر مدل است
            aggregation_strategy="simple",
            device=device
        )

        # برخی چک‌پوینت‌های لوکال 'model_max_length' درستی در تنظیمات توکنایزر ندارند،
        # در نتیجه HuggingFace هنگام truncate کردن هشدار "Default to no truncation" می‌دهد
        # و اصلاً هیچ‌چیز را truncate نمی‌کند. اگر یک chunk بیشتر از ظرفیت position embedding
        # مدل (معمولاً ۵۱۲) توکن subword تولید کند، این باعث کرش می‌شود
        # (RuntimeError: tensor size mismatch در embeddings += position_embeddings).
        # اینجا صریحاً model_max_length را از خود کانفیگ مدل می‌گیریم تا truncation واقعاً
        # فعال شود و دیگر کرش نکنیم.
        model_max_len = getattr(self.ner_pipeline.model.config, "max_position_embeddings", 512)
        self.ner_pipeline.tokenizer.model_max_length = model_max_len
        print(f"ℹ️ سقف طول توکن برای truncation ایمن روی {model_max_len} تنظیم شد.")

        self.batch_size = batch_size
        # chunk_size بر اساس تعداد کلمه است، نه توکن. چون فارسی معمولاً ۲ تا ۲.۵ برابر
        # subword تولید می‌کند، مقدار پیش‌فرض را کاهش دادیم تا کمتر به سقف truncation بخوریم
        # و اطلاعات کمتری از انتهای متن‌های بلند از دست برود (truncation همچنان به‌عنوان
        # شبکه‌ی ایمنی نهایی فعال است، نه راه‌حل اصلی).
        self.chunk_size = chunk_size

    def extract_entities(self, df: pd.DataFrame, text_col: str = 'txtContent') -> pd.DataFrame:
        """
        استخراج هم‌زمان موجودیت‌های مکانی (location) و رویدادی (event) از متن.
        دو ستون 'location_entities' و 'event_entities' به دیتافریم اضافه می‌شود.
        """
        texts_chunks = []
        chunk_to_row_mapping = [] 

        print("✂️ در حال قطعه‌بندی متن‌های طولانی...")
        for row_idx, text in enumerate(df[text_col]):
            if pd.notna(text) and str(text).strip():
                words = str(text).strip().split()

                if not words:
                    texts_chunks.append(" ")
                    chunk_to_row_mapping.append(row_idx)
                    continue

                for i in range(0, len(words), self.chunk_size):
                    chunk = " ".join(words[i : i + self.chunk_size])
                    texts_chunks.append(chunk)
                    chunk_to_row_mapping.append(row_idx)
            else:
                texts_chunks.append(" ")
                chunk_to_row_mapping.append(row_idx)

        print(f"📦 تعداد کل تکه‌های تولید شده: {len(texts_chunks)}")
        print("🌍 شروع استخراج موجودیت‌های مکانی و رویدادی (GPU)...")

        all_extracted_locations = []
        all_extracted_events = []

        for i in tqdm(range(0, len(texts_chunks), self.batch_size), desc="Processing Batches"):
            batch_texts = texts_chunks[i : i + self.batch_size]
            # بدون پاس دادن batch_size اینجا، pipeline متن‌ها را یکی‌یکی روی GPU پردازش
            # می‌کند حتی اگر لیستی از آن‌ها را با هم بدهیم؛ این پارامتر باعث batching واقعی
            # (و در نتیجه سرعت بیشتر روی GPU) می‌شود.
            batch_outputs = self.ner_pipeline(batch_texts, batch_size=self.batch_size)

            if not isinstance(batch_outputs, list) or (len(batch_outputs) > 0 and isinstance(batch_outputs[0], dict)):
                batch_outputs = [batch_outputs]

            for out in batch_outputs:
                locations = [
                    ent['word'] for ent in out
                    if 'loc' in ent['entity_group'].lower()
                ]
                events = [
                    ent['word'] for ent in out
                    if 'eve' in ent['entity_group'].lower()
                ]
                all_extracted_locations.append(locations)
                all_extracted_events.append(events)

        # تجمیع نتایج تکه‌ها به ازای هر سطر، هم برای مکان و هم برای رویداد
        row_locations_map = {i: set() for i in range(len(df))}
        row_events_map = {i: set() for i in range(len(df))}

        for chunk_idx, (loc_list, ev_list) in enumerate(zip(all_extracted_locations, all_extracted_events)):
            original_row_idx = chunk_to_row_mapping[chunk_idx]
            if loc_list:
                row_locations_map[original_row_idx].update(loc_list)
            if ev_list:
                row_events_map[original_row_idx].update(ev_list)

        final_extracted_locations = []
        final_extracted_events = []
        for i in range(len(df)):
            locs = row_locations_map.get(i, set())
            final_extracted_locations.append("، ".join(sorted(list(locs))) if locs else None)

            evs = row_events_map.get(i, set())
            final_extracted_events.append("، ".join(sorted(list(evs))) if evs else None)

        # اضافه کردن ستون‌ها به دیتافریم
        df['location_entities'] = final_extracted_locations
        df['event_entities'] = final_extracted_events
        print("✅ پردازش مکان‌ها و رویدادها با موفقیت به پایان رسید!")

        return df

    def filter_relevant_rows(
        self,
        df: pd.DataFrame,
        loc_col: str = 'location_entities',
        event_col: str = 'event_entities',
    ) -> pd.DataFrame:
        """
        برای جلوگیری از حجیم شدن فایل خروجی نهایی، فقط سطرهایی نگه داشته می‌شوند
        که هم موجودیت مکانی و هم موجودیت رویدادی برایشان استخراج شده باشد.
        """
        mask = df[loc_col].notna() & df[event_col].notna()
        filtered_df = df.loc[mask].copy()
        print(
            f"🧹 فیلتر نهایی: از {len(df)} سطر، {len(filtered_df)} سطر دارای "
            f"هم مکان و هم رویداد بودند و نگه داشته شدند."
        )
        return filtered_df

# تابع واسط برای استفاده راحت‌تر در main
def process_location_data(
    df: pd.DataFrame,
    model_path: str,
    text_col: str = 'txtContent',
    device: int = 0,
    filter_output: bool = True,
) -> pd.DataFrame:
    """
    filter_output=True (پیش‌فرض): دیتافریم برگشتی فقط شامل سطرهایی است که
    هم location_entities و هم event_entities مقدار دارند (خروجی سبک‌تر برای ذخیره‌سازی نهایی).
    filter_output=False: دیتافریم کامل با هر دو ستون (بدون فیلتر) برگردانده می‌شود،
    مثلاً وقتی می‌خواهید خودتان فیلتر دیگری اعمال کنید یا نیاز به تمام سطرها دارید.
    """
    extractor = LocationExtractor(model_path=model_path, device=device)
    full_df = extractor.extract_entities(df, text_col=text_col)

    if filter_output:
        return extractor.filter_relevant_rows(full_df)
    return full_df


