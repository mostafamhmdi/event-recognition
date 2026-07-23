# import pandas as pd
# import jdatetime
# import pickle
# import torch
# import gc
# import os
# import re
# import emoji
# import psutil  # <--- کتابخانه جدید برای مانیتورینگ سیستم (RAM و CPU)
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# # فرض بر این است که این ماژول‌ها در کنار فایل شما قرار دارند
# from data_loader import DataLoader 

# def print_resource_usage(stage_name=""):
#     """تابع کمکی اصلاح‌شده برای محاسبه و چاپ دقیق میزان مصرف منابع در اوبونتو"""
#     print(f"\n📊 --- میزان مصرف منابع سیستم ({stage_name}) ---")
    
#     # ۱. محاسبه میزان رم و پردازنده سیستم
#     process = psutil.Process(os.getpid())
#     # اصلاح فرمول: بایت به گیگابایت باید تقسیم بر 1024 به توان 3 شود
#     ram_usage = process.memory_info().rss / (1024 ** 3)  
    
#     # افزایش زمان اینتروال به 0.5 ثانیه برای محاسبه دقیق‌تر CPU در اوبونتو
#     cpu_usage = psutil.cpu_percent(interval=0.5)
    
#     print(f"🖥️ میزان مصرف پردازنده (CPU): {cpu_usage}%")
#     print(f"🧠 میزان رم مصرفی سیستم (RAM): {ram_usage:.2f} GB")
    
#     # ۲. محاسبه میزان حافظه کارت گرافیک (VRAM)
#     if torch.cuda.is_available():
#         # اصلاح فرمول‌های VRAM (تقسیم بر 1024 به توان 3)
#         vram_allocated = torch.cuda.memory_allocated() / (1024 ** 3)
#         vram_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
#         vram_max = torch.cuda.max_memory_allocated() / (1024 ** 3)
        
#         print(f"🎮 حافظه گرافیک فعال (VRAM Allocated): {vram_allocated:.2f} GB")
#         print(f"🎮 حافظه گرافیک رزرو شده (VRAM Reserved): {vram_reserved:.2f} GB")
#         print(f"🎮 حداکثر گرافیک استفاده شده (Max VRAM Allocated): {vram_max:.2f} GB")
#     else:
#         print("🎮 کارت گرافیک (GPU) شناسایی نشد؛ برنامه روی CPU در حال اجراست.")
#     print("=" * 50)


# # class TookaClassifier:
# #     def __init__(self, model_path: str):
# #         self.model_path = model_path
# #         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
# #         print("⏳ در حال بارگذاری مدل دسته‌بند و توکنایزر...")
# #         self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
# #         self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
# #         self.model.to(self.device)
# #         self.model.eval()
# #         print(f"✅ مدل دسته‌بند با موفقیت روی {self.device} بارگذاری شد!")

# #     @staticmethod
# #     def clean_text(text: str) -> str:
# #         if not isinstance(text, str):
# #             return ""
# #         text = re.sub(r'https?://\S+|www\.\S+', '', text)
# #         text = re.sub(r'@\S+', '', text)
# #         text = re.sub(r'#\S+', '', text)
        
# #         if emoji:
# #             text = emoji.replace_emoji(text, replace='')
# #         else:
# #             text = re.sub(r'[^\w\s\,\.\!\?؟\،\؛\:]', '', text)
            
# #         text = text.replace('\u200c', ' ')
# #         text = re.sub(r'\s+', ' ', text).strip()
# #         return text

# #     def predict_and_filter(self, raw_messages: list, raw_timestamps: list, batch_size: int = 32) -> tuple:
# #         """
# #         دریافت پیام‌ها و زمان‌ها، پیش‌بینی دسته‌ای و بازگرداندن نتایج ساختاریافته
# #         """
# #         processed_results = []
# #         filtered_out_count = 0

# #         # پاک‌سازی اولیه متون
# #         cleaned_messages = [self.clean_text(msg) for msg in raw_messages]

# #         for i in range(0, len(cleaned_messages), batch_size):
# #             batch_clean = cleaned_messages[i : i + batch_size]
# #             batch_raw = raw_messages[i : i + batch_size]
# #             batch_ts = raw_timestamps[i : i + batch_size]
            
# #             inputs = self.tokenizer(
# #                 batch_clean,
# #                 padding="max_length",
# #                 truncation=True,
# #                 max_length=256,
# #                 return_tensors="pt"
# #             ).to(self.device)

# #             with torch.no_grad():
# #                 outputs = self.model(**inputs)
# #                 predictions = torch.argmax(outputs.logits, dim=-1).cpu().numpy()

# #             for j, pred in enumerate(predictions):
# #                 # شرط فیلتر شما (کلاس 1 و طول بیشتر از 2 کاراکتر)
# #                 is_valid = (pred == 1 and len(batch_clean[j].strip()) > 2)
                
# #                 # ذخیره اطلاعات جامع هر سطر برای خروجی CSV
# #                 processed_results.append({
# #                     "timestamp": batch_ts[j],
# #                     "raw_text": batch_raw[j],
# #                     "clean_text": batch_clean[j],
# #                     "predicted_class": int(pred),
# #                     "passed_filter": is_valid
# #                 })
                
# #                 if not is_valid:
# #                     filtered_out_count += 1

# #         return processed_results, filtered_out_count


# class TookaClassifier:
#     def __init__(self, model_path: str):
#         self.model_path = model_path
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
#         print("⏳ در حال بارگذاری مدل دسته‌بند و توکنایزر...")
#         self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
#         # تغییر مهم: لود مدل با دقت 16 بیتی برای دو برابر شدن سرعت
#         dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
#         self.model = AutoModelForSequenceClassification.from_pretrained(
#             self.model_path, 
#             torch_dtype=dtype # لود مستقیم به صورت 16 بیتی
#         )
#         self.model.to(self.device)
#         self.model.eval()
#         print(f"✅ مدل دسته‌بند با موفقیت روی {self.device} بارگذاری شد!")

#     # ... تابع clean_text بدون تغییر ...
#     @staticmethod
#     def clean_text(text: str) -> str:
#         if not isinstance(text, str):
#             return ""
#         text = re.sub(r'https?://\S+|www\.\S+', '', text)
#         text = re.sub(r'@\S+', '', text)
#         text = re.sub(r'#\S+', '', text)
            
#         if emoji:
#             text = emoji.replace_emoji(text, replace='')
#         else:
#             text = re.sub(r'[^\w\s\,\.\!\?؟\،\؛\:]', '', text)
                
#         text = text.replace('\u200c', ' ')
#         text = re.sub(r'\s+', ' ', text).strip()
#         return text


#     def predict_and_filter(self, raw_messages: list, raw_timestamps: list, batch_size: int = 256) -> tuple:
#         # تغییر مهم: بچ سایز پیش‌فرض را به 256 افزایش دادیم
#         processed_results = []
#         filtered_out_count = 0

#         # پاک‌سازی اولیه متون
#         cleaned_messages = [self.clean_text(msg) for msg in raw_messages]

#         for i in range(0, len(cleaned_messages), batch_size):
#             batch_clean = cleaned_messages[i : i + batch_size]
#             batch_raw = raw_messages[i : i + batch_size]
#             batch_ts = raw_timestamps[i : i + batch_size]
            
#             # تغییر مهم: استفاده از padding داینامیک
#             inputs = self.tokenizer(
#                 batch_clean,
#                 padding=True,          # فقط تا اندازه بلندترین جمله همین بچ پد می‌کند
#                 truncation=True,
#                 max_length=256,        # جملات بلندتر از 256 توکن را برش می‌زند
#                 return_tensors="pt"
#             ).to(self.device)

#             # تغییر مهم: استفاده از inference_mode که کمی سریع‌تر از no_grad است
#             with torch.inference_mode():
#                 outputs = self.model(**inputs)
#                 predictions = torch.argmax(outputs.logits, dim=-1).cpu().numpy()

#             for j, pred in enumerate(predictions):
#                 is_valid = (pred == 1 and len(batch_clean[j].strip()) > 2)
                
#                 processed_results.append({
#                     "timestamp": batch_ts[j],
#                     "raw_text": batch_raw[j],
#                     "clean_text": batch_clean[j],
#                     "predicted_class": int(pred),
#                     "passed_filter": is_valid
#                 })
                
#                 if not is_valid:
#                     filtered_out_count += 1

#         return processed_results, filtered_out_count




import pandas as pd
import jdatetime
import pickle
import torch
import gc
import os
import re
import time
import emoji
import psutil  # <-- used for RAM/CPU monitoring
from transformers import AutoTokenizer, AutoModelForSequenceClassification
# Assumes these modules live next to this file
from data_loader import DataLoader


def print_resource_usage(stage_name=""):
    """Print current RAM / CPU / VRAM usage."""
    print(f"\n[Resources] --- Usage at stage: {stage_name} ---")

    # 1. RAM and CPU usage
    process = psutil.Process(os.getpid())
    ram_usage = process.memory_info().rss / (1024 ** 3)  # bytes -> GB

    # 0.5s interval gives a more accurate CPU reading
    cpu_usage = psutil.cpu_percent(interval=0.5)

    print(f"[Resources] CPU usage: {cpu_usage}%")
    print(f"[Resources] RAM usage (this process): {ram_usage:.2f} GB")

    # 2. GPU / VRAM usage
    if torch.cuda.is_available():
        vram_allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        vram_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        vram_max = torch.cuda.max_memory_allocated() / (1024 ** 3)

        print(f"[Resources] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[Resources] VRAM allocated: {vram_allocated:.2f} GB")
        print(f"[Resources] VRAM reserved:  {vram_reserved:.2f} GB")
        print(f"[Resources] Max VRAM allocated so far: {vram_max:.2f} GB")
    else:
        print("[Resources] No GPU detected - the program is running on CPU.")
    print("=" * 50)


class TookaClassifier:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("\n" + "=" * 60)
        print("[Classifier] STARTING: loading classifier model + tokenizer")
        print("=" * 60)
        print(f"[Classifier] Model path: {self.model_path}")
        print(f"[Classifier] torch.cuda.is_available(): {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"[Classifier] GPU device: {torch.cuda.get_device_name(0)}")
        print(f"[Classifier] Selected device: {self.device}")

        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        t1 = time.time()
        print(f"[Classifier] Tokenizer loaded in {t1 - t0:.2f}s")

        dtype, dtype_reason = self._select_dtype(self.device)
        print(f"[Classifier] dtype selected: {dtype} - {dtype_reason}")

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path,
            dtype=dtype
        )
        self.model.to(self.device)
        self.model.eval()
        t2 = time.time()

        print(f"[Classifier] Model moved to '{self.device}' in {t2 - t1:.2f}s")
        print(f"[Classifier] TOTAL load time: {t2 - t0:.2f}s")

        if self.device.type == "cuda":
            vram = torch.cuda.memory_allocated() / (1024 ** 3)
            print(f"[Classifier] VRAM currently used by the model: {vram:.2f} GB")
        else:
            print("[Classifier] WARNING: the model is running on CPU, which is "
                  "much slower than GPU. Check that torch.cuda.is_available() "
                  "is True and that you have a CUDA build of PyTorch installed "
                  "(pip show torch / nvidia-smi) if you expected GPU inference.")

        print("[Classifier] READY - model loaded, inference can begin now.")
        print("=" * 60 + "\n")

    @staticmethod
    def _select_dtype(device: torch.device):
        """
        Pick float16 only on GPUs that actually run it fast.

        Volta and newer (compute capability >= 7.0, e.g. V100/RTX20xx+) have
        full-rate fp16 tensor cores. Consumer Pascal cards (GTX 10-series,
        compute capability 6.1/6.2 - e.g. GTX 1080) deliberately run fp16 at
        about 1/64th the speed of fp32 (only the server part, P100, compute
        6.0, got fast fp16). Loading in fp16 on a GTX 1080 makes inference
        *slower*, not faster, even though it uses less VRAM. This is not a
        quality/quantization trade-off - the weights and math are identical,
        we're just picking the precision the hardware runs efficiently.
        """
        if device.type != "cuda":
            return torch.float32, "no GPU available, using float32 on CPU"

        major, minor = torch.cuda.get_device_capability(device)
        device_name = torch.cuda.get_device_name(device)

        if major >= 7:
            return torch.float16, (f"{device_name} (compute {major}.{minor}) has "
                                    f"fast fp16 tensor cores")
        if major == 6 and minor == 0:
            return torch.float16, (f"{device_name} (compute {major}.{minor}) is "
                                    f"P100-class and has fast fp16")

        return torch.float32, (f"{device_name} (compute {major}.{minor}) has "
                                f"crippled fp16 throughput on consumer Pascal/older "
                                f"GPUs - float32 will run faster here")

    @staticmethod
    def clean_text(text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'@\S+', '', text)
        text = re.sub(r'#\S+', '', text)

        text = emoji.replace_emoji(text, replace='')

        text = text.replace('\u200c', ' ')
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def predict_and_filter(self, raw_messages: list, raw_timestamps: list, batch_size: int = 256) -> tuple:
        processed_results = []
        filtered_out_count = 0
        total = len(raw_messages)

        print(f"[Classifier] predict_and_filter STARTED - {total} messages, "
              f"batch_size={batch_size}, device={self.device}")
        t_start = time.time()

        # Initial text cleaning (CPU-bound, done once up front)
        t_clean_start = time.time()
        cleaned_messages = [self.clean_text(msg) for msg in raw_messages]
        t_clean_end = time.time()
        print(f"[Classifier] Text cleaning finished in {t_clean_end - t_clean_start:.2f}s "
              f"for {total} messages")

        n_batches = (total + batch_size - 1) // batch_size if total else 0
        log_every = max(1, n_batches // 10) if n_batches else 1  # ~10 progress lines

        for batch_idx, i in enumerate(range(0, total, batch_size), start=1):
            batch_clean = cleaned_messages[i: i + batch_size]
            batch_raw = raw_messages[i: i + batch_size]
            batch_ts = raw_timestamps[i: i + batch_size]

            t_batch_start = time.time()

            # Dynamic padding: pads only to the longest sequence in this batch
            inputs = self.tokenizer(
                batch_clean,
                padding=True,
                truncation=True,
                max_length=256,  # sequences longer than 256 tokens are truncated
                return_tensors="pt"
            ).to(self.device)

            # inference_mode is slightly faster than no_grad
            with torch.inference_mode():
                outputs = self.model(**inputs)
                predictions = torch.argmax(outputs.logits, dim=-1).cpu().numpy()

            for j, pred in enumerate(predictions):
                is_valid = (pred == 1 and len(batch_clean[j].strip()) > 2)

                processed_results.append({
                    "timestamp": batch_ts[j],
                    "raw_text": batch_raw[j],
                    "clean_text": batch_clean[j],
                    "predicted_class": int(pred),
                    "passed_filter": is_valid
                })

                if not is_valid:
                    filtered_out_count += 1

            t_batch_end = time.time()

            if batch_idx % log_every == 0 or batch_idx == n_batches:
                elapsed = t_batch_end - t_start
                rows_done = min(i + batch_size, total)
                rate = rows_done / elapsed if elapsed > 0 else 0
                print(f"[Classifier] batch {batch_idx}/{n_batches} | "
                      f"{rows_done}/{total} rows | "
                      f"last batch: {t_batch_end - t_batch_start:.3f}s | "
                      f"elapsed: {elapsed:.1f}s | "
                      f"throughput: {rate:.1f} rows/s")

        t_end = time.time()
        total_elapsed = t_end - t_start
        throughput = total / total_elapsed if total_elapsed > 0 else 0
        print(f"[Classifier] predict_and_filter FINISHED - {total} rows in "
              f"{total_elapsed:.2f}s ({throughput:.1f} rows/s) | "
              f"filtered out: {filtered_out_count}")

        return processed_results, filtered_out_count