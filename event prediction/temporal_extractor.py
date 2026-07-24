# import re
# import pandas as pd
# import jdatetime
# from parstdex import Parstdex

# class TwitterTemporalNormalizer:
#     def __init__(self):
#         self.dex = Parstdex()
        
#         self.months_map = {
#             "فروردین": 1, "اردیبهشت": 2, "خرداد": 3, "تیر": 4,
#             "مرداد": 5, "شهریور": 6, "مهر": 7, "آبان": 8,
#             "آذر": 9, "دی": 10, "بهمن": 11, "اسفند": 12
#         }
        
#         self.weekdays_map = {
#             'شنبه': 0, 'یکشنبه': 1, 'دوشنبه': 2, 'سه شنبه': 3, 'سه‌شنبه': 3, 
#             'چهارشنبه': 4, 'پنجشنبه': 5, 'پنج‌شنبه': 5, 'جمعه': 6
#         }
        
#         self.relative_map = dict(sorted({
#             "پسون‌فردا": 3, "پسون فردا": 3, "پریروز": -2, "پریشب": -2,
#             "پس فردا": 2, "پس‌فردا": 2, "دیروز": -1, "دیشب": -1, 
#             "امروز": 0, "امشب": 0, "الان": 0, "حالا": 0, "هم اکنون": 0,
#             "فردا شب": 1, "فردا": 1, "آخر هفته": "weekend"
#         }.items(), key=lambda item: len(item[0]), reverse=True))
        
#         self.number_words = {
#             "یک": 1, "دو": 2, "سه": 3, "چهار": 4, "پنج": 5,
#             "شش": 6, "هفت": 7, "هشت": 8, "نه": 9, "ده": 10,
#             "یازده": 11, "دوازده": 12, "بیست": 20, "سی": 30
#         }
        
#         self.word_to_num = {
#             'بیست و نهم': '29', 'بیست و هشتم': '28', 'بیست و هفتم': '27', 
#             'بیست و ششم': '26', 'بیست و پنجم': '25', 'بیست و چهارم': '24', 
#             'بیست و سوم': '23', 'بیست و دوم': '22', 'بیست و یکم': '21', 
#             'سی و یکم': '31', 'سی و یک': '31', 'نوزدهم': '19', 'هجدهم': '18', 
#             'هفدهم': '17', 'شانزدهم': '16', 'پانزدهم': '15', 'چهاردهم': '14', 
#             'سیزدهم': '13', 'دوازدهم': '12', 'یازدهم': '11', 'بیست': '20', 
#             'سی': '30', 'دهم': '10', 'نهم': '9', 'هشتم': '8', 'هفتم': '7', 
#             'ششم': '6', 'پنجم': '5', 'چهارم': '4', 'سوم': '3', 'دوم': '2', 
#             'یکم': '1', 'یک': '1', 'دو': '2', 'سه': '3', 'چهار': '4', 'پنج': '5'
#         }

#         self.direction_words = {
#             "پیش": -1, "قبل": -1, "گذشته": -1,
#             "بعد": 1, "دیگه": 1, "دیگر": 1, "آینده": 1
#         }
        
#         self.priority_map = {
#             'exact_full': 1,       
#             'exact_short': 2,      
#             'exact_approx': 3,     
#             'relative_complex': 4, 
#             'relative_simple': 5,  
#             'relative_word': 6,    
#             'bare_weekday': 7      
#         }

#         # ==========================================
#         # بخش بهینه‌سازی: پیش‌کامپایل کردن تمام Regex ها
#         # ==========================================
#         month_names_str = "|".join(self.months_map.keys())
#         self.rx_months = re.compile(fr'({month_names_str})\s*ماه')
#         self.rx_entities = re.compile(r'(خیابان|میدان|کوچه|پلاک|کتاب|فیلم|سریال|شماره|کد)\s+[\w\d]+(?:\s+[\w\d]+)?')
#         self.rx_http = re.compile(r'http\S+')
#         self.rx_mention = re.compile(r'@\w+')
#         self.rx_hashtag = re.compile(r'#\w+')
#         self.rx_punc = re.compile(r'[،.؟!؛,;!؟]')
        
#         # ترکیب ۳۷ کلید به یک Regex برای تابع words_to_digits
#         # مرتب‌سازی بر اساس طول (کلمات طولانی‌تر اولویت دارند تا در تطابق اشتباه نشود)
#         sorted_word_nums = sorted(self.word_to_num.keys(), key=len, reverse=True)
#         self.rx_word_to_num = re.compile(fr'(?<![آ-ی])({"|".join(sorted_word_nums)})(?![آ-ی])')

#         # کامپایل Regexهای مربوط به استخراج زمان
#         self.rx_full_date = re.compile(r'(?:^|\s)(1[34]\d{2}|\d{2})[-/\.](0?[1-9]|1[0-2])[-/\.](0?[1-9]|[12]\d|3[01])(?:\s|$)')
#         self.rx_short_date = re.compile(r'(?:^|\s)(0?[1-9]|[12]\d|3[01])[-/\.](0?[1-9]|1[0-2])(?:\s|$)')
#         self.rx_curr_month = re.compile(r'(?:^|\s)(\d{1,2})\s+(?:همین|این)\s*ماه(?:\s|$)')
#         self.rx_day_month = re.compile(fr'(?:^|\s)(\d{{1,2}})\s+({month_names_str})(?:\s|$)')
#         self.rx_approx_month = re.compile(fr'(اواخر|اواسط|اوایل)\s+({month_names_str})')
#         self.rx_range = re.compile(r'(?:بین|از)\s+.*?\s+(?:تا|الی)\s+(.*)')
        
#         dir_pattern = r'(آینده|گذشته|دیگه|دیگر|بعد|پیش|قبل)'
#         wk_pattern = r'(چهارشنبه|پنجشنبه|پنج‌شنبه|سه‌شنبه|سه شنبه|دوشنبه|یکشنبه|جمعه|شنبه)'
#         self.rx_weekday = re.compile(fr'{wk_pattern}(?:\s+(هفته\s+)?{dir_pattern})?')
        
#         unit_pattern = r'(روز|هفته|ماه|سال)(?:ها|های|ان)?'
#         number_pattern = r'(\d+|' + r'|'.join(self.number_words.keys()) + r')'
#         self.rx_rel_complex = re.compile(fr'{number_pattern}(?:\s+و\s+نیم)?\s+({unit_pattern})(?:\s+و\s+نیم)?\s+{dir_pattern}')
#         self.rx_rel_simple = re.compile(fr'(?:^|\s)({unit_pattern})\s+{dir_pattern}(?:\s|$)')
        
#         self.trans_table = str.maketrans('۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩', '01234567890123456789')

#     def _convert_digits_to_en(self, text):
#         return text.translate(self.trans_table)

#     def _clean_tweet(self, text):
#         text = self._convert_digits_to_en(text)
#         text = text.replace('سی‌دی', 'لوح فشرده').replace('دی‌وی‌دی', 'لوح فشرده').replace('\u200c', ' ')
        
#         text = self.rx_months.sub(r'\1', text)
        
#         text = text.replace('۲شنبه', 'دوشنبه').replace('چارشنبه', 'چهارشنبه').replace('پیش رو', 'آینده')
        
#         text = self.rx_entities.sub(' ', text)
#         text = text.replace('پنج‌شنبه بازار', ' ').replace('پنجشنبه بازار', ' ').replace('هفت روز هفته', ' ')
        
#         text = self.rx_http.sub('', text)
#         text = self.rx_mention.sub('', text)
#         text = self.rx_hashtag.sub('', text)
#         return text.strip()

#     def _words_to_digits(self, text):
#         # جایگزینی ۳۷ حلقه for با یک بار پیمایش Regex بسیار سریع
#         return self.rx_word_to_num.sub(lambda m: self.word_to_num[m.group(1)], text)

#     def _add_months(self, date, months_to_add):
#         m = date.month + months_to_add
#         y = date.year + (m - 1) // 12
#         if y < 1 or y > 9999:
#             raise ValueError("Year out of range")
#         m = (m - 1) % 12 + 1
#         if m <= 6: max_days = 31
#         elif m <= 11: max_days = 30
#         else: max_days = 30 if jdatetime.date(y, 1, 1).isleap() else 29
#         return jdatetime.date(y, m, min(date.day, max_days))

#     def _parse_extracted_time(self, temporal_text, anchor_date):
#         temporal_text = self._words_to_digits(temporal_text)
#         text = f" {self.rx_punc.sub(' ', temporal_text)} "
        
#         full_date_match = self.rx_full_date.search(text)
#         if full_date_match:
#             y, m, d = int(full_date_match.group(1)), int(full_date_match.group(2)), int(full_date_match.group(3))
#             if y < 100: y += 1400 if y < 50 else 1300
#             return jdatetime.date(y, m, d), 'exact_full'
            
#         short_date_match = self.rx_short_date.search(text)
#         if short_date_match:
#             d, m = int(short_date_match.group(1)), int(short_date_match.group(2))
#             return jdatetime.date(anchor_date.year, m, d), 'exact_short'

#         current_month_match = self.rx_curr_month.search(text)
#         if current_month_match:
#             d = int(current_month_match.group(1))
#             return jdatetime.date(anchor_date.year, anchor_date.month, d), 'exact_short'

#         day_month_match = self.rx_day_month.search(text)
#         if day_month_match:
#             d, m = int(day_month_match.group(1)), self.months_map[day_month_match.group(2)]
#             return jdatetime.date(anchor_date.year, m, d), 'exact_full'

#         approx_match = self.rx_approx_month.search(text)
#         if approx_match:
#             part, m = approx_match.group(1), self.months_map[approx_match.group(2)]
#             d = 5 if part == 'اوایل' else (15 if part == 'اواسط' else 25)
#             return jdatetime.date(anchor_date.year, m, d), 'exact_approx'

#         text = self.rx_range.sub(r'\1', text)
        
#         wk_match = self.rx_weekday.search(text)
#         if wk_match:
#             day_name = wk_match.group(1).replace(" ", " ")
#             has_week = bool(wk_match.group(2))
#             modifier = wk_match.group(3)
            
#             target_wd = self.weekdays_map[day_name]
#             current_wd = anchor_date.weekday()
#             days_ahead = target_wd - current_wd
            
#             if modifier:
#                 direction = self.direction_words.get(modifier, 1)
#                 if direction == 1: 
#                     if days_ahead <= 0: days_ahead += 7
#                     if has_week: days_ahead += 7
#                 else: 
#                     if days_ahead >= 0: days_ahead -= 7
#                     if has_week: days_ahead -= 7
#                 return anchor_date + jdatetime.timedelta(days=days_ahead), 'relative_complex'
#             else:
#                 return anchor_date + jdatetime.timedelta(days=days_ahead), 'bare_weekday'

#         rel_complex_match = self.rx_rel_complex.search(text)
#         if rel_complex_match:
#             num_str, unit_full, unit_base, dir_str = rel_complex_match.groups()
#             num = float(num_str) if num_str.isdigit() else float(self.number_words.get(num_str, 1))
            
#             if 'و نیم' in text: num += 0.5
#             direction = self.direction_words.get(dir_str, 1)
            
#             if unit_base == 'روز': return anchor_date + jdatetime.timedelta(days=int(num * direction)), 'relative_complex'
#             elif unit_base == 'هفته': return anchor_date + jdatetime.timedelta(days=int(num * direction * 7)), 'relative_complex'
#             elif unit_base == 'ماه': 
#                 res_date = self._add_months(anchor_date, int(num) * direction)
#                 if num % 1 != 0: res_date += jdatetime.timedelta(days=15 * direction)
#                 return res_date, 'relative_complex'
#             elif unit_base == 'سال': 
#                 target_year = anchor_date.year + (int(num) * direction)
#                 if target_year < 1 or target_year > 9999: return None
#                 target_month, target_day = anchor_date.month, anchor_date.day
#                 if target_month == 12 and target_day == 30 and not jdatetime.date(target_year, 1, 1).isleap():
#                     target_day = 29
#                 res_date = jdatetime.date(target_year, target_month, target_day)
#                 if num % 1 != 0: res_date = self._add_months(res_date, 6 * direction)
#                 return res_date, 'relative_complex'

#         rel_simple_match = self.rx_rel_simple.search(text)
#         if rel_simple_match:
#             unit_full, unit_base, dir_str = rel_simple_match.groups()
#             direction = self.direction_words.get(dir_str, 1)
#             num = 2 if ('ها' in unit_full or 'ان' in unit_full) else 1
            
#             if unit_base == 'روز': return anchor_date + jdatetime.timedelta(days=num * direction), 'relative_simple'
#             elif unit_base == 'هفته': return anchor_date + jdatetime.timedelta(days=7 * num * direction), 'relative_simple'
#             elif unit_base == 'ماه': return self._add_months(anchor_date, num * direction), 'relative_simple'
#             elif unit_base == 'سال': 
#                 target_year = anchor_date.year + (num * direction)
#                 if target_year < 1 or target_year > 9999: return None
#                 target_month, target_day = anchor_date.month, anchor_date.day
#                 if target_month == 12 and target_day == 30 and not jdatetime.date(target_year, 1, 1).isleap():
#                     target_day = 29
#                 return jdatetime.date(target_year, target_month, target_day), 'relative_simple'

#         # استفاده از regex از پیش‌کامپایل‌شده امکان‌پذیر نیست چون کلیدها متفاوتند، 
#         # اما با توجه به حجم کم دیکشنری، جستجوی متنی ساده پایتون سریعتر از Regex است
#         for word, offset in self.relative_map.items():
#             if f" {word} " in text or text.startswith(f"{word} ") or text.endswith(f" {word}") or text == word:
#                 if offset == "weekend":
#                     days_ahead = self.weekdays_map['جمعه'] - anchor_date.weekday()
#                     if days_ahead <= 0: days_ahead += 7
#                     return anchor_date + jdatetime.timedelta(days=days_ahead), 'relative_word'
#                 return anchor_date + jdatetime.timedelta(days=offset), 'relative_word'
            
#         return None

#     def extract_and_normalize(self, text, anchor_date):
#         clean_text = self._clean_tweet(text)
#         spans = self.dex.extract_span(clean_text)
        
#         extracted_dates = []
        
#         if spans and spans.get('datetime'):
#             dts = spans['datetime']
            
#             if isinstance(dts, dict):
#                 parsed_dts = []
#                 for key in dts.keys():
#                     match = re.match(r'\[(\d+),\s*(\d+)\]', key)
#                     if match:
#                         parsed_dts.append([int(match.group(1)), int(match.group(2))])
#                 dts = sorted(parsed_dts)
                
#             if isinstance(dts, list) and len(dts) > 0:
#                 merged_spans = []
#                 curr_start, curr_end = dts[0]
#                 for start, end in dts[1:]:
#                     if start - curr_end <= 5: 
#                         curr_end = max(curr_end, end)
#                     else:
#                         merged_spans.append((curr_start, curr_end))
#                         curr_start, curr_end = start, end
#                 merged_spans.append((curr_start, curr_end))
                
#                 for start, end in merged_spans:
#                     temporal_phrase = clean_text[start:end]
#                     try:
#                         parsed_result = self._parse_extracted_time(temporal_phrase, anchor_date)
#                         if parsed_result:
#                             extracted_dates.append(parsed_result)
#                     except ValueError:
#                         pass
        
#         try:
#             whole_text_result = self._parse_extracted_time(clean_text, anchor_date)
#             if whole_text_result:
#                 extracted_dates.append(whole_text_result)
#         except ValueError:
#             pass

#         if not extracted_dates:
#             return None

#         def sort_key(item):
#             date, date_type = item
#             priority_score = self.priority_map.get(date_type, 99)
#             is_past = 1 if date < anchor_date else 0
#             return (priority_score, is_past, -date.toordinal())
            
#         extracted_dates.sort(key=sort_key)
#         return extracted_dates[0][0] 


# # --- توابع واسط برای اعمال روی دیتافریم ---

# def _gregorian_to_jalali(gregorian_str):
#     try:
#         dt = pd.to_datetime(gregorian_str)
#         jdt = jdatetime.datetime.fromgregorian(datetime=dt)
#         return jdt.strftime("%Y-%m-%d %H:%M:%S")
#     except:
#         return None 

# def process_temporal_data(df: pd.DataFrame, text_col: str = '0', date_col: str = '2') -> pd.DataFrame:
#     print("⏳ در حال تبدیل تاریخ‌های میلادی به شمسی...")
#     df['jalali_date'] = df[date_col].apply(_gregorian_to_jalali)
    
#     date_str = df['jalali_date'].dropna().str.split().str[0]
#     df['j_date'] = date_str.apply(lambda x: jdatetime.datetime.strptime(x, "%Y-%m-%d").date() if pd.notna(x) else None)

#     normalizer = TwitterTemporalNormalizer()

#     def extract_future_from_row(row):
#         text = row[text_col]
#         anchor_date = row['j_date']
        
#         if pd.isna(text) or pd.isna(anchor_date):
#             return None
            
#         norm_date = normalizer.extract_and_normalize(text, anchor_date)
        
#         if norm_date is not None and norm_date > anchor_date:
#             return norm_date.strftime('%Y/%m/%d')
#         else:
#             return None

#     print("⏳ در حال استخراج و اعتبارسنجی تاریخ‌های آینده از متن...")
#     df['future_date'] = df.apply(extract_future_from_row, axis=1)
    
#     return df


import re
import pandas as pd
import jdatetime
from parstdex import Parstdex
from multiprocessing import Pool, cpu_count


class TwitterTemporalNormalizer:
    # ==========================================================
    # این کلاس دقیقاً بدون تغییر است — هیچ منطقی دست نخورده است
    # ==========================================================
    def __init__(self):
        self.dex = Parstdex()
        
        self.months_map = {
            "فروردین": 1, "اردیبهشت": 2, "خرداد": 3, "تیر": 4,
            "مرداد": 5, "شهریور": 6, "مهر": 7, "آبان": 8,
            "آذر": 9, "دی": 10, "بهمن": 11, "اسفند": 12
        }
        
        self.weekdays_map = {
            'شنبه': 0, 'یکشنبه': 1, 'دوشنبه': 2, 'سه شنبه': 3, 'سه‌شنبه': 3, 
            'چهارشنبه': 4, 'پنجشنبه': 5, 'پنج‌شنبه': 5, 'جمعه': 6
        }
        
        self.relative_map = dict(sorted({
            "پسون‌فردا": 3, "پسون فردا": 3, "پریروز": -2, "پریشب": -2,
            "پس فردا": 2, "پس‌فردا": 2, "دیروز": -1, "دیشب": -1, 
            "امروز": 0, "امشب": 0, "الان": 0, "حالا": 0, "هم اکنون": 0,
            "فردا شب": 1, "فردا": 1, "آخر هفته": "weekend"
        }.items(), key=lambda item: len(item[0]), reverse=True))
        
        self.number_words = {
            "یک": 1, "دو": 2, "سه": 3, "چهار": 4, "پنج": 5,
            "شش": 6, "هفت": 7, "هشت": 8, "نه": 9, "ده": 10,
            "یازده": 11, "دوازده": 12, "بیست": 20, "سی": 30
        }
        
        self.word_to_num = {
            'بیست و نهم': '29', 'بیست و هشتم': '28', 'بیست و هفتم': '27', 
            'بیست و ششم': '26', 'بیست و پنجم': '25', 'بیست و چهارم': '24', 
            'بیست و سوم': '23', 'بیست و دوم': '22', 'بیست و یکم': '21', 
            'سی و یکم': '31', 'سی و یک': '31', 'نوزدهم': '19', 'هجدهم': '18', 
            'هفدهم': '17', 'شانزدهم': '16', 'پانزدهم': '15', 'چهاردهم': '14', 
            'سیزدهم': '13', 'دوازدهم': '12', 'یازدهم': '11', 'بیست': '20', 
            'سی': '30', 'دهم': '10', 'نهم': '9', 'هشتم': '8', 'هفتم': '7', 
            'ششم': '6', 'پنجم': '5', 'چهارم': '4', 'سوم': '3', 'دوم': '2', 
            'یکم': '1', 'یک': '1', 'دو': '2', 'سه': '3', 'چهار': '4', 'پنج': '5'
        }

        self.direction_words = {
            "پیش": -1, "قبل": -1, "گذشته": -1,
            "بعد": 1, "دیگه": 1, "دیگر": 1, "آینده": 1
        }
        
        self.priority_map = {
            'exact_full': 1,       
            'exact_short': 2,      
            'exact_approx': 3,     
            'relative_complex': 4, 
            'relative_simple': 5,  
            'relative_word': 6,    
            'bare_weekday': 7      
        }

        # ==========================================
        # بخش بهینه‌سازی: پیش‌کامپایل کردن تمام Regex ها
        # ==========================================
        month_names_str = "|".join(self.months_map.keys())
        self.rx_months = re.compile(fr'({month_names_str})\s*ماه')
        self.rx_entities = re.compile(r'(خیابان|میدان|کوچه|پلاک|کتاب|فیلم|سریال|شماره|کد)\s+[\w\d]+(?:\s+[\w\d]+)?')
        self.rx_http = re.compile(r'http\S+')
        self.rx_mention = re.compile(r'@\w+')
        self.rx_hashtag = re.compile(r'#\w+')
        self.rx_punc = re.compile(r'[،.؟!؛,;!؟]')
        
        # ترکیب ۳۷ کلید به یک Regex برای تابع words_to_digits
        # مرتب‌سازی بر اساس طول (کلمات طولانی‌تر اولویت دارند تا در تطابق اشتباه نشود)
        sorted_word_nums = sorted(self.word_to_num.keys(), key=len, reverse=True)
        self.rx_word_to_num = re.compile(fr'(?<![آ-ی])({"|".join(sorted_word_nums)})(?![آ-ی])')

        # کامپایل Regexهای مربوط به استخراج زمان
        self.rx_full_date = re.compile(r'(?:^|\s)(1[34]\d{2}|\d{2})[-/\.](0?[1-9]|1[0-2])[-/\.](0?[1-9]|[12]\d|3[01])(?:\s|$)')
        self.rx_short_date = re.compile(r'(?:^|\s)(0?[1-9]|[12]\d|3[01])[-/\.](0?[1-9]|1[0-2])(?:\s|$)')
        self.rx_curr_month = re.compile(r'(?:^|\s)(\d{1,2})\s+(?:همین|این)\s*ماه(?:\s|$)')
        self.rx_day_month = re.compile(fr'(?:^|\s)(\d{{1,2}})\s+({month_names_str})(?:\s|$)')
        self.rx_approx_month = re.compile(fr'(اواخر|اواسط|اوایل)\s+({month_names_str})')
        self.rx_range = re.compile(r'(?:بین|از)\s+.*?\s+(?:تا|الی)\s+(.*)')
        
        dir_pattern = r'(آینده|گذشته|دیگه|دیگر|بعد|پیش|قبل)'
        wk_pattern = r'(چهارشنبه|پنجشنبه|پنج‌شنبه|سه‌شنبه|سه شنبه|دوشنبه|یکشنبه|جمعه|شنبه)'
        self.rx_weekday = re.compile(fr'{wk_pattern}(?:\s+(هفته\s+)?{dir_pattern})?')
        
        unit_pattern = r'(روز|هفته|ماه|سال)(?:ها|های|ان)?'
        number_pattern = r'(\d+|' + r'|'.join(self.number_words.keys()) + r')'
        self.rx_rel_complex = re.compile(fr'{number_pattern}(?:\s+و\s+نیم)?\s+({unit_pattern})(?:\s+و\s+نیم)?\s+{dir_pattern}')
        self.rx_rel_simple = re.compile(fr'(?:^|\s)({unit_pattern})\s+{dir_pattern}(?:\s|$)')
        
        self.trans_table = str.maketrans('۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩', '01234567890123456789')

    def _convert_digits_to_en(self, text):
        return text.translate(self.trans_table)

    def _clean_tweet(self, text):
        text = self._convert_digits_to_en(text)
        text = text.replace('سی‌دی', 'لوح فشرده').replace('دی‌وی‌دی', 'لوح فشرده').replace('\u200c', ' ')
        
        text = self.rx_months.sub(r'\1', text)
        
        text = text.replace('۲شنبه', 'دوشنبه').replace('چارشنبه', 'چهارشنبه').replace('پیش رو', 'آینده')
        
        text = self.rx_entities.sub(' ', text)
        text = text.replace('پنج‌شنبه بازار', ' ').replace('پنجشنبه بازار', ' ').replace('هفت روز هفته', ' ')
        
        text = self.rx_http.sub('', text)
        text = self.rx_mention.sub('', text)
        text = self.rx_hashtag.sub('', text)
        return text.strip()

    def _words_to_digits(self, text):
        # جایگزینی ۳۷ حلقه for با یک بار پیمایش Regex بسیار سریع
        return self.rx_word_to_num.sub(lambda m: self.word_to_num[m.group(1)], text)

    def _add_months(self, date, months_to_add):
        m = date.month + months_to_add
        y = date.year + (m - 1) // 12
        if y < 1 or y > 9999:
            raise ValueError("Year out of range")
        m = (m - 1) % 12 + 1
        if m <= 6: max_days = 31
        elif m <= 11: max_days = 30
        else: max_days = 30 if jdatetime.date(y, 1, 1).isleap() else 29
        return jdatetime.date(y, m, min(date.day, max_days))

    def _parse_extracted_time(self, temporal_text, anchor_date):
        temporal_text = self._words_to_digits(temporal_text)
        text = f" {self.rx_punc.sub(' ', temporal_text)} "
        
        full_date_match = self.rx_full_date.search(text)
        if full_date_match:
            y, m, d = int(full_date_match.group(1)), int(full_date_match.group(2)), int(full_date_match.group(3))
            if y < 100: y += 1400 if y < 50 else 1300
            return jdatetime.date(y, m, d), 'exact_full'
            
        short_date_match = self.rx_short_date.search(text)
        if short_date_match:
            d, m = int(short_date_match.group(1)), int(short_date_match.group(2))
            return jdatetime.date(anchor_date.year, m, d), 'exact_short'

        current_month_match = self.rx_curr_month.search(text)
        if current_month_match:
            d = int(current_month_match.group(1))
            return jdatetime.date(anchor_date.year, anchor_date.month, d), 'exact_short'

        day_month_match = self.rx_day_month.search(text)
        if day_month_match:
            d, m = int(day_month_match.group(1)), self.months_map[day_month_match.group(2)]
            return jdatetime.date(anchor_date.year, m, d), 'exact_full'

        approx_match = self.rx_approx_month.search(text)
        if approx_match:
            part, m = approx_match.group(1), self.months_map[approx_match.group(2)]
            d = 5 if part == 'اوایل' else (15 if part == 'اواسط' else 25)
            return jdatetime.date(anchor_date.year, m, d), 'exact_approx'

        text = self.rx_range.sub(r'\1', text)
        
        wk_match = self.rx_weekday.search(text)
        if wk_match:
            day_name = wk_match.group(1).replace(" ", " ")
            has_week = bool(wk_match.group(2))
            modifier = wk_match.group(3)
            
            target_wd = self.weekdays_map[day_name]
            current_wd = anchor_date.weekday()
            days_ahead = target_wd - current_wd
            
            if modifier:
                direction = self.direction_words.get(modifier, 1)
                if direction == 1: 
                    if days_ahead <= 0: days_ahead += 7
                    if has_week: days_ahead += 7
                else: 
                    if days_ahead >= 0: days_ahead -= 7
                    if has_week: days_ahead -= 7
                return anchor_date + jdatetime.timedelta(days=days_ahead), 'relative_complex'
            else:
                return anchor_date + jdatetime.timedelta(days=days_ahead), 'bare_weekday'

        rel_complex_match = self.rx_rel_complex.search(text)
        if rel_complex_match:
            num_str, unit_full, unit_base, dir_str = rel_complex_match.groups()
            num = float(num_str) if num_str.isdigit() else float(self.number_words.get(num_str, 1))
            
            if 'و نیم' in text: num += 0.5
            direction = self.direction_words.get(dir_str, 1)
            
            if unit_base == 'روز': return anchor_date + jdatetime.timedelta(days=int(num * direction)), 'relative_complex'
            elif unit_base == 'هفته': return anchor_date + jdatetime.timedelta(days=int(num * direction * 7)), 'relative_complex'
            elif unit_base == 'ماه': 
                res_date = self._add_months(anchor_date, int(num) * direction)
                if num % 1 != 0: res_date += jdatetime.timedelta(days=15 * direction)
                return res_date, 'relative_complex'
            elif unit_base == 'سال': 
                target_year = anchor_date.year + (int(num) * direction)
                if target_year < 1 or target_year > 9999: return None
                target_month, target_day = anchor_date.month, anchor_date.day
                if target_month == 12 and target_day == 30 and not jdatetime.date(target_year, 1, 1).isleap():
                    target_day = 29
                res_date = jdatetime.date(target_year, target_month, target_day)
                if num % 1 != 0: res_date = self._add_months(res_date, 6 * direction)
                return res_date, 'relative_complex'

        rel_simple_match = self.rx_rel_simple.search(text)
        if rel_simple_match:
            unit_full, unit_base, dir_str = rel_simple_match.groups()
            direction = self.direction_words.get(dir_str, 1)
            num = 2 if ('ها' in unit_full or 'ان' in unit_full) else 1
            
            if unit_base == 'روز': return anchor_date + jdatetime.timedelta(days=num * direction), 'relative_simple'
            elif unit_base == 'هفته': return anchor_date + jdatetime.timedelta(days=7 * num * direction), 'relative_simple'
            elif unit_base == 'ماه': return self._add_months(anchor_date, num * direction), 'relative_simple'
            elif unit_base == 'سال': 
                target_year = anchor_date.year + (num * direction)
                if target_year < 1 or target_year > 9999: return None
                target_month, target_day = anchor_date.month, anchor_date.day
                if target_month == 12 and target_day == 30 and not jdatetime.date(target_year, 1, 1).isleap():
                    target_day = 29
                return jdatetime.date(target_year, target_month, target_day), 'relative_simple'

        # استفاده از regex از پیش‌کامپایل‌شده امکان‌پذیر نیست چون کلیدها متفاوتند، 
        # اما با توجه به حجم کم دیکشنری، جستجوی متنی ساده پایتون سریعتر از Regex است
        for word, offset in self.relative_map.items():
            if f" {word} " in text or text.startswith(f"{word} ") or text.endswith(f" {word}") or text == word:
                if offset == "weekend":
                    days_ahead = self.weekdays_map['جمعه'] - anchor_date.weekday()
                    if days_ahead <= 0: days_ahead += 7
                    return anchor_date + jdatetime.timedelta(days=days_ahead), 'relative_word'
                return anchor_date + jdatetime.timedelta(days=offset), 'relative_word'
            
        return None

    def extract_and_normalize(self, text, anchor_date):
        clean_text = self._clean_tweet(text)
        spans = self.dex.extract_span(clean_text)
        
        extracted_dates = []
        
        if spans and spans.get('datetime'):
            dts = spans['datetime']
            
            if isinstance(dts, dict):
                parsed_dts = []
                for key in dts.keys():
                    match = re.match(r'\[(\d+),\s*(\d+)\]', key)
                    if match:
                        parsed_dts.append([int(match.group(1)), int(match.group(2))])
                dts = sorted(parsed_dts)
                
            if isinstance(dts, list) and len(dts) > 0:
                merged_spans = []
                curr_start, curr_end = dts[0]
                for start, end in dts[1:]:
                    if start - curr_end <= 5: 
                        curr_end = max(curr_end, end)
                    else:
                        merged_spans.append((curr_start, curr_end))
                        curr_start, curr_end = start, end
                merged_spans.append((curr_start, curr_end))
                
                for start, end in merged_spans:
                    temporal_phrase = clean_text[start:end]
                    try:
                        parsed_result = self._parse_extracted_time(temporal_phrase, anchor_date)
                        if parsed_result:
                            extracted_dates.append(parsed_result)
                    except ValueError:
                        pass
        
        try:
            whole_text_result = self._parse_extracted_time(clean_text, anchor_date)
            if whole_text_result:
                extracted_dates.append(whole_text_result)
        except ValueError:
            pass

        if not extracted_dates:
            return None

        def sort_key(item):
            date, date_type = item
            priority_score = self.priority_map.get(date_type, 99)
            is_past = 1 if date < anchor_date else 0
            return (priority_score, is_past, -date.toordinal())
            
        extracted_dates.sort(key=sort_key)
        return extracted_dates[0][0] 


# ==========================================================================
# توابع واسط برای اعمال روی دیتافریم — این بخش بازنویسی شده تا برای حجم
# بالای داده بهینه باشد. منطق کلاس بالا کاملاً دست‌نخورده مانده است.
# ==========================================================================

def _gregorian_series_to_jalali(date_series: pd.Series):
    """
    تبدیل وکتورایز-شده‌ی ستون تاریخ میلادی به jdatetime.
    برخلاف نسخه‌ی قبلی که برای هر سطر جداگانه pd.to_datetime صدا می‌زد،
    اینجا فقط یک بار روی کل ستون اجرا می‌شود (که در pandas به‌شدت سریع‌تر است).
    خروجی یک Series از اشیای jdatetime.datetime (یا None) است تا از
    رفت‌وبرگشتِ اضافیِ string -> strptime در نسخه‌ی قبلی جلوگیری شود.
    """
    gdates = pd.to_datetime(date_series, errors='coerce')

    def _to_jalali(dt):
        if pd.isna(dt):
            return None
        try:
            return jdatetime.datetime.fromgregorian(datetime=dt)
        except Exception:
            return None

    return gdates.apply(_to_jalali)


# --- زیرساخت پردازش موازی ---
# هر پردازه (process) فقط یک بار Parstdex را می‌سازد (initializer) و برای
# تمام سطرهای همان chunk از آن استفاده می‌کند، به‌جای ساختن آن به ازای هر سطر.
_worker_normalizer = None


def _init_worker():
    global _worker_normalizer
    _worker_normalizer = TwitterTemporalNormalizer()


def _process_chunk(chunk):
    """
    chunk: لیستی از تاپل‌های (text, anchor_date)
    خروجی: لیستی هم‌طول از رشته‌ی تاریخ آینده (یا None)
    """
    global _worker_normalizer
    if _worker_normalizer is None:
        _worker_normalizer = TwitterTemporalNormalizer()

    results = []
    for text, anchor_date in chunk:
        if pd.isna(text) or anchor_date is None:
            results.append(None)
            continue
        try:
            norm_date = _worker_normalizer.extract_and_normalize(text, anchor_date)
        except ValueError:
            norm_date = None

        if norm_date is not None and norm_date > anchor_date:
            results.append(norm_date.strftime('%Y/%m/%d'))
        else:
            results.append(None)
    return results


def _chunk_list(lst, n_chunks):
    n = len(lst)
    if n == 0:
        return []
    size = max(1, -(-n // n_chunks))  # سقف تقسیم (ceil division)
    return [lst[i:i + size] for i in range(0, n, size)]


def process_temporal_data(
    df: pd.DataFrame,
    text_col: str = 'txtContent',
    date_col: str = 'date',
    n_jobs: int = None,
    parallel_threshold: int = 2000,
) -> pd.DataFrame:
    """
    خروجی دقیقاً همان ستون‌های نسخه‌ی قبلی را دارد
    (jalali_date, j_date, future_date) با این تفاوت که دیتافریم نهایی
    فقط شامل سطرهایی است که یک تاریخ آینده در آن‌ها شناسایی شده باشد.

    n_jobs: تعداد پردازه‌های موازی (پیش‌فرض: تعداد هسته‌های سیستم)
    parallel_threshold: کمتر از این تعداد سطر، پردازش موازی به‌صرفه نیست
                         و overhead ساخت پردازه‌ها از سود آن بیشتر می‌شود
    """
    print("⏳ در حال تبدیل تاریخ‌های میلادی به شمسی (وکتورایز شده)...")
    jdt_series = _gregorian_series_to_jalali(df[date_col])
    df['jalali_date'] = jdt_series.apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S") if x is not None else None)
    df['j_date'] = jdt_series.apply(lambda x: x.date() if x is not None else None)

    n = len(df)
    texts = df[text_col].tolist()
    anchors = df['j_date'].tolist()

    if n_jobs is None:
        n_jobs = cpu_count()

    if n < parallel_threshold or n_jobs <= 1:
        print(f"⏳ در حال استخراج و اعتبارسنجی تاریخ‌های آینده از متن ({n} سطر، تک‌پردازه‌ای)...")
        _init_worker()
        future_dates = _process_chunk(list(zip(texts, anchors)))
    else:
        print(f"⏳ در حال استخراج و اعتبارسنجی تاریخ‌های آینده از متن ({n} سطر، {n_jobs} پردازه موازی)...")
        pairs = list(zip(texts, anchors))
        chunks = _chunk_list(pairs, n_jobs)
        with Pool(processes=n_jobs, initializer=_init_worker) as pool:
            chunk_results = pool.map(_process_chunk, chunks)
        future_dates = [item for sub in chunk_results for item in sub]

    df['future_date'] = future_dates

    # فقط سطرهایی که به تاریخ آینده اشاره می‌کنند در خروجی باقی می‌مانند
    result_df = df[df['future_date'].notna()].reset_index(drop=True)

    print(f"✅ از {n} سطر، {len(result_df)} سطر شامل اشاره به تاریخ آینده بودند.")
    return result_df
