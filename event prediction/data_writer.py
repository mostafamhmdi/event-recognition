# # python3 main.py --db-name "x" --table-name "tweets_2" --start-date "1404-06-01" --end-date "1404-06-14"

import json
import os
import time
from datetime import datetime
import pandas as pd
import jdatetime  # اضافه شدن کتابخانه تاریخ شمسی
from clickhouse_connect import get_client

def _as_datetime(value):
    return pd.Timestamp(value).to_pydatetime()

def _sample_json(messages, limit=15):
    return json.dumps({"messages": list((messages or [])[:limit])}, ensure_ascii=False)

def _parse_and_standardize_datetime(value):
    """
    تاریخ را بررسی کرده و در صورت شمسی بودن، آن را به میلادی استاندارد می‌کند.
    در صورت نامعتبر بودن یا خالی بودن، تاریخ پیش‌فرض ارسال می‌شود تا رکورد از دست نرود و خطا هم ندهد.
    """
    if not value:
        return datetime(1970, 1, 1)
    
    val_str = str(value).strip()
    
    # بررسی فرمت شمسی (مثلاً با 13 یا 14 شروع شده باشد)
    if val_str.startswith('13') or val_str.startswith('14'):
        try:
            # جایگزینی اسلش با خط تیره و جداسازی تاریخ از ساعت احتمالی
            date_part = val_str.replace('/', '-').split(' ')[0]
            y, m, d = map(int, date_part.split('-'))
            
            # تبدیل شمسی به میلادی
            gregorian_date = jdatetime.date(y, m, d).togregorian()
            return datetime(gregorian_date.year, gregorian_date.month, gregorian_date.day)
        except Exception:
            return datetime(1970, 1, 1)

    # اگر شمسی نبود، با همان پانداس (میلادی) پارس شود
    try:
        dt = pd.Timestamp(value)
        if pd.isna(dt):
            return datetime(1970, 1, 1)
        return dt.to_pydatetime()
    except Exception:
        return datetime(1970, 1, 1)

class VerifiedEventsWriter:
    def __init__(self, db_name="raya_sepehr_analytical", source="telegram",
                 event_table="predicted_events"):
        self.db_name = db_name
        self.source = source
        self.event_table = event_table
        self._client = None

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

    def save_events(self, verified_events, execution_time):
        if not verified_events:
            return True

        exec_dt = _as_datetime(execution_time)
        created_dt = datetime.now()

        try:
            client = self._get_client()
        except Exception as e:
            print(f"[ClickHouse] ERROR connecting for insert into {self.event_table}: {e}")
            raise

        rows = []
        # 'id' used to be left out of the INSERT entirely, so ClickHouse
        # silently filled it with the column's default (0) on every single
        # row - that's why the table shows id = 0 for everything. If
        # 'predicted_events' is a Replacing/Collapsing/AggregatingMergeTree
        # with 'id' in its ORDER BY (a very common pattern for a table
        # named like this), ClickHouse treats every id=0 row as the SAME
        # logical row and can silently collapse them down to just one
        # survivor on the next background merge - i.e. real, silent loss
        # of previously "saved" events. Giving every row its own unique id
        # fixes that regardless of which MergeTree variant the table uses.
        # Nanosecond epoch + row offset keeps ids unique within a batch and
        # comfortably fits a UInt64 id column (no overflow risk).
        base_id = time.time_ns()
        for i, event in enumerate(verified_events):
            # استفاده از تابع جدید برای تبدیل اتوماتیک شمسی به میلادی
            pred_time_dt = _parse_and_standardize_datetime(event.get('predicted_time'))

            rows.append([
                base_id + i,
                event.get('event_title') or 'N/A',
                event.get('event_summary') or 'N/A',
                pred_time_dt,
                str(event.get('predicted_location') or ''),
                exec_dt,
                self.source,
                _sample_json(event.get('sample_messages') or event.get('messages')),
                created_dt,
            ])

        try:
            client.insert(
                self.event_table,
                rows,
                column_names=[
                    'id', 'title', 'summary', 'predicted_time',
                    'predicted_location', 'execution_time', 'source',
                    'sample_messages', 'created_at',
                ],
            )
            print(f"[ClickHouse] Inserted {len(rows)} row(s) into {self.event_table}")
            return True
        except Exception as e:
            # این خطا قبلاً فقط پرینت می‌شد و کد صدازننده هیچ‌وقت متوجه
            # شکست نمی‌شد (چون خروجی این تابع چک نمی‌شد) - یعنی اگر اینسرت
            # به هر دلیلی (مثلاً mismatch نوع ستون) fail می‌شد، لاگ‌ها هنوز
            # "موفق" نشون می‌دادن. الان return False می‌کنیم تا صدازننده
            # (main.py) در صورت تمایل بتونه شکست رو تشخیص بده.
            print(f"[ClickHouse] ERROR inserting into {self.event_table}: {e}")
            return False