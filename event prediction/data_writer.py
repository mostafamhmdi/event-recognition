# python3 main.py --db-name "x" --table-name "tweets_2" --start-date "1404-06-01" --end-date "1404-06-14"


import json
import os
from datetime import datetime

import pandas as pd
from clickhouse_connect import get_client


def _as_datetime(value):
    """Accepts a pandas Timestamp / python datetime / date-like / string
    and returns a plain python datetime, which clickhouse_connect expects
    for DateTime columns."""
    return pd.Timestamp(value).to_pydatetime()


def _sample_json(messages, limit=15):
    return json.dumps({"messages": list((messages or [])[:limit])}, ensure_ascii=False)


class VerifiedEventsWriter:
    """
    One public method - save_events(...) - inserts one row per
    LLM-confirmed event into `self.event_table`.
    """

    def __init__(self, db_name="raya_sepehr_analytical", source="telegram",
                 event_table="predicted_events"):
        self.db_name = db_name
        self.source = source
        self.event_table = event_table
        self._client = None

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

    # ---- insert ----

    def save_events(self, verified_events, execution_time):
        if not verified_events:
            return

        exec_dt = _as_datetime(execution_time)
        created_dt = datetime.now()

        try:
            client = self._get_client()
        except Exception as e:
            print(f"[ClickHouse] ERROR connecting for insert into {self.event_table}: {e}")
            raise

        rows = []
        for event in verified_events:
            # بررسی و تبدیل زمان به آبجکت datetime به جای string
            pred_time_raw = event.get('predicted_time')
            pred_time_dt = _as_datetime(pred_time_raw) if pred_time_raw else None

            rows.append([
                event.get('event_title') or 'N/A',
                event.get('event_summary') or 'N/A',
                pred_time_dt,  # <--- استفاده از آبجکت datetime به جای str()
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
                    'title', 'summary', 'predicted_time',
                    'predicted_location', 'execution_time', 'source',
                    'sample_messages', 'created_at',
                ],
            )
            print(f"[ClickHouse] Inserted {len(rows)} row(s) into "
                  f"{self.event_table}")
        except Exception as e:
            print(f"[ClickHouse] ERROR inserting into {self.event_table}: {e}")