# """
# Writes Stage 5 (LLM-verified) events into ClickHouse, into a single table,
# with exactly the schema requested:

#     id, title, summary, predicted_time, predicted_location,
#     execution_time, source, sample_messages, created_at

# Connects to the SAME ClickHouse instance/credentials as
# ClickHouseResultsWriter (data_writer.py) - same host/port/user/pass
# defaults, same CH_HOST/CH_PORT/CH_USER/CH_PASS env-var overrides - so this
# can point at the exact same database, just a different table.

# Field-mapping notes:

#   title / summary
#     <- event['event_title'] / event['event_summary'], set by
#     QwenEventVerifier.verify_candidates() in ev.py.

#   predicted_time / predicted_location
#     <- event['predicted_time'] / event['predicted_location'], set by
#     build_candidates_from_df() in main.py (the mode of future_date /
#     location_entities across the cluster's messages, straight from the
#     Stage-4 dataframe).

#   sample_messages
#     <- event['sample_messages'], the exact (<=15) messages sampled and
#     actually sent to the LLM - added onto each confirmed candidate by the
#     one-line edit already made in ev.py's verify_candidates(). Stored as a
#     ClickHouse JSON column; wrapped in {"messages": [...]} the same way
#     data_writer.py does, since ClickHouse's JSON column type only accepts
#     an object at the root, not a bare array.

#   id / created_at
#     Generated here at insert time: id is a fresh UUID4 (string) per row,
#     created_at is "now" (when the row is written to the DB) - separate
#     from execution_time, which is when the underlying batch of messages
#     was processed by the pipeline (see below).

#   execution_time
#     Passed in by the caller (main.py passes day_start - the calendar day
#     whose messages this event came from). Stored as-is for every row in
#     a given save_events() call.

# Suggested DDL (adjust types/engine to taste - this is just one that fits
# the field list above):

#     CREATE TABLE IF NOT EXISTS verified_events
#     (
#         id                 String,
#         title              String,
#         summary            String,
#         predicted_time     String,
#         predicted_location String,
#         execution_time     DateTime,
#         source             String,
#         sample_messages    JSON,
#         created_at         DateTime
#     )
#     ENGINE = MergeTree
#     ORDER BY (execution_time, id);
# """

# import os
# import uuid
# from datetime import datetime

# import pandas as pd
# from clickhouse_connect import get_client


# def _as_datetime(value):
#     """Accepts a pandas Timestamp / python datetime / date-like / string
#     and returns a plain python datetime, which clickhouse_connect expects
#     for DateTime columns."""
#     return pd.Timestamp(value).to_pydatetime()


# def _sample(messages, limit=15):
#     """ClickHouse's JSON column type only accepts an object at the root
#     (not a bare array) - hence the {"messages": [...]} wrapper instead of
#     just a list. Capped at 15 since that's the sample size verify_candidates()
#     actually sends to the LLM."""
#     return {"messages": list((messages or [])[:limit])}


# class VerifiedEventsWriter:
#     """
#     One public method - save_events(...) - inserts one row per
#     LLM-confirmed event into `self.event_table`.
#     """

#     def __init__(self, db_name="raya_sepehr_analytical", source="telegram",
#                  event_table="predicted_events"):
#         self.db_name = db_name
#         self.source = source
#         self.event_table = event_table
#         self._client = None

#     # ---- connection ----

#     def _get_client(self):
#         if self._client is None:
#             self._client = get_client(
#                 host=os.getenv("CH_HOST", '172.20.70.191'),
#                 port=int(os.getenv("CH_PORT", 8123)),
#                 database=self.db_name,
#                 username=os.getenv("CH_USER", 'labafi'),
#                 password=os.getenv("CH_PASS", 'l@b@fi@1234')
#             )
#         return self._client

#     def close(self):
#         if self._client is not None:
#             self._client.close()
#             self._client = None

#     # ---- insert ----

#     def save_events(self, verified_events, execution_time):
#         """
#         Inserts one row per item in `verified_events` - i.e. exactly what
#         QwenEventVerifier.verify_candidates() returns (the list of
#         candidates it confirmed as real events). Each item is expected to
#         already carry:
#             event_title, event_summary   (added by ev.py on confirmation)
#             predicted_time, predicted_location
#                                           (added by build_candidates_from_df()
#                                            in main.py, before verification)
#             sample_messages               (added by ev.py on confirmation)

#         `execution_time` is the calendar day this whole batch belongs to
#         (main.py's day_start) - the same value is stored on every row
#         inserted by this call.

#         Safe to call with an empty/None list (no-op).
#         """
#         if not verified_events:
#             return

#         exec_dt = _as_datetime(execution_time)
#         created_dt = datetime.now()

#         rows = []
#         for event in verified_events:
#             rows.append([
#                 str(uuid.uuid4()),
#                 event.get('event_title') or 'N/A',
#                 event.get('event_summary') or 'N/A',
#                 str(event.get('predicted_time') or ''),
#                 str(event.get('predicted_location') or ''),
#                 exec_dt,
#                 self.source,
#                 _sample(event.get('sample_messages') or event.get('messages')),
#                 created_dt,
#             ])

#         try:
#             client = self._get_client()
#             client.insert(
#                 self.event_table,
#                 rows,
#                 column_names=[
#                     'id', 'title', 'summary', 'predicted_time',
#                     'predicted_location', 'execution_time', 'source',
#                     'sample_messages', 'created_at',
#                 ],
#             )
#             print(f"[ClickHouse] Inserted {len(rows)} row(s) into "
#                   f"{self.event_table}")
#         except Exception as e:
#             print(f"[ClickHouse] ERROR inserting into {self.event_table}: {e}")

# python3 main.py --db-name "x" --table-name "tweets_2" --start-date "1404-06-01" --end-date "1404-06-14"


import json
import os
from datetime import datetime

import pandas as pd
from clickhouse_connect import get_client

START_ID = 10000


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

    # ---- id sequencing ----

    def _get_next_id(self, client) -> int:
        """
        Returns the next id to use for this save_events() call, i.e.
        (current max id in the table) + 1, or START_ID if the table is
        empty or doesn't exist yet.

        `id` is a String column, so the max is computed by casting each
        value to an integer first (toInt64OrZero -> 0 for anything
        non-numeric/empty, so old/garbage ids can't break this).
        """
        try:
            result = client.query(
                f'SELECT max(toInt64OrZero(id)) FROM {self.event_table}'
            )
            max_id = result.result_rows[0][0] if result.result_rows else None
        except Exception as e:
            print(f"[ClickHouse] WARNING: could not read current max id from "
                  f"{self.event_table} ({e}); starting from {START_ID}.")
            max_id = None

        if not max_id:
            return START_ID
        return int(max_id) + 1

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

        next_id = self._get_next_id(client)

        rows = []
        for offset, event in enumerate(verified_events):
            rows.append([
                str(next_id + offset),
                event.get('event_title') or 'N/A',
                event.get('event_summary') or 'N/A',
                str(event.get('predicted_time') or ''),
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
            print(f"[ClickHouse] Inserted {len(rows)} row(s) into "
                  f"{self.event_table}")
        except Exception as e:
            print(f"[ClickHouse] ERROR inserting into {self.event_table}: {e}")