# # # python3 main.py --db-name "x" --table-name "tweets_2" --start-date "1404-10-01" --end-date "1404-10-10"
# # # Twitter/X (Jalali dates, shdate column):
# # # python3 main.py --db-name "x" --table-name "tweets_2" --start-date "1404-10-01" --end-date "1404-10-10"
# # # Telegram (Jalali dates, shdate column):
# # # python3 main.py --db-name "telegram" --table-name "posts" --start-date "1404-10-01" --end-date "1404-10-10"




import os
import time
import re
from typing import Optional

import pandas as pd
from clickhouse_connect import get_client
import pg8000


class DataLoader:

    TWITTER_DB_NAMES = {"x", "twitter", "telegram", "eita", "rubika", "bale"}

    # Jalali dates validation (YYYY-MM-DD)
    JALALI_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

    def __init__(self, db_name: str, table_name: str):
        """
        Initialize the loader with a ClickHouse database name and table name.
        """
        self.db_name = db_name
        self.table_name = table_name
        self.is_twitter = self.db_name.strip().lower() in self.TWITTER_DB_NAMES

    def _get_ch_client(self):
        """
        Create a connection to ClickHouse using environment variables
        (falling back to default values if they are not set).
        """
        return get_client(
            host=os.getenv("CH_HOST"),
            port=int(os.getenv("CH_PORT")),
            database=self.db_name,
            username=os.getenv("CH_USER"),
            password=os.getenv("CH_PASS")
        )

    @staticmethod
    def _get_pg_conn():
        """Connection to the Postgres DB that holds 'topics' / 'topic_keywords'
        (same DB/credentials sts_job_fin.py uses)."""
        return pg8000.connect(
            host=os.getenv("PG_HOST"),
            port=int(os.getenv("PG_PORT")),
            database=os.getenv("PG_DB"),
            user=os.getenv("PG_USER"),
            password=os.getenv("PG_PASS")
        )

    @classmethod
    def _fetch_topic_keywords(cls, topic_id: int) -> list:
        """Looks up the keyword list for a given topic_id from Postgres
        (same 'topic_keywords' table sts_job_fin.py reads from)."""
        conn = cls._get_pg_conn()
        try:
            cur = conn.cursor()
            cur.execute("SELECT keyword FROM topic_keywords WHERE topic_id = %s", (topic_id,))
            return [r[0] for r in cur.fetchall()]
        finally:
            conn.close()

    @staticmethod
    def _ch_array_str(strings: list) -> str:
        """Formats a Python list of strings as a ClickHouse Array(String)
        literal (same escaping sts_job_fin.py uses)."""
        if not strings:
            return "[]"
        escaped = [s.replace("'", "\\'") for s in strings]
        return "[" + ",".join(f"'{s}'" for s in escaped) + "]"

    @classmethod
    def _validate_jalali_date(cls, date_str: str) -> None:
        if not cls.JALALI_DATE_RE.match(date_str):
            raise ValueError(
                f"Invalid Jalali date format: {date_str!r}. Expected 'YYYY-MM-DD' "
                f"(e.g. '1404-10-01')."
            )

    @staticmethod
    def _shamsi_to_gregorian(shdate_str: str) -> Optional[str]:
        try:
            import jdatetime
            y, m, d = map(int, shdate_str.split('-'))
            return jdatetime.date(y, m, d).togregorian().strftime('%Y-%m-%d')
        except ImportError:
            print("[DataLoader] WARNING: 'jdatetime' is missing. Gregorian fallback aborted.")
            return None
        except (ValueError, TypeError) as e:
            # Defensive: an invalid/unparseable Jalali date string should
            # skip the Gregorian fallback for THIS date, not blow up the
            # whole load_and_prepare() call (and with it, the whole day/run).
            print(f"[DataLoader] WARNING: could not convert {shdate_str!r} to Gregorian ({e}). "
                  f"Gregorian fallback aborted for this date.")
            return None

    def load_and_prepare(
        self,
        text_col: Optional[str] = None,
        date_col: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        filter_by_topic: bool = False,
        topic_id: Optional[int] = None,
    ) -> pd.DataFrame:

        text_col = text_col or 'txtContent'
        current_date_col = date_col or 'shdate'

        # ---------------------------------------------------------
        # Optional topic-keyword condition (same idea as sts_job_fin.py:
        # fetch_topics_map + multiSearchAny). This only ADDS a keyword
        # condition on top of whatever date range is already requested -
        # the existing date-range / social-network (db_name) logic below
        # is untouched.
        # ---------------------------------------------------------
        keyword_condition_sql = None
        if filter_by_topic:
            if topic_id is None:
                print("[DataLoader] WARNING: filter_by_topic=True but no topic_id was given. "
                      "Skipping keyword filter.")
            else:
                keywords = self._fetch_topic_keywords(topic_id)
                if keywords:
                    kws_sql = self._ch_array_str(keywords)
                    keyword_condition_sql = f'multiSearchAny("{text_col}", {kws_sql})'
                    print(f"[DataLoader] Topic keyword filter ENABLED | topic_id={topic_id} | "
                          f"{len(keywords)} keyword(s)")
                else:
                    print(f"[DataLoader] WARNING: no keywords found in Postgres for "
                          f"topic_id={topic_id}. Skipping keyword filter.")

        client = None

        try:
            db_kind = "x" if self.is_twitter else "insta"
            print(f"[DataLoader] Connecting to ClickHouse | database: {self.db_name} ({db_kind}) | table: {self.table_name}")
            t0 = time.time()

            # Open the connection
            client = self._get_ch_client()

            # ---------------------------------------------------------
            # Phase 1: query using shdate (Jalali string, e.g. '1404-10-01')
            # ---------------------------------------------------------
            query = f"SELECT * FROM {self.table_name}"
            conditions = []

            if start_date:
                self._validate_jalali_date(start_date)
                conditions.append(f'"{current_date_col}" >= \'{start_date}\'')
            if end_date:
                self._validate_jalali_date(end_date)
                conditions.append(f'"{current_date_col}" < \'{end_date}\'')
            if keyword_condition_sql:
                conditions.append(keyword_condition_sql)

            if conditions:
                query += " WHERE " + " AND ".join(conditions)

            print(f"[DataLoader] Running Phase 1 (shdate): {query}")
            df = client.query_df(query)

            if df.empty and (start_date or end_date):
                print(f"[DataLoader] No records found using '{current_date_col}'. Trying Phase 2 (Gregorian fallback on 'date').")
                g_start = self._shamsi_to_gregorian(start_date) if start_date else None
                g_end = self._shamsi_to_gregorian(end_date) if end_date else None

                if g_start or g_end:
                    fallback_query = f"SELECT * FROM {self.table_name}"
                    fallback_conds = []

                    if g_start:
                        fallback_conds.append(f'toDate("date") >= \'{g_start}\'')
                    if g_end:
                        fallback_conds.append(f'toDate("date") < \'{g_end}\'')
                    if keyword_condition_sql:
                        fallback_conds.append(keyword_condition_sql)

                    if fallback_conds:
                        fallback_query += " WHERE " + " AND ".join(fallback_conds)

                    print(f"[DataLoader] Running Phase 2 (date): {fallback_query}")
                    df = client.query_df(fallback_query)
                    current_date_col = 'date'

            elapsed = time.time() - t0
            print(f"[DataLoader] Query finished in {elapsed:.2f}s | rows fetched: {len(df)}")

            # --- ??????? ?? ??? ???????? ???? ---
            if df.empty:
                return df

            # Check that the expected columns exist in the returned DataFrame
            if text_col in df.columns and current_date_col in df.columns:
                before = len(df)

                # Drop rows with no text or no date
                df = df.dropna(subset=[text_col, current_date_col])

                if current_date_col == 'shdate':
                    # ??? ???? ???? ???? FixedString ????????? ?? ????? ??????
                    df[current_date_col] = df[current_date_col].apply(
                        lambda v: v.decode('utf-8') if isinstance(v, (bytes, bytearray)) else v
                    ).astype(str).str.strip()
                else:
                    # ?????? ??? ?? ????
                    df[current_date_col] = pd.to_datetime(df[current_date_col])

                df = df.sort_values(by=current_date_col)

                print(f"[DataLoader] Dropped {before - len(df)} row(s) with missing text/date. "
                      f"{len(df)} valid row(s) remain, sorted by '{current_date_col}'.")
            else:
                print(f"[DataLoader] WARNING: column '{text_col}' or '{current_date_col}' was not found in the table. "
                      f"Skipped the dropna/sort preparation step.")

            return df

        except Exception as e:
            print(f"[DataLoader] ERROR while loading data from the database: {e}")
            raise

        finally:
            if client:
                client.close()
