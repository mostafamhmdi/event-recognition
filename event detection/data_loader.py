# # python3 main.py --db-name "x" --table-name "tweets_2" --start-date "1404-10-01" --end-date "1404-10-10"
# # Twitter/X (Jalali dates, shdate column):
# # python3 main.py --db-name "x" --table-name "tweets_2" --start-date "1404-10-01" --end-date "1404-10-10"
# # Telegram:
# # python3 main.py --db-name "telegram" --table-name "posts" --start-date "2024-01-01" --end-date "2024-01-10"



import os
import time
import re
from typing import Optional

import pandas as pd
from clickhouse_connect import get_client


class DataLoader:
    # db_name values that should be treated as "Twitter/X"
    TWITTER_DB_NAMES = {"x", "twitter"}

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
            host=os.getenv("CH_HOST", '172.20.70.191'),
            port=int(os.getenv("CH_PORT", 8123)),
            database=self.db_name,
            username=os.getenv("CH_USER", 'labafi'),
            password=os.getenv("CH_PASS", 'l@b@fi@1234')
        )

    @classmethod
    def _validate_jalali_date(cls, date_str: str) -> None:
        if not cls.JALALI_DATE_RE.match(date_str):
            raise ValueError(
                f"Invalid Jalali date format: {date_str!r}. Expected 'YYYY-MM-DD' "
                f"(e.g. '1404-10-01')."
            )

    def load_and_prepare(
        self,
        text_col: Optional[str] = None,
        date_col: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        
        if self.is_twitter:
            text_col = text_col or 'txtContent'
            date_col = date_col or 'shdate'
        else:
            text_col = text_col or 'txtContent'
            date_col = date_col or 'date'

        try:
            db_kind = "x" if self.is_twitter else "telegram"
            print(f"[DataLoader] Connecting to ClickHouse | database: {self.db_name} ({db_kind}) | table: {self.table_name}")
            t0 = time.time()

            # Open the connection
            client = self._get_ch_client()

            # Build the query
            query = f"SELECT * FROM {self.table_name}"
            conditions = []
            parameters = {}

            if self.is_twitter:
                if start_date:
                    self._validate_jalali_date(start_date)
                    conditions.append(f'"{date_col}" >= \'{start_date}\'')
                if end_date:
                    self._validate_jalali_date(end_date)
                    conditions.append(f'"{date_col}" < \'{end_date}\'')
            else:
                if start_date:
                    conditions.append(f'toDate("{date_col}") >= {{start_date:Date}}')
                    parameters['start_date'] = start_date
                if end_date:
                    conditions.append(f'toDate("{date_col}") < {{end_date:Date}}')
                    parameters['end_date'] = end_date

            if conditions:
                query += " WHERE " + " AND ".join(conditions)

            print(f"[DataLoader] Running query: {query} | params: {parameters}")

            df = client.query_df(query, parameters=parameters) if parameters else client.query_df(query)

            # Close the connection
            client.close()

            elapsed = time.time() - t0
            print(f"[DataLoader] Query finished in {elapsed:.2f}s | rows fetched: {len(df)}")

            # --- ??????? ?? ??? ???????? ???? ---
            if df.empty:
                return df

            # Check that the expected columns exist in the returned DataFrame
            if text_col in df.columns and date_col in df.columns:
                before = len(df)

                # Drop rows with no text or no date
                df = df.dropna(subset=[text_col, date_col])

                if self.is_twitter:
                    # ??? ???? ???? ???? FixedString ????????? ?? ????? ??????
                    df[date_col] = df[date_col].apply(
                        lambda v: v.decode('utf-8') if isinstance(v, (bytes, bytearray)) else v
                    ).astype(str).str.strip()
                    df = df.sort_values(by=date_col)
                else:
                    # ?????? ??? ?? ????
                    df[date_col] = pd.to_datetime(df[date_col])
                    df = df.sort_values(by=date_col)

                print(f"[DataLoader] Dropped {before - len(df)} row(s) with missing text/date. "
                      f"{len(df)} valid row(s) remain, sorted by '{date_col}'.")
            else:
                print(f"[DataLoader] WARNING: column '{text_col}' or '{date_col}' was not found in the table. "
                      f"Skipped the dropna/sort preparation step.")

            return df

        except Exception as e:
            print(f"[DataLoader] ERROR while loading data from the database: {e}")
            raise