import os
import time
import pandas as pd
from clickhouse_connect import get_client


class DataLoader:
    def __init__(self, db_name: str, table_name: str):
        """
        Initialize the loader with a ClickHouse database name and table name.
        """
        self.db_name = db_name
        self.table_name = table_name

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

    def load_and_prepare(self, text_col: str = '0', date_col: str = '2') -> pd.DataFrame:
        """
        Read data from the ClickHouse table, drop rows with missing
        text/date, and sort by date.
        """
        try:
            print(f"[DataLoader] Connecting to ClickHouse | database: {self.db_name} | table: {self.table_name}")
            t0 = time.time()

            # Open the connection
            client = self._get_ch_client()

            # Run the query and get the result directly as a Pandas DataFrame
            query = f"SELECT * FROM {self.table_name}"
            print(f"[DataLoader] Running query: {query}")
            df = client.query_df(query)

            # Close the connection
            client.close()

            elapsed = time.time() - t0
            print(f"[DataLoader] Query finished in {elapsed:.2f}s | rows fetched: {len(df)}")

            # Check that the expected columns exist in the returned DataFrame
            if text_col in df.columns and date_col in df.columns:
                before = len(df)

                # Drop rows with no text or no date
                df = df.dropna(subset=[text_col, date_col])

                # Convert the date column to datetime and sort
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
