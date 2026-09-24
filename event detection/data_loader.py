
## python3 main.py --db-name "x" --table-name "tweets_2" --start-date "1404-04-01" --end-date "1404-04-10"
## python3 main.py --db-name "telegram" --table-name "posts" --start-date "1404-06-06" --end-date "1404-06-08"

import os
import time
import re
from typing import Optional

import pandas as pd
from clickhouse_connect import get_client
import pg8000


class DataLoader:
    # db_name values that should be treated as "Twitter/X"
    TWITTER_DB_NAMES = {"x", "twitter"}

    # ???? ?????????? ??????? (?? ????? ????? ????)
    EMOTION_MAP = {
        'خنثی': 0, 'neutral': 0,
        'شادی': 1, 'خوشحالی': 1, 'happy': 1, 'happiness': 1, 'joy': 1,
        'غم': 2, 'غمگین': 2, 'ناراحتی': 2, 'sad': 2, 'sadness': 2, 'grief': 2,
        'عصبانیت': 3, 'خشم': 3, 'anger': 3, 'angry': 3,
        'ترس': 4, 'fear': 4, 'afraid': 4, 'scared': 4,
        'نفرت': 5, 'انزجار': 5, 'hate': 5, 'hatred': 5, 'disgust': 5,
        'تعجب': 6, 'شگفتی': 6, 'surprise': 6, 'surprised': 6,
    }


    # Jalali dates validation (YYYY-MM-DD)
    JALALI_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

    def __init__(self, db_name: str, table_name: str):
        self.db_name = db_name
        self.table_name = table_name
        self.is_twitter = self.db_name.strip().lower() in self.TWITTER_DB_NAMES

    def _get_ch_client(self):
        return get_client(
            host=os.getenv("CH_HOST"),
            port=int(os.getenv("CH_PORT")),
            database=self.db_name,
            username=os.getenv("CH_USER"),
            password=os.getenv("CH_PASS")
        )

    @staticmethod
    def _get_pg_conn():
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
        # Same escaping/formatting sts_job_fin.py uses to build a ClickHouse Array(String) literal
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
            db_kind = "x" if self.is_twitter else "telegram"
            print(f"[DataLoader] Connecting to ClickHouse | database: {self.db_name} ({db_kind}) | table: {self.table_name}")
            t0 = time.time()

            client = self._get_ch_client()


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
                    
                    if g_start: fallback_conds.append(f'toDate("date") >= \'{g_start}\'')
                    if g_end: fallback_conds.append(f'toDate("date") < \'{g_end}\'')
                    if keyword_condition_sql: fallback_conds.append(keyword_condition_sql)
                    
                    if fallback_conds:
                        fallback_query += " WHERE " + " AND ".join(fallback_conds)
                        
                    print(f"[DataLoader] Running Phase 2 (date): {fallback_query}")
                    df = client.query_df(fallback_query)
                    current_date_col = 'date'

            elapsed = time.time() - t0
            print(f"[DataLoader] Base query finished in {elapsed:.2f}s | rows fetched: {len(df)}")

            if df.empty:
                return df

            # ---------------------------------------------------------
            # ??? ?: ??????? ? ?????????
            # ---------------------------------------------------------
            if text_col in df.columns and current_date_col in df.columns:
                before = len(df)
                df = df.dropna(subset=[text_col, current_date_col])

                if current_date_col == 'shdate':
                    df[current_date_col] = df[current_date_col].apply(
                        lambda v: v.decode('utf-8') if isinstance(v, (bytes, bytearray)) else v
                    ).astype(str).str.strip()
                else:
                    df[current_date_col] = pd.to_datetime(df[current_date_col])
                    
                df = df.sort_values(by=current_date_col)
                print(f"[DataLoader] Dropped {before - len(df)} row(s). {len(df)} valid row(s) remain.")
            
            # ?????? ???? ???? ????? ? ???????? ????????
            def _safe_str(v):
                return v.decode('utf-8') if isinstance(v, (bytes, bytearray)) else str(v)

            def _clean_id(v):
                if pd.isna(v): return ""
                if isinstance(v, float): return str(int(v))
                return _safe_str(v)

            # ---------------------------------------------------------
            # ??? ?: ?????? sentiment ? emotion_label ???? ??????
            # ---------------------------------------------------------
            if self.is_twitter and not df.empty:
                if 'user_id' in df.columns and 'tweet_id' in df.columns:
                    print("[DataLoader] Fetching sentiment & emotion from analytical DB for Twitter data...")
                    df['row_key'] = df['user_id'].apply(_clean_id) + "||" + df['tweet_id'].apply(_clean_id)
                    keys_list = df['row_key'].unique().tolist()
                    
                    an_query = """
                        SELECT row_key, sentiment, emotion_label 
                        FROM raya_sepehr_analytical.post_analysis 
                        WHERE row_key IN {keys:Array(String)}
                    """
                    try:
                        an_df = client.query_df(an_query, parameters={'keys': keys_list})
                        if not an_df.empty:
                            an_df['row_key'] = an_df['row_key'].apply(_safe_str)
                            df = df.merge(an_df, on='row_key', how='left')
                            print(f"[DataLoader] Joined labels for {len(an_df)} unique tweets.")
                        else:
                            df['sentiment'] = None
                            df['emotion_label'] = None
                    except Exception as an_err:
                        print(f"[DataLoader] ERROR fetching analytical data: {an_err}")
                        df['sentiment'] = None
                        df['emotion_label'] = None
                    
                    df = df.drop(columns=['row_key'])

            elif not self.is_twitter and not df.empty:
                if 'channel' in df.columns and 'msgid' in df.columns:
                    print("[DataLoader] Fetching comments and analytical data for Telegram posts...")
                    
                    df['post_key'] = df['channel'].apply(_clean_id) + "||" + df['msgid'].apply(_clean_id)
                    post_keys_list = df['post_key'].unique().tolist()
                    
                    try:
                        batch_size = 1000
                        all_comments_dfs = []
                        
                        for i in range(0, len(post_keys_list), batch_size):
                            batch_keys = post_keys_list[i:i + batch_size]
                            
                            c_query = """
                                SELECT channel, msgid, comment_id
                                FROM telegram.comments
                                WHERE concat(toString(channel), '||', toString(msgid)) IN {pk:Array(String)}
                            """
                            batch_df = client.query_df(c_query, parameters={'pk': batch_keys})
                            all_comments_dfs.append(batch_df)
                            
                        comments_df = pd.concat(all_comments_dfs, ignore_index=True) if all_comments_dfs else pd.DataFrame()
                        
                        if not comments_df.empty:
                            comments_df['post_key'] = comments_df['channel'].apply(_clean_id) + "||" + comments_df['msgid'].apply(_clean_id)
                            comments_df['row_key'] = comments_df['post_key'] + "||" + comments_df['comment_id'].apply(_clean_id)
                            
                            c_keys = comments_df['row_key'].unique().tolist()
                            
                            all_an_dfs = []
                            for i in range(0, len(c_keys), batch_size):
                                batch_an_keys = c_keys[i:i + batch_size]
                                an_query = """
                                    SELECT row_key, sentiment, emotion_label 
                                    FROM raya_sepehr_analytical.post_analysis 
                                    WHERE row_key IN {keys:Array(String)}
                                """
                                batch_an_df = client.query_df(an_query, parameters={'keys': batch_an_keys})
                                all_an_dfs.append(batch_an_df)
                                
                            an_df = pd.concat(all_an_dfs, ignore_index=True) if all_an_dfs else pd.DataFrame()
                            
                            if not an_df.empty:
                                an_df['row_key'] = an_df['row_key'].apply(_safe_str)
                                comments_df = comments_df.merge(an_df, on='row_key', how='left')
                                
                                comments_df['emotion_norm'] = comments_df['emotion_label'].map(self.EMOTION_MAP)
                                
                                print("[DataLoader] Aggregating sentiments and emotions back to posts...")
                                
                                total_counts = comments_df.groupby('post_key').size()
                                sent_null_counts = comments_df['sentiment'].isna().groupby(comments_df['post_key']).sum()
                                emo_null_counts = comments_df['emotion_norm'].isna().groupby(comments_df['post_key']).sum()
                                
                                valid_sent_mask = (sent_null_counts / total_counts) <= 0.6
                                valid_emo_mask = (emo_null_counts / total_counts) <= 0.6
                                
                                sent_mean = comments_df.groupby('post_key')['sentiment'].mean()
                                sent_std = comments_df.groupby('post_key')['sentiment'].std()
                                
                                def get_mode(s):
                                    m = s.dropna().mode()
                                    return m.iloc[0] if not m.empty else None
                                    
                                emo_mode = comments_df.groupby('post_key')['emotion_norm'].apply(get_mode)
                                
                                agg_df = pd.DataFrame({
                                    'valid_sent': valid_sent_mask,
                                    'valid_emo': valid_emo_mask,
                                    'sentiment': sent_mean,
                                    'sentiment_std': sent_std,
                                    'emotion_label': emo_mode
                                })
                                
                                agg_df.loc[~agg_df['valid_sent'], ['sentiment', 'sentiment_std']] = None
                                agg_df.loc[~agg_df['valid_emo'], 'emotion_label'] = None
                                
                                agg_df = agg_df.drop(columns=['valid_sent', 'valid_emo']).reset_index()
                                df = df.merge(agg_df, on='post_key', how='left')
                                
                                print(f"[DataLoader] Telegram aggregation complete. Sentiments appended.")
                            else:
                                print("[DataLoader] No analytical data found for these comments.")
                        else:
                            print("[DataLoader] No comments found in telegram.comments for these posts.")
                            
                    except Exception as e:
                        print(f"[DataLoader] ERROR processing Telegram comments: {e}")
                        
                    if 'post_key' in df.columns:
                        df = df.drop(columns=['post_key'])
                else:
                    print("[DataLoader] WARNING: 'channel' or 'msgid' columns not found. Cannot fetch Telegram comments.")

            for col in ['sentiment', 'sentiment_std', 'emotion_label']:
                if col not in df.columns:
                    df[col] = None

            return df

        except Exception as e:
            print(f"[DataLoader] ERROR while loading data: {e}")
            raise
        
        finally:
            if client:
                client.close()
