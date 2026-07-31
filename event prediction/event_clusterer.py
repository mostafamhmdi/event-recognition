
import pandas as pd
import networkx as nx


class EventClusterer:
    def __init__(
        self,
        text_col: str = 'txtContent',
        future_date_col: str = 'future_date',
        loc_col: str = 'location_entities',
        event_col: str = 'event_entities',
    ):
        self.text_col = text_col
        self.future_date_col = future_date_col
        self.loc_col = loc_col
        self.event_col = event_col

    @staticmethod
    def _to_entity_set(raw_value) -> set:
        """تبدیل رشته‌ی 'A، B، C' یا 'A, B, C' به یک Set برای مقایسه‌ی راحت."""
        if pd.isna(raw_value) or str(raw_value).strip() == '' or str(raw_value) == 'None':
            return set()
        s = str(raw_value).replace('،', ',')
        return set(x.strip() for x in s.split(',') if x.strip())

    @staticmethod
    def _find_components(nodes_data: dict, edge_predicate) -> list:
        """
        روی مجموعه‌ای از گره‌ها (idx -> {'locations': set, 'events': set}) گراف می‌سازد
        و بر اساس edge_predicate(data1, data2) یال می‌کشد. فقط اجزای متصل با حداقل
        ۲ عضو را برمی‌گرداند (گره‌های تک‌افتاده خوشه محسوب نمی‌شوند).
        """
        G = nx.Graph()
        G.add_nodes_from(nodes_data.keys())

        items = list(nodes_data.items())
        for i in range(len(items)):
            idx1, data1 = items[i]
            for j in range(i + 1, len(items)):
                idx2, data2 = items[j]
                if edge_predicate(data1, data2):
                    G.add_edge(idx1, idx2)

        return [comp for comp in nx.connected_components(G) if len(comp) > 1]

    def cluster_Events(self, df: pd.DataFrame) -> pd.DataFrame:
        print("\n🔍 شروع خوشه‌بندی سه‌سطحی (تاریخ مشترک + مکان/رویداد مشترک)...")
        working_df = df.copy()
        working_df['cluster_id'] = None
        working_df['cluster_type'] = None  # 'location_and_event' | 'location_only' | 'event_only'

        # شرط پایه: فقط تاریخ آینده معتبر لازم است. مکان و رویداد اختیاری‌اند
        # چون ممکن است فقط یکی از این دو برای یک پیام استخراج شده باشد.
        has_date = (
            working_df[self.future_date_col].notna()
            & (working_df[self.future_date_col] != 'None')
        )
        has_loc = working_df[self.loc_col].notna() & (working_df[self.loc_col] != 'None')
        has_event = working_df[self.event_col].notna() & (working_df[self.event_col] != 'None')

        mask = has_date & (has_loc | has_event)
        filtered_df = working_df[mask].copy()

        if filtered_df.empty:
            print("⚠️ هیچ داده‌ای برای خوشه‌بندی یافت نشد (رکوردی فیلترها را پاس نکرد).")
            return working_df

        print(f"تعداد رکوردهای معتبر برای خوشه‌بندی (دارای تاریخ و حداقل مکان یا رویداد): {len(filtered_df)}")

        grouped_by_future = filtered_df.groupby(self.future_date_col)
        cluster_global_id = 1

        for future_date, group_df in grouped_by_future:
            safe_date = str(future_date).replace('/', '')

            # داده‌ی هر گره را یک‌بار می‌سازیم تا در هر سه سطح دوباره پارس نشود
            nodes_data = {
                idx: {
                    'locations': self._to_entity_set(row[self.loc_col]),
                    'events': self._to_entity_set(row[self.event_col]),
                }
                for idx, row in group_df.iterrows()
            }

            assigned = set()

            # ------------------------------------------------------------------
            # سطح ۱ (قوی‌ترین سیگنال): اشتراک هم در مکان و هم در رویداد
            # ------------------------------------------------------------------
            remaining = {idx: d for idx, d in nodes_data.items() if idx not in assigned}
            tier1 = self._find_components(
                remaining,
                lambda d1, d2: bool(d1['locations'] & d2['locations']) and bool(d1['events'] & d2['events']),
            )
            for comp in tier1:
                comp = list(comp)
                cluster_name = f"Cluster_{cluster_global_id}_{safe_date}_LocEvent"
                working_df.loc[comp, 'cluster_id'] = cluster_name
                working_df.loc[comp, 'cluster_type'] = 'location_and_event'
                print(f"  * [سطح۱ مکان+رویداد] تاریخ: {future_date} | تعداد: {len(comp)} | شناسه: {cluster_name}")
                cluster_global_id += 1
                assigned.update(comp)

            # ------------------------------------------------------------------
            # سطح ۲: فقط اشتراک مکان، در میان گره‌هایی که هنوز خوشه نگرفته‌اند
            # ------------------------------------------------------------------
            remaining = {idx: d for idx, d in nodes_data.items() if idx not in assigned}
            tier2 = self._find_components(
                remaining,
                lambda d1, d2: bool(d1['locations'] & d2['locations']),
            )
            for comp in tier2:
                comp = list(comp)
                cluster_name = f"Cluster_{cluster_global_id}_{safe_date}_Loc"
                working_df.loc[comp, 'cluster_id'] = cluster_name
                working_df.loc[comp, 'cluster_type'] = 'location_only'
                print(f"  * [سطح۲ فقط مکان] تاریخ: {future_date} | تعداد: {len(comp)} | شناسه: {cluster_name}")
                cluster_global_id += 1
                assigned.update(comp)

            # ------------------------------------------------------------------
            # سطح ۳: فقط اشتراک رویداد، در میان گره‌هایی که هنوز خوشه نگرفته‌اند
            # ------------------------------------------------------------------
            remaining = {idx: d for idx, d in nodes_data.items() if idx not in assigned}
            tier3 = self._find_components(
                remaining,
                lambda d1, d2: bool(d1['events'] & d2['events']),
            )
            for comp in tier3:
                comp = list(comp)
                cluster_name = f"Cluster_{cluster_global_id}_{safe_date}_Event"
                working_df.loc[comp, 'cluster_id'] = cluster_name
                working_df.loc[comp, 'cluster_type'] = 'event_only'
                print(f"  * [سطح۳ فقط رویداد] تاریخ: {future_date} | تعداد: {len(comp)} | شناسه: {cluster_name}")
                cluster_global_id += 1
                assigned.update(comp)

            # ------------------------------------------------------------------
            # سطح ۴ (fallback): گره‌هایی که در هیچ‌کدام از سطوح بالا جفت پیدا نکردند
            # نباید بی‌خوشه (None) بمانند — چون خودشان حداقل دو موجودیت معتبر
            # (تاریخ + مکان یا رویداد) دارند و باید به‌عنوان یک خوشه‌ی تک‌عضوی
            # وارد مرحله‌ی post_cluster شوند تا آنجا بر اساس شباهت معنایی به
            # خوشه‌ی مناسب (مثلاً همان A/B) ملحق شوند.
            # نوع خوشه بر اساس موجودیت‌های خودِ همان گره تعیین می‌شود، نه اشتراک با کسی.
            # ------------------------------------------------------------------
            remaining = {idx: d for idx, d in nodes_data.items() if idx not in assigned}
            for idx, data in remaining.items():
                has_loc_here = bool(data['locations'])
                has_event_here = bool(data['events'])

                if has_loc_here and has_event_here:
                    suffix, c_type = "LocEvent", "location_and_event"
                elif has_loc_here:
                    suffix, c_type = "Loc", "location_only"
                else:  # فقط رویداد (چون mask تضمین کرده حداقل یکی هست)
                    suffix, c_type = "Event", "event_only"

                cluster_name = f"Cluster_{cluster_global_id}_{safe_date}_{suffix}_Solo"
                working_df.loc[[idx], 'cluster_id'] = cluster_name
                working_df.loc[[idx], 'cluster_type'] = c_type
                print(f"  * [سطح۴ تک‌عضوی] تاریخ: {future_date} | نوع: {c_type} | شناسه: {cluster_name}")
                cluster_global_id += 1
                assigned.add(idx)

        print(f"✅ خوشه‌بندی پایان یافت. مجموعاً {cluster_global_id - 1} خوشه استخراج شد.")
        return working_df


# تابع واسط برای استفاده در main
def process_clusters(df: pd.DataFrame, text_col: str = 'txtContent') -> pd.DataFrame:
    clusterer = EventClusterer(text_col=text_col)
    return clusterer.cluster_Events(df)