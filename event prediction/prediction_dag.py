import pendulum
from airflow import DAG
from airflow.operators.bash import BashOperator

# تابع کمکی برای تبدیل تاریخ میلادی به تاریخ شمسی
def get_jalali_ds(logical_date_str):
    try:
        import jdatetime
        g_date = pendulum.parse(logical_date_str).date()
        j_date = jdatetime.date.fromgregorian(date=g_date)
        return j_date.strftime("%Y-%m-%d")
    except ImportError:
        return logical_date_str

with DAG(
    dag_id="social_media_processing_pipeline",
    start_date=pendulum.datetime(2026, 8, 24, tz="Asia/Tehran"),
    schedule="0 0 * * *", 
    catchup=False,
    max_active_runs=1,
    tags=["ml", "twitter", "telegram", "daily"],
    user_defined_macros={"jalali_ds": get_jalali_ds}
) as dag:

    # تعریف پیکربندی منابع مختلف (توییتر، تلگرام و هر منبع دیگری که در آینده اضافه شود)
    sources = [
        {
            "name": "twitter",
            "db_name": "x",
            "table_name": "tweets_2"
        },
        {
            "name": "telegram",
            "db_name": "telegram",        
            "table_name": "posts" 
        }
    ]

    # ساخت داینامیک تسک‌ها با استفاده از یک حلقه for
    for source in sources:
        BashOperator(
            task_id=f"run_{source['name']}_model",
            bash_command=f'python3 /app/main.py --db-name "{source["db_name"]}" --table-name "{source["table_name"]}" --start-date "{{{{ jalali_ds(ds) }}}}" --end-date "{{{{ jalali_ds(ds) }}}}"',
            retries=2,
            retry_delay=pendulum.duration(minutes=5),
        )