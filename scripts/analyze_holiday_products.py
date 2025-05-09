import os
import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, date_format, to_date, sum as spark_sum, concat_ws

# ─────── LOGGING SETUP ───────
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "analyze_holiday_products.log"),
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)
log = logging.getLogger(__name__)
log.info("Starting holiday product analysis")

# ─────── SPARK SESSION ───────
spark = SparkSession.builder.appName(
    "Retail Demand Forecasting - Holiday Product Analysis"
).getOrCreate()
log.info("Spark session started")

# ─────── LOAD RAW TRANSACTIONS ───────
df = spark.read.option("header", "true").csv(
    "data/online_retail_raw.csv", inferSchema=True
)
df = df.withColumn("InvoiceDate", to_date("InvoiceDate"))
df = df.withColumn("InvoiceMonth", date_format(col("InvoiceDate"), "yyyy-MM"))
df = df.withColumn("TotalPrice", col("Quantity") * col("Price"))

log.info(f"Loaded {df.count()} transaction rows")

from pyspark.sql.functions import concat_ws

# ─────── LOAD & FORMAT HOLIDAY DATA ───────
holidays = spark.read.option("header", "true").csv(
    "data/public_holidays_uk_2009_2011.csv", inferSchema=True
)

# Reconstruct full date from year + month/day text
holidays = holidays.withColumn("DateString", concat_ws(" ", col("Year"), col("Date")))
holidays = holidays.withColumn("DateParsed", to_date(col("DateString"), "yyyy MMM d"))
holidays = holidays.withColumn(
    "holiday_month", date_format(col("DateParsed"), "yyyy-MM")
)

# Select distinct months (filtering any nulls)
holiday_months = (
    holidays.select("holiday_month").distinct().filter(col("holiday_month").isNotNull())
)

log.info(f"Detected {holiday_months.count()} unique holiday months")


# ─────── FILTER FOR HOLIDAY MONTHS ───────
df_holiday = df.join(
    holiday_months, df.InvoiceMonth == holiday_months.holiday_month, "inner"
)

log.info(f"Transactions in holiday months: {df_holiday.count()}")

# ─────── TOP PRODUCTS IN HOLIDAY MONTHS ───────
top_holiday_products = (
    df_holiday.groupBy("Description")
    .agg(
        spark_sum("Quantity").alias("total_quantity"),
        spark_sum("TotalPrice").alias("total_sales"),
    )
    .orderBy(col("total_sales").desc())
)

output_path = "output/seasonality/top_holiday_products"
top_holiday_products.coalesce(1).write.option("header", "true").option(
    "quote", '"'
).option("escape", '"').option("multiLine", False).csv(output_path, mode="overwrite")


log.info(f"✅ Top holiday products written to: {output_path}")
