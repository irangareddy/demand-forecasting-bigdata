import os
import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, round, date_format, to_date, concat_ws

concat_ws

# ─────── LOGGING SETUP ───────
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "analyze_holiday_impact.log"),
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)
log = logging.getLogger(__name__)
log.info("Starting holiday impact analysis")

# ─────── SPARK SESSION ───────
spark = SparkSession.builder.appName(
    "Retail Demand Forecasting - Holiday Impact Analysis"
).getOrCreate()
log.info("Spark session started")

# ─────── LOAD FEATURES ───────
features = spark.read.option("header", "true").csv("output/features", inferSchema=True)
features = features.withColumn("month_key", date_format(col("InvoiceMonth"), "yyyy-MM"))
row_count = features.count()
log.info(f"Loaded features with {row_count} rows")
features.select("InvoiceMonth", "month_key").show(5, truncate=False)

# ─────── LOAD AND PARSE HOLIDAYS ───────

# First read all columns (ensure "Year" and "Date" are split)
holidays = spark.read.option("header", "true").csv(
    "data/public_holidays_uk_2009_2011.csv", inferSchema=True
)

# Combine Year and Date → parse full date
holidays = holidays.withColumn(
    "ParsedDate", to_date(concat_ws("-", col("Year"), col("Date")), "yyyy-MMM d")
)

# Extract yyyy-MM
holidays = holidays.withColumn(
    "holiday_month", date_format(col("ParsedDate"), "yyyy-MM")
)

log.info(f"Loaded holidays with {holidays.count()} rows")
holidays.select("Date", "holiday_month", "Name").show(5, truncate=False)

# ─────── JOIN FEATURES WITH HOLIDAYS ───────
joined = features.join(
    holidays.select("holiday_month", "Name"),
    features.month_key == holidays.holiday_month,
    "left",
)
joined.cache()
log.info(f"Joined dataset has {joined.count()} rows")
joined.select("month_key", "Name", "TotalPrice").show(10, truncate=False)

# ─────── AGGREGATE IMPACT ───────
holiday_impact = (
    joined.groupBy("Name")
    .agg(round(avg("TotalPrice"), 2).alias("avg_sales"))
    .filter(col("Name").isNotNull())
    .orderBy(col("avg_sales").desc())
)

log.info(f"Non-null holiday matches: {holiday_impact.count()}")
holiday_impact.show(10, truncate=False)

# ─────── SAVE OUTPUT ───────
output_path = "output/seasonality/holiday_impact"
holiday_impact.coalesce(1).write.option("header", "true").csv(
    output_path, mode="overwrite"
)
log.info(f"✅ Holiday-wise sales impact written to: {output_path}")
