import os
import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, round, when, date_format

# ─────── LOGGING SETUP ───────
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "analyze_seasonality.log"),
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)
log = logging.getLogger(__name__)
log.info("Starting seasonality analysis")

# ─────── SPARK SESSION ───────
spark = SparkSession.builder.appName(
    "Retail Demand Forecasting - Seasonality Analysis"
).getOrCreate()
log.info("Spark session started")

# ─────── LOAD FEATURE DATA ───────
features = spark.read.option("header", "true").csv("output/features", inferSchema=True)
log.info(f"Loaded features with {features.count()} rows")

# ─────── ADD QUARTER COLUMN ───────
features = features.withColumn(
    "Quarter",
    when(col("Month").between(1, 3), "Q1")
    .when(col("Month").between(4, 6), "Q2")
    .when(col("Month").between(7, 9), "Q3")
    .when(col("Month").between(10, 12), "Q4"),
)

# ─────── QUARTER-WISE AVERAGE SALES ───────
quarter_summary = (
    features.groupBy("Description", "Quarter")
    .agg(round(avg("TotalPrice"), 2).alias("avg_quarter_sales"))
    .orderBy("Description", "Quarter")
)

quarter_output_path = "output/seasonality/quarterly_summary"
quarter_summary.coalesce(1).write.option("header", "true").csv(
    quarter_output_path, mode="overwrite"
)
log.info(f"✅ Quarterly demand summary written to: {quarter_output_path}")

# ─────── NEW YEAR VS CHRISTMAS COMPARISON ───────
# Extract Year from InvoiceMonth (assumed to be in YYYY-MM format or similar)
features = features.withColumn("Year", date_format(col("InvoiceMonth"), "yyyy"))

seasonal_summary = (
    features.filter((col("Month") == 1) | (col("Month") == 12))
    .groupBy("Description", "Year", "Month")
    .agg(round(avg("TotalPrice"), 2).alias("avg_monthly_sales"))
    .orderBy("Description", "Year", "Month")
)

nyc_output_path = "output/seasonality/newyear_vs_christmas"
seasonal_summary.coalesce(1).write.option("header", "true").csv(
    nyc_output_path, mode="overwrite"
)
log.info(f"✅ Year-wise New Year vs Christmas summary written to: {nyc_output_path}")
