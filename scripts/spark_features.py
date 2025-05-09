import os
import logging
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import (
    col,
    lag,
    avg,
    month,
    year,
    date_format,
    to_date,
    concat_ws,
)

# ─────── LOGGING SETUP ───────
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "spark_features.log"),
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)
log = logging.getLogger(__name__)
log.info("Starting Spark feature engineering script")

# ─────── SPARK SESSION ───────
spark = SparkSession.builder.appName(
    "Retail Forecasting - Feature Engineering"
).getOrCreate()
log.info("Spark session started")

# ─────── LOAD AGGREGATED DATA ───────
input_path = "output/aggregated_sales_monthly"
df = spark.read.option("header", "true").csv(input_path, inferSchema=True)
log.info(f"Loaded aggregated data with {df.count()} rows")

# ─────── TIME FEATURES ───────
df = df.withColumn("Year", year("InvoiceMonth")).withColumn(
    "Month", month("InvoiceMonth")
)

# ─────── LAG + ROLLING FEATURES ───────
window_spec = Window.partitionBy("Description").orderBy("InvoiceMonth")
df = df.withColumn("prev_month_sales", lag("TotalPrice", 1).over(window_spec))
df = df.withColumn(
    "rolling_3m_avg", avg("TotalPrice").over(window_spec.rowsBetween(-2, 0))
)
df = df.dropna(subset=["prev_month_sales", "rolling_3m_avg"])

# ─────── LOAD HOLIDAY DATA ───────

# Load all columns from your CSV
holidays = spark.read.option("header", "true").csv(
    "data/public_holidays_uk_2009_2011.csv", inferSchema=True
)

# Combine Year + Date columns (e.g., 2010 + Dec 25)
holidays = holidays.withColumn("DateString", concat_ws(" ", col("Year"), col("Date")))
holidays = holidays.withColumn("DateParsed", to_date(col("DateString"), "yyyy MMM d"))

# Extract yyyy-MM
holidays = holidays.withColumn(
    "holiday_month", date_format(col("DateParsed"), "yyyy-MM")
)

# Log preview
holidays.select("Date", "DateParsed", "holiday_month", "Name").show(5, truncate=False)

holiday_months = holidays.select("holiday_month").distinct()

df = df.withColumn("sales_delta", col("TotalPrice") - col("rolling_3m_avg"))

# ─────── JOIN HOLIDAY MONTHS ───────
features = df.withColumn("InvoiceMonthYM", date_format(col("InvoiceMonth"), "yyyy-MM"))
features = (
    features.join(
        holiday_months, features.InvoiceMonthYM == holiday_months.holiday_month, "left"
    )
    .withColumn("is_holiday_month", col("holiday_month").isNotNull().cast("int"))
    .drop("holiday_month")
    .drop("InvoiceMonthYM")
)

# ─────── WRITE FEATURES ───────
output_path = "output/features"
features.coalesce(1).write.option("header", "true").csv(output_path, mode="overwrite")
log.info(f"✅ Feature set (with holiday flags) written to: {output_path}")

from pyspark.sql.functions import when

# ─────── STOCK DECISION LABEL ───────
features = features.withColumn("sales_delta", col("TotalPrice") - col("rolling_3m_avg"))
features = features.withColumn(
    "stock_decision",
    when(col("TotalPrice") > col("rolling_3m_avg") * 1.2, "Increase")
    .when(col("TotalPrice") < col("rolling_3m_avg") * 0.8, "Reduce")
    .otherwise("Maintain"),
)

# ─────── WRITE EXTENDED FEATURES ───────
output_path = "output/features_with_labels"
features.coalesce(1).write.option("header", "true").csv(output_path, mode="overwrite")
log.info(f"✅ Feature set with labels written to: {output_path}")
