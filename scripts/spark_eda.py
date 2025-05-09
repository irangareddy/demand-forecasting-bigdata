import os
import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, sum as spark_sum, expr
from pyspark.sql.types import FloatType

# ─────── LOGGING SETUP ───────
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "spark_eda.log"),
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)
log = logging.getLogger(__name__)

log.info("Starting Spark EDA script")

# ─────── SPARK SESSION ───────
spark = SparkSession.builder.appName("Retail Demand Forecasting - EDA").getOrCreate()
log.info("Spark session started")

# ─────── LOAD DATA ───────
try:
    df = (
        spark.read.option("header", "true")
        .option("inferSchema", "true")
        .csv("data/online_retail_raw.csv")
    )
    log.info(f"Loaded dataset with {df.count()} rows")
except Exception as e:
    log.error("Failed to load data", exc_info=True)
    raise e

# ─────── CLEANING ───────
log.info("Starting data cleaning")
df_clean = (
    df.filter(~col("Invoice").startswith("C"))
    .filter((col("Quantity") > 0) & (col("Price") > 0))
    .filter(col("Customer ID").isNotNull())
    .withColumn("TotalPrice", col("Quantity") * col("Price"))
    .withColumn("InvoiceDate", to_date("InvoiceDate"))
    .withColumn(
        "InvoiceMonth", expr("make_date(year(InvoiceDate), month(InvoiceDate), 1)")
    )
)
log.info("Cleaning complete")

# ─────── TOP PRODUCTS ───────
log.info("Selecting top 10 products by revenue")
top_products = (
    df_clean.groupBy("Description")
    .agg(spark_sum("TotalPrice").alias("Revenue"))
    .orderBy(col("Revenue").desc())
    .limit(1000)
    .select("Description")
)

df_top = df_clean.join(top_products, on="Description", how="inner")

# ─────── AGGREGATION ───────
log.info("Aggregating monthly sales per product")
monthly_sales = (
    df_top.groupBy("Description", "InvoiceMonth")
    .agg(spark_sum("TotalPrice").cast(FloatType()).alias("TotalPrice"))
    .orderBy("Description", "InvoiceMonth")
)

# ─────── OUTPUT ───────
output_path = "output/aggregated_sales_monthly"
try:
    monthly_sales.coalesce(1).write.option("header", "true").csv(
        output_path, mode="overwrite"
    )
    log.info(f"Saved output to {output_path}")
except Exception as e:
    log.error("Failed to write output", exc_info=True)
    raise e

log.info("✅ Spark EDA script completed successfully")
