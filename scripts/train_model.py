import os
import logging
import json
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder

# ─────── LOGGING SETUP ───────
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "train_model.log"),
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)
log = logging.getLogger(__name__)
log.info("Starting training script")

# ─────── SPARK SESSION ───────
spark = SparkSession.builder.appName(
    "Retail Demand Forecasting - Train Model"
).getOrCreate()

log.info("Spark session started")

# ─────── LOAD FEATURE DATA ───────
df = spark.read.option("header", "true").csv("output/features", inferSchema=True)
log.info(f"Loaded features with {df.count()} rows")

# ─────── ENCODE + VECTORIZE ───────
indexer = StringIndexer(inputCol="Description", outputCol="DescriptionIndex")
df = indexer.fit(df).transform(df)
feature_cols = [
    "prev_month_sales",  # retains lag signal
    "sales_delta",  # better than Month
    "rolling_3m_avg",  # strongest signal
    "is_holiday_month",  # keep as binary context
    "DescriptionIndex",  # keep for product-specific bias
]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df = assembler.transform(df)

# ─────── SPLIT DATA ───────
train_data, val_data, test_data = df.randomSplit([0.7, 0.15, 0.15], seed=42)
log.info(
    f"Train: {train_data.count()}, Val: {val_data.count()}, Test: {test_data.count()}"
)

# ─────── SETUP MODEL & GRID ───────
rf = RandomForestRegressor(featuresCol="features", labelCol="TotalPrice", maxBins=2048)

paramGrid = (
    ParamGridBuilder()
    .addGrid(rf.numTrees, [10, 20])
    .addGrid(rf.maxDepth, [5, 10])
    .build()
)

evaluator = RegressionEvaluator(
    labelCol="TotalPrice", predictionCol="prediction", metricName="rmse"
)

tvs = TrainValidationSplit(
    estimator=rf, estimatorParamMaps=paramGrid, evaluator=evaluator, trainRatio=0.8
)

# ─────── TRAIN ───────
model = tvs.fit(train_data)
log.info("Model training and validation complete")
# Log best hyperparameters
best_model = model.bestModel
log.info(f"Best numTrees: {best_model.getNumTrees}")
log.info(f"Best maxDepth: {best_model.getOrDefault('maxDepth')}")

# ─────── TEST EVALUATION ───────
predictions = model.transform(test_data)
rmse = evaluator.evaluate(predictions)
mae = RegressionEvaluator(
    labelCol="TotalPrice", predictionCol="prediction", metricName="mae"
).evaluate(predictions)

log.info(f"Test RMSE: {rmse:.2f}")
log.info(f"Test MAE: {mae:.2f}")

# ─────── SAVE PREDICTIONS ───────
predictions.select("Description", "InvoiceMonth", "TotalPrice", "prediction").coalesce(
    1
).write.option("header", "true").csv("output/predictions", mode="overwrite")

log.info("Predictions written to output/predictions/")


# ─────── METRICS EXPORT ───────
metrics = {
    "RMSE": round(rmse, 2),
    "MAE": round(mae, 2),
    "Best Hyperparameters": {
        "numTrees": best_model.getNumTrees,
        "maxDepth": best_model.getOrDefault("maxDepth"),
    },
}

output_json_path = "output/predictions/model_metrics.json"
with open(output_json_path, "w") as f:
    json.dump(metrics, f, indent=4)

log.info(f"📊 Exported model metrics to: {output_json_path}")
