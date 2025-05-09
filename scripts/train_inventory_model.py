import os
import json
import logging
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler, IndexToString
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder

# ─────── LOGGING SETUP ───────
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "train_inventory_model.log"),
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)
log = logging.getLogger(__name__)
log.info("Starting inventory classifier training")

# ─────── SPARK SESSION ───────
spark = SparkSession.builder.appName("Inventory Planning Classifier").getOrCreate()
log.info("Spark session started")

# ─────── LOAD DATA ───────
df = spark.read.option("header", "true").csv(
    "output/features_with_labels", inferSchema=True
)
log.info(f"Loaded {df.count()} rows")

# ─────── ENCODE LABELS ───────
label_indexer = StringIndexer(inputCol="stock_decision", outputCol="label")
label_model = label_indexer.fit(df)
df = label_model.transform(df)

# ─────── ENCODE FEATURES ───────
desc_indexer = StringIndexer(inputCol="Description", outputCol="DescriptionIndex")
df = desc_indexer.fit(df).transform(df)

feature_cols = [
    "prev_month_sales",
    "sales_delta",  # New, medium correlation
    "is_holiday_month",  # Categorical binary
    "DescriptionIndex",  # Or ProductCategoryIndex
]


assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df = assembler.transform(df)

# ─────── SPLIT DATA ───────
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)
log.info(f"Train: {train_data.count()}, Test: {test_data.count()}")

# ─────── MODEL TRAINING ───────
rf = RandomForestClassifier(labelCol="label", featuresCol="features", maxBins=2048)
paramGrid = (
    ParamGridBuilder()
    .addGrid(rf.numTrees, [10, 20])
    .addGrid(rf.maxDepth, [5, 10])
    .build()
)

evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy"
)
tvs = TrainValidationSplit(
    estimator=rf, estimatorParamMaps=paramGrid, evaluator=evaluator, trainRatio=0.8
)

model = tvs.fit(train_data)
log.info("Model training and validation complete")

# ─────── EVALUATE ───────
predictions = model.transform(test_data)

metrics = {
    "accuracy": evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"}),
    "f1": MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="f1"
    ).evaluate(predictions),
    "weightedPrecision": MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="weightedPrecision"
    ).evaluate(predictions),
    "weightedRecall": MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="weightedRecall"
    ).evaluate(predictions),
}

for metric, value in metrics.items():
    log.info(f"{metric.capitalize()}: {value:.4f}")

# ─────── PREDICTION → LABEL STRING ───────
label_converter = IndexToString(
    inputCol="prediction", outputCol="predicted_label", labels=label_model.labels
)
predictions = label_converter.transform(predictions)

# ─────── EXPORT PREDICTIONS ───────
predictions.select(
    "Description", "InvoiceMonth", "stock_decision", "predicted_label"
).coalesce(1).write.option("header", "true").csv(
    "output/predictions_inventory", mode="overwrite"
)

log.info("✅ Inventory prediction output written to: output/predictions_inventory/")

# ─────── EXPORT METRICS JSON ───────
metrics_output_path = "output/predictions_inventory/model_metrics_inventory.json"
with open(metrics_output_path, "w") as f:
    json.dump({k: round(v, 4) for k, v in metrics.items()}, f, indent=4)

log.info(f"📊 Exported inventory classifier metrics to: {metrics_output_path}")
