import os
import logging
from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, StopWordsRemover
from pyspark.ml.clustering import KMeans

# ─────── LOGGING SETUP ───────
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "spark_nlp_clustering.log"),
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)
log = logging.getLogger(__name__)
log.info("Starting NLP clustering on product descriptions")

# ─────── SPARK SESSION ───────
spark = SparkSession.builder.appName("NLP Product Clustering").getOrCreate()
log.info("Spark session started")

# ─────── LOAD DATA ───────
df = spark.read.option("header", "true").csv(
    "output/features_with_labels", inferSchema=True
)
log.info(f"Loaded {df.count()} rows")

# ─────── NLP PIPELINE ───────
tokenizer = Tokenizer(inputCol="Description", outputCol="words")
wordsData = tokenizer.transform(df)

remover = StopWordsRemover(inputCol="words", outputCol="filtered")
filteredData = remover.transform(wordsData)

from pyspark.ml.feature import CountVectorizer

# Vectorize tokens using CountVectorizer
vectorizer = CountVectorizer(
    inputCol="filtered", outputCol="features", vocabSize=1000, minDF=5
)
vector_model = vectorizer.fit(filteredData)
rescaledData = vector_model.transform(filteredData)


# ─────── KMEANS CLUSTERING ───────
kmeans = KMeans(k=10, seed=42)
model = kmeans.fit(rescaledData)
clustered = model.transform(rescaledData).withColumnRenamed(
    "prediction", "NLP_Category"
)

# ─────── SAVE RESULT ───────
clustered.select("Description", "NLP_Category").dropDuplicates().coalesce(
    1
).write.option("header", "true").csv("output/nlp_clusters_pyspark", mode="overwrite")

log.info("✅ NLP category clusters saved to: output/nlp_clusters_pyspark")
