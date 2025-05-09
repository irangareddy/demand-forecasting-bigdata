# import pandas as pd
# import os
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.cluster import KMeans
#
# # ─────── SETTINGS ───────
# INPUT_PATH = "/Users/rangareddy/Development/OSS/demand-forecasting-bigdata/output/features_with_labels/part-00000-17c9ec0b-e308-4904-9546-991fcd22f5be-c000.csv"  # update filename as needed
# OUTPUT_PATH = "output/features_with_nlp_categories.csv"
# N_CLUSTERS = 10
#
# # ─────── LOAD DATA ───────
# df = pd.read_csv(INPUT_PATH)
# df["Description"] = df["Description"].fillna("").str.lower()
#
# # ─────── TF-IDF VECTORIZATION ───────
# vectorizer = TfidfVectorizer(
#     stop_words="english",
#     max_features=100,
#     ngram_range=(1, 2)
# )
# X = vectorizer.fit_transform(df["Description"])
#
# # ─────── K-MEANS CLUSTERING ───────
# kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42)
# df["NLP_Category"] = kmeans.fit_predict(X)
#
# # ─────── TOP TERMS PER CLUSTER ───────
# print("\n📊 Top Terms per Cluster:")
# terms = vectorizer.get_feature_names_out()
# order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
# for i in range(N_CLUSTERS):
#     top_words = [terms[ind] for ind in order_centroids[i, :5]]
#     print(f"Cluster {i}: {', '.join(top_words)}")
#
# # ─────── SAVE OUTPUT ───────
# os.makedirs("output", exist_ok=True)
# df.to_csv(OUTPUT_PATH, index=False)
# print(f"\n✅ NLP-based categories saved to: {OUTPUT_PATH}")

import pandas as pd
import os

# Auto-detect the file
file_path = "/Users/rangareddy/Development/OSS/demand-forecasting-bigdata/output/features_with_labels/part-00000-17c9ec0b-e308-4904-9546-991fcd22f5be-c000.csv"

# Load the data
df = pd.read_csv(file_path)

# Compute correlations
numeric_cols = [
    "TotalPrice",
    "prev_month_sales",
    "rolling_3m_avg",
    "is_holiday_month",
    "Month",
    "sales_delta",
]
corr_matrix = df[numeric_cols].corr()

# Save to text file
os.makedirs("output/diagnostics", exist_ok=True)
correlation_path = "output/diagnostics/correlation_matrix.txt"
with open(correlation_path, "w") as f:
    f.write(corr_matrix.round(2).to_string())

correlation_path
