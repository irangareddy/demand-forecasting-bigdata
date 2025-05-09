# 🛍️ Retail Demand Forecasting using Spark + Streamlit

This project builds a scalable pipeline for **product-level demand forecasting** and **inventory recommendation** using PySpark, MLlib, and Streamlit. It processes over 1M transactions and reveals sales patterns around holidays, seasons, and product trends.

---

## 🧭 Project Narrative

> How can a retailer avoid stockouts and overstock — especially during seasonal spikes?

We explored 1M+ UK retail transactions across 2 years and built:

* 📈 A product-month demand forecasting model (Random Forest, RMSE \~\$421)
* 📦 A classifier that predicts whether to **Increase**, **Maintain**, or **Reduce** stock (92% accuracy)
* 🎯 Insights about **holiday impact** and **seasonal trends**

---

## 💡 Problem Statement

> Predict monthly sales and recommend inventory actions per product by learning from past sales, seasonality, and public holiday effects.

---

## 🔧 Tech Stack

* `PySpark` – Big data processing + MLlib modeling
* `Pandas` – Local EDA + metric summaries
* `Streamlit` – Dashboard UI for business users
* `Docker + Taskfile` – Reproducible build + run steps

---

## 📁 Folder Layout

```
data/                          → raw retail & holiday data  
scripts/                       → Spark jobs & analysis scripts  
output/                        → Model inputs, predictions, metrics  
streamlit_app.py               → Interactive forecast dashboard  
Taskfile.yml                   → Run pipeline with `task all:train`  
```

---

## 📈 Dashboard Sections (Narrative-Driven)

| Section                      | What It Answers                          |
| ---------------------------- | ---------------------------------------- |
| 🧭 Intro                     | Why forecasting matters                  |
| 🔮 Product Forecast Explorer | What future sales look like              |
| 🎁 Holiday Products          | What sells best during holidays          |
| 📆 Quarterly Trends          | When each product peaks                  |
| 🎄 Holiday Comparison        | Is New Year beating Christmas?           |
| 🏆 Holiday Impact            | Which holidays actually move revenue?    |
| 📦 Inventory Classifier      | What stock action to take (92% accuracy) |

---

## 🧪 Modeling Pipeline

1. **`spark_eda.py`** – Cleans raw transactions and monthly aggregates
2. **`spark_features.py`** – Adds lag, rolling avg, sales\_delta, holiday flags
3. **`train_model.py`** – Trains RandomForestRegressor (uses `TrainValidationSplit`)
4. **`train_inventory_model.py`** – Trains classifier on stock\_decision (accuracy \~92%)
5. **`analyze_*.py` scripts** – Seasonal + holiday trends

---

## 📊 Key Results

* 🔁 Regression RMSE: **\$421.12**, MAE: **\$174.21**
* 📦 Classifier Accuracy: **91.89%**
* 🏆 Spring Bank Holiday outperformed Christmas in revenue
* 🎄 January sales exceeded December in some years

---

## 🚀 Run the Full Pipeline

```bash
docker-compose build

# Feature generation & modeling
task spark:eda
task spark:features
task train:regression
task train:inventory

# Exploratory analysis
task analyze:holidays
task analyze:products
task analyze:seasonality

# Streamlit dashboard
task serve
```

---

## 📁 Sample Output

`model_metrics.json`:

```json
{
  "RMSE": 421.12,
  "MAE": 174.21,
  "Best Hyperparameters": {
    "numTrees": 10,
    "maxDepth": 10
  }
}
```

`model_metrics_inventory.json`:

```json
{
  "accuracy": 0.9189,
  "f1": 0.9176,
  "weightedPrecision": 0.9200,
  "weightedRecall": 0.9189
}
```

---

## 🧠 Lessons Learned

* Demand varies significantly by product and season — models need **context-aware features**
* Holidays like **Substitute Boxing Day** and **Spring Bank** matter more than expected
* Class imbalance and label encoding can silently impact classifiers — verify them visually
* Small MAE at the global level can hide **high variance at product level** → use product-specific MAE

---

## 📦 Future Enhancements

* Use **Word2Vec / BERT** embeddings for product names
* Extend pipeline for **weekly forecasts** or **real-time Kafka integration**
* Add **profit-aware recommendations** (margin × demand × inventory)