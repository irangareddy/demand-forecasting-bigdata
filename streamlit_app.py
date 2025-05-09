import streamlit as st
import pandas as pd
import json
import os
import glob
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime


# ─────── Helper to load part-*.csv ───────
def load_spark_output_csv(folder_path: str) -> pd.DataFrame:
    part_files = glob.glob(os.path.join(folder_path, "part-*.csv"))
    if not part_files:
        st.warning(f"No CSV file found in: {folder_path}")
        return pd.DataFrame()
    return pd.read_csv(part_files[0], engine="python", on_bad_lines="skip")


# ─────── Load model metrics ───────
def load_metrics(path: str):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


# ─────── Data Loading ───────
df_preds = load_spark_output_csv("output/predictions")
df_features = load_spark_output_csv("output/features")
df_holiday_impact = load_spark_output_csv("output/seasonality/holiday_impact")
df_newyear_vs_christmas = load_spark_output_csv(
    "output/seasonality/newyear_vs_christmas"
)
df_quarterly = load_spark_output_csv("output/seasonality/quarterly_summary")
df_top_holiday_products = load_spark_output_csv(
    "output/seasonality/top_holiday_products"
)
metrics = load_metrics("output/predictions/model_metrics.json")

# ─────── Streamlit UI ───────
st.set_page_config(
    layout="wide", page_title="Retail Demand Forecasting", page_icon="📈"
)

# Custom CSS for better styling
st.markdown(
    """
    <style>
    .main {
        padding: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .stMetric:hover {
        background-color: #e6e9ef;
    }
    </style>
""",
    unsafe_allow_html=True,
)

st.title("📈 Retail Demand Forecasting Dashboard")

st.markdown("""
## 🧭 Introduction: Making Demand Predictable

> "How can a retailer plan smarter stock decisions when demand is volatile, seasonal, and holiday-driven?"

This dashboard showcases:
- 🔍 Insights into product- and holiday-driven demand cycles
- ⚙️ Predictive models for sales and inventory classification
- 📦 Stocking recommendations based on actual purchase behavior

Explore how we transformed 1M+ raw retail records into an actionable forecasting system using PySpark and MLlib.
""")

# ─────── Section 1: Forecasting Model Summary ───────
st.header("📊 How Accurate Is the Forecasting Model?")
st.caption(
    "This section summarizes the overall performance of our regression model across the test set. Lower values indicate better predictive accuracy."
)

if metrics:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "📉 RMSE (Root Mean Square Error)",
            f"${metrics['RMSE']:,.2f}",
            help="Lower is better – penalizes large errors",
        )
    with col2:
        st.metric(
            "📏 MAE (Mean Absolute Error)",
            f"${metrics['MAE']:,.2f}",
            help="Average absolute difference between predicted and actual sales",
        )
    with col3:
        st.metric(
            "🧠 Model Complexity",
            f"{metrics['Best Hyperparameters']['numTrees']} trees",
            help=f"Max depth: {metrics['Best Hyperparameters']['maxDepth']}",
        )
else:
    st.warning("No regression metrics found. Please retrain the model.")


# ─────── Section 2: Product Forecast Explorer ───────
st.header("🔮 What Do Future Sales Look Like?")
st.caption(
    "This section lets you explore product-level sales forecasts month by month. The model uses past sales, seasonality, and holiday impact to predict demand."
)

if not df_preds.empty:
    col1, col2 = st.columns([1, 2])
    with col1:
        product = st.selectbox(
            "🛒 Select a product",
            df_preds["Description"].unique(),
            help="Choose a product to view actual vs predicted sales",
        )
        date_range = st.date_input(
            "📆 Select date range",
            value=(
                pd.to_datetime(df_preds["InvoiceMonth"].min()),
                pd.to_datetime(df_preds["InvoiceMonth"].max()),
            ),
            help="Filter to a specific time period",
        )

    filtered_preds = df_preds[df_preds["Description"] == product]
    filtered_preds["InvoiceMonth"] = pd.to_datetime(filtered_preds["InvoiceMonth"])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=filtered_preds["InvoiceMonth"],
            y=filtered_preds["TotalPrice"],
            name="Actual Sales",
            line=dict(color="#1f77b4"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=filtered_preds["InvoiceMonth"],
            y=filtered_preds["prediction"],
            name="Predicted Sales",
            line=dict(color="#ff7f0e", dash="dash"),
        )
    )

    fig.update_layout(
        title=f"Sales Forecast for '{product}'",
        xaxis_title="Month",
        yaxis_title="Sales Amount ($)",
        hovermode="x unified",
        showlegend=True,
        height=500,
    )

    st.plotly_chart(fig, use_container_width=True)

    # Add prediction accuracy metrics for selected product
    mae = abs(filtered_preds["TotalPrice"] - filtered_preds["prediction"]).mean()
    st.metric(
        "📏 Product-specific MAE",
        f"${mae:,.2f}",
        help="Mean Absolute Error for this specific product's forecast",
    )
else:
    st.info("Prediction data not available.")
# ─────── Section 3: Holiday Product Demand ───────
st.header("🎁 Which Products Peak During the Holidays?")
st.caption(
    "By summing sales during public holiday months, we reveal the most in-demand products — crucial for Q4 inventory and promotion planning."
)

# Clean and prepare
df_top_holiday_products.columns = df_top_holiday_products.columns.str.strip()

st.write(df_top_holiday_products.head())

if "total_sales" in df_top_holiday_products.columns:
    # Ensure numeric
    df_top_holiday_products["total_sales"] = pd.to_numeric(
        df_top_holiday_products["total_sales"], errors="coerce"
    )
    df_top_holiday_products = df_top_holiday_products.dropna(subset=["total_sales"])

    # Sort by sales descending
    df_top_holiday_products = df_top_holiday_products.sort_values(
        by="total_sales", ascending=False
    )

# Check final condition
expected_cols = {"Description", "total_sales"}
if (
    expected_cols.issubset(df_top_holiday_products.columns)
    and not df_top_holiday_products.empty
):
    col1, col2 = st.columns([1, 2])
    with col1:
        top_n = st.slider(
            "Top N Products",
            5,
            20,
            10,
            help="Adjust to display more or fewer top-selling products",
        )

    top_products = df_top_holiday_products.head(top_n)

    fig = px.bar(
        top_products,
        x="Description",
        y="total_sales",
        title="Top Selling Products During Holiday Season",
        labels={"total_sales": "Total Sales ($)", "Description": "Product"},
        color="total_sales",
        color_continuous_scale="Sunsetdark",
    )
    fig.update_layout(height=500, xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Data is filtered to sales that occurred in months containing public holidays."
    )
else:
    st.info("No holiday praoduct data available.")


# ─────── Section 4: Quarterly Demand ───────
st.header("📆 When Does Each Product Sell the Most?")
st.caption(
    "This heatmap shows average sales per quarter for each product. Use it to identify seasonal patterns and inform when to stock or promote specific items."
)

if not df_quarterly.empty:
    df_quarterly.columns = df_quarterly.columns.str.strip()
    df_quarterly["avg_quarter_sales"] = pd.to_numeric(
        df_quarterly["avg_quarter_sales"], errors="coerce"
    )

    # Optional: Filter to top N products by total sales across quarters
    top_products = (
        df_quarterly.groupby("Description")["avg_quarter_sales"]
        .sum()
        .sort_values(ascending=False)
        .head(12)  # Adjust N here
        .index
    )
    df_filtered = df_quarterly[df_quarterly["Description"].isin(top_products)]

    # Pivot for heatmap
    quarterly_pivot = df_filtered.pivot(
        index="Quarter", columns="Description", values="avg_quarter_sales"
    )

    fig = px.imshow(
        quarterly_pivot,
        title="📊 Average Quarterly Sales by Product",
        labels=dict(x="Product", y="Quarter", color="Average Sales"),
        color_continuous_scale="Viridis",
    )
    fig.update_layout(height=600, xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Quarterly data not available.")

# ─────── Section 5: Holiday Demand Comparison ───────
st.header("🎄 Is New Year the New Christmas?")
st.caption(
    "This comparison shows how product sales differ between December (Christmas) and January (New Year) across years. Some products surprisingly peak after the holidays."
)

if not df_newyear_vs_christmas.empty:
    df_newyear_vs_christmas.columns = df_newyear_vs_christmas.columns.str.strip()
    df_newyear_vs_christmas["Year"] = df_newyear_vs_christmas["Year"].astype(str)

    if (
        "Description" in df_newyear_vs_christmas.columns
        and df_newyear_vs_christmas["Description"].nunique() > 1
    ):
        product = st.selectbox(
            "🛍️ Select a product to compare",
            df_newyear_vs_christmas["Description"].unique(),
            key="nyc_product",
        )
        df_plot = df_newyear_vs_christmas[
            df_newyear_vs_christmas["Description"] == product
        ]
    else:
        df_plot = df_newyear_vs_christmas.copy()

    # Label months
    month_map = {1: "New Year (Jan)", 12: "Christmas (Dec)"}
    df_plot["MonthLabel"] = df_plot["Month"].map(month_map)

    # Bar chart
    fig = px.bar(
        df_plot,
        x="Year",
        y="avg_monthly_sales",
        color="MonthLabel",
        barmode="group",
        text="avg_monthly_sales",
        labels={
            "avg_monthly_sales": "Average Monthly Sales ($)",
            "Year": "Year",
            "MonthLabel": "Holiday",
        },
        color_discrete_map={"New Year (Jan)": "#1f77b4", "Christmas (Dec)": "#ff7f0e"},
        title="📊 Year-by-Year Holiday Sales Comparison",
    )
    fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
    fig.update_layout(yaxis=dict(tickformat=",d"), height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Some years show stronger demand in January than December — highlighting the importance of post-holiday planning."
    )
else:
    st.info("Holiday comparison data not available.")

# ─────── Section 6: Holiday Impact Analysis ───────
st.header("🏆 Which Holidays Actually Drive Revenue?")
st.caption(
    "Not all holidays are equal. This analysis reveals which UK public holidays correspond to the highest average sales — helping identify which ones truly drive demand."
)

if not df_holiday_impact.empty:
    df_holiday_impact.columns = df_holiday_impact.columns.str.strip()
    df_holiday_impact["avg_sales"] = pd.to_numeric(
        df_holiday_impact["avg_sales"], errors="coerce"
    )
    df_holiday_impact = df_holiday_impact.dropna(subset=["avg_sales"])

    df_sorted = df_holiday_impact.sort_values(by="avg_sales", ascending=False)

    fig = px.bar(
        df_sorted,
        x="Name",
        y="avg_sales",
        title="📊 Average Monthly Sales by Holiday",
        labels={"avg_sales": "Average Sales ($)", "Name": "Holiday"},
        color="avg_sales",
        color_continuous_scale="Tealgrn",
    )
    fig.update_layout(height=450, xaxis_tickangle=-30)
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Surprisingly, substitute holidays and spring bank holidays often outperform Christmas — indicating when real consumer activity peaks."
    )
else:
    st.info("Holiday impact data not available.")

# ─────── Section 7: Inventory Planning Classifier ───────
st.header("📦 What Should We Do With Inventory?")
st.caption(
    "This section shows how we predict stock decisions — Increase, Maintain, or Reduce — using past demand trends, seasonal effects, and holiday sensitivity."
)

# Load classifier output
df_inventory_preds = load_spark_output_csv("output/predictions_inventory")
metrics_inventory = load_metrics(
    "output/predictions_inventory/model_metrics_inventory.json"
)

# Display metrics
if metrics_inventory:
    st.subheader("🧠 Classifier Evaluation Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("✅ Accuracy", f"{metrics_inventory['accuracy']:.2%}")
    col2.metric("🎯 F1 Score", f"{metrics_inventory['f1']:.2%}")
    col3.metric("📊 Precision", f"{metrics_inventory['weightedPrecision']:.2%}")
    col4.metric("📈 Recall", f"{metrics_inventory['weightedRecall']:.2%}")
else:
    st.warning("No inventory classification metrics found.")

# Show prediction results
if not df_inventory_preds.empty:
    st.subheader("🔍 Inventory Predictions Overview")

    df_inventory_preds["match"] = (
        df_inventory_preds["stock_decision"] == df_inventory_preds["predicted_label"]
    )
    df_inventory_preds["match"] = df_inventory_preds["match"].map(
        {True: "✔️ Correct", False: "❌ Wrong"}
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        st.dataframe(df_inventory_preds)

    with col2:
        match_counts = df_inventory_preds["match"].value_counts().reset_index()
        match_counts.columns = ["Match", "Count"]
        fig = px.pie(
            match_counts,
            names="Match",
            values="Count",
            title="📊 Prediction Match Rate",
            color_discrete_map={"✔️ Correct": "green", "❌ Wrong": "red"},
        )
        st.plotly_chart(fig, use_container_width=True)

    # Breakdown by predicted class
    st.subheader("📦 Class Distribution of Predictions")
    fig = px.histogram(
        df_inventory_preds,
        x="predicted_label",
        color="match",
        barmode="group",
        labels={"predicted_label": "Predicted Class"},
        color_discrete_map={"✔️ Correct": "green", "❌ Wrong": "red"},
        title="Class Prediction Outcomes by Match",
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "This classifier enables operational decision support — helping teams proactively manage inventory based on demand shifts."
    )

else:
    st.info("Inventory predictions not available.")


# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Retail Demand Forecasting Dashboard | Powered by Spark + MLlib</p>
        <p style='color: #666; font-size: 0.8em;'>Last updated: {}</p>
    </div>
""".format(datetime.now().strftime("%Y-%m-%d %H:%M")),
    unsafe_allow_html=True,
)
