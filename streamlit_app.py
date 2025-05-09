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
    return pd.read_csv(part_files[0])


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

# ─────── Section 1: Executive Summary ───────
st.header("📊 Model Performance Summary")
if metrics:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Root Mean Square Error",
            f"${metrics['RMSE']:,.2f}",
            help="Lower is better - measures prediction accuracy",
        )
    with col2:
        st.metric(
            "Mean Absolute Error",
            f"${metrics['MAE']:,.2f}",
            help="Average absolute difference between predictions and actual values",
        )
    with col3:
        st.metric(
            "Model Complexity",
            f"{metrics['Best Hyperparameters']['numTrees']} trees",
            help=f"Max depth: {metrics['Best Hyperparameters']['maxDepth']}",
        )
else:
    st.warning("No model metrics found.")

# ─────── Section 2: Predictions ───────
st.header("🔮 Forecast Results")
if not df_preds.empty:
    col1, col2 = st.columns([1, 2])
    with col1:
        product = st.selectbox(
            "Select a product",
            df_preds["Description"].unique(),
            help="Choose a product to view its forecast",
        )
        date_range = st.date_input(
            "Select date range",
            value=(
                pd.to_datetime(df_preds["InvoiceMonth"].min()),
                pd.to_datetime(df_preds["InvoiceMonth"].max()),
            ),
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
        title=f"Sales Forecast for {product}",
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
        "Product-specific MAE",
        f"${mae:,.2f}",
        help="Mean Absolute Error for this specific product",
    )
else:
    st.info("Prediction data not available.")

# ─────── Section 3: Holiday Product Demand ───────
st.header("🎁 Holiday Season Analysis")
if not df_top_holiday_products.empty:
    col1, col2 = st.columns([1, 2])
    with col1:
        top_n = st.slider(
            "Number of top products to display",
            5,
            20,
            10,
            help="Select how many top-selling holiday products to show",
        )

    fig = px.bar(
        df_top_holiday_products.head(top_n),
        x="Description",
        y="total_sales",
        title="Top Selling Products During Holiday Season",
        labels={"total_sales": "Total Sales ($)", "Description": "Product"},
        color="total_sales",
        color_continuous_scale="Viridis",
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

# ─────── Section 4: Quarter-wise Demand ───────
st.header("📆 Quarterly Trends")
if not df_quarterly.empty:
    # Create a heatmap of quarterly sales
    quarterly_pivot = df_quarterly.pivot(
        index="Quarter", columns="Description", values="avg_quarter_sales"
    )

    fig = px.imshow(
        quarterly_pivot,
        title="Quarterly Sales Heatmap",
        labels=dict(x="Product", y="Quarter", color="Average Sales"),
        color_continuous_scale="Viridis",
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

# ─────── Section 5: Holiday Comparison ───────
st.header("🎄 Holiday Season Comparison")
if not df_newyear_vs_christmas.empty:
    # Ensure Year is string for plotting
    df_newyear_vs_christmas["Year"] = df_newyear_vs_christmas["Year"].astype(str)
    # Product selector if multiple products
    if (
        "Description" in df_newyear_vs_christmas.columns
        and df_newyear_vs_christmas["Description"].nunique() > 1
    ):
        product = st.selectbox(
            "Select a product for year-wise comparison",
            df_newyear_vs_christmas["Description"].unique(),
            key="nyc_product",
        )
        df_plot = df_newyear_vs_christmas[
            df_newyear_vs_christmas["Description"] == product
        ]
    else:
        df_plot = df_newyear_vs_christmas.copy()
    # Map months to labels
    month_map = {1: "New Year (Jan)", 12: "Christmas (Dec)"}
    df_plot["MonthLabel"] = df_plot["Month"].map(month_map)
    # Grouped bar chart
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
        title="Year-wise New Year vs Christmas Sales Comparison",
    )
    fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
    fig.update_layout(yaxis=dict(tickformat=",d"), height=500)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Comparison of average sales in January (New Year) and December (Christmas) for each year."
    )

# ─────── Section 6: Holiday Impact ───────
st.header("🏆 Holiday Impact Analysis")
if not df_holiday_impact.empty:
    fig = px.bar(
        df_holiday_impact,
        x="Name",
        y="avg_sales",
        title="Sales Impact by Holiday",
        labels={"avg_sales": "Average Sales ($)", "Name": "Holiday"},
        color="avg_sales",
        color_continuous_scale="Viridis",
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

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
