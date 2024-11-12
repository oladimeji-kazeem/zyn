import streamlit as st
import plotly.express as px
import pandas as pd
import datetime

# Load the dataset
df = pd.read_csv('data/ResellerSalesDetail1.csv')

# Calculate Profit if not already present
if 'Profit' not in df.columns:
    df['Profit'] = df['SalesAmount'] - df['Cost']

# Preprocess date columns
df['OrderDate'] = pd.to_datetime(df['OrderDate'], errors='coerce')
df = df.dropna(subset=['OrderDate'])  # Remove any rows with null OrderDate
df['OrderMonth'] = df['OrderDate'].dt.month
df['OrderYear'] = df['OrderDate'].dt.year

# Sidebar date range filter
st.sidebar.header("Select Order Date Range")
min_date = datetime.date(2011, 1, 1)
max_date = datetime.date(2014, 1, 31)
start_date = st.sidebar.date_input("Start Date", min_date)
end_date = st.sidebar.date_input("End Date", max_date)

# Validate date range
if start_date < end_date:
    st.sidebar.success(f"Start date: `{start_date}`\nEnd date: `{end_date}`")
else:
    st.sidebar.error("Error: End date must fall after start date.")

# Additional filters
st.sidebar.header("Additional Filters")
territory_filter = st.sidebar.selectbox("Select Sales Territory Group", options=df['TerritoryGroup'].unique())
category_filter = st.sidebar.selectbox("Select Product Subcategory", options=df['ProductSubcategory'].unique())

# Filter data based on selected dates and additional filters
filtered_df = df[(df['OrderDate'] >= pd.to_datetime(start_date)) & 
                 (df['OrderDate'] <= pd.to_datetime(end_date)) &
                 (df['TerritoryGroup'] == territory_filter) &
                 (df['ProductSubcategory'] == category_filter)]

# Display the filtered data summary
st.title("Sales and Profitability Dashboard")
st.write(f"Displaying data from {start_date} to {end_date} for Territory Group '{territory_filter}' and Product Subcategory '{category_filter}'")

# Split the layout into two columns for charts
col1, col2 = st.columns(2)

# Sales by Territory Group and Product Line
with col1:
    territory_product_fig = px.bar(
        filtered_df, 
        x="TerritoryGroup", 
        y="SalesAmount", 
        color="ProductSubcategory", 
        title="Sales by Territory Group and Product Line",
        labels={"SalesAmount": "Sales Amount (£)", "TerritoryGroup": "Territory Group"}
    )
    st.plotly_chart(territory_product_fig)

# Sales and Profit Trends
with col2:
    sales_trend_fig = px.line(
        filtered_df.groupby(['OrderYear', 'OrderMonth'])['SalesAmount'].sum().reset_index(),
        x="OrderMonth", 
        y="SalesAmount", 
        color="OrderYear",
        title="Sales Trend by Month and Year",
        labels={"OrderMonth": "Month", "SalesAmount": "Sales Amount (£)", "OrderYear": "Year"}
    )
    st.plotly_chart(sales_trend_fig)

# Profitability by Product Subcategory - Scatter Plot
with col1:
    scatter_fig = px.scatter(
        filtered_df, 
        x="SalesAmount", 
        y="Profit", 
        color="ProductSubcategory", 
        title="Sales Amount vs Profit by Product Subcategory",
        labels={"SalesAmount": "Sales Amount (£)", "Profit": "Profit (£)"}
    )
    st.plotly_chart(scatter_fig)

# Profitability by Product Subcategory - Box Plot
with col2:
    top_categories = df['ProductSubcategory'].value_counts().nlargest(10).index
    box_fig = px.box(
        filtered_df[filtered_df['ProductSubcategory'].isin(top_categories)], 
        x="ProductSubcategory", 
        y="Profit", 
        title="Profit by Product Subcategory (Top 10)",
        labels={"ProductSubcategory": "Product Subcategory", "Profit": "Profit (£)"}
    )
    st.plotly_chart(box_fig)