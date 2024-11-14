import streamlit as st
import pandas as pd
import plotly.express as px

# Set Streamlit page configuration
st.set_page_config(page_title="Dashboard", page_icon="🌍", layout="wide")

# Load the external CSS file
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load data
dfc = pd.read_csv('data/InternetSales.csv', encoding='latin1')

# Define the data template
def create_template():
    columns = [
        "OrderDate", "ProductCategory", "Country", "Color", "Style", "Region", 
        "SalesAmount", "Cost", "CustomerID", "ProductName", "SalesOrderNumber", "ReturnAmount", "Freight", "Gender", "Age", "MaritalStatus"
    ]
    # Create an empty DataFrame with these columns
    template_df = pd.DataFrame(columns=columns)
    return template_df

# Download button for the data template
st.sidebar.header("Data Template")
template_df = create_template()
csv_template = template_df.to_csv(index=False).encode('utf-8')
st.sidebar.download_button(
    label="Download Data Template",
    data=csv_template,
    file_name="data_template.csv",
    mime="text/csv"
)

# File upload for user-populated data
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload your populated data template (CSV format)", type="csv")

# Load data from the uploaded file or use default data
if uploaded_file:
    try:
        dfc = pd.read_csv(uploaded_file)
        st.success("Data uploaded successfully!")
    except Exception as e:
        st.error(f"Error loading file: {e}")
else:
    # Load default data if no file is uploaded
    dfc = pd.read_csv('data/InternetSales.csv', encoding='latin1')

# Check if the necessary columns are available before creating the Profit column
if 'SalesAmount' in dfc.columns and 'Cost' in dfc.columns:
    dfc['Profit'] = dfc['SalesAmount'] - dfc['Cost']
else:
    st.error("The columns 'SalesAmount' or 'Cost' are missing from the dataset.")

# Ensure OrderDate column is parsed as datetime and calculate min and max dates
if 'OrderDate' in dfc.columns:
    dfc['OrderDate'] = pd.to_datetime(dfc['OrderDate'], errors='coerce')
    min_date = dfc['OrderDate'].min()
    max_date = dfc['OrderDate'].max()
    
    # Sidebar date range filter
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    # Convert date_range to datetime format and filter data
    date_range = pd.to_datetime(date_range)
    dfc = dfc[(dfc['OrderDate'] >= date_range[0]) & (dfc['OrderDate'] <= date_range[1])]

    # Extract Month and Year from OrderDate
    dfc['Month'] = dfc['OrderDate'].dt.month
    dfc['Year'] = dfc['OrderDate'].dt.year
else:
    st.error("The column 'OrderDate' is missing from the dataset.")

# Sidebar additional filters
st.sidebar.header("Additional Filters")
product_category = st.sidebar.selectbox("Product Category", dfc['ProductCategory'].unique(), index=0)
sales_country = st.sidebar.selectbox("Country", dfc['Country'].unique(), index=0)
color = st.sidebar.selectbox("Color", dfc['Color'].unique(), index=0)
style = st.sidebar.selectbox("Style", dfc['Style'].unique(), index=0)
region = st.sidebar.selectbox("Region", dfc['Region'].unique(), index=0)

# Apply filters to the DataFrame
filtered_df = dfc[
    (dfc['ProductCategory'] == product_category) &
    (dfc['Country'] == sales_country) &
    (dfc['Color'] == color) &
    (dfc['Style'] == style) &
    (dfc['Region'] == region)
]

# Display key metrics
st.header("Business Analytics Dashboard")
total_sales = filtered_df['SalesAmount'].sum()
total_profit = filtered_df['Profit'].sum()
total_customers = filtered_df['CustomerID'].nunique()
total_products = filtered_df['ProductName'].nunique()
total_countries = filtered_df['Country'].nunique()
total_orders = filtered_df['SalesOrderNumber'].nunique()
total_returns = filtered_df['ReturnAmount'].sum() if 'ReturnAmount' in filtered_df.columns else 0
total_shipping_cost = filtered_df['Freight'].sum() if 'Freight' in filtered_df.columns else 0

# Display key metrics with icons
st.subheader("Key Metrics")
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.markdown(f"<div class='embossed-card'><div class='icon icon-sales'>💰</div><h3>Revenue</h3><p>${total_sales:,.2f}</p></div>", unsafe_allow_html=True)    
with col2:
    st.markdown(f"<div class='embossed-card'><div class='icon icon-profit'>📈</div><h3>Profit</h3><p>${total_profit:,.2f}</p></div>", unsafe_allow_html=True)    
with col3:
    st.markdown(f"<div class='embossed-card'><div class='icon icon-customers'>👥</div><h3>Customers</h3><p>{total_customers}</p></div>", unsafe_allow_html=True)    
with col4:
    st.markdown(f"<div class='embossed-card'><div class='icon icon-products'>📦</div><h3>Products</h3><p>{total_products}</p></div>", unsafe_allow_html=True)    
with col5:
    st.markdown(f"<div class='embossed-card'><div class='icon icon-countries'>🌍</div><h3>Countries</h3><p>{total_countries}</p></div>", unsafe_allow_html=True)
    

# Tabs for detailed insights
tab1, tab2, tab3 = st.tabs(["Sales Insights", "Customer Insights", "Product Insights"])

# Sales Insights Tab
with tab1:
    st.subheader("Sales Insights")
    
    # Sales by month and year area chart
    fig_sales_time = px.area(
        filtered_df,
        x='Month',
        y='SalesAmount',
        color='Year',
        title="Sales by Month and Year",
        labels={'SalesAmount': 'Sales Amount', 'Month': 'Month'},
    )
    fig_sales_time.update_layout(yaxis_tickformat="£,.0f", xaxis=dict(tickmode="array", tickvals=list(range(1, 13)), ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']))

    st.plotly_chart(fig_sales_time, use_container_width=True)

# Create age groups with string labels for JSON serialization compatibility
age_bins = [0, 30, 40, 50, 60, 70, 100]
dfc['AgeGroup'] = pd.cut(dfc['Age'], bins=age_bins).astype(str)

# Customer Insights Tab
with tab2:
    st.subheader("Customer Insights")
    # Sales by gender
    fig_sales_gender = px.pie(dfc, names='Gender', values='SalesAmount', title="Sales Distribution by Gender")
    st.plotly_chart(fig_sales_gender, use_container_width=True)

    # Sales by age group
    fig_sales_age = px.bar(dfc, x='AgeGroup', y='SalesAmount', title="Sales by Age Group")
    st.plotly_chart(fig_sales_age, use_container_width=True)

    # Profit by marital status
    fig_profit_marital = px.bar(dfc, x='MaritalStatus', y='Profit', title="Profit by Marital Status")
    st.plotly_chart(fig_profit_marital, use_container_width=True)

    

# Product Insights Tab
with tab3:
    st.subheader("Product Insights")
    # Sales by product category
    fig_sales_category = px.bar(dfc, x='ProductCategory', y='SalesAmount', title="Sales by Product Category")
    st.plotly_chart(fig_sales_category, use_container_width=True)

    # Profit by product model
    fig_profit_model = px.bar(dfc, x='ModelName', y='Profit', title="Profit by Model")
    st.plotly_chart(fig_profit_model, use_container_width=True)

    # Sales by promotion type
    fig_sales_promo = px.bar(dfc, x='PromotionType', y='SalesAmount', color='Gender', barmode='group', title="Sales by Promotion Type")
    st.plotly_chart(fig_sales_promo, use_container_width=True)
