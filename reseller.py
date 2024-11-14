import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_option_menu import option_menu
from numerize.numerize import numerize
import time
from streamlit_extras.metric_cards import style_metric_cards
#st.set_option('deprecation.showPyplotGlobalUse', False)
import plotly.graph_objs as go
theme_plotly = None

#uncomment this line if you use mysql
#from query import *

# Load data
df = pd.read_csv('data/ResellerSalesDetail1.csv')

st.set_page_config(page_title="Dashboard",page_icon="🌍",layout="wide")
st.header("Business Analytics")

# Convert OrderDate to datetime format and handle errors
df['OrderDate'] = pd.to_datetime(df['OrderDate'], errors='coerce')

# load Style css
with open('style.css')as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html = True)


# Sidebar for filtering data
st.sidebar.header("Filters")

# Date range filter based on OrderDate with calendar pop-up
min_date = df['OrderDate'].min().date()
max_date = df['OrderDate'].max().date()
date_range = st.sidebar.date_input(
    "Select Order Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Split the tuple into start and end dates
start_date, end_date = date_range

product_category = st.sidebar.multiselect("Select Product Category", options=df['ProductCategory'].unique())
territory = st.sidebar.multiselect("Select Sales Territory", options=df['Country'].unique())
color = st.sidebar.multiselect("Select Color", options=df['Color'].unique())
product_style = st.sidebar.multiselect("Product Style", options=df["Style"].unique())
region = st.sidebar.multiselect("Sales Region", options=df["Region"].unique())
businesstype = st.sidebar.multiselect("Reseller Business Type", options=df["BusinessType"].unique())
product_class = st.sidebar.multiselect("Product Class", options=df["Class"].unique())
#order_date = st.date_input()




#df_selection=df.query(
    #"ProductCategory==@product_category & TerritoryGroup==@territory & Color==@color & Style==@product_style & Region==@region & BusinessType==@businesstype & Class==@product_class"
#)

# Apply filtering conditions manually
df_selection = df.copy()

# Filter based on date range
df_selection = df_selection[(df_selection['OrderDate'] >= pd.to_datetime(start_date)) &
                            (df_selection['OrderDate'] <= pd.to_datetime(end_date))]

if product_category:
    df_selection = df_selection[df_selection["ProductCategory"].isin(product_category)]

if product_class:
    df_selection = df_selection[df_selection["Class"].isin(product_class)]

if color:
    df_selection = df_selection[df_selection["Color"].isin(color)]

if product_style:
    df_selection = df_selection[df_selection["Style"].isin(product_style)]

if region:
    df_selection = df_selection[df_selection["Region"].isin(region)]

if territory:
    df_selection = df_selection[df_selection["Country"].isin(territory)]

if territory:
    df_selection = df_selection[df_selection["TerritoryGroup"].isin(territory)]

if businesstype:
    df_selection = df_selection[df_selection["BusinessType"].isin(businesstype)]

def Home():
    st.subheader("Filtered Sales Data")
    with st.expander("Tabular"):
        showData = st.multiselect('Select columns to display:', df_selection.columns, default=df_selection.columns)
        if showData:
            st.write(df_selection[showData])
        else:
            st.write("No columns selected for display. Please select columns from the dropdown.")

# Display the Home function content
Home()


def metrics():
    from streamlit_extras.metric_cards import style_metric_cards
    card1, card2, card3, card4, card5 = st.columns(5)

    card1.metric("Sales", value=f"${df_selection.SalesAmount.sum()/ 1e6:.1f}M", delta="Total Sales")
    card2.metric("Orders", value=f"{df_selection.SalesOrderNumber.nunique()/1e3:.1f}K", delta="Total Orders")
    card3.metric("Resellers", value=f"{df_selection.ResellerID.nunique()/1e3:.1f}K", delta="Resellers")
    card4.metric("Products", value=f"{df_selection.ProductName.nunique()/1e3:.1f}K", delta="Products")
    card5.metric("Countries", value=df_selection.Country.nunique(), delta="Countries")

    style_metric_cards(background_color="#071021", border_left_color="#1f66bd")
        
metrics()

# Next Level charts
div1, div2 = st.columns(2)
def productcategory_pie():
    with div1:
        fig=px.pie(df_selection, values="SalesAmount", names="ProductCategory", title="Sales by Product Category")
        fig.update_layout(legend_title="Product Category", legend_y=0.9)
        fig.update_traces(textinfo="percent+label", textposition="inside")
        st.plotly_chart(fig, use_container_width=True, theme=theme_plotly)

productcategory_pie()

def businesstype_bar():
    with div2:
        theme_plotly = None
        fig=px.bar(df_selection, y="SalesAmount", x="BusinessType", text_auto='.2s', title="Sales by Type of Business")
        #fig.update_layout(legend_title="Business Type", legend_y=0.9)
        fig.update_traces(textfont_size=14,textangle=0, textposition="outside")
        st.plotly_chart(fig, use_container_width=True, theme=theme_plotly)

businesstype_bar()


col1, col2 = st.columns(2)

def territorygroup_tree():
    # Create two columns in Streamlit for side-by-side visualization
    # Plot 1: Treemap for Sales by Territory Group
    with col1:
        treemap_fig = px.treemap(
            df_selection,
            path=["TerritoryGroup"],
            values="SalesAmount",
            title="Sales by Territory Group"
    )
    treemap_fig.update_traces(textinfo="label+value", texttemplate='%{label}<br>£%{value:.2f}M')
    st.plotly_chart(treemap_fig, use_container_width=True)

territorygroup_tree()


# # Plot 2: Bar Chart for Sales by Product Line and Sales Territory Group
# territoryproduct():
# with col2:
#     bar_chart_fig = px.bar(
#         df_selection,
#         x="ProductLine",
#         y="SalesAmount",
#         color="SalesTerritoryGroup",
#         title="Sales by Product Line and Sales Territory Group",
#         labels={"SalesAmount": "Sales Amount (£)", "ProductLine": "Product Line", "SalesTerritoryGroup": "Sales Territory Group"}
#     )
#     bar_chart_fig.update_layout(legend_title="Sales Territory Group")
#     st.plotly_chart(bar_chart_fig, use_container_width=True)



# Group by both Year and Month for Sales and Profit Analysis
df_selection['Year'] = df_selection['OrderDate'].dt.year
df_selection['Month'] = df_selection['OrderDate'].dt.month_name().str[:3]  # Get abbreviated month names

# Time Series Analysis
st.subheader("Time Series Analysis")
col1, col2 = st.columns(2)

# Sales Over Time
with col1:
    # Group by month and use month names
    # Sales by Month and Year
    monthly_sales = df_selection.groupby(['Year', 'Month'])['SalesAmount'].sum().reset_index()
    monthly_sales['Month'] = pd.Categorical(monthly_sales['Month'], categories=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], ordered=True)
    monthly_sales = monthly_sales.sort_values('Month')

    # Plot Sales by Month and Year
    sales_trend_fig = px.line(
        monthly_sales,
        x='Month', y='SalesAmount', color='Year',
        title="Sales by Month and Year",
        labels={'SalesAmount': 'Sales Amount (£)', 'Month': 'Month', 'Year': 'Year'}
    )
    sales_trend_fig.update_traces(mode="lines+markers", line_shape="linear")
    st.plotly_chart(sales_trend_fig, use_container_width=True)

# Profit Over Time
with col2:
    # Group by month and use month names
    # Profit by Month and Year
    df_selection['Profit'] = df_selection['SalesAmount'] - df_selection['Cost']
    monthly_profit = df_selection.groupby(['Year', 'Month'])['Profit'].sum().reset_index()
    monthly_profit['Month'] = pd.Categorical(monthly_profit['Month'], categories=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], ordered=True)
    monthly_profit = monthly_profit.sort_values('Month')

    # Plot Profit by Month and Year
    profit_trend_fig = px.line(
        monthly_profit,
        x='Month', y='Profit', color='Year',
        title="Profit by Month and Year",
        labels={'Profit': 'Profit (£)', 'Month': 'Month', 'Year': 'Year'}
    )
    profit_trend_fig.update_traces(mode="lines+markers", line_shape="linear")
    st.plotly_chart(profit_trend_fig, use_container_width=True)


# Regional Analysis
st.subheader("Regional Analysis")
col1, col2 = st.columns(2)

with col1:
    # Sales by Region
    sales_by_region_fig = px.bar(
        df_selection.groupby('Region')['SalesAmount'].sum().reset_index(),
        x='Region', y='SalesAmount',
        title="Sales by Region"
    )
    st.plotly_chart(sales_by_region_fig, use_container_width=True)

with col2:
    # Profit by Region
    profit_by_region_fig = px.bar(
        df_selection.groupby('Region')['Profit'].sum().reset_index(),
        x='Region', y='Profit',
        title="Profit by Region"
    )
    st.plotly_chart(profit_by_region_fig, use_container_width=True)


# Product Analysis
st.subheader("Product Analysis")
col1, col2 = st.columns(2)

with col1:
    # Top Product Categories by Sales
    top_products_sales_fig = px.bar(
        df_selection.groupby('ProductCategory')['SalesAmount'].sum().nlargest(10).reset_index(),
        x='ProductCategory', y='SalesAmount',
        title="Top Product Categories by Sales"
    )
    st.plotly_chart(top_products_sales_fig, use_container_width=True)

with col2:
    # Top Product Categories by Profit
    top_products_profit_fig = px.bar(
        df_selection.groupby('ProductCategory')['Profit'].sum().nlargest(10).reset_index(),
        x='ProductCategory', y='Profit',
        title="Top Product Categories by Profit"
    )
    st.plotly_chart(top_products_profit_fig, use_container_width=True)


# Customer Preferences
st.subheader("Customer Preferences")
col1, col2 = st.columns(2)

with col1:
    # Sales by Color
    sales_by_color_fig = px.bar(
        df_selection.groupby('Color')['SalesAmount'].sum().reset_index(),
        x='Color', y='SalesAmount',
        title="Sales by Color"
    )
    st.plotly_chart(sales_by_color_fig, use_container_width=True)

with col2:
    # Sales by Style
    sales_by_style_fig = px.bar(
        df_selection.groupby('Style')['SalesAmount'].sum().reset_index(),
        x='Style', y='SalesAmount',
        title="Sales by Style"
    )
    st.plotly_chart(sales_by_style_fig, use_container_width=True)

# # Monthly Sales Distribution
# st.subheader("Monthly Sales Distribution")
# monthly_sales_distribution_fig = px.bar(
#     df_selection.groupby(df_selection['OrderDate'].dt.month)['SalesAmount'].sum().reset_index(),
#     x='OrderDate', y='SalesAmount',
#     title="Sales Distribution by Month"
# )
# st.plotly_chart(monthly_sales_distribution_fig, use_container_width=True)

# # Group data by year and calculate total sales for each year
# annual_sales = df_selection.groupby(df_selection['OrderDate'].dt.year)['SalesAmount'].sum().reset_index()
# annual_sales.columns = ['Year', 'TotalSales']

# # Calculate year-over-year growth in sales
# annual_sales['YoY_Growth'] = annual_sales['TotalSales'].pct_change() * 100  # Convert to percentage

# # Plotting Year-over-Year Growth
# yoy_growth_fig = px.bar(
#     annual_sales,
#     x='Year', y='YoY_Growth',
#     title="Year-over-Year Growth in Sales",
#     labels={'YoY_Growth': 'YoY Growth (%)', 'Year': 'Year'},
#     text='YoY_Growth'
# )
# yoy_growth_fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')

# # Display the plot
# st.plotly_chart(yoy_growth_fig, use_container_width=True)



# # Step 1: Group data by year and month, and calculate total sales for each month
# monthly_sales = df_selection.groupby([df_selection['OrderDate'].dt.year, df_selection['OrderDate'].dt.month])['SalesAmount'].sum().reset_index()
# monthly_sales.columns = ['Year', 'Month', 'TotalSales']

# # Sort data by Year and Month
# monthly_sales = monthly_sales.sort_values(['Year', 'Month'])

# # Step 2: Calculate Rolling Average for a 3-month window
# monthly_sales['RollingAvgSales'] = monthly_sales['TotalSales'].rolling(window=3).mean()

# # Step 3: Calculate Cumulative Sales
# monthly_sales['CumulativeSales'] = monthly_sales['TotalSales'].cumsum()

# # Step 4: Calculate Month-over-Month growth for non-cumulative analysis
# monthly_sales['MoM_Growth'] = monthly_sales['TotalSales'].pct_change() * 100  # Convert to percentage
# # Reset MoM growth for the first month of each year to NaN, as it doesn't have a previous month to compare
# monthly_sales.loc[monthly_sales['Month'] == 1, 'MoM_Growth'] = None

# # Step 5: Plot Non-Cumulative Growth with Rolling Average
# non_cumulative_fig = px.line(
#     monthly_sales,
#     x=monthly_sales.apply(lambda row: f"{row['Year']}-{row['Month']:02}", axis=1),
#     y=['TotalSales', 'RollingAvgSales'],
#     title="Non-Cumulative Sales with Rolling Average",
#     labels={'value': 'Sales Amount (£)', 'variable': 'Legend', 'x': 'Year-Month'},
# )
# non_cumulative_fig.for_each_trace(lambda trace: trace.update(name={'TotalSales': 'Monthly Sales', 'RollingAvgSales': '3-Month Rolling Average'}[trace.name]))
# st.plotly_chart(non_cumulative_fig, use_container_width=True)

# # Step 6: Plot Cumulative Growth
# cumulative_growth_fig = px.line(
#     monthly_sales,
#     x=monthly_sales.apply(lambda row: f"{row['Year']}-{row['Month']:02}", axis=1),
#     y='CumulativeSales',
#     title="Cumulative Sales Growth",
#     labels={'CumulativeSales': 'Cumulative Sales (£)', 'x': 'Year-Month'}
# )
# st.plotly_chart(cumulative_growth_fig, use_container_width=True)

# # Step 7: Plot Month-over-Month Growth
# mom_growth_fig = px.bar(
#     monthly_sales,
#     x=monthly_sales.apply(lambda row: f"{row['Year']}-{row['Month']:02}", axis=1),
#     y='MoM_Growth',
#     title="Month-over-Month Sales Growth",
#     labels={'MoM_Growth': 'MoM Growth (%)', 'x': 'Year-Month'}
# )
# mom_growth_fig.update_traces(texttemplate='%{y:.2f}%', textposition='outside')
# mom_growth_fig.update_layout(xaxis_tickangle=-45, yaxis_tickformat=".2f%")
# st.plotly_chart(mom_growth_fig, use_container_width=True)