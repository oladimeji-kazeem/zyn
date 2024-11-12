import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
import plotly.express as px
from numerize.numerize import numerize

# Set paths for data and images
st.sidebar.image("images/app_logo.png", caption="Zynalytics")

# Application name
st.title("Comprehensive Sales Analytics")

# Load the data
df = pd.read_csv("data/ResellerSalesDetail1.csv")

# Sidebar filters
st.sidebar.header("Please Filter")
product_category = st.sidebar.multiselect(
    "Select Product Category",
    options=df["ProductCategory"].unique(),
)

product_subcategory = st.sidebar.multiselect(
    "Select Product Subcategory",
    options=df["ProductSubcategory"].unique(),
)

product_class = st.sidebar.multiselect(
    "Product Class",
    options=df["Class"].unique(),
)

product_color = st.sidebar.multiselect(
    "Product Color",
    options=df["Color"].unique(),
)

product_style = st.sidebar.multiselect(
    "Product Style",
    options=df["Style"].unique(),
)

region = st.sidebar.multiselect(
    "Sales Region",
    options=df["Region"].unique(),
)

country = st.sidebar.multiselect(
    "Sales Country",
    options=df["Country"].unique(),
)

territory = st.sidebar.multiselect(
    "Sales Territory Group",
    options=df["TerritoryGroup"].unique(),
)

businesstype = st.sidebar.multiselect(
    "Reseller Business Type",
    options=df["BusinessType"].unique(),
)

# Apply filtering conditions manually
df_selection = df.copy()

if product_category:
    df_selection = df_selection[df_selection["ProductCategory"].isin(product_category)]

if product_subcategory:
    df_selection = df_selection[df_selection["ProductSubcategory"].isin(product_subcategory)]

if product_class:
    df_selection = df_selection[df_selection["Class"].isin(product_class)]

if product_color:
    df_selection = df_selection[df_selection["Color"].isin(product_color)]

if product_style:
    df_selection = df_selection[df_selection["Style"].isin(product_style)]

if region:
    df_selection = df_selection[df_selection["Region"].isin(region)]

if country:
    df_selection = df_selection[df_selection["Country"].isin(country)]

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


# Top Analytics card
df_selection["SalesAmount"] = pd.to_numeric(df_selection["SalesAmount"], errors="coerce")
total_sales = df_selection["SalesAmount"].sum()
#total_sales = float(df_selection["SalesAmount"]).sum()
sales_orders = df_selection["SalesOrderNumber"].nunique()
resellers = df_selection["ResellerID"].nunique()
products = df_selection["ProductName"].nunique()
countries = df_selection["Country"].nunique()

# Custom CSS for embossed effect and background color
st.markdown("""
    <style>
    .metric-container {
        background-color: #f0f8ff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 5px 5px 10px rgba(0, 0, 0, 0.1), -5px -5px 10px rgba(255, 255, 255, 0.7);
        text-align: center;
    }
    .metric-label {
        font-size: 1.2em;
        color: #333;
        font-weight: bold;
    }
    .metric-value {
        font-size: 1.8em;
        color: #007ACC;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)


col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Sales</div>
            <div class="metric-value">${total_sales / 1e6:.1f}M</div>
        </div>
        """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Orders</div>
            <div class="metric-value">{sales_orders / 1e3:.1f}K</div>
        </div>
        """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Resellers</div>
            <div class="metric-value">{resellers}</div>
        </div>
        """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Products</div>
            <div class="metric-value">{products}</div>
        </div>
        """, unsafe_allow_html=True)

with col5:
    st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Countries</div>
            <div class="metric-value">{countries}</div>
        </div>
        """, unsafe_allow_html=True)





# Plot 1: Sales by Territory Group (Treemap)
fig1 = px.treemap(
    df_selection['TerritoryGroup'],
    path=["TerritoryGroup"],
    values="SalesAmount",
    title="Sales by Territory Group",
    color="TerritoryGroup",
    color_discrete_map={
        "North America": "teal",
        "Pacific": "coral",
        "Europe": "gray"
    }
)
fig1.update_layout(margin=dict(t=30, l=0, r=0, b=0))

# Plot 2: Sales by Month and Year (Line chart)
fig2 = px.area(
    df_selection['TerritoryGroup'],
    x="Month",
    y="SalesAmount",
    color="Year",
    line_group="Year",
    title="Sales by Month and Year",
    color_discrete_sequence=["teal", "orange", "gray", "coral"]
)
fig2.update_layout(margin=dict(t=30, l=0, r=0, b=0))

# Plot 3: Sales by Product Line and Territory Group (Bar chart)
fig3 = px.bar(
    df_selection['ProductCategory'],
    x="ProductLine",
    y="SalesAmount",
    color="TerritoryGroup",
    barmode="group",
    title="Sales by Product Line and Sales Territory Group",
    color_discrete_map={
        "Europe": "gray",
        "North America": "teal",
        "Pacific": "coral"
    }
)
fig3.update_layout(margin=dict(t=30, l=0, r=0, b=0))

# Display the plots in three columns
col1, col2, col3 = st.columns(3)

with col1:
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.plotly_chart(fig2, use_container_width=True)

with col3:
    st.plotly_chart(fig3, use_container_width=True)