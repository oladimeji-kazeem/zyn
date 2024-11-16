import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from io import BytesIO
from prophet import Prophet
from prophet.plot import plot_plotly
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import datetime as dt

# Set Streamlit page configuration
st.set_page_config(page_title="Business Analytics Dashboard", page_icon="🌍", layout="wide")

# Load the external CSS file
try:
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    st.warning("Custom CSS file not found. Proceeding with default styling.")

# Load data
@st.cache_data
def load_default_data():
    try:
        return pd.read_csv('data/InternetSales.csv', encoding='latin1')
    except FileNotFoundError:
        st.error("Default data file not found.")
        return pd.DataFrame()

# Define the data template
def create_template():
    columns = [
        "OrderDate", "ProductCategory", "Country", "Color", "Style", "Region",
        "SalesAmount", "Cost", "CustomerID", "ProductName", "SalesOrderNumber",
        "ReturnAmount", "Freight", "Gender", "Age", "MaritalStatus"
    ]
    return pd.DataFrame(columns=columns)

# Sidebar: Download data template
st.sidebar.header("Data Template")
template_df = create_template()
csv_template = template_df.to_csv(index=False).encode('utf-8')
st.sidebar.download_button(
    label="Download Data Template",
    data=csv_template,
    file_name="data_template.csv",
    mime="text/csv"
)

# Sidebar: Upload data
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload your populated data template (CSV format)", type="csv")

# Load data
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("Data uploaded successfully!")
    except Exception as e:
        st.error(f"Error loading file: {e}")
        df = pd.DataFrame()
else:
    df = load_default_data()

# Ensure necessary columns exist
required_columns = {"OrderDate", "SalesAmount", "Cost"}
if not df.empty:
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        st.error(f"Missing columns: {', '.join(missing_columns)}. Please upload a valid file.")
        df = pd.DataFrame()
    else:
        df['OrderDate'] = pd.to_datetime(df['OrderDate'], errors='coerce')
        df['Profit'] = df['SalesAmount'] - df['Cost']

# Sidebar: Date range filter
if not df.empty and 'OrderDate' in df.columns:
    st.sidebar.header("Date Range Filter")
    min_date = df['OrderDate'].min()
    max_date = df['OrderDate'].max()
    start_date = st.sidebar.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date)
    end_date = st.sidebar.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)
    df = df[(df['OrderDate'] >= pd.to_datetime(start_date)) & (df['OrderDate'] <= pd.to_datetime(end_date))]

# Sidebar: Additional filters
if not df.empty:
    st.sidebar.header("Additional Filters")
    filters = {}
    for column in ["ProductCategory", "Country", "Color", "Style", "Region"]:
        if column in df.columns:
            options = ["All"] + list(df[column].dropna().unique())
            selected_option = st.sidebar.selectbox(column, options)
            if selected_option != "All":
                filters[column] = selected_option
    for column, value in filters.items():
        df = df[df[column] == value]

def metrics():
    from streamlit_extras.metric_cards import style_metric_cards
    card1, card2, card3, card4, card5 = st.columns(5)

    card1.metric("Sales", value=f"${df.SalesAmount.sum()/ 1e6:.1f}M", delta="Total Sales")
    card2.metric("Orders", value=f"{df.Profit.sum()/ 1e6:.1f}M", delta="Profit")
    card3.metric("Resellers", value=f"{df.CustomerID.nunique()/1e3:.1f}K", delta="Customers")
    card4.metric("Products", value=f"{df.ProductName.nunique()}", delta="Products")
    card5.metric("Countries", value=df.Country.nunique(), delta="Countries")

    style_metric_cards(background_color="#dfdfdf", border_left_color="#1f66bd")
        


# Dropdown menu control for navigation
menu = st.selectbox(
    "Select a Section",
    options=["Home", "Sales Insights", "Customer Insights", "Product Insights", "Segmentation", "CLV Prediction", "Forecasting"]
)

# Home Section
def display_home():
    st.subheader("Filtered Sales Data")
    if not df.empty:
        with st.expander("Tabular Data"):
            # Add a unique key to avoid duplicate element ID errors
            selected_columns = st.multiselect(
                "Select columns to display:",
                df.columns,
                default=df.columns,
                key="home_display_columns"
            )
            if selected_columns:
                filtered_table = df[selected_columns]
                st.dataframe(filtered_table)
                    
                # Export button for filtered table data
                csv_data = filtered_table.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Export Filtered Data as CSV",
                    data=csv_data,
                    file_name="filtered_sales_data.csv",
                    mime="text/csv"
                )
            else:
                st.write("No columns selected for display. Please select columns from the dropdown.")
    else:
        st.error("No data available to display.")

def display_trends():
    st.subheader("Trends")

    # Annual trends for Sales and Profit
    col1, col2 = st.columns(2)

    # Annual Sales Trend
    with col1:
        st.write("**Annual Sales Trend**")
        annual_sales = df.groupby(df['OrderDate'].dt.year)['SalesAmount'].sum().reset_index()
        annual_sales.columns = ['Year', 'SalesAmount']
        fig_annual_sales = px.line(
            annual_sales, 
            x='Year', 
            y='SalesAmount', 
            title="Annual Sales Trend",
            labels={'SalesAmount': 'Sales Amount (Millions)', 'Year': 'Year'},
            markers=True
        )
        fig_annual_sales.update_layout(
            yaxis_tickformat="£,.0f",  # Format values as currency
            yaxis=dict(tickprefix="£", tickformat=".1s"),  # Show values in millions
            xaxis_title="Year"
        )
        st.plotly_chart(fig_annual_sales, use_container_width=True)

    # Annual Profit Trend
    with col2:
        st.write("**Annual Profit Trend**")
        annual_profit = df.groupby(df['OrderDate'].dt.year)['Profit'].sum().reset_index()
        annual_profit.columns = ['Year', 'Profit']
        fig_annual_profit = px.line(
            annual_profit, 
            x='Year', 
            y='Profit', 
            title="Annual Profit Trend",
            labels={'Profit': 'Profit (Millions)', 'Year': 'Year'},
            markers=True
        )
        fig_annual_profit.update_layout(
            yaxis_tickformat="£,.0f",  # Format values as currency
            yaxis=dict(tickprefix="£", tickformat=".1s"),  # Show values in millions
            xaxis_title="Year"
        )
        st.plotly_chart(fig_annual_profit, use_container_width=True)

    # Year-over-Year Trends
    st.subheader("Year-over-Year Trends")
    col3, col4 = st.columns(2)

    # Year-over-Year Sales Change
    with col3:
        st.write("**Year-over-Year Sales Change**")
        annual_sales['YoY Change'] = annual_sales['SalesAmount'].pct_change() * 100  # Calculate percentage change
        fig_yoy_sales = px.bar(
            annual_sales,
            x='Year',
            y='YoY Change',
            title="Year-over-Year Sales Change (%)",
            labels={'YoY Change': 'YoY Change (%)', 'Year': 'Year'},
            text='YoY Change'
        )
        fig_yoy_sales.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        fig_yoy_sales.update_layout(
            yaxis_title="Percentage Change (%)",
            xaxis_title="Year",
            yaxis=dict(tickformat=".0f")
        )
        st.plotly_chart(fig_yoy_sales, use_container_width=True)

    # Year-over-Year Profit Change
    with col4:
        st.write("**Year-over-Year Profit Change**")
        annual_profit['YoY Change'] = annual_profit['Profit'].pct_change() * 100  # Calculate percentage change
        fig_yoy_profit = px.bar(
            annual_profit,
            x='Year',
            y='YoY Change',
            title="Year-over-Year Profit Change (%)",
            labels={'YoY Change': 'YoY Change (%)', 'Year': 'Year'},
            text='YoY Change'
        )
        fig_yoy_profit.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        fig_yoy_profit.update_layout(
            yaxis_title="Percentage Change (%)",
            xaxis_title="Year",
            yaxis=dict(tickformat=".0f")
        )
        st.plotly_chart(fig_yoy_profit, use_container_width=True)

    # Monthly trends for Sales and Profit
    col5, col6 = st.columns(2)

    # Monthly Sales Trend
    with col5:
        st.write("**Monthly Sales Trend**")
        monthly_sales = df.groupby(df['OrderDate'].dt.month)['SalesAmount'].sum().reset_index()
        monthly_sales.columns = ['Month', 'SalesAmount']
        fig_monthly_sales = px.line(
            monthly_sales, 
            x='Month', 
            y='SalesAmount', 
            title="Monthly Sales Trend",
            labels={'SalesAmount': 'Sales Amount (Millions)', 'Month': 'Month'},
            markers=True
        )
        fig_monthly_sales.update_layout(
            yaxis_tickformat="£,.0f",  # Format values as currency
            yaxis=dict(tickprefix="£", tickformat=".1s"),  # Show values in millions
            xaxis=dict(
                tickmode="array", 
                tickvals=list(range(1, 13)), 
                ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            )
        )
        st.plotly_chart(fig_monthly_sales, use_container_width=True)

    # Monthly Profit Trend
    with col6:
        st.write("**Monthly Profit Trend**")
        monthly_profit = df.groupby(df['OrderDate'].dt.month)['Profit'].sum().reset_index()
        monthly_profit.columns = ['Month', 'Profit']
        fig_monthly_profit = px.line(
            monthly_profit, 
            x='Month', 
            y='Profit', 
            title="Monthly Profit Trend",
            labels={'Profit': 'Profit (Millions)', 'Month': 'Month'},
            markers=True
        )
        fig_monthly_profit.update_layout(
            yaxis_tickformat="£,.0f",  # Format values as currency
            yaxis=dict(tickprefix="£", tickformat=".1s"),  # Show values in millions
            xaxis=dict(
                tickmode="array", 
                tickvals=list(range(1, 13)), 
                ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            )
        )
        st.plotly_chart(fig_monthly_profit, use_container_width=True)

# Call the display_trends function
if menu == "Home":
    metrics()
    
    display_trends()

def display_sales_insights():
    st.subheader("Sales Insights")

    # SalesAmount and Profit on the same axes (Line Graphs)
    st.markdown("### SalesAmount and Profit Trends")
    col1, col2 = st.columns(2)

    # Yearly Trends: SalesAmount and Profit
    with col1:
        st.write("**Yearly Sales and Profit Trends**")
        yearly_data = df.groupby(df['OrderDate'].dt.year).agg({
            'SalesAmount': 'sum',
            'Profit': 'sum'
        }).reset_index()
        yearly_data.columns = ['Year', 'SalesAmount', 'Profit']
        fig_yearly_sales_profit = px.line(
            yearly_data,
            x='Year',
            y=['SalesAmount', 'Profit'],
            title="Yearly Sales and Profit Trends",
            labels={'value': 'Amount (£)', 'Year': 'Year', 'variable': 'Metric'},
            markers=True
        )
        fig_yearly_sales_profit.update_layout(yaxis_tickformat="£,.0f", legend_title="Metrics")
        st.plotly_chart(fig_yearly_sales_profit, use_container_width=True)

    # Monthly Trends: SalesAmount and Profit
    with col2:
        st.write("**Monthly Sales and Profit Trends**")
        monthly_data = df.groupby(df['OrderDate'].dt.month).agg({
            'SalesAmount': 'sum',
            'Profit': 'sum'
        }).reset_index()
        monthly_data.columns = ['Month', 'SalesAmount', 'Profit']
        fig_monthly_sales_profit = px.line(
            monthly_data,
            x='Month',
            y=['SalesAmount', 'Profit'],
            title="Monthly Sales and Profit Trends",
            labels={'value': 'Amount (£)', 'Month': 'Month', 'variable': 'Metric'},
            markers=True
        )
        fig_monthly_sales_profit.update_layout(
            yaxis_tickformat="£,.0f",
            legend_title="Metrics",
            xaxis=dict(
                tickmode="array",
                tickvals=list(range(1, 13)),
                ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            )
        )
        st.plotly_chart(fig_monthly_sales_profit, use_container_width=True)

    # Tax and Freight on the same axes (Line Graphs)
    st.markdown("### Tax and Freight Trends")
    col3, col4 = st.columns(2)

    # Yearly Trends: Tax and Freight
    with col3:
        st.write("**Yearly Tax and Freight Trends**")
        yearly_tax_freight = df.groupby(df['OrderDate'].dt.year).agg({
            'Freight': 'sum',
            'TaxAmt': 'sum'  # Ensure 'TaxAmt' column exists in the dataset
        }).reset_index()
        yearly_tax_freight.columns = ['Year', 'Freight', 'TaxAmt']
        fig_yearly_tax_freight = px.line(
            yearly_tax_freight,
            x='Year',
            y=['Freight', 'TaxAmt'],
            title="Yearly Tax and Freight Trends",
            labels={'value': 'Amount (£)', 'Year': 'Year', 'variable': 'Metric'},
            markers=True
        )
        fig_yearly_tax_freight.update_layout(yaxis_tickformat="£,.0f", legend_title="Metrics")
        st.plotly_chart(fig_yearly_tax_freight, use_container_width=True)

    # Monthly Trends: Tax and Freight
    with col4:
        st.write("**Monthly Tax and Freight Trends**")
        monthly_tax_freight = df.groupby(df['OrderDate'].dt.month).agg({
            'Freight': 'sum',
            'TaxAmt': 'sum'  # Ensure 'TaxAmt' column exists in the dataset
        }).reset_index()
        monthly_tax_freight.columns = ['Month', 'Freight', 'TaxAmt']
        fig_monthly_tax_freight = px.line(
            monthly_tax_freight,
            x='Month',
            y=['Freight', 'TaxAmt'],
            title="Monthly Tax and Freight Trends",
            labels={'value': 'Amount (£)', 'Month': 'Month', 'variable': 'Metric'},
            markers=True
        )
        fig_monthly_tax_freight.update_layout(
            yaxis_tickformat="£,.0f",
            legend_title="Metrics",
            xaxis=dict(
                tickmode="array",
                tickvals=list(range(1, 13)),
                ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            )
        )
        st.plotly_chart(fig_monthly_tax_freight, use_container_width=True)
    
    # Other sales insight
    st.markdown("### Other Sales insights")
    col5, col6 = st.columns(2)

    # Treemap of SalesAmount by Territory Groups
    with col5:
        st.write("**Treemap of SalesAmount by Territory Groups**")
        if 'TerritoryGroup' in df.columns:
            treemap_data = df.groupby('TerritoryGroup').agg({'SalesAmount': 'sum'}).reset_index()
            fig_treemap = px.treemap(
                treemap_data,
                path=['TerritoryGroup'],
                values='SalesAmount',
                title="SalesAmount by Territory Groups",
                labels={'SalesAmount': 'Sales Amount (£)'}
            )
            fig_treemap.update_layout(margin=dict(t=30, l=0, r=0, b=0))
            st.plotly_chart(fig_treemap, use_container_width=True)
        else:
            st.warning("Territory Group data is not available.")

    # Line Graph: Number of SalesOrders vs Months
    with col6:
        st.write("**Number of SalesOrders vs Months**")
        sales_orders_monthly = df.groupby(df['OrderDate'].dt.month)['SalesOrderNumber'].nunique().reset_index()
        sales_orders_monthly.columns = ['Month', 'NumSalesOrders']
        fig_orders_months = px.line(
            sales_orders_monthly,
            x='Month',
            y='NumSalesOrders',
            title="Number of SalesOrders vs Months",
            labels={'NumSalesOrders': 'Number of SalesOrders', 'Month': 'Month'},
            markers=True
        )
        fig_orders_months.update_layout(
            xaxis=dict(
                tickmode="array",
                tickvals=list(range(1, 13)),
                ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            )
        )
        st.plotly_chart(fig_orders_months, use_container_width=True)

     # Row 2: Line Graph (Nos of SalesOrders vs Year) and Bar Chart (SalesAmount vs Country)
    col7, col8 = st.columns(2)
    
    # Line Graph: Number of SalesOrders vs Year
    with col7:
        st.write("**Number of SalesOrders vs Year**")
        sales_orders_yearly = df.groupby(df['OrderDate'].dt.year)['SalesOrderNumber'].nunique().reset_index()
        sales_orders_yearly.columns = ['Year', 'NumSalesOrders']
        fig_orders_years = px.line(
            sales_orders_yearly,
            x='Year',
            y='NumSalesOrders',
            title="Number of SalesOrders vs Year",
            labels={'NumSalesOrders': 'Number of SalesOrders', 'Year': 'Year'},
            markers=True
        )
        st.plotly_chart(fig_orders_years, use_container_width=True)

    # Vertical Bar Chart: SalesAmount vs Country
    with col8:
        st.write("**SalesAmount vs Country**")
        if 'Country' in df.columns:
            sales_country = df.groupby('Country').agg({'SalesAmount': 'sum'}).reset_index()
            fig_sales_country = px.bar(
                sales_country,
                x='Country',
                y='SalesAmount',
                title="SalesAmount by Country",
                labels={'SalesAmount': 'Sales Amount (£)', 'Country': 'Country'}
            )
            fig_sales_country.update_layout(yaxis_tickformat="£,.0f")
            st.plotly_chart(fig_sales_country, use_container_width=True)
        else:
            st.warning("Country data is not available.")

    # Row 3: Bar Chart (SalesAmount vs Currency) and Bar Chart (SalesAmount vs Region)
    col9, col10 = st.columns(2)
    
    # Bar Chart: SalesAmount vs Currency
    with col9:
        st.write("**SalesAmount vs Currency**")
        if 'Currency' in df.columns:
            sales_currency = df.groupby('Currency').agg({'SalesAmount': 'sum'}).reset_index()
            fig_sales_currency = px.bar(
                sales_currency,
                x='Currency',
                y='SalesAmount',
                title="SalesAmount by Currency",
                labels={'SalesAmount': 'Sales Amount (£)', 'Currency': 'Currency'}
            )
            fig_sales_currency.update_layout(yaxis_tickformat="£,.0f")
            st.plotly_chart(fig_sales_currency, use_container_width=True)
        else:
            st.warning("Currency data is not available.")

    # Bar Chart: SalesAmount vs Region
    with col10:
        st.write("**SalesAmount vs Region**")
        if 'Region' in df.columns:
            sales_region = df.groupby('Region').agg({'SalesAmount': 'sum'}).reset_index()
            fig_sales_region = px.bar(
                sales_region,
                x='Region',
                y='SalesAmount',
                title="SalesAmount by Region",
                labels={'SalesAmount': 'Sales Amount (£)', 'Region': 'Region'}
            )
            fig_sales_region.update_layout(yaxis_tickformat="£,.0f")
            st.plotly_chart(fig_sales_region, use_container_width=True)
        else:
            st.warning("Region data is not available.")

    # Row 1: Grouped Bar Chart and Bar Chart
    col11, col12 = st.columns(2)

    # Grouped Bar Chart: SalesAmount vs TerritoryGroup and PromotionCategory
    with col11:
        st.write("**SalesAmount vs TerritoryGroup and PromotionCategory**")
        if 'TerritoryGroup' in df.columns and 'PromotionCategory' in df.columns:
            sales_promotion_grouped = df.groupby(['TerritoryGroup', 'PromotionCategory']).agg({'SalesAmount': 'sum'}).reset_index()
            fig_grouped_bar = px.bar(
                sales_promotion_grouped,
                x='TerritoryGroup',
                y='SalesAmount',
                color='PromotionCategory',
                title="SalesAmount by TerritoryGroup and PromotionCategory",
                labels={'SalesAmount': 'Sales Amount (£)', 'TerritoryGroup': 'Territory Group'},
                barmode='group'
            )
            fig_grouped_bar.update_layout(yaxis_tickformat="£,.0f")
            st.plotly_chart(fig_grouped_bar, use_container_width=True)
        else:
            st.warning("Territory Group or Promotion Category data is not available.")

    # Bar Chart: Number of SalesOrders vs PromotionName
    with col12:
        st.write("**Number of SalesOrders vs PromotionName**")
        if 'PromotionName' in df.columns:
            sales_orders_promotion = df.groupby('PromotionName').agg({'SalesOrderNumber': 'nunique'}).reset_index()
            sales_orders_promotion.columns = ['PromotionName', 'NumSalesOrders']
            fig_bar_promotion_name = px.bar(
                sales_orders_promotion,
                x='PromotionName',
                y='NumSalesOrders',
                title="Number of SalesOrders by PromotionName",
                labels={'NumSalesOrders': 'Number of SalesOrders', 'PromotionName': 'Promotion Name'}
            )
            st.plotly_chart(fig_bar_promotion_name, use_container_width=True)
        else:
            st.warning("Promotion Name data is not available.")

# Row 2: Pie Chart and Doughnut Chart
    col13, col14 = st.columns(2)

    # Pie Chart: SalesAmount by PromotionName
    with col13:
        st.write("**SalesAmount by PromotionName**")
        if 'PromotionName' in df.columns:
            sales_amount_promotion = df.groupby('PromotionName').agg({'SalesAmount': 'sum'}).reset_index()
            sales_amount_promotion.columns = ['PromotionName', 'SalesAmount']
            fig_pie_promotion_name = px.pie(
                sales_amount_promotion,
                names='PromotionName',
                values='SalesAmount',
                title="SalesAmount by PromotionName",
                labels={'PromotionName': 'Promotion Name'},
                hole=0  # Regular pie chart
            )
            fig_pie_promotion_name.update_traces(
                textinfo='percent+label', 
                hovertemplate='Promotion: %{label}<br>Sales Amount: £%{value:,.2f}<extra></extra>'
            )
            st.plotly_chart(fig_pie_promotion_name, use_container_width=True)
        else:
            st.warning("Promotion Name data is not available.")

    # Doughnut Chart: SalesAmount by PromotionType
    with col14:
        st.write("**SalesAmount by PromotionType**")
        if 'PromotionType' in df.columns:
            sales_amount_promotion_type = df.groupby('PromotionType').agg({'SalesAmount': 'sum'}).reset_index()
            sales_amount_promotion_type.columns = ['PromotionType', 'SalesAmount']
            fig_doughnut_promotion_type = px.pie(
                sales_amount_promotion_type,
                names='PromotionType',
                values='SalesAmount',
                title="SalesAmount by PromotionType",
                labels={'PromotionType': 'Promotion Type'},
                hole=0.5  # Doughnut chart
            )
            fig_doughnut_promotion_type.update_traces(
                textinfo='percent+label', 
                hovertemplate='Promotion Type: %{label}<br>Sales Amount: £%{value:,.2f}<extra></extra>'
            )
            st.plotly_chart(fig_doughnut_promotion_type, use_container_width=True)
        else:
            st.warning("Promotion Type data is not available.")


def display_customer_insights():
    st.subheader("Customer Insights")
    
    def customer_metrics():
        from streamlit_extras.metric_cards import style_metric_cards
        card6, card7, card8, card9, card10 = st.columns(5)

        card6.metric("Sales", value=f"${df.SalesAmount.sum()/ 1e6:.1f}M", delta="Total Sales")
        card7.metric("Orders", value=f"{df.Profit.sum()/ 1e6:.1f}M", delta="Profit")
        card8.metric("Freight", value=f"${df.Freight.sum()/ 1e6:.1f}M", delta="Total Freight")
        card9.metric("Tax Amount", value=f"${df.TaxAmt.sum()/ 1e6:.1f}M", delta="Total Tax")
        card10.metric("Customers", value=f"{df.CustomerID.nunique()/1e3:.1f}K", delta="Customers")
        

        style_metric_cards(background_color="#dfdfdf", border_left_color="#1f66bd")
    customer_metrics()
    
    # Grouped Bar Chart: SalesAmount vs TerritoryGroup and Gender
    col1, col2 = st.columns(2)

    with col1:
        st.write("**SalesAmount by TerritoryGroup and Gender**")
        if 'TerritoryGroup' in df.columns and 'Gender' in df.columns:
            df['Gender'] = df['Gender'].replace({'M': 'Male', 'F': 'Female'})  # Replace M/F
            sales_gender = df.groupby(['TerritoryGroup', 'Gender']).agg({'SalesAmount': 'sum'}).reset_index()
            fig_gender_territory = px.bar(
                sales_gender,
                x='TerritoryGroup',
                y='SalesAmount',
                color='Gender',
                #title="SalesAmount by TerritoryGroup and Gender",
                labels={'SalesAmount': 'Sales Amount (£)', 'TerritoryGroup': 'Territory Group'},
                barmode='group'
            )
            fig_gender_territory.update_layout(yaxis_tickformat="£,.0f", legend_title="Gender")
            st.plotly_chart(fig_gender_territory, use_container_width=True)
        else:
            st.warning("Territory Group or Gender data is not available.")

    # Grouped Bar Chart: SalesAmount vs TerritoryGroup and Marital Status
    with col2:
        st.write("**SalesAmount by TerritoryGroup and Marital Status**")
        if 'TerritoryGroup' in df.columns and 'MaritalStatus' in df.columns:
            df['MaritalStatus'] = df['MaritalStatus'].replace({'M': 'Married', 'S': 'Single'})  # Replace M/S
            sales_marital = df.groupby(['TerritoryGroup', 'MaritalStatus']).agg({'SalesAmount': 'sum'}).reset_index()
            fig_marital_territory = px.bar(
                sales_marital,
                x='TerritoryGroup',
                y='SalesAmount',
                color='MaritalStatus',
                #title="SalesAmount by TerritoryGroup and Marital Status",
                labels={'SalesAmount': 'Sales Amount (£)', 'TerritoryGroup': 'Territory Group'},
                barmode='group'
            )
            fig_marital_territory.update_layout(yaxis_tickformat="£,.0f", legend_title="Marital Status")
            st.plotly_chart(fig_marital_territory, use_container_width=True)
        else:
            st.warning("Territory Group or Marital Status data is not available.")

    # Row 2: CustomerID Count vs Country and Number of SalesOrders vs Country
    col3, col4 = st.columns(2)

    # Bar Chart: CustomerID Count vs Country
    with col3:
        st.write("**CustomerID Count vs Country**")
        if 'Country' in df.columns:
            customer_country = df.groupby('Country').agg({'CustomerID': 'nunique'}).reset_index()
            #customer_country['Freight'] = customer_country['Freight'] / 1e6
            customer_country.columns = ['Country', 'CustomerCount']
            fig_customer_country = px.bar(
                customer_country,
                x='Country',
                y='CustomerCount',
                #title="Customer Count by Country",
                labels={'CustomerCount': 'Customer Count', 'Country': 'Country'}
            )
            st.plotly_chart(fig_customer_country, use_container_width=True)
        else:
            st.warning("Country data is not available.")

    # Bar Chart: Number of SalesOrders vs Country
    with col4:
        st.write("**Number of SalesOrders vs Country**")
        if 'Country' in df.columns:
            orders_country = df.groupby('Country').agg({'SalesOrderNumber': 'nunique'}).reset_index()
            orders_country.columns = ['Country', 'NumSalesOrders']
            fig_orders_country = px.bar(
                orders_country,
                x='Country',
                y='NumSalesOrders',
                #title="Number of SalesOrders by Country",
                labels={'NumSalesOrders': 'Number of SalesOrders', 'Country': 'Country'}
            )
            st.plotly_chart(fig_orders_country, use_container_width=True)
        else:
            st.warning("Country data is not available.")

    # Row 3: Top 10 CustomerNames by SalesAmount (Horizontal Bar Chart)
    st.write("**Top 10 CustomerNames by SalesAmount**")
    if 'CustomerName' in df.columns:
        top_customers = df.groupby('CustomerName').agg({'SalesAmount': 'sum'}).reset_index().sort_values(by='SalesAmount', ascending=False).head(10)
        fig_top_customers = px.bar(
            top_customers,
            x='SalesAmount',
            y='CustomerName',
            orientation='h',
            #title="Top 10 Customers by SalesAmount",
            labels={'SalesAmount': 'Sales Amount (£)', 'CustomerName': 'Customer Name'}
        )
        fig_top_customers.update_layout(xaxis_tickformat="£,.0f")
        st.plotly_chart(fig_top_customers, use_container_width=True)
    else:
        st.warning("Customer Name data is not available.")

    # Row 4: Freight and TaxAmount by Country
    col5, col6 = st.columns(2)

    # Bar Chart: Freight by Country
    with col5:
        st.write("**Freight by Country**")
        if 'Country' in df.columns and 'Freight' in df.columns:
            freight_country = df.groupby('Country').agg({'Freight': 'sum'}).reset_index()
            freight_country['Freight'] = freight_country['Freight'] / 1e6
            freight_country['Percentage'] = 100 * freight_country['Freight'] / freight_country['Freight'].sum()
            fig_freight_country = px.bar(
                freight_country,
                x='Country',
                y='Freight',
                #title="Freight by Country",
                text='Percentage',
                labels={'Freight': 'Freight (£) X 1M', 'Country': 'Country'}
            )
            fig_freight_country.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
            fig_freight_country.update_layout(yaxis_tickformat="£,.0f")
            st.plotly_chart(fig_freight_country, use_container_width=True)
        else:
            st.warning("Country or Freight data is not available.")

    # Bar Chart: TaxAmount by Country
    with col6:
        st.write("**TaxAmount by Country**")
        if 'Country' in df.columns and 'TaxAmt' in df.columns:
            tax_country = df.groupby('Country').agg({'TaxAmt': 'sum'}).reset_index()
            tax_country['TaxAmt'] = tax_country['TaxAmt'] / 1e6
            tax_country['Percentage'] = 100 * tax_country['TaxAmt'] / tax_country['TaxAmt'].sum()
            fig_tax_country = px.bar(
                tax_country,
                x='Country',
                y='TaxAmt',
                #title="TaxAmount by Country",
                text='Percentage',
                labels={'TaxAmt': 'Tax Amount (£) x 1M', 'Country': 'Country'}
            )
            fig_tax_country.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
            fig_tax_country.update_layout(yaxis_tickformat="£,.0f")
            st.plotly_chart(fig_tax_country, use_container_width=True)
        else:
            st.warning("Country or TaxAmount data is not available.")

    # Row: Age Group Charts
    col7, col8 = st.columns(2)

    # Grouped Bar Chart: SalesAmount vs AgeGroup by Gender
    with col7:
        st.write("**SalesAmount vs AgeGroup by Gender (in Millions)**")
        if 'Age' in df.columns and 'Gender' in df.columns:
            # Define age bins and create AgeGroup column
            age_bins = [0, 30, 40, 50, 60, 70, 100]
            df['AgeGroup'] = pd.cut(df['Age'], bins=age_bins, labels=['0-30', '30-40', '40-50', '50-60', '60-70', '70+'])
            
            # Group by AgeGroup and Gender
            sales_age_gender = df.groupby(['AgeGroup', 'Gender']).agg({'SalesAmount': 'sum'}).reset_index()
            sales_age_gender['SalesAmount'] = sales_age_gender['SalesAmount'] / 1e6  # Convert to millions
            
            # Create grouped bar chart
            fig_sales_age_gender = px.bar(
                sales_age_gender,
                x='AgeGroup',
                y='SalesAmount',
                color='Gender',
                #title="SalesAmount vs AgeGroup by Gender",
                labels={'SalesAmount': 'Sales Amount (Millions)', 'AgeGroup': 'Age Group'},
                barmode='group'
            )
            fig_sales_age_gender.update_layout(yaxis_tickformat=".2fM", legend_title="Gender")
            st.plotly_chart(fig_sales_age_gender, use_container_width=True)
        else:
            st.warning("Age or Gender data is not available.")

    # Grouped Bar Chart: Number of SalesOrders vs AgeGroup by Gender
    with col8:
        st.write("**Number of SalesOrders vs AgeGroup by Gender (in Thousands)**")
        if 'Age' in df.columns and 'Gender' in df.columns:
            sales_orders_age_gender = df.groupby(['AgeGroup', 'Gender']).agg({'SalesOrderNumber': 'nunique'}).reset_index()
            sales_orders_age_gender.columns = ['AgeGroup', 'Gender', 'NumSalesOrders']
            sales_orders_age_gender['NumSalesOrders'] = sales_orders_age_gender['NumSalesOrders'] / 1e3  # Convert to thousands
            
            fig_orders_age_gender = px.bar(
                sales_orders_age_gender,
                x='AgeGroup',
                y='NumSalesOrders',
                color='Gender',
                #title="Number of SalesOrders vs AgeGroup by Gender",
                labels={'NumSalesOrders': 'Number of SalesOrders (Thousands)', 'AgeGroup': 'Age Group'},
                barmode='group'
            )
            fig_orders_age_gender.update_layout(yaxis_tickformat=".1fK", legend_title="Gender")
            st.plotly_chart(fig_orders_age_gender, use_container_width=True)
        else:
            st.warning("Age or Gender data is not available.")

    # Row: TotalChildren Charts
    col1, col2 = st.columns(2)

    # Bar Chart: SalesAmount vs TotalChildren
    with col1:
        st.write("**SalesAmount vs TotalChildren**")
        if 'TotalChildren' in df.columns:
            total_children_sales = df.groupby('TotalChildren').agg({'SalesAmount': 'sum'}).reset_index()
            total_children_sales['SalesAmount'] = total_children_sales['SalesAmount'] / 1e6  # Convert to millions
            
            fig_sales_total_children = px.bar(
                total_children_sales,
                x='TotalChildren',
                y='SalesAmount',
                #title="SalesAmount vs TotalChildren",
                labels={'SalesAmount': 'Sales Amount (Millions)', 'TotalChildren': 'Total Children'}
            )
            fig_sales_total_children.update_layout(yaxis_tickformat=".2fM")
            st.plotly_chart(fig_sales_total_children, use_container_width=True)
        else:
            st.warning("TotalChildren data is not available.")

        # Bar Chart: Number of SalesOrders vs TotalChildren
        with col2:
            st.write("**Number of SalesOrders vs TotalChildren**")
            if 'TotalChildren' in df.columns:
                total_children_orders = df.groupby('TotalChildren').agg({'SalesOrderNumber': 'nunique'}).reset_index()
                total_children_orders.columns = ['TotalChildren', 'NumSalesOrders']
                total_children_orders['NumSalesOrders'] = total_children_orders['NumSalesOrders'] / 1e3  # Convert to thousands
            
                fig_orders_total_children = px.bar(
                    total_children_orders,
                    x='TotalChildren',
                    y='NumSalesOrders',
                    #title="Number of SalesOrders vs TotalChildren",
                    labels={'NumSalesOrders': 'Number of SalesOrders (Thousands)', 'TotalChildren': 'Total Children'}
                )
                fig_orders_total_children.update_layout(yaxis_tickformat=".1fK")
                st.plotly_chart(fig_orders_total_children, use_container_width=True)
            else:
                st.warning("TotalChildren data is not available.")

    # Add similar horizontal bar charts for Education, CommuteDistance, and Occupation as specified.

def display_product_insights():
    
    def product_metrics():
        from streamlit_extras.metric_cards import style_metric_cards
        card11, card12, card13, card14, card15 = st.columns(5)

        card11.metric("Sales", value=f"${df.SalesAmount.sum()/ 1e6:.1f}M", delta="Total Sales")
        card12.metric("Orders", value=f"{df.Profit.sum()/ 1e6:.1f}M", delta="Profit")
        card13.metric("Cost", value=f"${df.Cost.sum()/ 1e6:.1f}M", delta="Total Cost")
        card14.metric("Freight", value=f"${df.TaxAmt.sum()/ 1e6:.1f}M", delta="Total Freight")
        card15.metric("Product", value=f"{df.ProductName.nunique()/1e3:.1f}K", delta="Products")
        

        style_metric_cards(background_color="#dfdfdf", border_left_color="#1f66bd")
    product_metrics()
    
    st.subheader("Product Insights")
    # Row 1: SalesAmount and SalesOrderNumber vs ProductName
    col1, col2 = st.columns(2)

    # Horizontal Bar Chart: SalesAmount vs ProductName (Top 10)
    with col1:
        st.write("**Top 10 Products by SalesAmount**")
        if 'ProductName' in df.columns:
            product_sales = df.groupby('ProductName').agg({'SalesAmount': 'sum'}).reset_index().sort_values(by='SalesAmount', ascending=False).head(10)
            product_sales['SalesAmount'] = product_sales['SalesAmount'] / 1e6  # Convert to millions
            fig_sales_product = px.bar(
                product_sales,
                x='SalesAmount',
                y='ProductName',
                orientation='h',
                #title="Top 10 Products by SalesAmount",
                labels={'SalesAmount': 'Sales Amount (Millions)', 'ProductName': 'Product Name'}
            )
            fig_sales_product.update_layout(xaxis_tickformat=".2fM")
            st.plotly_chart(fig_sales_product, use_container_width=True)
        else:
            st.warning("Product Name data is not available.")

    # Horizontal Bar Chart: SalesOrderNumber vs ProductName (Top 10)
    with col2:
        st.write("**Top 10 Products by Number of SalesOrders**")
        if 'ProductName' in df.columns:
            product_orders = df.groupby('ProductName').agg({'SalesOrderNumber': 'nunique'}).reset_index().sort_values(by='SalesOrderNumber', ascending=False).head(10)
            product_orders['SalesOrderNumber'] = product_orders['SalesOrderNumber'] / 1e3  # Convert to thousands
            fig_orders_product = px.bar(
                product_orders,
                x='SalesOrderNumber',
                y='ProductName',
                orientation='h',
                #title="Top 10 Products by Number of SalesOrders",
                labels={'SalesOrderNumber': 'Number of SalesOrders (Thousands)', 'ProductName': 'Product Name'}
            )
            fig_orders_product.update_layout(xaxis_tickformat=".1fK")
            st.plotly_chart(fig_orders_product, use_container_width=True)
        else:
            st.warning("Product Name data is not available.")
    
    # Row 2: SalesAmount and SalesOrderNumber vs ProductCategory
    col3, col4 = st.columns(2)

    # Pie Chart: SalesAmount vs ProductCategory
    with col3:
        st.write("**SalesAmount by ProductCategory**")
        if 'ProductCategory' in df.columns:
            category_sales = df.groupby('ProductCategory').agg({'SalesAmount': 'sum'}).reset_index()
            fig_sales_category = px.pie(
                category_sales,
                names='ProductCategory',
                values='SalesAmount',
                #title="SalesAmount by ProductCategory",
                labels={'ProductCategory': 'Product Category'},
                hole=0
            )
            fig_sales_category.update_traces(textinfo='percent+label')
            st.plotly_chart(fig_sales_category, use_container_width=True)
        else:
            st.warning("Product Category data is not available.")

    # Doughnut Chart: SalesOrderNumber vs ProductCategory
    with col4:
        st.write("**Number of SalesOrders by ProductCategory**")
        if 'ProductCategory' in df.columns:
            category_orders = df.groupby('ProductCategory').agg({'SalesOrderNumber': 'nunique'}).reset_index()
            fig_orders_category = px.pie(
                category_orders,
                names='ProductCategory',
                values='SalesOrderNumber',
                #title="Number of SalesOrders by ProductCategory",
                labels={'ProductCategory': 'Product Category'},
                hole=0.5  # Doughnut chart
            )
            fig_orders_category.update_traces(textinfo='percent+label')
            st.plotly_chart(fig_orders_category, use_container_width=True)
        else:
            st.warning("Product Category data is not available.")

    # Row 3: SalesAmount and SalesOrderNumber vs ModelName
    col5, col6 = st.columns(2)

    # Horizontal Bar Chart: SalesAmount vs ModelName (Top 10)
    with col5:
        st.write("**Top 10 Models by SalesAmount**")
        if 'ModelName' in df.columns:
            model_sales = df.groupby('ModelName').agg({'SalesAmount': 'sum'}).reset_index().sort_values(by='SalesAmount', ascending=False).head(10)
            model_sales['SalesAmount'] = model_sales['SalesAmount'] / 1e6  # Convert to millions
            fig_sales_model = px.bar(
                model_sales,
                x='SalesAmount',
                y='ModelName',
                orientation='h',
                #title="Top 10 Models by SalesAmount",
                labels={'SalesAmount': 'Sales Amount (Millions)', 'ModelName': 'Model Name'}
            )
            fig_sales_model.update_layout(xaxis_tickformat=".2fM")
            st.plotly_chart(fig_sales_model, use_container_width=True)
        else:
            st.warning("Model Name data is not available.")

    # Horizontal Bar Chart: SalesOrderNumber vs ModelName (Top 10)
    with col6:
        st.write("**Top 10 Models by Number of SalesOrders**")
        if 'ModelName' in df.columns:
            model_orders = df.groupby('ModelName').agg({'SalesOrderNumber': 'nunique'}).reset_index().sort_values(by='SalesOrderNumber', ascending=False).head(10)
            model_orders['SalesOrderNumber'] = model_orders['SalesOrderNumber'] / 1e3  # Convert to thousands
            fig_orders_model = px.bar(
                model_orders,
                x='SalesOrderNumber',
                y='ModelName',
                orientation='h',
                #title="Top 10 Models by Number of SalesOrders",
                labels={'SalesOrderNumber': 'Number of SalesOrders (Thousands)', 'ModelName': 'Model Name'}
            )
            fig_orders_model.update_layout(xaxis_tickformat=".1fK")
            st.plotly_chart(fig_orders_model, use_container_width=True)
        else:
            st.warning("Model Name data is not available.")

    # Row 4: SalesAmount and Number of SalesOrders vs Color
    col7, col8 = st.columns(2)

    # Bar Chart: SalesAmount vs Color
    with col7:
        st.write("**SalesAmount by Color**")
        if 'Color' in df.columns:
            color_sales = df.groupby('Color').agg({'SalesAmount': 'sum'}).reset_index()
            color_sales['SalesAmount'] = color_sales['SalesAmount'] / 1e6  # Convert to millions
            fig_sales_color = px.bar(
                color_sales,
                x='Color',
                y='SalesAmount',
                text='SalesAmount',
                #title="SalesAmount by Color",
                labels={'SalesAmount': 'Sales Amount (Millions)', 'Color': 'Color'}
            )
            fig_sales_color.update_traces(texttemplate='%{text:.2f}M', textposition='outside')
            st.plotly_chart(fig_sales_color, use_container_width=True)
        else:
            st.warning("Color data is not available.")

    # Bar Chart: Number of SalesOrders vs Color
    with col8:
        st.write("**Number of SalesOrders by Color (in Thousands)**")
        if 'Color' in df.columns:
            color_orders = df.groupby('Color').agg({'SalesOrderNumber': 'nunique'}).reset_index()
            color_orders['SalesOrderNumber'] = color_orders['SalesOrderNumber'] / 1e3  # Convert to thousands
            fig_orders_color = px.bar(
                color_orders,
                x='Color',
                y='SalesOrderNumber',
                text='SalesOrderNumber',
                #title="Number of SalesOrders by Color",
                labels={'SalesOrderNumber': 'Number of SalesOrders (Thousands)', 'Color': 'Color'}
            )
            fig_orders_color.update_traces(texttemplate='%{text:.1f}K', textposition='outside')
            st.plotly_chart(fig_orders_color, use_container_width=True)
        else:
            st.warning("Color data is not available.")

    # Row 5: SalesAmount and Number of SalesOrders vs Class
    col9, col10 = st.columns(2)

    # Bar Chart: SalesAmount vs Class
    with col9:
        st.write("**SalesAmount by Class**")
        if 'Class' in df.columns:
            class_sales = df.groupby('Class').agg({'SalesAmount': 'sum'}).reset_index()
            class_sales['SalesAmount'] = class_sales['SalesAmount'] / 1e6  # Convert to millions
            fig_sales_class = px.bar(
                class_sales,
                x='Class',
                y='SalesAmount',
                text='SalesAmount',
                #title="SalesAmount by Class",
                labels={'SalesAmount': 'Sales Amount (Millions)', 'Class': 'Class'}
            )
            fig_sales_class.update_traces(texttemplate='%{text:.2f}M', textposition='outside')
            st.plotly_chart(fig_sales_class, use_container_width=True)
        else:
            st.warning("Class data is not available.")

    # Bar Chart: Number of SalesOrders vs Class
    with col10:
        st.write("**Number of SalesOrders by Class (in Thousands)**")
        if 'Class' in df.columns:
            class_orders = df.groupby('Class').agg({'SalesOrderNumber': 'nunique'}).reset_index()
            class_orders['SalesOrderNumber'] = class_orders['SalesOrderNumber'] / 1e3  # Convert to thousands
            fig_orders_class = px.bar(
                class_orders,
                x='Class',
                y='SalesOrderNumber',
                text='SalesOrderNumber',
                #title="Number of SalesOrders by Class",
                labels={'SalesOrderNumber': 'Number of SalesOrders (Thousands)', 'Class': 'Class'}
            )
            fig_orders_class.update_traces(texttemplate='%{text:.1f}K', textposition='outside')
            st.plotly_chart(fig_orders_class, use_container_width=True)
        else:
            st.warning("Class data is not available.")

    # Row 6: SalesAmount and Number of SalesOrders vs Style
    col11, col12 = st.columns(2)

    # Bar Chart: SalesAmount vs Style
    with col11:
        st.write("**SalesAmount by Style**")
        if 'Style' in df.columns:
            style_sales = df.groupby('Style').agg({'SalesAmount': 'sum'}).reset_index()
            style_sales['SalesAmount'] = style_sales['SalesAmount'] / 1e6  # Convert to millions
            fig_sales_style = px.bar(
                style_sales,
                x='Style',
                y='SalesAmount',
                text='SalesAmount',
                #title="SalesAmount by Style",
                labels={'SalesAmount': 'Sales Amount (Millions)', 'Style': 'Style'}
            )
            fig_sales_style.update_traces(texttemplate='%{text:.2f}M', textposition='outside')
            st.plotly_chart(fig_sales_style, use_container_width=True)
        else:
            st.warning("Style data is not available.")

    # Bar Chart: Number of SalesOrders vs Style
    with col12:
        st.write("**Number of SalesOrders by Style (in Thousands)**")
        if 'Style' in df.columns:
            style_orders = df.groupby('Style').agg({'SalesOrderNumber': 'nunique'}).reset_index()
            style_orders['SalesOrderNumber'] = style_orders['SalesOrderNumber'] / 1e3  # Convert to thousands
            fig_orders_style = px.bar(
                style_orders,
                x='Style',
                y='SalesOrderNumber',
                text='SalesOrderNumber',
                #title="Number of SalesOrders by Style",
                labels={'SalesOrderNumber': 'Number of SalesOrders (Thousands)', 'Style': 'Style'}
            )
            fig_orders_style.update_traces(texttemplate='%{text:.1f}K', textposition='outside')
            st.plotly_chart(fig_orders_style, use_container_width=True)
        else:
            st.warning("Style data is not available.")

    # Row 7: SalesAmount and Number of SalesOrders vs ProductSubCategory
    col13, col14 = st.columns(2)

    # Horizontal Bar Chart: SalesAmount vs ProductSubCategory (Top 10)
    with col13:
        st.write("**Top 10 ProductSubCategories by SalesAmount**")
        if 'ProductSubcategory' in df.columns:
            subcategory_sales = df.groupby('ProductSubcategory').agg({'SalesAmount': 'sum'}).reset_index().sort_values(by='SalesAmount', ascending=False).head(10)
            subcategory_sales['SalesAmount'] = subcategory_sales['SalesAmount'] / 1e6  # Convert to millions
            fig_sales_subcategory = px.bar(
                subcategory_sales,
                x='SalesAmount',
                y='ProductSubcategory',
                orientation='h',
                #title="Top 10 ProductSubCategories by SalesAmount",
                labels={'SalesAmount': 'Sales Amount (Millions)', 'ProductSubcategory': 'Product SubCategory'}
            )
            fig_sales_subcategory.update_layout(xaxis_tickformat=".2fM")
            st.plotly_chart(fig_sales_subcategory, use_container_width=True)
        else:
            st.warning("ProductSubCategory data is not available.")

    # Horizontal Bar Chart: SalesOrderNumber vs ProductSubCategory (Top 10)
    with col14:
        st.write("**Top 10 ProductSubCategories by Number of SalesOrders**")
        if 'ProductSubcategory' in df.columns:
            subcategory_orders = df.groupby('ProductSubcategory').agg({'SalesOrderNumber': 'nunique'}).reset_index().sort_values(by='SalesOrderNumber', ascending=False).head(10)
            subcategory_orders['SalesOrderNumber'] = subcategory_orders['SalesOrderNumber'] / 1e3  # Convert to thousands
            fig_orders_subcategory = px.bar(
                subcategory_orders,
                x='SalesOrderNumber',
                y='ProductSubcategory',
                orientation='h',
                #title="Top 10 ProductSubCategories by Number of SalesOrders",
                labels={'SalesOrderNumber': 'Number of SalesOrders (Thousands)', 'ProductSubcategory': 'Product SubCategory'}
            )
            fig_orders_subcategory.update_layout(xaxis_tickformat=".1fK")
            st.plotly_chart(fig_orders_subcategory, use_container_width=True)
        else:
            st.warning("ProductSubcategory data is not available.")


def display_segmentation():
    st.subheader("Customer Segmentation")
    if not df.empty:
        # Select features for segmentation
        segmentation_features = st.multiselect(
        "Select features for segmentation",
        options=['Age', 'SalesAmount', 'Freight', 'Profit'],
        default=['Age', 'SalesAmount']
    )
    
    # Check if there are selected features
    if segmentation_features:
        # Scale features
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df[segmentation_features].fillna(0))
        
        # Perform K-Means clustering
        num_clusters = st.slider("Number of Clusters", min_value=2, max_value=10, value=4)
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        df['Segment'] = kmeans.fit_predict(scaled_data)
        
        # Visualize customer segments with scatter plot
        fig_segment = px.scatter(
            df,
            x=segmentation_features[0],
            y=segmentation_features[1] if len(segmentation_features) > 1 else segmentation_features[0],
            color='Segment',
            title="Customer Segments",
            labels={'Segment': 'Customer Segment'},
            hover_data=['CustomerID', 'SalesAmount', 'Profit', 'Age']
        )
        
        st.plotly_chart(fig_segment, use_container_width=True)

        # Display average metrics for each segment in a table
        st.subheader("Segment Analysis Table")
        segment_analysis = df.groupby('Segment').agg({
            'SalesAmount': 'mean',
            'Profit': 'mean',
            'Age': 'mean',
            'Freight': 'mean'
        }).reset_index()
        segment_analysis.columns = ['Segment', 'Avg SalesAmount', 'Avg Profit', 'Avg Age', 'Avg Freight']
        
        st.dataframe(segment_analysis)

        # Add a bar chart to visualize average metrics by segment
        st.subheader("Segment Insights Visualization")
        fig_segment_metrics = px.bar(
            segment_analysis,
            x='Segment',
            y=['Avg SalesAmount', 'Avg Profit', 'Avg Age', 'Avg Freight'],
            title="Average Metrics by Customer Segment",
            labels={'value': 'Average Value', 'variable': 'Metrics'},
            barmode='group'
        )
        st.plotly_chart(fig_segment_metrics, use_container_width=True)
    else:
        st.warning("Please select at least one feature for segmentation.")

    # Interpretation of segmentation's effect on sales insights
    with st.expander("How Customer Segmentation Affects Sales Insights"):
        st.write("""
        **Customer segmentation** is a powerful tool that can help you better understand your customers and tailor strategies for each group. Here’s how segmentation can impact sales insights:

        1. **Personalized Marketing and Promotions**:
            - By understanding the unique needs and behaviors of each segment, businesses can tailor their marketing strategies to be more effective.
            - For example, if one segment shows high sales and engagement with promotions, you can target similar promotions to that segment, increasing conversion rates and customer satisfaction.

        2. **Product Recommendations**:
            - Segmentation can reveal which products are more popular with certain customer groups.
            - For instance, a segment with younger customers might prefer certain product categories, while another segment with higher average spending might be more interested in premium products.
            - Knowing this can help optimize inventory and drive targeted product recommendations.

        3. **Pricing Strategy**:
            - Segments with different spending patterns may respond differently to pricing strategies. For example, price-sensitive segments might be more responsive to discounts, while premium segments might value quality over price.
            - Understanding these patterns allows businesses to implement pricing strategies that maximize profitability without alienating certain customer groups.

        4. **Resource Allocation**:
            - By identifying high-value segments (e.g., those with higher average sales or profit), businesses can prioritize resources towards nurturing relationships with those segments.
            - This could involve providing premium support, exclusive offers, or personalized experiences to retain high-value customers and increase their lifetime value.

        5. **Customer Retention**:
            - Analyzing customer segments helps identify groups with high churn rates, allowing the business to implement targeted retention strategies.
            - For example, if a segment with lower engagement and sales is at risk of churning, the business can reach out with incentives, loyalty programs, or targeted messaging to retain those customers.

        6. **Strategic Business Decisions**:
            - By observing trends within each segment, companies can make data-driven decisions about product development, new market opportunities, or business expansion.
            - For instance, if one segment shows high demand for a specific product type, it might be worth expanding that product line or introducing similar products.

        """)

def display_clv_prediction():
    st.subheader("Customer Lifetime Value (CLV) Prediction")
    if not df.empty:
        st.subheader("Customer Lifetime Value (CLV) Prediction")

        # Define CLV calculation based on historical data
        # Here we estimate CLV as (average purchase value) * (purchase frequency) * (expected customer lifespan)
        
        # Average purchase value (SalesAmount per customer)
        avg_purchase_value = df.groupby('CustomerID')['SalesAmount'].mean()
        
        # Purchase frequency (total orders per customer)
        purchase_frequency = df.groupby('CustomerID')['SalesOrderNumber'].nunique()
        
        # Assuming an arbitrary customer lifespan (in years)
        avg_customer_lifespan = 3  # You can adjust this based on your domain knowledge
        
        # Calculate CLV
        df['CLV'] = df['CustomerID'].map(avg_purchase_value) * df['CustomerID'].map(purchase_frequency) * avg_customer_lifespan
        
        # Display top customers by CLV
        st.subheader("Top Customers by CLV")
        top_customers = df.groupby('CustomerID')['CLV'].mean().sort_values(ascending=False).head(10).reset_index()
        st.dataframe(top_customers)

        # Visualization: CLV by Segment
        st.subheader("CLV by Customer Segment")
        if 'Segment' in df.columns:
            clv_segment = df.groupby('Segment')['CLV'].mean().reset_index()
            fig_clv_segment = px.bar(clv_segment, x='Segment', y='CLV', title="Average CLV by Customer Segment")
            st.plotly_chart(fig_clv_segment, use_container_width=True)
        else:
            st.warning("Customer Segmentation is required to view CLV by Segment.")

        # Interpretation of CLV Prediction
        with st.expander("Understanding Customer Lifetime Value (CLV) Prediction"):
            st.write("""
            **Customer Lifetime Value (CLV)** is a predictive metric that estimates the total revenue a customer is likely to bring to a business over their lifetime. By analyzing past purchasing behavior, we can estimate each customer's CLV, which allows us to make informed business decisions. Here’s how CLV prediction can impact business strategy:

            1. **Prioritizing High-Value Customers**:
                - By identifying customers with high CLV, businesses can focus resources on retaining these valuable customers through personalized offers, loyalty programs, or premium support.

            2. **Optimizing Marketing Spend**:
                - With CLV prediction, businesses can tailor marketing efforts to maximize ROI. For example, spending more to acquire and retain high-CLV customers can yield greater returns than treating all customers equally.

            3. **Personalizing Customer Experience**:
                - By understanding each customer’s potential lifetime value, businesses can personalize interactions based on their value. High-CLV customers might receive exclusive offers or access to premium services.

            4. **Strategic Product Development**:
                - CLV can inform product and service development. For example, if high-CLV customers are mostly purchasing specific products, businesses can focus on expanding or enhancing these products.

            
            """)
    else:
        st.warning("No data available for CLV Prediction.")

def display_forecasting():
    st.subheader("Sales Forecasting")
    if not df.empty:
        df_monthly = df.resample('M', on='OrderDate').sum().reset_index()[['OrderDate', 'SalesAmount']]
        df_monthly.columns = ['ds', 'y']
        model = Prophet(yearly_seasonality=True)
        model.fit(df_monthly)
        future_dates = model.make_future_dataframe(periods=12, freq='M')
        forecast = model.predict(future_dates)
        fig = plot_plotly(model, forecast)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No data available for Forecasting.")

# Render the selected section
if menu == "Home":
    display_home()
elif menu == "Sales Insights":
    display_sales_insights()
elif menu == "Customer Insights":
    display_customer_insights()
elif menu == "Product Insights":
    display_product_insights()
elif menu == "Segmentation":
    display_segmentation()
elif menu == "CLV Prediction":
    display_clv_prediction()
elif menu == "Forecasting":
    display_forecasting()