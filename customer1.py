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
    try:
        dfc = pd.read_csv('data/InternetSales.csv', encoding='latin1')
        st.info("Using default dataset.")
    except Exception as e:
        st.error(f"Error loading default dataset: {e}")
        dfc = pd.DataFrame()  # Fallback to an empty DataFrame

# Check if necessary columns exist
if not dfc.empty:
    if 'SalesAmount' in dfc.columns and 'Cost' in dfc.columns:
        dfc['Profit'] = dfc['SalesAmount'] - dfc['Cost']
    else:
        st.error("The columns 'SalesAmount' or 'Cost' are missing from the dataset.")

    # Ensure OrderDate is parsed as datetime and calculate min/max dates
    if 'OrderDate' in dfc.columns:
        dfc['OrderDate'] = pd.to_datetime(dfc['OrderDate'], errors='coerce')
        min_date = dfc['OrderDate'].min()
        max_date = dfc['OrderDate'].max()

        if pd.notnull(min_date) and pd.notnull(max_date):
            # Sidebar separate date inputs
            st.sidebar.header("Date Range Filter")
            start_date = st.sidebar.date_input(
                "Start Date", value=min_date, min_value=min_date, max_value=max_date
            )
            end_date = st.sidebar.date_input(
                "End Date", value=max_date, min_value=min_date, max_value=max_date
            )

            # Filter data based on selected start and end dates
            filtered_df = dfc[
                (dfc['OrderDate'] >= pd.to_datetime(start_date)) &
                (dfc['OrderDate'] <= pd.to_datetime(end_date))
            ]

            # Extract Month and Year from OrderDate
            filtered_df['Month'] = filtered_df['OrderDate'].dt.month
            filtered_df['Year'] = filtered_df['OrderDate'].dt.year
        else:
            st.error("Invalid dates in 'OrderDate' column.")
            filtered_df = dfc
    else:
        st.error("The column 'OrderDate' is missing from the dataset.")
        filtered_df = dfc

    # Sidebar additional filters if dataset is valid
    st.sidebar.header("Additional Filters")
    if not filtered_df.empty:
        if 'ProductCategory' in filtered_df.columns:
            product_category = st.sidebar.selectbox(
                "Product Category", filtered_df['ProductCategory'].unique(), index=0
            )
        else:
            product_category = None

        if 'Country' in filtered_df.columns:
            sales_country = st.sidebar.selectbox(
                "Country", filtered_df['Country'].unique(), index=0
            )
        else:
            sales_country = None

        if 'Color' in filtered_df.columns:
            color = st.sidebar.selectbox(
                "Color", filtered_df['Color'].unique(), index=0
            )
        else:
            color = None

        if 'Style' in filtered_df.columns:
            style = st.sidebar.selectbox(
                "Style", filtered_df['Style'].unique(), index=0
            )
        else:
            style = None

        if 'Region' in filtered_df.columns:
            region = st.sidebar.selectbox(
                "Region", filtered_df['Region'].unique(), index=0
            )
        else:
            region = None

        # Apply filters to the DataFrame
        final_filtered_df = filtered_df
        if product_category:
            final_filtered_df = final_filtered_df[
                final_filtered_df['ProductCategory'] == product_category
            ]
        if sales_country:
            final_filtered_df = final_filtered_df[
                final_filtered_df['Country'] == sales_country
            ]
        if color:
            final_filtered_df = final_filtered_df[
                final_filtered_df['Color'] == color
            ]
        if style:
            final_filtered_df = final_filtered_df[
                final_filtered_df['Style'] == style
            ]
        if region:
            final_filtered_df = final_filtered_df[
                final_filtered_df['Region'] == region
            ]

        # Display the final filtered DataFrame
        st.write("Filtered Data", final_filtered_df)
    else:
        st.error("Filtered dataset is empty.")
else:
    st.error("Dataset is empty or invalid.")

def Home():
    st.subheader("Filtered Sales Data")
    with st.expander("Tabular"):
        showData = st.multiselect('Select columns to display:', final_filtered_df.columns, default=final_filtered_df.columns)
        if showData:
            st.write(final_filtered_df[showData])
        else:
            st.write("No columns selected for display. Please select columns from the dropdown.")

# Display the Home function content
Home()

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

# # Display key metrics with icons
# st.subheader("Key Metrics")
# col1, col2, col3, col4, col5 = st.columns(5)
# with col1:
#     st.markdown(f"<div class='embossed-card'><div class='icon icon-sales'>💰</div><h3>Revenue</h3><p>${total_sales:,.2f}</p></div>", unsafe_allow_html=True)    
# with col2:
#     st.markdown(f"<div class='embossed-card'><div class='icon icon-profit'>📈</div><h3>Profit</h3><p>${total_profit:,.2f}</p></div>", unsafe_allow_html=True)    
# with col3:
#     st.markdown(f"<div class='embossed-card'><div class='icon icon-customers'>👥</div><h3>Customers</h3><p>{total_customers}</p></div>", unsafe_allow_html=True)    
# with col4:
#     st.markdown(f"<div class='embossed-card'><div class='icon icon-products'>📦</div><h3>Products</h3><p>{total_products}</p></div>", unsafe_allow_html=True)    
# with col5:
#     st.markdown(f"<div class='embossed-card'><div class='icon icon-countries'>🌍</div><h3>Countries</h3><p>{total_countries}</p></div>", unsafe_allow_html=True)
selected_section = st.selectbox(
    "Select a section to view",
    [
        "Sales Insights",
        "Customer Insights",
        "Product Insights",
        "Customer Segmentation",
        "Customer Segmentation & Predictive Profiling",
        "CLV Prediction",
        "Sales Forecasting"
    ]
)

def metrics():
    from streamlit_extras.metric_cards import style_metric_cards
    card1, card2, card3, card4, card5 = st.columns(5)

    card1.metric("Sales", value=f"${filtered_df.SalesAmount.sum()/ 1e6:.1f}M", delta="Total Sales")
    card2.metric("Orders", value=f"{filtered_df.Profit.sum()/ 1e6:.1f}M", delta="Profit")
    card3.metric("Resellers", value=f"{filtered_df.CustomerID.nunique()/1e3:.1f}K", delta="Customers")
    card4.metric("Products", value=f"{filtered_df.ProductName.nunique()}", delta="Products")
    card5.metric("Countries", value=filtered_df.Country.nunique(), delta="Countries")

    style_metric_cards(background_color="#071021", border_left_color="#1f66bd")
        
metrics()


# Tabs for detailed insights
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Sales Insights", "Customer Insights", "Product Insights", "Customer Segmentation", "Segmentation Predictive Profiling", "CLV Prediction", "Sales Forecast"])

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


# Customer Segmentation Tab
with tab4:
    st.subheader("Customer Segmentation")
    
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
        scaled_data = scaler.fit_transform(filtered_df[segmentation_features].fillna(0))
        
        # Perform K-Means clustering
        num_clusters = st.slider("Number of Clusters", min_value=2, max_value=10, value=4)
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        filtered_df['Segment'] = kmeans.fit_predict(scaled_data)
        
        # Visualize customer segments with scatter plot
        fig_segment = px.scatter(
            filtered_df,
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
        segment_analysis = filtered_df.groupby('Segment').agg({
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

        **In summary**, segmentation transforms raw sales data into actionable insights, allowing businesses to optimize marketing, pricing, and product strategies based on the characteristics of each customer group. This approach increases sales, improves customer loyalty, and supports more strategic, data-driven decision-making.
        """)

        # Customer Segmentation & Predictive Profiling Tab

    with tab5:
        st.subheader("Customer Segmentation & Predictive Profiling")

        # Define high spender threshold
        high_spender_threshold = st.number_input("Define High Spender Threshold", min_value=1000, max_value=10000, value=3000)

        # Add 'HighSpender' column for predictive profiling target
        filtered_df['HighSpender'] = (filtered_df['SalesAmount'] > high_spender_threshold).astype(int)

        # Segmentation setup
        segmentation_features = st.multiselect(
            "Select features for segmentation",
            options=['Age', 'SalesAmount', 'Freight', 'Profit'],
            default=['Age', 'SalesAmount'],
            key="segmentation_features"
        )

        if segmentation_features:
            # Scale features for segmentation
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(filtered_df[segmentation_features].fillna(0))

            # Perform K-Means clustering with a unique key for the slider
            num_clusters = st.slider("Number of Clusters", min_value=2, max_value=10, value=4, key="num_clusters_slider")
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            filtered_df['Segment'] = kmeans.fit_predict(scaled_data)

            # Predictive Modeling for High Spenders
            st.subheader("Predictive Profiling of High Spenders by Segment")

            # Select features for predictive profiling
            predictive_features = st.multiselect(
                "Select features for predictive profiling",
                options=segmentation_features + ['Segment'],
                default=segmentation_features + ['Segment'],
                key="predictive_features"
            )

            # Check if there are selected predictive features
            if predictive_features:
                # Prepare data for model training
                X = filtered_df[predictive_features].fillna(0)
                y = filtered_df['HighSpender']

                # Train/Test split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

                # Train the Random Forest model
                model = RandomForestClassifier(random_state=42)
                model.fit(X_train, y_train)

                # Predict probabilities for high spending within each segment
                filtered_df['HighSpender_Probability'] = model.predict_proba(X)[:, 1]  # Probability of being a high spender

                # Display average probability of high spenders for each segment
                segment_profiling = filtered_df.groupby('Segment').agg({
                    'HighSpender_Probability': 'mean',
                    'SalesAmount': 'mean',
                    'Profit': 'mean',
                    'Age': 'mean',
                    'Freight': 'mean'
                }).reset_index()
                segment_profiling.columns = ['Segment', 'Avg HighSpender Probability', 'Avg SalesAmount', 'Avg Profit', 'Avg Age', 'Avg Freight']

                st.dataframe(segment_profiling)

                # Explanation of the High Spender Chart
                st.markdown("""
                    **Explanation of High Spender Probability Chart:**
                    The bar chart below shows the average probability of customers being high spenders across different segments.
                    Segments with a higher probability indicate groups more likely to contain high spenders, which may be valuable for targeted marketing and retention efforts.
                """)

                # Visualize predictive profiling results by segment
                fig = px.bar(
                    segment_profiling,
                    x='Segment',
                    y='Avg HighSpender Probability',
                    title="Average Probability of High Spending by Segment",
                    labels={'Avg HighSpender Probability': 'Avg High Spender Probability'},
                    text='Avg HighSpender Probability'
                )
                fig.update_traces(texttemplate='%{text:.2%}', textposition='outside')
                st.plotly_chart(fig, use_container_width=True)

                # Export predictive profiling results to CSV
                csv_profiling = segment_profiling.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Predictive Profiling Results as CSV",
                    data=csv_profiling,
                    file_name="predictive_profiling_by_segment.csv",
                    mime="text/csv"
                )

                # Display High and Low Spenders
                st.subheader("High and Low Spenders")

                # High Spenders
                high_spenders_df = filtered_df[filtered_df['HighSpender'] == 1]
                st.write("**Top 5 High Spenders**")
                st.dataframe(high_spenders_df[['CustomerID', 'SalesAmount', 'Profit', 'Age', 'Country', 'ProductCategory']].head(5).astype({'CustomerID': str}))

                # Export full list of high spenders
                csv_high_spenders = high_spenders_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Full High Spenders List as CSV",
                    data=csv_high_spenders,
                    file_name="high_spenders_list.csv",
                    mime="text/csv"
                )

                # Low Spenders
                low_spenders_df = filtered_df[filtered_df['HighSpender'] == 0]
                st.write("**Top 5 Low Spenders**")
                st.dataframe(low_spenders_df[['CustomerID', 'SalesAmount', 'Profit', 'Age', 'Country', 'ProductCategory']].head(5).astype({'CustomerID': str}))

                # Export full list of low spenders
                csv_low_spenders = low_spenders_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Full Low Spenders List as CSV",
                    data=csv_low_spenders,
                    file_name="low_spenders_list.csv",
                    mime="text/csv"
                )

                # Comparison of High vs Low Spenders
                st.subheader("Comparison of High vs Low Spenders")
                spender_comparison = filtered_df.groupby('HighSpender').agg({
                    'SalesAmount': 'mean',
                    'Profit': 'mean'
                }).reset_index()
                spender_comparison['Spender Type'] = spender_comparison['HighSpender'].map({1: 'High Spender', 0: 'Low Spender'})

                fig_spender_comparison = px.bar(
                    spender_comparison,
                    x='Spender Type',
                    y=['SalesAmount', 'Profit'],
                    title="Average Sales and Profit Comparison: High vs Low Spenders",
                    labels={'value': 'Average Amount', 'variable': 'Metric'},
                    barmode='group'
                )
                st.plotly_chart(fig_spender_comparison, use_container_width=True)

                # Demographics Breakdown by Segment
                st.subheader("Demographics Breakdown by Segment")
                demographics_by_segment = filtered_df.groupby(['Segment', 'Country']).agg({
                    'Age': 'mean',
                    'SalesAmount': 'mean',
                    'Profit': 'mean'
                }).reset_index()

                st.dataframe(demographics_by_segment)

                

            else:
                st.warning("Please select at least one feature for predictive profiling.")
        else:
            st.warning("Please select at least one feature for segmentation.")

        st.markdown("""
            ### Understanding High Spenders and Low Spenders

            **High Spenders** and **Low Spenders** represent different customer groups with unique behaviors and potential impacts on the business. Here’s a breakdown of the implications of each group and the business value of understanding them:

            #### High Spenders
            - **Characteristics**: High spenders are customers whose purchases exceed a set threshold. They often contribute a significant portion of revenue and tend to be more engaged with premium products or services.
            - **Business Value**: High spenders are invaluable for revenue generation and can be more receptive to cross-selling and upselling. They often show brand loyalty and are less sensitive to price changes.
            - **Recommended Actions**:
            1. **Personalized Offers**: Use targeted marketing campaigns to offer exclusive discounts, early access to new products, or premium loyalty rewards to retain high spenders.
            2. **Upselling and Cross-Selling**: Leverage data on high spenders’ preferences to suggest complementary products or higher-value items.
            3. **Customer Retention**: Invest in customer retention strategies such as loyalty programs, personalized service, or VIP experiences to ensure these valuable customers stay engaged with your brand.

            #### Low Spenders
            - **Characteristics**: Low spenders are customers with spending below the defined high-spender threshold. They might engage less frequently or purchase lower-cost items, making them more price-sensitive.
            - **Business Value**: While individually they contribute less revenue, low spenders can represent a large portion of the customer base. Converting even a small percentage of low spenders into higher spenders can significantly boost overall revenue.
            - **Recommended Actions**:
            1. **Engagement Campaigns**: Use incentives, such as discounts or loyalty points, to increase purchase frequency and spending among low spenders.
            2. **Educational Content**: Provide content that educates low spenders on the value of your premium products or services to encourage them to upgrade their purchases.
            3. **Identify Growth Opportunities**: Analyze the behavior of low spenders to identify potential growth segments. For example, if low spenders prefer certain product types, consider developing a targeted strategy for those products to increase engagement.

            #### Summary of Business Value
            Identifying high and low spenders allows you to:
            - **Optimize Marketing Spend**: Allocate resources toward high-value customers while implementing cost-effective engagement strategies for low spenders.
            - **Increase Revenue and Profitability**: By focusing on converting low spenders and retaining high spenders, you can boost total revenue and enhance long-term customer loyalty.
            - **Data-Driven Strategy**: Use these insights to shape product development, marketing, and customer service strategies, aligning business goals with customer behavior patterns.

            By understanding these groups, businesses can create more targeted, data-driven strategies to maximize customer lifetime value, drive revenue, and improve customer satisfaction.
            """)
    
    # Customer Lifetime Value (CLV) Prediction Tab
#with st.tabs(["Sales Insights", "Customer Insights", "Product Insights", "Customer Segmentation", "CLV Prediction"]) as (tab1, tab2, tab3, tab4, tab5, tab6):
    with tab6:
        st.subheader("Customer Lifetime Value (CLV) Prediction")

        # Define CLV calculation based on historical data
        # Here we estimate CLV as (average purchase value) * (purchase frequency) * (expected customer lifespan)
        
        # Average purchase value (SalesAmount per customer)
        avg_purchase_value = filtered_df.groupby('CustomerID')['SalesAmount'].mean()
        
        # Purchase frequency (total orders per customer)
        purchase_frequency = filtered_df.groupby('CustomerID')['SalesOrderNumber'].nunique()
        
        # Assuming an arbitrary customer lifespan (in years)
        avg_customer_lifespan = 3  # You can adjust this based on your domain knowledge
        
        # Calculate CLV
        filtered_df['CLV'] = filtered_df['CustomerID'].map(avg_purchase_value) * filtered_df['CustomerID'].map(purchase_frequency) * avg_customer_lifespan
        
        # Display top customers by CLV
        st.subheader("Top Customers by CLV")
        top_customers = filtered_df.groupby('CustomerID')['CLV'].mean().sort_values(ascending=False).head(10).reset_index()
        st.dataframe(top_customers)

        # Visualization: CLV by Segment
        st.subheader("CLV by Customer Segment")
        if 'Segment' in filtered_df.columns:
            clv_segment = filtered_df.groupby('Segment')['CLV'].mean().reset_index()
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

    # Sales Forecasting Tab
# with st.tabs(["Sales Insights", "Customer Insights", "Product Insights", "Customer Segmentation", "CLV Prediction", "Sales Forecasting"]) as (tab1, tab2, tab3, tab4, tab5, tab6, tab7):
    with tab7:
        
        st.subheader("Sales Forecasting for the Next 5 Years")

        # Preprocess data for forecasting: aggregate monthly sales
        df_sales_monthly = filtered_df.resample('M', on='OrderDate').sum().reset_index()[['OrderDate', 'SalesAmount']]
        df_sales_monthly.columns = ['ds', 'y']  # Prophet requires these column names

        # Build the Prophet model
        model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        model.fit(df_sales_monthly)

        # Create future dates dataframe for the next 5 years
        future_dates = model.make_future_dataframe(periods=60, freq='M')  # 5 years of monthly predictions
        forecast = model.predict(future_dates)

        # Monthly Forecast Plot
        st.subheader("Monthly Sales Forecast")
        fig_monthly = plot_plotly(model, forecast)
        fig_monthly.update_layout(title="Monthly Sales Forecast for 5 Years", xaxis_title="Date", yaxis_title="Sales Amount")
        st.plotly_chart(fig_monthly, use_container_width=True)

        # Quarterly Forecast
        st.subheader("Quarterly Sales Forecast")
        forecast['quarter'] = forecast['ds'].dt.to_period('Q').astype(str)  # Convert Period to string
        quarterly_forecast = forecast.groupby('quarter')['yhat'].sum().reset_index()
        fig_quarterly = px.bar(quarterly_forecast, x='quarter', y='yhat', title="Quarterly Sales Forecast for 5 Years")
        fig_quarterly.update_layout(xaxis_title="Quarter", yaxis_title="Predicted Sales")
        st.plotly_chart(fig_quarterly, use_container_width=True)

        # Half-Year Forecast
        st.subheader("Half-Yearly Sales Forecast")
        forecast['half_year'] = forecast['ds'].apply(lambda x: f"{x.year} H1" if x.month <= 6 else f"{x.year} H2")
        half_year_forecast = forecast.groupby('half_year')['yhat'].sum().reset_index()
        fig_half_year = px.bar(half_year_forecast, x='half_year', y='yhat', title="Half-Yearly Sales Forecast for 5 Years")
        fig_half_year.update_layout(xaxis_title="Half-Year", yaxis_title="Predicted Sales")
        st.plotly_chart(fig_half_year, use_container_width=True)

        # Annual Forecast
        st.subheader("Annual Sales Forecast")
        forecast['year'] = forecast['ds'].dt.year
        annual_forecast = forecast.groupby('year')['yhat'].sum().reset_index()
        fig_annual = px.bar(annual_forecast, x='year', y='yhat', title="Annual Sales Forecast for 5 Years")
        fig_annual.update_layout(xaxis_title="Year", yaxis_title="Predicted Sales")
        st.plotly_chart(fig_annual, use_container_width=True)

        # Interpretation of Sales Forecast
        with st.expander("Understanding Sales Forecasting and 5-Year Projections"):
            st.write("""
            **Sales Forecasting** provides predictions of future sales by analyzing historical data, seasonality, and trends. Extending the forecast over a 5-year period allows businesses to plan long-term, making informed decisions about resources, budgeting, and market strategies. Key benefits of a 5-year projection include:

            1. **Strategic Planning**:
                - Understanding long-term sales trends helps businesses set realistic growth targets, allocate resources effectively, and make strategic decisions.

            2. **Capacity Planning and Expansion**:
                - Forecasting demand over a 5-year horizon supports capacity planning, whether in production, staffing, or infrastructure, preparing the business for expected growth.

            3. **Financial Forecasting**:
                - Long-term sales projections assist in budgeting and financial planning, ensuring that the company is financially prepared for anticipated growth or slowdowns.

            4. **Risk Management**:
                - Predicting sales over a longer period provides insights into potential risks, such as market saturation or seasonal downturns, allowing for proactive mitigation strategies.

            **In summary**, a 5-year sales forecast is a valuable tool for ensuring long-term stability and growth, helping businesses make strategic, data-driven decisions based on predicted sales trends.
            """)