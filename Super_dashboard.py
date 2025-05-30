import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

st.set_page_config(page_title="Sales Dashboard", layout="wide")
# Dashboard title
st.title("Sales Dashboard")

# Data upload
@st.cache_data
def load_data():
    df = pd.read_csv("filtered_sales.csv", parse_dates=['Order Date']) 
    df['Year'] = df['Order Date'].dt.year
    df['Year_Month'] = df['Order Date'].dt.to_period('M').astype(str)
    return df

data = load_data()

# Sidebar with Filters
st.sidebar.header("Filters")

# Unique values
years = sorted(data['Year'].unique())
segments = data['Segment'].unique()
categories = data['Category'].unique()

# Vigets
selected_years = st.sidebar.multiselect("Year", options=years, default=years)
selected_segments = st.sidebar.multiselect("Segment", options=segments, default=segments)
selected_categories = st.sidebar.multiselect("Category", options=categories, default=categories)

# Filters
filtered_data = data[
    (data['Year'].isin(selected_years)) &
    (data['Segment'].isin(selected_segments)) &
    (data['Category'].isin(selected_categories))
]

# KPI
total_sales = filtered_data['Sales'].sum()
avg_order_value = filtered_data.groupby('Order ID')['Sales'].sum().mean()

col1, col2 = st.columns(2)
col1.metric("Total Sales", f"${total_sales:,.2f}")
col2.metric("Average Order Value", f"${avg_order_value:,.2f}")

# Monthly Sales Trend
monthly_sales = filtered_data.groupby('Year_Month')['Sales'].sum().reset_index()
fig = px.line(monthly_sales, x='Year_Month', y='Sales', title='Monthly Sales Trend')
st.plotly_chart(fig, use_container_width=True)

# Sales by Category
st.subheader("Sales by Category")

category_sales = (
    filtered_data.groupby('Category')['Sales']
    .sum()
    .reset_index()
    .sort_values(by='Sales', ascending=False)
)

fig2 = px.bar(
    category_sales,
    x='Category',
    y='Sales',
    text='Sales',
    title='💼 Total Sales by Category',
    color='Category',  # добавим цвета
    color_discrete_sequence=px.colors.qualitative.Set2
)
fig2.update_traces(texttemplate='%{text:.2s}', textposition='outside')
fig2.update_layout(showlegend=False)

st.plotly_chart(fig2, use_container_width=True)

# TOP 10 PRODUCTS
top_products = filtered_data.groupby('Product Name')['Sales'].sum().sort_values(ascending=False).head(10).reset_index()
fig3 = px.bar(top_products, x='Sales', y='Product Name', orientation='h',
              title='Top 10 Products by Sales', text='Sales', color='Sales')
fig3.update_layout(yaxis={'categoryorder': 'total ascending'})
st.plotly_chart(fig3, use_container_width=True)

# CATEGORY SHARE PIE
category_share = filtered_data.groupby('Category')['Sales'].sum().reset_index()
category_share['Share'] = category_share['Sales'] / category_share['Sales'].sum() * 100
fig4 = px.pie(category_share, names='Category', values='Share', title='📊 Category Sales Share', hole=0.4)
fig4.update_traces(textinfo='percent+label')
st.plotly_chart(fig4, use_container_width=True)

# US STATE MAP
us_state_abbrev = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR',
    'California': 'CA', 'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE',
    'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI', 'Idaho': 'ID',
    'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS',
    'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
    'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS',
    'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV',
    'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM',
    'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND',
    'Ohio': 'OH', 'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA',
    'Rhode Island': 'RI', 'South Carolina': 'SC', 'South Dakota': 'SD',
    'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT',
    'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV',
    'Wisconsin': 'WI', 'Wyoming': 'WY'
}

state_sales = filtered_data.groupby('State')['Sales'].sum().reset_index()
state_sales['State Code'] = state_sales['State'].map(us_state_abbrev)

fig5 = px.choropleth(
    state_sales,
    locations='State Code',
    locationmode='USA-states',
    color='Sales',
    hover_name='State',
    scope='usa',
    color_continuous_scale='Blues',
    title='Sales by US State'
)
st.plotly_chart(fig5, use_container_width=True)

# Sales Heatmap: Category vs Month
heatmap_data = data.copy()
heatmap_data['Year_Month'] = pd.to_datetime(heatmap_data['Order Date']).dt.to_period('M').astype(str)
category_month_sales = heatmap_data.groupby(['Category', 'Year_Month'])['Sales'].sum().reset_index()
heatmap_pivot = category_month_sales.pivot(index='Category', columns='Year_Month', values='Sales')
heatmap_data = data.copy()
selected_heatmap_segment = st.selectbox("Select Segment for Heatmap", options=data['Segment'].unique(), index=0)
heatmap_data = data[data['Segment'] == selected_heatmap_segment].copy()
heatmap_data['Year_Month'] = pd.to_datetime(heatmap_data['Order Date']).dt.to_period('M').astype(str)
category_month_sales = heatmap_data.groupby(['Category', 'Year_Month'])['Sales'].sum().reset_index()
heatmap_pivot = category_month_sales.pivot(index='Category', columns='Year_Month', values='Sales')
heatmap_pivot = heatmap_pivot.fillna(0)

fig_heatmap = px.imshow(
    heatmap_pivot.values,
    labels=dict(x="Month", y="Category", color="Sales"),
    x=heatmap_pivot.columns,
    y=heatmap_pivot.index,
    aspect="auto",
    color_continuous_scale='Blues'
)
fig_heatmap.update_layout(title='Sales Heatmap: Category vs Month', xaxis_nticks=20)
st.plotly_chart(fig_heatmap, use_container_width=True)

#  HYPOTHESIS and VALIDATION

# 1 Repeat customers VS one time customers

oreders_per_customer = data.groupby('Customer ID')['Order ID'].nunique().reset_index()
oreders_per_customer.columns = ['Customer ID', 'Order Count']
data_with_orders = data.merge(oreders_per_customer, on='Customer ID')
data_with_orders['Is Repeat'] = data_with_orders['Order Count'] > 2
avg_sales_by_group = data_with_orders.groupby('Is Repeat')['Sales'].mean().reset_index()
avg_sales_by_group['Group'] = avg_sales_by_group['Is Repeat'].map({True: 'Repeat', False: 'One-time'})

fig6 = px.bar(avg_sales_by_group, x='Group', y='Sales', text='Sales',
              title='Avg Sale: Repeat vs One-time Customers',
              color='Group', color_discrete_sequence=px.colors.qualitative.Set2)
fig6.update_traces(texttemplate='$%{text:.2f}', textposition='outside')
fig6.update_layout(showlegend=False)
st.plotly_chart(fig6, use_container_width=True)

# HYPOTHESIS 2: Forecast Monthly Sales (XGBoost Forecasting with Features)
# copy data
monthly_sales = data.copy()

# transformation to year-month format
monthly_sales['Year_Month'] = pd.to_datetime(monthly_sales['Order Date'], dayfirst=True).dt.to_period('M')
monthly_sales = monthly_sales.groupby('Year_Month')['Sales'].sum().reset_index()

# transform back to datetime for visualisation
monthly_sales['Year_Month'] = monthly_sales['Year_Month'].dt.to_timestamp()
monthly_sales = monthly_sales.sort_values('Year_Month')

# median for monthly sales
monthly_median = monthly_sales['Sales'].median()
st.markdown(f"**Median monthly sales:** ${monthly_median:,.2f}")

# add parameters
monthly_sales['Month'] = monthly_sales['Year_Month'].dt.month
monthly_sales['Year'] = monthly_sales['Year_Month'].dt.year
monthly_sales['Month_Num'] = np.arange(len(monthly_sales))

# Devide date for test and train
train = monthly_sales.iloc[:-3]
test = monthly_sales.iloc[-3:]

# XGBoost model realisation
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

X_train = train[['Month', 'Year', 'Month_Num']]
y_train = train['Sales']
X_test = test[['Month', 'Year', 'Month_Num']]

param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.05, 0.1, 0.2]
}

base_model = XGBRegressor(objective='reg:squarederror', random_state=42)
grid_search = GridSearchCV(estimator=base_model, param_grid=param_grid, cv=3, scoring='neg_mean_absolute_error')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
future_preds = best_model.predict(X_test)

best_params = grid_search.best_params_
best_mae = -grid_search.best_score_

# parameter outputs and MAE
st.markdown(f"**Best Cross-validated MAE:** {best_mae:,.2f}")
st.markdown(f"**Best XGBoost parameters:** {best_params}")

# DataFrame with forecast for last 3 months
future_df = test.copy()
future_df['Sales'] = future_preds

# full series
full_series = pd.concat([train[['Year_Month', 'Sales']], future_df[['Year_Month', 'Sales']]])

# Dynamic forecast for N next months 
st.subheader("Forecast sales for N months ahead")
forecast_months = st.slider("Select Forecast Horizon", 1, 6, 3)

# future months generator
last_month_num = monthly_sales['Month_Num'].max()
future_month_nums = np.arange(last_month_num + 1, last_month_num + 1 + forecast_months)

future_months_df = pd.DataFrame({
    'Month_Num': future_month_nums,
    'Month': [(m % 12) + 1 for m in future_month_nums],
    'Year': [monthly_sales['Year'].iloc[-1] + ((m + 1) // 12) for m in range(forecast_months)]
})

# Forecast
future_preds_dynamic = best_model.predict(future_months_df[['Month', 'Year', 'Month_Num']])
future_dynamic_dates = pd.date_range(start=monthly_sales['Year_Month'].max() + pd.DateOffset(months=1),
                                     periods=forecast_months, freq='MS')

future_dynamic_df = pd.DataFrame({
    'Year_Month': future_dynamic_dates,
    'Sales': future_preds_dynamic
})

# concatenation both dataframes 
full_forecast = pd.concat([monthly_sales[['Year_Month', 'Sales']], future_dynamic_df])

# forecast visualisation
fig7 = px.line(full_forecast, x='Year_Month', y='Sales',
              title=f'🔮 XGBoost Forecast for next {forecast_months} months', markers=True)
fig7.add_scatter(x=future_dynamic_df['Year_Month'], y=future_dynamic_df['Sales'],
                mode='markers+lines', name='Прогноз',
                line=dict(dash='dash'))
st.plotly_chart(fig7, use_container_width=True)


# HYPOTHESIS 3: Pareto in Furniture category
furniture_data = data[data['Category'] == 'Furniture']
customer_furn_sales = furniture_data.groupby('Customer ID')['Sales'].sum().reset_index().sort_values(by='Sales', ascending=False)
customer_furn_sales['Cumulative Sales'] = customer_furn_sales['Sales'].cumsum()
customer_furn_sales['Cumulative %'] = 100 * customer_furn_sales['Cumulative Sales'] / customer_furn_sales['Sales'].sum()
customer_furn_sales['Customer %'] = 100 * (np.arange(len(customer_furn_sales)) + 1) / len(customer_furn_sales)

fig8 = px.line(customer_furn_sales, x='Customer %', y='Cumulative %',
               title='Pareto Analysis: Furniture Sales by Customers')
fig8.add_shape(type='line', x0=20, y0=0, x1=20, y1=100,
               line=dict(color='Red', dash='dash'), name='20% Customers')
fig8.add_shape(type='line', x0=0, y0=80, x1=100, y1=80,
               line=dict(color='Green', dash='dash'), name='80% Sales')
fig8.update_layout(xaxis_title='% of Customers', yaxis_title='% of Cumulative Sales')
st.plotly_chart(fig8, use_container_width=True)

# IMPLEMENTATION 2: Prioritized Clients Table
st.subheader("Priority Clients (Top Revenue Generators)")
priority_clients = (
    data.groupby(['Customer ID'])['Sales']
    .sum()
    .reset_index()
    .sort_values(by='Sales', ascending=False)
    .head(20)
)
st.dataframe(priority_clients, use_container_width=True)

import io
csv = priority_clients.to_csv(index=False).encode('utf-8')
st.download_button("Download CSV", data=csv, file_name='priority_clients.csv', mime='text/csv')

# IMPLEMENTATION 3: Export Summary Report
from fpdf import FPDF

def create_pdf_report():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Sales Summary Report", ln=1, align='C')
    pdf.ln(10)
    pdf.multi_cell(0, 10, txt="Total Sales: ${:,.2f}\nTop Customers:".format(total_sales))
    pdf.ln(5)
    pdf.set_fill_color(240, 240, 240)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(60, 10, "Customer ID", border=1, align='C', fill=True)
    pdf.cell(60, 10, "Sales", border=1, align='C', fill=True)
    pdf.ln()
    pdf.set_font("Arial", size=12)
    for idx, row in priority_clients.iterrows():
        pdf.cell(60, 10, str(row['Customer ID']), border=1)
        pdf.cell(60, 10, f"${row['Sales']:.2f}", border=1)
        pdf.ln()
    pdf.output("report.pdf")

if st.button("Generate Summary Report (PDF)"):
    create_pdf_report()
    with open("report.pdf", "rb") as file:
        st.download_button("Download Report", data=file, file_name="sales_summary_report.pdf")

