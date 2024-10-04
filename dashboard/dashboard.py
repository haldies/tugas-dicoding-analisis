import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(layout="wide", page_title="Brazilian E-Commerce Dashboard")

st.markdown("""
<style>
    .stApp {
        background-color: #f0f2f6;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 10px;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stMetric .metric-label {
        font-weight: bold;
        color: #1f77b4;
    }
    .stMetric .metric-value {
        font-size: 24px;
        color: #2c3e50;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    try:
        df = pd.read_csv("dashboard/all_data_cleaned.csv")
        print("Data berhasil dimuat dari file lokal.")
    except FileNotFoundError:
        print("File lokal tidak ditemukan. Mencoba opsi kedua...")

        try:
            df = pd.read_csv("all_data_cleaned.csv")
            print("Data berhasil dimuat dari file lokal.")

        except Exception as e:
            print(f"Gagal memuat data dari URL: {e}. Mencoba opsi ketiga...")
            
            try:
                df = pd.read_csv(
                    "https://raw.githubusercontent.com/haldies/tugas-dicoding-data-analisis/refs/heads/main/dashboard/all_data_cleaned.csv")
                print("Data berhasil dimuat dari URL.")
            except FileNotFoundError:
                print("Sumber alternatif tidak ditemukan. Proses gagal.")
                return None 

    df['order_purchase_timestamp'] = pd.to_datetime(
        df['order_purchase_timestamp'])
    # Menambah kolom tahun, bulan, dan kuartal
    df['year'] = df['order_purchase_timestamp'].dt.year
    df['month'] = df['order_purchase_timestamp'].dt.month
    df['quarter'] = df['order_purchase_timestamp'].dt.quarter
    df['purchase_month'] = df['order_purchase_timestamp'].dt.to_period('M')

    return df


all_df = load_data()

st.title("Brazilian E-Commerce Dashboard")

# Sidebar for filters
st.sidebar.header("Filters")
year_options = ['All'] + sorted(all_df['year'].unique().tolist())
selected_year = st.sidebar.selectbox("Year", options=year_options)

category_options = [
    'All'] + sorted(all_df['product_category_name'].dropna().unique().tolist())
selected_category = st.sidebar.selectbox(
    "Product category", options=category_options)

state_options = ['All'] + sorted(all_df['customer_state'].unique().tolist())
selected_state = st.sidebar.selectbox("Customer State", options=state_options)

status_options = ['All'] + sorted(all_df['order_status'].unique().tolist())
selected_status = st.sidebar.selectbox("Order Status", options=status_options)

# Filtering the DataFrame
filtered_df = all_df.copy()
if selected_year != 'All':
    filtered_df = filtered_df[filtered_df['year'] == selected_year]
if selected_category != 'All':
    filtered_df = filtered_df[filtered_df['product_category_name']
                              == selected_category]
if selected_state != 'All':
    filtered_df = filtered_df[filtered_df['customer_state'] == selected_state]
if selected_status != 'All':
    filtered_df = filtered_df[filtered_df['order_status'] == selected_status]


def calculate_rfm(df):

    df['recency'] = (df['order_purchase_timestamp'].max() -
                     df['order_purchase_timestamp']).dt.days

    frequency_df = df.groupby('customer_unique_id').agg(
        {'order_id': 'nunique'}).reset_index()
    frequency_df.columns = ['customer_unique_id', 'frequency']

    monetary_df = df.groupby('customer_unique_id').agg(
        {'payment_value': 'sum'}).reset_index()
    monetary_df.columns = ['customer_unique_id', 'monetary']

    rfm_df = df[['customer_unique_id', 'recency']]
    rfm_df = rfm_df.merge(frequency_df, on='customer_unique_id')
    rfm_df = rfm_df.merge(monetary_df, on='customer_unique_id')

    rfm_df['r_rank'] = rfm_df['recency'].rank(ascending=False)
    rfm_df['f_rank'] = rfm_df['frequency'].rank(ascending=True)
    rfm_df['m_rank'] = rfm_df['monetary'].rank(ascending=True)

    rfm_df['r_rank_norm'] = (rfm_df['r_rank'] / rfm_df['r_rank'].max()) * 100
    rfm_df['f_rank_norm'] = (rfm_df['f_rank'] / rfm_df['f_rank'].max()) * 100
    rfm_df['m_rank_norm'] = (rfm_df['m_rank'] / rfm_df['m_rank'].max()) * 100

    rfm_df.drop(columns=['r_rank', 'f_rank', 'm_rank'], inplace=True)

    rfm_df['RFM_score'] = 0.15 * rfm_df['r_rank_norm'] + 0.28 * \
        rfm_df['f_rank_norm'] + 0.57 * rfm_df['m_rank_norm']
    rfm_df['RFM_score'] *= 0.05
    rfm_df = rfm_df.round(2)

    rfm_df["customer_segment"] = np.where(
        rfm_df['RFM_score'] > 4.5, "Top customers",
        np.where(rfm_df['RFM_score'] > 4, "High value customer",
                 np.where(rfm_df['RFM_score'] > 3, "Medium value customer",
                          np.where(rfm_df['RFM_score'] > 1.6, 'Low value customers', 'lost customers'))))

    return rfm_df


# KPIs
st.header("Key Performance Indicators")
kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)

total_orders = filtered_df['order_id'].nunique()
kpi1.metric(label="Total Orders", value=f"{total_orders:,}")

total_items = filtered_df['order_item_id'].sum()
kpi2.metric(label="Total Items", value=f"{total_items:,}")

total_products = filtered_df['product_id'].nunique()
kpi3.metric(label="Total Products", value=f"{total_products:,}")

total_sales = filtered_df['price'].sum()
kpi4.metric(label="Total Sales", value=f"Rp{total_sales:,.2f}")

total_freight = filtered_df['freight_value'].sum()
kpi5.metric(label="Total Freight", value=f"Rp{total_freight:,.2f}")

avg_rating = filtered_df['review_score'].mean()
kpi6.metric(label="Average Rating", value=f"{avg_rating:.2f}")

# Sales and Revenue Analysis
st.header("Sales and Revenue Analysis")
col1, col2 = st.columns(2)

sales_revenue = filtered_df.groupby('purchase_month').agg(
    total_sales=('order_id', 'count'),
    total_revenue=('price', 'sum')
).reset_index()

sales_revenue['purchase_month'] = sales_revenue['purchase_month'].dt.to_timestamp()

with col1:
    st.subheader("Total Sales per Month")
    fig_sales, ax_sales = plt.subplots(figsize=(14, 7))
    sns.lineplot(x=sales_revenue['purchase_month'].astype(
        str), y=sales_revenue['total_sales'], marker='o', ax=ax_sales)
    ax_sales.set_title('Total Sales per Month', fontsize=16)
    ax_sales.set_xlabel('Purchase Month', fontsize=14)
    ax_sales.set_ylabel('Number of Orders', fontsize=14)
    ax_sales.grid(True)

    for i, v in enumerate(sales_revenue['total_sales']):
        ax_sales.text(i, v, str(v), ha='center', va='bottom', fontsize=9)

    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig_sales)


with col2:
    st.subheader("Total Revenue and Total Sales per Month")
    fig_combined, ax_combined = plt.subplots(figsize=(14, 7))

    sns.barplot(x=sales_revenue['purchase_month'].astype(str),
                y=sales_revenue['total_revenue'],
                color='green',
                alpha=0.6,
                label='Total Revenue',
                ax=ax_combined)

    sns.lineplot(x=sales_revenue['purchase_month'].astype(str),
                 y=sales_revenue['total_sales'],
                 marker='o',
                 color='blue',
                 label='Total Sales',
                 ax=ax_combined)

    ax_combined.set_ylabel('Total Revenue (Rp) / Number of Sales', fontsize=14)
    ax_combined.legend()

    for i, v in enumerate(sales_revenue['total_revenue']):
        ax_combined.text(
            i, v, f'Rp {v:,.0f}', ha='center', va='bottom', fontsize=8, rotation=90)

    ax_combined.set_title(
        'Total Revenue and Total Sales per Month', fontsize=16)
    ax_combined.set_xlabel('Purchase Month', fontsize=14)
    ax_combined.set_xticklabels(ax_combined.get_xticklabels(), rotation=45)
    ax_combined.grid(True)

    plt.tight_layout()
    st.pyplot(fig_combined)

st.header("Customer and Product Analysis")
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Top Customer States")
    top_states = filtered_df.groupby('customer_state')[
        'order_id'].count().nlargest(10).reset_index()
    fig3, ax3 = plt.subplots(figsize=(5, 5))
    colors = plt.cm.Set3(np.linspace(0, 1, len(top_states)))
    wedges, texts, autotexts = ax3.pie(top_states['order_id'], labels=top_states['customer_state'],
                                       autopct='%1.1f%%', startangle=30, colors=colors)
    ax3.set_title("Top 10 Customer States")
    st.pyplot(fig3)

with col2:
    st.subheader("Top 5 Highest Rated Products")
    top_rated = filtered_df.groupby('product_category_name')[
        'review_score'].mean().nlargest(5).reset_index()
    fig4, ax4 = plt.subplots(figsize=(8, 6))
    sns.barplot(data=top_rated, x='review_score',
                y='product_category_name', color='#d62728', ax=ax4)
    ax4.set_title("Top 5 Highest Rated Product Categories")
    ax4.set_xlabel("Average Rating")
    ax4.set_ylabel("Product Category")
    st.pyplot(fig4)

with col3:
    st.subheader("Bottom 5 Lowest Rated Products")
    bottom_rated = filtered_df.groupby('product_category_name')[
        'review_score'].mean().nsmallest(5).reset_index()
    fig5, ax5 = plt.subplots(figsize=(8, 6))
    sns.barplot(data=bottom_rated, x='review_score',
                y='product_category_name', color='#9467bd', ax=ax5)
    ax5.set_title("Bottom 5 Lowest Rated Product Categories")
    ax5.set_xlabel("Average Rating")
    ax5.set_ylabel("Product Category")
    st.pyplot(fig5)

st.header("Sales and Reviews Analysis")

sales_per_category = filtered_df.groupby('product_category_name').agg({
    'price': 'sum',
    'order_id': 'count',
    'review_score': 'mean'
}).reset_index()
sales_per_category.columns = ['product_category_name',
                              'total_sales', 'items_sold', 'average_review_score']

top_sales_categories = sales_per_category.nlargest(5, 'total_sales')
top_review_categories = sales_per_category.nlargest(5, 'average_review_score')

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Top 5 Categories by Sales Volume")
    fig6, ax6 = plt.subplots(figsize=(10, 6))
    sales_plot = sns.barplot(data=top_sales_categories, x='total_sales',
                             y='product_category_name', palette='Blues_d', ax=ax6)
    ax6.set_title('Top 5 Product Categories by Sales Volume')
    ax6.set_xlabel('Total Sales (Rp)')
    ax6.set_ylabel('Product Category')
    ax6.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, p: f'Rp{x:,.0f}'))
    for i, v in enumerate(top_sales_categories['total_sales']):
        ax6.text(v, i, f'Rp{v:,.0f}', va='center', ha='left')
    st.pyplot(fig6)

with col2:
    st.subheader("Top 5 Categories by Average Review Score")
    fig7, ax7 = plt.subplots(figsize=(10, 6))
    review_plot = sns.barplot(data=top_review_categories, x='average_review_score',
                              y='product_category_name', palette='Greens_d', ax=ax7)
    ax7.set_title('Top 5 Product Categories by Average Review Score')
    ax7.set_xlabel('Average Review Score')
    ax7.set_ylabel('Product Category')
    for i, v in enumerate(top_review_categories['average_review_score']):
        ax7.text(v, i, f'{v:.2f}', va='center', ha='left')
    st.pyplot(fig7)
with col3:
    st.subheader("Average Review Scores for Top Sales Categories")
    top_sales_ratings = sales_per_category[sales_per_category['product_category_name'].isin(
        top_sales_categories['product_category_name'])]
    fig8, ax8 = plt.subplots(figsize=(12, 6))
    rating_plot = sns.barplot(data=top_sales_ratings, x='average_review_score',
                              y='product_category_name', palette='Reds_d', ax=ax8)
    ax8.set_title(
        'Average Review Scores for Top 5 Product Categories by Sales Volume')
    ax8.set_xlabel('Average Review Score')
    ax8.set_ylabel('Product Category')
    for i, v in enumerate(top_sales_ratings['average_review_score']):
        ax8.text(v, i, f'{v:.2f}', va='center', ha='left')
    st.pyplot(fig8)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Customer Segmentation Analysis")
    rfm_df = calculate_rfm(filtered_df)
    customer_segment_df = rfm_df.groupby(
        by="customer_segment", as_index=False).customer_unique_id.nunique()
    customer_segment_df['customer_segment'] = pd.Categorical(customer_segment_df['customer_segment'], [
        "lost customers", "Low value customers", "Medium value customer",
        "High value customer", "Top customers"
    ])
    sorted_customer_segment_df = customer_segment_df.sort_values(
        by="customer_unique_id", ascending=False)

    plt.figure(figsize=(10, 5))
    colors_ = ["#72BCD4", "#72BCD4", "#D3D3D3", "#D3D3D3", "#D3D3D3"]

    bar_plot = sns.barplot(
        x="customer_unique_id",
        y="customer_segment",
        data=sorted_customer_segment_df,
        palette=colors_)

    plt.title("Number of Customers for Each Segment",
              loc="center", fontsize=15)
    plt.ylabel(None)
    plt.xlabel(None)
    plt.tick_params(axis='y', labelsize=12)

    for bar in bar_plot.patches:
        bar_plot.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f'{int(bar.get_width())}',
            va='center', ha='left', fontsize=10
        )

    plt.tight_layout()
    st.pyplot(plt.gcf())
