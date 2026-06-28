import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ─── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="🛍️",
    layout="wide"
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { background-color: #0f1117; }

    .hero {
        background: linear-gradient(135deg, #1a1f35 0%, #0f1117 100%);
        border: 1px solid #2a2f45;
        border-radius: 16px;
        padding: 40px;
        margin-bottom: 32px;
        text-align: center;
    }
    .hero h1 {
        font-size: 2.4rem;
        font-weight: 600;
        color: #ffffff;
        margin: 0 0 8px 0;
    }
    .hero p {
        color: #8b92a5;
        font-size: 1rem;
        margin: 0;
    }

    .metric-card {
        background: #1a1f35;
        border: 1px solid #2a2f45;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
    }
    .metric-card .value {
        font-size: 2rem;
        font-weight: 600;
        color: #ffffff;
    }
    .metric-card .label {
        font-size: 0.8rem;
        color: #8b92a5;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 4px;
    }

    .segment-card {
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 12px;
        border: 1px solid;
    }
    .segment-card h4 { margin: 0 0 6px 0; font-size: 1rem; font-weight: 600; }
    .segment-card p  { margin: 0; font-size: 0.85rem; opacity: 0.85; }

    .champions { background: #1a2e1a; border-color: #2d5a2d; color: #7dd87d; }
    .loyal     { background: #1a1f35; border-color: #2a3a6e; color: #7ab4f5; }
    .atrisk    { background: #2e1f0a; border-color: #5a3a0a; color: #f5b84a; }
    .lost      { background: #2e1a1a; border-color: #5a2a2a; color: #f57a7a; }

    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #ffffff;
        margin: 32px 0 16px 0;
        padding-bottom: 8px;
        border-bottom: 1px solid #2a2f45;
    }

    .stDataFrame { border-radius: 12px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ─── Helper: load & process data ───────────────────────────────────────────────
@st.cache_data
def load_and_process(file):
    dataset = pd.read_excel(file)
    dataset = dataset.dropna(subset=['CustomerID'])
    dataset['CustomerID'] = dataset['CustomerID'].astype(int)
    dataset = dataset[dataset['Quantity'] > 0]
    dataset = dataset[dataset['UnitPrice'] > 0]
    dataset['TotalPrice'] = dataset['Quantity'] * dataset['UnitPrice']

    reference_date = dataset['InvoiceDate'].max() + pd.Timedelta(days=1)

    rfm = dataset.groupby('CustomerID').agg(
        Recency   = ('InvoiceDate',  lambda x: (reference_date - x.max()).days),
        Frequency = ('InvoiceNo',    'nunique'),
        Monetary  = ('TotalPrice',   'sum')
    ).reset_index()

    for col in ['Recency', 'Frequency', 'Monetary']:
        Q1 = rfm[col].quantile(0.25)
        Q3 = rfm[col].quantile(0.75)
        rfm[col] = rfm[col].clip(upper=Q3 + 3 * (Q3 - Q1))

    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

    kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

    # auto-label clusters by Monetary mean
    order = rfm.groupby('Cluster')['Monetary'].mean().rank(ascending=False).astype(int)
    label_map = {
        cluster: ['Champions', 'Loyal', 'At Risk', 'Lost'][rank - 1]
        for cluster, rank in order.items()
    }
    rfm['Segment'] = rfm['Cluster'].map(label_map)

    return rfm, rfm_scaled, kmeans

# ─── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛍️ Customer Segmentation")
    st.markdown("---")
    uploaded = st.file_uploader("Upload Online Retail (.xlsx)", type=["xlsx"])
    st.markdown("---")
    st.markdown("**How it works**")
    st.markdown("""
- Cleans & preprocesses transactions
- Builds RFM scores per customer
- Clusters into 4 segments using K-Means
- Visualizes results interactively
    """)
    st.markdown("---")
    st.caption("Built with K-Means + RFM Analysis")

# ─── HERO ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🛍️ Customer Segmentation Dashboard</h1>
    <p>RFM Analysis · K-Means Clustering · Actionable Business Insights</p>
</div>
""", unsafe_allow_html=True)

# ─── MAIN ──────────────────────────────────────────────────────────────────────
if uploaded is None:
    st.info("👈 Upload the **Online Retail .xlsx** file from the sidebar to get started.")
    st.stop()

with st.spinner("Processing data and running clustering..."):
    rfm, rfm_scaled, kmeans = load_and_process(uploaded)

# ── Top metrics ────────────────────────────────────────────────────────────────
counts = rfm['Segment'].value_counts()
c1, c2, c3, c4, c5 = st.columns(5)
metrics = [
    ("Total Customers",  f"{len(rfm):,}",                          ""),
    ("Champions",        f"{counts.get('Champions', 0):,}",        "champions"),
    ("Loyal",            f"{counts.get('Loyal', 0):,}",            "loyal"),
    ("At Risk",          f"{counts.get('At Risk', 0):,}",          "atrisk"),
    ("Lost",             f"{counts.get('Lost', 0):,}",             "lost"),
]
for col, (label, value, _) in zip([c1, c2, c3, c4, c5], metrics):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="value">{value}</div>
            <div class="label">{label}</div>
        </div>""", unsafe_allow_html=True)

# ── Segment cards ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Segment Profiles</div>', unsafe_allow_html=True)
profiles = rfm.groupby('Segment')[['Recency', 'Frequency', 'Monetary']].mean().round(1)

card_cfg = {
    'Champions': ('champions', '👑', 'Recent, frequent, highest spenders. Reward and retain.'),
    'Loyal':     ('loyal',     '💙', 'Regular buyers with decent spend. Nurture toward champions.'),
    'At Risk':   ('atrisk',    '⚠️',  'Going quiet. Launch re-engagement campaigns now.'),
    'Lost':      ('lost',      '❌', 'Inactive for months. Hard to win back — low priority.'),
}

cols = st.columns(4)
for col, seg in zip(cols, ['Champions', 'Loyal', 'At Risk', 'Lost']):
    if seg in profiles.index:
        css, icon, desc = card_cfg[seg]
        r = profiles.loc[seg, 'Recency']
        f = profiles.loc[seg, 'Frequency']
        m = profiles.loc[seg, 'Monetary']
        with col:
            st.markdown(f"""
            <div class="segment-card {css}">
                <h4>{icon} {seg}</h4>
                <p>{desc}</p>
                <br/>
                <p>📅 Recency: <b>{r} days</b></p>
                <p>🔁 Frequency: <b>{f} orders</b></p>
                <p>💰 Monetary: <b>£{m:,.0f}</b></p>
                <p>👥 Count: <b>{counts.get(seg, 0):,}</b></p>
            </div>""", unsafe_allow_html=True)

# ── Charts ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Visual Analysis</div>', unsafe_allow_html=True)

COLORS = {'Champions': '#7dd87d', 'Loyal': '#7ab4f5', 'At Risk': '#f5b84a', 'Lost': '#f57a7a'}

tab1, tab2, tab3, tab4 = st.tabs(["📊 Segment Distribution", "📈 RFM Scatter", "📉 Elbow Method", "🔥 Correlation Heatmap"])

with tab1:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor='#0f1117')
    seg_counts = rfm['Segment'].value_counts()
    colors = [COLORS[s] for s in seg_counts.index]

    axes[0].pie(seg_counts.values, labels=seg_counts.index, colors=colors,
                autopct='%1.1f%%', textprops={'color': 'white', 'fontsize': 11},
                wedgeprops={'linewidth': 2, 'edgecolor': '#0f1117'})
    axes[0].set_facecolor('#0f1117')
    axes[0].set_title('Customer Distribution', color='white', fontsize=12, pad=15)

    axes[1].barh(seg_counts.index, seg_counts.values, color=colors, height=0.5)
    axes[1].set_facecolor('#0f1117')
    axes[1].tick_params(colors='white')
    axes[1].spines[:].set_color('#2a2f45')
    axes[1].set_title('Customer Count by Segment', color='white', fontsize=12)
    for i, v in enumerate(seg_counts.values):
        axes[1].text(v + 10, i, str(v), color='white', va='center', fontsize=10)

    fig.patch.set_facecolor('#0f1117')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with tab2:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor='#0f1117')
    pairs = [('Recency', 'Frequency'), ('Recency', 'Monetary'), ('Frequency', 'Monetary')]

    for ax, (x, y) in zip(axes, pairs):
        for seg, color in COLORS.items():
            subset = rfm[rfm['Segment'] == seg]
            ax.scatter(subset[x], subset[y], c=color, label=seg, alpha=0.5, s=15)
        ax.set_xlabel(x, color='#8b92a5')
        ax.set_ylabel(y, color='#8b92a5')
        ax.set_facecolor('#1a1f35')
        ax.tick_params(colors='#8b92a5')
        ax.spines[:].set_color('#2a2f45')
        ax.legend(fontsize=8, facecolor='#0f1117', labelcolor='white')

    fig.patch.set_facecolor('#0f1117')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with tab3:
    wcss = []
    for i in range(1, 11):
        km = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
        km.fit(rfm_scaled)
        wcss.append(km.inertia_)

    fig, ax = plt.subplots(figsize=(8, 4), facecolor='#0f1117')
    ax.plot(range(1, 11), wcss, marker='o', color='#7ab4f5', linewidth=2, markersize=7)
    ax.axvline(x=4, color='#f5b84a', linestyle='--', alpha=0.7, label='Chosen k=4')
    ax.set_xlabel('Number of clusters (k)', color='#8b92a5')
    ax.set_ylabel('WCSS', color='#8b92a5')
    ax.set_title('Elbow Method', color='white', fontsize=12)
    ax.set_facecolor('#1a1f35')
    ax.tick_params(colors='#8b92a5')
    ax.spines[:].set_color('#2a2f45')
    ax.legend(facecolor='#0f1117', labelcolor='white')
    fig.patch.set_facecolor('#0f1117')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with tab4:
    fig, ax = plt.subplots(figsize=(6, 4), facecolor='#0f1117')
    corr = rfm[['Recency', 'Frequency', 'Monetary']].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax,
                linewidths=0.5, linecolor='#0f1117',
                annot_kws={'color': 'white', 'fontsize': 11})
    ax.set_facecolor('#0f1117')
    ax.tick_params(colors='white')
    ax.set_title('RFM Correlation', color='white', fontsize=12)
    fig.patch.set_facecolor('#0f1117')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ── Customer Lookup ────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">🔍 Customer Lookup</div>', unsafe_allow_html=True)

customer_id = st.number_input("Enter Customer ID", min_value=int(rfm['CustomerID'].min()),
                               max_value=int(rfm['CustomerID'].max()), step=1)

if st.button("Look up customer"):
    result = rfm[rfm['CustomerID'] == customer_id]
    if result.empty:
        st.warning("Customer not found.")
    else:
        row = result.iloc[0]
        seg = row['Segment']
        css, icon, desc = card_cfg[seg]
        st.markdown(f"""
        <div class="segment-card {css}" style="max-width:400px">
            <h4>{icon} Customer {int(row['CustomerID'])} — {seg}</h4>
            <p>{desc}</p><br/>
            <p>📅 Last purchase: <b>{int(row['Recency'])} days ago</b></p>
            <p>🔁 Total orders: <b>{int(row['Frequency'])}</b></p>
            <p>💰 Total spend: <b>£{row['Monetary']:,.2f}</b></p>
        </div>""", unsafe_allow_html=True)

# ── Raw data ───────────────────────────────────────────────────────────────────
with st.expander("📋 View full RFM table"):
    st.dataframe(rfm[['CustomerID', 'Recency', 'Frequency', 'Monetary', 'Segment']],
                 use_container_width=True)