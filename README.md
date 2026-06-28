# 🛍️ Customer Segmentation using RFM Analysis & K-Means Clustering

> Segmenting 4,338 customers from 500k+ retail transactions into actionable behavioral groups using RFM Analysis, K-Means Clustering, and Hierarchical Clustering — deployed as an interactive Streamlit dashboard.

🔗 **Live Demo:** [customer-segmentation-izkvtqe7a8kkgg6qztetbj.streamlit.app](https://customer-segmentation-izkvtqe7a8kkgg6qztetbj.streamlit.app)

---

## 📌 Problem Statement

Retail businesses struggle to treat all customers the same way. A customer who buys every week deserves different treatment than one who hasn't purchased in 10 months. This project automatically segments customers based on their purchasing behavior — enabling targeted marketing, retention strategies, and personalized outreach.

---

## 📊 Dataset

- **Source:** [Online Retail Dataset — UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/online+retail)
- **Size:** 541,909 transactions
- **Period:** December 2010 – December 2011
- **Region:** UK-based online retail store

| Column | Description |
|---|---|
| InvoiceNo | Unique transaction ID |
| StockCode | Product code |
| Description | Product name |
| Quantity | Units purchased |
| InvoiceDate | Date and time of transaction |
| UnitPrice | Price per unit (£) |
| CustomerID | Unique customer identifier |
| Country | Customer's country |

---

## 🧠 Methodology

### 1. Data Cleaning
- Removed 135,080 rows with missing `CustomerID`
- Removed 8,905 cancelled transactions (negative quantity)
- Removed 40 rows with zero or negative unit price
- **Final clean dataset: 397,884 transactions**

### 2. RFM Feature Engineering
Created 3 behavioral features per customer:

| Feature | Definition | Business Meaning |
|---|---|---|
| **Recency** | Days since last purchase | How recently did they buy? |
| **Frequency** | Number of unique orders | How often do they buy? |
| **Monetary** | Total amount spent (£) | How much have they spent? |

### 3. Preprocessing
- **Outlier capping** using IQR method (Q3 + 3×IQR threshold)
- **Feature scaling** using StandardScaler (mean=0, std=1)

### 4. Clustering

**K-Means Clustering**
- Used Elbow Method to determine optimal k=4
- Initialization: K-Means++ for smart centroid placement

**Hierarchical Clustering (Validation)**
- Used Ward linkage with Agglomerative approach
- Dendrogram confirmed k=4 as optimal
- Both methods independently produced identical segment profiles — confirming robustness

---

## 👥 Customer Segments

| Segment | Recency | Frequency | Monetary | Count | Strategy |
|---|---|---|---|---|---|
| 👑 Champions | 17 days | 14 orders | £5,090 | 360 | Reward & retain |
| 💙 Loyal | 34 days | 6 orders | £2,488 | 822 | Nurture & upsell |
| ⚠️ At Risk | 52 days | 2 orders | £630 | 2,144 | Re-engagement campaigns |
| ❌ Lost | 253 days | 1 order | £430 | 1,012 | Low priority win-back |

---

## 🖥️ Streamlit Dashboard Features

- **Segment overview** — customer counts per segment with metric cards
- **Segment profiles** — avg Recency, Frequency, Monetary per group
- **Visual analysis** — pie chart, bar chart, RFM scatter plots, elbow curve, correlation heatmap
- **Customer lookup** — enter any CustomerID to instantly see their segment
- **Full RFM table** — expandable view of all 4,338 customers

---

## 🗂️ Project Structure

```
customer-segmentation/
│
├── data/
│   └── online_retail.xlsx        # Raw dataset
├── notebook/
│   └── analysis.ipynb            # Full EDA + ML pipeline
├── app.py                        # Streamlit dashboard
├── requirements.txt              # Dependencies
└── README.md
```

---

## ⚙️ Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/oorrvii/customer-segmentation.git
cd customer-segmentation

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

Then upload the `online_retail.xlsx` file in the sidebar.

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.x |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Deployment | Streamlit |
| Version Control | Git, GitHub |

---

## 📈 Key Results

- Processed **541,909 transactions** down to **397,884** clean records
- Engineered **3 RFM features** from raw transactional data
- Identified **4 distinct customer segments** validated by both K-Means and Hierarchical Clustering
- Champions (8.3% of customers) account for the highest average spend at **£5,090 per customer**
- 49.4% of customers fall in the At Risk segment — the biggest re-engagement opportunity

---

## 👩‍💻 Author

**Oorvi Kulshreshtha**
B.Tech CSE | Hindustan College of Science and Technology
🔗 [Portfolio](https://oorvi-portfolio.vercel.app)
