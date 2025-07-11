# Sales-Data-Analyzer
# Sales Data Analyzer

---

## 🚀 Project Overview

**Sales Data Analyzer** is a robust end-to-end pipeline designed to clean, analyze, and visualize sales data. It transforms raw transactional records into insightful dashboards that help stakeholders understand business performance, product trends, customer behavior, and seasonal dynamics.

This project reflects real-world data science workflows, from data preprocessing to exploratory analysis and executive-level visualization, making it ideal for portfolios, business use, or academic demonstration.

---

## 📆 Data Source & Structure

* **Input**: CSV file with sales transactions (`sample_sales_data.csv`)
* **Key Columns**: Date, Revenue, ProductID, ProductName, Quantity, CustomerID, Category, Region
* **Size**: \~1,000+ records, various customer/product interactions across time

---

## 🧰 Project Modules

### 1. `data_processor.py`

* Loads raw CSV
* Cleans missing values, converts date/time, ensures column integrity
* Adds features: `Weekday`, `Quarter`, etc.
* Exports cleaned dataset to `data/cleaned_sales_data.csv`

### 2. `analyzer.py`

* Performs key statistical summaries
* Computes revenue stats, unique customer/product counts, date range, etc.
* Provides structured summary for logging and reporting

### 3. `visualizer.py`

* Generates 5 powerful visual dashboards:

  * **Revenue Trend Analysis**
  * **Product Performance Overview**
  * **Customer Analysis Dashboard**
  * **Seasonal Pattern Detection**
  * **Executive Sales Dashboard**
* Uses `matplotlib`, `seaborn`, and `numpy` for plotting
* Saves all charts in `outputs/visualizations`

---

## 📊 Key Insights

* Revenue and customer trends by time, product, and category
* Top 10 performing products & customer segments
* Seasonal and daily sales behavior patterns
* Region-based performance (if available)
* Executive KPIs in a single dashboard (Total Revenue, Unique Customers, etc.)

---

## ⚙️ How to Use

1. Place raw data in `data/`
2. Run the complete analysis using:

```bash
python main.py
```

3. Visuals and logs will be auto-generated under `outputs/`

---

## 🔍 Sample Outputs

Visuals generated:

* `revenue_trend_analysis.png`
* `product_performance_analysis.png`
* `customer_analysis_dashboard.png`
* `seasonal_patterns_analysis.png`
* `executive_dashboard.png`

---

## 💼 Author & Contact

**Habiba Isah – Data Scientist**

* [LinkedIn](https://www.linkedin.com/in/habiba-isah-6120241a3/)
* [GitHub](https://github.com/bibexaliyy)
* Email: [hisah075@gmail.com](mailto:hisah075@gmail.com)

---

> This project showcases my skills in data cleaning, statistical analysis, and storytelling with data. If you're a recruiter, tech enthusiast, or business looking for insight-driven solutions, I'd love to connect!

---
