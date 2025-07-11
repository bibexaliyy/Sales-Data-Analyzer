"""
Sales Analyzer Module
Performs statistical analysis and generates business insights
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class SalesAnalyzer:
    """Main analysis class for sales data"""
    
    def __init__(self, data):
        """
        Initialize the analyzer with cleaned data
        
        Args:
            data (pd.DataFrame): Cleaned sales data
        """
        self.data = data.copy()
        self.insights = {}
        self.analysis_results = {}
        
        # Validate required columns
        required_cols = ['Date', 'CustomerID', 'ProductID', 'Revenue', 'Quantity']
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    def calculate_monthly_revenue(self):
        """Calculate monthly revenue trends"""
        monthly_revenue = self.data.groupby(self.data['Date'].dt.to_period('M')).agg({
            'Revenue': 'sum',
            'CustomerID': 'nunique',
            'ProductID': 'nunique'
        }).reset_index()
        
        monthly_revenue.columns = ['Month', 'Total_Revenue', 'Unique_Customers', 'Unique_Products']
        monthly_revenue['Month'] = monthly_revenue['Month'].astype(str)
        
        # Calculate month-over-month growth
        monthly_revenue['Revenue_Growth'] = monthly_revenue['Total_Revenue'].pct_change() * 100
        monthly_revenue['Customer_Growth'] = monthly_revenue['Unique_Customers'].pct_change() * 100
        
        self.analysis_results['monthly_revenue'] = monthly_revenue
        return monthly_revenue
    
    def calculate_product_performance(self):
        """Analyze comprehensive product performance metrics"""
        product_stats = self.data.groupby(['ProductID', 'ProductName']).agg({
            'Revenue': ['sum', 'mean', 'count'],
            'Quantity': ['sum', 'mean'],
            'Price': 'mean',
            'CustomerID': 'nunique'
        }).round(2)
        
        # Flatten column names
        product_stats.columns = [
            'Total_Revenue', 'Avg_Revenue_Per_Sale', 'Total_Orders',
            'Total_Quantity_Sold', 'Avg_Quantity_Per_Order', 'Avg_Price',
            'Unique_Customers'
        ]
        product_stats = product_stats.reset_index()
        
        # Calculate additional metrics
        product_stats['Revenue_Per_Customer'] = (
            product_stats['Total_Revenue'] / product_stats['Unique_Customers']
        ).round(2)
        
        # Calculate market share
        total_revenue = product_stats['Total_Revenue'].sum()
        product_stats['Market_Share'] = (
            product_stats['Total_Revenue'] / total_revenue * 100
        ).round(2)
        
        # Rank products
        product_stats['Revenue_Rank'] = product_stats['Total_Revenue'].rank(
            method='dense', ascending=False
        ).astype(int)
        
        # Sort by total revenue
        product_stats = product_stats.sort_values('Total_Revenue', ascending=False)
        
        self.analysis_results['product_performance'] = product_stats
        return product_stats
    
    def analyze_customer_segments(self):
        """Comprehensive customer segmentation analysis"""
        customer_stats = self.data.groupby('CustomerID').agg({
            'Revenue': ['sum', 'count', 'mean'],
            'Date': ['min', 'max'],
            'Quantity': 'sum',
            'ProductID': 'nunique'
        }).round(2)
        
        # Flatten column names
        customer_stats.columns = [
            'Total_Spent', 'Order_Count', 'Avg_Order_Value',
            'First_Purchase', 'Last_Purchase', 'Total_Items_Bought',
            'Unique_Products_Bought'
        ]
        customer_stats = customer_stats.reset_index()
        
        # Calculate additional metrics
        customer_stats['Customer_Lifetime_Days'] = (
            customer_stats['Last_Purchase'] - customer_stats['First_Purchase']
        ).dt.days
        
        customer_stats['Purchase_Frequency'] = np.where(
            customer_stats['Customer_Lifetime_Days'] > 0,
            customer_stats['Order_Count'] / (customer_stats['Customer_Lifetime_Days'] / 30),
            customer_stats['Order_Count']
        ).round(2)
        
        # Calculate recency (days since last purchase)
        max_date = self.data['Date'].max()
        customer_stats['Recency_Days'] = (
            max_date - customer_stats['Last_Purchase']
        ).dt.days
        
        # RFM Analysis (Recency, Frequency, Monetary)
        customer_stats['R_Score'] = pd.qcut(
            customer_stats['Recency_Days'], 5, labels=[5, 4, 3, 2, 1]
        ).astype(int)
        
        customer_stats['F_Score'] = pd.qcut(
            customer_stats['Order_Count'], 5, labels=[1, 2, 3, 4, 5]
        ).astype(int)
        
        customer_stats['M_Score'] = pd.qcut(
            customer_stats['Total_Spent'], 5, labels=[1, 2, 3, 4, 5]
        ).astype(int)
        
        customer_stats['RFM_Score'] = (
            customer_stats['R_Score'].astype(str) + 
            customer_stats['F_Score'].astype(str) + 
            customer_stats['M_Score'].astype(str)
        )
        
        # Customer segments based on spending
        customer_stats['Spending_Segment'] = pd.cut(
            customer_stats['Total_Spent'],
            bins=[0, 100, 500, 1000, 2000, float('inf')],
            labels=['Low Value', 'Medium Value', 'High Value', 'Premium', 'VIP']
        )
        
        # Customer lifecycle stage
        def get_lifecycle_stage(row):
            if row['Order_Count'] == 1:
                return 'New Customer'
            elif row['Recency_Days'] > 365:
                return 'Lost Customer'
            elif row['Recency_Days'] > 180:
                return 'At Risk'
            elif row['Order_Count'] >= 10:
                return 'Loyal Customer'
            else:
                return 'Regular Customer'
        
        customer_stats['Lifecycle_Stage'] = customer_stats.apply(get_lifecycle_stage, axis=1)
        
        self.analysis_results['customer_segments'] = customer_stats
        return customer_stats
    
    def calculate_growth_metrics(self):
        """Calculate comprehensive growth metrics"""
        monthly_data = self.calculate_monthly_revenue()
        
        if len(monthly_data) < 2:
            return {"error": "Insufficient data for growth analysis"}
        
        # Revenue growth metrics
        total_growth = ((monthly_data['Total_Revenue'].iloc[-1] / monthly_data['Total_Revenue'].iloc[0]) - 1) * 100
        avg_monthly_growth = monthly_data['Revenue_Growth'].mean()
        
        # Customer growth metrics
        customer_growth = monthly_data['Customer_Growth'].mean()
        
        # Find best and worst performing months
        best_month_idx = monthly_data['Total_Revenue'].idxmax()
        worst_month_idx = monthly_data['Total_Revenue'].idxmin()
        
        growth_metrics = {
            'total_revenue_growth': f"{total_growth:.1f}%",
            'avg_monthly_revenue_growth': f"{avg_monthly_growth:.1f}%",
            'avg_monthly_customer_growth': f"{customer_growth:.1f}%",
            'analysis_period_months': len(monthly_data),
            'best_month': {
                'month': monthly_data.iloc[best_month_idx]['Month'],
                'revenue': monthly_data.iloc[best_month_idx]['Total_Revenue']
            },
            'worst_month': {
                'month': monthly_data.iloc[worst_month_idx]['Month'],
                'revenue': monthly_data.iloc[worst_month_idx]['Total_Revenue']
            }
        }
        
        self.analysis_results['growth_metrics'] = growth_metrics
        return growth_metrics
    
    def analyze_seasonal_patterns(self):
        """Analyze seasonal sales patterns"""
        # Monthly patterns
        monthly_pattern = self.data.groupby(self.data['Date'].dt.month).agg({
            'Revenue': 'mean',
            'Quantity': 'mean'
        }).round(2)
        monthly_pattern.index = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        # Weekday patterns
        weekday_pattern = self.data.groupby('Weekday').agg({
            'Revenue': 'mean',
            'Quantity': 'mean'
        }).round(2)
        
        # Reorder weekdays
        weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekday_pattern = weekday_pattern.reindex(weekday_order)
        
        seasonal_patterns = {
            'monthly': monthly_pattern,
            'weekday': weekday_pattern,
            'peak_month': monthly_pattern['Revenue'].idxmax(),
            'peak_weekday': weekday_pattern['Revenue'].idxmax(),
            'seasonality_strength': monthly_pattern['Revenue'].std() / monthly_pattern['Revenue'].mean()
        }
        
        self.analysis_results['seasonal_patterns'] = seasonal_patterns
        return seasonal_patterns
    
    def generate_comprehensive_insights(self):
        """Generate comprehensive business insights"""
        print("🔍 Generating comprehensive business insights...")
        
        # Run all analyses
        monthly_revenue = self.calculate_monthly_revenue()
        product_performance = self.calculate_product_performance()
        customer_segments = self.analyze_customer_segments()
        growth_metrics = self.calculate_growth_metrics()
        seasonal_patterns = self.analyze_seasonal_patterns()
        
        # Basic metrics
        total_revenue = self.data['Revenue'].sum()
        total_orders = len(self.data)
        avg_order_value = self.data['Revenue'].mean()
        unique_customers = self.data['CustomerID'].nunique()
        unique_products = self.data['ProductID'].nunique()
        
        # Date range
        date_range = self.data['Date'].max() - self.data['Date'].min()
        
        # Top performers
        top_product = product_performance.iloc[0]
        
        # Customer insights
        segment_distribution = customer_segments['Spending_Segment'].value_counts()
        lifecycle_distribution = customer_segments['Lifecycle_Stage'].value_counts()
        
        # Regional insights (if available)
        if 'CustomerRegion' in self.data.columns:
            regional_revenue = self.data.groupby('CustomerRegion')['Revenue'].sum().sort_values(ascending=False)
            top_region = regional_revenue.index[0]
            top_region_revenue = regional_revenue.iloc[0]
        else:
            top_region = "N/A"
            top_region_revenue = 0
        
        # Compile insights
        insights = {
            "💰 Total Revenue": f"${total_revenue:,.2f}",
            "🛒 Total Orders": f"{total_orders:,}",
            "💵 Average Order Value": f"${avg_order_value:.2f}",
            "👥 Unique Customers": f"{unique_customers:,}",
            "📦 Products Sold": f"{unique_products:,}",
            "📅 Analysis Period": f"{date_range.days} days",
            "🏆 Top Product": f"{top_product['ProductName']} (${top_product['Total_Revenue']:,.2f})",
            "🌍 Top Region": f"{top_region} (${top_region_revenue:,.2f})" if top_region != "N/A" else "N/A",
            "👑 VIP Customers": f"{segment_distribution.get('VIP', 0)} customers",
            "🔄 Loyal Customers": f"{lifecycle_distribution.get('Loyal Customer', 0)} customers",
            "📈 Revenue Growth": growth_metrics.get('avg_monthly_revenue_growth', 'N/A'),
            "📊 Peak Sales Month": seasonal_patterns['peak_month'],
            "📅 Peak Sales Day": seasonal_patterns['peak_weekday'],
            "🎯 Customer Retention": f"{(lifecycle_distribution.get('Regular Customer', 0) + lifecycle_distribution.get('Loyal Customer', 0)) / len(customer_segments) * 100:.1f}%"
        }
        
        self.insights = insights
        return insights
    
    def generate_detailed_report(self, file_path, insights=None):
        """Generate a comprehensive business report"""
        if insights is None:
            insights = self.generate_comprehensive_insights()
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("COMPREHENSIVE SALES ANALYSIS REPORT\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Analysis Period: {self.data['Date'].min().strftime('%Y-%m-%d')} to {self.data['Date'].max().strftime('%Y-%m-%d')}\n\n")
            
            # Executive Summary
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-" * 30 + "\n")
            for key, value in insights.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
            
            # Top Products
            f.write("TOP 10 PRODUCTS BY REVENUE\n")
            f.write("-" * 35 + "\n")
            top_products = self.analysis_results.get('product_performance', pd.DataFrame()).head(10)
            for _, product in top_products.iterrows():
                f.write(f"{product['ProductName']}: ${product['Total_Revenue']:,.2f} "
                       f"({product['Market_Share']:.1f}% market share)\n")
            f.write("\n")
            
            # Customer Segments
            f.write("CUSTOMER SEGMENT DISTRIBUTION\n")
            f.write("-" * 35 + "\n")
            if 'customer_segments' in self.analysis_results:
                segment_dist = self.analysis_results['customer_segments']['Spending_Segment'].value_counts()
                for segment, count in segment_dist.items():
                    percentage = (count / len(self.analysis_results['customer_segments'])) * 100
                    f.write(f"{segment}: {count} customers ({percentage:.1f}%)\n")
            f.write("\n")
            
            # Growth Analysis
            f.write("GROWTH METRICS\n")
            f.write("-" * 20 + "\n")
            if 'growth_metrics' in self.analysis_results:
                growth = self.analysis_results['growth_metrics']
                f.write(f"Total Revenue Growth: {growth.get('total_revenue_growth', 'N/A')}\n")
                f.write(f"Average Monthly Growth: {growth.get('avg_monthly_revenue_growth', 'N/A')}\n")
                f.write(f"Customer Growth Rate: {growth.get('avg_monthly_customer_growth', 'N/A')}\n")
            f.write("\n")
            
            # Seasonal Insights
            f.write("SEASONAL PATTERNS\n")
            f.write("-" * 20 + "\n")
            if 'seasonal_patterns' in self.analysis_results:
                patterns = self.analysis_results['seasonal_patterns']
                f.write(f"Peak Sales Month: {patterns['peak_month']}\n")
                f.write(f"Peak Sales Weekday: {patterns['peak_weekday']}\n")
                f.write(f"Seasonality Strength: {patterns['seasonality_strength']:.2f}\n")
        
        print(f"✅ Detailed report saved to {file_path}")


# Example usage
if __name__ == "__main__":
    # This would typically be used with cleaned data from DataProcessor
    print("Sales Analyzer module - Run this with cleaned data from DataProcessor")