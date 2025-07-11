"""
Data Processor Module
Handles data loading, cleaning, and preprocessing
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    """Handles all data preprocessing operations"""
    
    def __init__(self, file_path):
        """
        Initialize the data processor
        
        Args:
            file_path (str): Path to the CSV file
        """
        self.file_path = file_path
        self.raw_data = None
        self.cleaned_data = None
        self.cleaning_report = {}
    
    def load_data(self):
        """Load data from CSV file"""
        try:
            self.raw_data = pd.read_csv(self.file_path)
            print(f"✅ Data loaded: {self.raw_data.shape[0]} rows, {self.raw_data.shape[1]} columns")
            return self.raw_data
        except FileNotFoundError:
            print("📁 Data file not found. Generating sample data...")
            self._generate_sample_data()
            return self.raw_data
    
    def clean_data(self):
        """Clean and preprocess the data"""
        if self.raw_data is None:
            self.load_data()
        
        print("🧹 Starting data cleaning process...")
        
        # Create a copy for cleaning
        df = self.raw_data.copy()
        initial_rows = len(df)
        
        # 1. Convert date column to datetime
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            print("✅ Date column converted to datetime")
        
        # 2. Handle missing values
        missing_before = df.isnull().sum().sum()
        
        # Drop rows with missing critical information
        df = df.dropna(subset=['CustomerID', 'ProductID'])
        
        # Fill missing values with appropriate defaults
        df['Quantity'] = df['Quantity'].fillna(0)
        df['Price'] = df['Price'].fillna(df['Price'].mean())
        
        missing_after = df.isnull().sum().sum()
        print(f"✅ Missing values: {missing_before} → {missing_after}")
        
        # 3. Remove invalid data
        df = df[(df['Quantity'] > 0) & (df['Price'] > 0)]
        valid_rows = len(df)
        
        # 4. Calculate derived columns
        df['Revenue'] = df['Quantity'] * df['Price']
        df['Month'] = df['Date'].dt.to_period('M')
        df['Year'] = df['Date'].dt.year
        df['Quarter'] = df['Date'].dt.quarter
        df['Weekday'] = df['Date'].dt.day_name()
        df['Month_Name'] = df['Date'].dt.month_name()
        
        print("✅ Derived columns added: Revenue, Month, Year, Quarter, Weekday")
        
        # 5. Remove outliers using IQR method
        Q1 = df['Revenue'].quantile(0.25)
        Q3 = df['Revenue'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers_removed = len(df[(df['Revenue'] < lower_bound) | (df['Revenue'] > upper_bound)])
        df = df[~((df['Revenue'] < lower_bound) | (df['Revenue'] > upper_bound))]
        
        final_rows = len(df)
        
        # Store cleaning report
        self.cleaning_report = {
            'initial_rows': initial_rows,
            'after_missing_removal': len(df),
            'after_invalid_removal': valid_rows,
            'outliers_removed': outliers_removed,
            'final_rows': final_rows,
            'data_retention_rate': f"{(final_rows/initial_rows)*100:.1f}%"
        }
        
        print(f"✅ Outliers removed: {outliers_removed}")
        print(f"✅ Data cleaning complete: {final_rows} records remaining ({self.cleaning_report['data_retention_rate']} retention)")
        
        self.cleaned_data = df
        return df
    
    def get_data_summary(self):
        """Get summary statistics of the cleaned data"""
        if self.cleaned_data is None:
            raise ValueError("No cleaned data available. Run clean_data() first.")
        
        summary = {
            'shape': self.cleaned_data.shape,
            'date_range': {
                'start': self.cleaned_data['Date'].min(),
                'end': self.cleaned_data['Date'].max(),
                'days': (self.cleaned_data['Date'].max() - self.cleaned_data['Date'].min()).days
            },
            'revenue_stats': {
                'total': self.cleaned_data['Revenue'].sum(),
                'mean': self.cleaned_data['Revenue'].mean(),
                'median': self.cleaned_data['Revenue'].median(),
                'std': self.cleaned_data['Revenue'].std()
            },
            'unique_counts': {
                'customers': self.cleaned_data['CustomerID'].nunique(),
                'products': self.cleaned_data['ProductID'].nunique(),
                'categories': self.cleaned_data['Category'].nunique() if 'Category' in self.cleaned_data.columns else 0
            }
        }
        
        return summary
    
    def _generate_sample_data(self):
        """Generate realistic sample sales data for demonstration"""
        print("🎲 Generating sample sales data...")
        
        np.random.seed(42)  # For reproducible results
        
        # Parameters for realistic data
        n_customers = 1000
        n_products = 50
        n_records = 15000
        
        # Generate date range (2 years of data)
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2024, 12, 31)
        
        # Create seasonal patterns
        dates = pd.date_range(start_date, end_date, freq='D')
        
        # Generate realistic sample data
        categories = ['Electronics', 'Clothing', 'Books', 'Home & Garden', 'Sports', 'Beauty', 'Toys']
        regions = ['North America', 'Europe', 'Asia', 'South America', 'Africa', 'Oceania']
        
        # Generate data with some realistic patterns
        data = []
        
        for i in range(n_records):
            # Random date with some seasonal weighting
            date = pd.to_datetime(np.random.choice(dates)).to_pydatetime()

            
            # Seasonal adjustment for quantities (higher in Nov-Dec)
            seasonal_multiplier = 1.5 if date.month in [11, 12] else 1.0
            
            # Generate record
            record = {
                'Date': date,
                'CustomerID': np.random.randint(1, n_customers + 1),
                'ProductID': np.random.randint(1, n_products + 1),
                'ProductName': f'Product_{np.random.randint(1, n_products + 1)}',
                'Category': np.random.choice(categories),
                'Quantity': max(1, int(np.random.poisson(2) * seasonal_multiplier)),
                'Price': round(np.random.gamma(2, 25), 2),  # Realistic price distribution
                'CustomerRegion': np.random.choice(regions)
            }
            data.append(record)
        
        self.raw_data = pd.DataFrame(data)
        
        # Introduce some realistic missing data patterns
        missing_indices = np.random.choice(len(self.raw_data), size=int(0.02 * len(self.raw_data)), replace=False)
        self.raw_data.loc[missing_indices, 'Price'] = np.nan
        
        # Save the generated data
        os.makedirs('data', exist_ok=True)
        self.raw_data.to_csv(self.file_path, index=False)
        print(f"✅ Sample data generated and saved: {len(self.raw_data)} records")
        
        return self.raw_data
    
    def export_cleaned_data(self, output_path=None):
        """Export cleaned data to CSV"""
        if self.cleaned_data is None:
            raise ValueError("No cleaned data available. Run clean_data() first.")
        
        if output_path is None:
            output_path = 'data/cleaned_sales_data.csv'
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.cleaned_data.to_csv(output_path, index=False)
        print(f"✅ Cleaned data exported to {output_path}")


# Example usage and testing
if __name__ == "__main__":
    # Test the data processor
    processor = DataProcessor('data/sample_sales_data.csv')
    
    # Load and clean data
    raw_data = processor.load_data()
    cleaned_data = processor.clean_data()
    
    # Get summary
    summary = processor.get_data_summary()
    
    print("\n📊 Data Summary:")
    print(f"Shape: {summary['shape']}")
    print(f"Date Range: {summary['date_range']['start'].strftime('%Y-%m-%d')} to {summary['date_range']['end'].strftime('%Y-%m-%d')}")
    print(f"Total Revenue: ${summary['revenue_stats']['total']:,.2f}")
    print(f"Unique Customers: {summary['unique_counts']['customers']}")
    print(f"Unique Products: {summary['unique_counts']['products']}")
    
    # Export cleaned data
    processor.export_cleaned_data()