"""
Sales Visualizer Module - GitHub Codespaces Ready
Creates comprehensive visualizations for sales data analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import sys

# Configure matplotlib for headless environments (like Codespaces)
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('visualizer.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Set style for better-looking plots
try:
    plt.style.use('seaborn-v0_8')
except OSError:
    # Fallback if seaborn style is not available
    plt.style.use('default')
    logger.warning("seaborn-v0_8 style not available, using default")

sns.set_palette("husl")

class SalesVisualizerError(Exception):
    """Custom exception for SalesVisualizer errors"""
    pass

class SalesVisualizer:
    """Creates comprehensive visualizations for sales data"""
    
    def __init__(self, data: pd.DataFrame, output_dir: str = 'outputs/visualizations'):
        """
        Initialize the visualizer
        
        Args:
            data (pd.DataFrame): Cleaned sales data
            output_dir (str): Directory to save visualizations
        
        Raises:
            SalesVisualizerError: If required columns are missing or data is invalid
        """
        try:
            if data is None or data.empty:
                raise SalesVisualizerError("Data cannot be None or empty")
            
            self.data = data.copy()
            self.output_dir = Path(output_dir)
            
            # Create output directory if it doesn't exist
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Output directory created: {self.output_dir}")
            
            # Set up plotting parameters
            self.figsize = (12, 8)
            self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
            
            # Validate required columns
            self._validate_data()
            
        except Exception as e:
            logger.error(f"Error initializing SalesVisualizer: {str(e)}")
            raise SalesVisualizerError(f"Initialization failed: {str(e)}")
    
    def _validate_data(self):
        """Validate that required columns exist in the data"""
        # Check what columns we actually have
        logger.info(f"Available columns: {list(self.data.columns)}")
        
        # Required columns based on your project description
        required_cols = ['Date', 'Revenue', 'CustomerID', 'ProductID']
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            logger.info("Available columns in data:")
            for col in self.data.columns:
                logger.info(f"  - {col}")
            raise SalesVisualizerError(f"Missing required columns: {missing_cols}")
        
        # Validate data types
        if not pd.api.types.is_datetime64_any_dtype(self.data['Date']):
            logger.warning("Date column is not datetime type, attempting conversion")
            try:
                self.data['Date'] = pd.to_datetime(self.data['Date'])
            except Exception as e:
                raise SalesVisualizerError(f"Cannot convert Date column to datetime: {str(e)}")
        
        # Check for missing critical data
        if self.data['Revenue'].isna().all():
            raise SalesVisualizerError("Revenue column contains no valid data")
        
        # Add derived columns if missing
        if 'Weekday' not in self.data.columns:
            self.data['Weekday'] = self.data['Date'].dt.day_name()
            logger.info("Added Weekday column")
        
        # Handle missing ProductName - use ProductID if ProductName doesn't exist
        if 'ProductName' not in self.data.columns:
            if 'ProductID' in self.data.columns:
                self.data['ProductName'] = 'Product_' + self.data['ProductID'].astype(str)
                logger.info("Created ProductName column from ProductID")
            else:
                raise SalesVisualizerError("Neither ProductName nor ProductID found")
        
        # Handle missing Quantity column
        if 'Quantity' not in self.data.columns:
            self.data['Quantity'] = 1  # Assume 1 item per transaction
            logger.info("Created Quantity column with default value 1")
        
        logger.info(f"Data validation completed. Shape: {self.data.shape}")
    
    def save_plot(self, filename: str, dpi: int = 300, bbox_inches: str = 'tight') -> bool:
        """
        Save the current plot with consistent formatting
        
        Args:
            filename (str): Name of the file to save
            dpi (int): DPI for the saved image
            bbox_inches (str): Bbox inches parameter for matplotlib
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches, facecolor='white')
            logger.info(f"✅ Plot saved: {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving plot {filename}: {str(e)}")
            return False
    
    def plot_revenue_trend(self):
        """Create comprehensive revenue trend analysis"""
        try:
            logger.info("Creating revenue trend analysis...")
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Revenue Trend Analysis', fontsize=16, fontweight='bold')
            
            # 1. Monthly Revenue Trend
            monthly_revenue = self.data.groupby(self.data['Date'].dt.to_period('M'))['Revenue'].sum()
            monthly_revenue.index = monthly_revenue.index.astype(str)
            
            if len(monthly_revenue) > 0:
                ax1.plot(monthly_revenue.index, monthly_revenue.values, marker='o', linewidth=2, markersize=6)
                ax1.set_title('Monthly Revenue Trend', fontweight='bold')
                ax1.set_xlabel('Month')
                ax1.set_ylabel('Revenue ($)')
                ax1.tick_params(axis='x', rotation=45)
                ax1.grid(True, alpha=0.3)
                ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
            else:
                ax1.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax1.transAxes)
            
            # 2. Daily Revenue with Moving Average
            daily_revenue = self.data.groupby('Date')['Revenue'].sum()
            if len(daily_revenue) > 7:
                moving_avg = daily_revenue.rolling(window=7).mean()
                
                ax2.plot(daily_revenue.index, daily_revenue.values, alpha=0.3, color='lightblue', label='Daily Revenue')
                ax2.plot(moving_avg.index, moving_avg.values, color='darkblue', linewidth=2, label='7-day Moving Average')
                ax2.set_title('Daily Revenue with Moving Average', fontweight='bold')
                ax2.set_xlabel('Date')
                ax2.set_ylabel('Revenue ($)')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
            else:
                ax2.text(0.5, 0.5, 'Insufficient data for moving average', ha='center', va='center', transform=ax2.transAxes)
            
            # 3. Quarterly Comparison
            quarterly_data = self.data.groupby([self.data['Date'].dt.year, self.data['Date'].dt.quarter])['Revenue'].sum()
            if len(quarterly_data) > 0:
                quarterly_labels = [f"{year}-Q{quarter}" for year, quarter in quarterly_data.index]
                
                bars = ax3.bar(range(len(quarterly_data)), quarterly_data.values, color=self.colors[0], alpha=0.7)
                ax3.set_xticks(range(len(quarterly_data)))
                ax3.set_xticklabels(quarterly_labels, rotation=45)
                ax3.set_title('Quarterly Revenue Comparison', fontweight='bold')
                ax3.set_xlabel('Quarter')
                ax3.set_ylabel('Revenue ($)')
                ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
                
                # Add value labels on bars
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    ax3.text(bar.get_x() + bar.get_width()/2., height,
                            f'${height:,.0f}', ha='center', va='bottom', fontsize=9)
            
            # 4. Revenue Growth Rate
            if len(monthly_revenue) > 1:
                monthly_growth = monthly_revenue.pct_change() * 100
                monthly_growth = monthly_growth.dropna()
                
                if len(monthly_growth) > 0:
                    colors = ['green' if x > 0 else 'red' for x in monthly_growth.values]
                    bars = ax4.bar(range(len(monthly_growth)), monthly_growth.values, color=colors, alpha=0.7)
                    ax4.set_xticks(range(len(monthly_growth)))
                    ax4.set_xticklabels(monthly_growth.index, rotation=45)
                    ax4.set_title('Month-over-Month Growth Rate', fontweight='bold')
                    ax4.set_xlabel('Month')
                    ax4.set_ylabel('Growth Rate (%)')
                    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                    ax4.grid(True, alpha=0.3)
                    
                    # Add value labels on bars
                    for i, bar in enumerate(bars):
                        height = bar.get_height()
                        ax4.text(bar.get_x() + bar.get_width()/2., height,
                                f'{height:.1f}%', ha='center', va='bottom' if height > 0 else 'top', fontsize=8)
            
            plt.tight_layout()
            success = self.save_plot('revenue_trend_analysis.png')
            plt.close()  # Close the figure to free memory
            
            if success:
                logger.info("✅ Revenue trend analysis completed")
            
        except Exception as e:
            logger.error(f"Error creating revenue trend plot: {str(e)}")
            plt.close()  # Ensure we close the figure even on error
            raise SalesVisualizerError(f"Revenue trend visualization failed: {str(e)}")
    
    def plot_product_performance(self):
        """Create comprehensive product performance visualization"""
        try:
            logger.info("Creating product performance analysis...")
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Product Performance Analysis', fontsize=16, fontweight='bold')
            
            # Prepare product data
            product_stats = self.data.groupby('ProductName').agg({
                'Revenue': 'sum',
                'Quantity': 'sum',
                'CustomerID': 'nunique'
            }).round(2)
            
            # 1. Top 10 Products by Revenue
            if len(product_stats) > 0:
                top_10_revenue = product_stats.nlargest(min(10, len(product_stats)), 'Revenue')
                bars = ax1.barh(range(len(top_10_revenue)), top_10_revenue['Revenue'], color=self.colors[1])
                ax1.set_yticks(range(len(top_10_revenue)))
                ax1.set_yticklabels([name[:20] + '...' if len(name) > 20 else name for name in top_10_revenue.index], fontsize=10)
                ax1.set_title('Top Products by Revenue', fontweight='bold')
                ax1.set_xlabel('Revenue ($)')
                ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
                
                # Add value labels
                for i, bar in enumerate(bars):
                    width = bar.get_width()
                    ax1.text(width, bar.get_y() + bar.get_height()/2,
                            f'${width:,.0f}', ha='left', va='center', fontsize=9)
            
            # 2. Product Category Performance (if available)
            if 'Category' in self.data.columns:
                category_stats = self.data.groupby('Category')['Revenue'].sum()
                if len(category_stats) > 0:
                    category_stats = category_stats.sort_values(ascending=False)
                    
                    wedges, texts, autotexts = ax2.pie(category_stats.values, labels=category_stats.index, 
                                                      autopct='%1.1f%%', colors=self.colors[:len(category_stats)])
                    ax2.set_title('Revenue by Product Category', fontweight='bold')
                    
                    # Improve text readability
                    for autotext in autotexts:
                        autotext.set_color('white')
                        autotext.set_fontweight('bold')
                else:
                    ax2.text(0.5, 0.5, 'No category data available', ha='center', va='center', transform=ax2.transAxes)
            else:
                ax2.text(0.5, 0.5, 'Category column not found', ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Revenue by Product Category', fontweight='bold')
            
            # 3. Quantity vs Revenue Scatter
            if len(product_stats) > 0:
                ax3.scatter(product_stats['Quantity'], product_stats['Revenue'], 
                           alpha=0.6, s=60, color=self.colors[2])
                ax3.set_xlabel('Total Quantity Sold')
                ax3.set_ylabel('Total Revenue ($)')
                ax3.set_title('Product Performance: Quantity vs Revenue', fontweight='bold')
                ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
                ax3.grid(True, alpha=0.3)
                
                # Add trend line if we have enough data points
                if len(product_stats) > 1:
                    try:
                        z = np.polyfit(product_stats['Quantity'], product_stats['Revenue'], 1)
                        p = np.poly1d(z)
                        ax3.plot(product_stats['Quantity'], p(product_stats['Quantity']), 
                                "r--", alpha=0.8, linewidth=2)
                    except:
                        pass  # Skip trend line if calculation fails
            
            # 4. Customer Reach by Product
            if len(product_stats) > 0:
                top_10_customers = product_stats.nlargest(min(10, len(product_stats)), 'CustomerID')
                bars = ax4.bar(range(len(top_10_customers)), top_10_customers['CustomerID'], 
                              color=self.colors[3], alpha=0.7)
                ax4.set_xticks(range(len(top_10_customers)))
                ax4.set_xticklabels([name[:15] + '...' if len(name) > 15 else name 
                                    for name in top_10_customers.index], rotation=45, ha='right')
                ax4.set_title('Top Products by Customer Reach', fontweight='bold')
                ax4.set_ylabel('Unique Customers')
                
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    ax4.text(bar.get_x() + bar.get_width()/2., height,
                            f'{int(height)}', ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            success = self.save_plot('product_performance_analysis.png')
            plt.close()  # Close the figure to free memory
            
            if success:
                logger.info("✅ Product performance analysis completed")
            
        except Exception as e:
            logger.error(f"Error creating product performance plot: {str(e)}")
            plt.close()  # Ensure we close the figure even on error
            raise SalesVisualizerError(f"Product performance visualization failed: {str(e)}")
    
    def create_all_visualizations(self):
        """Generate all visualizations with error handling"""
        logger.info("🎨 Creating comprehensive visualization suite...")
        
        success_count = 0
        total_plots = 2
        
        plots_to_create = [
            ("Revenue Trend Analysis", self.plot_revenue_trend),
            ("Product Performance Analysis", self.plot_product_performance),
        ]
        
        for plot_name, plot_function in plots_to_create:
            try:
                logger.info(f"Creating {plot_name}...")
                plot_function()
                success_count += 1
                logger.info(f"✅ {plot_name} completed successfully")
            except Exception as e:
                logger.error(f"❌ Failed to create {plot_name}: {str(e)}")
                continue
        
        logger.info(f"✅ {success_count}/{total_plots} visualizations created successfully!")
        logger.info(f"📁 Check the '{self.output_dir}' folder for all charts")
        
        return success_count == total_plots


def load_and_validate_data(csv_path: str) -> pd.DataFrame:
    """
    Load and validate CSV data with proper error handling
    
    Args:
        csv_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Validated dataframe
        
    Raises:
        SalesVisualizerError: If data loading or validation fails
    """
    try:
        # Check if file exists
        if not os.path.exists(csv_path):
            # Try alternative paths
            alternative_paths = [
                'data/cleaned_sales_data.csv',
                'cleaned_sales_data.csv',
                'sample_sales_data.csv',
                'data/sample_sales_data.csv'
            ]
            
            for alt_path in alternative_paths:
                if os.path.exists(alt_path):
                    csv_path = alt_path
                    logger.info(f"Using alternative path: {csv_path}")
                    break
            else:
                raise SalesVisualizerError(f"CSV file not found at: {csv_path} or alternative paths")
        
        # Load data
        df = pd.read_csv(csv_path)
        logger.info(f"Data loaded successfully from {csv_path}. Shape: {df.shape}")
        logger.info(f"Columns found: {list(df.columns)}")
        
        # Convert Date column
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        
        # Add missing fields if necessary
        if 'Weekday' not in df.columns and 'Date' in df.columns:
            df['Weekday'] = df['Date'].dt.day_name()
            logger.info("Added Weekday column")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise SalesVisualizerError(f"Data loading failed: {str(e)}")


def main():
    """Main function with proper error handling"""
    try:
        logger.info("🎯 Starting SalesVisualizer for GitHub Codespaces...")
        
        # Get script directory and build path to data
        script_dir = Path(__file__).resolve().parent
        
        # Try multiple possible data paths based on your project structure
        possible_paths = [
            script_dir / 'data' / 'cleaned_sales_data.csv',
            script_dir / '..' / 'data' / 'cleaned_sales_data.csv',
            script_dir / 'data' / 'sample_sales_data.csv',
            script_dir / '..' / 'data' / 'sample_sales_data.csv',
            script_dir / 'cleaned_sales_data.csv',
            script_dir / 'sample_sales_data.csv'
        ]
        
        csv_path = None
        for path in possible_paths:
            if path.exists():
                csv_path = str(path)
                break
        
        if csv_path is None:
            logger.error("No data file found. Looking for:")
            for path in possible_paths:
                logger.error(f"  - {path}")
            raise SalesVisualizerError("No data file found in expected locations")
        
        # Load and validate data
        df = load_and_validate_data(csv_path)
        
        # Create visualizer and generate plots
        visualizer = SalesVisualizer(df)
        success = visualizer.create_all_visualizations()
        
        if success:
            logger.info("✅ All visualizations generated successfully!")
            print("\n🎉 SUCCESS: All visualizations created!")
            print(f"📁 Check the 'outputs/visualizations' folder for your charts")
        else:
            logger.warning("⚠️  Some visualizations failed to generate")
            print("\n⚠️  WARNING: Some visualizations failed")
            
    except SalesVisualizerError as e:
        logger.error(f"❌ SalesVisualizer Error: {e}")
        print(f"\n❌ Error: {e}")
        return 1
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        print(f"\n❌ Unexpected error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())