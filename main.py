from src.data_processor import DataProcessor
from src.sales_analyzer import SalesAnalyzer

# Step 1: Load & clean data
processor = DataProcessor("data/sample_sales_data.csv")
processor.load_data()
cleaned = processor.clean_data()

# Step 2: Analyze data
analyzer = SalesAnalyzer(cleaned)
insights = analyzer.generate_comprehensive_insights()

# Step 3: Save report
analyzer.generate_detailed_report("report/sales_summary.txt", insights)
