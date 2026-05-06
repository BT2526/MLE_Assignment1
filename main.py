import os
import glob
from datetime import datetime

import pyspark
import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DoubleType

import utils.data_processing_bronze_table
import utils.data_processing_silver_table
import utils.data_processing_gold_table

import utils.data_processing_bronze_table_features
import utils.data_processing_silver_table_features
import utils.data_processing_gold_table_features


# Initialize SparkSession
spark = pyspark.sql.SparkSession.builder \
    .appName("dev") \
    .master("local[*]") \
    .getOrCreate()

# Set log level to ERROR to hide warnings
spark.sparkContext.setLogLevel("ERROR")

# Set up config
snapshot_date_str = "2023-01-01"

start_date_str = "2023-01-01"
end_date_str = "2024-12-01"

# Create directories for lms and features
bronze_lms_directory = "datamart/bronze/lms/"
silver_loan_daily_directory = "datamart/silver/loan_daily/"
gold_label_store_directory = "datamart/gold/label_store/"

bronze_features_directory = "datamart/bronze/features/"
silver_features_directory = "datamart/silver/features/"
gold_features_directory = "datamart/gold/features/"

os.makedirs(bronze_lms_directory, exist_ok=True)
os.makedirs(silver_loan_daily_directory, exist_ok=True)
os.makedirs(gold_label_store_directory, exist_ok=True)

os.makedirs(bronze_features_directory, exist_ok=True)
os.makedirs(silver_features_directory, exist_ok=True)
os.makedirs(gold_features_directory, exist_ok=True)

# Generate list of dates to process
def generate_first_of_month_dates(start_date_str, end_date_str):
    # Convert the date strings to datetime objects
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    
    # List to store the first of month dates
    first_of_month_dates = []

    # Start from the first of the month of the start_date
    current_date = datetime(start_date.year, start_date.month, 1)

    while current_date <= end_date:
        # Append the date in yyyy-mm-dd format
        first_of_month_dates.append(current_date.strftime("%Y-%m-%d"))
        
        # Move to the first of the next month
        if current_date.month == 12:
            current_date = datetime(current_date.year + 1, 1, 1)
        else:
            current_date = datetime(current_date.year, current_date.month + 1, 1)

    return first_of_month_dates

dates_str_lst = generate_first_of_month_dates(start_date_str, end_date_str)
print(dates_str_lst)

# Build Bronze Tables & Run Bronze Backfill
for date_str in dates_str_lst:
    utils.data_processing_bronze_table.process_bronze_table(date_str, bronze_lms_directory, spark)
    utils.data_processing_bronze_table_features.process_bronze_table_features(date_str, bronze_features_directory, spark)


# Build Silver Tables & Run Silver Backfill
for date_str in dates_str_lst:
    utils.data_processing_silver_table.process_silver_table(date_str, bronze_lms_directory, silver_loan_daily_directory, spark)
    utils.data_processing_silver_table_features.process_silver_table_features(date_str, bronze_features_directory, silver_features_directory, spark)


# Build Gold Tables & Run Gold Backfill
for date_str in dates_str_lst:
    utils.data_processing_gold_table.process_labels_gold_table(date_str, silver_loan_daily_directory, gold_label_store_directory, spark, dpd = 30, mob = 6)
    utils.data_processing_gold_table_features.process_gold_features_table(date_str, silver_features_directory, gold_features_directory, spark)

# Inspect Label Store
folder_path = gold_label_store_directory
files_list = [folder_path+os.path.basename(f) for f in glob.glob(os.path.join(folder_path, '*'))]
df = spark.read.option("header", "true").parquet(*files_list)
print("row_count:",df.count())

df.show()

# Inspect Feature Store
folder_path = gold_features_directory
files_list = [folder_path+os.path.basename(f) for f in glob.glob(os.path.join(folder_path, '*'))]
df = spark.read.option("header", "true").parquet(*files_list)
print("row_count:",df.count())

df.show()