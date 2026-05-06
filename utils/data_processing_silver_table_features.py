import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F
import argparse

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType, DoubleType

# Global Schema Definitions
# Dictionary specifying columns and their desired datatypes
column_type_map = {
    "Customer_ID": StringType(),
    "Name": StringType(),
    "Age": IntegerType(),
    "Occupation": StringType(),
    "Annual_Income": FloatType(),
    "Monthly_Inhand_Salary": FloatType(),
    "Num_Bank_Accounts": IntegerType(),
    "Num_Credit_Card": IntegerType(),
    "Interest_Rate": IntegerType(),
    "Num_of_Loan": IntegerType(),
    "Type_of_Loan": StringType(),
    "Delay_from_due_date": IntegerType(),
    "Num_of_Delayed_Payment": IntegerType(),
    "Changed_Credit_Limit": FloatType(),
    "Num_Credit_Inquiries": IntegerType(),
    "Credit_Mix": StringType(),
    "Outstanding_Debt": FloatType(),
    "Credit_Utilization_Ratio": FloatType(),
    "Credit_History_Age": DoubleType(),
    "Payment_of_Min_Amount": StringType(),
    "Total_EMI_per_month": FloatType(),
    "Amount_invested_monthly": FloatType(),
    "Payment_Behaviour": StringType(),
    "Monthly_Balance": FloatType(),
    "snapshot_date": StringType(), 
}

def enforce_column_types(df):    
    for column, new_type in column_type_map.items():
        if column in df.columns:
            df = df.withColumn(column, col(column).cast(new_type))

    return df

def clean_numerical_columns(df, column_type_map):
    """
    Clean numerical columns by 
    - Removing non-numeric characters (special characters e.g. _, !@#$, etc.) except negative sign and decimals
    - Cast to float
    """
    for col_name, col_type in column_type_map.items():
        # Check if the map defines the column as numeric
        if col_name in df.columns and isinstance(col_type, (FloatType, IntegerType)):
            # Remove special characters from the column except negative sign and decimals
            df = df.withColumn(col_name, F.regexp_replace(col(col_name).cast(StringType()), r"[^0-9.\-]", ""))
            # Cast the column to the defined type
            df = df.withColumn(col_name, col(col_name).cast(col_type))
            # Round floating point columns to 2 decimal places
            if isinstance(col_type, FloatType):
                df = df.withColumn(col_name, F.round(col(col_name), 2))
    return df

def clean_categorical_columns(df, column_type_map):
    """
    Clean categorical columns by 
    - Lowercasing
    - Trimming whitespace
    - Replacing nulls, blanks, and symbol-only values with 'na'
    """
    for col_name, col_type in column_type_map.items():
        if col_name.lower() == 'customer_id':
            continue

        # Check if the map defines the column as categorical
        if col_name in df.columns and isinstance(col_type, StringType):
            # Lowercase and trim whitespace
            df = df.withColumn(col_name, F.trim(F.lower(col(col_name))))
            # Replace nulls, blanks, and special characters with 'na'
            df = df.withColumn(col_name, F.when(
                F.col(col_name).isNull() | (F.col(col_name) == "") |
                (F.regexp_replace(F.col(col_name), r"[^a-zA-Z0-9]", "") == ""),
                "na"
            ).otherwise(F.col(col_name)))
    return df

def clean_type_of_loan(df):
    # Lowercase and replace 'and' with a comma to create a uniform separator
    df = df.withColumn("Type_of_Loan", F.lower(F.col("Type_of_Loan")))
    df = df.withColumn("Type_of_Loan", F.regexp_replace(F.col("Type_of_Loan"), r"\band\b", ","))
    
    # Split into an array by comma
    df = df.withColumn("Type_of_Loan", F.split(F.col("Type_of_Loan"), ","))

    # Clean each element in the array:
    #    - trim(): removes the extra spaces left behind by the 'and' replacement
    #    - filter(): removes blanks and 'not specified' entries
    #    - array_distinct(): removes duplicates (e.g., if it was 'Auto Loan, Auto Loan')
    df = df.withColumn("Type_of_Loan", F.expr("""
        array_distinct(
            filter(
                transform(Type_of_Loan, x -> trim(x)),
                x -> x != '' AND x != 'Not Specified' AND x IS NOT NULL
            )
        )
    """))
    
    # Replace the original messy column with a clean string version
    df = df.withColumn("Type_of_Loan", F.array_join(F.col("Type_of_Loan"), ", "))    
    
    return df

def clean_credit_history_age(df):
    # Extract the digits for years and months
    df = df.withColumn("years", F.regexp_extract(F.col("Credit_History_Age"), r"(\d+)\s+Years", 1).cast(IntegerType()))
    df = df.withColumn("months", F.regexp_extract(F.col("Credit_History_Age"), r"(\d+)\s+Months", 1).cast(IntegerType()))
    
    # Fill missing extracts with 0 to prevent nulls
    df = df.fillna(0, subset=["years", "months"])
    
    # Calculate total months
    df = df.withColumn("Credit_History_Age", (F.col("years") * 12 + F.col("months")).cast(IntegerType()))
    
    # Cleanup
    df = df.drop("years", "months")

    return df

def clean_payment_of_min_amount(df):
    df = df.withColumn("Payment_of_Min_Amount", F.trim(F.lower(F.col("Payment_of_Min_Amount"))))
    df = df.withColumn("Payment_of_Min_Amount", F.when(
        F.col("Payment_of_Min_Amount").isin(["yes", "no"]),
        F.col("Payment_of_Min_Amount")
    ).otherwise("na"))
    return df

def clean_payment_behaviour(df):
    # Standardizes the Payment_Behaviour column by ensuring only valid categories exist
    valid_behaviours = [
        "low_spent_small_value_payments",
        "low_spent_medium_value_payments",
        "low_spent_large_value_payments",
        "high_spent_small_value_payments",
        "high_spent_medium_value_payments",
        "high_spent_large_value_payments"
    ]
    # Remove whitespace and lowercase the strings to match the list (valid behaviours) above
    df = df.withColumn("Payment_Behaviour", F.trim(F.lower(F.col("Payment_Behaviour"))))
    # Keep only the valid behaviours
    df = df.withColumn("Payment_Behaviour", F.when(
        F.col("Payment_Behaviour").isin(valid_behaviours),
        F.col("Payment_Behaviour")
    ).otherwise("unknown"))
    
    return df

def process_silver_table_features(snapshot_date_str, bronze_features_directory, silver_features_directory, spark):
    # Prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # Connect to bronze table
    # Reconstruct the 3 filenames for this specific snapshot date
    date_suffix = snapshot_date_str.replace('-', '_') + ".csv"
    path_click = os.path.join(bronze_features_directory, f"bronze_table_features_clickstream_{date_suffix}")
    path_attr  = os.path.join(bronze_features_directory, f"bronze_table_features_attributes_{date_suffix}")
    path_fin   = os.path.join(bronze_features_directory, f"bronze_table_features_financials_{date_suffix}")
    # Read the files into separate DataFrames
    df_clickstream = spark.read.csv(path_click, header=True, inferSchema=True)
    df_attributes  = spark.read.csv(path_attr,  header=True, inferSchema=True)
    df_financials  = spark.read.csv(path_fin,   header=True, inferSchema=True)

    # Lowercase Customer_ID for all dataframes to ensure consistency
    df_clickstream = df_clickstream.withColumn('Customer_ID', F.trim(F.lower(F.col('Customer_ID').cast("string"))))
    df_attributes = df_attributes.withColumn('Customer_ID', F.trim(F.lower(F.col('Customer_ID').cast("string"))))
    df_financials = df_financials.withColumn('Customer_ID', F.trim(F.lower(F.col('Customer_ID').cast("string"))))

    # Drop redundant columns that are not needed in the silver table
    df_clickstream = df_clickstream.drop('snapshot_date')
    df_attributes = df_attributes.drop('Name', 'SSN', 'snapshot_date')
    df_financials = df_financials.drop('snapshot_date')

    # Clean numerical columns
    df_clickstream = clean_numerical_columns(df_clickstream, column_type_map)
    df_attributes = clean_numerical_columns(df_attributes, column_type_map)
    df_financials = clean_numerical_columns(df_financials, column_type_map)

    # Clean categorical columns
    df_clickstream = clean_categorical_columns(df_clickstream, column_type_map)
    df_attributes = clean_categorical_columns(df_attributes, column_type_map)
    df_financials = clean_categorical_columns(df_financials, column_type_map)

    # Process columns: Type_of_Loan, Credit_History_Age, Payment_of_Min_Amount, Payment_Behaviour
    df_financials = clean_type_of_loan(df_financials)
    df_financials = clean_credit_history_age(df_financials)
    df_financials = clean_payment_of_min_amount(df_financials)
    df_financials = clean_payment_behaviour(df_financials)

    # Enforce column types
    df_clickstream = enforce_column_types(df_clickstream)
    df_attributes = enforce_column_types(df_attributes)
    df_financials = enforce_column_types(df_financials)

    # Cast fe_ cols to float
    for i in range(1, 21):
        df_clickstream = df_clickstream.withColumn(f"fe_{i}", F.col(f"fe_{i}").cast(FloatType()))
    
    # Add snapshot date
    df_clickstream = df_clickstream.withColumn('snapshot_date', F.lit(snapshot_date_str))
    df_attributes = df_attributes.withColumn('snapshot_date', F.lit(snapshot_date_str))
    df_financials = df_financials.withColumn('snapshot_date', F.lit(snapshot_date_str))

    # Count no. of rows for each dataframe
    print(f"[{snapshot_date_str}] clickstream: {df_clickstream.count()} rows")
    print(f"[{snapshot_date_str}] attributes:  {df_attributes.count()} rows")
    print(f"[{snapshot_date_str}] financials:  {df_financials.count()} rows")

    # Save silver table - IRL connect to database to write
    for name, df in [("clickstream", df_clickstream), ("attributes", df_attributes), ("financials", df_financials)]:
        partition_name = f"silver_features_{name}_{snapshot_date_str.replace('-','_')}.parquet"
        filepath = os.path.join(silver_features_directory, partition_name)
        df.write.mode("overwrite").parquet(filepath)
        print(f"saved to: {filepath}")

    return df_clickstream, df_attributes, df_financials