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

# Pad fe_ cols with null for customers with no clickstream match
def pad_features(df):
    """Add fe_1 to fe_20 as null DoubleType if not present (customers with no clickstream match)."""
    for i in range(1, 21):
        col_name = f"fe_{i}"
        if col_name not in df.columns:
            df = df.withColumn(col_name, F.lit(None).cast(DoubleType()))
        else:
            df = df.withColumn(col_name, F.col(col_name).cast(DoubleType()))
    return df

# Clamp outliers to [1st, 99th] percentile bounds. Values outside of [1st, 99th] percentile will be set to 1st and 99th percentile values respectively.
def clamp_outliers(df):
    clamp_cols = [
        'Annual_Income',
        'Monthly_Inhand_Salary',
        'Num_Bank_Accounts',
        'Num_Credit_Card',
        'Interest_Rate',
        'Num_of_Loan',
        'Delay_from_due_date',
        'Num_of_Delayed_Payment',
        'Num_Credit_Inquiries',
        'Outstanding_Debt',
        'Total_EMI_per_month',
        'Amount_invested_monthly',
        'Monthly_Balance',
        'Changed_Credit_Limit',
        'Credit_History_Age',
    ]
 
    for c in clamp_cols:
        if c not in df.columns:
            continue
        # Compute percentile bounds
        bounds = df.approxQuantile(c, [0.01, 0.99], 0.001)
        p1, p99 = bounds[0], bounds[1]
        df = df.withColumn(
            c,
            F.when(F.col(c) < p1, p1)
             .when(F.col(c) > p99, p99)
             .otherwise(F.col(c))
            .cast(DoubleType())
        )
        print(f"  clamped {c}: [{p1:.2f}, {p99:.2f}]")
 
    return df

# Encode credit_mix column with ordinal encoding: bad=0, standard=1, good=2, na=-1 (unknown)
def encode_credit_mix(df):

    df = df.withColumn('Credit_Mix', F.trim(F.lower(F.col('Credit_Mix'))))
    df = df.withColumn('Credit_Mix_encoded', 
        F.when(F.col('Credit_Mix') == 'bad',      0)
         .when(F.col('Credit_Mix') == 'standard', 1)
         .when(F.col('Credit_Mix') == 'good',     2)
         .otherwise(-1)  # 'na' / unknown
        .cast(IntegerType())
    )
    return df

# Encode payment_of_min_amount with binary encoding: yes=1, no=0, na=-1 (unknown)
def encode_payment_of_min_amount(df):
    df = df.withColumn('Payment_of_Min_Amount', F.trim(F.lower(F.col('Payment_of_Min_Amount'))))
    df = df.withColumn('Payment_of_Min_Amount_encoded',
        F.when(F.col('Payment_of_Min_Amount') == 'yes', 1)
         .when(F.col('Payment_of_Min_Amount') == 'no',  0)
         .otherwise(-1)  # 'na' / unknown
        .cast(IntegerType())
    )
    return df

# Encode payment_behaviour 
def encode_payment_behaviour(df):
    """
    Ordinal encode Payment_Behaviour by combining spend level and payment size
    Encodes two sub-dimensions:
      - spent_level:    low=0, high=1
      - payment_size:   small=0, medium=1, large=2
    Unknown / corrupt values -> -1
    """
    df = df.withColumn('Payment_Behaviour', F.trim(F.lower(F.col('Payment_Behaviour'))))
 
    df = df.withColumn('pb_spent_level',
        F.when(F.col('Payment_Behaviour').startswith('low_spent'),  0)
         .when(F.col('Payment_Behaviour').startswith('high_spent'), 1)
         .otherwise(-1)
        .cast(IntegerType())
    )
    df = df.withColumn('pb_payment_size',
        F.when(F.col('Payment_Behaviour').endswith('small_value_payments'),  0)
         .when(F.col('Payment_Behaviour').endswith('medium_value_payments'), 1)
         .when(F.col('Payment_Behaviour').endswith('large_value_payments'),  2)
         .otherwise(-1)
        .cast(IntegerType())
    )
    return df

# Encode occupation
def encode_occupation(df):
    valid_occupations = [
        'lawyer', 'architect', 'engineer', 'accountant', 'scientist',
        'teacher', 'mechanic', 'media_manager', 'developer', 'doctor',
        'manager', 'entrepreneur', 'journalist', 'musician', 'writer', 'nurse'
    ]
    df = df.withColumn('Occupation', F.trim(F.lower(F.col('Occupation'))))
    # Normalise invalid entries to 'unknown'
    df = df.withColumn('Occupation',
        F.when(F.col('Occupation').isin(valid_occupations), F.col('Occupation'))
         .otherwise('unknown')
    )
    # One-hot encode
    for occ in valid_occupations:
        df = df.withColumn(
            f'occ_{occ}',
            F.when(F.col('Occupation') == occ, 1).otherwise(0).cast(IntegerType())
        )
    return df

# Feature engineering with new features that capture credit risk signals more directly
def engineer_features(df):
    # Debt burden relative to income
    df = df.withColumn('debt_to_income',
        F.when(F.col('Annual_Income') > 0,
               F.round(F.col('Outstanding_Debt') / F.col('Annual_Income'), 4))
         .otherwise(F.lit(None).cast(DoubleType()))
    )
 
    # Monthly repayment burden relative to take-home pay
    df = df.withColumn('emi_to_salary',
        F.when(F.col('Monthly_Inhand_Salary') > 0,
               F.round(F.col('Total_EMI_per_month') / F.col('Monthly_Inhand_Salary'), 4))
         .otherwise(F.lit(None).cast(DoubleType()))
    )
 
    # Savings rate: how much of salary is invested
    df = df.withColumn('savings_rate',
        F.when(F.col('Monthly_Inhand_Salary') > 0,
               F.round(F.col('Amount_invested_monthly') / F.col('Monthly_Inhand_Salary'), 4))
         .otherwise(F.lit(None).cast(DoubleType()))
    )
 
    # Payment stress: combines how often and how late payments are missed
    df = df.withColumn('payment_stress',
        (F.col('Delay_from_due_date') * F.col('Num_of_Delayed_Payment')).cast(DoubleType())
    )
 
    # Credit card utilisation per card: how spread out the debt is
    df = df.withColumn('loans_per_credit_card',
        F.when(F.col('Num_Credit_Card') > 0,
               F.round(F.col('Num_of_Loan') / F.col('Num_Credit_Card'), 4))
         .otherwise(F.lit(None).cast(DoubleType()))
    )
 
    # Monthly surplus: what's left after EMI obligations
    df = df.withColumn('monthly_surplus',
        F.round(F.col('Monthly_Inhand_Salary') - F.col('Total_EMI_per_month'), 2)
        .cast(DoubleType())
    )
 
    return df

# Drop columns not intended for further analysis
def drop_columns(df):
    """
    Drop columns that are:
    - PII (Customer_ID kept as index only, dropped before model training)
    - Raw categorical columns that have been encoded (keep only encoded versions)
    - Free-text / high-cardinality columns with no direct ML signal
    """
    cols_to_drop = [
        'Type_of_Loan',          # free-text multi-label, high cardinality
        'Credit_Mix',            # replaced by Credit_Mix_encoded
        'Payment_of_Min_Amount', # replaced by Payment_of_Min_Amount_encoded
        'Payment_Behaviour',     # replaced by pb_spent_level + pb_payment_size
        'Occupation',            # replaced by occ_* one-hot columns
    ]
    existing = [c for c in cols_to_drop if c in df.columns]
    return df.drop(*existing)

def fix_negative_values(df):
    """
    Replace nonsensical negative values with null.
    These columns have no valid negative interpretation.
    """
    # Counts — can never be negative
    count_cols = ['Num_of_Loan', 'Num_of_Delayed_Payment', 'Num_Bank_Accounts',
                  'Num_Credit_Card', 'Interest_Rate', 'Num_Credit_Inquiries',
                  'Delay_from_due_date', 'Credit_History_Age']
    for c in count_cols:
        if c in df.columns:
            df = df.withColumn(c,
                F.when(F.col(c) < 0, F.lit(None).cast(DoubleType()))
                 .otherwise(F.col(c))
            )

    # Negative values in Changed_Credit_Limit is valid as it represents limit was lowered
    # Extreme negatives like -100 are assessed to be data errors, hence null out below -50
    if 'Changed_Credit_Limit' in df.columns:
        df = df.withColumn('Changed_Credit_Limit',
            F.when(F.col('Changed_Credit_Limit') < -50, F.lit(None).cast(DoubleType()))
             .otherwise(F.col('Changed_Credit_Limit'))
        )

    return df   

def process_gold_features_table(snapshot_date_str, silver_features_directory, gold_features_directory, spark):
 
    # Prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
 
    # Load silver tables
    date_suffix = snapshot_date_str.replace('-', '_')
    path_click = os.path.join(silver_features_directory, f"silver_features_clickstream_{date_suffix}.parquet")
    path_attr  = os.path.join(silver_features_directory, f"silver_features_attributes_{date_suffix}.parquet")
    path_fin   = os.path.join(silver_features_directory, f"silver_features_financials_{date_suffix}.parquet")
 
    df_clickstream = spark.read.parquet(path_click)
    df_attributes  = spark.read.parquet(path_attr)
    df_financials  = spark.read.parquet(path_fin)
 
    print(f"[{snapshot_date_str}] loaded | financials: {df_financials.count()} | attributes: {df_attributes.count()} | clickstream: {df_clickstream.count()}")
 
    # Join tables
    df_gold = df_financials.drop('snapshot_date') \
                .join(df_attributes.drop('snapshot_date'), on='Customer_ID', how='left') \
                .join(df_clickstream.drop('snapshot_date'), on='Customer_ID', how='left')
 
    # Pad fe_ cols with null for customers with no clickstream match
    df_gold = pad_features(df_gold)
 
    # Log clickstream match rate
    total   = df_gold.count()
    matched = df_gold.filter(F.col('fe_1').isNotNull()).count()
    print(f"[{snapshot_date_str}] after join: {total} rows | clickstream matched: {matched} ({matched/total*100:.1f}%) | unmatched: {total - matched}")
 
    # Fix negative values
    print(f"[{snapshot_date_str}] fixing negative values...")
    df_gold = fix_negative_values(df_gold)

    # Clamp outliers
    print(f"[{snapshot_date_str}] clamping outliers...")
    df_gold = clamp_outliers(df_gold)
 
    # Encode categorical columns
    print(f"[{snapshot_date_str}] encoding categorical columns...")
    df_gold = encode_credit_mix(df_gold)
    df_gold = encode_payment_of_min_amount(df_gold)
    df_gold = encode_payment_behaviour(df_gold)
    df_gold = encode_occupation(df_gold)
 
    # Feature engineering
    print(f"[{snapshot_date_str}] engineering features...")
    df_gold = engineer_features(df_gold)
 
    # Drop raw columns replaced by engineered/encoded versions
    df_gold = drop_columns(df_gold)
 
    # Add snapshot_date back
    df_gold = df_gold.withColumn('snapshot_date', F.lit(snapshot_date_str))
 
    # Save
    partition_name = f"gold_features_{date_suffix}.parquet"
    filepath = os.path.join(gold_features_directory, partition_name)
    df_gold.write.mode("overwrite").parquet(filepath)
    print(f"[{snapshot_date_str}] saved to: {filepath} | final columns: {len(df_gold.columns)}")
 
    return df_gold