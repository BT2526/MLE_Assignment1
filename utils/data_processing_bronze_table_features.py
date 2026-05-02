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
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType


def process_bronze_table_features(snapshot_date_str, bronze_features_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to source back end - IRL connect to back end source system
    csv_file_path_clickstream = "data/feature_clickstream.csv"
    csv_file_path_attributes = "data/features_attributes.csv"
    csv_file_path_financials = "data/features_financials.csv"

    # load data - IRL ingest from back end source system
    df_clickstream = spark.read.csv(csv_file_path_clickstream, header=True, inferSchema=True).filter(col('snapshot_date') == snapshot_date)
    df_attributes = spark.read.csv(csv_file_path_attributes, header=True, inferSchema=True).filter(col('snapshot_date') == snapshot_date)
    df_financials = spark.read.csv(csv_file_path_financials, header=True, inferSchema=True).filter(col('snapshot_date') == snapshot_date)
    print(snapshot_date_str + 'row count:', df_clickstream.count())
    print(snapshot_date_str + 'row count:', df_attributes.count())
    print(snapshot_date_str + 'row count:', df_financials.count())
    
    # save bronze table to datamart - IRL connect to database to write
    for name, df in [("clickstream", df_clickstream), ("attributes", df_attributes), ("financials", df_financials)]:
        filename = f"bronze_table_features_{name}_" + snapshot_date_str.replace('-', '_') + ".csv"
        filepath = os.path.join(bronze_features_directory, filename)
        df.toPandas().to_csv(filepath, index=False)
        print("saved to:", filepath)

    return df_clickstream, df_attributes, df_financials
