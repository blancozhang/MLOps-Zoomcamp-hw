#!/usr/bin/env python
# coding: utf-8

import pickle
import argparse
import pandas as pd
import numpy as np 
import os

# Define categorical features 
categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    """
    Reads parquet data, calculates duration, filters, and prepares categorical features.
    """
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    # Convert categorical columns to string after filling NaNs
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

def run_scoring(year: int, month: int, output_file: str):
    """
    Main function to run the scoring pipeline.
    """

    try:
        with open('model.bin', 'rb') as f_in:
            dv, model = pickle.load(f_in)
    except FileNotFoundError:
        print("Error: 'model.bin' not found. Please ensure your trained model and DictVectorizer are saved as 'model.bin' in the same directory.")
        exit(1) # Exit if model not found


    data_url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet"
    print(f"Loading data from: {data_url}")

    # 3. Read the data
    try:
        df = read_data(data_url)
    except Exception as e:
        print(f"Error loading data from {data_url}: {e}")
        print("Please ensure the year and month are correct and the file exists.")
        exit(1) # Exit if data loading fails

    # 4. Create 'ride_id' using the passed year and month
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    # 5. Prepare features for prediction
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    
    # 6. Make predictions
    y_pred = model.predict(X_val)

    # 7. Calculate and print the mean predicted duration (for Q5)
    mean_predicted_duration = np.mean(y_pred)
    print(f"The mean predicted duration for {year}-{month:02d} is: {mean_predicted_duration:.2f}")

    # Optional: Calculate and print standard deviation if still needed
    std_dev_predicted_duration = np.std(y_pred)
    print(f"The standard deviation of predicted duration is: {std_dev_predicted_duration:.2f}")

    # 8. Prepare df_result and export
    df_result = pd.DataFrame({
        'ride_id': df['ride_id'],
        'predicted_duration': y_pred
    })
    
    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )
    print(f"Results saved to {output_file}")

    # Optional: Verify the file exists and its size
    if os.path.exists(output_file):
        file_size_bytes = os.path.getsize(output_file)
        file_size_mb = file_size_bytes / (1024 * 1024)
        print(f"File '{output_file}' created successfully. Size: {file_size_mb:.2f} MB")

# --- Main execution block when the script is run ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate predicted duration for taxi trips.")
    parser.add_argument(
        "--year", 
        type=int, 
        required=True, 
        help="Year of the trip data (e.g., 2023)"
    )
    parser.add_argument(
        "--month", 
        type=int, 
        required=True, 
        help="Month of the trip data (e.g., 3 for March, 4 for April)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results.parquet", # Default output file name
        help="Output file name for predictions (e.g., results.parquet)"
    )

    args = parser.parse_args()

    run_scoring(args.year, args.month, args.output)

