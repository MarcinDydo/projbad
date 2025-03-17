import json
import pandas as pd
import statistics
from datetime import datetime
from collections import defaultdict
from utils import *

def numerical_outliers(data, threshold=2.0):
    """
    Given a list of numeric values, returns a dict with:
      {
        'mean': ...,
        'stdev': ...,
        'outliers': [list_of_indices]
      }
    An "outlier" is defined here as any point more than (threshold * stdev) away from the mean.
    """
    if len(data) < 2:
        return {
            "mean": None,
            "stdev": None,
            "outliers": []
        }
    
    mean_val = statistics.mean(data)
    stdev_val = statistics.pstdev(data)  # population stdev or sample stdev
    
    # If stdev == 0 (all values identical), no outliers
    if stdev_val == 0:
        return {
            "mean": mean_val,
            "stdev": 0,
            "outliers": []
        }
    
    outliers = []
    for i, val in enumerate(data):
        if abs(val - mean_val) > threshold * stdev_val:
            outliers.append(i)
    
    return {
        "mean": mean_val,
        "stdev": stdev_val,
        "outliers": outliers
    }

def timeseries_outliers(counts, threshold=3.0):
    """
    Returns list of indices in `counts` that are more than `threshold * stdev` above the mean.
    e.g., threshold=3 means anything above (mean + 3*stdev) is flagged as spike.

    You could also consider time-based or rolling windows, but this is a simple global approach.
    """
    if len(counts) < 2:
        return []

    mean_val = statistics.mean(counts)
    stdev_val = statistics.pstdev(counts)
    if stdev_val == 0:
        return []

    spike_indices = []
    for i, c in enumerate(counts):
        if c > mean_val + threshold * stdev_val:
            spike_indices.append(i)
    return spike_indices

def analysis(docs):

    # This will store: fields_data[field_name] = list of (dt, value)
    fields_data = defaultdict(list)
    
    for doc in docs:
        src = doc.get("_source", {})
        if not src:
            continue
        
        # 1) Flatten
        flattened = flatten_dict(src)
        
        # 2) Parse the @timestamp
        raw_ts = flattened.get("@timestamp")
        if not raw_ts:
            # If no timestamp, skip or handle differently
            continue
        
        try:
            # Convert string to datetime
            dt = datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
        except ValueError:
            # If timestamp parsing fails, skip
            continue
        
        # 3) For each flattened field, if it's numeric, store (dt, value)
        for field_key, field_val in flattened.items():
            # skip the special @timestamp field itself, or any non-numeric
            if field_key == "@timestamp":
                continue
            
            if isinstance(field_val, (int, float)):
                fields_data[field_key].append((dt, field_val))
    
    # 4) For each numeric field, sort by timestamp and detect outliers
    outliers_result = {}
    
    for field_name, pairs in fields_data.items():
        # pairs = [(timestamp, val), ...]
        if not pairs:
            continue
        
        # Sort by timestamp
        pairs.sort(key=lambda x: x[0])
        
        # Extract the numeric values in time order
        values = [pair[1] for pair in pairs]
        
        # Run outlier detection
        stats_info = numerical_outliers(values)
        outlier_indices = stats_info["outliers"]
        
        if outlier_indices:
            # Store details of the outliers, including timestamps
            outlier_points = [pairs[i] for i in outlier_indices]
            outliers_result[field_name] = {
                "mean": stats_info["mean"],
                "stdev": stats_info["stdev"],
                "count": len(outlier_indices),
                "points": outlier_points  # list of (datetime, value)
            }
    
    # 5) Print results
    if not outliers_result:
        print("No outliers found for any numeric field.")
    else:
        print("Outliers found in the following fields:")
        for fld, info in outliers_result.items():
            print(f"\nField: {fld}")
            print(f"  Mean = {info['mean']:.2f}, Stdev = {info['stdev']:.2f}")
            print(f"  Outlier count: {info['count']}")
            print("  Outlier points:")
            for (dt, val) in info["points"]:
                print(f"    {dt.isoformat()} -> {val}")
    
def frequency_buckets(df, col, bucket_size = "5min"):
    if df[col].apply(lambda x: isinstance(x, list)).any():
        df[col] = df[col].apply(lambda x: tuple(x) if isinstance(x, list) else x)
    freq_df = df.resample(bucket_size)[col].value_counts().unstack(fill_value=0)
    return freq_df 
