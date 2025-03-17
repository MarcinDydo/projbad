import json
from collections import defaultdict
from datetime import datetime
import pandas as pd
import numpy as np


def flatten_dict(d, parent_key="", sep="."):
    """
    Recursively flattens a nested dictionary.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def get_dataframe(docs):
    flattened_records = []

    for doc in docs:
        source = doc.get("_source", {})
        # Flatten the _source dict
        flattened = flatten_dict(source)
        
        # Optionally include _id, _index, etc. in the flattened record:
        flattened["_id"] = doc.get("_id")
        flattened["_index"] = doc.get("_index")
        
        flattened_records.append(flattened)
    df = pd.DataFrame(flattened_records)
    df["@timestamp"] = pd.to_datetime(df["@timestamp"], errors="coerce")
    return(df)

def sort_time(all_docs):
    # 2) Group by 'key'
    #    Store as: data_by_key[key] = [ (timestamp, value), ... ]
    data_by_key = {}
    for doc in all_docs:
        source = doc.get("_source", {})
        timestamp_str = source.get("@timestamp")
        
        if timestamp_str is None:
            continue
        try:
            ts = datetime.fromisoformat(timestamp_str.replace("Z",""))
        except ValueError:
            continue
        
    sorted_docs = sorted(all_docs, key=lambda doc: doc["_source"]["@timestamp"])
    return sorted_docs

def round_to_bucket(dt, bucket_minutes=5):
    """
    Round a datetime 'dt' down to the nearest 'bucket_minutes'.
    """
    # Convert total minutes since midnight
    total_minutes = dt.hour * 60 + dt.minute
    # Floor to nearest bucket
    floored = (total_minutes // bucket_minutes) * bucket_minutes

    # Convert back to hours/minutes
    new_hour = floored // 60
    new_minute = floored % 60

    return dt.replace(hour=new_hour, minute=new_minute, second=0, microsecond=0)

def compute_entropy(series):
    #Shannon entropy of a pandas Series.
    counts = series.value_counts(normalize=True)
    if len(counts) == 0:
        return 0.0
    entropy = -np.sum(counts * np.log2(counts + 1e-9))
    return entropy

def compute_normalized_entropy(series):
    # entropy / log2(n_unique)
    counts = series.value_counts(normalize=True)
    n_unique = len(counts)
    if n_unique <= 1:
        return 0.0
    entropy = compute_entropy(series)
    max_entropy = np.log2(n_unique)
    normalized_entropy = entropy / max_entropy
    return normalized_entropy

def get_top_common_from_column(series, top_n=1):
    series_clean = series.dropna()
    value_counts = series_clean.value_counts()
    
    # Extract the top_n most common values as a list of (value, count) tuples.
    top_common = series_clean.value_counts().head(top_n)
    return top_common