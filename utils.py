import json
from collections import defaultdict
from datetime import datetime
import logging
import pandas as pd
import numpy as np
import ipaddress
from skrub import StringEncoder

def int_transform(series: pd.Series) -> np.ndarray:
    """
    Convert a numeric or numeric-like series to integers and return a 2D float array.
    Any non-convertible values are coerced to NaN then filled with 0.
    """
    # Convert series to numeric (coerce errors), fill missing with 0, and cast to int
    numeric_series = pd.to_numeric(series, errors='coerce').fillna(0).astype(int)
    # Return as 2D array with float type
    return numeric_series.values.reshape(-1, 1)

def float_transform(series: pd.Series) -> np.ndarray:
    """
    Convert a series to floats and return a 2D array.
    Non-numeric values are coerced to NaN and replaced with 0.0.
    """
    numeric_series = pd.to_numeric(series, errors='coerce').fillna(0.0).astype(float)
    return numeric_series.values.reshape(-1, 1)

def frequency_transform(series: pd.Series) -> np.ndarray:
    """
    series = series.astype(str)
    dummies = pd.get_dummies(series, prefix=series.name)
    # Convert to float and return the NumPy array
    return dummies.values.astype(float)
    """
    freq = series.value_counts()
    # Map each value in the original series to its frequency.
    encoded_series = series.map(freq)
    result = encoded_series.values.astype(float)
    return result.reshape(-1, 1)

def string_transform(series: pd.Series, maxN=30) -> np.ndarray:
    n = len(series.value_counts())
    if n > maxN: n = maxN
    enc = StringEncoder(n_components=n)
    series = series.fillna("").astype(str)
    result = enc.fit_transform(series).values
    return result

def bagging_transform(df: pd.DataFrame, bucket_size='5s') -> np.ndarray:
    # Ensure the DataFrame is indexed with a datetime index.
    a =1
    # Group by time bucket and the single column's values, then count occurrences.
    pivot_df = df.groupby([pd.Grouper(freq=bucket_size), df]).size().unstack(fill_value=0)
    b=3
    return pivot_df.values.astype(float)

def ensure_string_or_number(x):
    # If x is an int, float, or str, keep it as is.
    if isinstance(x, (int, float, str)):
        return x
    # Otherwise, convert it to a string representation.
    return str(x)

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


def get_notsingles_from_column(series):
    # Remove missing values
    series_clean = series.dropna()
    # Count the frequency of each value in the series
    value_counts = series_clean.value_counts()
    # Filter and return values that appear more than once
    result = value_counts[value_counts > 1]
    return result

def entropy_filter(series,min=0,max=0.95) -> bool: 
    s = series.astype(str)
    norm_entropy = compute_normalized_entropy(s.dropna())
    if min < norm_entropy < max:
        return True
    else:
        logging.warning(f"series diqualified entropy: {norm_entropy}")
        return False

def extract_properties(properties, parent_key=""):
    """
    Recursively traverse a properties dict and return a flat dict with dot-separated keys and their types.
    """
    result = {}
    for key, value in properties.items():
        # Build the dot-separated property name.
        full_key = f"{parent_key}.{key}" if parent_key else key
        # If this field has a defined type, add it.
        if "type" in value:
            result[full_key] = value["type"]
        # If the field has nested properties, traverse them recursively.
        if "properties" in value:
            nested = extract_properties(value["properties"], full_key)
            result.update(nested)
    return result