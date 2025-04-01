import json
from collections import defaultdict
from datetime import datetime
import logging
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from skrub import StringEncoder

def int_transform(series: pd.Series) -> np.ndarray:
    numeric_series = pd.to_numeric(series, errors='coerce').astype(float)
    mean_val = numeric_series.mean()
    numeric_series.fillna(mean_val)
    std_val = numeric_series.std()
    # Avoid division by zero: if std is 0, use 1 as the divisor.
    if std_val == 0:
        std_val = 1.0    
    # Compute the z-score: how far each number is from the mean, scaled by std.
    transformed = (numeric_series - mean_val) / std_val
    # Return the transformed values as a 2D array of floats.
    return transformed.values.reshape(-1, 1)

def log_transform(series: pd.Series) -> np.ndarray:
    # Convert series to numeric, handling non-numeric values as needed.
    numeric_series = pd.to_numeric(series, errors='coerce').fillna(0).astype(float)
    mean_val = numeric_series.mean()
    transformed = (numeric_series - mean_val) /mean_val
    # Apply the logarithmic transformation.
    # np.log1p is used to handle zeros, as np.log1p(0)=0.
    res = np.log1p(transformed)
    
    return res.values.reshape(-1, 1)


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
    # Group by time bucket and the single column's values, then count occurrences.
    pivot_df = df.groupby([pd.Grouper(freq=bucket_size), df]).size().unstack(fill_value=0)
    return pivot_df.values.astype(float)

def count_vectorizer_transform(series: pd.Series) -> np.ndarray:
    # Ensure the series values are strings.
    text_data = series.astype(str)
    
    # Initialize CountVectorizer with a token pattern that matches words with 4 or more characters.
    vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w{4,}\b")
    
    # Transform the text data to a document-term matrix.
    dt_matrix = vectorizer.fit_transform(text_data)
    
    # Convert the sparse matrix to a dense numpy array and cast the data type to float.
    res = dt_matrix.toarray().astype(float)
    return res

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