import json
import ast
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
    # Convert series to numeric and fill missing values with the series mean.
    numeric_series = pd.to_numeric(series, errors='coerce').astype(float).fillna(series.mean())
    min_val = numeric_series.min()
    max_val = numeric_series.max()
    range_val = max_val - min_val
    # Avoid division by zero: if range is 0, use 1 as the divisor.
    if range_val == 0:
        range_val = 1.0
    transformed = (numeric_series - min_val) / range_val
    return transformed.values.reshape(-1, 1)

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
    # Replace missing values with a distinct marker.
    filled_series = series.fillna("none").astype(str)
    unique_values = filled_series.value_counts()
    n_unique = len(unique_values)
    
    # If there's less than two unique values, return a constant array.
    if n_unique < 2:
        return np.zeros((len(series), 1))
    
    # Limit the number of components to the lesser of maxN or the number of unique values.
    n_components = min(n_unique, maxN)
    enc = StringEncoder(n_components=n_components)
    
    # Suppress warnings from catastrophic cancellation and invalid operations.

    result = enc.fit_transform(filled_series).values
    result = np.nan_to_num(result)

    return result

def bucket_transform(df: pd.DataFrame, bucket_size='5s') -> np.ndarray:
    # Ensure the DataFrame is indexed with a datetime index.
    # Group by time bucket and the single column's values, then count occurrences.
    pivot_df = df.groupby([pd.Grouper(freq=bucket_size), df]).size().unstack(fill_value=0)
    return pivot_df.values.astype(float)

def count_vectorizer_transform(series: pd.Series, max_features=1000) -> np.ndarray:
    # Ensure the series values are strings.
    series.fillna("").astype(str)
    text_data = series.astype(str)
    
    # Initialize CountVectorizer with a token pattern that matches words with 4 or more characters.
    vectorizer = CountVectorizer(token_pattern=r"(?u)\b[^=;+\:,&\s]{4,}\b",max_features=max_features)
    
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

def compute_accuracy(df,col,labels,filters):
    if not filters:
        return
    key = filters[0]["key"]
    assert key == col
    combined = []
    for f in filters:
        #get statistics of df[labels] for a subset of rows where df[col] has values from f["value"]  
        combined.append(f["value"])
        subset = df[df[col].astype(str) == f["value"]]
        metrics = subset[labels].value_counts() / len(subset) 
        res = metrics.to_string().replace("\n","=")
        res2 = subset[labels].value_counts().to_string().replace("\n","  =")
        print(f"TEST: filter {f['key']}:{f['value']} > returned [*100%] {res} [in subset] {res2}")

    subset = df[df[col].astype(str).isin(combined)]
    metrics = subset[labels].value_counts() / len(subset) # how much returned results are 
    res = metrics.to_string().replace("\n"," ")
    total_counts = df[labels].value_counts()
    matches_counts = subset[labels].value_counts()
    # Create a new DataFrame that contains both sets of counts
    total = pd.DataFrame({
        "total": total_counts,
        "matches": matches_counts
    })
    total = total.fillna(0)
    tres = ""
    for index, row in total.iterrows():
        tres += f"{str(index)}: {str(row['matches'])} out of {str(row['total'])} ({str(row['matches'] / row['total']*100)})%; "
    print(f"TEST: COMBINED filters for: {col} returned: {res}, RESULTING IN: {tres}")
         


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