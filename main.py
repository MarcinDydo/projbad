from elasticsearch import Elasticsearch, RequestsHttpConnection
from pyod.models.iforest import IForest
from pyod.models.ecod import ECOD
import logging
import argparse
import json
import os
import csv
from collections import defaultdict
from datetime import datetime
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from utils import *
import hashlib

class Model:
    def __init__(self, host: str, port: int, api_key: str, use_ssl: bool = True, verify_certs: bool = False):
        """
        Initialize the Elasticsearch client.
        """
        self.elastic = Elasticsearch(
            [{'host': host, 'port': port}],
            connection_class=RequestsHttpConnection,
            api_key=api_key,
            use_ssl=use_ssl,
            verify_certs=verify_certs,
            timeout=10
        )
        self.mapping = {}
        self.slice = None
        self.function_scores= None
        self.filename = None
        self.filters = None
    
    def get_hits(self, index: str, query: dict, size:int) -> list:
        all_hits = []
        body = {
        "query": query,
        "sort": [{"@timestamp": "asc"}]
        }
        boundaries = list(range(0, size, 10000))
        if boundaries[-1] != size:
            boundaries.append(size)
        search_after = None
        for i in range(len(boundaries) - 1):
            batch_size = boundaries[i+1] - boundaries[i]
            if search_after:
                body["search_after"] = search_after
                res = self.elastic.search(index=index, body=body, size=batch_size)
            else:
                res = self.elastic.search(index=index, body=body, size=batch_size)
            
            hits = res['hits']['hits']
            if not hits:
                break
            all_hits.extend(hits)
            # Update search_after with the sort value of the last hit.
            search_after = hits[-1]['sort']
        return all_hits
    
    def cache_hits(self, query, hits, cache_dir="cache"):
        """Save hits data to a JSON file using the query hash as filename."""
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        query_hash = hashlib.md5(json.dumps(query, sort_keys=True).encode('utf-8')).hexdigest()
        filename = os.path.join(cache_dir, f"{query_hash}.json")
        with open(filename, 'w') as f:
            json.dump(hits, f)
        logging.info(f"Cached hits to {filename}")

    def load_hits(self, query, cache_dir="cache") -> list: 
        """Load hits data from a JSON file if it exists, based on the query hash."""
        query_hash = hashlib.md5(json.dumps(query, sort_keys=True).encode('utf-8')).hexdigest()
        filename = os.path.join(cache_dir, f"{query_hash}.json")
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                hits = json.load(f)
            logging.info(f"Loaded cached hits from {filename}")
            return hits
        else:
            return None
        
    def index_mapping(self, index, use_cache=False, file="./cache/mapping_cache.json"):     
        if use_cache and os.path.exists(file):
            # Load mapping from the cache file
            with open(file, 'r') as f:
                es_mapping = json.load(f)
        else:
            # Retrieve mapping from Elasticsearch
            es_mapping = self.elastic.indices.get_mapping(index)
            if use_cache:
                # Ensure the cache directory exists
                os.makedirs(os.path.dirname(file), exist_ok=True)
                # Save the retrieved mapping to the cache file
                with open(file, 'w') as f:
                    json.dump(es_mapping, f)
        
        # Process the mapping to extract properties
        for _, mapping_info in es_mapping.items():
            index_mappings = mapping_info.get("mappings", {})
            properties = index_mappings.get("properties", {})
            extracted = extract_properties(properties)
            # Merge the extracted properties with the existing mapping (using the | operator for dict merge)
            self.mapping = self.mapping | extracted
    
    def transform_slice(self, f=None):
        copy = self.slice.copy()
        switcher = {
            'keyword': string_transform,
            'geo_point': frequency_transform,
            'short': int_transform,
            'ip': frequency_transform,
            'double': float_transform,
            'long': frequency_transform,
            'object': string_transform,
            'match_only_text': string_transform,
            'scaled_float': float_transform,
            'nested': frequency_transform,
            'boolean': frequency_transform,
            'flattened': frequency_transform,
            'text': string_transform,
            'constant_keyword': frequency_transform,
            'date': frequency_transform,
            'wildcard': frequency_transform,
            'float': float_transform,
            'integer': int_transform,
            'time_keyword': bucket_transform,
            'long_text': count_vectorizer_transform
        }
        for i in copy.columns:
            if isinstance(copy.index, pd.DatetimeIndex) and "@timestamp" not in copy.columns:
                type = 'time_keyword'
            else:
                try:
                    type = self.mapping[i]
                except KeyError:
                    type = 'keyword'
            f = switcher.get(type)
            if f is None: f = frequency_transform
            tmp = copy[i].apply(ensure_string_or_number).dropna() #TODO:increase dimensions (now 1 pass only)
        return f(tmp)
    
    def generate_query(self,function,minf=3):
        results = []
        uniq_f = set()
        logging.info(f"Currently using {self.slice.columns} in query")

        counts = self.slice[self.slice.columns[0]].value_counts()
        counts.index = counts.index.astype(str)
        unique_values = set(counts[counts < minf].index)
        duplicate_values = set(counts[counts >= minf].index)
        outlier_set = function() #list of outliers from column as a SET
        singular_outliers = list(outlier_set.intersection(unique_values))
        outliers =list(outlier_set.intersection(duplicate_values))
        if singular_outliers:
            outliers.append(min(singular_outliers))
            outliers.append(max(singular_outliers))
        for o in outliers:
            kpi = self.calculate_kpi(o)
            res = {"key":self.slice.columns[0],"value":o,"support":kpi[0],"lift":kpi[1],"z_score":kpi[2],"frequency":kpi[3]}
            uniq_f.add(kpi[3])
            #print(f"possible filter (type {self.mapping[res['key']]} > {res['key']} : {res['value']}; \
            #      frequency = {res['frequency']}, support = {res['support']}%, lift={res['lift']}, z_score = {res['z_score']}")
            results.append(res)

        for u in uniq_f:
            l=[]
            m=[]
            for d in results:
                if d.get("frequency") == u:
                    m = [d["support"],d["lift"],d["z_score"],d["frequency"]] #lift is against equal distribution of uniq values
                    l.append(d["value"])
            print(f"FILTER: type: {self.mapping[self.slice.columns[0]]} > {self.slice.columns[0]}: {str(l)} > support = {m[0]}%, lift={m[1]}, z_score = {m[2]}, frequency = {m[3]}")
        # Generate a filename
        filepath = os.path.join(os.path.join("cache", "output"), self.filename)
        with open(filepath, "a", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["key","value","support","lift","z_score","frequency"])
            if not os.path.isfile(filepath):
                writer.writeheader()
            writer.writerows(results)
        self.filters = results

    def calculate_kpi(self, value):
        total_count = len(self.slice)
        # Convert all values to string for consistency
        counts = self.slice.astype(str).value_counts()

        # Frequency is simply the count for this value
        count = counts.get(str(value), 0)
        frequency = count

        # Support as a percentage of total occurrences
        support_percentage = (count / total_count) * 100 if total_count > 0 else 0

        # Calculate lift.
        # Under a uniform baseline, expected probability = 1 / (number of unique values)
        num_unique = len(counts)
        if num_unique > 0 and total_count > 0:
            observed_probability = count / total_count
            expected_probability = 1 / num_unique
            lift = observed_probability / expected_probability
        else:
            lift = 0

        # Calculate z-score: (observed count - mean count) / standard deviation of counts
        mean_count = counts.mean() if len(counts) > 0 else 0
        std_count = counts.std() if len(counts) > 1 else 0
        z_score = (count - mean_count) / std_count if std_count != 0 else 0

        return support_percentage, lift, z_score, frequency

    def time_iforest(self):
        # Optionally, you can normalize or scale X here if needed
        X = self.transform_slice()
        # Initialize and fit the Isolation Forest model
        m = IForest()
        m.fit(X)
        
        # Predict outliers: In PyOD, typically 1 indicates an outlier and 0 an inlier.
        labels = m.predict(X)
        scores = m.decision_function(X)
        self.function_scores = scores
        
        # Identify indices of outliers
        outlier_indices = np.where(labels == 1)[0]
       
        bucket_timedelta = pd.to_timedelta("5s")
        results = set()
        base_time = self.slice.index[0] 
        for b in outlier_indices:
            # Skip any invalid bucket indices.
            if b < 0 or b >= len(X):
                continue

            bucket_start = base_time + b * bucket_timedelta
            bucket_end = bucket_start + bucket_timedelta
            
            # Filter the original DataFrame for rows within the time span of this bucket
            mask = (self.slice.index >= bucket_start) & (self.slice.index < bucket_end)
            results.update(self.slice.loc[mask, self.slice.columns[0]].unique().tolist())
        
        return set(map(str, results))
    
    def iforest(self):
        # Optionally, you can normalize or scale X here if needed
        X = self.transform_slice()
        # Initialize and fit the Isolation Forest model
        m = ECOD()
        m.fit(X)
        
        # Predict outliers: In PyOD, typically 1 indicates an outlier and 0 an inlier.
        labels = m.predict(X)
        scores = m.decision_function(X)
        
        # Identify indices of outliers
        outlier_indices = np.where(labels == 1)[0]
        values_array = self.slice.astype(str).values
        # Use numpy's advanced indexing to get the first element of each desired row
        results = set(values_array[outlier_indices, 0]) #TODO: implement something beter than first element on list
        return results

def main():
    host = os.environ["IP"]
    port = 9200
    api_key = os.environ["API_KEY"]

    model = Model(host, port, api_key)

    if True:  # TODO: argparse
        logging.info("Runtime arguments not specified, defaults used..")
        index = "logs-*"
        query = {
            "bool": {
            "must": [
                {
                "term": {
                    "event.dataset": "suricata.alert"
                }
                },
                {
                "range": {
                    "@timestamp": {
                    "gte": "now-1d/d"
                    }
                }
                }
            ]
            }
        }
        size = 50000
    
    if model.elastic.ping():
        logging.info("Elasticsearch cluster is up!")
        model.index_mapping(index)
    else:
        logging.warning("Elasticsearch cluster is down!")
        model.index_mapping(index, use_cache=True)
    
    #uniquemappings = set(model.mapping.values())
    # Use cached data if available
    logging.warning(f"Loading hits to dataframe")
    cached_data = model.load_hits(query)
    if cached_data is not None:
        logging.info("Using cached data.")
        data = cached_data
    else:
        logging.warning("Query-ing Elasticsearch.")
        data = model.get_hits(index, query,size)
        model.cache_hits(query, data)

    # Preprocessing
    df = get_dataframe(data)
    model.filename = datetime.now().strftime("%Y-%m-%d.%H.%M.%S") + ".csv"
    #df.set_index("@timestamp", inplace=True)
    headers = df.columns.values.tolist()
    headers.remove("@timestamp")
    logging.warning(f"Performing Isolation Forest outlier detection with headers: {headers}")
    for col in headers:
        if col =="network.data.decoded" or col == "rule.rule":
            model.mapping[col] = 'long_text'
            model.slice = df[[col]].copy()
            model.generate_query(model.iforest)
        elif col =="rule.severity" or col =="event.severity":
            model.slice = df[["@timestamp",col]].copy()#time slicing
            model.slice.set_index("@timestamp", inplace=True) 
            model.generate_query(model.time_iforest)
        else:
            model.slice = df[[col]].copy() #basic slicing
            if entropy_filter(model.slice): #basic filtering
                model.generate_query(model.iforest)
            else:
                continue

if __name__ == '__main__': 
    load_dotenv()
    main()
