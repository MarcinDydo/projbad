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
            timeout=280
        )
        self.mapping = {}
        self.slice = None
    
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
        
    def index_mapping(self, index):
        es_mapping = self.elastic.indices.get_mapping(index)
        for _, mapping_info in es_mapping.items():
            index_mappings = mapping_info.get("mappings", {})
            properties = index_mappings.get("properties", {})
            extracted = extract_properties(properties)
            self.mapping = self.mapping | extracted
    
    def transform_slice(self, f=None):
        copy = self.slice.copy()
        switcher = {
            'keyword': string_transform,
            'geo_point': frequency_transform,
            'short': int_transform,
            'ip': frequency_transform,
            'double': float_transform,
            'long': float_transform,
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
            'time_keyword': bagging_transform
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
            tmp = copy[i].apply(ensure_string_or_number).dropna() #TODO:increase dimensions (now 1 pass only)
        return f(tmp)
    
    def generate_query(self,function):
        results = []
        logging.info(f"Currently using {self.slice.columns} in query")
        outliers = function() #list of outliers from column
        for o in outliers:
            res = {"key":self.slice.columns[0],"value":o,"support":self.calculate_kpi(o)}
            print(f"possible filter (type {self.mapping[res['key']]} > {res['key']} : {res['value']}; support = {res['support']}%")
            results.append(res)
        # Generate a filename
        filename = datetime.now().strftime("%Y-%m-%d") + ".csv"
        filepath = os.path.join(os.path.join("cache", "output"), filename)
        with open(filepath, "a", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["key","value","support"])
            if not os.path.isfile(filepath):
                writer.writeheader()
            writer.writerows(results)

    def calculate_kpi(self, value):
        total_count = len(self.slice)
        counts = self.slice.astype(str).value_counts()
        
        count = counts.get(str(value), 0)
        support_percentage = (count / total_count) * 100  # percentage of total occurrences
        return support_percentage
                
   
    def time_iforest(self):
        # Optionally, you can normalize or scale X here if needed
        X = self.transform_slice()
        # Initialize and fit the Isolation Forest model
        m = IForest()
        m.fit(X)
        
        # Predict outliers: In PyOD, typically 1 indicates an outlier and 0 an inlier.
        labels = m.predict(X)
        scores = m.decision_function(X)
        
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
        return list(map(str, results))
    
    def iforest(self):
        # Optionally, you can normalize or scale X here if needed
        X = self.transform_slice()
        # Initialize and fit the Isolation Forest model
        m = IForest()
        m.fit(X)
        
        # Predict outliers: In PyOD, typically 1 indicates an outlier and 0 an inlier.
        labels = m.predict(X)
        scores = m.decision_function(X)
        
        # Identify indices of outliers
        outlier_indices = np.where(labels == 1)[0]
        results = set(map(lambda i: str(self.slice.values[i][0]), outlier_indices)) #TODO: implement something beter than first element on list
        return results

def main():
    host = os.environ["IP"]
    port = 9200
    api_key = os.environ["API_KEY"]

    model = Model(host, port, api_key)

    if model.elastic.ping():
        logging.info("Elasticsearch cluster is up!")
    else:
        logging.info("Elasticsearch cluster is down!")
        return

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

    model.index_mapping(index)    #uniquemappings = set(model.mapping.values())
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
    #df.set_index("@timestamp", inplace=True)
    headers = df.columns.values.tolist()
    headers.remove("@timestamp")
    logging.warning(f"Performing Isolation Forest outlier detection with headers: {headers}")
    for col in headers:
        
        if col =="rule.severity" or col =="event.severity":
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
