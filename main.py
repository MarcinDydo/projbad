from elasticsearch import Elasticsearch, RequestsHttpConnection
from pyod.models.iforest import IForest
import logging
import argparse
import json
import os
from collections import defaultdict
from datetime import datetime
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from utils import *

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
            timeout=60
        )
        self.df = None
    
    def get_hits(self, index: str, query: dict):
        res = self.elastic.search(index=index, body={"query": query})
        return res['hits']['hits']  
    
    def entropy_filter(self,min=0.1,max=0.95): #returing headers with limited entropy
        headers = self.df.columns.values
        selected_fields = []
        for key in headers:
            series = self.df[key].dropna()
            norm_entropy = compute_normalized_entropy(series)
            if min < norm_entropy < max:
                selected_fields.append(key)
                #print(f"Entropy of: {key} = {norm_entropy}")
        return selected_fields
    
    def generate_query(self,headers,function):
        for col in headers: # for category in fields
            logging.info(f"Currently using {self.df[col]} in query")
            outliers = function(col)
            for i in outliers["rows"]["max_category"]:
                print(f"possible filter > {col} : {i}")
                
    def iforest_outliers(self, col, bucket_size="5s", top_n=10):
        anomalies = {}
        column = self.df[col]
        top_values = get_top_common_from_column(column,top_n)
        freq_df = pd.DataFrame()
        for value in top_values.axes[0]:
            df =  self.df[["@timestamp", col]]
            df.set_index("@timestamp", inplace=True)
            df.loc[:, col] = df[col].apply(lambda x: 1 if x == value else 0)
            freq_series = df.resample(bucket_size)[col].sum()
            freq_df[str(value)] = freq_series
            
        pd.set_option('future.no_silent_downcasting', True)
        freq_df = freq_df.fillna(0).infer_objects(copy=False)

        X = freq_df.values
        logging.info(f"prepparing matrix for PyOD on column {col}:  each time bucket count becomes a sample. of {bucket_size}")
        m = IForest(contamination=0.1, random_state=42)
        m.fit(X)
        labels = m.predict(X)
        scores = m.decision_function(X)

        outliers = np.where(labels == 1)[0]
        bucket_timestamps = freq_df.index[outliers]
        #keys = top_values.take(outliers)
        selected_rows = freq_df.loc[bucket_timestamps].copy()

        selected_rows["max_category"] = selected_rows.idxmax(axis=1)
        result_dict = {
        "field":col,
        "value":value,
        "outlier_label": labels,
        "ts": bucket_timestamps,
        "rows": selected_rows,
        "anomaly_score": scores.flatten()
        }
        logging.info(f"Isolation Forest returning: {result_dict}")
        # Predict outliers: in PyOD, typically 1 indicates an outlier and 0 indicates an inlier.
        # Create a results DataFrame that aligns each time bucket with its value and anomaly info.
        return result_dict


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
    
    if True: #TODO:argparse
        logging.info("Runtime arguments not specified, defaults used..")
        index = "logs-*"
        query = {"term": {"event.dataset":"suricata.alert"}}
    

    logging.warning("Query-ing elasticsearch.")
    data = model.get_hits(index, query)
    model.df = get_dataframe(data)
    headers = model.entropy_filter()
    logging.warning("Performing Isolation Forest outlier detection")
    model.generate_query(headers, model.iforest_outliers)
    print(f"Search results + filter:{headers}")

if __name__ == '__main__':
    load_dotenv()
    main()
