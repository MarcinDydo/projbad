import os
import json
import logging
import pandas as pd
from dotenv import load_dotenv
from main import Model
from utils import *

def main():
    host = os.environ["IP"]
    port = 9200
    api_key = os.environ["API_KEY"]
    model = Model(host, port, api_key)

    config_file = "test_config.json"  # Update with the correct path if needed
    with open(config_file, "r") as f:
        config = json.load(f)

    #o = [] #open config file and load config to json - list of dict
    #for item in o:
    #    df = [] #TODO: load dataframe from csv specified in item which is in config.sjon
    #    mapping = 0 #TODO: load mapping from config.json

    #and for each loaded dataframe and mapping perform OD which will save the csv to separate file 

    for item in config:
        logging.warning(f"working on {item['path']}")
        model.filename = datetime.now().strftime("%Y-%m-%d.%H.%M.%S") + ".csv"
        print(f"RUNNING: {model.filename } working on {item['path']}")
        csv_path = item["path"]
        mapping = item["mapping"]
        # Load the dataset as a DataFrame.
        df = pd.read_csv(csv_path)
        # Set the model's mapping from the config.
        model.mapping = mapping
        # Process headers; remove '@timestamp' if present.
        headers = df.columns.tolist()
        logging.warning(f"Performing Isolation Forest outlier detection with headers: {headers}")
        labels = next((x for x in headers if mapping[x] == "labels"), None)

        # For each column in the dataset, run the corresponding outlier detection process.
        for col in headers:
            if col == labels:
                continue
            model.slice = df[[col]].copy()
            if entropy_filter(model.slice):
                model.generate_query(model.iforest)
                compute_accuracy(df[[col, labels]],col,labels, model.filters)
            else:
                continue

        logging.warning("DONE!")

if __name__ == '__main__': 
    load_dotenv()
    main()
