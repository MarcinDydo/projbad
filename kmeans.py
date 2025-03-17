import argparse
import requests
from sklearn.cluster import KMeans
from collections import Counter
   
def load_elasticsearch_json(filter_str=None):

    es_url = "https://10.44.30.201"   # Elasticsearch base URL
    index = "alerts"                # Elasticsearch index name

    if filter_str and ':' in filter_str:
        field, value = filter_str.split(':', 1)
        field, value = field.strip(), value.strip()
        query = {
            "query": {
                "term": {
                    field: value
                }
            },
            "size": 1000
        }
    else:
        query = {
            "query": {
                "match_all": {}
            },
            "size": 1000
        }

    search_url = f"{es_url.strip('/')}/{index}/_search"
    response = requests.post(search_url, json=query)
    response.raise_for_status()
    data = response.json()
    hits = data.get('hits', {}).get('hits', [])
    documents = [h.get('_source', {}) for h in hits]
    return documents

def preprocess_documents(documents, fields_for_clustering=None):
    if not documents:
        return [], [], []

    if fields_for_clustering is None:
        # Automatically pick numeric fields from the first document
        sample_doc = documents[0]
        fields_for_clustering = [
            key for key, val in sample_doc.items() if isinstance(val, (int, float))
        ]
    
    X = []
    valid_docs = []
    for doc in documents:
        feature_vector = []
        is_valid = True
        for field in fields_for_clustering:
            val = doc.get(field)
            if isinstance(val, (int, float)):
                feature_vector.append(val)
            else:
                # If a required field is missing or not numeric, skip this doc
                is_valid = False
                break
        if is_valid:
            X.append(feature_vector)
            valid_docs.append(doc)
    
    return X, valid_docs, fields_for_clustering

def perform_kmeans_clustering(X, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)
    return kmeans, labels

def find_top_field_value_pairs(documents, cluster_labels, n_clusters):
    """
    For each cluster, find the single (field, value) pair that occurs most frequently.
    We do this by scanning all documents in that cluster, counting all (field, value) pairs,
    and picking the top one by frequency.
    
    Returns a list of length n_clusters, where each element is (field, value).
    """
    from collections import defaultdict
    cluster_counters = [Counter() for _ in range(n_clusters)]
    
    for doc, label in zip(documents, cluster_labels):
        for field, value in doc.items():
            if value is not None:
                pair = (field, str(value))
                cluster_counters[label][pair] += 1
    
    top_pairs = []
    for i in range(n_clusters):
        if len(cluster_counters[i]) == 0:
            top_pairs.append(("N/A", "N/A"))
            continue
        
        (top_pair, _freq) = cluster_counters[i].most_common(1)[0]
        top_pairs.append(top_pair) 
    
    return top_pairs

def pretty_kibana_filter(field_value_pairs):
    filters = []
    for field, val in field_value_pairs:
        filters.append(f'{field}:"{val}"')
    return filters

def main():
    #1. parse args
    parser = argparse.ArgumentParser(description="K-Means clustering on ES Kibana JSON with optional filter.")
    parser.add_argument("--filter", help="Initial filter in the format 'field:value' to filter docs before clustering.", default=None)
    parser.add_argument("--n_clusters", type=int, default=3, help="Number of clusters for K-Means.")
    parser.add_argument("--fields", nargs='+', default=None, help="List of numeric fields to use for clustering. If not provided, auto-detect numeric fields.")
    args = parser.parse_args()
    
    # 2. Load JSON
    documents = load_elasticsearch_json()
    if not documents:
        print(f"No documents found")
        return

    # 3. Convert documents to numeric feature matrix
    X, valid_docs, used_fields = preprocess_documents(documents, fields_for_clustering=args.fields)
    if not X:
        print("No valid numeric documents found for clustering.")
        return
    
    print(f"Using fields for clustering: {used_fields}")
    print(f"Number of documents used for clustering: {len(X)}")

    # 4. Perform K-Means clustering
    kmeans, cluster_labels = perform_kmeans_clustering(X, n_clusters=args.n_clusters)
    
    # 5. Find top field:value pair for each cluster
    top_pairs = find_top_field_value_pairs(valid_docs, cluster_labels, args.n_clusters)
    
    # 6. Convert pairs to Kibana filter query
    kibana_filters = pretty_kibana_filter(top_pairs)
    
    # Print results
    print("\n--- Cluster Summary ---")
    for i, (field, value) in enumerate(top_pairs):
        print(f"Cluster {i}: {field}:{value}")
    
    print("\n--- Kibana Query Filters ---")
    for i, query in enumerate(kibana_filters):
        print(f"Cluster {i}: {query}")

if __name__ == "__main__":
    main()
