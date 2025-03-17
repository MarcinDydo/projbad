import json
from collections import defaultdict
from utils import flatten_dict, get_hits

hits = get_hits()
# 2. We'll store counts for each flattened key.
#    fields_counts[flattened_key][value] -> count
fields_counts = defaultdict(lambda: defaultdict(int))

# 3. Iterate over each document, flatten its _source, and count values.
for hit in hits:
    source_dict = hit.get("_source", {})
    flat_doc = flatten_dict(source_dict)  # Flatten nested keys

    for flattened_key, value in flat_doc.items():
        if isinstance(value, list):
            value = tuple(value)  # or str(value)
        fields_counts[flattened_key][value] += 1

# 4. For each flattened key, find the most common value and x other #TODO
results = {}
for flattened_key, value_count_map in fields_counts.items():
    # value_count_map is like {value1: count, value2: count, ...}
    most_common_value = max(value_count_map, key=value_count_map.get)
    most_common_count = value_count_map[most_common_value]
    results[flattened_key] = {
        "most_common_value": most_common_value,
        "count": most_common_count
    }

# 5. Print (or store) results in a nice format.
for field, info in results.items():
    print(f"Field: {field}")
    print(f"  Most common value: {info['most_common_value']!r} (count: {info['count']})\n")
