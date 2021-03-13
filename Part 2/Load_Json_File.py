import json

def load_json_file(path):
    with open(path, 'r') as f:
        opened_data = json.load(f)
        
    return opened_data