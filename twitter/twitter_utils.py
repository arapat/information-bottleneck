import json

def try_load(l):
    try:
        return json.loads(l)
    except:
        return None

def extract_tags(text):
    tokens = text.split()
    return [tag for tag in tokens if tag.startswith("#")]
