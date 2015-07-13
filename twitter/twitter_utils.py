import json

def json_validate(l):
    try:
        json.loads(l)
    except:
        return False
    return True

def extract_tags(text):
    tokens = text.split()
    return [tag for tag in tokens if tag.startswith("#")]
