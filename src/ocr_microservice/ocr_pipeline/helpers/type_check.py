import json

def is_json(data: str):
    try:
        json.loads(data)
    except ValueError as e:
        return False
    return True