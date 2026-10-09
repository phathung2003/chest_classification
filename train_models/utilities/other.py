import re

def to_snake_case(text):
    text = text.replace("-", "").lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = text.strip("_")

    return text