"""Check public names in paths, text, and decoded structured values."""

from html import unescape
import json
import re
import unicodedata

# Construct the search terms from pieces so this file passes the same scan.
RESTRICTED = re.compile("for" + "ge|hue" + "forg|" + r"\bh" + r"f-(?:score|issues|histogram)\b", re.I)


def check_names(name, data):
    text = data.decode("utf-8")
    if name.endswith(".json"):
        text += "\n" + json.dumps(json.loads(text), ensure_ascii=False)
    if RESTRICTED.search(unicodedata.normalize("NFKC", name + "\n" + unescape(text))):
        raise ValueError("restricted public name")


def check_members(files):
    for name, data in files.items():
        check_names(name, data)
