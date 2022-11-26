import time


def on_document(document, callback):
    for i in range(0, 10):
        callback(Document(text=f"test_{i}", tags = {
            "progress": (i + 1) / 10
        }))
        time.sleep(2)
import json
from jina import Document

while True:
    try:
        doc = input()
        doc = Document.from_base64(doc)
        on_document(doc, lambda doc: print(doc.to_base64()))
    except Exception as e:
        print("error")
