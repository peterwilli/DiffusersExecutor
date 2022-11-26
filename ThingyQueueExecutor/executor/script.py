import subprocess
from jina import Document
from threading import Thread

bootstrap = """
import json
from jina import Document

while True:
    try:
        doc = input()
        doc = Document.from_base64(doc)
        on_document(doc, lambda doc: print(doc.to_base64()))
    except Exception as e:
        print("error")
"""


class Script:
    def __init__(self, script):
        self.process = None
        self.script = script

    def start(self):
        with open("current_script.py", "w") as f:
            f.write(self.script)
            f.write(bootstrap)
        self.process = subprocess.Popen(['python', '-u', 'current_script.py'],
                                        stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE,
                                        bufsize=1,
                                        universal_newlines=True
                                        )

    def stop(self):
        self.process.terminate()

    def _run_worker(self, doc, callback):
        to_send = f"{doc.to_base64()}\n"
        print("to_send", to_send)
        self.process.stdin.write(to_send)
        while True:
            line = self.process.stdout.readline()
            print("line", line)
            try:
                doc = Document.from_base64(line)
                callback(doc)
                if "progress" not in doc.tags:
                    # Assume synchronous
                    doc.tags["progress"] = 1
                    return
                if doc.tags["progress"] == 1:
                    return
            except:  # In case we
                pass

    def run(self, doc: Document, callback):
        Thread(target=self._run_worker, args=(doc, callback)).start()
