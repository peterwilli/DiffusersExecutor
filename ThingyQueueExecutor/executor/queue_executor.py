from typing import Dict
import json
import subprocess
import os
import time
from .script import Script
from jina import Executor, requests, DocumentArray, Document

global_object = {
    'current_script': None,
    'current_status': []
}


def run_doc_entry(docs, index):
    if index == len(docs):
        return

    def callback(doc):
        global_object['current_status'][-1] = doc
        global_object['current_status'][-1].tags.update({
            'document_index': index
        })
        if doc.tags["progress"] == 1:
            global_object['current_status'].append(None)
            run_doc_entry(docs, index + 1)

    global_object['current_script'].run(docs[index], callback)


class ThingyQueueExecutor(Executor):
    @requests(on='/queue_entry_status')
    def queue_entry_status(self, docs: DocumentArray, parameters: Dict, **kwargs):
        if global_object['current_status'] is not None:
            index = int(parameters['index'])
            if len(global_object['current_status']) > index and global_object['current_status'][index] is not None:
                return DocumentArray(global_object['current_status'][index])
        return DocumentArray()

    @requests(on='/run_queue_entry')
    def run_queue_entry(self, docs: DocumentArray, parameters: Dict, **kwargs):
        new_script = parameters['script']
        should_spawn_new_script = False
        if global_object['current_script'] is None:
            should_spawn_new_script = Truewas
        elif global_object['current_script'].script != new_script:
            global_object['current_script'].stop()
            should_spawn_new_script = True
        if should_spawn_new_script:
            global_object['current_script'] = Script(new_script)
            global_object['current_script'].start()
        global_object['current_status'] = [None]
        run_doc_entry(docs, 0)
        return DocumentArray(Document(text="received"))
