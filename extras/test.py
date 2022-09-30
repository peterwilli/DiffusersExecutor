from MagicPromptExecutor import executor

from jina import Flow, Executor, requests

with Flow(protocol='HTTP', port=5000).add(name='mgp', uses=executor.MagicPromptExecutor) as f:
    f.block()