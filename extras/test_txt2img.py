from Txt2ImgExecutor import executor

from jina import Flow, Executor, requests

with Flow(protocol='HTTP', port=5000).add(name='mgp', uses=executor.Txt2ImgExecutor) as f:
    f.block()