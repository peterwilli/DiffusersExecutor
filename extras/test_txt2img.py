from Txt2ImgExecutor import executor

from jina import Flow, Executor, Client, Document, requests

c = Client(host='grpc://localhost:51001')
print(c.post('/stable_diffusion/txt2img', Document(text='Laughing mother in Greece'), parameters = {
    "seed": "30",
    "hf_auth_token": "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",
    "size": [512, 512],
    "guidance_scale": 7.5,
    "steps": 25
})[0].text)
