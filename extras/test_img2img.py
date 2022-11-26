from Txt2ImgExecutor import executor

from jina import Flow, Executor, Client, Document, requests

c = Client(host='grpc://localhost:51001')
image_doc = Document(uri="https://cdn.discordapp.com/attachments/964652278923526178/1045633983033118741/image.png")
image_doc.load_uri_to_image_tensor()
print(c.post('/stable_diffusion/img2img', image_doc, parameters = {
    "seed": "30",
    "hf_auth_token": "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",
    "size": [512, 512],
    "guidance_scale": 7.5,
    "steps": 25
})[0].text)
