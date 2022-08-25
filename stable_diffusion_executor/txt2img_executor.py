from torch import autocast
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
from jina import Executor, requests, DocumentArray, Document
from typing import Dict

class Txt2ImgExecutor(Executor):
    @requests(on='/txt2img')
    async def txt2img(self, docs: DocumentArray, parameters: Dict, **kwargs):
        lms = LMSDiscreteScheduler(
            beta_start=0.00085, 
            beta_end=0.012, 
            beta_schedule="scaled_linear"
        )

        pipe = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4", 
            scheduler=lms,
            use_auth_token=True
        ).to("cuda")

        image = None
        with autocast("cuda"):
            image = pipe(docs[0].text, guidance_scale=7.5)["sample"][0]  

        result_doc = Document()
        result_doc.load_pil_image_to_datauri(image)
        return result_doc
        