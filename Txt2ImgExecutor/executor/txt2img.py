from torch import autocast
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
from jina import Executor, requests, DocumentArray, Document
from typing import Dict
from .utils import free_memory

def _txt2img(docs, parameters):
    lms = LMSDiscreteScheduler(
        beta_start=0.00085, 
        beta_end=0.012, 
        beta_schedule="scaled_linear"
    )

    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        use_auth_token=parameters['hf_auth_token'],
        scheduler=lms
    ).to("cuda")

    def dummy(images, **kwargs):
        return images, False
    pipe.safety_checker = dummy

    image = None
    with autocast("cuda"):
        image = pipe(docs[0].text, guidance_scale=parameters["guidance_scale"], num_inference_steps=int(parameters['steps']))["sample"][0]  

    return Document().load_pil_image_to_datauri(image)

class Txt2ImgExecutor(Executor):
    @requests(on='/stable_diffusion/txt2img')
    def txt2img(self, docs: DocumentArray, parameters: Dict, **kwargs):
        result_doc = _txt2img(docs, parameters)
        free_memory()
        return DocumentArray(result_doc)