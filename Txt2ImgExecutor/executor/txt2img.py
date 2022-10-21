from torch import autocast
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
from jina import Executor, requests, DocumentArray, Document
from typing import Dict
from .utils import free_memory
import torch

def get_pipe(parameters):
    lms = LMSDiscreteScheduler(
        beta_start=0.00085, 
        beta_end=0.012, 
        beta_schedule="scaled_linear"
    )
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        use_auth_token=parameters['hf_auth_token'],
        scheduler=lms,
        revision="fp16",
        torch_dtype=torch.float16
    ).to("cuda")

    def dummy(images, **kwargs):
        return images, False
    pipe.safety_checker = dummy
    return pipe

global_object = {
    'pipe': None
}

def _txt2img(docs, parameters):
    generator = torch.manual_seed(int(parameters['seed']))
    if global_object['pipe'] is None:
        global_object['pipe'] = get_pipe(parameters)
    pipe = global_object['pipe']
    image = pipe(docs[0].text, guidance_scale=parameters["guidance_scale"], num_inference_steps=int(parameters['steps'])).images[0]  
    return Document().load_pil_image_to_datauri(image)

class Txt2ImgExecutor(Executor):
    @requests(on='/stable_diffusion/txt2img')
    def txt2img(self, docs: DocumentArray, parameters: Dict, **kwargs):
        result_doc = _txt2img(docs, parameters)
        free_memory()
        return DocumentArray(result_doc)