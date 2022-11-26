from torch import autocast
from jina import Executor, requests, DocumentArray, Document
from typing import Dict
from .utils import free_memory
import torch
from diffusers import DiffusionPipeline, EulerDiscreteScheduler

def get_pipe(parameters):
    repo_id = "stabilityai/stable-diffusion-2"
    device = "cuda"
    scheduler = EulerDiscreteScheduler.from_pretrained(repo_id, subfolder="scheduler", prediction_type="v_prediction")
    pipe = DiffusionPipeline.from_pretrained(repo_id, use_auth_token=parameters['hf_auth_token'], torch_dtype=torch.float16, revision="fp16", scheduler=scheduler)
    pipe = pipe.to(device)
    if 'no_nsfw_filter' in parameters:
        pipe.safety_checker = None
    return pipe

global_object = {
    'pipe': None
}

def next_divisible(n, d):
    divisor = n % d
    return n - divisor

def _txt2img(docs, parameters):
    generator = torch.manual_seed(int(parameters['seed']))
    if global_object['pipe'] is None:
        global_object['pipe'] = get_pipe(parameters)
    pipe = global_object['pipe']
    width = next_divisible(int(parameters['size'][0]), 8)
    height = next_divisible(int(parameters['size'][1]), 8)
    image = pipe(docs[0].text, width = width, height = height, guidance_scale=parameters["guidance_scale"], num_inference_steps=int(parameters['steps'])).images[0]  
    print("image", image)
    return Document().load_pil_image_to_datauri(image)

class Txt2ImgExecutor(Executor):
    @requests(on='/stable_diffusion/txt2img')
    def txt2img(self, docs: DocumentArray, parameters: Dict, **kwargs):
        result_doc = _txt2img(docs, parameters)
        free_memory()
        return DocumentArray(result_doc)