from jina import Executor, requests, DocumentArray, Document

import inspect
import warnings
from typing import List, Optional, Union, Dict

import torch
from torch import autocast
from tqdm.auto import tqdm
import torchvision.transforms as T

import PIL
from PIL import Image
import numpy as np

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DiffusionPipeline,
    PNDMScheduler,
    UNet2DConditionModel,
    StableDiffusionImg2ImgPipeline,
    LMSDiscreteScheduler
)
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from .utils import free_memory

tensor_to_pil_image = T.ToPILImage()

def _img2img(docs, parameters):
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
    doc = docs[0]
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        use_auth_token=parameters['hf_auth_token'],
        scheduler=scheduler
    ).to("cuda")

    def dummy(images, **kwargs):
        return images, False
    pipe.safety_checker = dummy

    doc.load_uri_to_image_tensor()
    image = tensor_to_pil_image(doc.tensor).convert("RGB")
    image = image.resize((512, 512))
    
    with autocast("cuda"):
        image = pipe(prompt=doc.text, strength=parameters["strength"], init_image = image, guidance_scale=parameters["guidance_scale"], num_inference_steps=int(parameters['steps']))["sample"][0]
    return Document().load_pil_image_to_datauri(image)

class Img2ImgExecutor(Executor):
    @requests(on='/stable_diffusion/img2img')
    def img2img(self, docs: DocumentArray, parameters: Dict, **kwargs):
        result_doc = _img2img(docs, parameters)
        free_memory()
        return DocumentArray(result_doc)