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
    DiffusionPipeline,
    UNet2DConditionModel,
    StableDiffusionImg2ImgPipeline,
    LMSDiscreteScheduler
)
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from .utils import free_memory

tensor_to_pil_image = T.ToPILImage()

def _img2img(docs, parameters):
    generator = torch.manual_seed(int(parameters['seed']))
    doc = docs[0]
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id,
        use_auth_token=parameters['hf_auth_token'],
        revision="fp16",
        torch_dtype=torch.float16
    ).to("cuda")

    def dummy(images, **kwargs):
        return images, False
    pipe.safety_checker = dummy

    doc.load_uri_to_image_tensor()
    image = tensor_to_pil_image(doc.tensor).convert("RGB")
    image = image.resize((512, 512))    
    image = pipe(prompt=doc.text, strength=parameters["strength"], init_image = image, guidance_scale=parameters["guidance_scale"], num_inference_steps=int(parameters['steps'])).images[0]
    return Document().load_pil_image_to_datauri(image)

class Img2ImgExecutor(Executor):
    @requests(on='/stable_diffusion/img2img')
    def img2img(self, docs: DocumentArray, parameters: Dict, **kwargs):
        result_doc = _img2img(docs, parameters)
        free_memory()
        return DocumentArray(result_doc)