import torch
from transformers import CLIPImageProcessor, CLIPTokenizer, CLIPVisionModelWithProjection, T5TokenizerFast

from diffusers import FluxPipeline
from diffusers.utils import load_image


tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", trust_remote_code=True)
tokenizer_2 = T5TokenizerFast.from_pretrained("t5-base", trust_remote_code=True)

image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    "openai/clip-vit-large-patch14", torch_dtype=torch.bfloat16
)
feature_extractor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    tokenizer=tokenizer,
    tokenizer_2=tokenizer_2,
    image_encoder=image_encoder,
    feature_extractor=feature_extractor,
    torch_dtype=torch.bfloat16,
)
pipe.load_ip_adapter("XLabs-AI/flux-ip-adapter-v2", weight_name="ip_adapter.safetensors")

pipe.to(device="cuda:1")

guidance_image = load_image(
    "https://github.com/exx8/differential-diffusion/blob/main/assets/input.jpg?raw=true",
)
seed = 646

pipe.set_ip_adapter_scale(0.5)
result = pipe(
    prompt="A car driving on a street.",
    ip_adapter_image=guidance_image,
    height=1024,
    width=1024,
    num_inference_steps=25,
    guidance_scale=3.5,
    generator=torch.Generator("cuda").manual_seed(seed),
).images[0]
result.save("output_with_ip_adapter0.5.png")

pipe.set_ip_adapter_scale(1.0)
result = pipe(
    prompt="A car driving on a street.",
    ip_adapter_image=guidance_image,
    height=1024,
    width=1024,
    num_inference_steps=25,
    guidance_scale=3.5,
    generator=torch.Generator("cuda").manual_seed(seed),
).images[0]
result.save("output_with_ip_adapter1.0.png")

pipe.set_ip_adapter_scale(0.0)
result = pipe(
    prompt="A car driving on a street.",
    ip_adapter_image=guidance_image,
    height=1024,
    width=1024,
    num_inference_steps=25,
    guidance_scale=3.5,
    generator=torch.Generator("cuda").manual_seed(seed),
).images[0]
result.save("output_without_ip_adapter.png")
