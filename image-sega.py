import torch
from transformers import CLIPImageProcessor, CLIPTokenizer, CLIPVisionModelWithProjection, T5TokenizerFast

from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import load_image
from examples.community.pipeline_flux_ip_semantic_guidance import FluxSemanticImageGuidancePipeline


# Load tokenizers
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", trust_remote_code=True)
tokenizer_2 = T5TokenizerFast.from_pretrained("t5-base", trust_remote_code=True)

# Load CLIP Vision model and feature extractor
image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    "openai/clip-vit-large-patch14", torch_dtype=torch.bfloat16
)
feature_extractor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")

# Create the pipeline with the FLUX model
pipe = FluxSemanticImageGuidancePipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    tokenizer=tokenizer,
    tokenizer_2=tokenizer_2,
    image_encoder=image_encoder,
    feature_extractor=feature_extractor,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

# Load guidance image
guidance_image = load_image(
    "https://github.com/exx8/differential-diffusion/blob/main/assets/input.jpg?raw=true",
)
pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda:1")

seed = 64

# Generate image without IP-Adapter guidance
prompt = "A car on a street"
image = pipe(
    prompt=prompt,
    num_inference_steps=28,
    generator=torch.Generator(device="cuda").manual_seed(seed),
).images[0]
image.save("flux-original.png")

# Load the specialized IP-Adapter weights from XLabs-AI
# pipe.load_ip_adapter("XLabs-AI/flux-ip-adapter-v2", weight_name="ip_adapter.safetensors")
pipe.load_ip_adapter(
    pretrained_model_name_or_path_or_dict="XLabs-AI/flux-ip-adapter",
    weight_name="ip_adapter.safetensors",
    # image_encoder_pretrained_model_name_or_path="openai/clip-vit-large-patch14"
)
pipe.to("cuda:1")
pipe.set_ip_adapter_scale(0.0)
prompt = "A car on a street"
image = pipe(
    prompt=prompt,
    num_inference_steps=28,
    sega_ip_adapter_image=guidance_image,
    use_sega=False,
    generator=torch.Generator(device="cuda").manual_seed(seed),
).images[0]
image.save("flux-original-scale0.png")

# Generate image with IP-Adapter guidance
prompt = "A car on a street"
image = pipe(
    prompt=prompt,
    num_inference_steps=28,
    sega_ip_adapter_image=guidance_image,
    sega_ip_adapter_scale=0.8,
    generator=torch.Generator(device="cuda").manual_seed(seed),
).images[0]
image.save("flux-ip-adapter-0.8.png")

image = pipe(
    prompt=prompt,
    num_inference_steps=28,
    sega_ip_adapter_image=guidance_image,
    sega_ip_adapter_scale=1.0,
    generator=torch.Generator(device="cuda").manual_seed(seed),
).images[0]
image.save("flux-ip-adapter-1.0.png")

# Try with stronger IP-Adapter influence
image = pipe(
    prompt=prompt,
    num_inference_steps=28,
    sega_ip_adapter_image=guidance_image,
    sega_ip_adapter_scale=8.0,  # Higher scale for stronger image guidance
    generator=torch.Generator(device="cuda").manual_seed(seed),
).images[0]
image.save("flux-ip-adapter-strong.png")

# Try with a different balance between text and image
image = pipe(
    prompt=prompt,
    num_inference_steps=28,
    sega_ip_adapter_image=guidance_image,
    sega_ip_adapter_scale=3.7,  # Moderate image guidance
    generator=torch.Generator(device="cuda").manual_seed(seed),
).images[0]
image.save("flux-ip-adapter-balanced.png")
