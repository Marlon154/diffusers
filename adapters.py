import torch
from transformers import CLIPTokenizer, T5TokenizerFast

from examples.community.pipeline_flux_semantic_guidance import FluxSemanticGuidancePipeline


# Load tokenizers
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", trust_remote_code=True)
tokenizer_2 = T5TokenizerFast.from_pretrained("t5-base", trust_remote_code=True)
#
# # Load CLIP Vision model for IP-Adapter
# clip_vision = CLIPVisionModel.from_pretrained(
#     "openai/clip-vit-large-patch14",
#     torch_dtype=torch.bfloat16
# )

# Create the pipeline
pipe = FluxSemanticGuidancePipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    tokenizer=tokenizer,
    tokenizer_2=tokenizer_2,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

# Move to GPU
pipe.to("cuda:1")
seed = 6543
prompt = "A cat holding a sign that says hello world"

image = pipe(
    prompt=prompt,
    guidance_scale=3.5,
    # true_cfg_scale=3.0,
    # negative_prompt="low quality, ugly, deformed, blurry, bad anatomy, bad proportions, extra limbs, watermark",
    edit_mom_beta=0.6,
    num_images_per_prompt=1,
    generator=torch.Generator(device="cuda").manual_seed(seed),
    num_inference_steps=28,
    width=1024,
    height=1024,
).images

print("len(image)", len(image))

for idx, image in enumerate(image):
    image.save(f"image-o-{idx}.png")
print("Saved image with IP-Adapter")

# Generate image with IP-Adapter
image = pipe(
    prompt=prompt,
    guidance_scale=3.5,
    # true_cfg_scale=3.0,
    # negative_prompt="low quality, ugly, deformed, blurry, bad anatomy, bad proportions, extra limbs, watermark",
    editing_prompt=["cat", "dog"],  # changes from cat to dog.
    reverse_editing_direction=[True, False],
    edit_warmup_steps=[6, 8],
    edit_guidance_scale=[6, 6.5],
    edit_threshold=[0.89, 0.89],
    edit_cooldown_steps=[25, 27],
    edit_momentum_scale=0.3,
    edit_mom_beta=0.6,
    num_images_per_prompt=1,
    generator=torch.Generator(device="cuda").manual_seed(seed),
    num_inference_steps=28,
    width=1024,
    height=1024,
).images

print("len(image)", len(image))

for idx, image in enumerate(image):
    image.save(f"image-{idx}.png")
print("Saved image with IP-Adapter")
