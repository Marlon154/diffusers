import torch

from diffusers import KandinskyV22PriorPipeline, KandinskyV22PriorEmb2EmbPipeline
from diffusers.utils import load_image
from pipeline_kandinsky2_2_semantic import KandinskyV22Pipeline

seed = 646
gpu = "cuda:2"
pipe_prior = KandinskyV22PriorPipeline.from_pretrained("kandinsky-community/kandinsky-2-2-prior")
pipe_prior.to(gpu)
prompt = "portrait of a woman, 4k photo"
out = pipe_prior(prompt, generator=torch.Generator(device=gpu).manual_seed(seed))
image_emb = out.image_embeds
zero_image_emb = out.negative_image_embeds

edit_prompt = "a women wearing sunglasses"
edit_out = pipe_prior(edit_prompt)
edit_image_emb = edit_out.image_embeds
edit_zero_image_emb = edit_out.negative_image_embeds

pipe = KandinskyV22Pipeline.from_pretrained("kandinsky-community/kandinsky-2-2-decoder")

pipe.to(gpu)
image = pipe(
    image_embeds=image_emb,
    edit_image_embeds=edit_image_emb,
    negative_image_embeds=zero_image_emb,
    enable_edit_guidance=True,
    height=768,
    width=768,
    num_inference_steps=50,
    edit_guidance_scale=7.0,
    edit_warmup_steps=15,
    edit_threshold=0.85,
    generator=torch.Generator(device="cuda").manual_seed(seed),
).images
image[0].save("basic_woman-sunglasses-edit1.png")

image = pipe(
    image_embeds=image_emb,
    edit_image_embeds=edit_image_emb,
    negative_image_embeds=zero_image_emb,
    enable_edit_guidance=True,
    height=768,
    width=768,
    num_inference_steps=50,
    edit_guidance_scale=7.5,
    edit_warmup_steps=15,
    generator=torch.Generator(device="cuda").manual_seed(seed),
).images
image[0].save("basic_woman-sunglasses-edit2.png")

image = pipe(
    image_embeds=image_emb,
    edit_image_embeds=edit_image_emb,
    negative_image_embeds=zero_image_emb,
    enable_edit_guidance=True,
    height=768,
    width=768,
    num_inference_steps=50,
    edit_guidance_scale=7.0,
    edit_warmup_steps=12,
    generator=torch.Generator(device="cuda").manual_seed(seed),
).images
image[0].save("basic_woman-sunglasses-edit3.png")

image = pipe(
    image_embeds=image_emb,
    # edit_image_embeds=guidance_image_embeds,
    negative_image_embeds=zero_image_emb,
    # enable_edit_guidance=True,
    height=768,
    width=768,
    num_inference_steps=50,
    edit_guidance_scale=3.0,
    edit_warmup_steps=30,
    generator=torch.Generator(device="cuda").manual_seed(seed),
).images
image[0].save("basic_woman-sunglasses-original.png")
