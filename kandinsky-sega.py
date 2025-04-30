from diffusers import KandinskyV22PriorEmb2EmbPipeline, KandinskyV22PriorPipeline
from diffusers.utils import load_image
from pipeline_kandinsky2_2_semantic import KandinskyV22Pipeline


pipe_prior = KandinskyV22PriorPipeline.from_pretrained("kandinsky-community/kandinsky-2-2-prior")
pipe_prior.to("cuda")
prompt = "portrait of a woman, 4k photo"
out = pipe_prior(prompt)
image_emb = out.image_embeds
zero_image_emb = out.negative_image_embeds

# guidance_image = load_image(
#     "https://github.com/exx8/differential-diffusion/blob/main/assets/input.jpg?raw=true",
# )
guidance_image = load_image("sunglasses.jpg")

pipe = KandinskyV22Pipeline.from_pretrained("kandinsky-community/kandinsky-2-2-decoder")
pipe_prior_image = KandinskyV22PriorEmb2EmbPipeline.from_pretrained("kandinsky-community/kandinsky-2-2-prior")
pipe_prior_image.to("cuda")

guidance_image_embeds = pipe_prior_image._encode_image([guidance_image], device="cuda", num_images_per_prompt=1)

pipe.to("cuda")
image = pipe(
    image_embeds=image_emb,
    edit_image_embeds=guidance_image_embeds,
    negative_image_embeds=zero_image_emb,
    enable_edit_guidance=True,
    height=768,
    width=768,
    num_inference_steps=50,
    edit_guidance_scale=4.0,
    edit_warmup_steps=17,
).images
image[0].save("woman-sunglasses-edit1.png")

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
).images
image[0].save("woman-sunglasses-original.png")
