import os

import torch
from transformers import CLIPTokenizer, T5TokenizerFast

from examples.community.pipeline_flux_semantic_guidance import FluxSemanticGuidancePipeline


def main():
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", trust_remote_code=True)
    tokenizer_2 = T5TokenizerFast.from_pretrained("t5-base", trust_remote_code=True)
    seed = 456745

    pipe = FluxSemanticGuidancePipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    pipe.to("cuda:6")
    prompt = "A couple in front of a Mediterranean harbor."
    joined_prompt = "A couple with sunglasses in front of a Mediterranean harbor."

    pics = "pics-comparison/" + str(seed)
    os.makedirs(pics, exist_ok=True)

    # # Original
    # image = pipe(
    #     prompt=prompt,
    #     guidance_scale=3.5,
    #     generator=torch.Generator(device="cuda").manual_seed(seed),
    # ).images[0]
    # image.save(f"{pics}/original.png")
    #
    # # Original joined
    # image = pipe(
    #     prompt=joined_prompt,
    #     guidance_scale=3.5,
    #     generator=torch.Generator(device="cuda").manual_seed(seed),
    # ).images[0]
    # image.save(f"{pics}/original-joined.png")
    #
    # # Edit
    # image = pipe(
    #     prompt=prompt,
    #     guidance_scale=3.5,
    #     editing_prompt=["sunglasses"],
    #     reverse_editing_direction=[False],
    #     edit_warmup_steps=[11],
    #     edit_guidance_scale=[3.5],
    #     edit_cooldown_steps=[27],
    #     edit_threshold=[0.96],
    #     edit_momentum_scale=0.3,
    #     edit_mom_beta=0.6,
    #     generator=torch.Generator(device="cuda").manual_seed(seed),
    # ).images[0]
    # image.save(f"{pics}/sega.png")
    #
    # # Original
    for seed in range(0, 20):
        pics = "pics-comparison-royals/" + str(seed)
        os.makedirs(pics, exist_ok=True)
        prompt = "An image of a king sitting on a throne."
        joined_prompt = "An image of a queen sitting on a throne."
        edit = ["king", "queen"]
        image = pipe(
            prompt=prompt,
            guidance_scale=3.5,
            generator=torch.Generator(device="cuda").manual_seed(seed),
        ).images[0]
        image.save(f"{pics}/original.png")

        # Original joined
        image = pipe(
            prompt=joined_prompt,
            guidance_scale=3.5,
            generator=torch.Generator(device="cuda").manual_seed(seed),
        ).images[0]
        image.save(f"{pics}/updated_prompt.png")

        # Edit
        image = pipe(
            prompt=prompt,
            guidance_scale=3.5,
            editing_prompt=edit,
            reverse_editing_direction=[True, False],
            edit_warmup_steps=[6, 7],
            edit_guidance_scale=[5.0, 6.0],
            edit_cooldown_steps=[27, 27],
            edit_threshold=[0.85, 0.85],
            edit_momentum_scale=0.3,
            edit_mom_beta=0.6,
            generator=torch.Generator(device="cuda").manual_seed(seed),
        ).images[0]
        image.save(f"{pics}/sega.png")


if __name__ == "__main__":
    main()
