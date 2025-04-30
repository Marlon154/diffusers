import multiprocessing
import os
from multiprocessing import Process

import torch
from transformers import CLIPTokenizer, T5TokenizerFast

from examples.community.pipeline_flux_semantic_guidance import FluxSemanticGuidancePipeline


def main(gpu, seeds, adapter_name=None, adapter_prompt=None, weight_name=None):
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", trust_remote_code=True)
    tokenizer_2 = T5TokenizerFast.from_pretrained("t5-base", trust_remote_code=True)

    pipe = FluxSemanticGuidancePipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    pipe.to(gpu)
    prompt = "A portrait of a men.'"

    for seed in seeds:
        pics = "pics/" + str(seed)
        if adapter_name and adapter_prompt and weight_name:
            pipe.load_lora_weights(adapter_name, weight_name=weight_name)
            prompt = adapter_prompt + prompt
            base = adapter_prompt
        else:
            base = "base"

        image = pipe(
            prompt=prompt,
            guidance_scale=3.5,
            generator=torch.Generator(device="cuda").manual_seed(seed),
        ).images[0]

        image.save(f"{pics}/{base}_original.png")

        image = pipe(
            prompt=prompt,
            guidance_scale=3.5,
            editing_prompt=["sunglasses"],
            reverse_editing_direction=[False],
            edit_warmup_steps=[11],
            edit_guidance_scale=[3],
            edit_cooldown_steps=[26],
            edit_threshold=[0.9],
            edit_momentum_scale=0.3,
            edit_mom_beta=0.6,
            generator=torch.Generator(device="cuda").manual_seed(seed),
        ).images[0]

        image.save(f"{pics}/{base}_edit-1.png")

        image = pipe(
            prompt=prompt,
            guidance_scale=3.5,
            editing_prompt=["very old, grandpa"],
            reverse_editing_direction=[False],
            edit_warmup_steps=[8],
            edit_guidance_scale=[5],
            edit_cooldown_steps=[26],
            edit_threshold=[0.82],
            edit_momentum_scale=0.3,
            edit_mom_beta=0.6,
            generator=torch.Generator(device="cuda").manual_seed(seed),
        ).images[0]

        image.save(f"{pics}/{base}_edit-2.png")

        # car
        # image = pipe(
        #     prompt=prompt,
        #     guidance_scale=3.5,
        #     editing_prompt=["sunglasses"],
        #     reverse_editing_direction=[False],
        #     edit_warmup_steps=[9],
        #     edit_guidance_scale=[5],
        #     edit_cooldown_steps=[26],
        #     edit_threshold=[0.85],
        #     edit_momentum_scale=0.3,
        #     edit_mom_beta=0.6,
        #     generator=torch.Generator(device="cuda").manual_seed(seed),
        # ).images[0]
        #
        # image.save(f"{pics}/{base}_edit-1.png")
        #
        # image = pipe(
        #     prompt=prompt,
        #     guidance_scale=3.5,
        #     editing_prompt=["sport cap"],
        #     reverse_editing_direction=[False],
        #     edit_warmup_steps=[8],
        #     edit_guidance_scale=[5],
        #     edit_cooldown_steps=[26],
        #     edit_threshold=[0.82],
        #     edit_momentum_scale=0.3,
        #     edit_mom_beta=0.6,
        #     generator=torch.Generator(device="cuda").manual_seed(seed),
        # ).images[0]
        #
        # image.save(f"{pics}/{base}_edit-2.png")

        # image = pipe(
        #     prompt=prompt,
        #     guidance_scale=3.5,
        #     editing_prompt=["apple tree", "dog sitting on the right side of a ca"],
        #     reverse_editing_direction=[False, False],
        #     edit_warmup_steps=[8, 8],
        #     edit_guidance_scale=[4, 4.5],
        #     edit_cooldown_steps=[26, 26],
        #     edit_threshold=[0.85, 0.92],
        #     edit_momentum_scale=0.3,
        #     edit_mom_beta=0.6,
        #     generator=torch.Generator(device="cuda").manual_seed(seed),
        # ).images[0]
        #
        # image.save(f"{pics}/{base}_edit-3.png")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")

    seeds = [y for y in range(16)]
    os.makedirs("pics", exist_ok=True)
    for seed in seeds:
        pics = "pics/" + str(seed)
        os.makedirs(pics, exist_ok=True)

    processes = []
    try:
        p1 = Process(target=main, args=("cuda:1", seeds, False))
        p2 = Process(
            target=main,
            args=(
                "cuda:2",
                seeds,
                "aleksa-codes/flux-ghibsky-illustration",
                "GHIBSKY style painting ",
                "lora.safetensors",
            ),
        )
        p3 = Process(
            target=main,
            args=("cuda:3", seeds, "goofyai/3D_Render_for_Flux", "3D render ", "3D_render_flux.safetensors"),
        )
        p4 = Process(
            target=main,
            args=(
                "cuda:4",
                seeds,
                "prithivMLmods/Red-Undersea-Flux-LoRA",
                "Red Undersea ",
                "Red-Undersea.safetensors",
            ),
        )
        p5 = Process(
            target=main,
            args=(
                "cuda:5",
                seeds,
                "Norod78/chalk-board-drawing-flux",
                "ChalkBoardDrawing ",
                "Chalk_Board_Drawing_FLUX.safetensors",
            ),
        )
        processes = [p1, p2, p3, p4, p5]

        for p in processes:
            p.start()

        for p in processes:
            p.join()

    except KeyboardInterrupt:
        print("Caught keyboard interrupt, terminating processes...")
        for p in processes:
            if p.is_alive():
                p.terminate()
                p.join()  # Wait for termination
        raise
    except Exception as e:
        print(f"An error occurred: {e}")
        for p in processes:
            if p.is_alive():
                p.terminate()
                p.join()
