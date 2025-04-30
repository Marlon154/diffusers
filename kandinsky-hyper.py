import itertools
import json
from pathlib import Path

import torch
from tqdm import tqdm

from diffusers import KandinskyV22PriorEmb2EmbPipeline, KandinskyV22PriorPipeline
from diffusers.utils import load_image
from pipeline_kandinsky2_2_semantic import KandinskyV22Pipeline


seed = 431298
output_path = Path("output-kandinsky")
output_path.mkdir(parents=True, exist_ok=True)

prompt_image_pairs = {
    "portrait of a woman, 4k photo": "sunglasses.jpg",
    "car driving over a street": "input.jpg",
    "a cat playing with a ball": "tennis-ball.jpg",
    "close-up of a person, high detail": "sunglasses-2.jpg",
    "kids playing in the park": "tennis-ball.jpg",
}

experiment_config = {
    "hyperparameters": {
        "edit_guidance_scale": [2.0, 4.0, 6.0, 8.0],
        "edit_warmup_steps": [10, 15, 25, 30, 35],
    }
}

# Initialize the pipelines
print("Initializing pipelines...")
pipe_prior = KandinskyV22PriorPipeline.from_pretrained("kandinsky-community/kandinsky-2-2-prior")
pipe_prior.to("cuda")

pipe = KandinskyV22Pipeline.from_pretrained("kandinsky-community/kandinsky-2-2-decoder")
pipe_prior_image = KandinskyV22PriorEmb2EmbPipeline.from_pretrained("kandinsky-community/kandinsky-2-2-prior")
pipe_prior_image.to("cuda")
pipe.to("cuda")

results = []
pipe.set_progress_bar_config(disable=True)

for prompt, img_path in tqdm(prompt_image_pairs.items(), total=len(prompt_image_pairs), desc="Processing pairs"):
    out = pipe_prior(prompt)
    image_emb = out.image_embeds
    zero_image_emb = out.negative_image_embeds

    try:
        guidance_image = load_image(img_path)
        guidance_image_embeds = pipe_prior_image._encode_image(
            [guidance_image], device="cuda", num_images_per_prompt=1
        )

        baseline_image = pipe(
            image_embeds=image_emb,
            negative_image_embeds=zero_image_emb,
            height=768,
            width=768,
            num_inference_steps=50,
            generator=torch.Generator(device="cuda").manual_seed(seed),
        ).images
        scene_path = output_path / prompt.replace(" ", "_")[:12]
        scene_path.mkdir(parents=True, exist_ok=True)
        baseline_filename = scene_path / ("original" + img_path.split(".")[0] + ".png")
        baseline_image[0].save(baseline_filename)
        results.append(
            {
                "prompt": prompt,
                "guidance_image": img_path,
                "edit_guidance": False,
                "hyperparameters": "baseline",
                "output_file": baseline_filename,
            }
        )

        hp_keys = list(experiment_config["hyperparameters"].keys())
        hp_values = list(experiment_config["hyperparameters"].values())

        for hp_combination in tqdm(
            list(itertools.product(*hp_values)), desc=f"  Running experiments for '{prompt[:20]}...'", leave=False
        ):
            hp_dict = dict(zip(hp_keys, hp_combination))

            hp_str = "-".join([f"{k}{v}" for k, v in hp_dict.items()])
            output_filename = scene_path / f"edit_{hp_str}.png"

            image = pipe(
                image_embeds=image_emb,
                edit_image_embeds=guidance_image_embeds,
                negative_image_embeds=zero_image_emb,
                enable_edit_guidance=True,
                height=768,
                width=768,
                generator=torch.Generator(device="cuda").manual_seed(seed),
                **hp_dict,
            ).images

            # Save the generated image
            image[0].save(output_filename)

            # Record the experiment configuration
            results.append(
                {
                    "prompt": prompt,
                    "guidance_image": str(img_path),
                    "edit_guidance": True,
                    "hyperparameters": hp_dict,
                    "output_file": output_filename,
                }
            )

    except Exception as e:
        print(f"  Error processing {img_path}: {e}")
        continue

# Save experiment results to a JSON file
with open(output_path / "experiment_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nExperiments completed. Generated {len(results)} images.")
print("Results saved to output-kandinsky/experiment_results.json")
