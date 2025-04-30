from concurrent.futures import ThreadPoolExecutor

import gradio as gr
import torch
from transformers import CLIPTokenizer, T5TokenizerFast

from examples.community.pipeline_flux_semantic_guidance import FluxSemanticGuidancePipeline


# Initialize models on GPUs
def init_pipeline(gpu_id, adapter_name=None, adapter_prompt=None, weight_name=None):
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", trust_remote_code=True)
    tokenizer_2 = T5TokenizerFast.from_pretrained("t5-base", trust_remote_code=True)

    pipe = FluxSemanticGuidancePipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    pipe = pipe.to(f"cuda:{gpu_id}")

    if adapter_name and adapter_prompt and weight_name:
        pipe.load_lora_weights(adapter_name, weight_name=weight_name)

    return pipe


# Initialize pipelines with different adapters
PIPES_CONFIG = {
    "base": {"gpu_id": 0},
    "edit": {"gpu_id": 1},
    "base-adapter": {
        "gpu_id": 2,
        "adapter_name": "aleksa-codes/flux-ghibsky-illustration",
        "adapter_prompt": "GHIBSKY style painting ",
        "weight_name": "lora.safetensors",
    },
    "edit-adapter": {
        "gpu_id": 3,
        "adapter_name": "aleksa-codes/flux-ghibsky-illustration",
        "adapter_prompt": "GHIBSKY style painting ",
        "weight_name": "lora.safetensors",
    },
    # "red-undersea": init_pipeline(3,
    #                               'prithivMLmods/Red-Undersea-Flux-LoRA',
    #                               "Red Undersea ",
    #                               'Red-Undersea.safetensors')
}

PIPES = {name: init_pipeline(**params) for name, params in PIPES_CONFIG.items()}


def generate_with_pipe(args):
    pipe_name, pipe, params = args
    return pipe(**params).images[0]


def generate_images(
    prompt,
    guidance_scale=3.5,
    num_inference_steps=28,
    seed=42,
    edit_momentum_scale=0.3,
    edit_mom_beta=0.6,
    # Edit 1 parameters
    editing_prompt_1=None,
    edit_guidance_scale_1=3.0,
    edit_warmup_steps_1=11,
    edit_threshold_1=0.9,
    edit_direction_1=False,
    edit_cooldown_steps_1=26,
    # Edit 2 parameters
    editing_prompt_2=None,
    edit_guidance_scale_2=3.0,
    edit_warmup_steps_2=11,
    edit_threshold_2=0.9,
    edit_direction_2=False,
    edit_cooldown_steps_2=26,
    # Edit 3 parameters
    editing_prompt_3=None,
    edit_guidance_scale_3=3.0,
    edit_warmup_steps_3=11,
    edit_threshold_3=0.9,
    edit_direction_3=False,
    edit_cooldown_steps_3=26,
):
    # Prepare edit parameters if any edit prompts exist
    editing_prompts = []
    edit_params = {
        "reverse_editing_direction": [],
        "edit_warmup_steps": [],
        "edit_guidance_scale": [],
        "edit_threshold": [],
        "edit_cooldown_steps": [],
    }

    # Collect active edit prompts and their parameters
    for edit_prompt, scale, warmup, threshold, direction, cooldown in [
        (
            editing_prompt_1,
            edit_guidance_scale_1,
            edit_warmup_steps_1,
            edit_threshold_1,
            edit_direction_1,
            edit_cooldown_steps_1,
        ),
        (
            editing_prompt_2,
            edit_guidance_scale_2,
            edit_warmup_steps_2,
            edit_threshold_2,
            edit_direction_2,
            edit_cooldown_steps_2,
        ),
        (
            editing_prompt_3,
            edit_guidance_scale_3,
            edit_warmup_steps_3,
            edit_threshold_3,
            edit_direction_3,
            edit_cooldown_steps_3,
        ),
    ]:
        if edit_prompt:
            editing_prompts.append(edit_prompt)
            edit_params["reverse_editing_direction"].append(direction)
            edit_params["edit_warmup_steps"].append(warmup)
            edit_params["edit_guidance_scale"].append(scale)
            edit_params["edit_threshold"].append(threshold)
            edit_params["edit_cooldown_steps"].append(cooldown)

    # Prepare generation parameters for each pipe
    pipe_tasks = []
    generators = {name: torch.Generator(device=f"cuda:{i}").manual_seed(seed) for i, name in enumerate(PIPES.keys())}

    for pipe_name, pipe in PIPES.items():
        # Prepare base parameters
        params = {
            "prompt": prompt if "adapter" not in pipe_name else PIPES_CONFIG[pipe_name]["adapter_prompt"] + prompt,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "generator": generators[pipe_name],
        }

        # Add edit parameters for edit pipes if edits exist
        if "edit" in pipe_name.lower() and editing_prompts:
            params.update(
                {
                    "editing_prompt": editing_prompts,
                    "edit_momentum_scale": edit_momentum_scale,
                    "edit_mom_beta": edit_mom_beta,
                    **edit_params,
                }
            )
            print(editing_prompts, edit_params)
        pipe_tasks.append((pipe_name, pipe, params))

    # Run generation in parallel
    with ThreadPoolExecutor(max_workers=len(PIPES)) as executor:
        results = list(executor.map(generate_with_pipe, pipe_tasks))

    # Return images in the order: base, edit, base-ghibsky, edit-ghibsky
    return tuple(results)


# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# FLUX Image Generation and Editing Demo")

    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Base Prompt")
            with gr.Row():
                guidance_scale = gr.Slider(0, 20, value=3.5, label="Guidance Scale")
                seed = gr.Slider(0, 1000, value=42, step=1, label="Seed")
                num_inference_steps = gr.Slider(6, 60, value=28, step=1, label="Num Inference Steps")

        with gr.Accordion("Momentum Parameters"):
            edit_momentum_scale = gr.Slider(0, 1, value=0.3, label="Edit Momentum Scale")
            edit_mom_beta = gr.Slider(0, 1, value=0.6, label="Edit Momentum Beta")

    with gr.Column():
        with gr.Row():
            base_image = gr.Image(label="Base Generated Image")
            edit_image = gr.Image(label="Base Edited Image")
        with gr.Row():
            adapter_image = gr.Image(label="Adapter Generated Image")
            adapter_edit_image = gr.Image(label="Adapter Edited Image")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Edit 1")
            editing_prompt_1 = gr.Textbox(label="Edit Prompt 1")
            edit_guidance_scale_1 = gr.Slider(0, 20, value=3.0, label="Guidance Scale")
            edit_warmup_steps_1 = gr.Slider(0, 50, step=1, value=11, label="Warmup Steps")
            edit_threshold_1 = gr.Slider(0, 1, value=0.9, label="Threshold")
            edit_direction_1 = gr.Checkbox(label="Reverse Direction")
            edit_cooldown_steps_1 = gr.Slider(0, 50, step=1, value=26, label="Cooldown Steps")

        with gr.Column():
            gr.Markdown("### Edit 2")
            editing_prompt_2 = gr.Textbox(label="Edit Prompt 2")
            edit_guidance_scale_2 = gr.Slider(0, 20, value=3.0, label="Guidance Scale")
            edit_warmup_steps_2 = gr.Slider(0, 50, step=1, value=11, label="Warmup Steps")
            edit_threshold_2 = gr.Slider(0, 1, value=0.9, label="Threshold")
            edit_direction_2 = gr.Checkbox(label="Reverse Direction")
            edit_cooldown_steps_2 = gr.Slider(0, 50, step=1, value=26, label="Cooldown Steps")

        with gr.Column():
            gr.Markdown("### Edit 3")
            editing_prompt_3 = gr.Textbox(label="Edit Prompt 3")
            edit_guidance_scale_3 = gr.Slider(0, 20, value=3.0, label="Guidance Scale")
            edit_warmup_steps_3 = gr.Slider(0, 50, value=11, label="Warmup Steps")
            edit_threshold_3 = gr.Slider(0, 1, value=0.9, label="Threshold")
            edit_direction_3 = gr.Checkbox(label="Reverse Direction")
            edit_cooldown_steps_3 = gr.Slider(0, 50, value=26, label="Cooldown Steps")

    generate_button = gr.Button("Generate")

    generate_button.click(
        fn=generate_images,
        inputs=[
            prompt,
            guidance_scale,
            num_inference_steps,
            seed,
            edit_momentum_scale,
            edit_mom_beta,
            editing_prompt_1,
            edit_guidance_scale_1,
            edit_warmup_steps_1,
            edit_threshold_1,
            edit_direction_1,
            edit_cooldown_steps_1,
            editing_prompt_2,
            edit_guidance_scale_2,
            edit_warmup_steps_2,
            edit_threshold_2,
            edit_direction_2,
            edit_cooldown_steps_2,
            editing_prompt_3,
            edit_guidance_scale_3,
            edit_warmup_steps_3,
            edit_threshold_3,
            edit_direction_3,
            edit_cooldown_steps_3,
        ],
        outputs=[base_image, edit_image, adapter_image, adapter_edit_image],
    )

if __name__ == "__main__":
    demo.launch(share=True)
