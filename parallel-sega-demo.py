from concurrent.futures.thread import ThreadPoolExecutor

import gradio as gr
import torch
from transformers import CLIPTokenizer, T5TokenizerFast

from examples.community.pipeline_flux_semantic_guidance import FluxSemanticGuidancePipeline

# Load the pipeline
model_id = "black-forest-labs/FLUX.1-dev"

# Create a list of available GPUs
gpu_ids = [2, 3, 5]
pipes = []

# with ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
#     results = list(executor.map(lambda x: SemanticFluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16, device="cuda:"+x), gpu_ids))

for gpu_id in gpu_ids:
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
    pipes.append(pipe)


def generate_image(args):
    pipe = args["pipe"]
    del args["pipe"]
    return pipe(**args).images[0]


def run_generation(
        prompt,
        prompt_w_edit,
        steps=30,
        guidance_scale=7.5,
        seed=42,
        edit_momentum_scale=0.3,
        edit_mom_beta=0.6,
        editing_prompt_1=None,
        edit_guidance_scale_1=6,
        edit_warmup_steps_1=15,
        edit_threshold_1=0.95,
        edit_direction_1=False,
        edit_cooldown_steps_1=0,
        editing_prompt_2=None,
        edit_guidance_scale_2=6,
        edit_warmup_steps_2=15,
        edit_threshold_2=0.95,
        edit_direction_2=False,
        edit_cooldown_steps_2=0,
        editing_prompt_3=None,
        edit_guidance_scale_3=6,
        edit_warmup_steps_3=15,
        edit_threshold_3=0.95,
        edit_direction_3=False,
        edit_cooldown_steps_3=1,
):
    negative_prompt = "ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, watermark, signature, cut off, low contrast, underexposed, overexposed, bad art, beginner art, amateur art, distorted face, blurry, draft, grainy, oversaturated, text, logo, advertisement, low resolution, jpeg artifacts, compression artifacts, cropped, worst quality, low quality, normal quality, gross proportions, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn eyes, mutation, deformed, dehydrated, bad proportions, extra limbs, stacked torsos, cloned face, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, error, duplicate layers, artifacts"

    generators = [torch.Generator(device="cuda").manual_seed(seed) for _ in range(len(pipes))]

    num_edits = 1 if editing_prompt_1 else 0 + 1 if editing_prompt_2 else 0 + 1 if editing_prompt_3 else 0
    editing_prompts = [editing_prompt_1, editing_prompt_2, editing_prompt_3][:num_edits]
    edit_guidance_scales = [edit_guidance_scale_1, edit_guidance_scale_2, edit_guidance_scale_3][:num_edits]
    edit_warmup_steps = [edit_warmup_steps_1, edit_warmup_steps_2, edit_warmup_steps_3][:num_edits]
    edit_threshold = [edit_threshold_1, edit_threshold_2, edit_threshold_3][:num_edits]
    edit_direction = [edit_direction_1, edit_direction_2, edit_direction_3][:num_edits]
    edit_cooldown_steps = [edit_cooldown_steps_1, edit_cooldown_steps_2, edit_cooldown_steps_3][:num_edits]

    normal_original = {
        "pipe": pipes[0],
        "prompt": prompt,
        "negative_prompt": None,
        "num_inference_steps": steps,
        "num_images_per_prompt": 1,
        "guidance_scale": guidance_scale,
        "editing_prompt": None,
        "generator": generators[0],
        "true_cfg_scale": 1.0,
    }

    normal_edit = {
        "pipe": pipes[1],
        "prompt": prompt_w_edit,
        "negative_prompt": None,
        "num_inference_steps": steps,
        "num_images_per_prompt": 1,
        "guidance_scale": guidance_scale,
        "editing_prompt": None,
        "generator": generators[1],
        "true_cfg_scale": 1.0,
    }

    sega_image = {
        "pipe": pipes[2],
        "prompt": prompt,
        "negative_prompt": None,
        "num_inference_steps": steps,
        "num_images_per_prompt": 1,
        "guidance_scale": guidance_scale,
        "editing_prompt": editing_prompts,
        "reverse_editing_direction": edit_direction,
        "edit_warmup_steps": edit_warmup_steps,
        "edit_guidance_scale": edit_guidance_scales,
        "edit_threshold": edit_threshold,
        "edit_momentum_scale": edit_momentum_scale,
        "edit_mom_beta": edit_mom_beta,
        "edit_cooldown_steps": edit_cooldown_steps,
        "generator": generators[2],
        "true_cfg_scale": 1.0,
    }

    params = [normal_original, normal_edit, sega_image]

    with ThreadPoolExecutor(max_workers=len(pipes)) as executor:
        results = list(executor.map(generate_image, params))
    return results[0], results[1], results[2]


# Define the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# SEGA: Semantic Guidance for Generating and Editing Images")
    gr.Markdown("This demo uses the FLUX model to generate and edit images based on a prompt.")
    gr.Markdown("**Note**: It takes about 1 minute to generate the images.")

    with gr.Row():
        original_image = gr.Image(label="Original Image")
        original_edited = gr.Image(label="Original Edit Prompt Image")
        sega_image = gr.Image(label="SeGa Image")

    with gr.Row():
        prompt = gr.Textbox(label="Prompt")
        prompt_w_edit = gr.Textbox(label="Prompt with Edit")
    with gr.Row():
        guidance_scale = gr.Slider(minimum=0, maximum=20, value=7.5, step=0.1, label="Guidance Scale")
        steps = gr.Slider(minimum=1, maximum=100, value=30, step=1, label="Steps")
        seed = gr.Number(value=46646, label="Seed")

    with gr.Column(variant="compact"):
        gr.Markdown("### Editing Prompt 1")
        with gr.Row():
            editing_prompt_1 = gr.Textbox(label="Editing Prompt 1")
            edit_guidance_scale_1 = gr.Slider(minimum=0, maximum=20, value=7.5, step=0.1, label="Edit Guidance Scale")
            edit_warmup_steps_1 = gr.Slider(minimum=0, maximum=50, value=8, step=1, label="Edit Warmup Steps")
        with gr.Row():
            edit_threshold_1 = gr.Slider(minimum=0, maximum=1, value=0.95, step=0.01, label="Edit Threshold")
            edit_direction_1 = gr.Checkbox(label="Reverse Edit Direction", value=False)
            edit_cooldown_steps_1 = gr.Slider(minimum=0, maximum=50, value=28, step=1, label="Edit Cooldown Steps")

    with gr.Column(variant="compact"):
        gr.Markdown("### Editing Prompt 2")
        with gr.Row():
            editing_prompt_2 = gr.Textbox(label="Editing Prompt 3")
            edit_guidance_scale_2 = gr.Slider(minimum=0, maximum=20, value=7.5, step=0.1, label="Edit Guidance Scale")
            edit_warmup_steps_2 = gr.Slider(minimum=0, maximum=50, value=8, step=1, label="Edit Warmup Steps")
        with gr.Row():
            edit_threshold_2 = gr.Slider(minimum=0, maximum=1, value=0.95, step=0.01, label="Edit Threshold")
            edit_direction_2 = gr.Checkbox(label="Reverse Edit Direction", value=False)
            edit_cooldown_steps_2 = gr.Slider(minimum=0, maximum=50, value=28, step=1, label="Edit Cooldown Steps")

    with gr.Column(variant="compact"):
        gr.Markdown("### Editing Prompt 3")
        with gr.Row():
            editing_prompt_3 = gr.Textbox(label="Editing Prompt 3")
            edit_guidance_scale_3 = gr.Slider(minimum=0, maximum=20, value=7.5, step=0.1, label="Edit Guidance Scale")
            edit_warmup_steps_3 = gr.Slider(minimum=0, maximum=50, value=8, step=1, label="Edit Warmup Steps")
        with gr.Row():
            edit_threshold_3 = gr.Slider(minimum=0, maximum=1, value=0.95, step=0.01, label="Edit Threshold")
            edit_direction_3 = gr.Checkbox(label="Reverse Edit Direction", value=False)
            edit_cooldown_steps_3 = gr.Slider(minimum=0, maximum=50, value=28, step=1, label="Edit Cooldown Steps")

    with gr.Row():
        edit_momentum_scale = gr.Slider(minimum=0, maximum=1, value=0.3, step=0.01, label="Edit Momentum Scale")
        edit_mom_beta = gr.Slider(minimum=0, maximum=1, value=0.6, step=0.01, label="Edit Momentum Beta")

    generate_button = gr.Button("Generate Images")

    generate_button.click(
        fn=run_generation,
        inputs=[
            prompt, prompt_w_edit, steps, guidance_scale, seed, edit_momentum_scale, edit_mom_beta,
            editing_prompt_1, edit_guidance_scale_1, edit_warmup_steps_1, edit_threshold_1, edit_direction_1, edit_cooldown_steps_1,
            editing_prompt_2, edit_guidance_scale_2, edit_warmup_steps_2, edit_threshold_2, edit_direction_2, edit_cooldown_steps_2,
            editing_prompt_3, edit_guidance_scale_3, edit_warmup_steps_3, edit_threshold_3, edit_direction_3, edit_cooldown_steps_3,
        ],
        outputs=[original_image, original_edited, sega_image]
    )

demo.launch(share=True)
