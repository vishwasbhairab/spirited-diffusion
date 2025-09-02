import gradio as gr
import numpy as np
import random
import torch
from diffusers import DiffusionPipeline
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

# --- Application Setup ---

# 1. Device and Data Type Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
print(f"Using device: {device}")

# 2. Model Loading from Hugging Face Hub (Corrected Two-Step Process)

# The official ID for the base model on the Hugging Face Hub.
base_model_id = "runwayml/stable-diffusion-v1-5"
# The ID of your Model Repository on Hugging Face.
your_model_repo_id = "vishwasbhairab/spirited-diffusion-model"
# The filename of your model within your repository.
your_model_filename = "diffusion_pytorch_model.safetensors"

print(f"Loading base pipeline from '{base_model_id}'...")
# Step 1: Load the standard pipeline. This provides the correct architecture.
pipe = DiffusionPipeline.from_pretrained(
    base_model_id,
    torch_dtype=torch_dtype,
    use_safetensors=True
)

print(f"Downloading your fine-tuned UNet weights from '{your_model_repo_id}'...")
# Step 2: Download your custom UNet weights file from your repository.
unet_weights_path = hf_hub_download(
    repo_id=your_model_repo_id,
    filename=your_model_filename
)

print("Injecting your custom weights into the pipeline...")
# Step 3: Load your weights and replace the pipeline's UNet with them.
unet_state_dict = load_file(unet_weights_path)
pipe.unet.load_state_dict(unet_state_dict)

pipe = pipe.to(device)
print("Model loaded successfully. Application is ready!")


# --- Constants ---
MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 1024

# --- Core Inference Function ---
def infer(
    prompt,
    negative_prompt,
    seed,
    randomize_seed,
    width,
    height,
    guidance_scale,
    num_inference_steps,
    progress=gr.Progress(track_tqdm=True),
):
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    generator = torch.Generator(device=device).manual_seed(seed)
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=int(num_inference_steps),
        width=width,
        height=height,
        generator=generator,
    ).images[0]
    return image, seed

# --- Gradio Web UI Definition ---
css = """
#col-container { margin: 0 auto; max-width: 768px; }
"""

with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown("""
            # 🎨 Spirited Diffusion: Ghibli-Style Art Generator
            Enter a prompt to generate an image in the beautiful art style of Studio Ghibli.
            **Remember to include your special trigger phrase ("ohwx")** to activate the style!
        """)
        with gr.Row():
            prompt_input = gr.Text(
                label="Prompt",
                show_label=False,
                max_lines=1,
                placeholder="e.g., a cozy cabin in a magical forest, ohwx",
                container=False,
            )
            run_button = gr.Button("Generate", scale=0, variant="primary")
        result_image = gr.Image(label="Result", show_label=False)
        with gr.Accordion("Advanced Settings", open=False):
            negative_prompt_input = gr.Text(
                label="Negative prompt",
                max_lines=1,
                placeholder="e.g., ugly, blurry, deformed",
                visible=True,
            )
            seed_slider = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=0)
            randomize_seed_checkbox = gr.Checkbox(label="Randomize seed", value=True)
            with gr.Row():
                width_slider = gr.Slider(label="Width", minimum=256, maximum=MAX_IMAGE_SIZE, step=64, value=768)
                height_slider = gr.Slider(label="Height", minimum=256, maximum=MAX_IMAGE_SIZE, step=64, value=768)
            with gr.Row():
                guidance_scale_slider = gr.Slider(label="Guidance scale", minimum=0.0, maximum=20.0, step=0.5, value=7.5)
                steps_slider = gr.Slider(label="Number of inference steps", minimum=1, maximum=100, step=1, value=30)
        example_prompts = [
            "A girl on a hill watching the clouds, ohwx",
            "A whimsical cat bus flying through a starry night, ohwx",
            "A bustling market in a fantasy city, ohwx",
            "A majestic castle floating in the sky, ohwx",
        ]
        gr.Examples(examples=example_prompts, inputs=[prompt_input])
    gr.on(
        triggers=[run_button.click, prompt_input.submit],
        fn=infer,
        inputs=[
            prompt_input,
            negative_prompt_input,
            seed_slider,
            randomize_seed_checkbox,
            width_slider,
            height_slider,
            guidance_scale_slider,
            steps_slider,
        ],
        outputs=[result_image, seed_slider],
        api_name="run"
    )

# --- Launch the Application ---
if __name__ == "__main__":
    demo.launch()

