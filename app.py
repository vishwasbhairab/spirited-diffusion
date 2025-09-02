import gradio as gr
import numpy as np
import random
import torch
from diffusers import DiffusionPipeline

# --- Application Setup ---
# This setup will run once when the application starts.

# 1. Device and Data Type Configuration
# Check for an available CUDA GPU and set the data type for faster processing.
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
print(f"Using device: {device}")

# 2. Model Loading
# IMPORTANT: Update this path to the exact name of your model file.
# This file should be in the same directory as your app.py.
model_path = "./your_ghibli_model.safetensors" # <-- RENAME THIS FILENAME

print("Loading the model... This might take a moment.")
# Use `from_single_file` to correctly load your full fine-tuned DreamBooth model.
pipe = DiffusionPipeline.from_single_file(
    model_path,
    torch_dtype=torch_dtype,
    use_safetensors=True
)
pipe = pipe.to(device)
print("Model loaded successfully.")

# --- Constants ---
MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 1024

# --- Core Inference Function ---
# This function is called every time a user clicks "Generate".

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
    """
    Generates an image based on the provided inputs.
    """
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
        
    # The generator uses the seed to ensure reproducibility.
    generator = torch.Generator(device=device).manual_seed(seed)
    
    # Generate the image using the pipeline.
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=int(num_inference_steps), # Ensure steps is an integer
        width=width,
        height=height,
        generator=generator,
    ).images[0]
    
    return image, seed

# --- Gradio Web UI Definition ---
# This block defines the layout and functionality of the web interface.

css = """
#col-container {
    margin: 0 auto;
    max-width: 768px;
}
"""

with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        # Title and introduction
        gr.Markdown("""
            # 🎨 Spirited Diffusion: Ghibli-Style Art Generator
            Enter a prompt to generate an image in the beautiful art style of Studio Ghibli.
            **Remember to include your special trigger phrase (e.g., "ohwx")** to activate the style!
        """)
        
        # Main UI components: prompt box, button, and result image
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

        # Accordion for advanced settings
        with gr.Accordion("Advanced Settings", open=False):
            negative_prompt_input = gr.Text(
                label="Negative prompt",
                max_lines=1,
                placeholder="e.g., ugly, blurry, deformed",
                visible=True,
            )
            seed_slider = gr.Slider(
                label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=0
            )
            randomize_seed_checkbox = gr.Checkbox(label="Randomize seed", value=True)
            
            with gr.Row():
                width_slider = gr.Slider(
                    label="Width", minimum=256, maximum=MAX_IMAGE_SIZE, step=64, value=768
                )
                height_slider = gr.Slider(
                    label="Height", minimum=256, maximum=MAX_IMAGE_SIZE, step=64, value=768
                )
            with gr.Row():
                guidance_scale_slider = gr.Slider(
                    label="Guidance scale", minimum=0.0, maximum=20.0, step=0.5, value=7.5
                )
                steps_slider = gr.Slider(
                    label="Number of inference steps", minimum=1, maximum=100, step=1, value=30
                )
        
        # Example prompts to guide the user
        example_prompts = [
            "A girl on a hill watching the clouds, ohwx",
            "A whimsical cat bus flying through a starry night, ohwx",
            "A bustling market in a fantasy city, ohwx",
            "A majestic castle floating in the sky, ohwx",
        ]
        gr.Examples(examples=example_prompts, inputs=[prompt_input])

    # Connect the UI components to the 'infer' function
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

