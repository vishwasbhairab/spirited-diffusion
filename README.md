# **🎨 Spirited Diffusion: Ghibli-Style AI Art Generator**

An advanced text-to-image generator that transforms any prompt into a beautiful piece of art in the iconic and beloved style of Studio Ghibli. This project involved fine-tuning a Stable Diffusion model on a custom dataset to teach it a new, unique artistic aesthetic.

## **✨ Live Demo**

Experience the magic yourself\! The model is deployed and permanently hosted on Hugging Face Spaces.

[**➡️ Launch the Spirited Diffusion App**](https://huggingface.co/spaces/vishwasbhairab/spirited-diffusion)

## **🖼️ Showcase: Before vs. After Fine-Tuning**

The power of fine-tuning is best shown visually. The base Stable Diffusion model understands the *words* "Ghibli style," but the fine-tuned model understands the *feeling*.

**Prompt:** a whimsical castle floating in the clouds

| Before (Base Stable Diffusion 1.5) | After (Spirited Diffusion) |
| :---- | :---- |
|  |  |

## **🚀 Project Overview**

Spirited Diffusion is a deep learning project that demonstrates advanced skills in generative AI and model customization. The core of this project was to take a powerful, pre-trained text-to-image model (Stable Diffusion v1.5) and retrain it on a small, curated dataset of images representing the Ghibli art style. The result is a specialized model that can be guided by a unique trigger phrase (ohwx) to create new, original artwork that captures the specific nuances of this aesthetic.

### **Key Features:**

* **Custom Fine-Tuned Model:** The model was trained using the DreamBooth technique to learn a new artistic style.  
* **Interactive Web UI:** A user-friendly interface built with Gradio allows anyone to generate images easily.  
* **Prompt Engineering:** The model responds to a special trigger word (ohwx) to apply the Ghibli style to any concept.  
* **Optimized Deployment:** Hosted on Hugging Face using a two-repository workflow for efficient management of large model files.

## **🛠️ Technology Stack**

This project was built using a modern stack of machine learning and web development tools.

| Technology | Description |
| :---- | :---- |
| **PyTorch** | The core deep learning framework for model training and inference. |
| **Hugging Face Diffusers** | The library used to load, fine-tune, and run the Stable Diffusion pipeline. |
| **Hugging Face Hub** | Used for both model storage (Git LFS) and application hosting (Spaces). |
| **Gradio** | The Python library used to rapidly create the interactive web interface. |
| **Git & Git LFS** | Essential for version control and managing the large (3.2 GB) model file. |

## **⚙️ How It Works**

The application is deployed using a robust, two-repository architecture on Hugging Face:

1. **Model Repository (spirited-diffusion-model):** This repository uses Git LFS to permanently store the large 3.2 GB fine-tuned .safetensors model file. It acts as a dedicated library for the heavy asset.  
2. **Space Repository (spirited-diffusion):** This repository contains only the lightweight application code (app.py, requirements.txt). When the application starts, it dynamically downloads the model from the Model Repository, separates storage from compute, and allows for a scalable and manageable deployment.

## **Usage Guide**

1. Navigate to the [**Live Demo**](https://huggingface.co/spaces/vishwasbhairab/spirited-diffusion).  
2. Enter a descriptive prompt for the scene you want to create.  
3. **Crucially, add the trigger phrase ohwx** at the end of your prompt to activate the Ghibli style.  
4. Click "Generate" and wait for your masterpiece\!

**Example Prompt:** A girl with a straw hat in a field of flowers, ohwx
