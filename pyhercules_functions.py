# -*- coding: utf-8 -*-
# hercules_functions.py
"""
Collection of embedding, LLM, and captioning functions for use with Hercules.
Handles model loading and API configuration.
"""

import os
import sys
import random
import re
import json
import time
import warnings
from typing import List, Any, Optional, Tuple
from dotenv import load_dotenv
# Load environment variables from .env file if it exists
# This makes the functions runnable in different environments
load_dotenv()
# --- Suppress specific warnings ---
# Suppress Hugging Face version warnings if they become noisy
warnings.filterwarnings("ignore", message=".*torch_dtype.*", category=UserWarning)
# Suppress potential tqdm warnings from SentenceTransformer if run non-interactively
# warnings.filterwarnings("ignore", message=".*Using `tqdm.*", category=FutureWarning)

# --- Essential Imports ---
import numpy as np

# --- Optional Imports (Load on Demand or Guarded) ---
_google_genai_available = False
try:
    import google.generativeai as genai
    _google_genai_available = True
    print("Google Generative AI SDK found.")
except ImportError:
    print("Warning: Google Generative AI SDK not found. Google functions will be unavailable.")
    print("Install with: pip install google-generativeai")

_sentence_transformers_available = False
try:
    from sentence_transformers import SentenceTransformer
    _sentence_transformers_available = True
    print("SentenceTransformers library found.")
except ImportError:
    print("Warning: SentenceTransformers library not found. Local text/image embedding functions will be unavailable.")
    print("Install with: pip install sentence-transformers torch") # torch often needed

_transformers_available = False
_pillow_available = False
_torch_available = False
_huggingface_hub_available = False
try:
    from transformers import AutoProcessor, Gemma3ForConditionalGeneration, BlipProcessor, BlipForConditionalGeneration
    _transformers_available = True
    print("Hugging Face Transformers library found.")
    from PIL import Image
    _pillow_available = True
    print("Pillow library found.")
    import torch
    _torch_available = True
    print("PyTorch library found.")
    from huggingface_hub import login as hf_login
    _huggingface_hub_available = True
    print("Hugging Face Hub library found.")
except ImportError as e:
    print(f"Warning: Missing libraries for local LLM/Captioning ({e}). These functions will be unavailable.")
    print("Install with: pip install transformers torch Pillow accelerate huggingface-hub") # accelerate often useful, huggingface-hub for login


# =============================================================================
# Configuration & Model Cache
# =============================================================================

# --- Google API Key Configuration ---
_google_configured = False
def configure_google_api():
    global _google_configured
    if not _google_genai_available or _google_configured:
        return _google_configured

    print("Attempting to configure Google API Key...")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        print("Error: GOOGLE_API_KEY environment variable not set.")
        print("Google Cloud functions require this key.")
        return False
    if "YOUR_API_KEY_HERE" in GOOGLE_API_KEY: # Simple placeholder check
         print("Error: GOOGLE_API_KEY seems to be a placeholder. Please set a valid key.")
         return False

    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        print("Google GenAI configured successfully.")
        _google_configured = True
        return True
    except Exception as e:
        print(f"Error configuring Google GenAI (check API key validity and permissions): {e}")
        _google_configured = False
        return False

# --- Hugging Face Hub Token Configuration ---
_huggingface_login_attempted = False # Flag to ensure login logic runs only once per session

def try_huggingface_login():
    """
    Attempts to log in to Hugging Face Hub using HUGGINGFACE_HUB_TOKEN environment variable.
    This is primarily for accessing gated models like Gemma.
    """
    global _huggingface_login_attempted
    if _huggingface_login_attempted: # Don't try again if already attempted
        return
    _huggingface_login_attempted = True

    if not _huggingface_hub_available:
        print("Hugging Face Hub library not found, cannot attempt login via token.")
        print("Set HUGGINGFACE_HUB_TOKEN or use `huggingface-cli login` for gated models.")
        return

    print("Checking Hugging Face Hub token for gated models...")
    HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")

    if HF_TOKEN:
        print("Found HUGGINGFACE_HUB_TOKEN. Attempting login to Hugging Face Hub...")
        try:
            hf_login(token=HF_TOKEN)
            print("Hugging Face Hub login successful with HUGGINGFACE_HUB_TOKEN.")
        except Exception as e:
            print(f"Warning: Hugging Face Hub login with HUGGINGFACE_HUB_TOKEN failed: {e}")
            print("Loading gated models (e.g., Gemma) from Hugging Face Hub may fail.")
            print("Ensure the token is valid and has permissions for the model.")
    else:
        print("HUGGINGFACE_HUB_TOKEN environment variable not set.")
        print("For gated models (e.g., Gemma), this token might be required if you haven't")
        print("logged in via `huggingface-cli login` previously.")


# --- Local Model Cache ---
# Use dictionaries to cache loaded models/processors by name
_loaded_embedding_models = {}
_loaded_llm_models = {}
_loaded_captioning_models = {}
_loaded_processors = {} # For models needing separate processors

_default_device = "cuda" if _torch_available and torch.cuda.is_available() else "cpu"
print(f"Using default device: {_default_device}")

# =============================================================================
# Dummy Functions (Fallbacks / Testing)
# =============================================================================

def dummy_text_embedding(texts: list[str]) -> np.ndarray:
    """Dummy text embedding function. Generates random 768-dim vectors."""
    DUMMY_DIM = 768
    DUMMY_SEED = 42
    print(f"Warning: Using DUMMY text embedding function ({DUMMY_DIM}-dim).")
    if not texts: return np.empty((0, DUMMY_DIM))
    rng = np.random.default_rng(seed=DUMMY_SEED)
    embeddings = rng.random((len(texts), DUMMY_DIM)).astype(np.float32) # Use float32 often
    if random.random() < 0.01 and len(texts) > 0: embeddings[random.randint(0, len(texts)-1), random.randint(0, DUMMY_DIM-1)] = np.nan
    return embeddings

def dummy_llm(prompt: str) -> str:
    """
    Dummy LLM function for testing text summarization/description.
    Parses expected IDs and returns JSON. Includes simulated latency/errors.
    """
    DUMMY_LATENCY_BASE = 0.1
    DUMMY_LATENCY_VAR = 0.2
    DUMMY_MALFORMED_JSON_RATE = 0.03
    DUMMY_MISSING_ID_RATE = 0.05
    DUMMY_MARKDOWN_RATE = 0.3
    DUMMY_TITLE_TRUNC = 20

    print(f"Warning: Using DUMMY LLM function for text/numeric/image description.")
    time.sleep(DUMMY_LATENCY_BASE + random.random() * DUMMY_LATENCY_VAR) # Simulate latency

    cluster_ids = re.findall(r'(?:Cluster|Item) ID: (\S+)', prompt)
    response_dict = {}
    if random.random() < DUMMY_MALFORMED_JSON_RATE:
         print("DUMMY LLM: Simulating malformed JSON error.")
         return '{"item_abc: {"title": "Bad JSON", "description": "Oops"}...'
    if random.random() < DUMMY_MISSING_ID_RATE and cluster_ids:
         if len(cluster_ids) > 1:
             missing_id_idx = random.randrange(len(cluster_ids))
             missing_id = cluster_ids.pop(missing_id_idx)
             print(f"DUMMY LLM: Simulating missing description for ID {missing_id}.")

    for cluster_id_str in cluster_ids:
         clean_id_str = str(cluster_id_str).replace('\\', '/')
         title_trunc = f"{clean_id_str[:DUMMY_TITLE_TRUNC]}{'...' if len(clean_id_str)>DUMMY_TITLE_TRUNC else ''}"
         response_dict[clean_id_str] = {
             "title": f"Dummy Title {title_trunc}",
             "description": f"Dummy description for item/cluster {clean_id_str} generated by dummy LLM."
         }

    if not response_dict and cluster_ids:
         print("DUMMY LLM: Found IDs but simulating missing all due to simulated error.")
    elif not response_dict and not cluster_ids:
         print("DUMMY LLM: Could not find Cluster/Item IDs in prompt.")
         return json.dumps({"error_no_ids_found": {"title": "Error", "description": "Dummy LLM could not find IDs."}})

    formatted_output = json.dumps(response_dict, indent=2)
    if random.random() < DUMMY_MARKDOWN_RATE:
        formatted_output = f"```json\n{formatted_output}\n```"
    return formatted_output

def dummy_image_embedding(image_identifiers: list[any]) -> np.ndarray:
    """Dummy image embedding function. Generates random 512-dim vectors."""
    DUMMY_DIM = 512
    DUMMY_SEED = 43
    print(f"Warning: Using DUMMY image embedding function ({DUMMY_DIM}-dim) for {len(image_identifiers)} items.")
    if not image_identifiers: return np.empty((0, DUMMY_DIM))
    rng = np.random.default_rng(seed=DUMMY_SEED)
    embeddings = rng.random((len(image_identifiers), DUMMY_DIM)).astype(np.float32)
    if random.random() < 0.01 and len(image_identifiers) > 0: embeddings[random.randint(0, len(image_identifiers)-1), random.randint(0, DUMMY_DIM-1)] = np.inf
    return embeddings

def dummy_image_captioning(image_identifiers: list[any], prompt: str | None = None) -> list[str]:
    """Dummy image captioning function."""
    DUMMY_SLEEP_PER_ITEM = 0.05
    DUMMY_ID_TRUNC_HEAD = 3
    DUMMY_ID_TRUNC_TAIL = 47
    DUMMY_ID_MAX_LEN = DUMMY_ID_TRUNC_HEAD + DUMMY_ID_TRUNC_TAIL + 3 # ...
    print(f"Warning: Using DUMMY image captioning function for {len(image_identifiers)} items.")
    if prompt: print(f"  Captioning Prompt Hint: {prompt}")
    captions = []
    for i, identifier in enumerate(image_identifiers):
        id_repr = str(identifier)
        if len(id_repr) > DUMMY_ID_MAX_LEN:
             id_repr = f"{id_repr[:DUMMY_ID_TRUNC_HEAD]}...{id_repr[-DUMMY_ID_TRUNC_TAIL:]}"
        captions.append(f"Dummy caption for image '{id_repr}' (item {i}).")
    time.sleep(DUMMY_SLEEP_PER_ITEM * len(image_identifiers))
    if random.random() < 0.02 and len(captions) > 0: captions[random.randrange(len(captions))] = ""
    return captions


# =============================================================================
# Google Cloud Functions
# =============================================================================

_google_embedding_model_name = "models/embedding-001"
_google_embedding_dim = 768

def google_text_embedding_001(texts: list[str]) -> np.ndarray:
    """Embeds text using Google's embedding-001 model."""
    if not _google_genai_available:
        print("Error: Google GenAI SDK not available for google_text_embedding_001.")
        return dummy_text_embedding(texts)
    if not _google_configured and not configure_google_api():
        print("Error: Google API not configured for google_text_embedding_001.")
        return dummy_text_embedding(texts)

    if not texts:
        return np.empty((0, _google_embedding_dim))

    try:
        result = genai.embed_content(
            model=_google_embedding_model_name,
            content=texts,
            task_type="clustering"
        )
        embeddings = np.array(result["embedding"])
        if embeddings.ndim != 2 or embeddings.shape[0] != len(texts) or embeddings.shape[1] != _google_embedding_dim:
             print(f"Warning: Unexpected Google embedding shape. Expected ({len(texts)}, {_google_embedding_dim}), Got {embeddings.shape}.")
             return dummy_text_embedding(texts)
        return embeddings
    except Exception as e:
        print(f"Error calling Google Embedding API ('{_google_embedding_model_name}'): {e}")
        return dummy_text_embedding(texts)

_gemini_model_name = "gemini-2.0-flash" # User-specified model name

def google_gemini_llm(prompt: str) -> str:
    """Generates text using Google's Gemini model."""
    if not _google_genai_available:
        print("Error: Google GenAI SDK not available for google_gemini_llm.")
        return dummy_llm(prompt)
    if not _google_configured and not configure_google_api():
        print("Error: Google API not configured for google_gemini_llm.")
        return dummy_llm(prompt)

    model_key = _gemini_model_name
    if model_key not in _loaded_llm_models:
        try:
            print(f"Initializing Google Gemini model: {model_key}")
            _loaded_llm_models[model_key] = genai.GenerativeModel(model_key)
            print(f"Gemini model '{model_key}' initialized.")
        except Exception as e:
            print(f"Error initializing Gemini model '{model_key}': {e}")
            return dummy_llm(prompt)

    model = _loaded_llm_models[model_key]
    try:
        response = model.generate_content(prompt)
        if not response.candidates:
             block_reason = "Unknown"
             try: block_reason = str(response.prompt_feedback)
             except Exception: pass
             print(f"Warning: Gemini response blocked or empty. Reason: {block_reason}")
             return ""
        try:
            result_text = response.text
        except ValueError:
            print("Warning: Gemini response generated empty content (ValueError).")
            result_text = ""
        except AttributeError:
             print("Warning: Gemini response missing '.text' attribute. Checking parts.")
             result_text = ""
             if response.parts:
                 result_text = "".join(part.text for part in response.parts if hasattr(part, 'text'))
        return result_text
    except Exception as e:
        print(f"Error calling Gemini API ('{model_key}'): {e}")
        return ""


# =============================================================================
# Local Text Embedding (SentenceTransformers)
# =============================================================================
_local_embedding_model_name = "all-MiniLM-L6-v2"
_local_embedding_dim = 384

def local_minilm_l6_v2_embedding(texts: list[str]) -> np.ndarray:
    """Embeds text using local SentenceTransformer all-MiniLM-L6-v2."""
    if not _sentence_transformers_available:
        print("Error: SentenceTransformers library not available for local_minilm_l6_v2_embedding.")
        return dummy_text_embedding(texts)

    model_key = _local_embedding_model_name
    if model_key not in _loaded_embedding_models:
        try:
            print(f"Loading local embedding model: {model_key}...")
            model = SentenceTransformer(model_key)
            dim = model.get_sentence_embedding_dimension()
            if dim != _local_embedding_dim:
                print(f"Warning: Expected dim {_local_embedding_dim} but got {dim} for {model_key}")
            _loaded_embedding_models[model_key] = model
            print(f"Local embedding model '{model_key}' loaded (Dim: {dim}).")
        except Exception as e:
            print(f"Error loading SentenceTransformer model '{model_key}': {e}")
            return dummy_text_embedding(texts)

    model = _loaded_embedding_models[model_key]
    if not texts:
        return np.empty((0, _local_embedding_dim))

    try:
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(texts) or embeddings.shape[1] != _local_embedding_dim:
             print(f"Warning: Unexpected local embedding shape. Expected ({len(texts)}, {_local_embedding_dim}), Got {embeddings.shape}.")
             return dummy_text_embedding(texts)
        return embeddings
    except Exception as e:
        print(f"Error during local embedding generation ('{model_key}'): {e}")
        return dummy_text_embedding(texts)

# =============================================================================
# Local LLM (Hugging Face Transformers - Gemma)
# =============================================================================
_local_gemma_model_id = "google/gemma-3-4b-it" # User-specified model ID

def local_gemma_3_4b_it_llm(prompt: str) -> str:
    """Generates text using local Gemma 3 4B Instruct model."""
    global _huggingface_login_attempted # Ensure we can modify this if needed (though try_huggingface_login handles it)

    if not _transformers_available or not _torch_available:
        print("Error: Transformers/Torch library not available for local_gemma_3_4b_it_llm.")
        return dummy_llm(prompt)

    # Attempt Hugging Face login once if HUGGINGFACE_HUB_TOKEN is set
    if not _huggingface_login_attempted:
        try_huggingface_login()

    model_key = _local_gemma_model_id
    processor_key = model_key

    if model_key not in _loaded_llm_models or processor_key not in _loaded_processors:
        try:
            print(f"Loading local LLM model/processor: {model_key}...")
            if _default_device == "cuda":
                dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
            else: # cpu
                dtype = torch.float32 # float16 not well supported for CPU compute
            
            model = Gemma3ForConditionalGeneration.from_pretrained(
                model_key, device_map=_default_device, torch_dtype=dtype
            ).eval()
            processor = AutoProcessor.from_pretrained(processor_key)
            _loaded_llm_models[model_key] = model
            _loaded_processors[processor_key] = processor
            print(f"Local LLM '{model_key}' loaded to {_default_device} with dtype {dtype}.")
        except Exception as e:
            print(f"Error loading local LLM model/processor '{model_key}': {e}")
            print("This might require significant RAM/VRAM, 'accelerate' library,")
            print("and for gated models like Gemma, a valid HUGGINGFACE_HUB_TOKEN or CLI login.")
            return dummy_llm(prompt)

    model = _loaded_llm_models[model_key]
    processor = _loaded_processors[processor_key]

    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant providing concise summaries."}]},
        {"role": "user", "content": [{"type": "text", "text": prompt}]}
    ]

    try:
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(model.device)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = model.generate(
                **inputs,
                max_new_tokens=16384,
                do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id
                )
            generation = generation[0][input_len:]

        decoded = processor.decode(generation, skip_special_tokens=True)
        return decoded.strip()
    except Exception as e:
        print(f"Error during local Gemma ('{model_key}') generation: {e}")
        if "CUDA out of memory" in str(e):
            print("CUDA Out of Memory Error. Try a smaller model or reduce batch size/input length if applicable.")
        return ""


# =============================================================================
# Local Image Embedding (SentenceTransformers - CLIP)
# =============================================================================
_clip_model_name = "clip-ViT-B-32"
_clip_embedding_dim = 512

def local_clip_vit_b_32_embedding(image_identifiers: List[Any]) -> np.ndarray:
    """Embeds images using local SentenceTransformer CLIP ViT-B/32."""
    if not _sentence_transformers_available or not _pillow_available:
        print("Error: SentenceTransformers/Pillow library not available for local_clip_vit_b_32_embedding.")
        return dummy_image_embedding(image_identifiers)

    model_key = _clip_model_name
    if model_key not in _loaded_embedding_models:
        try:
            print(f"Loading local CLIP model: {model_key}...")
            model = SentenceTransformer(model_key)
            dim = model.get_sentence_embedding_dimension()
            if dim != _clip_embedding_dim:
                 print(f"Warning: Expected dim {_clip_embedding_dim} but got {dim} for {model_key}")
            _loaded_embedding_models[model_key] = model
            print(f"Local CLIP model '{model_key}' loaded (Dim: {dim}).")
        except Exception as e:
            print(f"Error loading SentenceTransformer CLIP model '{model_key}': {e}")
            return dummy_image_embedding(image_identifiers)

    model = _loaded_embedding_models[model_key]
    if not image_identifiers:
        return np.empty((0, _clip_embedding_dim))

    images_to_process = []
    valid_indices = []
    for idx, identifier in enumerate(image_identifiers):
        try:
            if isinstance(identifier, str):
                if os.path.isfile(identifier):
                    img = Image.open(identifier).convert("RGB")
                    images_to_process.append(img)
                    valid_indices.append(idx)
                else:
                    print(f"Warning: Image file not found: '{identifier}'. Skipping.")
            elif isinstance(identifier, Image.Image):
                 images_to_process.append(identifier.convert("RGB"))
                 valid_indices.append(idx)
            else:
                 print(f"Warning: Skipping unsupported image identifier type: {type(identifier)}")
        except Exception as e:
            print(f"Warning: Error loading image '{identifier}': {e}. Skipping.")

    if not images_to_process:
        print("Warning: No valid images found to process for CLIP embedding.")
        return np.empty((len(image_identifiers), _clip_embedding_dim)) # Return correct shape with zeros if all fail or empty

    try:
        start_time = time.time()
        embeddings_subset = model.encode(images_to_process, batch_size=32, convert_to_numpy=True, show_progress_bar=False)
        end_time = time.time()

        final_embeddings = np.zeros((len(image_identifiers), _clip_embedding_dim), dtype=embeddings_subset.dtype if embeddings_subset.size > 0 else np.float32)
        for i, original_idx in enumerate(valid_indices):
            final_embeddings[original_idx] = embeddings_subset[i]
        return final_embeddings
    except Exception as e:
        print(f"Error during local CLIP ('{model_key}') embedding: {e}")
        return dummy_image_embedding(image_identifiers)


# =============================================================================
# Local Image Captioning (Hugging Face Transformers - BLIP)
# =============================================================================
_blip_model_name = "Salesforce/blip-image-captioning-large"

def local_blip_large_captioning(image_identifiers: List[Any], prompt: Optional[str] = None) -> List[Optional[str]]:
    """Captions images using local BLIP Large model."""
    if not _transformers_available or not _torch_available or not _pillow_available:
        print("Error: Transformers/Torch/Pillow not available for local_blip_large_captioning.")
        return dummy_image_captioning(image_identifiers, prompt)

    model_key = _blip_model_name
    processor_key = model_key

    if model_key not in _loaded_captioning_models or processor_key not in _loaded_processors:
        try:
            print(f"Loading local BLIP model/processor: {model_key}...")
            dtype = torch.float16 if _default_device == 'cuda' else torch.float32
            processor = BlipProcessor.from_pretrained(processor_key)
            model = BlipForConditionalGeneration.from_pretrained(model_key, torch_dtype=dtype).to(_default_device).eval()
            _loaded_captioning_models[model_key] = model
            _loaded_processors[processor_key] = processor
            print(f"Local BLIP '{model_key}' loaded to {_default_device} with dtype {dtype}.")
        except Exception as e:
            print(f"Error loading local BLIP model/processor '{model_key}': {e}")
            print("This might require significant RAM/VRAM.")
            return dummy_image_captioning(image_identifiers, prompt)

    model = _loaded_captioning_models[model_key]
    processor = _loaded_processors[processor_key]

    if not image_identifiers: return []

    images_to_process = []
    valid_indices = []

    start_time = time.time()
    for idx, identifier in enumerate(image_identifiers):
        try:
            if isinstance(identifier, str):
                if os.path.isfile(identifier):
                    img = Image.open(identifier).convert("RGB")
                    images_to_process.append(img)
                    valid_indices.append(idx)
                else:
                    print(f"Warning: Image file not found: '{identifier}'. Skipping.")
            elif isinstance(identifier, Image.Image):
                images_to_process.append(identifier.convert("RGB"))
                valid_indices.append(idx)
            else:
                 print(f"Warning: Skipping unsupported image identifier type: {type(identifier)}")
        except Exception as e:
            print(f"Warning: Error loading image '{identifier}': {e}. Skipping.")

    if not images_to_process:
        print("Warning: No valid images found to process for BLIP captioning.")
        return [None] * len(image_identifiers)

    try:
        model_dtype = model.dtype # Get actual model dtype after loading
        if prompt:
            inputs = processor(images=images_to_process, text=[prompt] * len(images_to_process), return_tensors="pt", padding=True).to(model.device)
            if inputs.pixel_values.dtype != model_dtype: # Ensure inputs match model dtype
                inputs.pixel_values = inputs.pixel_values.to(model_dtype)
        else:
            inputs = processor(images=images_to_process, return_tensors="pt", padding=True).to(model.device)
            if inputs.pixel_values.dtype != model_dtype:
                inputs.pixel_values = inputs.pixel_values.to(model_dtype)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=75, num_beams=4, early_stopping=True)

        generated_captions_subset = processor.batch_decode(outputs, skip_special_tokens=True)
        final_captions = [None] * len(image_identifiers)
        for i, caption in enumerate(generated_captions_subset):
             original_index = valid_indices[i]
             final_captions[original_index] = caption.strip()
        end_time = time.time()
        return final_captions
    except Exception as e:
        print(f"Error during local BLIP ('{model_key}') captioning: {e}")
        if "CUDA out of memory" in str(e):
             print("CUDA Out of Memory Error during BLIP captioning.")
        return dummy_image_captioning(image_identifiers, prompt)


# =============================================================================
# Function Registries (for Dash app)
# =============================================================================

AVAILABLE_TEXT_EMBEDDERS = {
    "Dummy": dummy_text_embedding,
}
if _sentence_transformers_available:
    AVAILABLE_TEXT_EMBEDDERS["Local MiniLM-L6-v2"] = local_minilm_l6_v2_embedding
if _google_genai_available:
    AVAILABLE_TEXT_EMBEDDERS["Google Embedding-001"] = google_text_embedding_001

AVAILABLE_LLMS = {
    "Dummy": dummy_llm,
}
if _transformers_available and _torch_available: # Gemma needs these
    AVAILABLE_LLMS["Local Gemma-3-4B-IT"] = local_gemma_3_4b_it_llm
if _google_genai_available:
    AVAILABLE_LLMS["Google Gemini-2.0-Flash"] = google_gemini_llm

AVAILABLE_IMAGE_EMBEDDERS = {
    "Dummy": dummy_image_embedding,
}
if _sentence_transformers_available and _pillow_available: # CLIP needs these
    AVAILABLE_IMAGE_EMBEDDERS["Local CLIP ViT-B/32"] = local_clip_vit_b_32_embedding

AVAILABLE_IMAGE_CAPTIONERS = {
    "Dummy": dummy_image_captioning,
}
if _transformers_available and _torch_available and _pillow_available: # BLIP needs these
    AVAILABLE_IMAGE_CAPTIONERS["Local BLIP Large"] = local_blip_large_captioning

def get_function_by_name(name: str, function_map: dict):
    """Safely retrieves a function from a map by its name."""
    func = function_map.get(name)
    if func is None:
        print(f"Warning: Function '{name}' not found in map. Returning dummy.")
        if "embed" in name.lower(): return dummy_text_embedding
        if "llm" in name.lower() or "gemini" in name.lower() or "gemma" in name.lower(): return dummy_llm
        if "caption" in name.lower() or "blip" in name.lower(): return dummy_image_captioning
        if "clip" in name.lower(): return dummy_image_embedding
        return None
    return func

if __name__ == "__main__":
    print("\n--- Hercules Function Availability ---")
    print("\nText Embedders:")
    for name in AVAILABLE_TEXT_EMBEDDERS: print(f"- {name}")
    print("\nLLMs:")
    for name in AVAILABLE_LLMS: print(f"- {name}")
    print("\nImage Embedders:")
    for name in AVAILABLE_IMAGE_EMBEDDERS: print(f"- {name}")
    print("\nImage Captioning:")
    for name in AVAILABLE_IMAGE_CAPTIONERS: print(f"- {name}")

    print("\n--- Configuration Notes ---")
    print("- Google functions require the GOOGLE_API_KEY environment variable.")
    print("- Local Hugging Face models (like Gemma) may require HUGGINGFACE_HUB_TOKEN environment variable")
    print("  or prior login via `huggingface-cli login` if they are gated.")
    print("- Local models may download on first use and require significant resources (RAM/VRAM).")
    print("- Ensure required libraries are installed (see warnings above if any).")
    print("  Recommended base install for local models: pip install transformers torch Pillow accelerate huggingface-hub sentence-transformers")

    # Example: Trigger Hugging Face login check if relevant models are available
    if "Local Gemma-3-4B-IT" in AVAILABLE_LLMS:
        print("\nAttempting to check/configure Hugging Face Hub access for Gemma...")
        try_huggingface_login()
    
    # Example: Trigger Google API configuration check
    if "Google Gemini-2.0-Flash" in AVAILABLE_LLMS or "Google Embedding-001" in AVAILABLE_TEXT_EMBEDDERS:
        print("\nAttempting to check/configure Google API access...")
        configure_google_api()
