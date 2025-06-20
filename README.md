# pyHercules

[![PyPI version](https://badge.fury.io/py/pyhercules.svg)](https://badge.fury.io/py/pyhercules)
[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**pyHercules** is a flexible Python framework for hierarchical clustering of text, numeric, or image data. The core algorithm, **Hercules**, uses recursive k-means and leverages Large Language Models (LLMs) for efficient and meaningful summarization of clusters at each level of the hierarchy. The project includes the core library (`pyhercules`), a set of "batteries-included" model functions, and a powerful Dash web application for interactive exploration.

### Key Features

-   **Hierarchical Clustering:** Automatically builds a tree of clusters from your data.
-   **Multi-Modal:** Natively handles text, numeric (NumPy, Pandas), and image data (file paths, URLs, PIL Images). (One modality at a time.)
-   **LLM-Powered Summarization:** Uses Large Language Models (LLMs) to generate human-readable titles and descriptions for each cluster.
-   **Flexible Representation:** Choose between `direct` mode (using original data embeddings) or `description` mode (using LLM-generated summary embeddings) for clustering at higher levels.
-   **Interactive Web App:** An included Dash application (`pyhercules_app.py`) allows for easy data upload, parameter configuration, and visualization of clustering results.
-   **Extensible:** The core library is dependency-light. Bring your own model functions or use the provided ones in `pyhercules_functions.py`.

### Project Structure

-   `pyhercules.py`: The core clustering library. Contains the `Hercules` and `Cluster` classes.
-   `pyhercules_functions.py`: A collection of ready-to-use functions for embedding, captioning, and LLM calls (using Hugging Face, Google Gemini, etc.).
-   `pyhercules_app.py`: A comprehensive Dash web application for interactive clustering and visualization.
-   `examples.ipynb`: A Jupyter Notebook demonstrating various use cases of the library.
-   `requirements-*.txt`: Dependency files for different use cases (for reference).
-   `setup.py`: The packaging configuration script.

### Installation

You can install `pyhercules` directly from PyPI. Several installation options are available depending on your needs.

**1. Core Library Only**

For using the `Hercules` class with your own model client functions. This is a minimal, lightweight installation.

```bash
pip install pyhercules
```

**2. Library with Model Functions**

To use the pre-built functions in `pyhercules_functions.py` (e.g., for running the `examples.ipynb` notebook).

```bash
pip install "pyhercules[models]"
```

**3. Full Web Application**

To run the interactive Dash application, which includes all dependencies.

```bash
pip install "pyhercules[app]"
```

### Configuration: API Keys

To use models from Google or gated models from Hugging Face (like Gemma), you must configure your API keys. The recommended way is to create a `.env` file in your project's working directory:

```env
# .env
GOOGLE_API_KEY="your-google-api-key-here"
HUGGINGFACE_HUB_TOKEN="your-hugging-face-token-for-gated-models"
```

The library will automatically load these variables. Alternatively, you can set them as system environment variables.

### Usage

#### 1. Running the Dash Web Application (Recommended)

The easiest way to get started is with the interactive app.

1.  **Install dependencies:**
    ```bash
    pip install "pyhercules[app]"
    ```
2.  **Set API keys:** Create a `.env` file as described in the Configuration section.
3.  **Run the app:**
    ```bash
    pyhercules-app
    ```

Then, open your web browser to `http://127.0.0.1:8050`.

#### 2. Using the Core Library in Python

You can use the `Hercules` class directly in your scripts. See `examples.ipynb` for more detailed use cases.

```python
from pyhercules import Hercules
from pyhercules_functions import local_minilm_l6_v2_embedding, local_gemma_3_4b_it_llm

# 1. Sample data
sample_texts = [
    "Introduction to machine learning concepts.",
    "Advanced techniques in deep neural networks.",
    "A guide to Python programming for beginners.",
    "Web development using Flask and Jinja.",
    "Understanding gradient descent and backpropagation.",
]

# 2. Instantiate Hercules with your chosen model clients
# Ensure you have set up your HUGGINGFACE_HUB_TOKEN in a .env file for Gemma
hercules = Hercules(
    level_cluster_counts=[3, 2],  # Desired hierarchy: 3 top-level, then subdivide
    representation_mode="direct",
    text_embedding_client=local_minilm_l6_v2_embedding,
    llm_client=local_gemma_3_4b_it_llm,
    verbose=1
)

# 3. Run clustering
top_clusters = hercules.cluster(sample_texts, topic_seed="computer science topics")

# 4. Print results
if top_clusters:
    for cluster in top_clusters:
        cluster.print_hierarchy(indent_increment=2, print_level_0=False)
```

### License

This project is licensed under the MIT License. See the `LICENSE` file for details.
