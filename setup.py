import os
import re
from setuptools import setup

# Function to read the version from the main library file
def get_version():
    """Reads the version from pyhercules.py"""
    version_file = os.path.join(os.path.dirname(__file__), "pyhercules.py")
    with open(version_file, "r", encoding="utf-8") as f:
        version_match = re.search(r"^    HERCULES_VERSION = \"([^\"]*)\"", f.read(), re.M)
        if version_match:
            return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

# Read the long description from the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Define dependencies
core_deps = [
    "numpy",
    "pandas",
    "scikit-learn",
    "Pillow",
]

models_deps = [
    "accelerate",
    "google-generativeai",
    "huggingface_hub",
    "sentence-transformers",
    "torch",
    "transformers",
    "requests",
    "python-dotenv",
]

app_deps = [
    "dash",
    "dash-bootstrap-components",
    "plotly",
]

setup(
    name="pyhercules",
    version=get_version(),
    author="Bandee",
    author_email="bandeerun@gmail.com",
    description="A flexible framework for hierarchical clustering of text, numeric, or image data using LLMs.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bandeerun/pyhercules",
    py_modules=["pyhercules", "pyhercules_functions", "pyhercules_app"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires='>=3.9',
    install_requires=core_deps,
    extras_require={
        "models": models_deps,
        "app": models_deps + app_deps,
    },
    entry_points={
        "console_scripts": [
            "pyhercules-app=pyhercules_app:server.run",
        ],
    },
)
