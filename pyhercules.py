#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hercules.py

HERCULES (Hierarchical Embedding-based Recursive Clustering Using LLMs for Efficient Summarization)

A package for hierarchical k-means clustering of text, numeric, or image data using LLMs.
"""

from __future__ import annotations

# Standard Library Imports
import json
import random
import re
import os
import uuid
import warnings
import time
import inspect
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Union, Optional, Callable, Any, Tuple, TYPE_CHECKING
from urllib.parse import urlparse

# Third-party Library Imports
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_score,
    completeness_score,
    homogeneity_completeness_v_measure,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

# Optional PIL / Pillow Import
try:
    from PIL import Image as PILImage
    PIL_IMAGE_TYPE = PILImage.Image
except ImportError:
    PILImage = None
    PIL_IMAGE_TYPE = type(None) # Fallback type if PIL is not installed, ensures isinstance checks don't fail
    if __name__ == '__main__': # Print warning only if running as script
        print("Warning: Pillow (PIL) library not installed. Support for PIL.Image.Image objects will be disabled.")

# Define common image file extensions
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'}

# Threshold for data size (number of elements or characters) to be excluded in light_dict
LIGHT_DICT_DATA_SIZE_THRESHOLD = 1000


# =============================================================================
# Helper Functions
# =============================================================================

def _is_string_url(text: str) -> bool:
    """Checks if a string starts with http:// or https://."""
    if not isinstance(text, str):
        return False
    return text.startswith("http://") or text.startswith("https://")

def _estimate_token_count(text: str) -> int:
    """Estimate token count for a text string (simple approximation)."""
    if not text: return 0
    return max(1, len(text) // 3 + 10)

# --- Dummy Clients ---
def _dummy_text_embedding_function(texts: list[str]) -> np.ndarray:
    """Dummy text embedding function. Generates random 768-dim vectors."""
    DUMMY_DIM = 768
    DUMMY_SEED = 42
    print(f"Warning: Using DUMMY text embedding function ({DUMMY_DIM}-dim).")
    if not texts: return np.empty((0, DUMMY_DIM))
    rng = np.random.default_rng(seed=DUMMY_SEED)
    embeddings = rng.random((len(texts), DUMMY_DIM)).astype(np.float32)
    if random.random() < 0.01 and len(texts) > 0: embeddings[random.randint(0, len(texts)-1), random.randint(0, DUMMY_DIM-1)] = np.nan
    return embeddings

def _dummy_image_embedding_function(image_identifiers: list[Any]) -> np.ndarray:
    """Dummy image embedding function. Generates random 512-dim vectors."""
    DUMMY_DIM = 512
    DUMMY_SEED = 43
    print(f"Warning: Using DUMMY image embedding function ({DUMMY_DIM}-dim) for {len(image_identifiers)} items.")
    if image_identifiers:
        first_id = image_identifiers[0]
        if PIL_IMAGE_TYPE and isinstance(first_id, PIL_IMAGE_TYPE):
            print(f"  (Dummy received PIL.Image.Image objects, e.g., size {getattr(first_id, 'size', 'N/A')})")
        elif isinstance(first_id, str) and _is_string_url(first_id):
            print(f"  (Dummy received URLs, e.g., '{str(first_id)[:50]}...')")
        elif isinstance(first_id, str):
            print(f"  (Dummy received string identifiers/paths, e.g., '{str(first_id)[:50]}...')")
    if not image_identifiers: return np.empty((0, DUMMY_DIM))
    rng = np.random.default_rng(seed=DUMMY_SEED)
    embeddings = rng.random((len(image_identifiers), DUMMY_DIM)).astype(np.float32)
    if random.random() < 0.01 and len(image_identifiers) > 0: embeddings[random.randint(0, len(image_identifiers)-1), random.randint(0, DUMMY_DIM-1)] = np.inf
    return embeddings

def _dummy_image_captioning_function(image_identifiers: list[Any], prompt: Optional[str] = None) -> list[str]:
    """Dummy image captioning function."""
    DUMMY_SLEEP_PER_ITEM = 0.05
    DUMMY_ID_TRUNC_HEAD = 3
    DUMMY_ID_TRUNC_TAIL = 47
    DUMMY_ID_MAX_LEN = DUMMY_ID_TRUNC_HEAD + DUMMY_ID_TRUNC_TAIL + 3
    print(f"Warning: Using DUMMY image captioning function for {len(image_identifiers)} items.")
    if image_identifiers:
        first_id = image_identifiers[0]
        if PIL_IMAGE_TYPE and isinstance(first_id, PIL_IMAGE_TYPE):
            print(f"  (Dummy received PIL.Image.Image objects for captioning, e.g., size {getattr(first_id, 'size', 'N/A')})")
        elif isinstance(first_id, str) and _is_string_url(first_id):
            print(f"  (Dummy received URLs for captioning, e.g., '{str(first_id)[:50]}...')")
        elif isinstance(first_id, str):
            print(f"  (Dummy received string identifiers/paths for captioning, e.g., '{str(first_id)[:50]}...')")
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

def _dummy_llm_function(prompt: str) -> str:
    """Dummy LLM function for generating batch descriptions in JSON format. Guaranteed to return a string."""
    DUMMY_LATENCY_BASE = 0.1
    DUMMY_LATENCY_VAR = 0.2
    DUMMY_MALFORMED_JSON_RATE = 0.03
    DUMMY_MISSING_ID_RATE = 0.05
    DUMMY_MARKDOWN_RATE = 0.3
    DUMMY_TITLE_TRUNC = 20

    print(f"Warning: Using DUMMY LLM function for text/numeric/image description.")
    time.sleep(DUMMY_LATENCY_BASE + random.random() * DUMMY_LATENCY_VAR)

    # Simulate malformed JSON first, as it's a direct return
    if random.random() < DUMMY_MALFORMED_JSON_RATE:
         print("DUMMY LLM: Simulating malformed JSON error.")
         return '{"item_abc: {"title": "Bad JSON", "description": "Oops"}...'

    cluster_ids = []
    if prompt and isinstance(prompt, str): # Ensure prompt is a valid string
        cluster_ids = re.findall(r'(?:Cluster|Item) ID: (\S+)', prompt)
    
    response_dict = {}

    if not cluster_ids:
        print("DUMMY LLM: Could not find Cluster/Item IDs in prompt or prompt was invalid.")
        response_dict["error_no_ids_found_in_prompt"] = {
            "title": "Error: No IDs",
            "description": "Dummy LLM could not find any Cluster/Item IDs in the provided prompt."
        }
    else:
        temp_cluster_ids = list(cluster_ids) # Work with a copy for modification
        if random.random() < DUMMY_MISSING_ID_RATE and len(temp_cluster_ids) > 1:
            missing_id_idx = random.randrange(len(temp_cluster_ids))
            missing_id = temp_cluster_ids.pop(missing_id_idx) # Remove from the temp list
            print(f"DUMMY LLM: Simulating missing description for ID {missing_id}.")

        # Simulate case where all descriptions might be "lost"
        if random.random() < 0.02 and temp_cluster_ids: # Small chance to lose all if some were found
            print("DUMMY LLM: Simulating losing all descriptions for found IDs due to a simulated internal error.")
            temp_cluster_ids = [] # Empty the list that will be iterated

        for cluster_id_str in temp_cluster_ids: # Iterate over potentially modified list
            clean_id_str = str(cluster_id_str).replace('\\', '/')
            title_trunc = f"{clean_id_str[:DUMMY_TITLE_TRUNC]}{'...' if len(clean_id_str)>DUMMY_TITLE_TRUNC else ''}"
            response_dict[clean_id_str] = {
                "title": f"Dummy Title {title_trunc}",
                "description": f"Dummy description for item/cluster {clean_id_str} generated by dummy LLM."
            }

    formatted_output = json.dumps(response_dict, indent=2)
    if random.random() < DUMMY_MARKDOWN_RATE and response_dict and "error_no_ids_found_in_prompt" not in response_dict : # Don't add markdown to error json
        formatted_output = f"```json\n{formatted_output}\n```"
        
    return formatted_output

def _parse_llm_response(llm_output_str: str, expected_ids: list[Union[int, str]]) -> dict | None:
    """
    Parses the LLM JSON output for batch descriptions.
    Handles string or int IDs and potential markdown fences.
    Returns dict {id: (title, description)} or None on failure.
    """
    if not llm_output_str:
        print("Error: Received empty LLM output string.")
        return None

    try:
        cleaned_output = re.sub(r"```json\n?(.*?)\n?```", r"\1", llm_output_str, flags=re.DOTALL | re.IGNORECASE).strip()
        if not cleaned_output:
            print("Error: LLM output empty after cleaning markdown.")
            return None

        data = json.loads(cleaned_output)
        if not isinstance(data, dict):
            print(f"Error: LLM output is not a JSON dictionary. Got: {type(data)}. Start: {cleaned_output[:100]}...")
            return None

        results = {}
        expected_keys_map = {str(cid).replace('\\', '/'): cid for cid in expected_ids}
        actual_keys_str = set(data.keys())

        found_keys_str = set(expected_keys_map.keys()).intersection(actual_keys_str)
        missing_keys_str = set(expected_keys_map.keys()) - actual_keys_str
        extra_keys_str = actual_keys_str - set(expected_keys_map.keys())

        if missing_keys_str:
            warnings.warn(f"LLM response missing expected IDs (keys): {missing_keys_str}. Corresponding original IDs: {[expected_keys_map[k] for k in missing_keys_str]}")
        if extra_keys_str:
            warnings.warn(f"LLM response contained unexpected keys (ignored): {extra_keys_str}")

        validation_passed_count = 0
        for key_str in found_keys_str:
            cluster_info = data.get(key_str)
            if cluster_info is None:
                warnings.warn(f"Value for key '{key_str}' is null in LLM response. Skipping this key.")
                continue

            original_id = expected_keys_map[key_str]

            if not isinstance(cluster_info, dict):
                warnings.warn(f"Value for key '{key_str}' (ID: {original_id}) is not a dictionary. Skipping.")
                continue
            title = cluster_info.get("title")
            description = cluster_info.get("description")
            if not (isinstance(title, str) and title.strip() and
                    isinstance(description, str) and description.strip()):
                warnings.warn(f"Missing/empty 'title' or 'description' for key '{key_str}' (ID: {original_id}). Skipping.")
                continue

            results[original_id] = (title.strip(), description.strip())
            validation_passed_count += 1

        if not results and expected_ids:
             print("Error: LLM response parsing yielded no valid results for any expected ID.")
             print(f"LLM Output causing failure (start): {cleaned_output[:200]}...")
             return None

        return results

    except json.JSONDecodeError as e:
        print(f"Error: Failed to decode LLM output as JSON: {e}")
        print(f"LLM Output causing JSON error (start): {llm_output_str[:200]}...")
        return None
    except Exception as e:
        print(f"Error: Unexpected error during LLM response parsing: {e}")
        print(f"LLM Output causing unexpected error (start): {llm_output_str[:200]}...")
        return None

# =============================================================================
# Cluster Data Structure
# =============================================================================

class Cluster:
    """Represents a node in the hierarchical clustering result."""
    _next_internal_id = 0

    def __init__(self, level: int,
                 original_data_type: str,
                 children: list[Cluster] | None = None,
                 title: str | None = None,
                 description: str | None = None,
                 original_id: str | int | None = None,
                 raw_item_data: Any | None = None
                 ):
        """
        Initializes a Cluster node.

        Args:
            level: The depth level in the hierarchy (0 = leaf/original item).
            original_data_type: The type of the underlying leaf data ('text', 'numeric', 'image').
            children: List of child Cluster objects (for non-leaf nodes).
            title: Cluster title.
            description: Cluster description.
            original_id: The ID of the original data point (used for level 0 nodes).
            raw_item_data: The original input item data for L0 nodes.
        """
        self.level = level
        self.id = Cluster._get_next_id()
        self.original_data_type = original_data_type
        self.children = children if children is not None else []
        self.parent: Cluster | None = None
        self.title = title
        self.description = description
        self.original_id = original_id
        self._raw_item_data = raw_item_data

        self.representation_vector: np.ndarray | None = None
        self.representation_vector_space: str | None = None
        self.description_embedding: np.ndarray | None = None
        self.representation_vector_reductions: dict[str, np.ndarray] = {}
        self.description_embedding_reductions: dict[str, np.ndarray] = {}
        self.original_numeric_data: np.ndarray | None = None

        if self.children:
            self._aggregate_numeric_data_from_children()

    @classmethod
    def _get_next_id(cls) -> int:
        """Generates unique sequential internal IDs."""
        current_id = cls._next_internal_id
        cls._next_internal_id += 1
        return current_id

    @classmethod
    def reset_id_counter(cls):
        """Resets the internal ID counter."""
        cls._next_internal_id = 0

    @property
    def num_items(self) -> int:
        """Recursively calculates the number of level 0 descendants."""
        if self.level == 0:
            return 1
        return sum(child.num_items for child in self.children)

    def _aggregate_numeric_data_from_children(self):
        """Aggregates original numeric data from children's L0 descendants."""
        if not self.children or self.level == 0 or self.original_data_type != 'numeric':
            return

        all_original_data = []
        for child in self.children:
             child_data = child.original_numeric_data
             if child_data is not None:
                 all_original_data.append(child_data if child_data.ndim == 2 else child_data.reshape(1, -1))

        if all_original_data:
            try:
                self.original_numeric_data = np.vstack(all_original_data)
            except ValueError as e:
                 warnings.warn(f"Cluster {self.id}: Error vstacking child numeric data: {e}. Data might be inconsistent.")
                 self.original_numeric_data = None
        else:
            self.original_numeric_data = None

    def get_level0_descendants(self) -> list['Cluster']:
        """Recursively collect all descendant clusters at level 0."""
        if self.level == 0:
            return [self]
        descendants = []
        for child in self.children:
            descendants.extend(child.get_level0_descendants())
        return descendants

    def get_representative_samples(self,
                                   sample_size: int = 3,
                                   strategy: str = "centroid_closest",
                                   variable_names: list[str] | None = None,
                                   numeric_repr_max_vals: int = 5,
                                   numeric_repr_precision: int = 2
                                   ) -> list[dict]:
        """
        Returns representative samples (data and IDs) from this cluster's L0 descendants.

        Args:
            sample_size: Max number of samples to return.
            strategy: How to select samples ('centroid_closest', 'random', or 'self').
            variable_names: Names of numeric features (for L0 display).
            numeric_repr_max_vals: Max values to show for numeric item display.
            numeric_repr_precision: Decimal places for numeric item display.

        Returns:
            List of dictionaries, each with {'id': original_id, 'data': formatted_string_or_raw}.
        """
        if self.level == 0 or strategy == 'self':
            data_repr = f"Item {self.original_id}"
            if self.original_data_type == "text":
                data_repr = self.description or self.title or str(self._raw_item_data or data_repr)
            elif self.original_data_type == "numeric":
                data_to_format = self._raw_item_data
                if data_to_format is not None and isinstance(data_to_format, np.ndarray):
                     values = data_to_format.flatten()
                     prefix = "Numeric Orig"
                     num_fmt = f"{{:.{numeric_repr_precision}f}}"
                     num_vars_to_show = min(len(values), numeric_repr_max_vals) if numeric_repr_max_vals > 0 else len(values)
                     vals_str_parts = []
                     for i in range(num_vars_to_show):
                          v_name = variable_names[i] if variable_names and i < len(variable_names) else f"v{i+1}"
                          vals_str_parts.append(f"{v_name}={num_fmt.format(values[i])}")
                     vals_str = ', '.join(vals_str_parts)
                     ellipsis = '...' if (numeric_repr_max_vals > 0 and len(values) > numeric_repr_max_vals) else ''
                     data_repr = f"{prefix} Item {self.original_id}: [{vals_str}{ellipsis}]"
                else:
                    data_repr = f"Numeric Item {self.original_id} (Data unavailable)"
            elif self.original_data_type == "image":
                 data_repr = self.description if self.description else f"Image Item {self.original_id} (Description Missing)"

            return [{"id": self.original_id, "data": data_repr}]

        l0_descendants = self.get_level0_descendants()
        if not l0_descendants: return []

        sampled_descendants: List[Cluster] = []
        n_descendants = len(l0_descendants)
        actual_sample_size = min(sample_size, n_descendants)

        if actual_sample_size <= 0: return []

        current_strategy = strategy
        if current_strategy == "random":
            sampled_descendants = random.sample(l0_descendants, actual_sample_size)

        elif current_strategy == "centroid_closest":
            if self.representation_vector is None or self.representation_vector_space is None:
                warnings.warn(f"Representation vector/space unavailable for L0 centroid sampling in Cluster {self.id}. Falling back to random.")
                current_strategy = "random"
            elif not np.all(np.isfinite(self.representation_vector)):
                 warnings.warn(f"Cluster {self.id} representation vector is non-finite. Falling back to random.")
                 current_strategy = "random"
            else:
                l0_vectors = []
                valid_l0_nodes = []
                for l0_node in l0_descendants:
                    vec = None
                    if l0_node.representation_vector_space == self.representation_vector_space and l0_node.representation_vector is not None:
                         vec = l0_node.representation_vector
                    elif self.representation_vector_space == 'embedding' and l0_node.representation_vector is not None:
                         # Allow comparing cluster centroid (direct or desc) to L0 direct embeddings if cluster space is 'embedding'
                         vec = l0_node.representation_vector

                    if vec is not None and np.all(np.isfinite(vec)):
                        l0_vectors.append(vec)
                        valid_l0_nodes.append(l0_node)

                if not valid_l0_nodes or len(valid_l0_nodes) < actual_sample_size:
                    warnings.warn(f"Not enough L0 descendants ({len(valid_l0_nodes)}) found with comparable vectors for centroid_closest sampling in cluster {self.id}. Falling back to random.")
                    current_strategy = "random"
                else:
                    l0_vectors_array = np.array(l0_vectors)
                    rep_vec = self.representation_vector.reshape(1, -1)

                    if l0_vectors_array.shape[1] != rep_vec.shape[1]:
                         warnings.warn(f"Dimension mismatch L0 vectors ({l0_vectors_array.shape[1]}) vs cluster representation vector ({rep_vec.shape[1]}) for cluster {self.id} in space '{self.representation_vector_space}'. Falling back to random.")
                         current_strategy = "random"
                    else:
                         try:
                              # Using Euclidean distance here as it's common for finding nearest neighbors.
                              distances = np.linalg.norm(l0_vectors_array - rep_vec, axis=1)
                              closest_indices = np.argsort(distances)[:actual_sample_size]
                              sampled_descendants = [valid_l0_nodes[i] for i in closest_indices]
                         except Exception as e:
                              warnings.warn(f"Error calculating distances for L0 centroid_closest sampling in cluster {self.id}: {e}. Falling back to random.")
                              current_strategy = "random"

            if current_strategy == "random": # Fallback if centroid failed
                sampled_descendants = random.sample(l0_descendants, actual_sample_size)

        results = []
        for desc in sampled_descendants:
            sample_info = desc.get_representative_samples(1, strategy='self',
                                                          variable_names=variable_names,
                                                          numeric_repr_max_vals=numeric_repr_max_vals,
                                                          numeric_repr_precision=numeric_repr_precision)
            if sample_info: results.append(sample_info[0])
        return results

    def _get_representative_child_samples(self,
                                          sample_size: int = 3,
                                          strategy: str = "random",
                                          ) -> list['Cluster']:
        """
        Returns representative immediate child Cluster objects.

        Args:
            sample_size: Max number of child clusters to return.
            strategy: How to select children ('centroid_closest', 'random').

        Returns:
            List of selected child Cluster objects.
        """
        if not self.children or self.level < 1:
            return []

        sampled_children: List[Cluster] = []
        n_children = len(self.children)
        actual_sample_size = min(sample_size, n_children)

        if actual_sample_size <= 0:
            return []

        current_strategy = strategy
        if current_strategy == "random":
            sampled_children = random.sample(self.children, actual_sample_size)

        elif current_strategy == "centroid_closest":
            parent_rep_vec = self.representation_vector
            parent_space = self.representation_vector_space

            if parent_rep_vec is None or parent_space is None:
                warnings.warn(f"Parent representation vector/space unavailable for child centroid sampling in Cluster {self.id}. Falling back to random.")
                current_strategy = "random"
            elif not np.all(np.isfinite(parent_rep_vec)):
                warnings.warn(f"Parent representation vector non-finite for child centroid sampling in Cluster {self.id}. Falling back to random.")
                current_strategy = "random"
            else:
                child_vectors = []
                valid_child_nodes = []
                for child in self.children:
                    vec = None
                    # Use the vector that corresponds to the parent's space for comparison
                    if child.representation_vector is not None and child.representation_vector_space == parent_space:
                        vec = child.representation_vector
                    # Special case: If parent is description-based, compare to child description embeddings
                    elif parent_space == 'embedding' and self.representation_mode == 'description' and child.description_embedding is not None:
                        vec = child.description_embedding

                    if vec is not None and np.all(np.isfinite(vec)):
                        child_vectors.append(vec)
                        valid_child_nodes.append(child)

                if not valid_child_nodes or len(valid_child_nodes) < actual_sample_size:
                    warnings.warn(f"Not enough children ({len(valid_child_nodes)}) found with comparable vectors for child centroid_closest sampling in cluster {self.id}. Falling back to random.")
                    current_strategy = "random"
                else:
                    child_vectors_array = np.array(child_vectors)
                    parent_rep_vec_2d = parent_rep_vec.reshape(1, -1)

                    if child_vectors_array.shape[1] != parent_rep_vec_2d.shape[1]:
                        warnings.warn(f"Dimension mismatch child vectors ({child_vectors_array.shape[1]}) vs parent representation vector ({parent_rep_vec_2d.shape[1]}) for cluster {self.id} in space '{parent_space}' (child sampling). Falling back to random.")
                        current_strategy = "random"
                    else:
                        try:
                            # Using Euclidean distance for centroid closeness
                            distances = np.linalg.norm(child_vectors_array - parent_rep_vec_2d, axis=1)
                            closest_indices = np.argsort(distances)[:actual_sample_size]
                            sampled_children = [valid_child_nodes[i] for i in closest_indices]
                        except Exception as e:
                            warnings.warn(f"Error calculating distances for child centroid_closest sampling in cluster {self.id}: {e}. Falling back to random.")
                            current_strategy = "random"

            if current_strategy == "random": # Fallback if centroid failed
                sampled_children = random.sample(self.children, actual_sample_size)

        return sampled_children

    def compute_numeric_statistics(self, variable_names: list[str],
                                   numeric_stats_precision: int = 2
                                   ) -> dict | None:
        """
        Computes statistics using aggregated *original, unscaled* numeric data.

        Args:
            variable_names: Names of the numeric features.
            numeric_stats_precision: Decimal places for storing stats (not used for computation).

        Returns:
            Dictionary of statistics per variable, or None.
        """
        if self.original_data_type != 'numeric': return None
        data_to_analyze = self.original_numeric_data if self.level > 0 else self._raw_item_data

        if data_to_analyze is None or data_to_analyze.shape[0] == 0:
            return None
        if data_to_analyze.ndim == 1: data_to_analyze = data_to_analyze.reshape(1, -1)

        if variable_names is None: variable_names = [f"var_{i}" for i in range(data_to_analyze.shape[1])]

        if len(variable_names) != data_to_analyze.shape[1]:
             warnings.warn(f"Var names ({len(variable_names)}) != data dim ({data_to_analyze.shape[1]}) for cluster {self.id}. Computing stats only for matching dimensions.")
             num_features = min(len(variable_names), data_to_analyze.shape[1])
             if num_features == 0: return None
             data = data_to_analyze[:, :num_features]
             var_names_to_use = variable_names[:num_features]
        else:
            data = data_to_analyze
            var_names_to_use = variable_names
            num_features = len(var_names_to_use)

        stats = {}
        for i in range(num_features):
             var_name = var_names_to_use[i]
             try:
                 var_data_clean = data[:, i]
                 var_data_clean = var_data_clean[~np.isnan(var_data_clean)]
                 if var_data_clean.size == 0: continue

                 is_single_item = (self.level == 0 or data.shape[0] == 1)

                 stats[var_name] = {
                     "mean": float(np.mean(var_data_clean)),
                     "median": float(np.median(var_data_clean)) if not is_single_item else float(np.mean(var_data_clean)),
                     "min": float(np.min(var_data_clean)) if not is_single_item else float(np.mean(var_data_clean)),
                     "max": float(np.max(var_data_clean)) if not is_single_item else float(np.mean(var_data_clean)),
                     "std": float(np.std(var_data_clean)) if not is_single_item else 0.0
                 }
             except Exception as e:
                  warnings.warn(f"Error computing stats for var '{var_name}' in cluster {self.id}: {e}")
                  continue
        return stats if stats else None

    def get_data_for_prompt(self,
                           l0_sample_size: int,
                           l0_sample_strategy: str,
                           l0_sample_numeric_repr_max_vals: int,
                           l0_sample_numeric_repr_precision: int,
                           include_immediate_children: bool,
                           immediate_child_sample_size: int,
                           immediate_child_sample_strategy: str,
                           variable_names: list[str] | None,
                           variable_metadata_by_name: Optional[Dict[str, Dict[str, Any]]],
                           numeric_stats_precision: int,
                           max_stats_vars: int,
                           child_sample_desc_trunc_len: int
                           ) -> dict:
        """
        Prepares data for generating LLM title/description prompt.

        Includes L0 descendant samples, optionally immediate child samples,
        and numeric statistics.

        Returns:
            Dictionary containing data formatted for the LLM prompt.
        """
        prompt_data = {
            "id": self.original_id if self.level == 0 else self.id,
            "internal_id": self.id,
            "level": self.level,
            "num_items": self.num_items,
            "original_data_type": self.original_data_type,
            "samples": [],
            "immediate_child_samples": [],
            "statistics": None,
            "variable_metadata_by_name": variable_metadata_by_name,
        }

        actual_l0_sample_strategy = l0_sample_strategy
        if l0_sample_strategy == "centroid_closest" and self.representation_vector is None:
             actual_l0_sample_strategy = "random"

        prompt_data["samples"] = self.get_representative_samples(
            sample_size=l0_sample_size,
            strategy=actual_l0_sample_strategy,
            variable_names=variable_names,
            numeric_repr_max_vals=l0_sample_numeric_repr_max_vals,
            numeric_repr_precision=l0_sample_numeric_repr_precision
        )

        if self.level >= 2 and include_immediate_children and self.children:
            actual_child_sample_strategy = immediate_child_sample_strategy
            if immediate_child_sample_strategy == "centroid_closest" and self.representation_vector is None:
                 actual_child_sample_strategy = "random"

            sampled_children = self._get_representative_child_samples(
                sample_size=immediate_child_sample_size,
                strategy=actual_child_sample_strategy
            )
            child_samples_list = []
            for child in sampled_children:
                 desc = child.description or ""
                 trunc_desc = (desc[:child_sample_desc_trunc_len] + "...") if len(desc) > child_sample_desc_trunc_len else desc
                 child_samples_list.append({
                     "id": child.id,
                     "title": child.title or f"Child Cluster {child.id}",
                     "description": trunc_desc
                 })
            prompt_data["immediate_child_samples"] = child_samples_list

        if self.original_data_type == 'numeric' and variable_names:
             prompt_data["statistics"] = self.compute_numeric_statistics(
                 variable_names,
                 numeric_stats_precision=numeric_stats_precision
             )

        if self.level == 0 and self.original_data_type == 'image':
            prompt_data['image_identifier'] = str(self._raw_item_data or self.original_id)

        return prompt_data

    def print_hierarchy(self, indent: int = 0, max_depth: int | None = None, indent_increment: int = 2, print_level_0: bool = True):
        """
        Recursively prints the cluster hierarchy.

        Args:
            indent: Current indentation level (spaces).
            max_depth: Maximum depth level to print relative to this node.
            indent_increment: Number of spaces to add per level.
            print_level_0: Whether to print Level 0 nodes (leaf items). Defaults to True.
        """
        if indent_increment < 1: indent_increment = 1

        # Skip printing L0 nodes if print_level_0 is False
        if self.level == 0 and not print_level_0:
            return

        current_relative_depth = indent // indent_increment
        if max_depth is not None and current_relative_depth > max_depth :
            if self.children: print(f"{' ' * indent}- ... (Depth limit {max_depth} reached, children not shown)")
            return

        prefix = " " * indent
        title_str = self.title or f"Cluster {self.id}"
        desc_str = f": {self.description}" if self.description else ""
        meta = f"(L{self.level}, ID:{self.id}, Items:{self.num_items}, Type:{self.original_data_type})"
        if self.level == 0:
            meta = f"(L0, OrigID: {self.original_id}, Type:{self.original_data_type})"
        else:
             if self.representation_vector_space: meta += f" RepVecSpace:{self.representation_vector_space}"
             if self.description_embedding is not None: meta+= " HasDescEmb"

        print(f"{prefix}- {title_str} {meta}{desc_str}")

        if max_depth is None or current_relative_depth < max_depth:
            sorted_children = sorted(self.children, key=lambda c: c.num_items, reverse=True)
            for child in sorted_children:
                # Pass print_level_0 to recursive calls
                child.print_hierarchy(indent + indent_increment, max_depth, indent_increment, print_level_0)

    def to_dict(self) -> dict:
        """Convert Cluster object to a dictionary for JSON serialization."""
        def np_to_list(arr): return arr.tolist() if isinstance(arr, np.ndarray) else None
        def reductions_to_list(rd): return {k: np_to_list(v) for k, v in rd.items()} if rd else {}

        raw_data_serializable = None
        raw_data = self._raw_item_data
        if isinstance(raw_data, np.ndarray):
            raw_data_serializable = np_to_list(raw_data)
        elif isinstance(raw_data, (list, tuple)):
            try: json.dumps(raw_data); raw_data_serializable = raw_data
            except TypeError: raw_data_serializable = str(raw_data)
        elif isinstance(raw_data, bytes):
            try: raw_data_serializable = raw_data.decode('utf-8', errors='replace')
            except Exception: raw_data_serializable = "[Undecodable Bytes Data]"
        elif raw_data is not None:
            try: json.dumps(raw_data); raw_data_serializable = raw_data
            except TypeError: raw_data_serializable = str(raw_data)
            except Exception: raw_data_serializable = "[Unserializable Raw Data]"

        return {
            "level": self.level,
            "id": self.id,
            "original_id": self.original_id,
            "original_data_type": self.original_data_type,
            "children_ids": [child.id for child in self.children],
            "parent_id": self.parent.id if self.parent else None,
            "title": self.title,
            "description": self.description,
            "representation_vector": np_to_list(self.representation_vector),
            "representation_vector_space": self.representation_vector_space,
            "description_embedding": np_to_list(self.description_embedding),
            "representation_vector_reductions": reductions_to_list(self.representation_vector_reductions),
            "description_embedding_reductions": reductions_to_list(self.description_embedding_reductions),
            "original_numeric_data": np_to_list(self.original_numeric_data),
            "_raw_item_data": raw_data_serializable,
            "_raw_item_data_type": type(self._raw_item_data).__name__,
        }

    def to_light_dict(self) -> dict:
        """
        Convert Cluster object to a lightweight dictionary, excluding large data fields.

        Excludes large vectors and optionally large raw/numeric data based on size threshold.
        """
        def np_to_list(arr): return arr.tolist() if isinstance(arr, np.ndarray) else None
        def reductions_to_list(rd): return {k: np_to_list(v) for k, v in rd.items()} if rd else {}

        light_dict = {
            "level": self.level,
            "id": self.id,
            "original_id": self.original_id,
            "original_data_type": self.original_data_type,
            "num_items": self.num_items,
            "children_ids": [child.id for child in self.children],
            "parent_id": self.parent.id if self.parent else None,
            "title": self.title,
            "description": self.description,
            "representation_vector_reductions": reductions_to_list(self.representation_vector_reductions),
            "description_embedding_reductions": reductions_to_list(self.description_embedding_reductions),
            "representation_vector_space": self.representation_vector_space,
            "_raw_item_data_type": type(self._raw_item_data).__name__,
            # Add flag indicating if desc embedding exists
            "has_description_embedding": self.description_embedding is not None,
        }

        numeric_data = self.original_numeric_data
        if numeric_data is not None:
             try:
                 if numeric_data.size <= LIGHT_DICT_DATA_SIZE_THRESHOLD:
                     light_dict["original_numeric_data"] = np_to_list(numeric_data)
             except AttributeError: pass

        raw_data = self._raw_item_data
        include_raw_data = False
        if raw_data is not None:
            if isinstance(raw_data, np.ndarray):
                try:
                     if raw_data.size <= LIGHT_DICT_DATA_SIZE_THRESHOLD: include_raw_data = True
                except AttributeError: pass
            elif isinstance(raw_data, (str, list, tuple, bytes)):
                try:
                    if len(raw_data) <= LIGHT_DICT_DATA_SIZE_THRESHOLD: include_raw_data = True
                except TypeError: pass
            else: include_raw_data = True

        if include_raw_data:
            raw_data_serializable = None
            if isinstance(raw_data, np.ndarray): raw_data_serializable = np_to_list(raw_data)
            elif isinstance(raw_data, (list, tuple)):
                try: json.dumps(raw_data); raw_data_serializable = raw_data
                except TypeError: raw_data_serializable = str(raw_data)
            elif isinstance(raw_data, bytes):
                 try: raw_data_serializable = raw_data.decode('utf-8', errors='replace')
                 except Exception: raw_data_serializable = "[Undecodable Bytes Data]"
            elif raw_data is not None:
                 try: json.dumps(raw_data); raw_data_serializable = raw_data
                 except TypeError: raw_data_serializable = str(raw_data)
                 except Exception: raw_data_serializable = "[Unserializable Raw Data]"
            if raw_data_serializable is not None: light_dict["_raw_item_data"] = raw_data_serializable

        return light_dict

    @classmethod
    def from_dict(cls, data: dict) -> 'Cluster':
        """Convert dict back to Cluster object (without hierarchy links initially)."""
        def list_to_np(lst): return np.array(lst) if lst is not None else None
        def reductions_from_list(rd): return {k: list_to_np(v) for k, v in rd.items()} if rd else {}

        internal_id = data.get("id")
        if internal_id is None:
             warnings.warn("Cluster data missing 'id', a new internal ID will be assigned.")
             internal_id = cls._get_next_id()
        else:
             # Make sure the next ID starts after the highest loaded ID
             cls._next_internal_id = max(cls._next_internal_id, internal_id + 1)

        raw_data = data.get("_raw_item_data")
        raw_data_type_str = data.get("_raw_item_data_type")
        if raw_data is not None:
             if raw_data_type_str == "ndarray":
                 try: raw_data = list_to_np(raw_data)
                 except Exception as e: warnings.warn(f"Failed to convert raw data to ndarray for ID {internal_id}: {e}")
             elif raw_data_type_str == "bytes":
                 try: raw_data = raw_data.encode('utf-8')
                 except Exception as e: warnings.warn(f"Failed to convert raw data to bytes for ID {internal_id}: {e}")

        cluster = cls(
            level=data["level"],
            original_data_type=data["original_data_type"],
            title=data.get("title"),
            description=data.get("description"),
            original_id=data.get("original_id"),
            raw_item_data=raw_data
        )
        cluster.id = internal_id # Assign the loaded ID

        cluster.representation_vector = list_to_np(data.get("representation_vector"))
        cluster.representation_vector_space = data.get("representation_vector_space")
        cluster.description_embedding = list_to_np(data.get("description_embedding"))
        cluster.representation_vector_reductions = reductions_from_list(data.get("representation_vector_reductions"))
        cluster.description_embedding_reductions = reductions_from_list(data.get("description_embedding_reductions"))
        cluster.original_numeric_data = list_to_np(data.get("original_numeric_data"))

        # Store linkage info temporarily
        cluster._saved_children_ids = data.get("children_ids", [])
        cluster._saved_parent_id = data.get("parent_id")
        return cluster

    @staticmethod
    def link_hierarchy(clusters_by_id: dict[int, 'Cluster']):
        """Links parent/child references after loading all clusters from dicts."""
        for cluster_id, cluster in clusters_by_id.items():
            child_ids = getattr(cluster, '_saved_children_ids', [])
            parent_id = getattr(cluster, '_saved_parent_id', None)

            valid_child_ids = [cid for cid in child_ids if cid in clusters_by_id]
            if len(valid_child_ids) != len(child_ids):
                 warnings.warn(f"Cluster {cluster_id}: Found missing child IDs during linking.")
            cluster.children = [clusters_by_id[cid] for cid in valid_child_ids]

            cluster.parent = clusters_by_id.get(parent_id) if parent_id is not None else None
            if parent_id is not None and cluster.parent is None:
                 warnings.warn(f"Cluster {cluster_id}: Could not find parent with ID {parent_id} during linking.")

            # Clean up temporary attributes
            if hasattr(cluster, '_saved_children_ids'): del cluster._saved_children_ids
            if hasattr(cluster, '_saved_parent_id'): del cluster._saved_parent_id

# =============================================================================
# Hercules Class
# =============================================================================

class Hercules:
    """
    Hercules Hierarchical Clustering.

    Handles text, numeric (with optional metadata), and image data with
    configurable parameters and evaluation. Uses k-means hierarchically with
    LLM-generated cluster descriptions. Offers 'direct' or 'description' based
    representation modes. Supports fixed or automatic cluster counts (k) per level.
    Includes optional sampling of immediate children for L2+ LLM prompts.
    """
    HERCULES_VERSION = "1.0.2"
    DEFAULT_REPRESENTATION_MODE = "direct"
    DEFAULT_MIN_CLUSTERS_PER_LEVEL = 2
    DEFAULT_FALLBACK_K = 2
    DEFAULT_AUTO_K_METHOD = "silhouette"
    DEFAULT_AUTO_K_MAX = 15
    DEFAULT_AUTO_K_METRIC_PARAMS = {}
    DEFAULT_N_REDUCTION_COMPONENTS = 2
    DEFAULT_REDUCTION_METHODS = ["pca"]
    DEFAULT_MAX_PROMPT_TOKENS = 1048576
    DEFAULT_MAX_PROMPT_TOKEN_BUFFER = 1.2
    DEFAULT_LLM_INITIAL_BATCH_SIZE = 32
    DEFAULT_LLM_BATCH_REDUCTION_FACTOR = 2.0
    DEFAULT_LLM_MIN_BATCH_SIZE = 1
    DEFAULT_LLM_RETRIES = 2
    DEFAULT_LLM_RETRY_DELAY = 1.0
    DEFAULT_PROMPT_L0_SAMPLE_SIZE = 5
    DEFAULT_PROMPT_L0_SAMPLE_STRATEGY = "centroid_closest"
    DEFAULT_PROMPT_L0_SAMPLE_TRUNC_LEN = 100
    DEFAULT_PROMPT_L0_NUMERIC_REPR_MAX_VALS = 5
    DEFAULT_PROMPT_L0_NUMERIC_REPR_PRECISION = 2
    DEFAULT_PROMPT_INCLUDE_IMMEDIATE_CHILDREN = True
    DEFAULT_PROMPT_IMMEDIATE_CHILD_SAMPLE_STRATEGY = "random"
    DEFAULT_PROMPT_IMMEDIATE_CHILD_SAMPLE_SIZE = 3
    DEFAULT_PROMPT_CHILD_SAMPLE_DESC_TRUNC_LEN = 75
    DEFAULT_PROMPT_MAX_STATS_VARS = 5
    DEFAULT_PROMPT_NUMERIC_STATS_PRECISION = 2
    DEFAULT_DIRECT_L0_TEXT_TRUNC_LEN = 150
    DEFAULT_CLUSTER_NUMERIC_STATS_PRECISION = 2
    DEFAULT_CLUSTER_PRINT_INDENT_INCREMENT = 2
    DEFAULT_RANDOM_STATE = 42
    DEFAULT_SAVE_RUN_DETAILS = True
    DEFAULT_SAVE_DIR = "hercules_run"
    DEFAULT_VERBOSE = 0
    DEFAULT_USE_LLM_FOR_L0_DESCRIPTIONS = False
    DEFAULT_USE_RESAMPLING = False
    DEFAULT_RESAMPLING_POINTS_PER_CLUSTER = 10
    DEFAULT_RESAMPLING_ITERATIONS = 10
    EMPTY_TEXT_PLACEHOLDER = "[EMPTY_CONTENT]"

    def __init__(self,
                 level_cluster_counts: Optional[list[int]],
                 text_embedding_client: Optional[Callable[[list[str]], np.ndarray]] = None,
                 llm_client: Optional[Callable[[str], str]] = None,
                 image_embedding_client: Optional[Callable[[list[Any]], np.ndarray]] = None,
                 image_captioning_client: Optional[Callable[[list[Any], Optional[str]], list[str]]] = None,
                 representation_mode: str = DEFAULT_REPRESENTATION_MODE,
                 auto_k_method: str = DEFAULT_AUTO_K_METHOD,
                 auto_k_max: int = DEFAULT_AUTO_K_MAX,
                 auto_k_metric_params: dict = DEFAULT_AUTO_K_METRIC_PARAMS,
                 n_reduction_components: int = DEFAULT_N_REDUCTION_COMPONENTS,
                 reduction_methods: list[str] | str = DEFAULT_REDUCTION_METHODS,
                 prompt_l0_sample_size: int = DEFAULT_PROMPT_L0_SAMPLE_SIZE,
                 prompt_l0_sample_strategy: str = DEFAULT_PROMPT_L0_SAMPLE_STRATEGY,
                 prompt_l0_sample_trunc_len: int = DEFAULT_PROMPT_L0_SAMPLE_TRUNC_LEN,
                 prompt_l0_numeric_repr_max_vals: int = DEFAULT_PROMPT_L0_NUMERIC_REPR_MAX_VALS,
                 prompt_l0_numeric_repr_precision: int = DEFAULT_PROMPT_L0_NUMERIC_REPR_PRECISION,
                 prompt_include_immediate_children: bool = DEFAULT_PROMPT_INCLUDE_IMMEDIATE_CHILDREN,
                 prompt_immediate_child_sample_strategy: str = DEFAULT_PROMPT_IMMEDIATE_CHILD_SAMPLE_STRATEGY,
                 prompt_immediate_child_sample_size: int = DEFAULT_PROMPT_IMMEDIATE_CHILD_SAMPLE_SIZE,
                 prompt_child_sample_desc_trunc_len: int = DEFAULT_PROMPT_CHILD_SAMPLE_DESC_TRUNC_LEN,
                 prompt_max_stats_vars: int = DEFAULT_PROMPT_MAX_STATS_VARS,
                 prompt_numeric_stats_precision: int = DEFAULT_PROMPT_NUMERIC_STATS_PRECISION,
                 max_prompt_tokens: int = DEFAULT_MAX_PROMPT_TOKENS,
                 max_prompt_token_buffer: float = DEFAULT_MAX_PROMPT_TOKEN_BUFFER,
                 llm_initial_batch_size: int = DEFAULT_LLM_INITIAL_BATCH_SIZE,
                 llm_batch_reduction_factor: float = DEFAULT_LLM_BATCH_REDUCTION_FACTOR,
                 llm_min_batch_size: int = DEFAULT_LLM_MIN_BATCH_SIZE,
                 llm_retries: int = DEFAULT_LLM_RETRIES,
                 llm_retry_delay: float = DEFAULT_LLM_RETRY_DELAY,
                 random_state: int | None = DEFAULT_RANDOM_STATE,
                 save_run_details: bool = DEFAULT_SAVE_RUN_DETAILS,
                 run_details_dir: str = DEFAULT_SAVE_DIR,
                 verbose: int = DEFAULT_VERBOSE,
                 direct_l0_text_trunc_len: int = DEFAULT_DIRECT_L0_TEXT_TRUNC_LEN,
                 min_clusters_per_level: int = DEFAULT_MIN_CLUSTERS_PER_LEVEL,
                 fallback_k: int = DEFAULT_FALLBACK_K,
                 cluster_numeric_stats_precision: int = DEFAULT_CLUSTER_NUMERIC_STATS_PRECISION,
                 cluster_print_indent_increment: int = DEFAULT_CLUSTER_PRINT_INDENT_INCREMENT,
                 use_llm_for_l0_descriptions: bool = DEFAULT_USE_LLM_FOR_L0_DESCRIPTIONS,
                 use_resampling: bool = DEFAULT_USE_RESAMPLING,
                 resampling_points_per_cluster: int = DEFAULT_RESAMPLING_POINTS_PER_CLUSTER,
                 resampling_iterations: int = DEFAULT_RESAMPLING_ITERATIONS
                 ):
        """
        Initializes the Hercules clusterer.

        Args:
            level_cluster_counts: List of desired clusters per level, or None for auto-k.
            text_embedding_client: Function for embedding text.
            llm_client: Function for LLM calls (description generation).
            image_embedding_client: Function for embedding images (optional).
            image_captioning_client: Function for captioning images (optional).
            representation_mode: 'direct' or 'description'.
            auto_k_method: Metric for auto-k ('silhouette', 'davies_bouldin', 'calinski_harabasz').
            auto_k_max: Max k to test per level for auto-k.
            auto_k_metric_params: Additional params for auto-k metric functions.
            n_reduction_components: Dimensions for PCA reduction.
            reduction_methods: List of reduction methods (currently only 'pca').
            prompt_l0_sample_size: Number of L0 descendant samples in prompts.
            prompt_l0_sample_strategy: Strategy for L0 samples ('centroid_closest', 'random').
            prompt_l0_sample_trunc_len: Max length for L0 text samples in prompts.
            prompt_l0_numeric_repr_max_vals: Max L0 numeric values shown in prompts.
            prompt_l0_numeric_repr_precision: Precision for L0 numeric values in prompts.
            prompt_include_immediate_children: Include immediate child summaries in L2+ prompts (bool).
            prompt_immediate_child_sample_strategy: Strategy for child samples ('centroid_closest', 'random').
            prompt_immediate_child_sample_size: Number of child samples to include.
            prompt_child_sample_desc_trunc_len: Max length for child descriptions in prompts.
            prompt_max_stats_vars: Max numeric variables shown in prompts.
            prompt_numeric_stats_precision: Precision for numeric stats in prompts.
            max_prompt_tokens: Estimated LLM context window limit.
            max_prompt_token_buffer: Safety multiplier for token estimation.
            llm_initial_batch_size: Starting batch size for LLM calls.
            llm_batch_reduction_factor: Factor to reduce batch size on failure (>=1.1).
            llm_min_batch_size: Minimum batch size for LLM calls.
            llm_retries: Number of times to retry a failing LLM call for a batch chunk.
            llm_retry_delay: Base delay in seconds between retries (increases linearly).
            random_state: Seed for reproducibility.
            save_run_details: Save prompt logs and run info.
            run_details_dir: Directory for saving run details.
            verbose: Controls the level of output (0=silent, 1=basic, 2=detailed).
            direct_l0_text_trunc_len: Truncation for L0 text descriptions (non-LLM).
            min_clusters_per_level: Minimum allowed clusters at any level (>=2).
            fallback_k: Number of clusters to use if config exhausted or auto-k fails.
            cluster_numeric_stats_precision: Precision for stored numeric stats.
            cluster_print_indent_increment: Spaces per level in print_hierarchy.
            use_llm_for_l0_descriptions: Use LLM for L0 text/numeric descriptions (default False).
            use_resampling: Whether to use iterative resampling-clustering.
            resampling_points_per_cluster: Number of points to sample from each cluster in resampling.
            resampling_iterations: Number of resampling iterations.
        """
        # Handle optional clients with fallbacks to dummies
        using_dummy_text_embed = False
        if text_embedding_client is None:
            self.text_embedding_client = _dummy_text_embedding_function
            using_dummy_text_embed = True
        elif not callable(text_embedding_client):
            raise TypeError("If provided, `text_embedding_client` must be callable.")
        else:
            self.text_embedding_client = text_embedding_client

        using_dummy_llm = False
        if llm_client is None:
            self.llm_client = _dummy_llm_function
            using_dummy_llm = True
        elif not callable(llm_client):
            raise TypeError("If provided, `llm_client` must be callable.")
        else:
            self.llm_client = llm_client

        using_dummy_image_embed = False
        if image_embedding_client is None:
            self.image_embedding_client = _dummy_image_embedding_function
            using_dummy_image_embed = True
        elif not callable(image_embedding_client):
             raise TypeError("If provided, `image_embedding_client` must be callable if provided.")
        else:
            self.image_embedding_client = image_embedding_client

        using_dummy_image_caption = False
        if image_captioning_client is None:
            self.image_captioning_client = _dummy_image_captioning_function
            using_dummy_image_caption = True
        elif not callable(image_captioning_client):
             raise TypeError("If provided, `image_captioning_client` must be callable if provided.")
        else:
            self.image_captioning_client = image_captioning_client

        self.verbose = max(0, int(verbose))

        # --- Explicit Notifications for Dummy Usage ---
        log_level_for_dummy_warning = 0
        if using_dummy_text_embed:
            self._log("INFO: `text_embedding_client` not provided. HERCULES will use a DUMMY text embedding function. Real embeddings are recommended for meaningful results.", level=log_level_for_dummy_warning)
        if using_dummy_llm:
            self._log("INFO: `llm_client` not provided. HERCULES will use a DUMMY LLM function for descriptions. Real LLM calls are recommended for meaningful summaries.", level=log_level_for_dummy_warning)
        if using_dummy_image_embed:
            self._log("INFO: `image_embedding_client` not provided. HERCULES will use a DUMMY image embedding function if image data is processed in 'direct' mode.", level=log_level_for_dummy_warning)
        if using_dummy_image_caption:
            self._log("INFO: `image_captioning_client` not provided. HERCULES will use a DUMMY image captioning function if image data descriptions are generated.", level=log_level_for_dummy_warning)

        self._function_metadata = {}
        self._function_metadata['text_embedding'] = self._get_function_metadata(text_embedding_client)
        self._function_metadata['llm'] = self._get_function_metadata(llm_client)
        self._function_metadata['image_embedding'] = self._get_function_metadata(self.image_embedding_client)
        self._function_metadata['image_captioning'] = self._get_function_metadata(self.image_captioning_client)

        if level_cluster_counts is not None:
            if not isinstance(level_cluster_counts, list) or not all(isinstance(k, int) and k > 0 for k in level_cluster_counts):
                raise ValueError("`level_cluster_counts`, if provided, must be a list of positive integers.")
        self.level_cluster_counts = level_cluster_counts
        if representation_mode not in ["direct", "description"]:
             raise ValueError("`representation_mode` must be 'direct' or 'description'")
        self.representation_mode = representation_mode

        if self.level_cluster_counts is None:
            valid_auto_k_methods = ['silhouette', 'davies_bouldin', 'calinski_harabasz']
            if auto_k_method not in valid_auto_k_methods:
                raise ValueError(f"`auto_k_method` must be one of {valid_auto_k_methods}")
            self.auto_k_method = auto_k_method
            self.auto_k_max = max(2, auto_k_max)
            self.auto_k_metric_params = auto_k_metric_params if isinstance(auto_k_metric_params, dict) else {}
            if self.auto_k_max < min_clusters_per_level:
                warnings.warn(f"auto_k_max ({self.auto_k_max}) is less than min_clusters_per_level ({min_clusters_per_level}). Effective minimum k will be {min_clusters_per_level}.")
        else:
            self.auto_k_method = auto_k_method
            self.auto_k_max = auto_k_max
            self.auto_k_metric_params = auto_k_metric_params

        self.n_reduction_components = max(1, n_reduction_components)
        if isinstance(reduction_methods, str): reduction_methods = [reduction_methods]
        if not isinstance(reduction_methods, list): raise ValueError("`reduction_methods` must be str or list[str].")
        self.reduction_methods = [m.lower() for m in reduction_methods if m.lower() == 'pca']
        if reduction_methods and not self.reduction_methods:
            warnings.warn(f"Specified reduction methods {reduction_methods} are not supported (only 'pca'). No reduction will be performed.")
        elif 'pca' not in self.reduction_methods and reduction_methods:
             warnings.warn(f"Specified reduction methods {reduction_methods} do not include 'pca'. Only 'pca' is supported. No reduction will be performed.")
             self.reduction_methods = []

        self.prompt_l0_sample_size = max(0, prompt_l0_sample_size)
        if prompt_l0_sample_strategy not in ["centroid_closest", "random"]: raise ValueError("`prompt_l0_sample_strategy` must be 'centroid_closest' or 'random'")
        self.prompt_l0_sample_strategy = prompt_l0_sample_strategy
        self.prompt_l0_sample_trunc_len = max(10, prompt_l0_sample_trunc_len)
        self.prompt_l0_numeric_repr_max_vals = max(0, prompt_l0_numeric_repr_max_vals)
        self.prompt_l0_numeric_repr_precision = max(0, prompt_l0_numeric_repr_precision)
        self.prompt_include_immediate_children = prompt_include_immediate_children
        if prompt_immediate_child_sample_strategy not in ["centroid_closest", "random"]: raise ValueError("`prompt_immediate_child_sample_strategy` must be 'centroid_closest' or 'random'")
        self.prompt_immediate_child_sample_strategy = prompt_immediate_child_sample_strategy
        self.prompt_immediate_child_sample_size = max(0, prompt_immediate_child_sample_size)
        self.prompt_child_sample_desc_trunc_len = max(10, prompt_child_sample_desc_trunc_len)
        self.prompt_max_stats_vars = max(0, prompt_max_stats_vars)
        self.prompt_numeric_stats_precision = max(0, prompt_numeric_stats_precision)
        self.max_prompt_tokens = max(100, max_prompt_tokens)
        self.max_prompt_token_buffer = max(1.0, max_prompt_token_buffer)
        self.llm_initial_batch_size = max(1, llm_initial_batch_size)
        self.llm_batch_reduction_factor = max(1.1, llm_batch_reduction_factor)
        self.llm_min_batch_size = max(1, llm_min_batch_size)
        self.llm_retries = max(0, llm_retries)
        self.llm_retry_delay = max(0.1, llm_retry_delay)

        self.random_state = random_state
        self._configure_randomness()
        self.save_run_details = save_run_details
        self.run_details_dir = run_details_dir

        self.direct_l0_text_trunc_len = max(20, direct_l0_text_trunc_len)
        self.min_clusters_per_level = max(2, min_clusters_per_level)
        self.fallback_k = max(1, fallback_k)
        self.cluster_numeric_stats_precision = max(0, cluster_numeric_stats_precision)
        self.cluster_print_indent_increment = max(1, cluster_print_indent_increment)
        self.use_llm_for_l0_descriptions = use_llm_for_l0_descriptions
        
        self.use_resampling = use_resampling
        self.resampling_points_per_cluster = max(1, resampling_points_per_cluster)
        self.resampling_iterations = max(1, resampling_iterations)

        self.variable_names: list[str] | None = None
        self.numeric_metadata_by_name: dict[str, dict[str, Any]] | None = None
        self.input_data_type: str | None = None
        self.original_numeric_data_: np.ndarray | None = None
        self._scaler: StandardScaler | None = None

        self._reducers_trained_ = {"text_embedding": False, "image_embedding": False, "numeric": False}
        self.reducers_ = {"text_embedding": {}, "image_embedding": {}, "numeric": {}}
        self.embedding_dims_ = {"text_embedding": None, "image_embedding": None, "numeric": None}
        
        self._run_id: str | None = None
        self._prompt_log: list[dict] = []
        self._current_topic_seed: str | None = None
        self._all_clusters_map: dict[int, Cluster] = {}
        self._l0_clusters_ordered: list[Cluster] = []
        self._max_level: int = 0
        self._original_id_to_index = {}

        if self.save_run_details and not os.path.exists(self.run_details_dir):
             try: os.makedirs(self.run_details_dir, exist_ok=True)
             except OSError as e: print(f"Warning: Could not create run details dir {self.run_details_dir}: {e}")

    def _get_function_metadata(self, func: Callable) -> Dict[str, str | None]:
        """Extracts metadata (name, module, signature, docstring) from a function."""
        if not callable(func):
            return {"name": "N/A (Not Callable)", "module": None, "signature": None, "docstring": None}
        metadata = {}
        try: metadata["name"] = getattr(func, '__name__', 'unknown')
        except Exception: metadata["name"] = 'unknown (error getting name)'
        try: metadata["module"] = getattr(func, '__module__', 'unknown')
        except Exception: metadata["module"] = 'unknown (error getting module)'
        try: metadata["signature"] = str(inspect.signature(func))
        except (ValueError, TypeError, Exception): metadata["signature"] = 'unknown (error getting signature)'
        try: metadata["docstring"] = inspect.getdoc(func)
        except Exception: metadata["docstring"] = 'unknown (error getting docstring)'
        return metadata

    def _configure_randomness(self):
        """Sets random seeds for reproducibility."""
        if self.random_state is not None:
            np.random.seed(self.random_state)
            random.seed(self.random_state)

    def _log(self, message: str, level: int = 1):
        """Prints message if verbose level is sufficient."""
        if self.verbose >= level:
            print(message)

    def _is_potential_image_identifier(self, item: Any) -> bool:
        """Checks if an item is a PIL Image, an image file path, or an image URL."""
        if PIL_IMAGE_TYPE != type(None) and isinstance(item, PIL_IMAGE_TYPE): # Check for PIL Image object
            return True
        if isinstance(item, str):
            try:
                if _is_string_url(item): # Check URL (using module-level helper)
                    parsed_url = urlparse(item)
                    return Path(parsed_url.path).suffix.lower() in IMAGE_EXTENSIONS
                else: # Check file path
                    if not item or len(item) > 1024 : return False # Basic sanity check for path string
                    return Path(item).suffix.lower() in IMAGE_EXTENSIONS
            except (TypeError, ValueError):
                return False
            except Exception:
                return False
        return False

    def _generate_and_assign_description_embeddings(self, clusters_to_process: list[Cluster]):
        """
        Generates description embeddings for given clusters and assigns them.
        Trains embedding reducers if not already trained.
        Ensures no empty strings are sent to the text_embedding_client.
        """
        if not clusters_to_process: return

        level_str = f"L{clusters_to_process[0].level}"
        self._log(f"Generating description embeddings for {len(clusters_to_process)} clusters ({level_str})...", level=2)

        embedding_space = 'text_embedding' # Description embeddings are always in this space

        texts_for_client = [] 
        clusters_that_will_get_embedding = []
        clusters_with_empty_original_desc = []

        for cluster in clusters_to_process:
            title = cluster.title or ""
            desc = cluster.description or ""
            combined_desc = f"{title}. {desc}".replace("..", ".").replace(". .", ".").strip()

            if not combined_desc:
                texts_for_client.append(Hercules.EMPTY_TEXT_PLACEHOLDER)
                clusters_that_will_get_embedding.append(cluster)
                clusters_with_empty_original_desc.append(cluster)
            else:
                texts_for_client.append(combined_desc)
                clusters_that_will_get_embedding.append(cluster)

        if clusters_with_empty_original_desc:
            example_ids = [(c.original_id if c.level == 0 else c.id) for c in clusters_with_empty_original_desc[:5]]
            warnings.warn(
                f"{len(clusters_with_empty_original_desc)} clusters (L{level_str}) had empty combined title/description. "
                f"Using placeholder '{Hercules.EMPTY_TEXT_PLACEHOLDER}' for their description embedding. "
                f"Example Cluster/Original IDs: {example_ids}{'...' if len(clusters_with_empty_original_desc)>5 else ''}"
            )
    
        if not texts_for_client:
            self._log(f"  No texts (not even placeholders) found to generate description embeddings for {level_str}.", level=2)
            return

        # --- Embed Descriptions ---
        self._log(f"  Calling text embedding client for {len(texts_for_client)} combined descriptions ({level_str})...", level=2)
        try:
            desc_embeddings = self.text_embedding_client(texts_for_client)
            if desc_embeddings.shape[0] != len(texts_for_client):
                 raise ValueError(f"Embedding client returned incorrect shape ({desc_embeddings.shape[0]} vs {len(texts_for_client)} expected).")

            # Clean up non-finite embeddings
            if not np.all(np.isfinite(desc_embeddings)):
                num_bad = np.sum(~np.isfinite(desc_embeddings))
                warnings.warn(f"{level_str} description embeddings contain {num_bad} non-finite values. Attempting nan_to_num.")
                desc_embeddings = np.nan_to_num(desc_embeddings, nan=0.0, posinf=0.0, neginf=0.0)
                if not np.all(np.isfinite(desc_embeddings)):
                     raise ValueError("Non-finite values persist after cleanup.")

            # --- Train Embedding Reducers (if needed) ---
            if self.reduction_methods and not self._reducers_trained_.get(embedding_space, False):
                 if desc_embeddings.shape[0] > 0:
                     self._log(f"  Training '{embedding_space}' space reducers using {level_str} description embeddings.", level=2)
                     self._train_reducers(desc_embeddings, embedding_space)
                     if self.embedding_dims_.get(embedding_space) is None: self.embedding_dims_[embedding_space] = desc_embeddings.shape[1]
                 else:
                     self._log(f"  Skipping embedding reducer training: No valid {level_str} description embeddings generated.", level=2)

            # --- Assign Embeddings ---
            num_assigned = 0
            for i, cluster in enumerate(clusters_that_will_get_embedding): 
                 cluster.description_embedding = desc_embeddings[i]
                 num_assigned += 1
            self._log(f"  Assigned {num_assigned} description embeddings ({level_str}).", level=2)

            all_assigned_cluster_ids = {c.id for c in clusters_that_will_get_embedding}
            for cluster in clusters_to_process:
                if cluster.id not in all_assigned_cluster_ids:
                    cluster.description_embedding = None
                    cluster.description_embedding_reductions = {}
                    warnings.warn(f"Cluster {cluster.id} (L{level_str}) was in process list but did not receive a description embedding. This indicates a potential logic flaw.")


        except Exception as e:
            warnings.warn(f"Error generating or assigning description embeddings for {level_str}: {e}. Affected clusters will lack description embeddings.")
            for cluster in clusters_that_will_get_embedding: 
                 cluster.description_embedding = None
                 cluster.description_embedding_reductions = {}

    def _apply_reductions_to_embeddings(self, clusters_to_process: list[Cluster], space_type: str):
        """Applies trained reducers to vectors in a specific space for given clusters."""
        if not self.reduction_methods or not clusters_to_process: return

        level_str = f"L{clusters_to_process[0].level}"
        self._log(f"Applying reductions to '{space_type}' vectors for {len(clusters_to_process)} clusters ({level_str})...", level=2)

        processed_count = 0
        for cluster in clusters_to_process:
            vector = None
            reduction_attr_name = None

            if space_type == 'description_embedding':
                 vector = cluster.description_embedding
                 reduction_space_config = 'text_embedding'
                 reduction_attr_name = 'description_embedding_reductions'
            elif space_type == 'representation_vector':
                 vector = cluster.representation_vector
                 reduction_space_config = cluster.representation_vector_space
                 reduction_attr_name = 'representation_vector_reductions'
            else:
                 warnings.warn(f"Unknown space type '{space_type}' requested for reduction application.")
                 continue

            if vector is not None and reduction_space_config is not None and reduction_attr_name is not None:
                 reductions = self._apply_reduction(vector, reduction_space_config)
                 if reductions:
                     setattr(cluster, reduction_attr_name, reductions)
                     processed_count += 1
            elif reduction_attr_name is not None:
                 setattr(cluster, reduction_attr_name, {})


        self._log(f"  Applied reductions to {processed_count} '{space_type}' vectors ({level_str}).", level=2)

    def _prepare_input_data(self, data: Any,
                          numeric_metadata: Optional[Dict[Union[str, int], Dict[str, Any]]] = None
                          ) -> tuple[list | np.ndarray, list, str]:
        """
        Validates, standardizes input, detects type ('text', 'numeric', 'image').
        Scales numeric data, stores original, and processes optional numeric metadata.

        Returns:
            standardized_data: List of texts/image_ids or SCALED numpy array (numeric).
            original_ids: List of original IDs/indices.
            input_type: 'text', 'numeric', or 'image'.
        """
        self._log("Starting input data preparation...", level=2)
        self.variable_names = None
        self.numeric_metadata_by_name = None
        self.original_numeric_data_ = None
        self._scaler = None
        input_type = None
        standardized_data = None
        original_ids = None
        numeric_data_unscaled = None
        original_numeric_keys: List[Union[str, int]] | None = None

        # --- Input Type Detection and Standardization ---
        if isinstance(data, list):
            if not data: raise ValueError("Input data list is empty.")
            first_item = data[0]
            self._log(f"Processing list input ({len(data)} items, first item type: {type(first_item).__name__})", level=2)

            if PIL_IMAGE_TYPE != type(None) and all(isinstance(item, PIL_IMAGE_TYPE) for item in data):
                input_type = "image"; standardized_data = data
                original_ids = list(range(len(data)))
                self._log("Detected input type: image (list of PIL.Image objects)", level=2)
            elif all(isinstance(item, str) for item in data):
                image_like_count = sum(1 for item in data if self._is_potential_image_identifier(item))
                if len(data) > 0 and image_like_count / len(data) > 0.8: # Heuristic for list of image paths/URLs
                    input_type = "image"; standardized_data = data
                    if len(set(data)) == len(data): original_ids = data
                    else: original_ids = list(range(len(data))); warnings.warn("Duplicate image identifiers (paths/URLs) found in list, using indices as IDs.")
                    self._log("Detected input type: image (list of paths/URLs)", level=2)
                else:
                    input_type = "text"; standardized_data = data; original_ids = list(range(len(data)))
                    self._log("Detected input type: text (list)", level=2)
            elif all(isinstance(item, (int, float, np.number)) for item in data):
                 input_type = "numeric"; numeric_data_unscaled = np.array(data).reshape(-1, 1)
                 original_numeric_keys = [0]
                 original_ids = list(range(len(data)))
                 self._log("Detected input type: numeric (list of scalars)", level=2)
            elif all(isinstance(item, (list, np.ndarray)) for item in data):
                try:
                     def is_numeric_vector(v):
                         if isinstance(v, np.ndarray): return np.issubdtype(v.dtype, np.number)
                         if isinstance(v, list): return all(isinstance(x, (int, float, np.number)) for x in v)
                         return False
                     if not all(is_numeric_vector(item) for item in data):
                         raise TypeError("List of lists/arrays detected, but not all sub-items are purely numeric.")

                     numeric_data_list = [np.asarray(item, dtype=float).flatten() for item in data]
                     first_len = len(numeric_data_list[0]) if numeric_data_list else 0
                     if not all(len(arr) == first_len for arr in numeric_data_list): raise ValueError("Inconsistent lengths in list of numeric vectors.")
                     numeric_data_unscaled = np.array(numeric_data_list)
                     if numeric_data_unscaled.ndim != 2: raise ValueError("List of numeric vectors must result in a 2D array.")
                     input_type = "numeric"
                     original_numeric_keys = list(range(numeric_data_unscaled.shape[1]))
                     original_ids = list(range(len(data)))
                     self._log("Detected input type: numeric (list of vectors)", level=2)
                except ValueError as e: raise TypeError(f"Could not process list of numeric lists/arrays: {e}")
                except TypeError as e: raise TypeError(f"Could not process list of lists/arrays as numeric: {e}")
            elif all(isinstance(item, dict) for item in data):
                 self._log("Input is list of dicts, attempting numeric conversion.", level=2)
                 try:
                     df = pd.DataFrame(data)
                     numeric_df = df.select_dtypes(include=np.number)
                     if numeric_df.empty: raise ValueError("List of dicts resulted in DataFrame with no numeric columns.")
                     numeric_data_unscaled = numeric_df.values.astype(float)
                     original_numeric_keys = numeric_df.columns.tolist()
                     input_type = "numeric"; original_ids = list(range(len(data)))
                     self._log("Detected input type: numeric (list of dicts)", level=2)
                 except Exception as e: raise TypeError(f"Could not process list of numeric dictionaries: {e}")
            else: raise TypeError("Input list contains mixed or unsupported item types for automatic detection.")

        elif isinstance(data, dict):
             if not data: raise ValueError("Input data dictionary is empty.")
             original_ids = list(data.keys())
             values = list(data.values())
             first_value = values[0]
             self._log(f"Processing dict input ({len(data)} items, first value type: {type(first_value).__name__})", level=2)

             if PIL_IMAGE_TYPE != type(None) and all(isinstance(v, PIL_IMAGE_TYPE) for v in values):
                 input_type = "image"; standardized_data = values
                 self._log("Detected input type: image (dict of PIL.Image objects)", level=2)
             elif all(isinstance(v, str) for v in values):
                 image_like_count = sum(1 for v in values if self._is_potential_image_identifier(v))
                 if len(values) > 0 and image_like_count / len(values) > 0.8: # Heuristic
                      input_type = "image"; standardized_data = values
                      self._log("Detected input type: image (dict of paths/URLs)", level=2)
                 else:
                      input_type = "text"; standardized_data = values
                      self._log("Detected input type: text (dict)", level=2)
             elif all(isinstance(v, (int, float, np.number)) for v in values):
                  input_type = "numeric"; numeric_data_unscaled = np.array(values).reshape(-1, 1)
                  original_numeric_keys = [0]
                  self._log("Detected input type: numeric (dict of scalars)", level=2)
             elif all(isinstance(v, (list, np.ndarray)) for v in values):
                 try:
                      def is_numeric_vector(vec):
                          if isinstance(vec, np.ndarray): return np.issubdtype(vec.dtype, np.number)
                          if isinstance(vec, list): return all(isinstance(x, (int, float, np.number)) for x in vec)
                          return False
                      if not all(is_numeric_vector(v_item) for v_item in values):
                          raise TypeError("Dict of lists/arrays detected, but not all values are purely numeric vectors.")

                      numeric_data_list = [np.asarray(item, dtype=float).flatten() for item in values]
                      first_len = len(numeric_data_list[0]) if numeric_data_list else 0
                      if not all(len(arr) == first_len for arr in numeric_data_list): raise ValueError("Inconsistent vector lengths in dict values.")
                      numeric_data_unscaled = np.array(numeric_data_list)
                      if numeric_data_unscaled.ndim != 2: raise ValueError("Dict of numeric vectors must result in a 2D array.")
                      input_type = "numeric"
                      original_numeric_keys = list(range(numeric_data_unscaled.shape[1]))
                      self._log("Detected input type: numeric (dict of vectors)", level=2)
                 except ValueError as e: raise TypeError(f"Could not process dictionary with numeric vectors: {e}")
                 except TypeError as e: raise TypeError(f"Could not process dictionary values as numeric vectors: {e}")
             else: raise TypeError("Input dictionary values must be all PIL.Image, all strings, or all compatible numerics.")

        elif isinstance(data, pd.Series):
             self._log(f"Processing Pandas Series input ({len(data)} items, dtype: {data.dtype})", level=2)
             if pd.api.types.is_string_dtype(data.dtype) or data.dtype == 'object':
                 series_list = data.tolist()
                 image_like_count = sum(1 for item in series_list if isinstance(item, str) and self._is_potential_image_identifier(item))
                 if len(series_list) > 0 and image_like_count / len(series_list) > 0.8: # Heuristic
                      input_type = "image"; standardized_data = series_list
                      original_ids = data.index.tolist()
                      self._log("Detected input type: image (Series of paths/URLs)", level=2)
                 else: # Assume text
                      input_type = "text"; standardized_data = series_list
                      original_ids = data.index.tolist()
                      self._log("Detected input type: text (Series)", level=2)
             elif pd.api.types.is_numeric_dtype(data.dtype):
                 input_type = "numeric"; numeric_data_unscaled = data.values.reshape(-1, 1)
                 original_numeric_keys = [data.name or 0]
                 original_ids = data.index.tolist()
                 self._log("Detected input type: numeric (Series)", level=2)
             elif PIL_IMAGE_TYPE != type(None) and data.dtype == 'object' and all(isinstance(item, PIL_IMAGE_TYPE) for item in data if pd.notna(item)):
                 valid_pil_data = [item for item in data if isinstance(item, PIL_IMAGE_TYPE)]
                 if not valid_pil_data: raise TypeError("Pandas Series of object type contains no valid PIL.Image.Image objects.")
                 input_type = "image"; standardized_data = valid_pil_data
                 original_ids = data.index[data.apply(lambda x: isinstance(x, PIL_IMAGE_TYPE))].tolist() if len(valid_pil_data) < len(data) else data.index.tolist()
                 if len(standardized_data) != len(original_ids):
                     warnings.warn("Length mismatch after filtering PIL objects from Series. ID alignment might be affected. Using indices for PIL objects.")
                     standardized_data = [item for item in data if isinstance(item, PIL_IMAGE_TYPE)]
                     original_ids = list(range(len(standardized_data)))

                 self._log("Detected input type: image (Series of PIL.Image objects)", level=2)
             else: raise TypeError(f"Unsupported Pandas Series dtype for automatic detection: {data.dtype}")
        
        elif isinstance(data, pd.DataFrame):
             original_ids = data.index.tolist()
             self._log(f"Processing Pandas DataFrame input (shape: {data.shape})", level=2)
             input_type = "numeric"
             try:
                 numeric_df = data.select_dtypes(include=np.number)
                 if numeric_df.empty: raise ValueError("Input DataFrame contains no numeric columns.")
                 if numeric_df.shape[1] < data.shape[1]:
                     warnings.warn(f"DataFrame has non-numeric columns. Only using: {numeric_df.columns.tolist()}")
                 numeric_data_unscaled = numeric_df.values.astype(float)
                 original_numeric_keys = numeric_df.columns.tolist()
                 self._log("Detected input type: numeric (DataFrame)", level=2)
             except Exception as e: raise TypeError(f"Could not convert DataFrame to numeric: {e}")

        elif isinstance(data, np.ndarray):
            self._log(f"Processing NumPy array input (shape: {data.shape}, dtype: {data.dtype})", level=2)
            if not np.issubdtype(data.dtype, np.number) and data.dtype != object:
                raise TypeError(f"NumPy array input must be numeric or object type for images, got dtype: {data.dtype}")

            if data.dtype == object:
                self._log("NumPy array is object type, checking for image identifiers...", level=2)
                if data.ndim != 1: raise TypeError("Object type NumPy array must be 1D (list of image identifiers).")
                
                list_data = data.tolist()
                if PIL_IMAGE_TYPE != type(None) and all(isinstance(item, PIL_IMAGE_TYPE) for item in list_data):
                    input_type = "image"; standardized_data = list_data
                    original_ids = list(range(len(list_data)))
                    self._log("Detected input type: image (NumPy array of PIL.Image objects)", level=2)
                elif all(isinstance(item, str) for item in list_data):
                    image_like_count = sum(1 for item in list_data if self._is_potential_image_identifier(item))
                    if len(list_data) > 0 and image_like_count / len(list_data) > 0.8:
                        input_type = "image"; standardized_data = list_data
                        if len(set(list_data)) == len(list_data): original_ids = list_data
                        else: original_ids = list(range(len(list_data))); warnings.warn("Duplicate image identifiers in NumPy array, using indices as IDs.")
                        self._log("Detected input type: image (NumPy array of image paths/URLs)", level=2)
                    else:
                        raise TypeError("Object type NumPy array of strings does not appear to be image identifiers.")
                else:
                    raise TypeError("Object type NumPy array contains mixed types or unsupported image identifiers.")

            elif data.ndim == 1:
                input_type = "numeric"; numeric_data_unscaled = data.reshape(-1, 1)
                original_numeric_keys = [0]
                original_ids = list(range(len(data)))
                self._log("Detected input type: numeric (1D NumPy array)", level=2)
            elif data.ndim == 2:
                input_type = "numeric"
                numeric_data_unscaled = data.astype(float)
                original_numeric_keys = list(range(numeric_data_unscaled.shape[1]))
                original_ids = list(range(numeric_data_unscaled.shape[0]))
                self._log("Detected input type: numeric (2D NumPy array)", level=2)
            else:
                raise TypeError(f"Numeric NumPy arrays with >2 dimensions are not directly supported. Found {data.ndim} dimensions.")
        else:
            raise TypeError(f"Unsupported input data type: {type(data)}.")

        # --- Numeric Data Processing (Scaling, Metadata) ---
        if input_type == "numeric":
            if numeric_data_unscaled is None or numeric_data_unscaled.size == 0: raise ValueError("Numeric data extraction failed.")
            if np.isnan(numeric_data_unscaled).any():
                 raise ValueError("Numeric data contains NaN values. Please handle them before clustering.")
            if not np.all(np.isfinite(numeric_data_unscaled)):
                 raise ValueError("Numeric data contains non-finite values (inf/-inf). Please handle them before clustering.")

            self.original_numeric_data_ = numeric_data_unscaled.copy()

            if original_numeric_keys is not None:
                num_features = len(original_numeric_keys)
                final_var_names = [f"feature_{i}" for i in range(num_features)]
                self.numeric_metadata_by_name = {}

                if numeric_metadata:
                    self._log("Processing provided numeric metadata...", level=2)
                    validated_metadata_count = 0
                    temp_metadata_by_name = {}
                    temp_final_var_names = list(final_var_names)

                    for i, key in enumerate(original_numeric_keys):
                        meta = numeric_metadata.get(key)
                        display_name = str(key)

                        if isinstance(meta, dict):
                             provided_name = meta.get('name')
                             if isinstance(provided_name, str) and provided_name.strip():
                                 display_name = provided_name.strip()

                             temp_final_var_names[i] = display_name
                             temp_metadata_by_name[display_name] = meta
                             validated_metadata_count += 1
                        else:
                             temp_final_var_names[i] = f"feature_{i}"

                    if len(set(temp_final_var_names)) != len(temp_final_var_names):
                         warnings.warn("Name collision detected in final variable names (from metadata or keys). Using default names ('feature_i') instead.")
                         final_var_names = [f"feature_{i}" for i in range(num_features)]
                         temp_metadata_by_name = {}
                    else:
                        final_var_names = temp_final_var_names
                        self.numeric_metadata_by_name = temp_metadata_by_name

                    self.variable_names = final_var_names
                    self._log(f"  Applied metadata to {validated_metadata_count}/{num_features} variables.", level=2)
                    self._log(f"  Final variable names: {self.variable_names}", level=2)
                else:
                    self.variable_names = [str(key) for key in original_numeric_keys]
                    self._log(f"  No numeric metadata provided. Using names: {self.variable_names}", level=2)

            self._log(f"Scaling numeric data (shape: {numeric_data_unscaled.shape})...", level=2)
            self._scaler = StandardScaler()
            try:
                standardized_data = self._scaler.fit_transform(numeric_data_unscaled)
                if not np.all(np.isfinite(standardized_data)):
                    warnings.warn("Non-finite values detected after scaling (e.g., due to zero variance). Attempting to replace with 0.")
                    standardized_data = np.nan_to_num(standardized_data, nan=0.0, posinf=0.0, neginf=0.0)
                    if not np.all(np.isfinite(standardized_data)):
                       raise ValueError("Non-finite values persist even after nan_to_num post-scaling.")
            except ValueError as e: raise ValueError(f"Error scaling numeric data: {e}")

        elif input_type in ["text", "image"]:
             if not standardized_data: raise ValueError(f"{input_type.capitalize()} data is empty.")

        # --- Final Checks ---
        if input_type is None or standardized_data is None or original_ids is None:
             raise RuntimeError("Internal error: Could not determine input type or process data.")

        self.input_data_type = input_type
        data_len = standardized_data.shape[0] if isinstance(standardized_data, np.ndarray) else len(standardized_data)
        if len(original_ids) != data_len:
             raise RuntimeError(f"Internal error: Length mismatch between processed data ({data_len}) and original IDs ({len(original_ids)}).")

        self._original_id_to_index = {orig_id: i for i, orig_id in enumerate(original_ids)}
        self._log("Input data preparation complete.", level=2)
        return standardized_data, original_ids, input_type

    def _train_reducers(self, vectors: np.ndarray, space_type: str):
        """Trains PCA reducer for a given space type ('numeric' or 'embedding')."""
        if not self.reduction_methods: return
        if space_type not in self.reducers_:
            self._log(f"Warning: Unknown space type '{space_type}' for reducer training. Skipping.", level=1)
            return
        if self._reducers_trained_.get(space_type, False): return

        if vectors is None or vectors.shape[0] < max(2, self.n_reduction_components):
            self._log(f"Warning: Not enough data ({vectors.shape[0] if vectors is not None else 0} items, min {max(2, self.n_reduction_components)} needed) for {self.n_reduction_components}-D reducers ({space_type}). Skipping training.", level=1)
            return

        if not np.all(np.isfinite(vectors)):
             warnings.warn(f"Non-finite values (NaN/inf) detected in vectors for training reducers ({space_type}). Skipping training.")
             return

        self._log(f"Training reducers for '{space_type}' space (data shape: {vectors.shape})...", level=2)
        self.reducers_[space_type] = {}

        if "pca" in self.reduction_methods:
             actual_n_pca = min(self.n_reduction_components, vectors.shape[0], vectors.shape[1])
             if actual_n_pca < 1:
                 self._log(f"  Skipping PCA ({space_type}): effective n_components ({actual_n_pca}) < 1.", level=2)
             else:
                 try:
                     pca = PCA(n_components=actual_n_pca, random_state=self.random_state)
                     pca.fit(vectors)
                     self.reducers_[space_type]["pca"] = pca
                     self.embedding_dims_[space_type] = vectors.shape[1]
                     self._log(f"    PCA trained for {space_type} (input dim: {vectors.shape[1]}, output dim: {actual_n_pca}).", level=2)
                 except Exception as e: warnings.warn(f"Error training PCA for {space_type}: {e}")

        self._reducers_trained_[space_type] = True

    def _apply_reduction(self, vector: np.ndarray, space_type: str) -> dict[str, np.ndarray]:
        """Applies trained reducers (PCA) to a single vector."""
        reductions = {}
        if not self.reduction_methods: return reductions
        if vector is None or space_type not in self.reducers_: return reductions
        if not self._reducers_trained_.get(space_type, False):
             return reductions

        vector_2d = vector.reshape(1, -1) if vector.ndim == 1 else vector
        if not np.all(np.isfinite(vector_2d)):
             return {}

        expected_dim = self.embedding_dims_.get(space_type)

        for method, reducer in self.reducers_[space_type].items():
            if reducer is None or method != 'pca': continue

            try:
                n_features_in = getattr(reducer, 'n_features_in_', -1)
                if n_features_in == -1 and expected_dim is not None: n_features_in = expected_dim

                if n_features_in != -1 and n_features_in != vector_2d.shape[1]:
                     warnings.warn(f"Dimension mismatch applying {method.upper()} ({space_type}). Reducer trained on {n_features_in} features, vector has {vector_2d.shape[1]}. Skipping reduction.")
                     continue

                reduced_vector = reducer.transform(vector_2d)

                if np.all(np.isfinite(reduced_vector)):
                    reductions[method] = reduced_vector[0]
                else:
                    warnings.warn(f"Non-finite values produced by {method.upper()} transform ({space_type}). Skipping result for this vector.")
            except Exception as e:
                 warnings.warn(f"Error applying {method.upper()} reduction ({space_type}) to vector: {e}. Skipping reduction for this vector.")

        return reductions

    def _determine_optimal_k(self, vectors: np.ndarray, min_k: int, max_k: int,
                             method: str, space_type: str) -> int:
        """Determines the optimal number of clusters (k) using a specified method."""
        n_samples = vectors.shape[0]
        actual_min_k = max(min_k, self.min_clusters_per_level)
        actual_max_k = min(max_k, self.auto_k_max, n_samples - 1)

        self._log(f"  Determining optimal k using '{method}' method...", level=1)
        self._log(f"    Testing k range: {actual_min_k} to {actual_max_k} (n_samples={n_samples})", level=2)

        if actual_max_k < actual_min_k:
            self._log(f"    Warning: Invalid k range [{actual_min_k}, {actual_max_k}]. Using fallback_k={self.fallback_k}.", level=1)
            return min(self.fallback_k, n_samples - 1) if n_samples > 1 else 1

        scores = {}
        best_score = -np.inf if method in ['silhouette', 'calinski_harabasz'] else np.inf
        best_k = actual_min_k

        k_range = range(actual_min_k, actual_max_k + 1)
        if not k_range:
             self._log(f"    Warning: Empty k range after adjustments. Using fallback_k={self.fallback_k}.", level=1)
             return min(self.fallback_k, n_samples - 1) if n_samples > 1 else 1

        for k in k_range:
            if k >= n_samples: continue
            try:
                # Use n_init='auto' for newer sklearn, fallback for older versions.
                try:
                    kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init='auto')
                    labels = kmeans.fit_predict(vectors)
                except TypeError:
                    kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
                    labels = kmeans.fit_predict(vectors)

                score = None
                if method == 'silhouette':
                    metric = self.auto_k_metric_params.get('metric', 'cosine' if space_type in ['text_embedding', 'image_embedding'] else 'euclidean')
                    valid_silhouette_params = {p:v for p,v in self.auto_k_metric_params.items() if p != 'metric'}
                    score = silhouette_score(vectors, labels, metric=metric, **valid_silhouette_params)
                    if score > best_score: best_score = score; best_k = k
                elif method == 'davies_bouldin':
                    score = davies_bouldin_score(vectors, labels)
                    if score < best_score: best_score = score; best_k = k
                elif method == 'calinski_harabasz':
                    score = calinski_harabasz_score(vectors, labels)
                    if score > best_score: best_score = score; best_k = k
                else: raise ValueError(f"Unsupported auto_k_method: {method}")
                scores[k] = score
                self._log(f"      k={k}, method='{method}', score={score:.4f}", level=3)
            except ValueError as ve:
                 self._log(f"    Warning: Skipping k={k} due to KMeans/metric error: {ve}", level=2)
                 scores[k] = None; continue
            except Exception as e:
                 self._log(f"    Warning: Error calculating score for k={k}: {e}", level=2)
                 scores[k] = None; continue

        valid_scores = {k: s for k, s in scores.items() if s is not None}
        if not valid_scores:
            self._log(f"    Warning: No valid scores found for any k in range. Using fallback_k={self.fallback_k}.", level=1)
            return min(self.fallback_k, n_samples - 1) if n_samples > 1 else 1

        if method == 'silhouette': best_k = max(valid_scores, key=valid_scores.get); best_score = valid_scores[best_k]
        elif method == 'davies_bouldin': best_k = min(valid_scores, key=valid_scores.get); best_score = valid_scores[best_k]
        elif method == 'calinski_harabasz': best_k = max(valid_scores, key=valid_scores.get); best_score = valid_scores[best_k]

        self._log(f"  Auto K determination complete. Method='{method}'. Best k={best_k} (Score: {best_score:.4f}).", level=1)

        if best_k >= n_samples:
            warnings.warn(f"Determined optimal k ({best_k}) is >= n_samples ({n_samples}). Using fallback_k={self.fallback_k}.")
            return min(self.fallback_k, n_samples - 1) if n_samples > 1 else 1

        return best_k

    def _build_llm_prompt(self, cluster_prompt_data_list: list[dict]) -> tuple[str | None, str | None, int, list[Union[int, str]]]:
        """
        Constructs the LLM prompt for batch description generation.

        Includes L0 descendant samples, optionally immediate child samples,
        and numeric statistics with metadata.

        Returns:
            Tuple of (prompt string | None, prompt ID | None, estimated token count, list of included IDs).
        """
        if not cluster_prompt_data_list:
            return None, str(uuid.uuid4()), self.max_prompt_tokens + 1, []

        prompt_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()

        first_item_data = cluster_prompt_data_list[0]
        level = first_item_data["level"]
        item_type = "Cluster" if level > 0 else "Data Item"
        item_type_plural = "Clusters" if level > 0 else "Data Items"
        item_ids_requested = [str(data["id"]).replace('\\', '/') for data in cluster_prompt_data_list]
        n_items_requested = len(item_ids_requested)
        self._log(f"Building LLM prompt {prompt_id} for {n_items_requested} {item_type_plural} (Level {level})...", level=2)

        prompt = f"""Generate a concise 'title' (max 5-7 words) and 'description' (1-2 sentences) for EACH of the {item_type_plural} below (Level {level}).
"""
        if self._current_topic_seed:
            escaped_seed = self._current_topic_seed.replace('"', '\\"').replace("'", "\\'")
            prompt += f"CONTEXT FOCUS: Orient towards '{escaped_seed}', if relevant.\n"

        prompt += f"""
RESPONSE FORMAT: Respond ONLY with a single, valid JSON object.
- Top-level keys MUST be the string representation of the '{item_type} ID' provided (e.g., "{item_ids_requested[0] if item_ids_requested else 'example_id'}").
- Values MUST be JSON objects containing non-empty "title" and "description" string keys.

EXAMPLE (for IDs "id_A", "id_B"):
{{
  "id_A": {{ "title": "Title A", "description": "Desc A." }},
  "id_B": {{ "title": "Title B", "description": "Desc B." }}
}}

IMPORTANT: Ensure the entire output is valid JSON. Do NOT include markdown fences (```json ... ```) or any text outside the JSON structure.

--- {item_type_plural} Information ---
"""
        prompt_tokens_estimate = _estimate_token_count(prompt)
        prompt_items_added = 0
        item_ids_included = []

        for i, data in enumerate(cluster_prompt_data_list):
            item_id_str = item_ids_requested[i]
            original_id_for_lookup = data["id"]
            internal_id = data["internal_id"]
            num_underlying = data.get("num_items", 1)
            orig_type = data.get("original_data_type", "unknown").capitalize()
            variable_metadata = data.get("variable_metadata_by_name")

            level_info = f"L{level}"
            if level > 0: level_info += f" ({num_underlying} base items)"
            level_info += f", BaseType: {orig_type}, InternalID: {internal_id}"

            cluster_block = f"\n--- {item_type} ID: {item_id_str} ({level_info}) ---\n"

            if data.get('image_identifier'):
                 cluster_block += f"Image Identifier: {data['image_identifier']}\n"

            l0_samples = data.get("samples", [])
            if l0_samples:
                 sample_desc = "Representative Content/Samples (from L0 Descendants)" if level > 0 else "Original Content/Data"
                 num_l0_samples_to_show = len(l0_samples)
                 if num_l0_samples_to_show > 0:
                      cluster_block += f"{sample_desc}:\n"
                      for sample_info in l0_samples:
                          s_id = sample_info.get('id', 'N/A')
                          s_data = str(sample_info.get('data', ''))
                          trunc_len = self.prompt_l0_sample_trunc_len
                          display_text = (s_data[:trunc_len] + '...') if len(s_data) > trunc_len else s_data
                          cluster_block += f'- (Orig. ID: {s_id}) "{display_text}"\n'

            child_samples = data.get("immediate_child_samples", [])
            if child_samples:
                 child_sample_desc = "Representative Immediate Children (Summaries)"
                 num_child_samples_to_show = len(child_samples)
                 if num_child_samples_to_show > 0:
                     cluster_block += f"{child_sample_desc}:\n"
                     for child_info in child_samples:
                         c_id = child_info.get('id', 'N/A')
                         c_title = child_info.get('title', '')
                         c_desc = child_info.get('description', '')
                         cluster_block += f'- Child ID {c_id}: "{c_title}" - {c_desc}\n'

            stats = data.get("statistics")
            if stats:
                 stat_desc = "Key Statistics (Original Scale)"
                 cluster_block += f"{stat_desc}:\n"
                 vars_to_show = list(stats.keys())[:self.prompt_max_stats_vars]
                 num_fmt = f"{{:.{self.prompt_numeric_stats_precision}f}}"

                 for var_name in vars_to_show:
                      s_data = stats[var_name]
                      var_meta = variable_metadata.get(var_name) if variable_metadata else None
                      unit_str = f" ({var_meta['unit']})" if var_meta and var_meta.get('unit') else ""
                      desc_str = f" # {var_meta['description']}" if var_meta and var_meta.get('description') else ""
                      is_single_item_or_l0 = (level == 0 or num_underlying == 1)

                      if not is_single_item_or_l0:
                          mean = s_data['mean']
                          min_val = s_data['min']
                          max_val = s_data['max']
                          std = s_data['std']
                          cluster_block += (f"- {var_name}{unit_str}: mean={num_fmt.format(mean)} "
                                            f"(range: {num_fmt.format(min_val)} to {num_fmt.format(max_val)})"
                                            f"{f', std={num_fmt.format(std)}' if std > 1e-9 else ''}"
                                            f"{desc_str}\n")
                      else:
                           cluster_block += f"- {var_name}{unit_str}: {num_fmt.format(s_data['mean'])}{desc_str}\n"

                 if len(stats) > self.prompt_max_stats_vars:
                      cluster_block += "- ... (more variables exist)\n"

            cluster_block += f"--- End {item_type} ID: {item_id_str} ---\n"

            cluster_block_tokens = _estimate_token_count(cluster_block)
            effective_token_limit = self.max_prompt_tokens * self.max_prompt_token_buffer

            if prompt_tokens_estimate + cluster_block_tokens > effective_token_limit:
                 if prompt_items_added == 0:
                      warnings.warn(f"First item ID {item_id_str} for prompt {prompt_id} exceeds token limit ({cluster_block_tokens} estimated). Cannot generate prompt.")
                      if self.save_run_details:
                          log_entry = {
                              "prompt_id": prompt_id, "run_id": self._run_id, "timestamp": timestamp,
                              "level": level, "item_ids_requested": [d["id"] for d in cluster_prompt_data_list],
                              "item_ids_included": [], "item_type": item_type, "topic_seed_used": self._current_topic_seed,
                              "prompt_text": "Prompt too large for first item.", "estimated_tokens": prompt_tokens_estimate + cluster_block_tokens,
                              "token_limit": self.max_prompt_tokens, "llm_response": None, "llm_error": None,
                              "parsed_output": None, "parsing_error": "Prompt generation failed (too large for first item)"}
                          self._prompt_log.append(log_entry)
                      return None, prompt_id, prompt_tokens_estimate + cluster_block_tokens, []
                 else:
                      warnings.warn(f"Prompt token limit ({effective_token_limit:.0f}) likely reached. Stopping additions for prompt {prompt_id}. Included {prompt_items_added}/{n_items_requested} requested items.")
                      break

            prompt += cluster_block
            prompt_tokens_estimate += cluster_block_tokens
            item_ids_included.append(original_id_for_lookup)
            prompt_items_added += 1

        if not item_ids_included:
             self._log(f"Warning: No items could be added to prompt {prompt_id} (e.g., due to token limits).", level=1)
             if self.save_run_details:
                 log_entry = {
                     "prompt_id": prompt_id, "run_id": self._run_id, "timestamp": timestamp,
                     "level": level, "item_ids_requested": [d["id"] for d in cluster_prompt_data_list],
                     "item_ids_included": [], "item_type": item_type, "topic_seed_used": self._current_topic_seed,
                     "prompt_text": "No items fit within token limits.", "estimated_tokens": prompt_tokens_estimate,
                     "token_limit": self.max_prompt_tokens, "llm_response": None, "llm_error": None,
                     "parsed_output": None, "parsing_error": "Prompt generation failed (no items fit)"}
                 self._prompt_log.append(log_entry)
             return None, prompt_id, prompt_tokens_estimate, []

        log_entry = next((p for p in reversed(self._prompt_log) if p.get("prompt_id") == prompt_id), None)
        if not log_entry:
            log_entry = {
                "prompt_id": prompt_id, "run_id": self._run_id, "timestamp": timestamp,
                "level": level, "item_ids_requested": [d["id"] for d in cluster_prompt_data_list],
                "item_ids_included": item_ids_included, "item_type": item_type,
                "topic_seed_used": self._current_topic_seed, "prompt_text": prompt,
                "estimated_tokens": 0, "token_limit": self.max_prompt_tokens,
                "llm_response": None, "llm_error": None, "parsed_output": None, "parsing_error": None
             }
            self._prompt_log.append(log_entry)

        included_ids_str = [str(id_val).replace('\\', '/') for id_val in item_ids_included]
        prompt += f"\n--- End {item_type_plural} Information ---"
        prompt += f"\n\nGenerate the JSON output for the {len(included_ids_str)} {item_type} ID(s): {', '.join(included_ids_str)}"
        final_estimated_tokens = _estimate_token_count(prompt)

        self._log(f"  Built prompt {prompt_id} with {len(item_ids_included)} items, estimated tokens: {final_estimated_tokens}.", level=2)

        if self.save_run_details:
            log_entry["prompt_text"] = prompt
            log_entry["estimated_tokens"] = final_estimated_tokens
            log_entry["item_ids_included"] = item_ids_included

        return prompt, prompt_id, final_estimated_tokens, item_ids_included

    def _save_prompt_log(self):
        """Saves prompt interactions log if enabled."""
        if not self.save_run_details or not self._prompt_log or not self._run_id: return
        log_dir = self.run_details_dir
        json_filename = os.path.join(log_dir, f"prompt_log_{self._run_id}.json")
        txt_filename = os.path.join(log_dir, f"prompt_log_{self._run_id}.txt")
        try:
             os.makedirs(log_dir, exist_ok=True)
             with open(json_filename, 'w', encoding='utf-8') as f:
                 json.dump(self._prompt_log, f, indent=2, default=str)
             with open(txt_filename, 'w', encoding='utf-8') as f:
                  for i, entry in enumerate(self._prompt_log):
                       f.write(f"--- PROMPT {i+1} / ID: {entry.get('prompt_id', 'N/A')} / Level: {entry.get('level', 'N/A')} ---\n")
                       f.write(f"Timestamp: {entry.get('timestamp', 'N/A')}\n")
                       f.write(f"Item Type: {entry.get('item_type', 'N/A')}\n")
                       f.write(f"Topic Seed: {entry.get('topic_seed_used', 'None')}\n")
                       f.write(f"Item IDs Requested: {[str(x) for x in entry.get('item_ids_requested', [])]}\n")
                       f.write(f"Item IDs Included : {[str(x) for x in entry.get('item_ids_included', [])]}\n")
                       f.write(f"Estimated Tokens: {entry.get('estimated_tokens', 'N/A')}\n")
                       f.write(f"Token Limit: {entry.get('token_limit', 'N/A')}\n")
                       f.write(f"\n--- Prompt Text Snippet ---\n{entry.get('prompt_text', '')[:1000]}...\n")
                       f.write(f"\n--- LLM Response Snippet ---\n{str(entry.get('llm_response', ''))[:1000]}...\n")
                       f.write(f"\n--- Status ---\n")
                       f.write(f"LLM Error: {entry.get('llm_error', 'None')}\n")
                       f.write(f"Parsing Error: {entry.get('parsing_error', 'None')}\n")
                       parsed = entry.get("parsed_output")
                       f.write(f"Parsed Count: {len(parsed) if parsed else 0}\n")
                       f.write("="*70 + "\n\n")
             self._log(f"Saved prompt log to {json_filename} (and {txt_filename})", level=1)
        except Exception as e: print(f"Error saving prompt log: {e}")

    def _get_llm_descriptions_batched(self, clusters_to_describe: list[Cluster]) -> tuple[dict[Union[int, str], tuple[str, str]], list[Cluster]]:
        """
        Handles LLM calls with adaptive batching and retries.

        Returns:
            Tuple of (dictionary {id: (title, description)}, list of persistently failed clusters).
        """
        processed_results: dict[Union[int, str], tuple[str, str]] = {}
        persistently_failed: list[Cluster] = []
        if not clusters_to_describe:
            return processed_results, persistently_failed

        level = clusters_to_describe[0].level
        item_type_plural = "clusters" if level > 0 else "items"
        n_total_items = len(clusters_to_describe)
        self._log(f"Generating LLM descriptions for {n_total_items} {item_type_plural} (Level {level})...", level=1)
        self._log(f"  LLM Retry Policy: {self.llm_retries} retries, {self.llm_retry_delay}s base delay.", level=2)

        items_to_process: list[Cluster] = list(clusters_to_describe)
        current_batch_size: int = min(len(items_to_process), self.llm_initial_batch_size)

        while items_to_process and current_batch_size >= self.llm_min_batch_size:
            num_remaining_before_iter = len(items_to_process)
            self._log(f"  Starting LLM batch iteration. Batch size <= {current_batch_size}, {num_remaining_before_iter} items remaining.", level=2)

            failed_ids_in_iteration = set()
            successfully_processed_count_this_iteration = 0

            for i in range(0, len(items_to_process), current_batch_size):
                batch_chunk_indices = range(i, min(i + current_batch_size, len(items_to_process)))
                batch_chunk: list[Cluster] = [items_to_process[idx] for idx in batch_chunk_indices]
                if not batch_chunk: continue

                chunk_ids = [(c.original_id if c.level == 0 else c.id) for c in batch_chunk]
                self._log(f"    Processing chunk {i//current_batch_size + 1}/{ (len(items_to_process) + current_batch_size - 1) // current_batch_size } (size {len(batch_chunk)}, IDs: {chunk_ids[:3]}{'...' if len(chunk_ids)>3 else ''})...", level=3)

                batch_prompt_data = [
                    c.get_data_for_prompt(
                        l0_sample_size=self.prompt_l0_sample_size,
                        l0_sample_strategy=self.prompt_l0_sample_strategy,
                        l0_sample_numeric_repr_max_vals=self.prompt_l0_numeric_repr_max_vals,
                        l0_sample_numeric_repr_precision=self.prompt_l0_numeric_repr_precision,
                        include_immediate_children=self.prompt_include_immediate_children,
                        immediate_child_sample_size=self.prompt_immediate_child_sample_size,
                        immediate_child_sample_strategy=self.prompt_immediate_child_sample_strategy,
                        variable_names=self.variable_names,
                        variable_metadata_by_name=self.numeric_metadata_by_name,
                        numeric_stats_precision=self.cluster_numeric_stats_precision,
                        max_stats_vars=self.prompt_max_stats_vars,
                        child_sample_desc_trunc_len=self.prompt_child_sample_desc_trunc_len
                    ) for c in batch_chunk
                ]
                prompt_str, prompt_id, estimated_tokens, included_ids_original_type = self._build_llm_prompt(batch_prompt_data)
                prompt_log_entry = next((p for p in reversed(self._prompt_log) if p.get("prompt_id") == prompt_id), None)

                id_to_cluster_map = {(c.original_id if c.level == 0 else c.id): c for c in batch_chunk}
                clusters_in_prompt = [id_to_cluster_map[cid] for cid in included_ids_original_type if cid in id_to_cluster_map]
                clusters_omitted_from_prompt = [c for c in batch_chunk if (c.original_id if c.level==0 else c.id) not in included_ids_original_type]

                if clusters_omitted_from_prompt:
                    omitted_ids = [(c.original_id if c.level==0 else c.id) for c in clusters_omitted_from_prompt]
                    self._log(f"    Prompt {prompt_id} omitted {len(clusters_omitted_from_prompt)} items due to token limits during build (IDs: {omitted_ids}). Marking as failed for this iteration.", level=2)
                    failed_ids_in_iteration.update(omitted_ids)

                if prompt_str is None or not clusters_in_prompt:
                    self._log(f"    Skipping chunk starting at index {i}: Prompt generation failed or no items included.", level=2)
                    failed_ids_in_iteration.update([(c.original_id if c.level == 0 else c.id) for c in clusters_in_prompt])
                    if prompt_log_entry and prompt_log_entry.get("parsing_error") is None:
                        prompt_log_entry["parsing_error"] = "Prompt generation failed or empty"
                    continue

                if estimated_tokens > self.max_prompt_tokens * self.max_prompt_token_buffer:
                    self._log(f"    Prompt {prompt_id} likely too long ({estimated_tokens} estimated > {self.max_prompt_tokens * self.max_prompt_token_buffer:.0f} limit). Marking items as failed for this iteration.", level=2)
                    failed_ids_in_iteration.update([(c.original_id if c.level == 0 else c.id) for c in clusters_in_prompt])
                    if prompt_log_entry: prompt_log_entry["parsing_error"] = "Prompt likely too large (post-build check)"
                    continue

                llm_output = None
                parsed_data = None
                chunk_succeeded = False
                final_llm_error_str = None
                final_parse_error_str = None

                for attempt in range(self.llm_retries + 1):
                    llm_error_str = None
                    parse_error_str = None
                    self._log(f"      Attempt {attempt + 1}/{self.llm_retries + 1} for prompt {prompt_id} ({len(included_ids_original_type)} items)...", level=2 if attempt > 0 else 3)

                    try:
                        llm_output = self.llm_client(prompt_str)
                        if prompt_log_entry: prompt_log_entry["llm_response"] = llm_output
                    except Exception as e:
                        llm_error_str = str(e)
                        self._log(f"      LLM call failed on attempt {attempt + 1}: {e}", level=2)
                        if prompt_log_entry: prompt_log_entry["llm_error"] = llm_error_str

                    if llm_output is not None and llm_error_str is None:
                        self._log(f"      Parsing response for prompt {prompt_id} (attempt {attempt+1})...", level=3)
                        parsed_data = _parse_llm_response(llm_output, expected_ids=included_ids_original_type)

                        if parsed_data is not None:
                            parsed_count = len(parsed_data)
                            processed_results.update(parsed_data)
                            if prompt_log_entry:
                                prompt_log_entry["parsed_output"] = parsed_data
                                prompt_log_entry["llm_error"] = None

                            successfully_parsed_ids = set(parsed_data.keys())
                            succeeded_in_prompt = [c for c in clusters_in_prompt if (c.original_id if c.level == 0 else c.id) in successfully_parsed_ids]
                            missing_in_response = [c for c in clusters_in_prompt if (c.original_id if c.level == 0 else c.id) not in successfully_parsed_ids]

                            successfully_processed_count_this_iteration += len(succeeded_in_prompt)
                            failed_ids_in_iteration.update([(c.original_id if c.level == 0 else c.id) for c in missing_in_response])

                            if missing_in_response:
                                missing_ids_str = [(c.original_id if c.level == 0 else c.id) for c in missing_in_response]
                                parse_error_str = f"Partial results ({parsed_count}/{len(included_ids_original_type)}) - Missing: {missing_ids_str[:5]}{'...' if len(missing_ids_str)>5 else ''}"
                                self._log(f"      {parse_error_str}", level=2)
                                if prompt_log_entry: prompt_log_entry["parsing_error"] = parse_error_str
                            else:
                                self._log(f"      Successfully parsed {parsed_count}/{len(included_ids_original_type)} descriptions from response {prompt_id} on attempt {attempt+1}.", level=2)
                                if prompt_log_entry: prompt_log_entry["parsing_error"] = False

                            chunk_succeeded = True
                            break

                        else:
                            parse_error_str = "Response parsing failed"
                            self._log(f"      {parse_error_str} on attempt {attempt+1}.", level=2)
                            if prompt_log_entry and prompt_log_entry.get("parsing_error") is None:
                                prompt_log_entry["parsing_error"] = parse_error_str

                    final_llm_error_str = llm_error_str
                    final_parse_error_str = parse_error_str

                    if attempt < self.llm_retries:
                        delay = self.llm_retry_delay * (attempt + 1)
                        self._log(f"      Waiting {delay:.1f}s before next retry...", level=2)
                        time.sleep(delay)
                    else:
                        self._log(f"      Chunk failed after {self.llm_retries + 1} attempts.", level=1)

                if not chunk_succeeded:
                    chunk_prompt_ids = [(c.original_id if c.level == 0 else c.id) for c in clusters_in_prompt]
                    self._log(f"    Marking {len(chunk_prompt_ids)} items from prompt {prompt_id} as failed for this iteration after retries.", level=2)
                    failed_ids_in_iteration.update(chunk_prompt_ids)
                    if prompt_log_entry:
                         if prompt_log_entry.get("llm_error") is None and final_llm_error_str: prompt_log_entry["llm_error"] = f"Failed after retries: {final_llm_error_str}"
                         if prompt_log_entry.get("parsing_error") is None and final_parse_error_str: prompt_log_entry["parsing_error"] = f"Failed after retries: {final_parse_error_str}"
                         elif prompt_log_entry.get("parsing_error") is None: prompt_log_entry["parsing_error"] = "Failed after retries (unknown reason)"


            items_to_process = [item for item in items_to_process if (item.original_id if item.level == 0 else item.id) not in processed_results]
            num_remaining_after_iter = len(items_to_process)

            self._log(f"  Finished LLM batch iteration. Processed in iter: {successfully_processed_count_this_iteration}, Total Processed: {len(processed_results)}, Iteration Failures: {len(failed_ids_in_iteration)}, Remaining: {num_remaining_after_iter}", level=2)

            if successfully_processed_count_this_iteration == 0 and num_remaining_after_iter > 0:
                 if current_batch_size <= self.llm_min_batch_size:
                     warnings.warn(f"Could not get descriptions for {num_remaining_after_iter} {item_type_plural} even at minimum batch size ({current_batch_size}) after retries. Marking as persistent failures.")
                     persistently_failed.extend(items_to_process)
                     items_to_process = []
                     break
                 else:
                     self._log(f"  No items successfully processed in this iteration (batch size {current_batch_size}). Reducing batch size.", level=2)
                     new_batch_size = max(self.llm_min_batch_size, int(current_batch_size / self.llm_batch_reduction_factor))
                     if new_batch_size == current_batch_size and current_batch_size > self.llm_min_batch_size:
                         new_batch_size -= 1
                     if new_batch_size < current_batch_size:
                         self._log(f"  Reducing batch size from {current_batch_size} to {new_batch_size}.", level=2)
                         current_batch_size = new_batch_size
                     else:
                          warnings.warn(f"Batch size reduction logic failed to reduce size from {current_batch_size}. Setting to minimum {self.llm_min_batch_size} to avoid infinite loop.")
                          current_batch_size = self.llm_min_batch_size

        if items_to_process:
             failed_ids = [(c.original_id if c.level == 0 else c.id) for c in items_to_process]
             warnings.warn(f"Processing loop finished unexpectedly with {len(failed_ids)} items remaining unprocessed: {failed_ids}. Marking as persistent failures.")
             persistently_failed.extend(items_to_process)

        final_failed_ids = set((c.original_id if c.level == 0 else c.id) for c in persistently_failed)
        default_assigned_count = 0
        for item in clusters_to_describe:
             item_key = item.original_id if item.level == 0 else item.id
             if item_key not in processed_results and item_key in final_failed_ids:
                  item_type_label = "Cluster" if item.level > 0 else "Item"
                  if item.title is None or not item.title.strip() or item.title.startswith(f"{item_type_label} {item_key} (LLM Failed)"):
                      item.title = f"{item_type_label} {item_key} (LLM Failed)"
                  item.description = "LLM description generation or parsing failed after retries."
                  default_assigned_count += 1

        if default_assigned_count > 0:
            self._log(f"Assigned default descriptions to {default_assigned_count} persistently failed items.", level=1)

        final_success_count = len(processed_results)
        final_fail_count = len(persistently_failed)

        self._log(f"LLM description generation complete. Success: {final_success_count}/{n_total_items}, Persistently Failed (default assigned): {final_fail_count}/{n_total_items}", level=1)

        return processed_results, persistently_failed

    def _initialize_l0_clusters(self, standardized_data: list | np.ndarray, original_ids: list) -> list[Cluster]:
        """Creates Level 0 Cluster objects and assigns raw/numeric data."""
        self._log(f"Initializing {len(original_ids)} Level 0 Cluster objects...", level=2)
        l0_clusters = []
        for i, orig_id in enumerate(original_ids):
            raw_data_item = None
            if self.input_data_type == "numeric":
                raw_data_item = self.original_numeric_data_[i] if self.original_numeric_data_ is not None else None
            else:
                raw_data_item = standardized_data[i]

            cluster = Cluster(level=0, original_data_type=self.input_data_type,
                              original_id=orig_id, raw_item_data=raw_data_item)
            l0_clusters.append(cluster)
            self._all_clusters_map[cluster.id] = cluster

            if self.input_data_type == 'numeric' and cluster._raw_item_data is not None:
                 try:
                      cluster.original_numeric_data = np.asarray(cluster._raw_item_data).reshape(1, -1)
                 except Exception as e:
                      warnings.warn(f"Could not reshape raw numeric data for L0 cluster {orig_id}: {e}")
                      cluster.original_numeric_data = None

        return l0_clusters

    def _generate_l0_titles_descriptions(self, l0_clusters: list[Cluster], standardized_data: list | np.ndarray) -> list[str | None]:
        """
        Generates titles and descriptions for L0 clusters based on config and numeric metadata.
        Returns list of combined "Title. Description" strings.
        """
        self._log("Generating Level 0 titles and descriptions...", level=2)
        if self.use_llm_for_l0_descriptions and self.input_data_type in ['text', 'numeric']:
             self._log("  Mode: LLM will be used for L0 Text/Numeric descriptions.", level=2)
        elif self.input_data_type == 'image' and self.image_captioning_client:
             self._log("  Mode: Image Captioning Client will be used for L0 Image descriptions.", level=2)
        else: self._log("  Mode: LLM-free L0 Text/Numeric/Image descriptions (using snippets/defaults).", level=2)

        l0_combined_descriptions = [None] * len(l0_clusters)
        items_needing_llm_description = []

        self._log("  Assigning initial L0 titles based on content extraction/type...", level=2)
        for i, cluster in enumerate(l0_clusters):
            cluster.title = self._extract_l0_title(cluster)
            cluster.description = f"{cluster.original_data_type.capitalize()} data for item {cluster.original_id}."

        if self.input_data_type == 'image':
            if self.image_captioning_client:
                self._log(f"  Attempting IMAGE captioning for {len(standardized_data)} items...", level=2)
                try:
                    image_identifiers = [c._raw_item_data for c in l0_clusters]
                    image_captions = self.image_captioning_client(image_identifiers, prompt=self._current_topic_seed)
                    if len(image_captions) != len(l0_clusters): raise ValueError("Captioning client returned incorrect number of captions.")
                    captions_generated = 0
                    for i, cluster in enumerate(l0_clusters):
                        caption = image_captions[i]
                        if isinstance(caption, str) and caption.strip():
                            cluster.description = caption.strip()
                            captions_generated += 1
                        
                            max_title_words = 7
                            max_title_chars = 50
                        
                            match = re.match(r"([^.!?]+[.!?])", cluster.description)
                            if match:
                                new_title = match.group(1).strip()
                            else:
                                words = cluster.description.split()
                                new_title = " ".join(words[:max_title_words])
                        
                            if len(new_title) > max_title_chars:
                                new_title = new_title[:max_title_chars].strip() + "..."
                            
                            if len(new_title.strip()) > 3:
                                cluster.title = new_title.strip()

                        else:
                            cluster.description = "(Captioning failed or empty)"
                            warnings.warn(f"Empty or invalid caption received for image {cluster.original_id}")
                    self._log(f"  Generated {captions_generated}/{len(l0_clusters)} image captions.", level=2)
                except Exception as e:
                    warnings.warn(f"Image captioning client failed: {e}. Using default descriptions for images.")
                    for cluster in l0_clusters: cluster.description = f"Image Item {cluster.original_id}"
            else:
                self._log("  No image captioning client provided. Using default descriptions for images.", level=2)
                for cluster in l0_clusters: cluster.description = f"Image Item {cluster.original_id}"

        llm_needed_for_text_numeric = self.use_llm_for_l0_descriptions and self.input_data_type in ['text', 'numeric']

        if llm_needed_for_text_numeric:
            items_needing_llm_description = [c for c in l0_clusters if c.original_data_type in ['text', 'numeric']]
            if items_needing_llm_description:
                self._log(f"  Generating LLM descriptions for {len(items_needing_llm_description)} L0 {self.input_data_type} items...", level=2)
                processed_llm_results, _ = self._get_llm_descriptions_batched(items_needing_llm_description)
                for cluster in items_needing_llm_description:
                    key = cluster.original_id
                    if key in processed_llm_results:
                        cluster.title, cluster.description = processed_llm_results[key]

        else:
            if self.input_data_type == 'text':
                self._log("  Generating LLM-free L0 descriptions for TEXT items (using snippets).", level=2)
                trunc_len = self.direct_l0_text_trunc_len
                for cluster in l0_clusters:
                    if cluster.original_data_type == 'text':
                        original_text = str(cluster._raw_item_data or "")
                        desc = original_text.strip()
                        cluster.description = (desc[:trunc_len] + "..." if len(desc) > trunc_len
                                               else desc or "(Original text empty)")
            elif self.input_data_type == 'numeric':
                self._log("  Generating LLM-free L0 descriptions for NUMERIC items (structured values).", level=2)
                max_vals = self.prompt_l0_numeric_repr_max_vals
                precision = self.prompt_l0_numeric_repr_precision
                num_fmt = f"{{:.{precision}f}}"
                for cluster in l0_clusters:
                     if cluster.original_data_type == 'numeric':
                        numeric_data = cluster._raw_item_data
                        desc_parts = []
                        if isinstance(numeric_data, np.ndarray):
                            values = numeric_data.flatten()
                            num_features = len(values)
                            num_features_to_show = min(num_features, max_vals) if max_vals > 0 else num_features

                            for idx in range(num_features_to_show):
                                var_name = self.variable_names[idx] if self.variable_names and idx < len(self.variable_names) else f"val_{idx+1}"
                                var_meta = self.numeric_metadata_by_name.get(var_name) if self.numeric_metadata_by_name else None
                                unit_str = f" {var_meta['unit']}" if var_meta and var_meta.get('unit') else ""
                                desc_parts.append(f"{var_name}={num_fmt.format(values[idx])}{unit_str}")

                            ellipsis = "..." if (max_vals > 0 and num_features > max_vals) else ""
                            cluster.description = f"Values: {', '.join(desc_parts)}{ellipsis}" if desc_parts else "(No values)"
                        else:
                            cluster.description = "(Numeric data unavailable)"

        for i, cluster in enumerate(l0_clusters):
             title = cluster.title if cluster.title else f"{cluster.original_data_type.capitalize()} {cluster.original_id}"
             desc = cluster.description if cluster.description else "(No description available)"
             combined = f"{title}. {desc}"
             combined = combined.replace("..", ".").replace(". .", ".").strip()
             if title.endswith(('.', '!', '?')) and desc:
                 combined = f"{title} {desc}"

             if not combined:
                 warnings.warn(f"L0 item {cluster.original_id} resulted in empty combined description after processing. Using fallback.")
                 l0_combined_descriptions[i] = f"{title}. (Description generation failed or was empty)"
             else:
                 l0_combined_descriptions[i] = combined

        return l0_combined_descriptions

    def _generate_l0_representations(self, l0_clusters: list[Cluster], standardized_data: list | np.ndarray) -> bool:
        """
        Generates L0 vectors (embeddings or scaled numerics) based on representation mode
        and trains relevant reducers.
        Returns True on success.
        """
        self._log("Generating Level 0 representations for clustering...", level=2)
        l0_vectors = None
        l0_representation_space = None

        if self.representation_mode == 'direct':
            self._log(f"  Mode: Direct Representation (using scaled numerics or item embeddings for L0).", level=2)
            if self.input_data_type == 'numeric':
                l0_vectors = standardized_data
                if not isinstance(l0_vectors, np.ndarray):
                    print("Error: Expected standardized_data to be a numpy array for numeric direct mode."); return False
                l0_representation_space = 'numeric'
                self._log(f"    Using scaled numeric data as L0 representation (space: {l0_representation_space}).", level=2)
            elif self.input_data_type == 'text':
                try:
                    original_texts_for_embedding = []
                    empty_text_indices_info = []
                    for i, c in enumerate(l0_clusters):
                        text_content = str(c._raw_item_data or "").strip()
                        if not text_content:
                            original_texts_for_embedding.append(Hercules.EMPTY_TEXT_PLACEHOLDER)
                            empty_text_indices_info.append((i, c.original_id))
                        else:
                            original_texts_for_embedding.append(text_content)

                    if empty_text_indices_info:
                        example_ids = [info[1] for info in empty_text_indices_info[:5]]
                        warnings.warn(
                            f"Found {len(empty_text_indices_info)} empty L0 text items for 'direct' representation. "
                            f"Using placeholder '{Hercules.EMPTY_TEXT_PLACEHOLDER}' for embedding. "
                            f"Example Original IDs: {example_ids}{'...' if len(empty_text_indices_info)>5 else ''}"
                        )
                
                    self._log(f"    Generating TEXT embeddings for {len(original_texts_for_embedding)} items...", level=2)
                    if not original_texts_for_embedding and l0_clusters:
                         print("Error: No texts (not even placeholders) to embed for L0 direct mode, though L0 clusters exist."); return False
                    if not original_texts_for_embedding and not l0_clusters:
                        self._log("    No L0 items to generate text embeddings for.", level=2)
                        l0_vectors = np.empty((0,0))
                    else:
                        l0_vectors = self.text_embedding_client(original_texts_for_embedding)
                    l0_representation_space = 'text_embedding'
                except Exception as e: print(f"Error: Text embedding client failed for L0: {e}"); return False
            elif self.input_data_type == 'image':
                if not self.image_embedding_client: print("Error: `image_embedding_client` is required for 'direct' mode with image data."); return False
                try:
                    image_identifiers = [c._raw_item_data for c in l0_clusters]
                    self._log(f"    Generating IMAGE embeddings for {len(image_identifiers)} items...", level=2)
                    l0_vectors = self.image_embedding_client(image_identifiers)
                    l0_representation_space = 'image_embedding'
                except Exception as e: print(f"Error: Image embedding client failed for L0: {e}"); return False

            if l0_vectors is None or (l0_clusters and l0_vectors.shape[0] != len(l0_clusters)):
                 print(f"Error: Failed to generate initial vectors for direct mode ({self.input_data_type}). Shape mismatch or None."); return False
            if not np.all(np.isfinite(l0_vectors)):
                 num_bad = np.sum(~np.isfinite(l0_vectors))
                 warnings.warn(f"Generated L0 vectors for direct mode ({self.input_data_type}) contain {num_bad} non-finite values. Attempting nan_to_num.")
                 l0_vectors = np.nan_to_num(l0_vectors, nan=0.0, posinf=0.0, neginf=0.0)
                 if not np.all(np.isfinite(l0_vectors)): print("Error: Non-finite values persist in L0 vectors after nan_to_num. Cannot proceed."); return False

            if l0_vectors.shape[0] > 0 :
                self._log(f"    Assigning direct L0 representation vectors (space: {l0_representation_space}, dim: {l0_vectors.shape[1]})...", level=2)
                if l0_vectors.shape[1] > 0:
                    self.embedding_dims_[l0_representation_space] = l0_vectors.shape[1]
                    self._train_reducers(l0_vectors, l0_representation_space)
                else:
                     self._log(f"    Warning: Generated L0 representation vectors have 0 dimension ({l0_representation_space}). Skipping reducer training.", level=1)

                for i, cluster in enumerate(l0_clusters):
                     cluster.representation_vector = l0_vectors[i]
                     cluster.representation_vector_space = l0_representation_space
            elif l0_clusters:
                self._log(f"    Warning: No L0 representation vectors generated for {len(l0_clusters)} items in direct mode. They will not have representation vectors.", level=1)


        elif self.representation_mode == 'description':
            self._log(f"  Mode: Description Representation (using embeddings of L0 descriptions/captions).", level=2)
            l0_representation_space = 'text_embedding'
            self._log(f"    L0 representation vectors will be assigned from description embeddings (space: {l0_representation_space}).", level=2)

        return True

    def _apply_l0_reductions(self, l0_clusters: list[Cluster]):
        """Applies dimensionality reduction (PCA) to L0 representations if configured."""
        if not self.reduction_methods: return
        self._log("Applying dimensionality reduction to L0 representations...", level=2)
        processed_count_rep = 0
        processed_count_desc = 0

        for cluster in l0_clusters:
            if cluster.representation_vector is not None and cluster.representation_vector_space is not None:
                 reductions = self._apply_reduction(cluster.representation_vector, cluster.representation_vector_space)
                 if reductions:
                     cluster.representation_vector_reductions = reductions
                     processed_count_rep +=1

            if cluster.description_embedding is not None and cluster.representation_vector is not cluster.description_embedding:
                 desc_reductions = self._apply_reduction(cluster.description_embedding, 'text_embedding')
                 if desc_reductions:
                     cluster.description_embedding_reductions = desc_reductions
                     processed_count_desc += 1

        total_processed = processed_count_rep + processed_count_desc
        self._log(f"Applied reductions to {total_processed} L0 vector instance(s) across {len(l0_clusters)} clusters.", level=2)


    def _extract_l0_title(self, cluster: Cluster, max_title_words: int = 7, max_title_chars: int = 50) -> str:
        """Attempts to extract a concise title for an L0 cluster."""
        fallback_prefix = f"{cluster.original_data_type.capitalize()} Item"
        fallback_title = f"{fallback_prefix} {cluster.original_id}"
        raw_data = cluster._raw_item_data

        try:
            if cluster.original_data_type == 'text':
                if isinstance(raw_data, str) and raw_data.strip():
                    text = raw_data.strip()
                    match = re.match(r"([^.!?]+[.!?])", text)
                    if match: title = match.group(1).strip()
                    else: words = text.split(); title = " ".join(words[:max_title_words])
                    if len(title) > max_title_chars: title = title[:max_title_chars].strip() + "..."
                    return title if len(title) > 3 else fallback_title
                else: return fallback_title
            elif cluster.original_data_type == 'image':
                if PIL_IMAGE_TYPE != type(None) and isinstance(raw_data, PIL_IMAGE_TYPE):
                    title_str = f"PIL Image ({getattr(raw_data, 'format', 'N/A')} {getattr(raw_data, 'width', 'W')}x{getattr(raw_data, 'height', 'H')})"
                    return title_str[:max_title_chars + 15]
                elif isinstance(raw_data, str):
                    cleaned_identifier = raw_data
                    prefix = "Image"
                    if _is_string_url(raw_data):
                        prefix = "Image URL"
                        try:
                            parsed_url = urlparse(raw_data)
                            cleaned_identifier = os.path.basename(parsed_url.path)
                        except Exception: pass
                    else:
                        prefix = "Image File"
                        cleaned_identifier = os.path.basename(raw_data)

                    try:
                        stem = Path(cleaned_identifier).stem
                        title = stem.replace('_', ' ').replace('-', ' ')
                        if not title.strip() and cleaned_identifier.strip():
                            title = Path(cleaned_identifier).name
                        
                        if len(title) > max_title_chars: title = title[:max_title_chars].strip() + "..."
                        return f"{prefix}: {title}" if title.strip() else fallback_title
                    except Exception: return fallback_title
                return fallback_title
            elif cluster.original_data_type == 'numeric':
                 if self.variable_names:
                      return f"Numeric: ID {cluster.original_id}"[:max_title_chars+15]
                 else: return fallback_title
            else: return fallback_title
        except Exception: return fallback_title

    def _perform_hierarchical_clustering_loop(self, l0_clusters: list[Cluster]) -> list[Cluster]:
        """
        Performs the iterative hierarchical clustering (Levels 1+).
        """
        current_clusters = l0_clusters
        max_level_iterations = len(self.level_cluster_counts) if self.level_cluster_counts is not None else 100
        last_successful_level = 0
        auto_k_mode = (self.level_cluster_counts is None)

        if auto_k_mode: self._log(f"Automatic K determination enabled (Method: {self.auto_k_method}, Max K per level: {self.auto_k_max})", level=1)
        else: self._log(f"Using fixed level cluster counts: {self.level_cluster_counts}", level=1)

        for level_idx in range(max_level_iterations):
            current_level_num = level_idx + 1
            self._log(f"\n--- Clustering Level {current_level_num} ---", level=1)

            clustering_basis_space: str | None = None
            vectors_for_kmeans = []
            valid_prev_level_clusters = []

            if self.representation_mode == 'direct':
                valid_clusters = [c for c in current_clusters if c.representation_vector is not None and c.representation_vector_space is not None]
                if valid_clusters:
                     clustering_basis_space = valid_clusters[0].representation_vector_space
                     valid_clusters = [c for c in valid_clusters if c.representation_vector_space == clustering_basis_space]
                     vectors_for_kmeans = [c.representation_vector for c in valid_clusters]
                     valid_prev_level_clusters = valid_clusters
                     self._log(f"Level {current_level_num}: Clustering {len(valid_prev_level_clusters)} prev level REPRESENTATION VECTORS (space: {clustering_basis_space}).", level=2)
                else: self._log(f"Level {current_level_num}: No valid representation vectors found from previous level for direct mode clustering.", level=1)

            elif self.representation_mode == 'description':
                valid_clusters = [c for c in current_clusters if c.description_embedding is not None]
                if valid_clusters:
                    clustering_basis_space = 'text_embedding'
                    vectors_for_kmeans = [c.description_embedding for c in valid_clusters]
                    valid_prev_level_clusters = valid_clusters
                    self._log(f"Level {current_level_num}: Clustering {len(valid_prev_level_clusters)} prev level DESCRIPTION EMBEDDINGS.", level=2)
                else: self._log(f"Level {current_level_num}: No valid description embeddings found from previous level for description mode clustering.", level=1)

            if not valid_prev_level_clusters or not vectors_for_kmeans:
                self._log("No valid clusters/vectors from previous level to cluster. Stopping hierarchy.", level=1); break
            n_items_to_cluster = len(valid_prev_level_clusters)
            if n_items_to_cluster < self.min_clusters_per_level:
                self._log(f"Only {n_items_to_cluster} valid cluster(s) remain (min required: {self.min_clusters_per_level}). Stopping hierarchy.", level=1); break

            try:
                 level_input_vectors = np.array(vectors_for_kmeans).astype(np.float32)
                 if level_input_vectors.ndim != 2: raise ValueError("Vectors are not 2D.")
                 if level_input_vectors.shape[0] != n_items_to_cluster: raise ValueError("Shape mismatch.")
                 if not np.all(np.isfinite(level_input_vectors)):
                      num_bad = np.sum(~np.isfinite(level_input_vectors))
                      warnings.warn(f"Input vectors for KMeans L{current_level_num} contain {num_bad} non-finite values. Attempting nan_to_num.")
                      level_input_vectors = np.nan_to_num(level_input_vectors, nan=0.0, posinf=0.0, neginf=0.0)
                      if not np.all(np.isfinite(level_input_vectors)): raise ValueError("Non-finite values persist after cleanup.")
            except ValueError as e: print(f"Error preparing vectors for KMeans L{current_level_num}: {e}. Stopping."); break
            except Exception as e: print(f"Unexpected error preparing vectors for KMeans L{current_level_num}: {e}. Stopping."); break

            expected_dim = self.embedding_dims_.get(clustering_basis_space)
            if expected_dim is not None and expected_dim != level_input_vectors.shape[1]:
                 print(f"Error: Dimension mismatch for L{current_level_num} clustering. Expected {expected_dim}, got {level_input_vectors.shape[1]} (space: {clustering_basis_space}). Stopping."); break
            elif expected_dim is None and level_input_vectors.shape[1] > 0:
                 self.embedding_dims_[clustering_basis_space] = level_input_vectors.shape[1]
                 self._log(f"Note: Clustering space '{clustering_basis_space}' dimension set to {level_input_vectors.shape[1]} at Level {current_level_num}.", level=2)


            k = -1
            if auto_k_mode:
                k = self._determine_optimal_k(vectors=level_input_vectors, min_k=self.min_clusters_per_level, max_k=self.auto_k_max, method=self.auto_k_method, space_type=clustering_basis_space)
            else:
                if level_idx >= len(self.level_cluster_counts): target_k = self.fallback_k; warnings.warn(f"Level index {level_idx} exceeds config length ({len(self.level_cluster_counts)}). Using fallback k={target_k}.")
                else: target_k = self.level_cluster_counts[level_idx]
                k = max(self.min_clusters_per_level, int(target_k))
                if n_items_to_cluster > 1: k = min(k, n_items_to_cluster - 1)
                k = max(1, k)

            if k < 1: self._log(f"Determined k={k} is invalid (< 1). Stopping hierarchy.", level=1); break
            if k >= n_items_to_cluster and n_items_to_cluster > 0: k = n_items_to_cluster - 1; self._log(f"Adjusting k to {k} (must be < n_items={n_items_to_cluster}).", level=1)
            if k < 1: self._log(f"Adjusted k={k} is invalid (< 1). Stopping hierarchy.", level=1); break
            if k == 1 and n_items_to_cluster > 1: self._log(f"Determined k=1 for {n_items_to_cluster} items. Stopping hierarchy at Level {current_level_num-1}.", level=1); break
            
            self._log(f"    Performing initial KMeans (k={k}) on input I (shape: {level_input_vectors.shape}, space: {clustering_basis_space}) for Level {current_level_num}...", level=2)
            
            if not np.all(np.isfinite(level_input_vectors)):
                warnings.warn(f"Input I for L{current_level_num} initial KMeans has non-finite values. Cleaning.")
                level_input_vectors_cleaned = np.nan_to_num(level_input_vectors, nan=0.0, posinf=0.0, neginf=0.0)
                if not np.all(np.isfinite(level_input_vectors_cleaned)):
                    print(f"Error: Non-finite values persist in input I for L{current_level_num} after cleaning. Stopping."); break
            else:
                level_input_vectors_cleaned = level_input_vectors

            try:
                initial_kmeans = KMeans(n_clusters=k, random_state=self.random_state, 
                                        n_init='auto' if 'auto' in KMeans().get_params() else 10)
                current_labels = initial_kmeans.fit_predict(level_input_vectors_cleaned)
                current_centroids = initial_kmeans.cluster_centers_
                
                if not np.all(np.isfinite(current_centroids)):
                    warnings.warn(f"Initial KMeans L{current_level_num} produced non-finite centroids. Cleaning.")
                    current_centroids = np.nan_to_num(current_centroids, nan=0.0, posinf=0.0, neginf=0.0)
                
                self._log(f"    Initial KMeans for L{current_level_num} complete. Centroids shape: {current_centroids.shape}, Labels count: {len(current_labels)}", level=2)
            
            except Exception as e_initial_kmeans:
                print(f"Error during initial KMeans for L{current_level_num}: {e_initial_kmeans}. Stopping hierarchy."); break

            if self.use_resampling and self.resampling_points_per_cluster > 0 and self.resampling_iterations > 0 and k > 0:
                self._log(f"  Starting resampling iterations (m={self.resampling_iterations}, rt={self.resampling_points_per_cluster}) for Level {current_level_num}...", level=2)
                
                for s_iter in range(self.resampling_iterations):
                    self._log(f"    Resampling iteration {s_iter + 1}/{self.resampling_iterations}...", level=3)
                    resampled_vectors_R_list = []
                    
                    for cluster_idx in range(k):
                        points_in_this_cluster_mask = (current_labels == cluster_idx)
                        actual_points_vector_subset_I = level_input_vectors_cleaned[points_in_this_cluster_mask]
                        
                        if actual_points_vector_subset_I.shape[0] == 0:
                            continue

                        centroid_of_this_cluster = current_centroids[cluster_idx].reshape(1, -1)
                        
                        try:
                            distances_to_centroid = np.linalg.norm(actual_points_vector_subset_I - centroid_of_this_cluster, axis=1)
                            num_to_sample_rt = min(self.resampling_points_per_cluster, actual_points_vector_subset_I.shape[0])
                            closest_indices_in_subset = np.argsort(distances_to_centroid)[:num_to_sample_rt]
                            
                            resampled_vectors_R_list.extend(actual_points_vector_subset_I[closest_indices_in_subset])
                        except Exception as e_resample_dist:
                            warnings.warn(f"Error during distance calculation for resampling (L{current_level_num}, cluster {cluster_idx}, iter {s_iter+1}): {e_resample_dist}")
                            continue
                            
                    if not resampled_vectors_R_list or len(resampled_vectors_R_list) < k:
                        self._log(f"    Not enough points collected for R ({len(resampled_vectors_R_list)}, need >= {k}) in iter {s_iter+1}. Stopping resampling iterations for this level.", level=2)
                        break

                    resampled_vectors_R_np = np.array(resampled_vectors_R_list).astype(np.float32)
                    self._log(f"      Collected {resampled_vectors_R_np.shape[0]} points for R.", level=3)

                    try:
                        self._log(f"      Running KMeans on R (shape {resampled_vectors_R_np.shape}, k={k}) to update C_t...", level=3)
                        kmeans_on_R = KMeans(n_clusters=k, random_state=self.random_state, 
                                             n_init='auto' if 'auto' in KMeans().get_params() else 10)
                        kmeans_on_R.fit(resampled_vectors_R_np)
                        updated_centroids = kmeans_on_R.cluster_centers_
                        
                        if not np.all(np.isfinite(updated_centroids)):
                            warnings.warn(f"KMeans on R (L{current_level_num}, iter {s_iter+1}) produced non-finite centroids. Cleaning.")
                            updated_centroids = np.nan_to_num(updated_centroids, nan=0.0, posinf=0.0, neginf=0.0)
                        current_centroids = updated_centroids
                    except Exception as e_kmeans_R:
                        warnings.warn(f"Error during KMeans on R (L{current_level_num}, iter {s_iter+1}): {e_kmeans_R}. Using centroids from previous iteration/initial step.")
                        break

                    try:
                        self._log(f"      Re-assigning input I (shape {level_input_vectors_cleaned.shape}) to updated C_t (shape {current_centroids.shape}) to update L_t...", level=3)
                        distances_I_to_new_Ct = euclidean_distances(level_input_vectors_cleaned, current_centroids)
                        updated_labels = np.argmin(distances_I_to_new_Ct, axis=1)
                        current_labels = updated_labels
                        self._log(f"      Re-assignment complete. Iter {s_iter+1} for L{current_level_num} finished.", level=3)
                    except Exception as e_reassign:
                        warnings.warn(f"Error during re-assignment of I to new C_t (L{current_level_num}, iter {s_iter+1}): {e_reassign}. Using labels from previous iteration/initial step.")
                        break
            else:
                self._log(f"  Resampling not applied for Level {current_level_num} (use_resampling={self.use_resampling}, rt={self.resampling_points_per_cluster}, m={self.resampling_iterations}, k={k}).", level=2)

            level_rep_vectors = current_centroids
            labels = current_labels
            
            self._log(f"Creating {k} new parent clusters for Level {current_level_num} based on final assignments...", level=2)
            new_parent_clusters_map: dict[int, Cluster] = {} 
            next_level_clusters: list[Cluster] = []
            
            if not valid_prev_level_clusters and len(labels) > 0:
                print(f"Error: Labels generated but no valid_prev_level_clusters (items from C_t-1) to assign them to at L{current_level_num}. Stopping."); break
            
            parent_data_type = valid_prev_level_clusters[0].original_data_type if valid_prev_level_clusters else self.input_data_type

            if len(labels) != len(valid_prev_level_clusters):
                print(f"Error: Mismatch between number of labels ({len(labels)}) and number of items from previous level ({len(valid_prev_level_clusters)}) at L{current_level_num}. Stopping."); break

            for i, label_val in enumerate(labels):
                child_cluster_from_prev_level = valid_prev_level_clusters[i]
                
                if label_val not in new_parent_clusters_map:
                    parent_cluster_level_t = Cluster(level=current_level_num, original_data_type=parent_data_type)
                    parent_cluster_level_t.representation_vector = level_rep_vectors[label_val]
                    parent_cluster_level_t.representation_vector_space = clustering_basis_space
                    
                    new_parent_clusters_map[label_val] = parent_cluster_level_t
                    next_level_clusters.append(parent_cluster_level_t)
                    self._all_clusters_map[parent_cluster_level_t.id] = parent_cluster_level_t
                else:
                    parent_cluster_level_t = new_parent_clusters_map[label_val]
                
                parent_cluster_level_t.children.append(child_cluster_from_prev_level)
                child_cluster_from_prev_level.parent = parent_cluster_level_t

            self._log(f"Processing {len(next_level_clusters)} new Level {current_level_num} clusters...", level=2)
            clusters_needing_llm_desc = []
            for parent in next_level_clusters:
                parent._aggregate_numeric_data_from_children()
                clusters_needing_llm_desc.append(parent)

            if clusters_needing_llm_desc:
                processed_llm_results, _ = self._get_llm_descriptions_batched(clusters_needing_llm_desc)
                for parent in next_level_clusters:
                    key = parent.id
                    if key in processed_llm_results:
                         parent.title, parent.description = processed_llm_results[key]

            self._generate_and_assign_description_embeddings(next_level_clusters)

            if self.reduction_methods:
                self._apply_reductions_to_embeddings(next_level_clusters, 'representation_vector')
                self._apply_reductions_to_embeddings(next_level_clusters, 'description_embedding')

            current_clusters = next_level_clusters
            last_successful_level = current_level_num
            self._log(f"Level {current_level_num} processing complete. {len(current_clusters)} clusters created.", level=1)

        self._max_level = last_successful_level
        return current_clusters

    def cluster(self, data: Any,
                topic_seed: str | None = None,
                numeric_metadata: Optional[Dict[Union[str, int], Dict[str, Any]]] = None
                ) -> list[Cluster]:
        """
        Performs hierarchical clustering on text, numeric, or image data.
        """
        self._configure_randomness()
        self._run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._prompt_log = []
        self._reducers_trained_ = {"text_embedding": False, "image_embedding": False, "numeric": False}
        self.reducers_ = {"text_embedding": {}, "image_embedding": {}, "numeric": {}}
        self.embedding_dims_ = {"text_embedding": None, "image_embedding": None, "numeric": None}
        self._all_clusters_map = {}
        self._l0_clusters_ordered = []
        self._max_level = 0
        self._original_id_to_index = {}
        self.variable_names = None
        self.numeric_metadata_by_name = None
        Cluster.reset_id_counter()
        self._current_topic_seed = topic_seed

        self._log(f"--- Starting Hercules Clustering Run: {self._run_id} ---", level=1)
        self._log(f"Hercules Version: {self.HERCULES_VERSION}", level=1)
        self._log(f"Verbosity Level: {self.verbose}", level=1)
        self._log(f"Representation Mode: {self.representation_mode}", level=1)
        if self.level_cluster_counts is None: self._log(f"Automatic K Mode: Enabled (Method: {self.auto_k_method}, Max K: {self.auto_k_max})", level=1)
        else: self._log(f"Automatic K Mode: Disabled (Using level_cluster_counts: {self.level_cluster_counts})", level=1)
        if self.reduction_methods: self._log(f"Reduction Methods: {self.reduction_methods} ({self.n_reduction_components} components)", level=1)
        if self._current_topic_seed: self._log(f"Topic Seed: '{self._current_topic_seed}'", level=1)
        if numeric_metadata: self._log("Numeric metadata provided.", level=1)
        if self.prompt_include_immediate_children: self._log(f"Prompting includes immediate children (Strategy: {self.prompt_immediate_child_sample_strategy}, Size: {self.prompt_immediate_child_sample_size}).", level=1)

        try:
            standardized_data, original_ids, input_type = self._prepare_input_data(data, numeric_metadata)
            self._log(f"Input data type detected: {input_type}", level=1)
            if isinstance(standardized_data, np.ndarray): self._log(f"  Scaled numeric data shape: {standardized_data.shape}", level=2)
            else: self._log(f"  Input items count: {len(standardized_data)}", level=2)
            if self.original_numeric_data_ is not None: self._log(f"  Original numeric data shape: {self.original_numeric_data_.shape}", level=2)
            if self.variable_names: self._log(f"  Using numeric variable names: {self.variable_names}", level=1)
        except (ValueError, TypeError, RuntimeError) as e:
            print(f"Error: Input data preparation failed: {e}"); return []

        self._l0_clusters_ordered = self._initialize_l0_clusters(standardized_data, original_ids)
        self._generate_l0_titles_descriptions(self._l0_clusters_ordered, standardized_data)
        self._generate_and_assign_description_embeddings(self._l0_clusters_ordered)

        success = self._generate_l0_representations(self._l0_clusters_ordered, standardized_data)
        if not success:
             print("Error: Failed to generate Level 0 representations for clustering. Aborting.")
             self._l0_clusters_ordered = []; self._all_clusters_map = {}; return []

        if self.representation_mode == 'description':
             self._log("Assigning L0 description embeddings to representation vectors (description mode)...", level=2)
             assigned_count = 0
             for cluster in self._l0_clusters_ordered:
                 if cluster.description_embedding is not None:
                     cluster.representation_vector = cluster.description_embedding
                     cluster.representation_vector_space = 'text_embedding'
                     assigned_count += 1
                 else:
                     cluster.representation_vector = None
                     cluster.representation_vector_space = None
             self._log(f"  Assigned {assigned_count} L0 description embeddings as representation vectors.", level=2)

        if self.reduction_methods:
             self._apply_reductions_to_embeddings(self._l0_clusters_ordered, 'representation_vector')
             self._apply_reductions_to_embeddings(self._l0_clusters_ordered, 'description_embedding')

        self._log(f"Level 0 processing complete. {len(self._l0_clusters_ordered)} items represented.", level=1)

        valid_l0_clusters = [c for c in self._l0_clusters_ordered if c.representation_vector is not None]
        if not valid_l0_clusters:
            print("Error: No valid L0 clusters have representations suitable for clustering. Cannot proceed with hierarchy.")
            return [] if not self._l0_clusters_ordered else self._l0_clusters_ordered

        self._log(f"Starting hierarchical loop with {len(valid_l0_clusters)} valid L0 clusters.", level=1)
        top_level_clusters = self._perform_hierarchical_clustering_loop(valid_l0_clusters)

        if self.save_run_details: self._save_prompt_log()
        self._log(f"\n--- Hercules Clustering Run Finished ---", level=1)
        self._log(f"Highest level achieved: {self._max_level}", level=1)
        self._log(f"Returning {len(top_level_clusters)} top-level clusters.", level=1)

        return top_level_clusters

    @property
    def max_level(self) -> int:
        """Returns the highest level achieved during the last clustering run."""
        return self._max_level

    @property
    def num_l0_items(self) -> int:
        """Returns the number of original L0 items processed."""
        return len(self._l0_clusters_ordered)

    def get_level_assignments(self, level: int) -> Tuple[np.ndarray | None, Dict[int, int]]:
        """
        Retrieves cluster assignments for all original L0 items at a specific level.

        Args:
            level: The target level (> 0).

        Returns:
            Tuple (assignments array | None, map {cluster_id -> label}).
            Assignments are -1 for unassigned items. Array length matches original L0 items.
        """
        if not self._l0_clusters_ordered: warnings.warn("Cannot get level assignments: Clustering has not been run."); return None, {}
        if not isinstance(level, int) or level <= 0: warnings.warn(f"Invalid level '{level}'. Must be > 0."); return None, {}
        if level > self._max_level: warnings.warn(f"Requested level {level} > max level achieved ({self._max_level}).");

        n_l0_items = len(self._l0_clusters_ordered)
        assignments = np.full(n_l0_items, -1, dtype=int)

        target_level_clusters = [c for c in self._all_clusters_map.values() if c.level == level]
        if not target_level_clusters:
             warnings.warn(f"No clusters found at level {level}.")
             return assignments, {}

        cluster_id_to_label = {cluster.id: i for i, cluster in enumerate(target_level_clusters)}
        assigned_count = 0

        self._log(f"Generating assignments for level {level} ({len(target_level_clusters)} clusters)...", level=2)

        for target_cluster in target_level_clusters:
            target_label = cluster_id_to_label[target_cluster.id]
            try:
                l0_descendants = target_cluster.get_level0_descendants()
                for l0_node in l0_descendants:
                    original_index = self._original_id_to_index.get(l0_node.original_id)
                    if original_index is not None:
                        if assignments[original_index] != -1:
                             warnings.warn(f"L0 item {l0_node.original_id} (index {original_index}) appears in multiple L{level} clusters. Overwriting.")
                        assignments[original_index] = target_label
                        assigned_count += 1
                    else:
                         warnings.warn(f"Cannot find original index for L0 node ID {l0_node.original_id}. Skipping assignment.")
            except Exception as e:
                 warnings.warn(f"Error processing descendants for cluster {target_cluster.id} at L{level}: {e}.")
                 continue

        unassigned_indices = np.where(assignments == -1)[0]
        if len(unassigned_indices) > 0:
             warnings.warn(f"{len(unassigned_indices)} L0 items could not be assigned to a cluster at level {level}. Label remains -1.")

        self._log(f"Generated assignments for {assigned_count}/{n_l0_items} L0 items at level {level}.", level=1)
        return assignments, cluster_id_to_label

    def evaluate_level(self, level: int,
                       ground_truth_labels: Optional[Union[list, np.ndarray, pd.Series, Dict[Any, Any]]] = None,
                       calculate_llm_internal_metrics: bool = True,
                       topic_seed: Optional[str] = None
                      ) -> Dict[str, Any]:
        """
        Evaluates the clustering quality at a specified level of the hierarchy.

        Calculates:
        - Traditional internal metrics (Silhouette, DB, CH) based on L-1 representations
          (mode-dependent) and L0 representations (representation_vector).
        - LLM-based internal metrics (Silhouette, DB, CH) based on L-1 description_embeddings
          and L0 description_embeddings (optional).
        - External metrics (ARI, NMI, V-Measure) if ground truth is provided.
        - Topic alignment score if a topic_seed is provided.

        Args:
            level: The hierarchy level to evaluate (> 0).
            ground_truth_labels: Optional ground truth labels for the original L0 items.
                                Should match the order or keys/index of the input data.
            calculate_llm_internal_metrics: Whether to calculate LLM-based internal metrics. Defaults to True.
            topic_seed: Optional seed text to evaluate topic alignment of clusters at the given level.

        Returns:
            Dictionary containing evaluation metric scores and metadata.
        """
        results: Dict[str, Any] = {
            "level_evaluated": level, "ground_truth_provided": False,
            "num_l0_items_total": 0, "num_l0_items_assigned": 0,
            "num_l0_items_in_gt_eval": 0,
            "num_clusters_found_at_level": 0,
            "num_clusters_with_desc_embedding": 0,

            "traditional_internal_metrics_calculated": True,
            "num_l_minus_1_items_in_traditional_internal_eval": 0,
            "silhouette_score_on_prev_level": None,
            "davies_bouldin_score_on_prev_level": None,
            "calinski_harabasz_score_on_prev_level": None,

            "num_l0_items_in_traditional_internal_eval": 0,
            "silhouette_score_on_l0": None,
            "davies_bouldin_score_on_l0": None,
            "calinski_harabasz_score_on_l0": None,

            "llm_internal_metrics_calculated": calculate_llm_internal_metrics,
            "num_l_minus_1_items_in_llm_internal_eval": 0,
            "llm_silhouette_on_prev_level": None,
            "llm_davies_bouldin_on_prev_level": None,
            "llm_calinski_harabasz_on_prev_level": None,

            "num_l0_items_in_llm_internal_eval": 0,
            "llm_silhouette_on_l0": None,
            "llm_davies_bouldin_on_l0": None,
            "llm_calinski_harabasz_on_l0": None,

            "topic_seed_provided": bool(topic_seed),
            "topic_alignment_score": None,
            "num_clusters_in_topic_alignment": 0,
        }
        self._log(f"Starting evaluation for level {level}...", level=1)
        if calculate_llm_internal_metrics:
            self._log(f"  LLM-based internal metrics: Will be calculated.", level=2)
        else:
            self._log(f"  LLM-based internal metrics: Will NOT be calculated (calculate_llm_internal_metrics=False).", level=2)
        if topic_seed:
            self._log(f"  Topic seed provided for alignment: '{topic_seed[:50]}...'", level=2)


        if not self._l0_clusters_ordered or not self._all_clusters_map:
            results["error"] = "Clustering has not been run or failed."; return results
        if not isinstance(level, int) or level <= 0:
            results["error"] = f"Invalid level '{level}'. Must be > 0."; return results
        if level > self._max_level:
             warnings.warn(f"Requested level {level} may be higher than max level performed ({self._max_level}). Evaluation might be incomplete.")

        n_l0_items = len(self._l0_clusters_ordered)
        results["num_l0_items_total"] = n_l0_items
        if n_l0_items == 0:
             results["error"] = "No L0 items found."; return results

        ground_truth_array_ordered = None
        if ground_truth_labels is not None:
            self._log(f"  Processing ground truth labels (type: {type(ground_truth_labels).__name__})...", level=2)
            results["ground_truth_provided"] = True
            try:
                if isinstance(ground_truth_labels, dict):
                    expected_ids = set(self._original_id_to_index.keys()); provided_ids_set = set(ground_truth_labels.keys())
                    expected_ids_str = set(map(str, expected_ids)); provided_ids_str = set(map(str, provided_ids_set))
                    missing_ids = expected_ids_str - provided_ids_str; extra_ids = provided_ids_str - expected_ids_str
                    temp_array = [None] * n_l0_items; valid_gt_count = 0
                    for orig_id, index in self._original_id_to_index.items():
                        label = ground_truth_labels.get(orig_id, ground_truth_labels.get(str(orig_id)))
                        if label is not None: temp_array[index] = label; valid_gt_count += 1
                    if valid_gt_count == 0: raise ValueError("No matching IDs found in ground truth dict.")
                    if missing_ids and valid_gt_count < len(expected_ids_str): warnings.warn(f"Ground truth dict missing labels for {len(missing_ids)} IDs (e.g., {list(missing_ids)[:5]}).")
                    if extra_ids: warnings.warn(f"Ground truth dict has {len(extra_ids)} extra IDs not in input data (e.g., {list(extra_ids)[:5]}).")
                    ground_truth_array_ordered = np.array(temp_array, dtype=object)
                    self._log(f"  Aligned ground truth labels from dictionary ({valid_gt_count}/{n_l0_items}).", level=2)
                elif isinstance(ground_truth_labels, pd.Series):
                    if len(ground_truth_labels) != n_l0_items: raise ValueError(f"GT Series length ({len(ground_truth_labels)}) != L0 items ({n_l0_items}).")
                    try:
                        ordered_ids = [c.original_id for c in self._l0_clusters_ordered]
                        ground_truth_array_ordered = ground_truth_labels.reindex(ordered_ids).values
                        if pd.isna(ground_truth_array_ordered).all(): raise Exception("All GTs NaN after reindex")
                    except:
                        warnings.warn("Could not reindex GT Series by original IDs, assuming direct alignment.")
                        ground_truth_array_ordered = ground_truth_labels.values
                elif isinstance(ground_truth_labels, (list, np.ndarray)):
                    if len(ground_truth_labels) != n_l0_items: raise ValueError(f"GT length ({len(ground_truth_labels)}) != L0 items ({n_l0_items}).")
                    ground_truth_array_ordered = np.asarray(ground_truth_labels)
                else: raise TypeError(f"Unsupported ground_truth_labels type: {type(ground_truth_labels)}")

                if ground_truth_array_ordered is not None:
                    unique_gt_labels = pd.unique(ground_truth_array_ordered[pd.notna(ground_truth_array_ordered)])
                    if len(unique_gt_labels) < 2:
                        warnings.warn("Ground truth has < 2 unique labels (excluding None/NaN). External metrics may be undefined.")
            except (ValueError, TypeError, KeyError, Exception) as e:
                 results["error"] = f"Error processing ground truth labels: {e}"; return results

        predicted_assignments, cluster_id_to_label_map_L_target = self.get_level_assignments(level)
        if predicted_assignments is None:
            results["error"] = f"Could not retrieve assignments for level {level}."; return results

        valid_assignment_mask = predicted_assignments != -1
        num_l0_items_assigned = np.sum(valid_assignment_mask)
        results["num_l0_items_assigned"] = num_l0_items_assigned

        if num_l0_items_assigned < n_l0_items:
            self._log(f"  Evaluating level {level} using {num_l0_items_assigned}/{n_l0_items} L0 items (excluding unassigned).", level=2)
        if num_l0_items_assigned < 2:
            results["error"] = f"Only {num_l0_items_assigned} valid L0 items assigned at L{level}. Cannot compute metrics."; return results

        predicted_assignments_valid = predicted_assignments[valid_assignment_mask]
        l0_indices_valid = np.where(valid_assignment_mask)[0]
        unique_pred_labels = np.unique(predicted_assignments_valid)
        num_clusters_found = len(unique_pred_labels)
        results["num_clusters_found_at_level"] = num_clusters_found
        self._log(f"  Found {num_clusters_found} distinct cluster labels among assigned L0 items.", level=2)

        if results["ground_truth_provided"] and ground_truth_array_ordered is not None:
            ground_truth_labels_valid = ground_truth_array_ordered[valid_assignment_mask]
            gt_not_none_mask = pd.notna(ground_truth_labels_valid)
            num_gt_valid = np.sum(gt_not_none_mask)
            results["num_l0_items_in_gt_eval"] = num_gt_valid

            if num_gt_valid < num_l0_items_assigned:
                warnings.warn(f"Excluding {num_l0_items_assigned - num_gt_valid} items from external evaluation due to missing ground truth labels.")

            if num_gt_valid < 2:
                 results["supervised_metrics_error"] = f"Only {num_gt_valid} items with valid assignments and non-missing GT labels."
                 self._log(f"  Warning: {results['supervised_metrics_error']}", level=2)
            else:
                predicted_for_external = predicted_assignments_valid[gt_not_none_mask]
                ground_truth_for_external = ground_truth_labels_valid[gt_not_none_mask]
                num_clusters_pred_final = len(np.unique(predicted_for_external))
                num_classes_gt_final = len(np.unique(ground_truth_for_external))

                if num_classes_gt_final < 2 or num_clusters_pred_final < 1:
                     warn_msg = f"Cannot calculate some external metrics: Found {num_clusters_pred_final} predicted clusters and {num_classes_gt_final} GT labels for the {num_gt_valid} comparable items."
                     warnings.warn(warn_msg); self._log(f"  Warning: {warn_msg}", level=2)
                     try: results["adjusted_rand_score"] = adjusted_rand_score(ground_truth_for_external, predicted_for_external)
                     except Exception as e: results["adjusted_rand_score_error"] = f"Failed: {e}"
                     try: results["normalized_mutual_info_score"] = normalized_mutual_info_score(ground_truth_for_external, predicted_for_external, average_method='arithmetic')
                     except Exception as e: results["normalized_mutual_info_score_error"] = f"Failed: {e}"
                     try: h, c, v = homogeneity_completeness_v_measure(ground_truth_for_external, predicted_for_external); results["homogeneity_score"], results["completeness_score"], results["v_measure_score"] = h, c, v
                     except Exception as e: results["v_measure_score_error"] = f"Failed: {e}"
                else:
                    try:
                        results["adjusted_rand_score"] = adjusted_rand_score(ground_truth_for_external, predicted_for_external)
                        results["normalized_mutual_info_score"] = normalized_mutual_info_score(ground_truth_for_external, predicted_for_external, average_method='arithmetic')
                        h, c, v = homogeneity_completeness_v_measure(ground_truth_for_external, predicted_for_external)
                        results["homogeneity_score"], results["completeness_score"], results["v_measure_score"] = h, c, v
                        self._log(f"  Calculated supervised metrics (ARI, NMI, V-Measure) using {num_gt_valid} items.", level=2)
                    except Exception as e:
                         results["supervised_metrics_error"] = str(e); warnings.warn(f"Error calculating supervised metrics: {e}")
        elif results["ground_truth_provided"]:
             results["supervised_metrics_error"] = "GT labels provided but could not be aligned or insufficient non-missing values."
             self._log(f"  Warning: {results['supervised_metrics_error']}", level=2)

        self._log(f"\n  Calculating Traditional Internal Metrics based on L{level-1} representations...", level=2)
        prev_level = level - 1
        if prev_level < 0:
             self._log("  Skipping Traditional L-1 internal metrics (not applicable for L1 evaluation).", level=2)
        else:
            prev_level_clusters_involved = set()
            target_level_cluster_objs_from_map = [c for c in self._all_clusters_map.values() if c.level == level]
            for target_cluster_obj in target_level_cluster_objs_from_map:
                prev_level_clusters_involved.update(c for c in target_cluster_obj.children if c.level == prev_level)

            if not prev_level_clusters_involved:
                 warn_msg = f"Could not find any L{prev_level} clusters as children of L{level} clusters. Cannot compute Traditional L-1 internal metrics."
                 warnings.warn(warn_msg); results["traditional_internal_metrics_prev_level_error"] = warn_msg
            else:
                 vectors_for_traditional_internal_eval_prev = []
                 labels_for_traditional_internal_eval_prev = []
                 clustering_space_for_level = None 
                 sample_prev_cluster = next(iter(prev_level_clusters_involved))

                 if self.representation_mode == 'direct': clustering_space_for_level = sample_prev_cluster.representation_vector_space
                 elif self.representation_mode == 'description': clustering_space_for_level = 'text_embedding'

                 if clustering_space_for_level is None:
                      warn_msg = f"Cannot determine clustering space for L{prev_level}->L{level} Traditional L-1 metrics."
                      warnings.warn(warn_msg); results["traditional_internal_metrics_prev_level_error"] = warn_msg
                 else:
                    processed_prev_clusters_trad = 0
                    for prev_cluster in prev_level_clusters_involved:
                        vector_to_use = None
                        if self.representation_mode == 'direct':
                            if prev_cluster.representation_vector is not None and prev_cluster.representation_vector_space == clustering_space_for_level: vector_to_use = prev_cluster.representation_vector
                        elif self.representation_mode == 'description': 
                            if prev_cluster.description_embedding is not None: vector_to_use = prev_cluster.description_embedding

                        parent_at_level = prev_cluster.parent; label = -1
                        if parent_at_level and parent_at_level.level == level: label = cluster_id_to_label_map_L_target.get(parent_at_level.id, -1)

                        if vector_to_use is not None and np.all(np.isfinite(vector_to_use)) and label != -1:
                            vectors_for_traditional_internal_eval_prev.append(vector_to_use); labels_for_traditional_internal_eval_prev.append(label); processed_prev_clusters_trad += 1
                    
                    results["num_l_minus_1_items_in_traditional_internal_eval"] = processed_prev_clusters_trad
                    if processed_prev_clusters_trad < 2:
                        warn_msg = f"Not enough valid vectors ({processed_prev_clusters_trad}) at L{prev_level} for Traditional L-1 metrics."
                        warnings.warn(warn_msg); results["traditional_internal_metrics_prev_level_error"] = f"Insufficient L{prev_level} data ({processed_prev_clusters_trad})"
                    else:
                        X_trad_internal_eval_prev = np.array(vectors_for_traditional_internal_eval_prev)
                        labels_trad_internal_eval_prev = np.array(labels_for_traditional_internal_eval_prev)
                        num_clusters_trad_internal_prev = len(np.unique(labels_trad_internal_eval_prev))
                        self._log(f"  Calculating Traditional L-1 internal metrics using {X_trad_internal_eval_prev.shape[0]} vectors (space: {clustering_space_for_level}).", level=2)

                        if num_clusters_trad_internal_prev < 2:
                            warnings.warn(f"Only {num_clusters_trad_internal_prev} clusters for Traditional L-1 metrics. Skipping Silhouette/DB.")
                        else:
                            silhouette_metric_trad_prev = 'cosine' if clustering_space_for_level in ['text_embedding', 'image_embedding'] else 'euclidean'
                            try: results["silhouette_score_on_prev_level"] = silhouette_score(X_trad_internal_eval_prev, labels_trad_internal_eval_prev, metric=silhouette_metric_trad_prev)
                            except Exception as e: results["silhouette_score_on_prev_level_error"] = str(e)
                            try: results["davies_bouldin_score_on_prev_level"] = davies_bouldin_score(X_trad_internal_eval_prev, labels_trad_internal_eval_prev)
                            except Exception as e: results["davies_bouldin_score_on_prev_level_error"] = str(e)
                        
                        min_samples_ch = 3 
                        if num_clusters_trad_internal_prev >= 2 and X_trad_internal_eval_prev.shape[0] >= min_samples_ch:
                            try: results["calinski_harabasz_score_on_prev_level"] = calinski_harabasz_score(X_trad_internal_eval_prev, labels_trad_internal_eval_prev)
                            except Exception as e: results["calinski_harabasz_score_on_prev_level_error"] = str(e)
        
        if calculate_llm_internal_metrics:
            self._log(f"\n  Calculating LLM-Based Internal Metrics based on L{level-1} description_embeddings...", level=2)
            if prev_level < 0:
                self._log("  Skipping LLM-Based L-1 internal metrics (not applicable for L1 evaluation).", level=2)
            else:
                if not prev_level_clusters_involved:
                    results["llm_internal_metrics_prev_level_error"] = "No L-1 child clusters found (same as traditional)."
                else:
                    vectors_for_llm_internal_eval_prev = []
                    labels_for_llm_internal_eval_prev = []
                    processed_prev_clusters_llm = 0
                    for prev_cluster in prev_level_clusters_involved:
                        vector_to_use = prev_cluster.description_embedding
                        parent_at_level = prev_cluster.parent; label = -1
                        if parent_at_level and parent_at_level.level == level: label = cluster_id_to_label_map_L_target.get(parent_at_level.id, -1)

                        if vector_to_use is not None and np.all(np.isfinite(vector_to_use)) and label != -1:
                            vectors_for_llm_internal_eval_prev.append(vector_to_use); labels_for_llm_internal_eval_prev.append(label); processed_prev_clusters_llm += 1
                    
                    results["num_l_minus_1_items_in_llm_internal_eval"] = processed_prev_clusters_llm
                    if processed_prev_clusters_llm < 2:
                        warn_msg = f"Not enough valid description_embeddings ({processed_prev_clusters_llm}) at L{prev_level} for LLM-Based L-1 metrics."
                        warnings.warn(warn_msg); results["llm_internal_metrics_prev_level_error"] = f"Insufficient L{prev_level} desc_emb data ({processed_prev_clusters_llm})"
                    else:
                        X_llm_internal_eval_prev = np.array(vectors_for_llm_internal_eval_prev)
                        labels_llm_internal_eval_prev = np.array(labels_for_llm_internal_eval_prev)
                        num_clusters_llm_internal_prev = len(np.unique(labels_llm_internal_eval_prev))
                        self._log(f"  Calculating LLM-Based L-1 internal metrics using {X_llm_internal_eval_prev.shape[0]} description_embeddings.", level=2)

                        if num_clusters_llm_internal_prev < 2:
                            warnings.warn(f"Only {num_clusters_llm_internal_prev} clusters for LLM-Based L-1 metrics. Skipping LLM Silhouette/DB on prev_level.")
                        else:
                            try: results["llm_silhouette_on_prev_level"] = silhouette_score(X_llm_internal_eval_prev, labels_llm_internal_eval_prev, metric='cosine')
                            except Exception as e: results["llm_silhouette_on_prev_level_error"] = str(e)
                            try: results["llm_davies_bouldin_on_prev_level"] = davies_bouldin_score(X_llm_internal_eval_prev, labels_llm_internal_eval_prev)
                            except Exception as e: results["llm_davies_bouldin_on_prev_level_error"] = str(e)
                        
                        min_samples_ch = 3
                        if num_clusters_llm_internal_prev >= 2 and X_llm_internal_eval_prev.shape[0] >= min_samples_ch :
                            try: results["llm_calinski_harabasz_on_prev_level"] = calinski_harabasz_score(X_llm_internal_eval_prev, labels_llm_internal_eval_prev)
                            except Exception as e: results["llm_calinski_harabasz_on_prev_level_error"] = str(e)
        else: 
            self._log(f"\n  Skipping calculation of LLM-Based L{level-1} internal metrics.", level=2)


        self._log(f"\n  Calculating Traditional Internal Metrics based on L0 representation_vectors...", level=2)
        l0_trad_vectors_for_eval = []
        l0_trad_indices_for_eval = []
        l0_trad_space = None

        for idx in l0_indices_valid:
            l0_cluster = self._l0_clusters_ordered[idx]
            vec = l0_cluster.representation_vector 
            space = l0_cluster.representation_vector_space

            if vec is not None and space is not None and np.all(np.isfinite(vec)):
                 if l0_trad_space is None: l0_trad_space = space
                 elif l0_trad_space != space:
                      warn_msg = f"Inconsistent L0 representation_vector spaces ('{l0_trad_space}' vs '{space}'). Cannot compute Traditional L0 internal metrics."
                      warnings.warn(warn_msg); results["traditional_internal_metrics_l0_error"] = warn_msg
                      l0_trad_vectors_for_eval = []; break
                 l0_trad_vectors_for_eval.append(vec)
                 l0_trad_indices_for_eval.append(idx)

        num_l0_items_in_trad_internal_eval = len(l0_trad_vectors_for_eval)
        results["num_l0_items_in_traditional_internal_eval"] = num_l0_items_in_trad_internal_eval

        if not l0_trad_vectors_for_eval:
            if "traditional_internal_metrics_l0_error" not in results:
                 results["traditional_internal_metrics_l0_error"] = f"No valid L0 representation_vectors for items at level {level}."
            self._log(f"  Warning: {results['traditional_internal_metrics_l0_error']}", level=2)
        elif num_l0_items_in_trad_internal_eval < 2:
            warn_msg = f"Not enough L0 items ({num_l0_items_in_trad_internal_eval}) with valid representation_vectors for Traditional L0 metrics."
            warnings.warn(warn_msg); results["traditional_internal_metrics_l0_error"] = f"Insufficient L0 data ({num_l0_items_in_trad_internal_eval})"
        else:
            mask_for_trad_labels_l0 = np.isin(l0_indices_valid, l0_trad_indices_for_eval)
            labels_for_trad_l0_eval = predicted_assignments_valid[mask_for_trad_labels_l0]
            X_trad_l0_eval = np.array(l0_trad_vectors_for_eval)

            if X_trad_l0_eval.shape[0] != labels_for_trad_l0_eval.shape[0]:
                 warn_msg = "Shape mismatch Traditional L0 vectors vs labels. Skipping."
                 warnings.warn(warn_msg); results["traditional_internal_metrics_l0_error"] = warn_msg
            else:
                 num_clusters_trad_l0_eval = len(np.unique(labels_for_trad_l0_eval))
                 self._log(f"  Calculating Traditional L0 internal metrics using {X_trad_l0_eval.shape[0]} representation_vectors (space: {l0_trad_space}, clusters: {num_clusters_trad_l0_eval}).", level=2)

                 if num_clusters_trad_l0_eval < 2:
                     warnings.warn(f"Only {num_clusters_trad_l0_eval} clusters for Traditional L0 metrics. Skipping Silhouette/DB.")
                 else:
                     silhouette_metric_trad_l0 = 'cosine' if l0_trad_space in ['text_embedding', 'image_embedding'] else 'euclidean'
                     try: results["silhouette_score_on_l0"] = silhouette_score(X_trad_l0_eval, labels_for_trad_l0_eval, metric=silhouette_metric_trad_l0)
                     except Exception as e: results["silhouette_score_on_l0_error"] = str(e)
                     try: results["davies_bouldin_score_on_l0"] = davies_bouldin_score(X_trad_l0_eval, labels_for_trad_l0_eval)
                     except Exception as e: results["davies_bouldin_score_on_l0_error"] = str(e)
                 
                 min_samples_ch = 3
                 if num_clusters_trad_l0_eval >= 2 and X_trad_l0_eval.shape[0] >= min_samples_ch :
                     try: results["calinski_harabasz_score_on_l0"] = calinski_harabasz_score(X_trad_l0_eval, labels_for_trad_l0_eval)
                     except Exception as e: results["calinski_harabasz_score_on_l0_error"] = str(e)

        if calculate_llm_internal_metrics:
            self._log(f"\n  Calculating LLM-Based Internal Metrics based on L0 description_embeddings...", level=2)
            l0_llm_desc_embeddings_for_eval = []
            l0_llm_indices_for_eval = []

            for idx in l0_indices_valid:
                l0_cluster = self._l0_clusters_ordered[idx]
                desc_emb = l0_cluster.description_embedding 

                if desc_emb is not None and np.all(np.isfinite(desc_emb)):
                     l0_llm_desc_embeddings_for_eval.append(desc_emb)
                     l0_llm_indices_for_eval.append(idx)
                else:
                     self._log(f"    L0 item {l0_cluster.original_id} missing valid description_embedding. Excluding from LLM L0 Metrics.", level=3)

            num_l0_items_in_llm_internal_eval = len(l0_llm_desc_embeddings_for_eval)
            results["num_l0_items_in_llm_internal_eval"] = num_l0_items_in_llm_internal_eval

            if not l0_llm_desc_embeddings_for_eval:
                results["llm_internal_metrics_l0_error"] = f"No valid L0 description_embeddings for items at level {level}."
                self._log(f"  Warning: {results['llm_internal_metrics_l0_error']}", level=2)
            elif num_l0_items_in_llm_internal_eval < 2:
                warn_msg = f"Not enough L0 items ({num_l0_items_in_llm_internal_eval}) with valid description_embeddings for LLM L0 Metrics."
                warnings.warn(warn_msg); results["llm_internal_metrics_l0_error"] = f"Insufficient L0 data for LLM Metrics ({num_l0_items_in_llm_internal_eval})"
            else:
                mask_for_llm_labels_l0 = np.isin(l0_indices_valid, l0_llm_indices_for_eval)
                labels_for_llm_l0_eval = predicted_assignments_valid[mask_for_llm_labels_l0]
                X_llm_l0_desc_emb_eval = np.array(l0_llm_desc_embeddings_for_eval)

                if X_llm_l0_desc_emb_eval.shape[0] != labels_for_llm_l0_eval.shape[0]:
                     warn_msg = "Shape mismatch LLM L0 description_embeddings vs labels. Skipping."
                     warnings.warn(warn_msg); results["llm_internal_metrics_l0_error"] = warn_msg
                else:
                     num_clusters_llm_l0 = len(np.unique(labels_for_llm_l0_eval))
                     self._log(f"  Calculating LLM L0 internal metrics using {X_llm_l0_desc_emb_eval.shape[0]} description_embeddings (clusters: {num_clusters_llm_l0}).", level=2)

                     if num_clusters_llm_l0 < 2:
                         warnings.warn(f"Only {num_clusters_llm_l0} clusters for LLM L0 metrics. Skipping LLM Silhouette/DB on L0.")
                     else:
                         try: results["llm_silhouette_on_l0"] = silhouette_score(X_llm_l0_desc_emb_eval, labels_for_llm_l0_eval, metric='cosine')
                         except Exception as e: results["llm_silhouette_on_l0_error"] = str(e)
                         try: results["llm_davies_bouldin_on_l0"] = davies_bouldin_score(X_llm_l0_desc_emb_eval, labels_for_llm_l0_eval)
                         except Exception as e: results["llm_davies_bouldin_on_l0_error"] = str(e)
                     
                     min_samples_ch = 3
                     if num_clusters_llm_l0 >= 2 and X_llm_l0_desc_emb_eval.shape[0] >= min_samples_ch :
                         try: results["llm_calinski_harabasz_on_l0"] = calinski_harabasz_score(X_llm_l0_desc_emb_eval, labels_for_llm_l0_eval)
                         except Exception as e: results["llm_calinski_harabasz_on_l0_error"] = str(e)
        else:
            self._log(f"\n  Skipping calculation of LLM-Based L0 internal metrics.", level=2)

        target_level_cluster_objs_with_desc_emb = [
            c for c in self._all_clusters_map.values()
            if c.level == level and c.description_embedding is not None and np.all(np.isfinite(c.description_embedding))
        ]
        results["num_clusters_with_desc_embedding"] = len(target_level_cluster_objs_with_desc_emb)

        if topic_seed:
            self._log(f"\n  Calculating Topic Alignment for level {level} with seed: '{topic_seed[:50]}...'", level=2)
            results["topic_seed_provided"] = True
            try:
                topic_seed_embedding = self.text_embedding_client([topic_seed])
                if topic_seed_embedding is None or topic_seed_embedding.shape[0] == 0 or not np.all(np.isfinite(topic_seed_embedding[0])):
                    raise ValueError("Failed to generate a valid embedding for the topic seed.")
                topic_seed_vec = topic_seed_embedding[0]

                cluster_desc_embeddings_for_alignment = []
                for cluster_obj in target_level_cluster_objs_from_map:
                    if cluster_obj.description_embedding is not None and np.all(np.isfinite(cluster_obj.description_embedding)):
                        cluster_desc_embeddings_for_alignment.append(cluster_obj.description_embedding)
                
                results["num_clusters_in_topic_alignment"] = len(cluster_desc_embeddings_for_alignment)

                if not cluster_desc_embeddings_for_alignment:
                    results["topic_alignment_error"] = "No clusters at this level have valid description embeddings for alignment."
                    self._log(f"  Warning: {results['topic_alignment_error']}", level=2)
                elif len(cluster_desc_embeddings_for_alignment) < 1 :
                    results["topic_alignment_error"] = f"Not enough clusters ({len(cluster_desc_embeddings_for_alignment)}) with description embeddings for alignment."
                    self._log(f"  Warning: {results['topic_alignment_error']}", level=2)

                else:
                    alignment_similarities = cosine_similarity(topic_seed_vec.reshape(1, -1), np.array(cluster_desc_embeddings_for_alignment))
                    results["topic_alignment_score"] = float(np.mean(alignment_similarities[0]))
                    self._log(f"  Calculated topic alignment score: {results['topic_alignment_score']:.4f} using {len(cluster_desc_embeddings_for_alignment)} clusters.", level=2)

            except Exception as e:
                results["topic_alignment_error"] = f"Error during topic alignment: {e}"
                warnings.warn(f"Topic alignment calculation failed: {e}")
                self._log(f"  Warning: {results['topic_alignment_error']}", level=2)
        
        self._log(f"Evaluation for level {level} complete.", level=1)
        final_results = {}
        for k, v in results.items():
            if not (k.endswith("_error") and (v is None or v is False)):
                 final_results[k] = v
        return final_results

    def get_cluster_membership_dataframe(self,
                                         include_l0_details: Optional[List[str]] = None
                                         ) -> pd.DataFrame:
        """
        Creates a DataFrame showing cluster membership of each L0 item across levels.

        Args:
            include_l0_details: Optional list of L0 Cluster attributes to include
                                (e.g., ['title', 'description']). Checks if attribute exists.

        Returns:
            pandas.DataFrame: Rows are L0 items, columns show original_id,
                              optional L0 details, and L{level}_cluster_id/title pairs.
                              Index matches the original order of L0 items.
        """
        self._log("Generating L0 cluster membership DataFrame...", level=1)

        if not self._l0_clusters_ordered:
            warnings.warn("Cannot generate membership DataFrame: Clustering not run or failed.")
            return pd.DataFrame()

        valid_l0_details_list = []
        if include_l0_details is not None:
            if not isinstance(include_l0_details, list) or not all(isinstance(item, str) for item in include_l0_details):
                raise TypeError("`include_l0_details` must be a list of strings.")
            if self._l0_clusters_ordered:
                sample_cluster = self._l0_clusters_ordered[0]
                valid_l0_details_list = [attr for attr in include_l0_details if hasattr(sample_cluster, attr)]
                invalid_attrs = set(include_l0_details) - set(valid_l0_details_list)
                if invalid_attrs: warnings.warn(f"Ignoring invalid L0 detail attributes: {invalid_attrs}")
            self._log(f"Including L0 details: {valid_l0_details_list}", level=1)

        if self._max_level == 0:
            warnings.warn("No higher levels (L1+) created. Returning DF with L0 details only.")
            if self._l0_clusters_ordered:
                 l0_only_data = []
                 for l0_cluster in self._l0_clusters_ordered:
                      row = {'original_id': l0_cluster.original_id}
                      for field_name in valid_l0_details_list:
                           row[field_name] = getattr(l0_cluster, field_name, None)
                      l0_only_data.append(row)
                 expected_cols = ['original_id'] + valid_l0_details_list
                 original_ids = [c.original_id for c in self._l0_clusters_ordered]
                 df_index = pd.RangeIndex(len(original_ids))
                 if len(set(original_ids)) == len(original_ids) and all(isinstance(id_val, (str, int, float)) for id_val in original_ids):
                      try: df_index = pd.Index(original_ids, name="original_id")
                      except Exception: warnings.warn("Could not use original IDs as index, using default RangeIndex.")

                 return pd.DataFrame(l0_only_data, index=df_index).reindex(columns=expected_cols)
            else: return pd.DataFrame()

        n_l0_items = len(self._l0_clusters_ordered)
        all_rows_data = []
        max_level_to_include = self._max_level
        self._log(f"Tracing ancestry for {n_l0_items} L0 items up to Level {max_level_to_include}.", level=2)

        original_ids_list = [c.original_id for c in self._l0_clusters_ordered]

        for i, l0_cluster in enumerate(self._l0_clusters_ordered):
            row_data = {'original_id': l0_cluster.original_id}
            for field_name in valid_l0_details_list:
                row_data[field_name] = getattr(l0_cluster, field_name, None)

            current_node = l0_cluster
            while current_node is not None and current_node.parent is not None:
                parent_node = current_node.parent
                parent_level = parent_node.level
                if 0 < parent_level <= max_level_to_include:
                    level_prefix = f"L{parent_level}"
                    row_data[f"{level_prefix}_cluster_id"] = parent_node.id
                    row_data[f"{level_prefix}_cluster_title"] = parent_node.title
                current_node = parent_node
                if parent_level >= max_level_to_include: break
            all_rows_data.append(row_data)

        if not all_rows_data: return pd.DataFrame()

        df_index = pd.RangeIndex(len(original_ids_list))
        if len(set(original_ids_list)) == len(original_ids_list) and all(isinstance(id_val, (str, int, float)) for id_val in original_ids_list):
             try: df_index = pd.Index(original_ids_list, name="original_id")
             except Exception: warnings.warn("Could not use original IDs as index, using default RangeIndex.")

        membership_df = pd.DataFrame(all_rows_data, index=df_index)

        expected_columns = ['original_id']
        expected_columns.extend(valid_l0_details_list)
        for level in range(1, max_level_to_include + 1):
            level_prefix = f"L{level}"
            expected_columns.append(f"{level_prefix}_cluster_id")
            expected_columns.append(f"{level_prefix}_cluster_title")

        membership_df = membership_df.reindex(columns=expected_columns)

        for level in range(1, max_level_to_include + 1):
            id_col = f"L{level}_cluster_id"
            if id_col in membership_df.columns:
                 try: membership_df[id_col] = membership_df[id_col].astype('Int64')
                 except Exception: pass

        if isinstance(membership_df.index, pd.Index) and membership_df.index.name == 'original_id' and 'original_id' in membership_df.columns:
            membership_df = membership_df.drop(columns=['original_id'])

        self._log("Cluster membership DataFrame generated successfully.", level=1)
        return membership_df


    def save_model(self, filepath: str, top_clusters: list[Cluster]):
        """
        Saves Hercules config, state, function metadata, reducer configs, and cluster hierarchy.
        """
        if not self._all_clusters_map and top_clusters:
             queue = list(top_clusters); visited = set(); temp_map = {}
             while queue:
                 c = queue.pop(0)
                 if c is None or c.id in visited: continue
                 visited.add(c.id); temp_map[c.id] = c
                 queue.extend(c.children)
             self._all_clusters_map = temp_map
             if temp_map: warnings.warn("Internal cluster map reconstructed for saving.")

        if not self._all_clusters_map: print("Error: No cluster data to save."); return

        if not filepath.endswith(".json"): filepath += ".json"
        self._log(f"Saving Hercules model to {filepath}...", level=1)

        all_clusters_dict_list = [c.to_dict() for c in self._all_clusters_map.values()]

        serializable_reducers = {}
        for space_type in ["text_embedding", "image_embedding", "numeric"]:
            if space_type in self.reducers_ and self.reducers_[space_type]:
                serializable_reducers[space_type] = {}
                for method, reducer_instance in self.reducers_[space_type].items():
                    if reducer_instance is None or method != 'pca': continue
                    try:
                        n_comps = getattr(reducer_instance, 'n_components_', None) or \
                                    getattr(reducer_instance, 'n_components', self.n_reduction_components)
                        serializable_reducers[space_type][method] = {
                            "type": method, "n_components": n_comps
                        }
                    except Exception as e: warnings.warn(f"Could not serialize PCA config ({space_type}): {e}")

        model_data = {
            "__hercules_version__": self.HERCULES_VERSION,
            "config": {
                 "level_cluster_counts": self.level_cluster_counts,
                 "representation_mode": self.representation_mode,
                 "auto_k_method": self.auto_k_method,
                 "auto_k_max": self.auto_k_max,
                 "auto_k_metric_params": self.auto_k_metric_params,
                 "n_reduction_components": self.n_reduction_components,
                 "reduction_methods": self.reduction_methods,
                 "random_state": self.random_state,
                 "save_run_details": self.save_run_details,
                 "run_details_dir": self.run_details_dir,
                 "verbose": self.verbose,
                 "prompt_l0_sample_size": self.prompt_l0_sample_size,
                 "prompt_l0_sample_strategy": self.prompt_l0_sample_strategy,
                 "prompt_l0_sample_trunc_len": self.prompt_l0_sample_trunc_len,
                 "prompt_l0_numeric_repr_max_vals": self.prompt_l0_numeric_repr_max_vals,
                 "prompt_l0_numeric_repr_precision": self.prompt_l0_numeric_repr_precision,
                 "prompt_include_immediate_children": self.prompt_include_immediate_children,
                 "prompt_immediate_child_sample_strategy": self.prompt_immediate_child_sample_strategy,
                 "prompt_immediate_child_sample_size": self.prompt_immediate_child_sample_size,
                 "prompt_child_sample_desc_trunc_len": self.prompt_child_sample_desc_trunc_len,
                 "prompt_max_stats_vars": self.prompt_max_stats_vars,
                 "prompt_numeric_stats_precision": self.prompt_numeric_stats_precision,
                 "max_prompt_tokens": self.max_prompt_tokens,
                 "max_prompt_token_buffer": self.max_prompt_token_buffer,
                 "llm_initial_batch_size": self.llm_initial_batch_size,
                 "llm_batch_reduction_factor": self.llm_batch_reduction_factor,
                 "llm_min_batch_size": self.llm_min_batch_size,
                 "llm_retries": self.llm_retries,
                 "llm_retry_delay": self.llm_retry_delay,
                 "direct_l0_text_trunc_len": self.direct_l0_text_trunc_len,
                 "min_clusters_per_level": self.min_clusters_per_level,
                 "fallback_k": self.fallback_k,
                 "cluster_numeric_stats_precision": self.cluster_numeric_stats_precision,
                 "cluster_print_indent_increment": self.cluster_print_indent_increment,
                 "use_llm_for_l0_descriptions": self.use_llm_for_l0_descriptions,
                 "use_resampling": self.use_resampling,
                 "resampling_points_per_cluster": self.resampling_points_per_cluster,
                 "resampling_iterations": self.resampling_iterations,
            },
            "state": {
                 "variable_names": self.variable_names,
                 "numeric_metadata_by_name": self.numeric_metadata_by_name,
                 "input_data_type": self.input_data_type,
                 "embedding_dims": self.embedding_dims_,
                 "max_level_achieved": self._max_level,
                 "scaler_params": {
                      "mean_": self._scaler.mean_.tolist(), "scale_": self._scaler.scale_.tolist(),
                      "n_features_in_": getattr(self._scaler, 'n_features_in_', None),
                      "n_samples_seen_": int(getattr(self._scaler, 'n_samples_seen_', 0))
                  } if self._scaler and hasattr(self._scaler, 'mean_') and self._scaler.mean_ is not None else None,
                 "_l0_original_ids_ordered": [c.original_id for c in self._l0_clusters_ordered] if self._l0_clusters_ordered else None
            },
            "function_metadata": self._function_metadata,
            "reducers_config": serializable_reducers,
            "clusters": all_clusters_dict_list,
            "top_cluster_ids": [c.id for c in top_clusters if c and c.id in self._all_clusters_map],
            "prompt_log": self._prompt_log if self.save_run_details else []
        }

        try:
            dir_name = os.path.dirname(os.path.abspath(filepath))
            if dir_name: os.makedirs(dir_name, exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                 json.dump(model_data, f, indent=2, default=str)
            self._log(f"Model successfully saved to {filepath}", level=1)
        except Exception as e: print(f"Error saving model: {e}")

    @classmethod
    def load_model(cls, filepath: str,
                   text_embedding_client: Optional[Callable[[list[str]], np.ndarray]] = None,
                   llm_client: Optional[Callable[[str], str]] = None,
                   image_embedding_client: Optional[Callable] = None,
                   image_captioning_client: Optional[Callable] = None
                   ) -> tuple[Hercules | None, list[Cluster]]:
        """
        Loads Hercules model. Requires compatible client functions.

        Restores configuration, state, and cluster hierarchy. Reducers are
        reconstructed but UNTRAINED. Need to re-run `cluster` or manually
        train reducers if reduction is needed post-load.

        Args:
            filepath: Path to the saved model JSON file.
            text_embedding_client: Function for embedding text (must match expected signature).
            llm_client: Function for LLM calls (must match expected signature).
            image_embedding_client: Function for embedding images (if used originally).
            image_captioning_client: Function for captioning images (if used originally).

        Returns:
            Tuple of (Hercules instance | None, list of top-level Cluster objects).
        """
        print(f"Loading Hercules model from {filepath}...")
        try:
            with open(filepath, 'r', encoding='utf-8') as f: model_data = json.load(f)
        except FileNotFoundError: print(f"Error: Model file not found at {filepath}"); return None, []
        except json.JSONDecodeError as e: print(f"Error: Failed to decode JSON from {filepath}: {e}"); return None, []
        except Exception as e: print(f"Error loading model file {filepath}: {e}"); return None, []

        saved_version = model_data.get("__hercules_version__", "Unknown")
        print(f"Loading model saved with Hercules version: {saved_version}. Current tool version: {cls.HERCULES_VERSION}")
        if saved_version.split('.')[0] != cls.HERCULES_VERSION.split('.')[0]:
             warnings.warn(f"Loading model from different major version ({saved_version}) than current ({cls.HERCULES_VERSION}). Compatibility not guaranteed.")

        config = model_data.get("config", {})
        state = model_data.get("state", {})
        reducers_config_saved = model_data.get("reducers_config", {})
        cluster_data_list = model_data.get("clusters", [])
        top_cluster_ids = model_data.get("top_cluster_ids", [])
        function_metadata_saved = model_data.get("function_metadata", {})

        try:
            init_args = {}
            init_sig = inspect.signature(cls.__init__)
            init_params = init_sig.parameters
            default_config = {}
            for name, param in init_params.items():
                 if name != 'self' and param.default != inspect.Parameter.empty:
                      default_const_name = f"DEFAULT_{name.upper()}"
                      if hasattr(cls, default_const_name):
                           default_config[name] = getattr(cls, default_const_name)
                      else:
                           default_config[name] = param.default
            init_args.update(default_config)

            saved_config_mapped = {k: v for k, v in config.items() if k in init_params}
            init_args.update(saved_config_mapped)

            init_args["text_embedding_client"] = text_embedding_client
            init_args["llm_client"] = llm_client
            init_args["image_embedding_client"] = image_embedding_client
            init_args["image_captioning_client"] = image_captioning_client

            init_param_names = list(init_params.keys())[1:]
            init_args_final = {k: v for k, v in init_args.items() if k in init_param_names}

            hercules_instance = cls(**init_args_final)
            hercules_instance._log(f"Hercules instance created from loaded config (verbose={hercules_instance.verbose}).", level=1)

        except Exception as e: print(f"Error instantiating Hercules from loaded config: {e}"); return None, []

        hercules_instance.variable_names = state.get("variable_names")
        hercules_instance.numeric_metadata_by_name = state.get("numeric_metadata_by_name")
        hercules_instance.input_data_type = state.get("input_data_type")
        hercules_instance.embedding_dims_ = state.get("embedding_dims", {"text_embedding": None, "image_embedding": None, "numeric": None})
        hercules_instance._max_level = state.get("max_level_achieved", 0)
        hercules_instance._reducers_trained_ = {"text_embedding": False, "image_embedding": False, "numeric": False}
        hercules_instance._prompt_log = model_data.get("prompt_log", []) if hercules_instance.save_run_details else []
        hercules_instance._function_metadata = function_metadata_saved
        if hercules_instance._function_metadata: hercules_instance._log("  Loaded function metadata.", level=2)

        scaler_params = state.get("scaler_params")
        if scaler_params and scaler_params.get("mean_") is not None and scaler_params.get("scale_") is not None:
             try:
                 hercules_instance._scaler = StandardScaler()
                 hercules_instance._scaler.mean_ = np.array(scaler_params["mean_"])
                 hercules_instance._scaler.scale_ = np.array(scaler_params["scale_"])
                 if "n_features_in_" in scaler_params and scaler_params["n_features_in_"] is not None: hercules_instance._scaler.n_features_in_ = int(scaler_params["n_features_in_"])
                 if "n_samples_seen_" in scaler_params and scaler_params["n_samples_seen_"] is not None: hercules_instance._scaler.n_samples_seen_ = int(scaler_params["n_samples_seen_"])
                 hercules_instance._log("  StandardScaler state restored.", level=2)
             except Exception as e: warnings.warn(f"Error restoring StandardScaler state: {e}")

        hercules_instance.reducers_ = {"text_embedding": {}, "image_embedding": {}, "numeric": {}}
        for space_type_key, methods_config in reducers_config_saved.items():
            current_space_types_for_key = []
            if space_type_key == "embedding":
                if "text_embedding" in hercules_instance.reducers_: current_space_types_for_key.append("text_embedding")
                if "image_embedding" in hercules_instance.reducers_: current_space_types_for_key.append("image_embedding")
            elif space_type_key in hercules_instance.reducers_:
                 current_space_types_for_key.append(space_type_key)

            for current_space_type in current_space_types_for_key:
                if current_space_type not in hercules_instance.reducers_: continue
                if not isinstance(methods_config, dict):
                    warnings.warn(f"Reducer methods config for space '{current_space_type}' is not a dictionary. Skipping reconstruction for this space.")
                    continue

                for method, r_config in methods_config.items():
                    if not isinstance(r_config, dict): continue
                    try:
                        if method == 'pca' and r_config.get('type') == 'pca':
                            n_comp = r_config.get('n_components', hercules_instance.n_reduction_components)
                            hercules_instance.reducers_[current_space_type][method] = PCA(n_components=n_comp, random_state=hercules_instance.random_state)
                            hercules_instance._log(f"  Reconstructed PCA reducer instance ({current_space_type}, n_components={n_comp}). Status: UNTRAINED.", level=2)
                    except Exception as e: warnings.warn(f"Error reconstructing {method} reducer ({current_space_type}): {e}")


        Cluster.reset_id_counter() 
        clusters_by_id: dict[int, Cluster] = {}
        top_clusters: list[Cluster] = []
        max_loaded_id = -1

        if not cluster_data_list: warnings.warn("No cluster data found in saved model.")
        else:
            hercules_instance._log(f"Deserializing {len(cluster_data_list)} cluster objects...", level=2)
            for cluster_data in cluster_data_list:
                try:
                    if not isinstance(cluster_data, dict): continue 
                    cluster = Cluster.from_dict(cluster_data) 
                    cluster.representation_mode = hercules_instance.representation_mode
                    clusters_by_id[cluster.id] = cluster
                    max_loaded_id = max(max_loaded_id, cluster.id)
                except Exception as e: warnings.warn(f"Error deserializing cluster data for ID {cluster_data.get('id', 'N/A')}: {e}")

            Cluster._next_internal_id = max(Cluster._next_internal_id, max_loaded_id + 1)

            if clusters_by_id:
                 hercules_instance._log("Linking cluster hierarchy...", level=2)
                 try: Cluster.link_hierarchy(clusters_by_id) 
                 except Exception as e: warnings.warn(f"Error linking cluster hierarchy: {e}")

                 loaded_top_clusters = [clusters_by_id[tid] for tid in top_cluster_ids if tid in clusters_by_id]
                 if len(loaded_top_clusters) != len(top_cluster_ids): warnings.warn("Some top cluster IDs from save file not found in loaded clusters.")

                 if not loaded_top_clusters and clusters_by_id:
                      hercules_instance._log("Inferring top-level clusters (parent is None)...", level=2)
                      loaded_top_clusters = [c for c in clusters_by_id.values() if c.parent is None]
                 top_clusters = loaded_top_clusters

        hercules_instance._all_clusters_map = clusters_by_id

        hercules_instance._l0_clusters_ordered = []
        hercules_instance._original_id_to_index = {}
        saved_l0_ids_ordered = state.get("_l0_original_ids_ordered")
        if saved_l0_ids_ordered and clusters_by_id:
             l0_map_temp = {c.original_id: c for c in clusters_by_id.values() if c.level == 0}
             restored_count = 0
             for i, orig_id in enumerate(saved_l0_ids_ordered):
                 l0_node = l0_map_temp.get(orig_id)
                 if l0_node:
                     hercules_instance._l0_clusters_ordered.append(l0_node)
                     hercules_instance._original_id_to_index[orig_id] = i 
                     restored_count += 1
                 else:
                     warnings.warn(f"Could not find loaded L0 cluster with original ID '{orig_id}' from saved order. Alignment might be affected.")
             hercules_instance._log(f"  Restored ordered list of {restored_count} L0 clusters.", level=2)
             if restored_count != len(l0_map_temp):
                  warnings.warn("Mismatch between number of restored ordered L0 clusters and total L0 clusters found.")
        elif clusters_by_id: 
             warnings.warn("Saved model state missing '_l0_original_ids_ordered'. Attempting fallback L0 ordering (sorted by original_id). Alignment not guaranteed.")
             l0_nodes = sorted([c for c in clusters_by_id.values() if c.level == 0], key=lambda c: str(c.original_id))
             hercules_instance._l0_clusters_ordered = l0_nodes
             hercules_instance._original_id_to_index = {c.original_id: i for i, c in enumerate(l0_nodes)}
             hercules_instance._log(f"  Performed fallback L0 cluster ordering for {len(l0_nodes)} clusters.", level=2)

        print(f"Hercules model loaded successfully. Restored {len(clusters_by_id)} clusters ({len(top_clusters)} top-level). Max level: {hercules_instance.max_level}.")
        print("Note: Reducers (e.g., PCA) are reconstructed but UNTRAINED.")

        return hercules_instance, top_clusters

    def get_l0_representations(self) -> List[Optional[np.ndarray]]:
        """
        Retrieves the representation vectors for all Level 0 items in their original order.

        The specific vector returned for each item depends on the 'representation_mode'
        used during clustering ('direct' uses original item embeddings/scaled numerics,
        'description' uses embeddings of the generated L0 descriptions/captions).

        Returns:
            A list where each element is the representation vector (numpy array) for the
            corresponding original L0 item, or None if the item had no valid representation.
            Returns an empty list if clustering has not been run or failed.
        """
        self._log("Retrieving ordered L0 representations...", level=2)

        if not hasattr(self, '_l0_clusters_ordered') or not self._l0_clusters_ordered:
            warnings.warn("Cannot retrieve L0 representations: Clustering has not been run or failed.")
            return []

        representations = []
        for i, l0_cluster in enumerate(self._l0_clusters_ordered):
            vec = l0_cluster.representation_vector
            if vec is not None and not np.all(np.isfinite(vec)):
                 warnings.warn(f"L0 cluster index {i} (ID: {l0_cluster.original_id}) representation vector is non-finite. Returning None for this item.")
                 representations.append(None)
            elif vec is None:
                 self._log(f"L0 cluster index {i} (ID: {l0_cluster.original_id}) has no representation vector. Returning None.", level=3)
                 representations.append(None)
            else:
                 representations.append(vec)

        num_retrieved = sum(1 for r in representations if r is not None)
        num_none = len(representations) - num_retrieved
        self._log(f"Retrieved {num_retrieved} L0 representations ({num_none} are None) out of {len(self._l0_clusters_ordered)} total L0 items.", level=1)

        return representations

    def get_function_info(self) -> dict:
        """Returns metadata about functions used when the model was created."""
        return getattr(self, '_function_metadata', {}).copy()

    def print_cluster_counts_per_level(self):
        """
        Prints the number of clusters found at each level of the hierarchy (L1+).
        """
        print("\n--- Cluster Counts Per Level ---")
        if not self._all_clusters_map or self._max_level == 0:
            print("No hierarchy data available or only Level 0 exists.")
            if self._l0_clusters_ordered:
                 print(f"Level 0: {len(self._l0_clusters_ordered)} items (original data points)")
            return

        counts_by_level = {}
        for cluster in self._all_clusters_map.values():
            level = cluster.level
            if level > 0: 
                counts_by_level[level] = counts_by_level.get(level, 0) + 1

        if self._l0_clusters_ordered:
            print(f"Level 0: {len(self._l0_clusters_ordered)} items (original data points)")

        if not counts_by_level:
            print("No clusters found at levels 1 or higher.")
            return

        for level in range(1, self._max_level + 1):
            num_clusters = counts_by_level.get(level, 0) 
            print(f"Level {level}: {num_clusters} clusters")

# --- Example Usage ---
if __name__ == '__main__':
    print(f"Running Hercules Minimalist Example (Version: {Hercules.HERCULES_VERSION})...")

    # --- 1. Configuration for this minimalist example ---
    EXAMPLE_DATA_TYPE = 'numeric_numpy'
    REPRESENTATION_MODE = 'direct'
    USE_AUTO_K = True # Set to False to use fixed_hierarchy_levels
    SAVE_AND_LOAD_MODEL = False # Set to True to test saving/loading
    VERBOSE_LEVEL = 1

    N_ITEMS = 50
    N_FEATURES = 4
    N_TRUE_CLUSTERS = 3 # For dummy labels, and for fixed_k if USE_AUTO_K is False

    MODEL_FILENAME = f"hercules_minimal_{EXAMPLE_DATA_TYPE}_{REPRESENTATION_MODE}.json"
    RUN_DETAILS_DIR = f"hercules_minimal_run_{EXAMPLE_DATA_TYPE}"

    # --- 2. Prepare Dummy Data ---
    print(f"\n--- Preparing dummy data: {EXAMPLE_DATA_TYPE} ---")
    input_data = None
    dummy_labels = [] # For evaluation reference

    all_data_np_list = []
    means = np.random.uniform(-3, 3, (N_TRUE_CLUSTERS, N_FEATURES))
    items_per_group = N_ITEMS // N_TRUE_CLUSTERS
    for i in range(N_TRUE_CLUSTERS):
        group_data = np.random.normal(means[i], 0.8, (items_per_group, N_FEATURES))
        all_data_np_list.append(group_data)
        dummy_labels.extend([i] * items_per_group)
    
    remaining_items = N_ITEMS - len(dummy_labels)
    if remaining_items > 0:
        group_data = np.random.normal(means[N_TRUE_CLUSTERS - 1], 0.8, (remaining_items, N_FEATURES))
        all_data_np_list.append(group_data)
        dummy_labels.extend([N_TRUE_CLUSTERS - 1] * remaining_items)
    
    input_data = np.concatenate(all_data_np_list)
    print(f"Using numeric data as 2D NumPy array, shape: {input_data.shape}")

    numeric_metadata_input = {
        i: {'name': f'Sensor_{i+1}', 'unit': 'V', 'description': f'Reading from sensor {i+1}'}
        for i in range(N_FEATURES)
    }
    print(f"Defined dummy numeric metadata for {N_FEATURES} features.")

    # --- 3. Hercules Setup & Clustering ---
    hercules_instance = None
    top_level_clusters = []

    if SAVE_AND_LOAD_MODEL and os.path.exists(MODEL_FILENAME):
        print(f"\n--- Attempting to load existing model from {MODEL_FILENAME} ---")
        hercules_instance, top_level_clusters = Hercules.load_model(
            MODEL_FILENAME,
            text_embedding_client=_dummy_text_embedding_function,
            llm_client=_dummy_llm_function,
            image_embedding_client=_dummy_image_embedding_function,
            image_captioning_client=_dummy_image_captioning_function
        )
        if not hercules_instance:
            print(f"Failed to load model. Will train a new one.")
            SAVE_AND_LOAD_MODEL = False
    
    if not hercules_instance: # Train a new model
        print("\n--- Instantiating and training new Hercules model ---")
        cluster_counts_arg = None if USE_AUTO_K else [N_TRUE_CLUSTERS, 2] # Example fixed counts
        
        hercules_instance = Hercules(
            level_cluster_counts=cluster_counts_arg,
            text_embedding_client=_dummy_text_embedding_function,
            llm_client=_dummy_llm_function,
            image_embedding_client=_dummy_image_embedding_function,
            image_captioning_client=_dummy_image_captioning_function,
            representation_mode=REPRESENTATION_MODE,
            auto_k_method='silhouette' if USE_AUTO_K else Hercules.DEFAULT_AUTO_K_METHOD,
            auto_k_max=8 if USE_AUTO_K else Hercules.DEFAULT_AUTO_K_MAX,
            reduction_methods=['pca'], # Optional: for visualizations or some metrics
            n_reduction_components=2,
            random_state=42,
            verbose=VERBOSE_LEVEL,
            save_run_details=True, # Logs prompts etc.
            run_details_dir=RUN_DETAILS_DIR,
            use_llm_for_l0_descriptions=(EXAMPLE_DATA_TYPE == 'text') # Example: only use for text L0
        )

        print(f"\nStarting clustering ({EXAMPLE_DATA_TYPE}, {REPRESENTATION_MODE} mode)...")
        start_time = time.time()
        top_level_clusters = hercules_instance.cluster(
            input_data,
            topic_seed=f"Insights from {EXAMPLE_DATA_TYPE} data", # Optional
            numeric_metadata=numeric_metadata_input
        )
        end_time = time.time()
        print(f"Clustering completed in {end_time - start_time:.2f} seconds.")

        if SAVE_AND_LOAD_MODEL and top_level_clusters and hercules_instance:
            print(f"\n--- Saving model to {MODEL_FILENAME} ---")
            hercules_instance.save_model(MODEL_FILENAME, top_level_clusters)

    # --- 4. Print Basic Results ---
    if top_level_clusters and hercules_instance:
        print(f"\n--- Results: {len(top_level_clusters)} Top-Level Clusters (Max Level: {hercules_instance.max_level}) ---")
        hercules_instance.print_cluster_counts_per_level()
        
        # Print hierarchy of the first top-level cluster (or a few)
        for i, cluster in enumerate(sorted(top_level_clusters, key=lambda c: c.num_items, reverse=True)[:2]): # Print first 2 largest
            print(f"\n--- Top Level Cluster {i+1} (ID: {cluster.id}, Size: {cluster.num_items}) ---")
            # Set print_level_0=True if you want to see L0 items in the hierarchy printout
            cluster.print_hierarchy(max_depth=2, indent_increment=2, print_level_0=False)
    else:
        print("\nClustering did not produce results or model loading failed.")

    # --- 5. Optional: Evaluation (if ground truth is available) ---
    if hercules_instance and hercules_instance.max_level > 0 and dummy_labels:
        print("\n--- Evaluating Clustering (Level 1) ---")
        eval_level = 1 # Evaluate the first level of clustering
        if eval_level <= hercules_instance.max_level:
            eval_results = hercules_instance.evaluate_level(
                eval_level,
                ground_truth_labels=dummy_labels,
                calculate_llm_internal_metrics=True # Set to False to speed up if not needed
            )
            print(f"Level {eval_level} Evaluation Metrics:")
            for metric, value in eval_results.items():
                if isinstance(value, float): print(f"  {metric}: {value:.4f}")
                elif not metric.endswith("_error") and metric != "error": print(f"  {metric}: {value}")
                elif value: print(f"  ERROR ({metric}): {value}") # Print if error value is not None/False
        else:
            print(f"Cannot evaluate level {eval_level}, max level achieved is {hercules_instance.max_level}")

    # --- 6. Optional: Membership DataFrame ---
    if hercules_instance and hercules_instance.max_level >= 0: # Can run even if only L0 exists
        print("\n--- Cluster Membership DataFrame (First 5 rows) ---")
        # For numeric_numpy, L0 descriptions might be useful if generated
        l0_details = ['description'] if EXAMPLE_DATA_TYPE == 'numeric_numpy' else None
        membership_df = hercules_instance.get_cluster_membership_dataframe(include_l0_details=l0_details)
        if not membership_df.empty:
            print(membership_df.head())
            # df_path = os.path.join(RUN_DETAILS_DIR, "membership_df.csv")
            # membership_df.to_csv(df_path)
            # print(f"Full membership DataFrame saved to {df_path}")
        else:
            print("Could not generate membership DataFrame.")

    print("\nMinimalist example finished.")
