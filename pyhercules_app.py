import base64
import contextlib
import io
import json
import math
import os
import shutil
import sys
import time
import traceback
import uuid
import warnings
import zipfile
from collections import Counter, deque
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()
from dash import (ALL, MATCH, ClientsideFunction, Input, Output, State,
                  callback, callback_context, dash_table, dcc, html, no_update)

# --- Import Hercules and its components ---
# Make sure hercules.py is in the same directory or Python path
try:
    from pyhercules import IMAGE_EXTENSIONS, Cluster, Hercules
    # Keep specific imports needed for evaluate_level reference
    from sklearn.metrics import (adjusted_rand_score,
                                 calinski_harabasz_score,
                                 davies_bouldin_score,
                                 homogeneity_completeness_v_measure,
                                 normalized_mutual_info_score,
                                 silhouette_score)
    print("Successfully imported Hercules and sklearn metrics.")
except ImportError as e:
    print(f"FATAL: A required library is not installed: {e}", file=sys.stderr)
    print("Please ensure 'hercules' and 'scikit-learn' are installed to run this application.", file=sys.stderr)
    sys.exit(1)

# --- Import Hercules Functions ---
try:
    import pyhercules_functions as hf
    print("Successfully imported hercules_functions.")
except ImportError:
    print("FATAL: Failed to import 'pyhercules_functions.py'.", file=sys.stderr)
    print("Please ensure 'pyhercules_functions.py' is in the same directory or Python path.", file=sys.stderr)
    sys.exit(1)

# --- Constants ---
UPLOAD_DIRECTORY = "hercules_uploads"
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)
TABULAR_EXTENSIONS = {'.csv', '.tsv', '.json', '.jsonl', '.parquet', '.xls', '.xlsx'}
TEXT_EXTENSIONS = {'.txt'}
NON_TABULAR_EXTENSIONS = IMAGE_EXTENSIONS.union(TEXT_EXTENSIONS)
GROUND_TRUTH_EXTENSIONS = {'.csv', '.tsv', '.json'}
L0_DISPLAY_THRESHOLD = 5000
DEFAULT_COLOR_SEQUENCE = px.colors.qualitative.Plotly
DEFAULT_OPACITY_NO_SELECTION = 0.8
DEFAULT_OPACITY_UNRELATED = 0.3
PREVIEW_TABLE_PAGE_SIZE = 10
PREVIEW_MAX_FILES_LIST = 10
PREVIEW_NROWS = 50
RUN_LOG_FILENAME = "run_log.txt"

# --- App Initialization ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX, dbc.icons.BOOTSTRAP], suppress_callback_exceptions=True)
app.title = "Hercules Clustering Explorer"
server = app.server

# =============================================================================
# Helper Functions
# =============================================================================
def get_download_filename(base: str, ext: str, session_id_str: Optional[str] = None) -> str:
    """Generates a filename for downloads with a timestamp and optional session ID."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if session_id_str:
        return f"{base}_{session_id_str}_{ts}.{ext}"
    return f"{base}_{ts}.{ext}"

def parse_cluster_counts(counts_str: str) -> Optional[List[int]]:
    if not counts_str:
        return None
    try:
        counts = [int(c.strip()) for c in counts_str.split(',') if c.strip()]
        if not counts or any(c <= 0 for c in counts):
            raise ValueError("Cluster counts must be positive integers.")
        return counts
    except ValueError:
        return None

def get_file_list(directory_path):
    filepaths = []
    for root, _, files in os.walk(directory_path):
        for filename in files:
            # FIX: Explicitly ignore the run log file
            if not filename.startswith('.') and filename != RUN_LOG_FILENAME:
                filepaths.append(os.path.join(root, filename))
    return filepaths

def infer_data_and_prepare(session_dir: str) -> Tuple[Any, Optional[str], Optional[str]]:
    """Infers data type and prepares input from a directory of files."""
    print(f"Note: Using directory data inference for {session_dir}.")
    all_files = get_file_list(session_dir)
    if not all_files:
        return None, None, "No valid (non-hidden) data files found in session directory."
    image_files = [f for f in all_files if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS]
    text_files = [f for f in all_files if os.path.splitext(f)[1].lower() in TEXT_EXTENSIONS]
    tabular_files = [f for f in all_files if os.path.splitext(f)[1].lower() in TABULAR_EXTENSIONS]

    # Prefer homogeneous types if all files match
    if len(image_files) == len(all_files) and image_files:
        return {os.path.basename(f): f for f in image_files}, 'image', None
    elif len(text_files) == len(all_files) and text_files:
        try:
            text_data = {os.path.basename(f): open(f, 'r', encoding='utf-8', errors='ignore').read() for f in text_files}
            return text_data, 'text', None
        except Exception as e:
            return None, None, f"Error reading text file(s): {e}"
    elif len(tabular_files) == 1 and len(all_files) == 1:
        df, err = load_tabular_data(tabular_files[0], header=0, index_col=None, nrows=None) # Default load for inference
        if err:
            return None, None, f"Error loading single tabular file: {err}"
        if df is None or df.empty:
            return None, None, "Single tabular file loaded as empty."
        df_numeric = df.select_dtypes(include=np.number)
        if df_numeric.empty:
            return None, None, "Single tabular file has no numeric columns."
        return df_numeric, 'numeric', None

    # Handle mixed types or specific primary types if unambiguous
    if image_files and not text_files and not tabular_files:
        return {os.path.basename(f): f for f in image_files}, 'image', None
    if text_files and not image_files and not tabular_files:
         try:
             text_data = {os.path.basename(f): open(f, 'r', encoding='utf-8', errors='ignore').read() for f in text_files}
             return text_data, 'text', None
         except Exception as e:
             return None, None, f"Error reading text file(s): {e}"

    return None, None, f"Could not reliably infer data type from mixed files: {[os.path.basename(f) for f in all_files[:5]]}..."


def load_tabular_data(filepath: str, header: Optional[int], index_col: Optional[Union[int, str]], nrows: Optional[int] = None) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    if not filepath or not os.path.exists(filepath):
        return None, "File path invalid or file does not exist."
    ext = os.path.splitext(filepath)[1].lower()
    read_opts = {'nrows': nrows, 'header': header}
    index_col_parsed: Optional[Union[int, str]] = None
    if index_col is not None:
        index_str = str(index_col).strip()
        if index_str and index_str.lower() != 'none':
             try:
                 index_col_parsed = int(index_str)
             except (ValueError, TypeError):
                 index_col_parsed = index_str
    try:
        df = None
        print(f"Loading tabular: '{os.path.basename(filepath)}' (H={header}, I='{index_col_parsed}', NRows={nrows})")
        if ext in ['.csv', '.tsv']:
            csv_opts = read_opts.copy()
            if index_col_parsed is not None:
                csv_opts['index_col'] = index_col_parsed
            try:
                df = pd.read_csv(filepath, sep=None, engine='python', **csv_opts)
            except (ValueError, pd.errors.ParserError) as ve:
                 print(f"Warning: pandas sep=None failed for {filepath} ({ve}). Trying common separators.")
                 try:
                     df = pd.read_csv(filepath, sep=',', **csv_opts)
                 except Exception:
                     try:
                         df = pd.read_csv(filepath, sep='\t', **csv_opts)
                     except Exception as final_csv_err:
                         raise ValueError(f"Failed with sep=',' and sep='\\t'. Original: {ve}. Last: {final_csv_err}") from final_csv_err
            except Exception as e:
                print(f"Error reading CSV/TSV {filepath}: {e}")
                raise
        elif ext in ['.json', '.jsonl']:
            df_loaded = pd.read_json(filepath, lines=(ext == '.jsonl'), nrows=nrows)
            if index_col_parsed is not None:
                 try:
                    if isinstance(index_col_parsed, str) and index_col_parsed in df_loaded.columns:
                        df_loaded = df_loaded.set_index(index_col_parsed, drop=True)
                    elif isinstance(index_col_parsed, int) and -len(df_loaded.columns) <= index_col_parsed < len(df_loaded.columns):
                        df_loaded = df_loaded.set_index(df_loaded.columns[index_col_parsed], drop=True)
                    else:
                        print(f"Warning: Cannot set index '{index_col_parsed}' for JSON.")
                 except Exception as idx_err:
                     print(f"Warning: Error setting index '{index_col_parsed}' for JSON: {idx_err}")
            df = df_loaded
        elif ext in ['.xls', '.xlsx']:
            excel_opts = read_opts.copy()
            if index_col_parsed is not None:
                excel_opts['index_col'] = index_col_parsed
            df = pd.read_excel(filepath, **excel_opts)
        elif ext == '.parquet':
            df_full = pd.read_parquet(filepath)
            if index_col_parsed is not None:
                 try:
                    if isinstance(index_col_parsed, str) and index_col_parsed in df_full.columns:
                        df_full = df_full.set_index(index_col_parsed, drop=True)
                    elif isinstance(index_col_parsed, int) and -len(df_full.columns) <= index_col_parsed < len(df_full.columns):
                        df_full = df_full.set_index(df_full.columns[index_col_parsed], drop=True)
                    elif index_col_parsed not in df_full.columns and index_col_parsed != df_full.index.name:
                        print(f"Warning: Index '{index_col_parsed}' not in Parquet. Using existing '{df_full.index.name}'.")
                 except Exception as idx_err:
                     print(f"Warning: Error setting index '{index_col_parsed}' for Parquet: {idx_err}.")
            df = df_full.head(nrows) if nrows is not None else df_full
        else:
            return None, f"Unsupported file extension: {ext}"
        if df is None:
            return None, "Failed to load data (DataFrame is None)."
        print(f"Loaded '{os.path.basename(filepath)}' - Shape: {df.shape}, Index: {df.index.name}")
        return df, None
    except Exception as e:
        tb_str = traceback.format_exc()
        print(f"Error loading tabular file '{os.path.basename(filepath)}': {e}\n{tb_str}", file=sys.stderr)
        return None, f"Error loading file: {e}"

def parse_ground_truth(content_bytes: bytes, filename: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    ext = os.path.splitext(filename)[1].lower()
    gt_dict = None
    error = None
    try:
        if ext in ['.csv', '.tsv']:
            df = pd.read_csv(io.BytesIO(content_bytes), sep=(',' if ext == '.csv' else '\t'), header=0, dtype=str)
            id_col = next((col for col in df.columns if col.lower() == 'id'), None)
            label_col = next((col for col in df.columns if col.lower() == 'label'), None)
            if id_col is None or label_col is None:
                 if len(df.columns) >= 2:
                     if len(df.columns) > 2:
                         warnings.warn(f"GT file '{filename}' missing 'id'/'label'. Assuming first='{df.columns[0]}'=ID, second='{df.columns[1]}'=Label.")
                     id_col, label_col = df.columns[0], df.columns[1]
                 else:
                     raise ValueError(f"CSV/TSV must have >= 2 columns. Found {len(df.columns)}.")
            gt_dict = {}
            skipped_rows = 0
            for _, row in df.iterrows():
                row_id = row[id_col]
                row_label = row[label_col]
                if pd.notna(row_id) and str(row_id).strip() and pd.notna(row_label):
                    gt_dict[str(row_id).strip()] = row_label
                else:
                    skipped_rows += 1
            if skipped_rows > 0:
                warnings.warn(f"Skipped {skipped_rows} rows in '{filename}' due to missing/empty ID or Label.")
        elif ext == '.json':
            data = json.load(io.BytesIO(content_bytes))
            if not isinstance(data, dict):
                raise ValueError("JSON must contain a dictionary {id: label}.")
            gt_dict = {}
            skipped_keys = 0
            for k, v in data.items():
                key_str = str(k).strip()
                if key_str and pd.notna(v):
                    gt_dict[key_str] = v
                else:
                    skipped_keys += 1
            if skipped_keys > 0:
                warnings.warn(f"Skipped {skipped_keys} entries in '{filename}' due to missing/empty key or null value.")
        else:
            error = f"Unsupported GT extension: {ext}. Use .csv, .tsv, or .json."
        if gt_dict is not None and not gt_dict:
            warnings.warn(f"Parsed GT file '{filename}' resulted in an empty dictionary.")
    except Exception as e:
        error = f"Error parsing GT file '{filename}': {e}"
        print(f"{error}\n{traceback.format_exc()}", file=sys.stderr)
        gt_dict = None
    return gt_dict, error

def reconstruct_cluster_map_from_data(clusters_data_list: List[Dict]) -> Dict[int, Cluster]:
    if not clusters_data_list:
        return {}
    clusters_by_id = {}
    max_loaded_id = -1
    Cluster.reset_id_counter()
    for cluster_data in clusters_data_list:
        try:
            cluster = Cluster.from_dict(cluster_data)
            clusters_by_id[cluster.id] = cluster
            max_loaded_id = max(max_loaded_id, cluster.id)
        except Exception as e:
            print(f"Error deserializing cluster data (ID: {cluster_data.get('id', 'N/A')}): {e}", file=sys.stderr)
            continue
    Cluster._next_internal_id = max(Cluster._next_internal_id, max_loaded_id + 1)
    if clusters_by_id:
        try:
            Cluster.link_hierarchy(clusters_by_id)
        except Exception as e:
            print(f"Error linking hierarchy: {e}", file=sys.stderr)
    return clusters_by_id

def get_cluster_by_id(cluster_id, cluster_map):
    try:
        return cluster_map.get(int(cluster_id)) if cluster_id is not None else None
    except (ValueError, TypeError):
        return None

def get_ancestors(cluster_id, cluster_map):
    ancestors = set()
    try:
         current_id = int(cluster_id)
         current = cluster_map.get(current_id)
         while current and current.parent and current.parent.id in cluster_map:
             parent = current.parent
             ancestors.add(parent.id)
             current = parent
    except (ValueError, TypeError):
        pass
    return ancestors

def get_descendants(cluster_id, cluster_map):
    descendants = set()
    q = deque()
    try:
        start_node = cluster_map.get(int(cluster_id))
        if start_node:
            q.append(start_node)
            descendants.add(start_node.id)
    except (ValueError, TypeError):
        return descendants
    visited = {start_node.id} if start_node else set()
    while q:
        curr_cluster = q.popleft()
        if curr_cluster.children:
             for child in curr_cluster.children:
                  if child and child.id in cluster_map and child.id not in visited:
                      visited.add(child.id)
                      descendants.add(child.id)
                      q.append(child)
    return descendants

def generate_hierarchy_list_group_item(cluster: Cluster, cluster_map: Dict[int, Cluster], show_l0: bool, max_depth: int = 10, current_depth: int = 0):
    if not cluster or cluster.id not in cluster_map or current_depth > max_depth:
        return None
    if not show_l0 and cluster.level == 0:
        return None
    item_label = f"L{cluster.level} ID:{cluster.id} ({cluster.num_items}) - {cluster.title or '(No Title)'}"
    item_button = dbc.Button(item_label, id={'type': 'select-cluster-button', 'index': cluster.id}, color="link", size="sm", className="p-0 text-start w-100 d-inline-block text-truncate", title=item_label, n_clicks=0, style={'maxWidth': '100%'})
    children_content = []
    valid_children = sorted([c for c in cluster.children if c and c.id in cluster_map], key=lambda c: (-c.num_items, c.id))
    if valid_children and current_depth < max_depth:
        child_items = [generate_hierarchy_list_group_item(child, cluster_map, show_l0, max_depth, current_depth + 1) for child in valid_children]
        child_items = [item for item in child_items if item]
        if child_items:
            children_content.append(dbc.ListGroup(child_items, flush=True, className="mt-1 ms-3 border-start ps-2"))
    return dbc.ListGroupItem([item_button] + children_content, className="border-0 px-0 py-1")

def scale_bubble_size(num_items, level, all_clusters_map, min_size=10, max_size=100, l0_size=10, scale='sqrt'):
    if level == 0:
        return l0_size
    if num_items <= 0:
        return min_size
    max_n_at_level = 1
    if all_clusters_map:
         items_at_level = [c.num_items for c in all_clusters_map.values() if c.level == level and c.num_items > 0]
         if items_at_level:
             max_n_at_level = max(items_at_level)
    max_n_at_level = max(1, max_n_at_level)
    if scale == 'sqrt':
        scale_factor = math.sqrt(max(0, min(1, num_items / max_n_at_level)))
    elif scale == 'linear':
        scale_factor = max(0, min(1, num_items / max_n_at_level))
    else:
        scale_factor = math.log1p(num_items) / math.log1p(max_n_at_level) # log scale
    scaled = min_size + (max_size - min_size) * max(0, min(1, scale_factor))
    return max(min_size, min(scaled, max_size))

def format_numeric_stats(stats_dict, precision=2):
    if not stats_dict:
        return html.P("No numeric statistics available.", className="small text-muted")
    items = []
    num_fmt = f"{{:.{precision}f}}"
    sorted_vars = sorted(stats_dict.keys())
    for var in sorted_vars:
        data = stats_dict[var]
        if not isinstance(data, dict) or 'mean' not in data:
            items.append(dbc.Row([
                dbc.Col(html.Strong(f"{var}:"), width="auto"),
                dbc.Col("Invalid/incomplete stats", className="text-muted small")
            ]))
            continue
        stat_cols = [dbc.Col(html.Strong(f"{var}:"), width="auto", className="pe-1 text-nowrap")]
        show_range = data.get('std', 0) > 1e-9 and 'min' in data and 'max' in data and data['min'] != data['max']
        if show_range:
            mean_std = f"Mean: {num_fmt.format(data.get('mean', float('nan')))} (±{num_fmt.format(data['std'])})"
            range_str = f"Range: [{num_fmt.format(data['min'])} - {num_fmt.format(data['max'])}]"
            stat_cols.extend([dbc.Col(mean_std, width=12, md=6, className="ps-md-2"), dbc.Col(range_str, width=12, md=6, className="ps-md-2")])
        else:
            val_str = f"{num_fmt.format(data.get('mean', data.get('min', float('nan'))))}" if not np.isnan(data.get('mean', data.get('min', float('nan')))) else "N/A"
            stat_cols.append(dbc.Col(val_str, width=True, className="ps-md-2"))
        items.append(dbc.Row(stat_cols, className="mb-1 align-items-center gx-2"))
    return html.Div(items)

def create_summary_content(cluster_id, cluster_map, config_data, state_data, eval_results, default_show_l0: bool, run_log: str, l0_clusters_ordered=None):
    cluster = get_cluster_by_id(cluster_id, cluster_map)
    if not cluster: # Overall Summary
        total_items = state_data.get("num_l0_items", 0)
        max_level_achieved = state_data.get("max_level_achieved", 0)
        input_data_type = state_data.get("input_data_type", "N/A").capitalize()
        top_level_clusters = [c for c in cluster_map.values() if c.parent is None or c.parent.id not in cluster_map]
        def get_config_str(key, default="N/A", data_dict=None):
             val = (data_dict if data_dict is not None else config_data).get(key, default)
             return ", ".join(map(str, val)) if isinstance(val, list) and val else str(val) if val not in [None, 'N/A'] else default
        config_items_display = [html.H6("Run Configuration", className="mt-2 mb-2 fw-bold")]
        def add_config_row(label, value, title=None, width1=6, width2=6):
             value_col = html.Span(value, title=title or str(value), className='d-inline-block text-truncate' if title else '', style={'maxWidth': '100%'} if title else {})
             config_items_display.append(dbc.Row([dbc.Col(html.Strong(label), width=width1), dbc.Col(value_col, width=width2)], className="mb-1 small"))
        data_source_desc = state_data.get("data_source_description", "N/A")
        add_config_row("Data Source:", data_source_desc, title=data_source_desc)
        add_config_row("Input Data Type:", input_data_type)
        tabular_params = state_data.get("tabular_load_params")
        if tabular_params and input_data_type.lower() == 'numeric':
            h_str = "First Row (0)" if tabular_params.get('header') == 0 else "None" if tabular_params.get('header') is None else str(tabular_params.get('header'))
            i_str = "None" if tabular_params.get('index_col') is None or str(tabular_params.get('index_col')).strip() == "" else str(tabular_params.get('index_col'))
            add_config_row("Tabular Header:", h_str)
            add_config_row("Tabular Index:", i_str)
        add_config_row("Rep. Mode:", get_config_str('representation_mode', data_dict=config_data))
        add_config_row("Hierarchy:", f"{get_config_str('level_cluster_counts', data_dict=config_data)} (Levels={len(config_data.get('level_cluster_counts',[]))})")
        add_config_row("Topic Seed:", get_config_str('topic_seed', default="(Not Set)", data_dict=config_data))
        add_config_row("Text Embedder:", get_config_str('selected_text_embedder', default='N/A', data_dict=config_data))
        add_config_row("LLM:", get_config_str('selected_llm', default='N/A', data_dict=config_data))
        if input_data_type.lower() == 'image':
             add_config_row("Image Embedder:", get_config_str('selected_image_embedder', default='N/A', data_dict=config_data))
             add_config_row("Image Captioner:", get_config_str('selected_image_captioner', default='N/A', data_dict=config_data))
        for space, dim in state_data.get('embedding_dims', {}).items():
             if dim:
                 add_config_row(f"{space.replace('_', ' ').capitalize()} Dim:", str(dim))
        eval_content = [html.H6("Evaluation Metrics", className="mt-3 mb-2 fw-bold")]
        if eval_results:
            try:
                sorted_levels = sorted([int(k) for k in eval_results.keys()])
            except ValueError:
                sorted_levels = list(eval_results.keys()) # Should be strings if int conversion fails
            for level in sorted_levels:
                results = eval_results.get(str(level), {}) # Ensure key is string
                if not results:
                    continue
                eval_content.append(html.Strong(f"Level {level}:", className="d-block mb-1 small"))
                if "error" in results:
                    eval_content.append(html.P(f"Error: {results['error']}", className="text-danger small ms-2 mb-2"))
                else:
                    metric_items = []
                    def add_metric_row(label, value, color="body", fmt="{:.3f}"):
                        if value is not None and value != 'N/A':
                            val_str = fmt.format(value) if isinstance(value, (int, float)) else str(value)
                            metric_items.append(dbc.Row([dbc.Col(label, width=7), dbc.Col(val_str, width=5, className=f"text-{color} text-end")], justify="between", className='gx-2 mb-0 small'))
                    add_metric_row("Num Clusters:", results.get('num_clusters'), fmt="{}")
                    gt_provided = results.get('ground_truth_provided', False)
                    add_metric_row("GT Used:", "Yes" if gt_provided else "No", color="info" if gt_provided else "muted", fmt="{}")
                    add_metric_row("Items Eval'd:", results.get('num_items_evaluated'), fmt="{}")
                    if gt_provided:
                         for metric in ['adjusted_rand_score', 'normalized_mutual_info_score', 'v_measure_score', 'homogeneity_score', 'completeness_score']:
                             add_metric_row(metric.replace('_', ' ').title()+":", results.get(metric))
                         if "supervised_metrics_error" in results:
                             metric_items.append(html.P(f"Note: {results['supervised_metrics_error']}", className="text-warning small mt-1 mb-0"))
                         if "gt_warnings" in results:
                             metric_items.append(html.P(f"GT Note: {results['gt_warnings']}", className="text-warning small mt-1 mb-0"))
                    for metric in ['silhouette_score_on_prev_level', 'davies_bouldin_score_on_prev_level', 'calinski_harabasz_score_on_prev_level', 'silhouette_score_on_l0', 'davies_bouldin_score_on_l0', 'calinski_harabasz_score_on_l0']:
                        add_metric_row(metric.replace('_', ' ').title()+":", results.get(metric))
                    if "unsupervised_metrics_error" in results:
                        metric_items.append(html.P(f"Note: {results['unsupervised_metrics_error']}", className="text-warning small mt-1 mb-0"))
                    eval_content.append(html.Div(metric_items, className="ms-2 mb-2 border-start ps-2"))
        else:
            eval_content.append(html.P("Evaluation metrics not available.", className="small text-muted"))
        hierarchy_content = [html.H6("Cluster Hierarchy", className="mt-3 mb-2 fw-bold")]
        if top_level_clusters:
            hierarchy_items = [generate_hierarchy_list_group_item(c, cluster_map, show_l0=default_show_l0) for c in top_level_clusters]
            hierarchy_items = [item for item in hierarchy_items if item]
            list_group_children = hierarchy_items or [html.P("No items to display (or L0 hidden).", className="small text-muted mt-1")]
            hierarchy_display = dbc.ListGroup(list_group_children, flush=True, style={'maxHeight': '400px', 'overflowY': 'auto'}, id='hierarchy-list-group')
            hierarchy_content.append(hierarchy_display)
            if hierarchy_items:
                hierarchy_content.append(html.P("Click cluster ID to select.", className="small text-muted mt-2"))
        else:
            hierarchy_content.append(html.P("No cluster hierarchy generated.", className="small text-muted mt-1"))

        downloads_section = html.Div([
            html.H6("Download Results", className="mt-4 mb-2 fw-bold"),
            dbc.Button([html.I(className="bi bi-table me-1"), "Membership DF (CSV)"], id="btn-download-membership-csv", color="info", outline=True, size="sm", className="me-2 mb-2"),
            dbc.Button([html.I(className="bi bi-graph-up me-1"), "Evaluation (JSON)"], id="btn-download-evaluation-json", color="info", outline=True, size="sm", className="me-2 mb-2"),
            dbc.Button([html.I(className="bi bi-diagram-3 me-1"), "Hierarchy (TXT)"], id="btn-download-hierarchy-txt", color="info", outline=True, size="sm", className="mb-2"),
        ], className="mt-3 border-top pt-3")

        log_accordion_item = dbc.AccordionItem(
            title="View Run Log",
            children=html.Div([
                dbc.Button(
                    [html.I(className="bi bi-download me-1"), "Download Log"],
                    id="btn-download-run-log",
                    color="info",
                    outline=True,
                    size="sm",
                    className="float-end mb-2"
                ),
                html.Pre(
                    run_log,
                    style={
                        'whiteSpace': 'pre-wrap',
                        'wordBreak': 'break-all',
                        'maxHeight': '400px',
                        'overflowY': 'scroll',
                        'border': '1px solid #ddd',
                        'padding': '10px',
                        'backgroundColor': '#f8f9fa',
                        'fontSize': '0.8em',
                        'clear': 'both'
                    }
                )
            ])
        )

        return dbc.CardBody([
            dbc.Row([dbc.Col(html.H4("Overall Summary", className="card-title mb-0"), width="auto")], justify="between", align="center", className="mb-3"),
                dbc.Row([dbc.Col(f"Total L0 Items: {total_items}", width=6), dbc.Col(f"Achieved Levels: {max_level_achieved}", width=6), dbc.Col(f"Top Clusters: {len(top_level_clusters)}", width=6), dbc.Col(f"Data Type: {input_data_type}", width=6)], className="mb-3 small"),
            html.Hr(className="my-3"),
            dbc.Accordion([
                dbc.AccordionItem(title="Run Configuration", children=html.Div(config_items_display, className="p-2")),
                dbc.AccordionItem(title="Evaluation Metrics", children=html.Div(eval_content, className="p-2")),
                dbc.AccordionItem(title="Cluster Hierarchy", children=html.Div(hierarchy_content, className="p-2")),
                log_accordion_item # <-- ADD THE NEW LOG ITEM HERE
            ], start_collapsed=True, flush=True, always_open=False, className="mt-3"),
            downloads_section
        ])
    else: # Selected cluster details
        body_content = [dbc.Row([dbc.Col(html.H4(f"Cluster Details", className="card-title mb-0"), width="auto"), dbc.Col(dbc.Button([html.I(className="bi bi-x-circle me-1"), "Clear"], id="clear-selection-button", color="warning", size="sm", n_clicks=0, outline=True), width="auto")], justify="between", align="center", className="mb-3")]
        body_content.append(dbc.Row([dbc.Col(f"ID: {cluster.id}", width=6, md=3), dbc.Col(f"Level: {cluster.level}", width=6, md=3), dbc.Col(f"Items: {cluster.num_items}", width=6, md=3), dbc.Col(f"Type: {getattr(cluster, 'original_data_type', 'N/A').capitalize()}", width=6, md=3)], className="mb-2 small"))
        if cluster.original_id is not None:
            body_content.append(dbc.Row([dbc.Col(f"Orig. ID: {cluster.original_id}", width=12)], className="mb-2 small text-muted"))
        body_content.append(html.Hr(className="my-2"))
        body_content.append(html.Strong("Title:", className="d-block mb-1"))
        body_content.append(html.P(cluster.title or html.Em("(No Title)"), className="mb-2"))
        body_content.append(html.Strong("Description:", className="d-block mb-1"))
        body_content.append(html.P(cluster.description or html.Em("(No Description)"), style={'maxHeight': '150px', 'overflowY': 'auto', 'fontSize': '0.9em'}, className="p-2 border rounded bg-light mb-3"))
        parent_content = [html.Strong("Parent:", className="me-2")]
        if cluster.parent and cluster.parent.id in cluster_map:
            p = cluster.parent
            parent_content.append(dbc.Button(f"ID: {p.id} - {p.title or ''}", id={'type': 'select-cluster-button', 'index': p.id}, color="link", size="sm", className="p-0", n_clicks=0))
        else:
            parent_content.append(html.Strong("None (Top Level)"))
        body_content.append(html.Div(parent_content, className="mb-2"))
        children_content = [html.Strong(f"Children ({len(cluster.children) if cluster.children else 0}):", className="me-2")]
        if cluster.children:
            child_links = []
            sorted_children = sorted([c for c in cluster.children if c and c.id in cluster_map], key=lambda c: (-c.num_items, c.id))[:10]
            for c_node in sorted_children:
                child_links.append(dbc.Button(f"ID:{c_node.id} ({c_node.num_items}) - {c_node.title or ''}", id={'type': 'select-cluster-button', 'index': c_node.id}, color="link", size="sm", className="p-0 d-block text-truncate w-100 text-start", n_clicks=0))
            if len([c for c in cluster.children if c and c.id in cluster_map]) > 10:
                child_links.append(html.Em(f"(top 10 of {len([c for c in cluster.children if c and c.id in cluster_map])})", className="small d-block mt-1"))
            children_content.append(html.Div(child_links, className="ms-2", style={'maxHeight': '150px', 'overflowY': 'auto'}))
        else:
            children_content.append(html.Em("None", className="small text-muted ms-1"))
        body_content.append(html.Div(children_content, className="mb-3"))
        try:
            try:
                sample_size = int(config_data.get('prompt_l0_sample_size', 3))
            except (ValueError, TypeError):
                sample_size = 3
            try:
                num_repr_vals = int(config_data.get('cluster_numeric_repr_max_vals', 5))
            except (ValueError, TypeError):
                num_repr_vals = 5
            try:
                num_precision = int(config_data.get('cluster_numeric_stats_precision', 2))
            except (ValueError, TypeError):
                num_precision = 2
            variable_names = state_data.get('variable_names')
            samples = cluster.get_representative_samples(
                sample_size=sample_size, 
                numeric_repr_max_vals=num_repr_vals, 
                numeric_repr_precision=num_precision,
                variable_names=variable_names
            ) if hasattr(cluster, 'get_representative_samples') else []
            if samples:
                body_content.append(html.Strong("Representative L0 Samples:", className="d-block mb-1"))
                sample_items = []
                for sample in samples:
                    sample_id = sample.get('id')
                    sample_data = sample.get('data', '(N/A)')
                    sample_preview = str(sample_data)[:100]
                    ellipsis = "..." if len(str(sample_data)) > 100 else ""
                    l0_cluster_obj = next((c for c in (l0_clusters_ordered or []) if str(c.original_id) == str(sample_id)), None)
                    if l0_cluster_obj:
                        sample_items.append(html.Li([dbc.Button(f"ID: {sample_id}", id={'type': 'select-cluster-button', 'index': l0_cluster_obj.id}, color="link", size="sm", className="p-0", n_clicks=0), f": {sample_preview}{ellipsis}"]))
                    else:
                        sample_items.append(html.Li(f"ID: {sample_id}: {sample_preview}{ellipsis}"))
                body_content.append(html.Ul(sample_items, className="list-unstyled ms-2 small"))
                body_content.append(html.Hr(className="my-2"))
            elif cluster.level == 0:
                body_content.append(html.Strong("L0 Item Data:", className="d-block mb-1"))
                raw_data = getattr(cluster, '_raw_item_data', '(N/A)')
                preview_str = np.array2string(raw_data, precision=3, separator=', ', max_line_width=100, threshold=20) if isinstance(raw_data, np.ndarray) else str(raw_data)
                body_content.append(html.Pre(preview_str[:200] + ("..." if len(preview_str) > 200 else ""), style={'fontSize': '0.85em', 'maxHeight': '100px', 'overflowY': 'auto'}, className="p-2 border rounded bg-light"))
                body_content.append(html.Hr(className="my-2"))
        except Exception as e:
            print(f"Error fetching/displaying samples for cluster {cluster.id}: {e}", file=sys.stderr)
            body_content.append(html.P(f"Error fetching samples: {e}", className="text-danger small"))
        cluster_data_type = getattr(cluster, 'original_data_type', 'unknown')
        if cluster_data_type == 'numeric':
            body_content.append(html.Strong("Numeric Statistics (Original Scale):", className="d-block mb-1"))
            try:
                var_names = state_data.get('variable_names')
                try:
                    stats_precision = int(config_data.get('cluster_numeric_stats_precision', 2))
                except (ValueError, TypeError):
                    stats_precision = 2
                stats = cluster.compute_numeric_statistics(variable_names=var_names, numeric_stats_precision=stats_precision) if hasattr(cluster, 'compute_numeric_statistics') else None
                if stats:
                    body_content.append(format_numeric_stats(stats, precision=stats_precision))
                else:
                    body_content.append(html.P("Stats N/A.", className="small text-muted"))
            except Exception as e:
                print(f"Error computing/displaying stats for cluster {cluster.id}: {e}", file=sys.stderr)
                body_content.append(html.P(f"Error computing stats.", className="text-danger small"))
        elif cluster_data_type == 'image' and cluster.level == 0 and hasattr(cluster, '_raw_item_data'):
            body_content.append(html.Strong("Image Identifier:", className="d-block mb-1"))
            body_content.append(html.P(f"{cluster._raw_item_data}", className="small ms-2"))
        return dbc.CardBody(body_content)

def create_bubble_chart_figure(level, selected_cluster_id, reduction_type, cluster_map, cluster_to_top_ancestor_map, top_level_color_map, l0_clusters_ordered):
    if level == 0 and len(l0_clusters_ordered) >= L0_DISPLAY_THRESHOLD:
        print(f"Skipping L0 plot generation: {len(l0_clusters_ordered)} items >= threshold {L0_DISPLAY_THRESHOLD}")
        return go.Figure(layout=go.Layout(title=dict(text=f"Level 0: Plot hidden ({len(l0_clusters_ordered)} items exceeds threshold {L0_DISPLAY_THRESHOLD})", font=dict(size=14)), xaxis={'visible': False}, yaxis={'visible': False}, height=150, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor='rgba(248, 249, 250, 0.5)', plot_bgcolor='rgba(0,0,0,0)', annotations=[dict(text=f"Plotting {len(l0_clusters_ordered)} items may cause performance issues.", showarrow=False, y=0.5, font=dict(size=11, color="#6c757d"))]))
    clusters_at_level = l0_clusters_ordered if level == 0 else [c for c in cluster_map.values() if c.level == level]
    if not clusters_at_level:
        return go.Figure(layout=go.Layout(title=f"Level {level}: No Clusters" if level > 0 else "Level 0: No Items", xaxis={'visible': False}, yaxis={'visible': False}, height=100, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'))
    plot_data = {'x': [], 'y': [], 'size': [], 'hover_texts': [], 'ids': [], 'colors': [], 'opacities': []}
    selected_id_int = int(selected_cluster_id) if selected_cluster_id is not None else None
    selected_cluster = get_cluster_by_id(selected_id_int, cluster_map)
    selected_cluster_level = selected_cluster.level if selected_cluster else -1
    ancestor_ids = get_ancestors(selected_id_int, cluster_map) if selected_id_int is not None else set()
    descendant_ids = get_descendants(selected_id_int, cluster_map) if selected_id_int is not None else set()
    on_marker_texts = []
    reduction_key = 'representation_vector_reductions' if reduction_type == 'representation_vector' else 'description_embedding_reductions'
    reduction_space = 'Representation' if reduction_type == 'representation_vector' else 'Description'
    available_methods = set()
    for c in clusters_at_level[:20]: # Check first few for available methods
        reductions = getattr(c, reduction_key, None)
        if reductions and isinstance(reductions, dict):
            available_methods.update(reductions.keys())
    plot_method = 'pca' if 'pca' in available_methods else (sorted(list(available_methods))[0] if available_methods else None)
    if not plot_method: # Fallback if primary reduction type not found for any method
         fallback_key = 'description_embedding_reductions' if reduction_key == 'representation_vector_reductions' else 'representation_vector_reductions'
         fallback_space = 'Description' if reduction_space == 'Representation' else 'Representation'
         fb_methods = set()
         for c in clusters_at_level[:20]:
             reductions = getattr(c, fallback_key, None)
             if reductions and isinstance(reductions, dict):
                 fb_methods.update(reductions.keys())
         plot_method = 'pca' if 'pca' in fb_methods else (sorted(list(fb_methods))[0] if fb_methods else None)
         if plot_method:
             reduction_key = fallback_key
             reduction_space = fallback_space
             print(f"Warning L{level}: Using fallback reduction '{plot_method.upper()}' from {fallback_space} space.")
    valid_count = 0
    missing_count = 0
    default_unselected_color = '#CCCCCC'
    MIN_SIZE_FOR_TEXT = 25
    MAX_TITLE_CHARS_ON_BUBBLE = 10
    for cluster in clusters_at_level:
        coords = None
        if plot_method:
            reductions = getattr(cluster, reduction_key, {})
            if isinstance(reductions, dict):
                coords = reductions.get(plot_method)
        if isinstance(coords, (list, np.ndarray)) and len(coords) >= 2 and not any(c is None or np.isnan(c) for c in coords[:2]):
            plot_data['x'].append(coords[0])
            plot_data['y'].append(coords[1])
            l0_bubble_size = 5 if level==0 and len(clusters_at_level) > 1000 else 10
            current_bubble_size = scale_bubble_size(cluster.num_items, cluster.level, cluster_map, l0_size=l0_bubble_size)
            plot_data['size'].append(current_bubble_size)
            hover = f"ID: {cluster.id}<br>Level: {level}<br>Items: {cluster.num_items}" + (f"<br>Orig. ID: {cluster.original_id}" if cluster.level == 0 and cluster.original_id else "") + f"<br>Title: {cluster.title or '(No Title)'}"
            plot_data['hover_texts'].append(hover)
            plot_data['ids'].append(cluster.id)
            valid_count += 1

            bubble_label_text = ""
            if level > 0 and current_bubble_size >= MIN_SIZE_FOR_TEXT:
                title = cluster.title or ""
                id_str = str(cluster.id)
                is_generic_title = title.lower().startswith("cluster ") and title[len("cluster "):].isdigit()
                if title and not is_generic_title:
                    if len(title) > MAX_TITLE_CHARS_ON_BUBBLE:
                        prefix = title[:MAX_TITLE_CHARS_ON_BUBBLE-1]
                        last_space = prefix.rfind(' ')
                        if last_space > (MAX_TITLE_CHARS_ON_BUBBLE -1) / 3:
                            bubble_label_text = title[:last_space] + "…"
                        else:
                            bubble_label_text = title[:MAX_TITLE_CHARS_ON_BUBBLE-1] + "…"
                    else:
                        bubble_label_text = title
                else:
                    bubble_label_text = id_str
            on_marker_texts.append(bubble_label_text)

            opacity = DEFAULT_OPACITY_UNRELATED
            color = default_unselected_color
            is_selected = (cluster.id == selected_id_int)
            is_descendant = (selected_id_int is not None and cluster.id in descendant_ids and cluster.id != selected_id_int)
            is_ancestor = (selected_id_int is not None and cluster.id in ancestor_ids)
            if is_selected:
                color = 'red'
                opacity = 1.0
            elif is_descendant:
                top_ancestor_id = cluster_to_top_ancestor_map.get(cluster.id)
                color = top_level_color_map.get(top_ancestor_id, 'orange')
                opacity = 0.8 if selected_cluster_level >= level else 0.6
            elif is_ancestor:
                color = 'purple'
                opacity = 0.8
            else:
                top_ancestor_id = cluster_to_top_ancestor_map.get(cluster.id)
                color = top_level_color_map.get(top_ancestor_id, default_unselected_color)
                opacity = DEFAULT_OPACITY_NO_SELECTION if selected_id_int is None else DEFAULT_OPACITY_UNRELATED
            plot_data['colors'].append(color)
            plot_data['opacities'].append(opacity)
        else:
            missing_count += 1
    if valid_count == 0:
         err_msg = f"Level {level}: No valid reduction data found" + (f" for '{plot_method.upper()}' in {reduction_space} space." if reduction_type and plot_method else f" in {reduction_space} space." if not plot_method else "")
         return go.Figure(layout=go.Layout(title=err_msg, xaxis={'visible': False}, yaxis={'visible': False}, height=100, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'))
    fig = go.Figure(data=[go.Scatter(
        x=plot_data['x'],
        y=plot_data['y'],
        mode='markers' if level == 0 else 'markers+text',
        marker=dict(
            size=plot_data['size'],
            color=plot_data['colors'],
            opacity=plot_data['opacities'],
            sizemode='diameter',
            line=dict(width=1, color='rgba(0,0,0,0.6)')
        ),
        text=on_marker_texts,
        textposition='middle center',
        textfont=dict(size=9, color='white'),
        hoverinfo='text',
        hovertext=plot_data['hover_texts'],
        customdata=plot_data['ids']
    )])
    chart_title = f"Level {level}" + (f" (Items - {valid_count} shown)" if level == 0 else f" (Clusters - {valid_count})") + (f" | {reduction_space} ({plot_method.upper()})" if plot_method else "") + (f" ({missing_count} missing data)" if missing_count > 0 else "")
    fig.update_layout(title=dict(text=chart_title, font=dict(size=14)), xaxis_title=None, yaxis_title=None, showlegend=False, clickmode='event+select', dragmode='pan', margin=dict(l=20, r=20, t=40, b=20), height=450, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False), hoverlabel=dict(bgcolor="rgba(255, 255, 255, 0.9)", font=dict(size=12,color="black"), bordercolor="rgba(50, 50, 50, 0.8)"))
    return fig

# =============================================================================
# Layout Definition Functions
# =============================================================================
def create_upload_layout():
    text_embedders = [{'label': name, 'value': name} for name in hf.AVAILABLE_TEXT_EMBEDDERS.keys()]
    llms = [{'label': name, 'value': name} for name in hf.AVAILABLE_LLMS.keys()]
    img_embedders = [{'label': name, 'value': name} for name in hf.AVAILABLE_IMAGE_EMBEDDERS.keys()]
    img_captioners = [{'label': name, 'value': name} for name in hf.AVAILABLE_IMAGE_CAPTIONERS.keys()]
    def_txt_emb = next(iter(text_embedders), {'value': None})['value']
    def_llm = next(iter(llms), {'value': None})['value']
    def_img_emb = next(iter(img_embedders), {'value': None})['value']
    def_img_cap = next(iter(img_captioners), {'value': None})['value']
    return dbc.Container([
        dbc.Row(dbc.Col(html.H1("Hercules Clustering Explorer", className="display-5"), width=12), className="mb-2 mt-3"),
        dbc.Row(dbc.Col(html.P("Upload data, configure parameters, and run Hercules to explore hierarchical clustering.", className="lead"), width=12), className="mb-4"),
        dbc.Row([
            dbc.Col(md=6, children=[dbc.Card(className="mb-4 shadow-sm", children=[
                    dbc.CardHeader(html.H5("1. Upload Data", className="mb-0")),
                    dbc.CardBody([
                        dbc.Label("Select tabular file, multiple text/images, or a ZIP archive.", className="mb-2"),
                        dcc.Upload(id='upload-data', children=html.Div([html.I(className="bi bi-cloud-upload-fill fs-1 text-secondary"), html.P(["Drag & Drop or ", html.A('Select Files/ZIP', className="text-primary fw-bold")])], className="py-3"), style={'width': '100%', 'borderWidth': '2px', 'borderStyle': 'dashed', 'borderRadius': '5px', 'textAlign': 'center', 'borderColor': '#ced4da'}, multiple=True, className="mb-3"),
                        html.Div(id='upload-status', className="mb-4", style={'minHeight': '60px'}), html.Hr(),
                        html.H6("Upload Ground Truth (Optional)", className="mb-2"), dbc.Label("CSV/TSV ('id', 'label') or JSON ({id: label}) for evaluation.", html_for='upload-ground-truth'), dbc.FormText("IDs should match data index/filenames.", className="mb-2"),
                        dcc.Upload(id='upload-ground-truth', children=html.Div(['Drag & Drop or ', html.A('Select GT File')]), style={'width': '100%', 'height': '50px', 'lineHeight': '50px', 'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px 0', 'backgroundColor': 'var(--bs-light)'}, multiple=False, className="mb-1"),
                        html.Div(id='ground-truth-upload-status', className="small text-muted", style={'minHeight': '40px'}),
                    ])])]),
            dbc.Col(md=6, children=[dbc.Card(className="mb-4 shadow-sm", children=[
                    dbc.CardHeader(html.H5("2. Configure Hercules", className="mb-0")),
                    dbc.CardBody([
                         dbc.Row([
                             dbc.Col(md=6, children=[
                                 dbc.Label("Representation Mode", html_for='representation-mode-input', className="fw-bold"),
                                 html.I(className="bi bi-info-circle ms-1", id="tooltip-rep-mode", style={'cursor': 'pointer'}),
                                 dbc.Tooltip(
                                     "Direct: Uses original item embeddings/numerics for L0, and centroids for higher levels. "
                                     "Description: Uses LLM-generated descriptions (and their embeddings) for all levels.",
                                     target="tooltip-rep-mode", placement="top"
                                 ),
                                 dcc.Dropdown(id='representation-mode-input', options=[{'label': 'Direct (Embed/Numeric)', 'value': 'direct'}, {'label': 'Description (LLM)', 'value': 'description'}], value='direct')
                             ]),
                             dbc.Col(md=6, children=[
                                 dbc.Label("Cluster Counts/Level", html_for='cluster-counts-input', className="fw-bold"),
                                 html.I(className="bi bi-info-circle ms-1", id="tooltip-counts", style={'cursor': 'pointer'}),
                                 dbc.Tooltip(
                                     "Comma-separated list of desired clusters per level (e.g., 10,5,2). "
                                     "Leave empty for automatic K determination (experimental).",
                                     target="tooltip-counts", placement="top"
                                 ),
                                 dbc.Input(id='cluster-counts-input', placeholder='e.g., 5, 3, 2 or blank for auto-k', value='5, 2'),
                                 dbc.FormText("Comma-separated.", className="small")
                             ]),
                         ], className="mb-3"),
                         dbc.Row([
                             dbc.Col([
                                 dbc.Label("Topic Seed (Optional)", html_for='topic-seed-input', className="fw-bold"),
                                 html.I(className="bi bi-info-circle ms-1", id="tooltip-seed", style={'cursor': 'pointer'}),
                                 dbc.Tooltip(
                                     "A phrase to guide LLM topic focus (e.g., 'financial performance', 'customer sentiment'). "
                                     "More effective with 'description' mode.",
                                     target="tooltip-seed", placement="top"
                                 ),
                                 dbc.Input(id='topic-seed-input', placeholder='Guides LLM topic focus')
                             ])
                         ], className="mb-3"),
                         html.Hr(),
                         dbc.Row([dbc.Col(md=6, children=[dbc.Label("Text Embedding", className="fw-bold"), dcc.Dropdown(id='text-embedder-select', options=text_embedders, value=def_txt_emb)]), dbc.Col(md=6, children=[dbc.Label("LLM", className="fw-bold"), dcc.Dropdown(id='llm-select', options=llms, value=def_llm)])], className="mb-3"),
                         dbc.Row([dbc.Col(md=6, children=[dbc.Label("Image Embedding", className="fw-bold"), dcc.Dropdown(id='image-embedder-select', options=img_embedders, value=def_img_emb), dbc.FormText("For image data.", className="small")]), dbc.Col(md=6, children=[dbc.Label("Image Captioning", className="fw-bold"), dcc.Dropdown(id='image-captioner-select', options=img_captioners, value=def_img_cap), dbc.FormText("For image data.", className="small")])])])])])]),
        dbc.Card(id='tabular-options-card', children=[
            dbc.CardHeader(html.H5("Tabular Data Options & Preview", className="mb-0")),
            dbc.CardBody([
                dbc.Alert(id='tabular-load-status', color="info", is_open=False, duration=6000, dismissable=True, className="d-flex align-items-center"),
                dbc.Row([dbc.Col(dbc.Checklist(options=[{"label": "Data has header row", "value": "header"}], value=["header"], id="tabular-header-checkbox", switch=True), md=4, className="mb-2 d-flex align-items-center"), dbc.Col(dbc.InputGroup([dbc.InputGroupText("Index Column", className="small"), dbc.Input(id="tabular-index-input", placeholder="None, 0, or name", type="text", value="", debounce=True, size="sm")]), md=8, className="mb-2")], className="mb-3 align-items-center"),
                html.Div(id='numeric-column-checklist-wrapper', children=[
                    html.H6("Select Numeric Columns for Clustering:", className="mt-3"),
                    dbc.Checklist(
                        id='numeric-column-selector',
                        options=[], value=[], # Initially empty
                        inline=True, switch=True, className="mb-2"
                    ),
                    dbc.FormText("Only selected numeric columns will be used in the analysis.")
                ], className="mt-3 mb-3 border-top pt-3", style={'display': 'none'}), # Initially hidden
                html.H6("Data Preview (first 50 rows):"),
                dash_table.DataTable(id='tabular-preview-table', page_size=PREVIEW_TABLE_PAGE_SIZE, style_table={'overflowX': 'auto', 'minWidth': '100%'}, style_cell={'textAlign': 'left', 'padding': '8px', 'maxWidth': 180, 'overflow': 'hidden', 'textOverflow': 'ellipsis', 'whiteSpace': 'nowrap', 'fontSize': '0.9em', 'border': '1px solid #eee'}, style_header={'backgroundColor': 'var(--bs-light)', 'fontWeight': 'bold', 'borderBottom': '2px solid #dee2e6'}, style_data={'border': '1px solid #eee'}, style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': 'rgba(0,0,0,.02)'}], tooltip_delay=0, tooltip_duration=None)])], className="mt-4 mb-4 shadow-sm", style={'display': 'none'}),
        html.H4("3. Run & Monitor", className="mb-3"),
        dbc.Row(dbc.Col(dbc.Button([html.I(className="bi bi-play-circle-fill me-2"), "Run Hercules Clustering"], id='run-button', n_clicks=0, color="primary", size="lg", className="fw-bold"), width={"size": 6, "offset": 3}, className="text-center mb-4")),
        dcc.Interval(id='log-interval', interval=1000, n_intervals=0, disabled=True),
        dbc.Row(dbc.Col(dcc.Loading(id="loading-output", type="default", children=[
             html.Div(id='status-updates', className="mb-3", style={'minHeight': '50px'}),
        ]), width=12)),
        dbc.Row(dbc.Col(
            html.Div(id='live-log-container', children=[
                html.H5("Run Log", className="mt-2"),
                html.Pre(id='run-log-live', style={
                    'whiteSpace': 'pre-wrap',
                    'wordBreak': 'break-all',
                    'maxHeight': '400px',
                    'overflowY': 'scroll',
                    'border': '1px solid #ddd',
                    'padding': '10px',
                    'backgroundColor': '#f8f9fa',
                    'fontSize': '0.8em',
                })
            ], style={'display': 'none'}) # Initially hidden
        ), className="mb-5"),
    ], fluid=True, id="upload-view-container")

def create_results_layout(clusters_data: List[Dict], config_data: Dict, state_data: Dict, eval_results: Dict, run_log: str):
    if not clusters_data:
        return dbc.Container([dbc.Alert([html.I(className="bi bi-exclamation-triangle-fill me-2"), "Error: No cluster data received."], color="danger")])
    all_clusters_map = reconstruct_cluster_map_from_data(clusters_data)
    if not all_clusters_map:
        return dbc.Container([dbc.Alert([html.I(className="bi bi-exclamation-triangle-fill me-2"), "Error: Failed to reconstruct cluster map."], color="danger")])
    max_level = state_data.get("max_level_achieved", 0)
    num_l0_items = state_data.get("num_l0_items", 0)
    show_l0_default = num_l0_items < L0_DISPLAY_THRESHOLD
    default_l0_toggle_value = ["show_l0"] if show_l0_default else []
    print(f"Initial L0 Visibility: {'Shown' if show_l0_default else 'Hidden'} ({num_l0_items} items vs threshold {L0_DISPLAY_THRESHOLD})")
    l0_clusters_ordered = []
    l0_orig_ids = state_data.get("_l0_original_ids_ordered", [])
    if l0_orig_ids and all_clusters_map:
        l0_map_temp = {str(c.original_id): c for c in all_clusters_map.values() if c.level == 0 and c.original_id is not None}
        l0_clusters_ordered = [l0_map_temp.get(str(orig_id)) for orig_id in l0_orig_ids if str(orig_id) in l0_map_temp]
        l0_clusters_ordered = [c for c in l0_clusters_ordered if c is not None] # Filter out None
        if len(l0_clusters_ordered) != len(l0_orig_ids):
            print(f"Warning: L0 order reconstruction mismatch. Found {len(l0_clusters_ordered)} / {len(l0_orig_ids)} IDs.")
        if not l0_clusters_ordered or abs(len(l0_clusters_ordered) - len(l0_orig_ids)) > 10: # Heuristic for major mismatch
             print("Reverting to sorted L0 clusters due to significant mismatch.")
             l0_clusters_ordered = sorted([c for c in all_clusters_map.values() if c.level == 0], key=lambda c: str(c.original_id) if c.original_id is not None else "")
    elif num_l0_items > 0 and all_clusters_map:
        l0_clusters_ordered = sorted([c for c in all_clusters_map.values() if c.level == 0], key=lambda c: str(c.original_id) if c.original_id is not None else "")
    available_reduction_sources = []
    if any(bool(getattr(c, 'representation_vector_reductions', {})) for c in all_clusters_map.values()):
        available_reduction_sources.append({'label': 'Representation', 'value': 'representation_vector'})
    if any(bool(getattr(c, 'description_embedding_reductions', {})) for c in all_clusters_map.values()):
        available_reduction_sources.append({'label': 'Description', 'value': 'description_embedding'})
    default_reduction_type = 'representation_vector' if any(s['value'] == 'representation_vector' for s in available_reduction_sources) else (available_reduction_sources[0]['value'] if available_reduction_sources else None)
    top_clusters = sorted([c for c in all_clusters_map.values() if c.parent is None or c.parent.id not in all_clusters_map], key=lambda c: c.id)
    color_map = {c.id: DEFAULT_COLOR_SEQUENCE[i % len(DEFAULT_COLOR_SEQUENCE)] for i, c in enumerate(top_clusters)}
    top_ancestor_map = {}
    queue = deque([c for c in top_clusters])
    visited_ancestry = set(c.id for c in top_clusters)
    for c_node in top_clusters:
        top_ancestor_map[c_node.id] = c_node.id
    while queue:
        parent = queue.popleft()
        parent_top_ancestor_id = top_ancestor_map.get(parent.id)
        if parent.children:
            for child in parent.children:
                 if child and child.id in all_clusters_map and child.id not in visited_ancestry:
                     top_ancestor_map[child.id] = parent_top_ancestor_id
                     visited_ancestry.add(child.id)
                     queue.append(child)
    initial_figures = {}
    if default_reduction_type:
        for lvl in range(max_level, -1, -1):
            try:
                initial_figures[lvl] = create_bubble_chart_figure(lvl, None, default_reduction_type, all_clusters_map, top_ancestor_map, color_map, l0_clusters_ordered)
            except Exception as fig_err:
                print(f"Error generating initial figure L{lvl}: {fig_err}", file=sys.stderr)
                initial_figures[lvl] = go.Figure(layout=go.Layout(title=f"Level {lvl}: Error creating plot", height=100, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'))
    else:
        no_data_layout = go.Layout(title="No reduction data available", height=100, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        for lvl in range(max_level, -1, -1):
            initial_figures[lvl] = go.Figure(layout=no_data_layout)

    return dbc.Container(fluid=True, children=[
        dcc.Store(id='selected-cluster-store', data=None), dcc.Store(id='reduction-type-store', data=default_reduction_type), dcc.Store(id='scroll-trigger-store', data=None),
        dbc.Row([
            dbc.Col(html.H1("Hercules Results", className="display-5"), width="auto"),
            dbc.Col([
                dbc.Button([html.I(className="bi bi-cloud-download me-1"), "Download Full Results (JSON)"], id='btn-download-full-results', color="info", outline=True, className="me-2"),
                dbc.Button([html.I(className="bi bi-arrow-left-circle me-1"), "Upload New Data"], id='back-to-upload-button', color="secondary", outline=True)
            ], width="auto", className="d-flex align-items-center ms-auto")
        ], justify="between", align="center", className="my-3"),
        dbc.Row([
            dbc.Col(md=4, children=[
                html.Div(id='cluster-summary-div', children=[dbc.Card([dbc.CardHeader(html.H5("Selection Summary", className="mb-0")), html.Div(id='cluster-summary-content', children=create_summary_content(None, all_clusters_map, config_data, state_data, eval_results, default_show_l0=show_l0_default, run_log=run_log, l0_clusters_ordered=l0_clusters_ordered))], className="mb-4 shadow-sm")]),
                dbc.Card(className="shadow-sm", children=[dbc.CardHeader(html.H5("Search Clusters", className="mb-0")), dbc.CardBody([dbc.InputGroup([dbc.InputGroupText(html.I(className="bi bi-search")), dbc.Input(id='search-input', placeholder="Title/Desc (min 2 chars)...", type='text', debounce=True)], className="mb-2"), dcc.Loading(type="circle", children=[html.Div(id='search-results-div', style={'maxHeight': '300px', 'overflowY': 'auto'})])])])], className="mb-4 mb-md-0"),
            dbc.Col(md=8, children=[
                dbc.Card(className="mb-4 shadow-sm", children=[dbc.CardHeader(html.H5("Visualization Controls", className="mb-0")), dbc.CardBody([dbc.Row([dbc.Col([dbc.Label("Plot Coordinates:", html_for='reduction-radio', className="fw-bold me-2"), dcc.RadioItems(id='reduction-type-radio', options=available_reduction_sources, value=default_reduction_type, inline=True, inputClassName="me-1", labelClassName="me-3 small") if available_reduction_sources else html.P("No reduction data available.")], width=12, lg=8, className="mb-2 mb-lg-0"), dbc.Col([dbc.Checklist(options=[{"label": "Show Level 0 Details", "value": "show_l0"}], value=default_l0_toggle_value, id="global-l0-visibility-toggle", inline=True, switch=True, labelClassName="small") if num_l0_items > 0 else html.Div()], width=12, lg=4, className="d-flex align-items-center justify-content-lg-end")])])]),
                html.Div([dbc.Card(className="mb-3 shadow-sm", children=[dbc.CardBody(html.Div(id={'type': 'plot-container', 'level': level}, children=[dcc.Graph(id={'type': 'bubble-chart', 'level': level}, figure=initial_figures.get(level, go.Figure()), config={'displayModeBar': False})]))]) for level in range(max_level, 0, -1)]) if max_level > 0 else dbc.Alert([html.I(className="bi bi-info-circle-fill me-2"), "No cluster hierarchy (L1+) to visualize."], color="info"),
                html.Div(id='l0-plot-container', children=[dbc.Card(className="mb-3 shadow-sm", children=[dbc.CardBody(dcc.Graph(id={'type': 'bubble-chart', 'level': 0}, figure=initial_figures.get(0, go.Figure()), config={'displayModeBar': False}))])], hidden=(not show_l0_default)) if num_l0_items > 0 else html.Div()])])], id="results-view-container", className="mb-5")

# =============================================================================
# App Layout (Main Controller)
# =============================================================================
app.layout = html.Div([
    dcc.Store(id='session-id-store'), dcc.Store(id='temp-data-dir-store'), dcc.Store(id='uploaded-file-info-store'),
    dcc.Store(id='validated-load-params-store'), dcc.Store(id='ground-truth-data-store'),
    dcc.Store(id='hercules-results-store', storage_type='memory'),
    dcc.Store(id='run-params-store'), dcc.Store(id='run-trigger-store'),
    dcc.Download(id='download-full-results-json'), dcc.Download(id='download-membership-csv'),
    dcc.Download(id='download-evaluation-json'), dcc.Download(id='download-hierarchy-txt'),
    dcc.Download(id='download-run-log-txt'),
    html.Div(id='page-content', children=create_upload_layout())
])

# =============================================================================
# Callbacks
# =============================================================================
@callback(Output('page-content', 'children', allow_duplicate=True), Input('hercules-results-store', 'data'), prevent_initial_call=True)
def display_results_page(results_data):
    if results_data and isinstance(results_data, dict):
        print("Rendering results view.")
        try:
            clusters = results_data.get('clusters')
            config = results_data.get('config')
            state = results_data.get('state')
            eval_res = results_data.get('eval_results', {})
            run_log = results_data.get('run_log', 'Run log was not found in the results.')

            if clusters is None or config is None or state is None:
                 err_msg = f"Error displaying results: Missing data for {[k for k,v in {'clusters':clusters,'config':config,'state':state}.items() if v is None]}."
                 print(err_msg, file=sys.stderr)
                 return dbc.Container([dbc.Alert([html.I(className="bi bi-exclamation-triangle-fill me-2"), err_msg], color="danger"), dbc.Button("Go Back", id={'type': 'error-back-to-upload-button', 'index':'display_error'}, color="warning")])
            
            return create_results_layout(clusters, config, state, eval_res, run_log)
        except Exception as e:
            tb_str = traceback.format_exc()
            print(f"Error creating results layout: {e}\n{tb_str}", file=sys.stderr)
            return dbc.Container([dbc.Alert([html.I(className="bi bi-exclamation-triangle-fill me-2"), f"Layout Error: {e}"], color="danger"), html.Details([html.Summary("Details"), html.Pre(tb_str)]), dbc.Button("Go Back", id={'type': 'error-back-to-upload-button', 'index':'layout_error'}, color="warning")], className="mt-5")
    return no_update

@callback(
    Output('page-content', 'children', allow_duplicate=True), Output('session-id-store', 'data', allow_duplicate=True),
    Output('temp-data-dir-store', 'data', allow_duplicate=True), Output('uploaded-file-info-store', 'data', allow_duplicate=True),
    Output('validated-load-params-store', 'data', allow_duplicate=True), Output('ground-truth-data-store', 'data', allow_duplicate=True),
    Output('hercules-results-store', 'data', allow_duplicate=True),
    Input('back-to-upload-button', 'n_clicks'), Input({'type': 'error-back-to-upload-button', 'index': ALL}, 'n_clicks'),
    State('session-id-store', 'data'), prevent_initial_call=True)
def go_back_to_upload(back_clicks, error_clicks, session_id_to_clean):
    ctx = callback_context
    if not ctx.triggered:
        return (no_update,) * 7
    button_clicked = (ctx.triggered_id == 'back-to-upload-button' and back_clicks and back_clicks > 0) or \
                     (isinstance(ctx.triggered_id, dict) and ctx.triggered_id.get('type') == 'error-back-to-upload-button' and any(n and n > 0 for n in error_clicks))
    if button_clicked:
        print("Returning to upload view. Clearing stores and session data.")
        if session_id_to_clean:
            session_dir = os.path.join(UPLOAD_DIRECTORY, session_id_to_clean)
            if os.path.exists(session_dir):
                try:
                    shutil.rmtree(session_dir)
                    print(f"Cleaned up: {session_dir}")
                except Exception as e:
                    print(f"Warning: Could not clean {session_dir}: {e}", file=sys.stderr)
        return create_upload_layout(), None, None, None, None, None, None
    return (no_update,) * 7

@callback(
    Output('upload-status', 'children'),
    Output('temp-data-dir-store', 'data', allow_duplicate=True),
    Output('session-id-store', 'data', allow_duplicate=True),
    Output('uploaded-file-info-store', 'data', allow_duplicate=True),
    Output('tabular-options-card', 'style'),
    Output('tabular-preview-table', 'data'),
    Output('tabular-preview-table', 'columns'),
    Output('validated-load-params-store', 'data', allow_duplicate=True),
    Output('ground-truth-data-store', 'data', allow_duplicate=True),
    Output('ground-truth-upload-status', 'children', allow_duplicate=True),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    State('upload-data', 'last_modified'),
    State('session-id-store', 'data'),
    prevent_initial_call=True
)
def handle_upload(list_of_contents, list_of_names, list_of_dates, existing_session_id):
    if list_of_contents is None:
        return (no_update,) * 10

    print("Main data upload triggered. Clearing existing ground truth data/status if any.")
    clear_gt_data = None
    clear_gt_status = ""
    session_id = existing_session_id or str(uuid.uuid4())
    session_dir = os.path.join(UPLOAD_DIRECTORY, session_id)
    
    # FIX: Aggressively clean the session directory on every new upload.
    if os.path.exists(session_dir):
        try:
            shutil.rmtree(session_dir)
            print(f"Cleaned previous session directory: {session_dir}")
        except Exception as e:
            print(f"Warning: Could not clean previous session directory {session_dir}: {e}", file=sys.stderr)
            
    try:
        os.makedirs(session_dir, exist_ok=True)
    except OSError as e:
        return (dbc.Alert([html.I(className="bi bi-x-octagon-fill me-2"), f"Error creating session directory: {e}"], color="danger"), None, session_id, None, {'display': 'none'}, None, None, None, clear_gt_data, clear_gt_status)
    upload_msgs = []
    saved_files = []
    try:
        for i, c in enumerate(list_of_contents):
            content_type, content_string = c.split(',')
            decoded = base64.b64decode(content_string)
            filename = list_of_names[i] if list_of_names and i < len(list_of_names) else f"file_{i}"
            safe_filename = filename.replace('/', '_').replace('\\', '_').replace(':', '_')
            filepath = os.path.join(session_dir, safe_filename)
            ext = os.path.splitext(safe_filename)[1].lower()
            if ext == '.zip':
                upload_msgs.append(html.P(f"Processing ZIP: '{safe_filename}'...", className="small"))
                try:
                    with zipfile.ZipFile(io.BytesIO(decoded), 'r') as zip_ref:
                        session_dir_abs = os.path.abspath(session_dir)
                        extracted_count = 0
                        for member in zip_ref.infolist():
                            if member.is_dir() or member.filename.startswith('__MACOSX') or os.path.basename(member.filename).startswith('.'):
                                continue
                            target_path = os.path.abspath(os.path.join(session_dir_abs, member.filename)) # Resolve to absolute
                            if not target_path.startswith(session_dir_abs): # Security: Ensure path is within session_dir
                                upload_msgs.append(html.P(f" - Skipped potentially unsafe path in ZIP: {member.filename}", className="small text-danger"))
                                continue
                            if os.path.splitext(member.filename)[1].lower() in GROUND_TRUTH_EXTENSIONS:
                                upload_msgs.append(html.P(f" - Skipped '{member.filename}' (potential GT file) inside data ZIP. Upload GT separately.", className="small text-warning"))
                                continue
                            os.makedirs(os.path.dirname(target_path), exist_ok=True)
                            with open(target_path, 'wb') as outfile:
                                outfile.write(zip_ref.read(member.filename))
                            saved_files.append({'path': target_path, 'name': member.filename})
                            extracted_count += 1
                        upload_msgs.append(html.P(f" - Extracted {extracted_count} file(s) from ZIP.", className="small"))
                except zipfile.BadZipFile:
                    upload_msgs.append(dbc.Alert([html.I(className="bi bi-x-octagon-fill me-2"), f"Invalid ZIP file: '{safe_filename}'"], color="danger"))
                except Exception as ze:
                    upload_msgs.append(dbc.Alert([html.I(className="bi bi-x-octagon-fill me-2"), f"Error processing ZIP '{safe_filename}': {ze}"], color="danger"))
            else:
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                open(filepath, 'wb').write(decoded)
                saved_files.append({'path': filepath, 'name': safe_filename})
    except Exception as e:
        tb = traceback.format_exc()
        print(f"Upload Error: {e}\n{tb}", file=sys.stderr)
        return (dbc.Alert([html.I(className="bi bi-x-octagon-fill me-2"), f"Error processing upload: {e}"], color="danger"), None, session_id, None, {'display': 'none'}, None, None, None, clear_gt_data, clear_gt_status)
    if not saved_files:
        return (upload_msgs + [dbc.Alert([html.I(className="bi bi-exclamation-triangle-fill me-2"), "No valid main data files found/extracted from upload."], color="warning")], None, session_id, None, {'display': 'none'}, None, None, None, clear_gt_data, clear_gt_status)

    is_single_tabular = False
    file_info_for_store = None
    if len(saved_files) == 1:
        f_info = saved_files[0]
        f_ext = os.path.splitext(f_info['path'])[1].lower()
        if f_ext in TABULAR_EXTENSIONS:
            is_single_tabular = True
            file_info_for_store = {'path': f_info['path'], 'type': 'tabular', 'ext': f_ext, 'name': f_info['name']}
    
    if is_single_tabular:
        print(f"Single tabular file detected: {file_info_for_store['path']}. Triggering preview update.")
        status = dbc.Alert([html.I(className="bi bi-check-circle-fill me-2"), f"Detected tabular file: '{file_info_for_store['name']}'. Configure options below."], color="success", duration=8000, dismissable=True)
        # Set stores to trigger the update_tabular_preview callback
        return (upload_msgs + [status], None, session_id, file_info_for_store, {'display': 'block'}, None, None, {'header': 0, 'index_col': None}, clear_gt_data, clear_gt_status)
    else: # Multiple files or non-tabular single file
        print(f"{len(saved_files)} files uploaded to session directory: {session_dir}")
        num_files = len(saved_files)
        f_types = Counter(os.path.splitext(f['path'])[1].lower() for f in saved_files)
        img_count = sum(c for ext, c in f_types.items() if ext in IMAGE_EXTENSIONS)
        txt_count = sum(c for ext, c in f_types.items() if ext in TEXT_EXTENSIONS)
        tab_count = sum(c for ext, c in f_types.items() if ext in TABULAR_EXTENSIONS)
        summary_list = [html.Strong(f"Processed {num_files} file(s):")]
        if img_count: summary_list.append(html.Li(f"{img_count} image file(s)"))
        if txt_count: summary_list.append(html.Li(f"{txt_count} text file(s)"))
        if tab_count: summary_list.append(html.Li(f"{tab_count} tabular file(s) (will use directory inference if run)"))
        if num_files - img_count - txt_count - tab_count: summary_list.append(html.Li(f"{num_files - img_count - txt_count - tab_count} other file(s)"))
        summary_list.extend([html.Hr(style={'margin': '8px 0'}), html.Strong(f"First {min(num_files, PREVIEW_MAX_FILES_LIST)} file(s):"), html.Ul([html.Li(f['name'], className="small text-muted text-truncate") for f in saved_files[:PREVIEW_MAX_FILES_LIST]], style={'paddingLeft': '20px', 'marginTop': '5px', 'maxHeight': '100px', 'overflowY': 'auto'})])
        if num_files > PREVIEW_MAX_FILES_LIST: summary_list.append(html.Em(f"...and {num_files - PREVIEW_MAX_FILES_LIST} more.", className="small"))
        summary_list.append(html.P("Configure Hercules below and click Run.", className="mt-2 mb-0"))
        summary = html.Div(summary_list)
        return (upload_msgs + [dbc.Alert([html.I(className="bi bi-info-circle-fill me-2"), summary], color="info", className="d-flex align-items-start")], session_dir, session_id, None, {'display': 'none'}, None, None, None, clear_gt_data, clear_gt_status)

@callback(
    Output('ground-truth-upload-status', 'children'), Output('ground-truth-data-store', 'data', allow_duplicate=True),
    Input('upload-ground-truth', 'contents'), State('upload-ground-truth', 'filename'), prevent_initial_call=True)
def handle_ground_truth_upload(contents, filename):
    if contents is None:
        return "Ground truth removed.", None
    if filename is None:
        return dbc.Alert("Error: No filename for GT.", color="warning"), None
    if os.path.splitext(filename)[1].lower() not in GROUND_TRUTH_EXTENSIONS:
        return dbc.Alert([html.I(className="bi bi-exclamation-triangle-fill me-2"), f"Unsupported GT type: '{filename}'. Use CSV, TSV, JSON."], color="warning"), None
    print(f"Parsing ground truth file: {filename}")
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    parsed_gt, error = parse_ground_truth(decoded, filename)
    if error:
        print(f"GT parsing error: {error}")
        return dbc.Alert([html.I(className="bi bi-x-octagon-fill me-2"), f"Error parsing '{filename}': {error}"], color="danger"), None
    elif parsed_gt is None or not isinstance(parsed_gt, dict):
        print(f"GT parsing failed for '{filename}', result not dict/None.")
        return dbc.Alert([html.I(className="bi bi-exclamation-triangle-fill me-2"), f"Could not parse GT from '{filename}'. Check format."], color="warning"), None
    elif not parsed_gt:
        warn_msg = f"Parsed GT file '{filename}' is empty."
        print(warn_msg, file=sys.stderr)
        return dbc.Alert([html.I(className="bi bi-exclamation-triangle-fill me-2"), warn_msg], color="warning"), None
    else:
        num_entries = len(parsed_gt)
        status_msg = f"Loaded '{filename}' ({num_entries} entr{'y' if num_entries == 1 else 'ies'})."
        print(status_msg)
        return dbc.Alert([html.I(className="bi bi-check-circle-fill me-2"), status_msg], color="success", duration=6000, dismissable=True), parsed_gt

@callback(
    Output('tabular-preview-table', 'data', allow_duplicate=True),
    Output('tabular-preview-table', 'columns', allow_duplicate=True),
    Output('tabular-preview-table', 'tooltip_data', allow_duplicate=True),
    Output('tabular-load-status', 'children', allow_duplicate=True),
    Output('tabular-load-status', 'color', allow_duplicate=True),
    Output('tabular-load-status', 'is_open', allow_duplicate=True),
    Output('validated-load-params-store', 'data', allow_duplicate=True),
    Output('numeric-column-selector', 'options'),
    Output('numeric-column-selector', 'value'),
    Output('numeric-column-checklist-wrapper', 'style'),
    Input('tabular-header-checkbox', 'value'),
    Input('tabular-index-input', 'value'),
    Input('uploaded-file-info-store', 'data'),
    prevent_initial_call=True)
def update_tabular_preview(header_check, index_input, uploaded_file_info):
    if not uploaded_file_info or 'path' not in uploaded_file_info:
        return (no_update,) * 10

    filepath = uploaded_file_info['path']
    header_row = 0 if "header" in (header_check or []) else None
    index_col = str(index_input).strip() if index_input is not None and str(index_input).strip() else None

    print(f"Updating preview for '{os.path.basename(filepath)}': header={header_row}, index='{index_col}'")
    df_preview, err = load_tabular_data(filepath, header=header_row, index_col=index_col, nrows=PREVIEW_NROWS)

    valid_params, preview_data, preview_cols, preview_tips = None, None, [], []
    selector_options, selector_value, selector_style = [], [], {'display': 'none'}
    icon_map = {"danger": "bi-x-octagon-fill", "warning": "bi-exclamation-triangle-fill", "success": "bi-check-circle-fill"}
    status_msg, status_color, status_open = "", "info", False

    if err:
        status_msg = f"Error updating preview: {err}"
        status_color, status_open = "danger", True
    elif df_preview is None:
        status_msg, status_color, status_open = "Error: Load returned None.", "danger", True
    else:
        valid_params = {'header': header_row, 'index_col': index_col}
        if df_preview.empty:
            status_msg = "Preview updated. Note: File loaded empty (or only header/index)."
            status_color, status_open = "warning", True
        else:
            status_msg, status_color, status_open = "Preview updated successfully.", "success", True
            preview_data = df_preview.to_dict('records')
            try:
                preview_cols = [{"name": list(map(str, i)), "id": ".".join(map(str, i))} for i in df_preview.columns] if isinstance(df_preview.columns, pd.MultiIndex) else [{"name": str(i), "id": str(i)} for i in df_preview.columns]
                if preview_data:
                    preview_tips = [{col['id']: {'value': str(row.get(col['id'], '')), 'type': 'markdown'} for col in preview_cols if col['id'] in row} for row in preview_data[:PREVIEW_TABLE_PAGE_SIZE]]
            except Exception as e:
                status_msg += f" (Warning creating preview: {e})"
                status_color = "warning"

        numeric_cols = df_preview.select_dtypes(include=np.number).columns.tolist()
        if numeric_cols:
            selector_options = [{'label': str(col), 'value': str(col)} for col in numeric_cols]
            selector_value = [str(col) for col in numeric_cols] # All selected by default
            selector_style = {'display': 'block'}
        elif df_preview is not None:
             # Keep it hidden but ensure the component ID is in the layout
             selector_style = {'display': 'none'}

    status_div = [html.I(className=f"bi {icon_map.get(status_color, 'bi-info-circle-fill')} me-2"), status_msg]
    return preview_data, preview_cols, preview_tips, status_div, status_color, status_open, valid_params, selector_options, selector_value, selector_style


@callback(
    Output('run-button', 'disabled'),
    Output('run-params-store', 'data'),
    Output('run-trigger-store', 'data'),
    Output('live-log-container', 'style'),
    Output('log-interval', 'disabled'),
    Output('status-updates', 'children', allow_duplicate=True),
    Input('run-button', 'n_clicks'),
    State('representation-mode-input', 'value'), State('cluster-counts-input', 'value'), State('topic-seed-input', 'value'),
    State('text-embedder-select', 'value'), State('llm-select', 'value'), State('image-embedder-select', 'value'),
    State('image-captioner-select', 'value'), State('numeric-column-selector', 'value'),
    prevent_initial_call=True
)
def initiate_run(n_clicks, rep_mode, counts_str, topic_seed, txt_emb, llm, img_emb, img_cap, selected_cols):
    if not n_clicks:
        return no_update, no_update, no_update, no_update, no_update, no_update

    params = {
        'rep_mode': rep_mode, 'counts_str': counts_str, 'topic_seed': topic_seed,
        'txt_emb_name': txt_emb, 'llm_name': llm, 'img_emb_name': img_emb,
        'img_cap_name': img_cap, 'selected_columns': selected_cols
    }
    initial_status = dbc.Alert([html.I(className="bi bi-hourglass-split me-2"), "Run initiated... Hercules is starting."], color="info")
    
    # Disable button, store params, trigger run, show log viewer, enable interval, set initial status
    return True, params, str(uuid.uuid4()), {'display': 'block'}, False, initial_status

@callback(
    Output('run-log-live', 'children'),
    Input('log-interval', 'n_intervals'),
    State('session-id-store', 'data'),
    prevent_initial_call=True
)
def update_live_log(n, session_id):
    if not session_id:
        return no_update
    
    log_filepath = os.path.join(UPLOAD_DIRECTORY, session_id, RUN_LOG_FILENAME)
    try:
        with open(log_filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return "Waiting for log file to be created..."
    except Exception as e:
        return f"Error reading log file: {e}"

@callback(
    Output('hercules-results-store', 'data', allow_duplicate=True),
    Output('status-updates', 'children', allow_duplicate=True),
    Output('log-interval', 'disabled', allow_duplicate=True),
    Output('run-button', 'disabled', allow_duplicate=True),
    Input('run-trigger-store', 'data'),
    State('run-params-store', 'data'),
    State('temp-data-dir-store', 'data'), State('uploaded-file-info-store', 'data'),
    State('validated-load-params-store', 'data'), State('ground-truth-data-store', 'data'), 
    State('session-id-store', 'data'),
    prevent_initial_call=True)
def run_hercules_clustering_logic(trigger, run_params, temp_data_dir, uploaded_file_info, valid_load_params, ground_truth_data, session_id):
    if not trigger or not run_params or not session_id:
        return no_update, dbc.Alert("Run trigger failed: Missing parameters or session.", color="danger"), True, False

    start_time = time.time()
    log_filepath = os.path.join(UPLOAD_DIRECTORY, session_id, RUN_LOG_FILENAME)
    results_pkg, final_status_alert = None, None

    # Unpack parameters
    rep_mode = run_params.get('rep_mode')
    counts_str = run_params.get('counts_str')
    topic_seed = run_params.get('topic_seed')
    txt_emb_name = run_params.get('txt_emb_name')
    llm_name = run_params.get('llm_name')
    img_emb_name = run_params.get('img_emb_name')
    img_cap_name = run_params.get('img_cap_name')
    selected_columns = run_params.get('selected_columns')

    try:
        with open(log_filepath, 'w', encoding='utf-8') as log_stream:
            print("--- run_hercules_clustering triggered ---", file=log_stream)
            input_data, data_type, load_params, data_desc, parsed_gt = None, None, None, "Unknown", None

            level_counts = parse_cluster_counts(counts_str)
            if level_counts is None and counts_str and counts_str.strip():
                raise ValueError("Invalid Cluster Counts. Use comma-separated positive integers (e.g., 10, 5).")

            seed = topic_seed.strip() if topic_seed else None
            print(f"Run params: Mode={rep_mode}, Counts={level_counts or 'Auto'}, Seed={seed}", file=log_stream)
            
            if ground_truth_data and isinstance(ground_truth_data, dict):
                parsed_gt = ground_truth_data
                print(f"Using provided ground truth ({len(parsed_gt)} entries).", file=log_stream)

            if uploaded_file_info and 'path' in uploaded_file_info and valid_load_params is not None: # Single tabular file
                filepath, filename = uploaded_file_info['path'], uploaded_file_info.get('name', '...'),
                header, index_col = valid_load_params.get('header'), valid_load_params.get('index_col')
                data_desc = f"Tabular: '{filename}' (H={header}, I='{index_col}')"
                print(f"Loading full data: {data_desc}", file=log_stream)
                df_full, err = load_tabular_data(filepath, header=header, index_col=index_col)
                if err: raise ValueError(f"Tabular Load Error: {err}")
                if df_full is None or df_full.empty: raise ValueError("Tabular file loaded empty.")
                df_numeric = df_full.select_dtypes(include=np.number)
                if df_numeric.empty: raise ValueError(f"File '{filename}' has no numeric columns.")
                
                if selected_columns is not None:
                    if not selected_columns: raise ValueError("No numeric columns were selected for clustering.")
                    final_cols = [c for c in selected_columns if c in df_numeric.columns]
                    if not final_cols: raise ValueError("None of the selected columns found in data.")
                    print(f"Using {len(final_cols)} of {len(df_numeric.columns)} numeric columns.", file=log_stream)
                    input_data = df_numeric[final_cols]
                else:
                    print("Warning: No column selection provided. Using all numeric columns.", file=log_stream)
                    input_data = df_numeric
                
                data_type = 'numeric'
                load_params = valid_load_params
            elif temp_data_dir: # Directory of files
                data_desc = f"Files in Session: {os.path.basename(temp_data_dir)}"
                print(f"Inferring data from directory: {temp_data_dir}", file=log_stream)
                input_data, data_type, err = infer_data_and_prepare(temp_data_dir)
                if err: raise ValueError(f"Data Inference Error: {err}")
            else:
                raise ValueError("No data source specified.")
            
            print(f"Interpreting data as {data_type.upper()} ({len(input_data) if isinstance(input_data, (dict, list)) else 'Shape ' + str(getattr(input_data, 'shape', 'N/A')) }).", file=log_stream)
            
            sel_txt_emb = hf.get_function_by_name(txt_emb_name, hf.AVAILABLE_TEXT_EMBEDDERS)
            sel_llm = hf.get_function_by_name(llm_name, hf.AVAILABLE_LLMS)
            sel_img_emb = hf.get_function_by_name(img_emb_name, hf.AVAILABLE_IMAGE_EMBEDDERS) if data_type == 'image' else None
            sel_img_cap = hf.get_function_by_name(img_cap_name, hf.AVAILABLE_IMAGE_CAPTIONERS) if data_type == 'image' else None

            print("Instantiating Hercules...", file=log_stream)
            hercules = Hercules(level_cluster_counts=level_counts, representation_mode=rep_mode, text_embedding_client=sel_txt_emb, llm_client=sel_llm, image_embedding_client=sel_img_emb, image_captioning_client=sel_img_cap, reduction_methods=['pca'], random_state=42, verbose=True, save_run_details=False)
            
            with contextlib.redirect_stdout(log_stream):
                print(f"\n--- Starting Hercules Clustering ({data_type.upper()}) ---")
                top_clusters = hercules.cluster(input_data, topic_seed=seed)
                print("\n--- Clustering Finished ---\n")
                if not hercules._all_clusters_map:
                    raise ValueError("Clustering finished, but no clusters were generated.")

                eval_results = {}
                if hercules.max_level > 0:
                    print("--- Running Evaluation ---")
                    for level_idx in range(1, hercules.max_level + 1):
                         with warnings.catch_warnings(record=True) as caught_warnings:
                             warnings.simplefilter("always")
                             try:
                                 res = hercules.evaluate_level(level=level_idx, ground_truth_labels=parsed_gt)
                                 eval_results[str(level_idx)] = res
                             except Exception as eval_err:
                                 print(f"  Level {level_idx} Eval FAILED: {eval_err}", file=sys.stderr)
                                 eval_results[str(level_idx)] = {"error": str(eval_err)}
                             for w in caught_warnings:
                                 print(f"    Eval Warning (L{level_idx}): {w.message}")
                    print("--- Evaluation Finished ---\n")

            clusters_serial = [c.to_light_dict() for c in hercules._all_clusters_map.values()]
            cfg = {k: getattr(hercules, k, 'N/A') for k in ['level_cluster_counts', 'representation_mode', 'reduction_methods', 'random_state']}
            cfg.update({"selected_text_embedder": txt_emb_name, "selected_llm": llm_name, "selected_image_embedder": img_emb_name, "selected_image_captioner": img_cap_name})
            state = {"variable_names": getattr(hercules, 'variable_names', None), "input_data_type": getattr(hercules, 'input_data_type', 'unknown'), "embedding_dims": getattr(hercules, 'embedding_dims_', {}), "max_level_achieved": getattr(hercules, 'max_level', 0), "num_l0_items": len(getattr(hercules, '_l0_clusters_ordered', [])), "_l0_original_ids_ordered": [str(c.original_id) for c in getattr(hercules, '_l0_clusters_ordered', [])], "tabular_load_params": load_params, "data_source_description": data_desc}
            results_pkg = {"clusters": clusters_serial, "config": cfg, "state": state, "eval_results": eval_results}
            
            duration = time.time() - start_time
            final_status_alert = dbc.Alert([html.I(className="bi bi-check-circle-fill me-2"), html.Strong(f"Finished in {duration:.2f}s. Results ready.")], color="success", duration=8000)
            print(f"--- run_hercules_clustering finished successfully in {duration:.2f}s ---", file=log_stream)

    except Exception as e:
        tb = traceback.format_exc()
        err_msg = f"Error during Hercules run: {e}"
        print(f"{err_msg}\nTraceback:\n{tb}", file=sys.stderr)
        final_status_alert = dbc.Alert([html.I(className="bi bi-x-octagon-fill me-2"), html.Strong(f"Error: {e}")], color="danger")
        with open(log_filepath, 'a', encoding='utf-8') as log_stream:
             print(f"\n--- HERCULES RUN FAILED ---\n{err_msg}\nTraceback:\n{tb}", file=log_stream)
        results_pkg = None
    finally:
        log_content = ""
        if os.path.exists(log_filepath):
            with open(log_filepath, 'r', encoding='utf-8') as f:
                log_content = f.read()
        if results_pkg:
            results_pkg['run_log'] = log_content

    return results_pkg, final_status_alert, True, False

# =============================================================================
# Results View Callbacks
# =============================================================================
@callback(Output('reduction-type-store', 'data', allow_duplicate=True), Input('reduction-type-radio', 'value'), prevent_initial_call=True)
def update_reduction_store(selected_type):
    return selected_type

@callback(Output('cluster-summary-content', 'children'), Input('selected-cluster-store', 'data'), Input('hercules-results-store', 'data'), State('global-l0-visibility-toggle', 'value'), prevent_initial_call=True)
def update_summary_box(selected_cluster_id, results_data, current_l0_toggle_value):
    ctx = callback_context
    trigger = ctx.triggered_id
    if not results_data or not isinstance(results_data, dict):
        return no_update
    # Trigger only if selection changes or if results data is loaded and no selection is active
    if not (trigger == 'selected-cluster-store' or (trigger == 'hercules-results-store' and selected_cluster_id is None)):
        return no_update
    try:
        clusters = results_data.get('clusters', [])
        cfg = results_data.get('config', {})
        state = results_data.get('state', {})
        evals = results_data.get('eval_results', {})
        run_log = results_data.get('run_log', 'Run log was not found.')

        cluster_map = reconstruct_cluster_map_from_data(clusters)
        l0_clusters_ordered = []
        l0_orig_ids = state.get("_l0_original_ids_ordered", [])
        if l0_orig_ids and cluster_map:
            l0_map_temp = {str(c.original_id): c for c in cluster_map.values() if c.level == 0 and c.original_id is not None}
            l0_clusters_ordered = [l0_map_temp.get(str(orig_id)) for orig_id in l0_orig_ids if l0_map_temp.get(str(orig_id)) is not None]
        if not cluster_map and clusters:
            return dbc.Alert("Error reconstructing cluster map for summary.", color="warning")
        default_show_l0 = "show_l0" in (current_l0_toggle_value or [])
        return create_summary_content(
            selected_cluster_id, cluster_map, cfg, state, evals,
            default_show_l0=default_show_l0, run_log=run_log, l0_clusters_ordered=l0_clusters_ordered
        )
    except Exception as e:
        print(f"Error updating summary box: {e}\n{traceback.format_exc()}", file=sys.stderr)
        return dbc.Alert([html.I(className="bi bi-exclamation-triangle-fill me-2"), f"Summary Error: {e}"], color="danger")

@callback(Output('hierarchy-list-group', 'children'), Input('global-l0-visibility-toggle', 'value'), State('hercules-results-store', 'data'), State('selected-cluster-store', 'data'), prevent_initial_call=True)
def update_hierarchy_list_visibility(global_toggle_value, results_data, selected_cluster_id):
    if selected_cluster_id is not None or not results_data or not isinstance(results_data, dict):
        return no_update # Only update if overall summary is shown
    try:
        clusters = results_data.get('clusters', [])
        cluster_map = reconstruct_cluster_map_from_data(clusters)
        if not cluster_map:
            return html.P("Error reconstructing map for hierarchy.", className="text-danger small")
        show_l0 = "show_l0" in (global_toggle_value or [])
        top_clusters = sorted( [c for c in cluster_map.values() if c.parent is None or c.parent.id not in cluster_map], key=lambda c: c.id )
        hierarchy_items = [generate_hierarchy_list_group_item(c, cluster_map, show_l0=show_l0) for c in top_clusters]
        hierarchy_items = [item for item in hierarchy_items if item]
        if not hierarchy_items:
            return html.P("No top-level clusters found." if not top_clusters else "No hierarchy items to display (or L0 details hidden).", className="small text-muted mt-1")
        return hierarchy_items
    except Exception as e:
        print(f"Error updating hierarchy list: {e}\n{traceback.format_exc()}", file=sys.stderr)
        return dbc.Alert([html.I(className="bi bi-exclamation-triangle-fill me-2"), f"Hierarchy Error: {e}"], color="warning")

@callback(Output({'type': 'bubble-chart', 'level': ALL}, 'figure'), Input('selected-cluster-store', 'data'), Input('reduction-type-store', 'data'), Input('hercules-results-store', 'data'), Input('global-l0-visibility-toggle', 'value'), State({'type': 'bubble-chart', 'level': ALL}, 'id'), prevent_initial_call=True)
def update_all_bubble_charts(selected_cluster_id, reduction_type, results_data, l0_toggle_value, chart_ids):
    ctx = callback_context
    if not ctx.triggered or not results_data or reduction_type is None or not isinstance(results_data, dict):
        return [no_update] * len(chart_ids)
    try:
        clusters = results_data.get('clusters', [])
        state = results_data.get('state', {})
        cluster_map = reconstruct_cluster_map_from_data(clusters)
        if not cluster_map:
            return [go.Figure(layout=go.Layout(title="Error: Cluster map empty", height=100, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'))] * len(chart_ids)
        top_clusters = sorted([c for c in cluster_map.values() if c.parent is None or c.parent.id not in cluster_map], key=lambda c: c.id)
        color_map = {c.id: DEFAULT_COLOR_SEQUENCE[i % len(DEFAULT_COLOR_SEQUENCE)] for i, c in enumerate(top_clusters)}
        top_ancestor_map = {}
        queue = deque([c for c in top_clusters])
        visited_ancestry = set(c.id for c in top_clusters)
        for c_node in top_clusters:
            top_ancestor_map[c_node.id] = c_node.id
        while queue:
             parent = queue.popleft()
             parent_top_ancestor_id = top_ancestor_map.get(parent.id)
             if parent.children:
                 for child in parent.children:
                      if child and child.id in cluster_map and child.id not in visited_ancestry:
                          top_ancestor_map[child.id] = parent_top_ancestor_id
                          visited_ancestry.add(child.id)
                          queue.append(child)
        l0_clusters_ordered = []
        l0_orig_ids = state.get("_l0_original_ids_ordered", [])
        num_l0_items = state.get("num_l0_items", 0)
        if l0_orig_ids and cluster_map:
            l0_map_temp = {str(c.original_id): c for c in cluster_map.values() if c.level == 0 and c.original_id is not None}
            l0_clusters_ordered = [l0_map_temp.get(str(orig_id)) for orig_id in l0_orig_ids if str(orig_id) in l0_map_temp]
            l0_clusters_ordered = [c for c in l0_clusters_ordered if c is not None]
        elif num_l0_items > 0 and cluster_map:
            l0_clusters_ordered = sorted([c for c in cluster_map.values() if c.level == 0], key=lambda c: str(c.original_id) if c.original_id is not None else "")
        figures = {chart_id['level']: create_bubble_chart_figure(chart_id['level'], selected_cluster_id, reduction_type, cluster_map, top_ancestor_map, color_map, l0_clusters_ordered) for chart_id in chart_ids}
        return [figures.get(chart_id['level'], go.Figure()) for chart_id in chart_ids]
    except Exception as e:
        print(f"Error updating bubble charts: {e}\n{traceback.format_exc()}", file=sys.stderr)
        return [go.Figure(layout=go.Layout(title=f"Plot Error: {e}", height=100, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'))] * len(chart_ids)

@callback(Output('l0-plot-container', 'hidden'), Input('global-l0-visibility-toggle', 'value'), prevent_initial_call=True)
def toggle_l0_plot_visibility(global_toggle_value):
    return "show_l0" not in (global_toggle_value or [])

@callback(Output('selected-cluster-store', 'data', allow_duplicate=True), Input({'type': 'bubble-chart', 'level': ALL}, 'clickData'), State('hercules-results-store', 'data'), prevent_initial_call=True)
def update_selection_on_bubble_click(click_data_list, results_data):
    ctx = callback_context
    if not ctx.triggered:
        return no_update
    click_data = ctx.triggered[0]['value'] # Get the specific clickData that triggered
    if not click_data or 'points' not in click_data or not click_data['points']:
        return no_update
    point_data = click_data['points'][0]
    if 'customdata' in point_data:
        selected_id = point_data['customdata']
        try:
             if results_data and isinstance(results_data, dict):
                 cluster_map = reconstruct_cluster_map_from_data(results_data.get('clusters', []))
                 if get_cluster_by_id(selected_id, cluster_map) is not None:
                     print(f"Selection updated via bubble click: ID {selected_id}")
                     return selected_id
                 else:
                     print(f"Warning: Clicked bubble ID {selected_id} invalid/not found in current cluster map.", file=sys.stderr)
             else:
                 print("Warning: Cannot validate bubble click ID, results data missing.", file=sys.stderr)
        except Exception as e:
            print(f"Error validating clicked bubble ID {selected_id}: {e}", file=sys.stderr)
    return no_update

@callback(
    Output('search-results-div', 'children'),
    Input('search-input', 'value'),
    Input('global-l0-visibility-toggle', 'value'),
    State('hercules-results-store', 'data'),
    prevent_initial_call=True
)
def update_search_results(search_term, global_l0_toggle_value, results_data):
    if not results_data or not isinstance(results_data, dict):
        return dbc.Alert("Results not loaded.", color="warning", className="small")

    if search_term is None or len(search_term) < 2:
        return html.P("Enter 2 or more characters to search.", className="text-muted small")

    try:
        cluster_map = reconstruct_cluster_map_from_data(results_data.get('clusters', []))
        if not cluster_map:
            return dbc.Alert("Error: Cluster map empty for search.", color="warning", className="small")

        include_l0 = "show_l0" in (global_l0_toggle_value or [])
        term = search_term.lower()
        matches = []
        for cluster in cluster_map.values():
            if not isinstance(cluster, Cluster):
                continue # Skip if not a valid cluster object
            if not include_l0 and cluster.level == 0:
                continue

            title = cluster.title or ""
            desc = cluster.description or ""
            orig_id_str = str(cluster.original_id) if cluster.level == 0 and cluster.original_id is not None else ""

            if term in title.lower() or \
               term in desc.lower() or \
               (include_l0 and cluster.level == 0 and term in orig_id_str.lower()):

                label_parts = [f"L{cluster.level} ID:{cluster.id}"]
                if cluster.level == 0 and cluster.original_id is not None:
                    orig_id_preview = str(cluster.original_id)
                    if len(orig_id_preview) > 20:
                        orig_id_preview = orig_id_preview[:20] + "…"
                    label_parts.append(f"(Orig: {orig_id_preview})")

                title_preview = str(title)
                if len(title_preview) > 40:
                    title_preview = title_preview[:40] + "…"
                elif not title_preview:
                    title_preview = '(No Title)'
                label_parts.append(f"- {title_preview}")
                label = " ".join(label_parts)
                full_title_attr = f"L{cluster.level} ID: {cluster.id}\nOrig ID: {orig_id_str}\nTitle: {title}\nDesc: {(desc[:100] + '...' if len(desc) > 100 else desc)}"

                matches.append(dbc.Button(
                    label,
                    id={'type': 'select-cluster-button', 'index': cluster.id},
                    color="secondary",
                    outline=True,
                    size="sm",
                    className="me-1 mb-1 d-block w-100 text-start text-truncate",
                    n_clicks=0,
                    title=full_title_attr
                ))
                if len(matches) >= 100:
                    matches.append(html.Em(f" (Showing first 100 results)", className="small d-block mt-1 text-muted"))
                    break

        return matches if matches else html.P("No matches found.", className="small text-muted")
    except Exception as e:
        print(f"Search Error: {e}\n{traceback.format_exc()}", file=sys.stderr)
        return dbc.Alert([html.I(className="bi bi-exclamation-triangle-fill me-2"), f"Search Error: {e}"], color="danger", className="small")

@callback(Output('selected-cluster-store', 'data', allow_duplicate=True), Input({'type': 'select-cluster-button', 'index': ALL}, 'n_clicks'), State('hercules-results-store', 'data'), prevent_initial_call=True)
def update_selection_on_button_click(n_clicks_list, results_data):
    ctx = callback_context
    if not ctx.triggered or not any(n and n > 0 for n in n_clicks_list):
        return no_update
    button_id_str = ctx.triggered[0]['prop_id'].split('.')[0]
    try:
        button_id_dict = json.loads(button_id_str)
        if button_id_dict.get('type') == 'select-cluster-button' and 'index' in button_id_dict:
            cluster_id = button_id_dict.get('index')
            if results_data and isinstance(results_data, dict):
                 cluster_map = reconstruct_cluster_map_from_data(results_data.get('clusters', []))
                 if get_cluster_by_id(cluster_id, cluster_map) is not None:
                     print(f"Selection updated via button/link click: ID {cluster_id}")
                     return cluster_id
                 else:
                     print(f"Warning: Clicked button ID {cluster_id} invalid or not found in map.", file=sys.stderr)
            else:
                print("Warning: Cannot validate clicked button ID, no results data available.", file=sys.stderr)
    except Exception as e:
        print(f"Error parsing clicked button ID '{button_id_str}': {e}", file=sys.stderr)
    return no_update

@callback(Output('selected-cluster-store', 'data', allow_duplicate=True), Input('clear-selection-button', 'n_clicks'), prevent_initial_call=True)
def clear_selection(n_clicks):
    if n_clicks and n_clicks > 0:
        print("Selection cleared.")
        return None
    return no_update

app.clientside_callback(
    """
    function(n_clicks_list, current_selection) {
        const ctx = dash_clientside.callback_context;
        if (!ctx.triggered || ctx.triggered.length === 0) return window.dash_clientside.no_update;
        const trigger = ctx.triggered[0];
        if (!trigger.value || trigger.value === 0) return window.dash_clientside.no_update; // No actual click
        const trigger_id_str = trigger.prop_id.split('.')[0];
        let is_select_button = false;
        try { is_select_button = (JSON.parse(trigger_id_str).type === 'select-cluster-button'); } catch(e) {}
        if (is_select_button) {
             const summaryDiv = document.getElementById('cluster-summary-div');
             if (summaryDiv) { summaryDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' }); }
        }
        return window.dash_clientside.no_update; // Only for side-effect (scroll)
    }
    """,
    Output('scroll-trigger-store', 'data'), # Dummy output for clientside callback
    Input({'type': 'select-cluster-button', 'index': ALL}, 'n_clicks'),
    State('selected-cluster-store', 'data'), prevent_initial_call=True)

# --- Download Callbacks ---
@callback(
    Output('download-full-results-json', 'data'),
    Input('btn-download-full-results', 'n_clicks'),
    State('hercules-results-store', 'data'),
    State('session-id-store', 'data'),
    prevent_initial_call=True
)
def download_full_results(n_clicks, results_data, session_id):
    if not n_clicks or not results_data:
        return no_update
    filename = get_download_filename("hercules_full_results", "json", session_id)
    content = json.dumps(results_data, indent=2, default=str) # Use default=str for broader type compatibility
    return dict(content=content, filename=filename)

def capture_print_hierarchy(top_clusters_list: List[Cluster], cluster_map: Dict[int, Cluster], indent_increment: int = 2, print_l0_setting: bool = False) -> str:
    log_stream = io.StringIO()
    # Ensure all items in top_clusters_list are actual Cluster objects before sorting
    valid_top_clusters = [c for c in top_clusters_list if isinstance(c, Cluster)]
    sorted_top_clusters = sorted(valid_top_clusters, key=lambda c: c.num_items, reverse=True)
    for i, cluster_obj in enumerate(sorted_top_clusters):
        log_stream.write(f"\n--- Top Level Cluster {i+1} (ID: {cluster_obj.id}, Size: {cluster_obj.num_items}) ---\n")
        with contextlib.redirect_stdout(log_stream): # Capture print output from method
            cluster_obj.print_hierarchy(max_depth=None, indent_increment=indent_increment, print_level_0=print_l0_setting)
    return log_stream.getvalue()

@callback(
    Output('download-hierarchy-txt', 'data'),
    Input('btn-download-hierarchy-txt', 'n_clicks'),
    State('hercules-results-store', 'data'),
    State('session-id-store', 'data'),
    State('global-l0-visibility-toggle', 'value'),
    prevent_initial_call=True
)
def download_hierarchy_text(n_clicks, results_data, session_id, l0_toggle_value):
    if not n_clicks or not results_data:
        return no_update
    clusters_serial = results_data.get('clusters')
    if not clusters_serial:
        return no_update
    cluster_map = reconstruct_cluster_map_from_data(clusters_serial)
    if not cluster_map:
        return no_update
    top_clusters = [c for c in cluster_map.values() if c.parent is None or c.parent.id not in cluster_map]
    indent_increment = results_data.get('config', {}).get('cluster_print_indent_increment', 2)
    print_l0_for_download = "show_l0" in (l0_toggle_value or [])
    hierarchy_text = capture_print_hierarchy(top_clusters, cluster_map, indent_increment, print_l0_for_download)
    filename = get_download_filename("hercules_hierarchy", "txt", session_id)
    return dict(content=hierarchy_text, filename=filename)

@callback(
    Output('download-evaluation-json', 'data'),
    Input('btn-download-evaluation-json', 'n_clicks'),
    State('hercules-results-store', 'data'),
    State('session-id-store', 'data'),
    prevent_initial_call=True
)
def download_evaluation_results(n_clicks, results_data, session_id):
    if not n_clicks or not results_data or 'eval_results' not in results_data:
        return no_update
    eval_res = results_data.get('eval_results', {})
    filename = get_download_filename("hercules_evaluation_results", "json", session_id)
    content = json.dumps(eval_res, indent=2, default=str)
    return dict(content=content, filename=filename)

def generate_membership_df_from_store(results_data_store: Dict) -> pd.DataFrame:
    clusters_serial = results_data_store.get('clusters')
    state_data = results_data_store.get('state', {})
    if not clusters_serial or not state_data:
        return pd.DataFrame()
    all_clusters_map = reconstruct_cluster_map_from_data(clusters_serial)
    l0_orig_ids_ordered_from_store = state_data.get("_l0_original_ids_ordered", [])
    l0_clusters_ordered_obj = []
    if l0_orig_ids_ordered_from_store and all_clusters_map:
        l0_map_temp = {str(c.original_id): c for c in all_clusters_map.values() if c.level == 0 and c.original_id is not None}
        l0_clusters_ordered_obj = [l0_map_temp.get(str(orig_id)) for orig_id in l0_orig_ids_ordered_from_store if str(orig_id) in l0_map_temp]
        l0_clusters_ordered_obj = [c for c in l0_clusters_ordered_obj if c is not None] # Filter out None if ID mismatch
    if not l0_clusters_ordered_obj:
        return pd.DataFrame()
    _max_level = state_data.get("max_level_achieved", 0)
    valid_l0_details_list = ['original_data_type', 'title']
    all_rows_data = []
    original_ids_list = [c.original_id for c in l0_clusters_ordered_obj]
    for i, l0_cluster in enumerate(l0_clusters_ordered_obj):
        row_data = {'original_id': l0_cluster.original_id}
        for field_name in valid_l0_details_list:
            row_data[field_name] = getattr(l0_cluster, field_name, None)
        current_node = l0_cluster
        while current_node is not None and current_node.parent is not None:
            parent_node = current_node.parent
            parent_level = parent_node.level
            if 0 < parent_level <= _max_level:
                level_prefix = f"L{parent_level}"
                row_data[f"{level_prefix}_cluster_id"] = parent_node.id
                row_data[f"{level_prefix}_cluster_title"] = parent_node.title
            current_node = parent_node
            if parent_level >= _max_level:
                break # Stop if we've reached or exceeded max level
        all_rows_data.append(row_data)
    if not all_rows_data:
        return pd.DataFrame()
    # Use original_id as index if they are unique and suitable, otherwise default RangeIndex
    df_index = pd.RangeIndex(len(original_ids_list))
    if len(set(original_ids_list)) == len(original_ids_list) and all(isinstance(id_val, (str, int, float)) for id_val in original_ids_list):
        try:
            df_index = pd.Index(original_ids_list, name="original_id_index")
        except Exception:
            pass # Fallback to RangeIndex if pd.Index creation fails
    membership_df = pd.DataFrame(all_rows_data, index=df_index)
    expected_columns = ['original_id'] + valid_l0_details_list
    for level_num in range(1, _max_level + 1):
        expected_columns.extend([f"L{level_num}_cluster_id", f"L{level_num}_cluster_title"])
    membership_df = membership_df.reindex(columns=expected_columns) # Ensure all expected columns exist
    for level_num in range(1, _max_level + 1): # Convert ID columns to nullable integers if possible
        id_col = f"L{level_num}_cluster_id"
        if id_col in membership_df.columns:
            try:
                membership_df[id_col] = membership_df[id_col].astype('Int64') # Pandas nullable integer
            except Exception:
                pass # Keep as is if conversion fails
    return membership_df

@callback(
    Output('download-membership-csv', 'data'),
    Input('btn-download-membership-csv', 'n_clicks'),
    State('hercules-results-store', 'data'),
    State('session-id-store', 'data'),
    prevent_initial_call=True
)
def download_membership_csv(n_clicks, results_data, session_id):
    if not n_clicks or not results_data:
        return no_update
    df = generate_membership_df_from_store(results_data)
    if df.empty:
        return dict(content="No membership data to download.", filename="error.txt")
    filename = get_download_filename("hercules_membership", "csv", session_id)
    csv_string = df.to_csv(index=False) # Do not write DataFrame index to CSV
    return dict(content=csv_string, filename=filename)

@callback(
    Output('download-run-log-txt', 'data'),
    Input('btn-download-run-log', 'n_clicks'),
    State('hercules-results-store', 'data'),
    State('session-id-store', 'data'),
    prevent_initial_call=True
)
def download_run_log(n_clicks, results_data, session_id):
    if not n_clicks or not results_data:
        return no_update

    log_content = results_data.get('run_log')

    if not log_content:
        return no_update

    log_text = log_content if isinstance(log_content, str) else "Log content not available as simple text."
    filename = get_download_filename("hercules_run_log", "txt", session_id)
    return dict(content=log_text, filename=filename)

# --- Run the App ---
if __name__ == '__main__':
    print("Starting Hercules Dash Application...")
    debug_mode = os.environ.get("DASH_DEBUG_MODE", "False").lower() in ["true", "1", "t"]
    host_ip = os.environ.get("DASH_HOST_IP", "0.0.0.0")
    port = int(os.environ.get("DASH_PORT", 8050))
    print(f"Running in {'DEBUG' if debug_mode else 'PRODUCTION'} mode.")
    print(f"Accessible at http://{host_ip}:{port} (or http://127.0.0.1:{port} if host is 0.0.0.0)")
    app.run(debug=debug_mode, host=host_ip, port=port)
