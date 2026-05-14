import csv
import os
import time
import warnings

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from config import (
    LLM_DO_SAMPLE,
    LLM_MAX_INPUT_LENGTH,
    LLM_MAX_NEW_TOKENS,
    LLM_MODEL_PATH,
    LLM_REPETITION_PENALTY,
    LLM_RESUME_SUFFIX,
    LLM_TEMPERATURE,
    LLM_TOP_P,
)


warnings.filterwarnings("ignore", category=UserWarning)

LOCAL_MODEL_PATH = LLM_MODEL_PATH
global_tokenizer, global_model, global_device = None, None, None


def load_file(root_path, dataset_name, epoch):
    dataset_dir = os.path.join(root_path, dataset_name)
    prompt_dir = os.path.join(dataset_dir, "prompt")
    prompt_files = {
        "fusion_knn": os.path.join(prompt_dir, f"{dataset_name}_fusion_knn_prompts.csv"),
        "structural": os.path.join(prompt_dir, f"{dataset_name}_structural_prompts.csv"),
    }
    node_info_path = os.path.join(dataset_dir, "node_info.csv")
    fixed_summary_path = os.path.join(dataset_dir, "node_summaries.csv")
    output_path_template = os.path.join(
        dataset_dir,
        str(epoch),
        f"{dataset_name}_refined_text_{{type}}_local_llm.csv",
    )

    return prompt_files, node_info_path, output_path_template, fixed_summary_path


def _clean_value(value):
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    return value


def _read_rows_by_id(path):
    if not path or not os.path.exists(path) or os.path.getsize(path) == 0:
        return {}

    try:
        df = pd.read_csv(path, dtype={"paper_id": str})
    except pd.errors.EmptyDataError:
        return {}

    if "paper_id" not in df.columns:
        return {}

    rows = {}
    for row in df.to_dict("records"):
        paper_id = str(row.get("paper_id", "")).strip()
        if paper_id:
            rows[paper_id] = {key: _clean_value(value) for key, value in row.items()}
    return rows


def _append_csv_row(path, fieldnames, row):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    needs_header = not os.path.exists(path) or os.path.getsize(path) == 0
    clean_row = {field: _clean_value(row.get(field, "")) for field in fieldnames}

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if needs_header:
            writer.writeheader()
        writer.writerow(clean_row)
        f.flush()
        os.fsync(f.fileno())


def _write_ordered_rows(path, rows_by_id, expected_ids, fieldnames, only_existing=False):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for paper_id in expected_ids:
            if paper_id not in rows_by_id:
                if only_existing:
                    continue
                raise ValueError(f"Missing row for paper_id={paper_id}")
            writer.writerow({field: _clean_value(rows_by_id[paper_id].get(field, "")) for field in fieldnames})
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def _prepare_resumable_csv(final_path, expected_ids, fieldnames):
    partial_path = f"{final_path}{LLM_RESUME_SUFFIX}"
    rows_by_id = {}
    rows_by_id.update(_read_rows_by_id(final_path))
    rows_by_id.update(_read_rows_by_id(partial_path))

    if all(paper_id in rows_by_id for paper_id in expected_ids):
        _write_ordered_rows(final_path, rows_by_id, expected_ids, fieldnames)
        if os.path.exists(partial_path):
            os.remove(partial_path)
        return True, partial_path, rows_by_id

    if rows_by_id:
        _write_ordered_rows(partial_path, rows_by_id, expected_ids, fieldnames, only_existing=True)
    return False, partial_path, rows_by_id


def _completed_count(rows_by_id, expected_ids):
    return sum(1 for paper_id in expected_ids if paper_id in rows_by_id)


def load_local_llm(model_path: str):
    global global_tokenizer, global_model, global_device

    if global_model is not None:
        print("LLM already loaded; reusing it.")
        return global_tokenizer, global_model, global_device

    if not os.path.isdir(model_path):
        print(f"Error: local LLM path does not exist or is not a directory: {model_path}")
        return None, None, None

    try:
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print("Loading local LLM...")
        start_time = time.time()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Detected device: {device}")

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        model.eval()

        end_time = time.time()
        print(f"LLM loaded successfully in {end_time - start_time:.2f}s on {model.device}")

        global_tokenizer, global_model, global_device = tokenizer, model, device
        return tokenizer, model, device

    except Exception as e:
        print(f"Failed to load local LLM: {e}")
        return None, None, None


VICUNA_PROMPT_TEMPLATE = (
    "A chat between a helpful research assistant and a curious user.\n\n"
    "USER: {user_input}\n"
    "ASSISTANT:"
)


def build_summarize_prompt(title: str, abstract: str) -> str:
    user_input = (
        f'The title of the paper is "{title}", '
        f'the abstract of the paper is "{abstract}". '
        f"Please summarize the paper."
    )
    return VICUNA_PROMPT_TEMPLATE.format(user_input=user_input)


def build_full_analysis_prompt(summary_text: str, neighbor_prompt_text: str) -> str:
    semantic_context = (
        "The core semantic content of the central node is summarized as follows: "
        f'"{summary_text}"\n\n'
    )
    analysis_instruction = (
        f"{neighbor_prompt_text}\n\n"
        "Based strictly on the semantic content of the central node and the presence of these neighbor IDs. "
        "Do not attempt to interpret or assume the content of the neighbor IDs. "
        "Similar to cluster assignment in K-means, identify the shared aspects that contribute to their "
        "feature-space similarity, and discard nodes exhibiting low similarity. "
        "Similar to moving centroids in K-means, state the derived insights in a single, concise, and dense paragraph. "
        "Finally, integrate these insights into a compact, refined representation for the target node."
    )
    return VICUNA_PROMPT_TEMPLATE.format(user_input=semantic_context + analysis_instruction)


def ask_llm_local(prompt: str, tokenizer: AutoTokenizer, model: AutoModelForCausalLM, device: torch.device) -> str:
    if not model or not tokenizer:
        return "Error: Local LLM not initialized."

    encoded_input = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=LLM_MAX_INPUT_LENGTH,
    ).to(device)

    generation_config = GenerationConfig(
        max_new_tokens=LLM_MAX_NEW_TOKENS,
        do_sample=LLM_DO_SAMPLE,
        top_p=LLM_TOP_P,
        temperature=LLM_TEMPERATURE,
        repetition_penalty=LLM_REPETITION_PENALTY,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        return_dict_in_generate=True,
        output_scores=False,
    )

    try:
        with torch.no_grad():
            output = model.generate(
                encoded_input["input_ids"],
                attention_mask=encoded_input["attention_mask"],
                generation_config=generation_config,
            )

        generated_text = tokenizer.decode(output.sequences[0], skip_special_tokens=True)
        start_tag = "ASSISTANT:"
        if start_tag in generated_text:
            return generated_text.split(start_tag, 1)[-1].strip()
        return generated_text.strip()

    except Exception as e:
        print(f"LLM inference error: {e}")
        return "Error during local model generation."


def _ensure_node_summaries(node_info_path, fixed_summary_path):
    if not os.path.exists(node_info_path):
        print(f"Error: node info file does not exist: {node_info_path}")
        return False

    node_info_df = pd.read_csv(node_info_path, dtype={"paper_id": str})
    expected_ids = node_info_df["paper_id"].astype(str).tolist()
    fieldnames = ["paper_id", "summarize_text"]

    complete, partial_path, rows_by_id = _prepare_resumable_csv(
        fixed_summary_path,
        expected_ids,
        fieldnames,
    )
    if complete:
        print(f"Node summaries already complete: {fixed_summary_path}")
        return True

    print(
        f"[Pre-check] Generating/resuming summaries: "
        f"{_completed_count(rows_by_id, expected_ids)} / {len(expected_ids)} complete"
    )

    tokenizer, model, device = load_local_llm(LOCAL_MODEL_PATH)
    if not model:
        return False

    with tqdm(
        total=len(node_info_df),
        initial=_completed_count(rows_by_id, expected_ids),
        desc="Generating Summaries",
    ) as pbar:
        for _, row in node_info_df.iterrows():
            paper_id = str(row["paper_id"])
            if paper_id in rows_by_id:
                continue

            title = str(row.get("title", "Unknown Title"))
            abstract = str(row.get("abstract", row.get("input_text", "No content available.")))
            summary = ask_llm_local(build_summarize_prompt(title, abstract), tokenizer, model, device)
            out_row = {
                "paper_id": paper_id,
                "summarize_text": summary,
            }
            _append_csv_row(partial_path, fieldnames, out_row)
            rows_by_id[paper_id] = out_row
            pbar.update(1)

    complete, _, rows_by_id = _prepare_resumable_csv(fixed_summary_path, expected_ids, fieldnames)
    if complete:
        print(f"Node summaries generated: {fixed_summary_path}")
    return complete


def _run_prompt_file(prompt_type, input_path, output_path, summary_dict, tokenizer, model, device):
    if not os.path.exists(input_path):
        print(f"Skipping {prompt_type}; prompt file does not exist: {input_path}")
        return None

    prompt_df = pd.read_csv(input_path, dtype={"paper_id": str})
    expected_ids = prompt_df["paper_id"].astype(str).tolist()
    fieldnames = ["paper_id", "output_label", "summarize_text", "refined_text", "neighbor_prompt"]

    complete, partial_path, rows_by_id = _prepare_resumable_csv(
        output_path,
        expected_ids,
        fieldnames,
    )
    if complete:
        print(f"{prompt_type} refined CSV already complete: {output_path}")
        return output_path

    print(
        f"\n=== Processing/resuming {prompt_type}: {input_path} "
        f"({_completed_count(rows_by_id, expected_ids)} / {len(expected_ids)} complete) ==="
    )

    with tqdm(
        total=len(prompt_df),
        initial=_completed_count(rows_by_id, expected_ids),
        desc=f"Processing {prompt_type}",
    ) as pbar:
        for _, row in prompt_df.iterrows():
            node_id = str(row["paper_id"])
            if node_id in rows_by_id:
                continue

            summary = summary_dict.get(node_id, "")
            neighbor_prompt = str(row.get("prompt_text", row.get(f"prompt_{prompt_type}", "")))
            if neighbor_prompt:
                refined = ask_llm_local(build_full_analysis_prompt(summary, neighbor_prompt), tokenizer, model, device)
            else:
                refined = ""

            out_row = {
                "paper_id": node_id,
                "output_label": row.get("output_text", ""),
                "summarize_text": summary,
                "refined_text": refined,
                "neighbor_prompt": neighbor_prompt,
            }
            _append_csv_row(partial_path, fieldnames, out_row)
            rows_by_id[node_id] = out_row
            pbar.update(1)

    complete, _, rows_by_id = _prepare_resumable_csv(output_path, expected_ids, fieldnames)
    if complete:
        print(f"Saved complete {prompt_type} results ({len(expected_ids)} rows): {output_path}")
        return output_path

    print(f"Warning: {prompt_type} refined CSV is still incomplete: {partial_path}")
    return None


def run_llm_inference(
    ROOT_PATH,
    DATASET_NAME,
    enable_structural: bool,
    read_path: str,
    thought,
    epoch,
):
    print("==========================================================")
    print(f"--- Starting LLM inference (Thought = {thought}, Epoch = {epoch}) ---")
    print(f"--- Structural enabled: {enable_structural} ---")
    print("==========================================================")

    prompt_files, node_info_path, output_path_template, fixed_summary_path = load_file(ROOT_PATH, DATASET_NAME, epoch)
    if not _ensure_node_summaries(node_info_path, fixed_summary_path):
        return None

    print(f"-> Loading summaries from {fixed_summary_path}")
    summary_df = pd.read_csv(fixed_summary_path, dtype={"paper_id": str})
    summary_dict = dict(zip(summary_df["paper_id"], summary_df["summarize_text"]))

    tokenizer, model, device = load_local_llm(LOCAL_MODEL_PATH)
    if not model:
        print("LLM was not loaded; cannot run inference.")
        return None

    prompt_inputs = []
    if enable_structural:
        prompt_inputs.append(("structural", prompt_files["structural"]))

    if thought == 1:
        prompt_inputs.append(("fusion_knn", prompt_files["fusion_knn"]))
    else:
        if not read_path:
            print("Error: missing dynamic KNN prompt path for thought > 1.")
            return None
        prompt_inputs.append(("fusion_knn", read_path))

    last_output_path = None
    for prompt_type, input_path in prompt_inputs:
        output_path = output_path_template.format(type=prompt_type)
        result_path = _run_prompt_file(
            prompt_type,
            input_path,
            output_path,
            summary_dict,
            tokenizer,
            model,
            device,
        )
        if result_path is not None:
            last_output_path = result_path

    print("\nLLM inference tasks completed.")
    return last_output_path
