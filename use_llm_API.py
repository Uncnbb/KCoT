import os

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

from config import API_BASE_URL, API_KEY, API_MAX_TOKENS, API_MODEL, API_TEMPERATURE
from use_llm import (
    _append_csv_row,
    _completed_count,
    _prepare_resumable_csv,
    build_full_analysis_prompt,
    build_summarize_prompt,
    load_file,
)


class ApiLLMPredictor:
    def __init__(self):
        if not API_KEY:
            raise ValueError("Missing API key. Set KCOT_API_KEY or OPENAI_API_KEY.")

        print(f"Initializing OpenAI-compatible API client: {API_MODEL}")
        self.client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

    def predict(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=API_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful research assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=API_TEMPERATURE,
                max_tokens=API_MAX_TOKENS,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            print(f"API inference error: {exc}")
            return ""


def _api_output_template(local_output_template: str) -> str:
    return local_output_template.replace("_local_llm.csv", "_api_llm.csv")


def _ensure_node_summaries_api(node_info_path, fixed_summary_path, predictor):
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
        f"[Pre-check] Generating/resuming summaries with API: "
        f"{_completed_count(rows_by_id, expected_ids)} / {len(expected_ids)} complete"
    )

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
            summary = predictor.predict(build_summarize_prompt(title, abstract))
            out_row = {
                "paper_id": paper_id,
                "summarize_text": summary,
            }
            _append_csv_row(partial_path, fieldnames, out_row)
            rows_by_id[paper_id] = out_row
            pbar.update(1)

    complete, _, _ = _prepare_resumable_csv(fixed_summary_path, expected_ids, fieldnames)
    if complete:
        print(f"Node summaries generated: {fixed_summary_path}")
    return complete


def _run_prompt_file_api(prompt_type, input_path, output_path, summary_dict, predictor):
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
        print(f"{prompt_type} API refined CSV already complete: {output_path}")
        return output_path

    print(
        f"\n=== Processing/resuming {prompt_type} with API: {input_path} "
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
            refined = predictor.predict(build_full_analysis_prompt(summary, neighbor_prompt)) if neighbor_prompt else ""
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

    complete, _, _ = _prepare_resumable_csv(output_path, expected_ids, fieldnames)
    if complete:
        print(f"Saved complete {prompt_type} API results ({len(expected_ids)} rows): {output_path}")
        return output_path

    print(f"Warning: {prompt_type} API refined CSV is still incomplete: {partial_path}")
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
    print(f"--- Starting API LLM inference (Thought = {thought}, Epoch = {epoch}) ---")
    print(f"--- Structural enabled: {enable_structural} ---")
    print("==========================================================")

    prompt_files, node_info_path, output_path_template, fixed_summary_path = load_file(ROOT_PATH, DATASET_NAME, epoch)
    output_path_template = _api_output_template(output_path_template)
    predictor = ApiLLMPredictor()

    if not _ensure_node_summaries_api(node_info_path, fixed_summary_path, predictor):
        return None

    print(f"-> Loading summaries from {fixed_summary_path}")
    summary_df = pd.read_csv(fixed_summary_path, dtype={"paper_id": str})
    summary_dict = dict(zip(summary_df["paper_id"], summary_df["summarize_text"]))

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
        result_path = _run_prompt_file_api(prompt_type, input_path, output_path, summary_dict, predictor)
        if result_path is not None:
            last_output_path = result_path

    print("\nAPI LLM inference tasks completed.")
    return last_output_path
