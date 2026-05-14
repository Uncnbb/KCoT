import os
from typing import Union

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from config import DATASET_NAME, MODEL_PATH, ROOT_PATH, global_model
from preprocess import PromptConfig, generate_prompts_dataset
from use_llm import run_llm_inference


def generate_prompt(thoughts: int, structural_prompt_enabled: bool, epoch: int):
    output_path = None

    print("=======================================================================")
    print("--- Generate Prompt by structural and evolving KNN neighbors ---")
    print(f"--- Config: structural prompt={structural_prompt_enabled} ---")
    print("=======================================================================")

    try:
        config = PromptConfig(
            ROOT_PATH,
            DATASET_NAME,
            thoughts,
            use_structural_prompt=structural_prompt_enabled,
            epoch=epoch,
        )
        output_path = generate_prompts_dataset(config)
        print("\nPrompt generation completed.")

    except NameError as e:
        print(f"\nMissing required prompt generation symbols: {e}")
    except Exception as e:
        print(f"\nUnexpected prompt generation error: {e}")
    return output_path


def use_llm(structural: bool, read_path, thought, epoch):
    print("\n--- Starting local LLM inference ---")
    return run_llm_inference(
        ROOT_PATH,
        DATASET_NAME,
        enable_structural=structural,
        read_path=read_path,
        thought=thought,
        epoch=epoch,
    )


def create_path(x, thought_num, epoch, dataset_name=DATASET_NAME):
    filename = f"{epoch}/{thought_num}_thought_embeddings.pt"
    full_path = os.path.join(ROOT_PATH, dataset_name, filename)

    parent_dir = os.path.dirname(full_path)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)
        print(f"Created missing directory: {parent_dir}")
    torch.save(x, full_path)
    print(f"Saved embedding to: {full_path}")
    return None


def load_sentence_transformer(model_path: str) -> Union[SentenceTransformer, None]:
    global global_model
    if global_model is not None:
        return global_model
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading SentenceTransformer to {device}...")
        model = SentenceTransformer(model_path, device=device)
        global_model = model
        return model
    except Exception as e:
        print(f"Load failed: {e}")
        return None


def generate_embeddings(data_name, read_path, thought, epoch):
    base_dir = os.path.join(ROOT_PATH, data_name)
    epoch_dir = os.path.join(base_dir, str(epoch))

    tasks = [
        {
            "name": "Structural Refined",
            "pt_path": os.path.join(epoch_dir, f"{data_name}_refined_structural_emb.pt"),
            "csv_path": os.path.join(epoch_dir, f"{data_name}_refined_text_structural_local_llm.csv"),
            "col": "refined_text",
        },
        {
            "name": "Fusion Refined",
            "pt_path": os.path.join(epoch_dir, f"{data_name}_refined_fusion_knn_emb.pt"),
            "csv_path": os.path.join(epoch_dir, f"{data_name}_refined_text_fusion_knn_local_llm.csv"),
            "col": "refined_text",
        },
    ]

    model = None

    print(f"\n--- Checking refined text embeddings (Epoch={epoch}) ---")

    for task in tasks:
        pt_path = task["pt_path"]
        csv_path = task["csv_path"]
        col_name = task["col"]

        if os.path.exists(pt_path):
            print(f"Exists: {os.path.basename(pt_path)}")
            continue

        if not os.path.exists(csv_path):
            print(f"Missing CSV: {csv_path}")
            continue

        print(f"Generating: {os.path.basename(pt_path)} from {os.path.basename(csv_path)}")

        if model is None:
            model = load_sentence_transformer(MODEL_PATH)
            if model is None:
                return

        try:
            df = pd.read_csv(csv_path)
            texts = df[col_name].astype(str).tolist()

            embeddings = model.encode(
                texts,
                batch_size=32,
                show_progress_bar=True,
                convert_to_tensor=True,
                device=model.device,
            )

            torch.save(embeddings, pt_path)
            print(f"Saved: {os.path.basename(pt_path)}")

        except Exception as e:
            print(f"Error: {e}")

    print("Embedding tasks processed.")


def load_thought(data, device, thought, epoch):
    epoch_dir = os.path.join(ROOT_PATH, data, str(epoch))

    p_fusion = os.path.join(epoch_dir, f"{data}_refined_fusion_knn_emb.pt")
    p_structural = os.path.join(epoch_dir, f"{data}_refined_structural_emb.pt")

    try:
        emb_f = torch.load(p_fusion, map_location=device)
        emb_s = torch.load(p_structural, map_location=device)

        print(f"Loaded 2 refined embeddings (Epoch {epoch})")
        return emb_f, emb_s

    except Exception as e:
        print(f"Load failed: {e}")
        return None, None
