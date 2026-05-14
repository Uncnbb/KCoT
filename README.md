# KCoT

PyTorch implementation of **Clustering as Reasoning: A K-Means Interpretation of Chain-of-Thought Graph Learning** for **ICML 2026**.

KCoT performs graph pretraining, generates structural and feature-space prompts, refines node semantics with an LLM, encodes the refined text with a sentence-transformer, and feeds the refined embeddings back into downstream graph learning.

## Project Layout

```text
.
|-- dataset
|   |-- cora
|   |   |-- node_info.csv
|   |   |-- processed_data.pt
|   |   |-- processed_data_link_notest.pt
|   |   |-- simteg_sbert_x.pt
|   |   |-- simteg_roberta_x.pt
|   |   |-- simteg_e5_x.pt
|   |   |-- node_summaries.csv
|   |   |-- prompt
|   |   `-- 1
|   |-- pubmed
|   |-- ogbn-arxiv
|   `-- ogbn-products
|-- llm
|   |-- all-mpnet-base-v2
|   `-- vicuna-7b-v1.5-16k
|-- config.py
|-- dataloader.py
|-- gcn.py
|-- main.py
|-- model.py
|-- preprocess.py
|-- use_llm.py
|-- use_llm_API.py
|-- utils.py
|-- requirements.txt
`-- README.md
```

Large data files, checkpoints, generated CSV files, and local LLM weights are not intended to be committed to Git.

## Data Preparation

We provide Cora as an example dataset, including several intermediate generated files. You can download it from:

[KCoT Cora example data on Google Drive](https://drive.google.com/drive/folders/1Po8vZdvGB_W8k-jTEjhyyT1fsyoKhszx?usp=sharing)

Place the extracted files under:

```text
dataset/cora/
```

The only additional text-attributed-graph file used by this project is:

```text
dataset/<dataset_name>/node_info.csv
```

`node_info.csv` should contain:

- `paper_id`: the external node or paper ID used in prompts.
- `title`: node text title.
- `abstract`: node text content.

All other graph data and feature files follow the data format used by [LLaGA](https://github.com/vita-group/llaga). In particular, KCoT expects PyTorch Geometric processed graph files and SimTeG feature tensors compatible with that format:

```text
processed_data.pt
processed_data_link_notest.pt
simteg_sbert_x.pt
simteg_roberta_x.pt
simteg_e5_x.pt
```


## LLM Preparation

KCoT uses two local Hugging Face models by default:

- [sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) for encoding refined LLM text.
- [lmsys/vicuna-7b-v1.5-16k](https://huggingface.co/lmsys/vicuna-7b-v1.5-16k) for local LLM inference.

The default local layout is:

```text
llm/all-mpnet-base-v2/
llm/vicuna-7b-v1.5-16k/
```

You can override these paths in `config.py` or with environment variables:

```bash
export KCOT_EMBED_MODEL_PATH=/path/to/all-mpnet-base-v2
export KCOT_LLM_MODEL_PATH=/path/to/vicuna-7b-v1.5-16k
```

`use_llm.py` is the default local Vicuna backend. `use_llm_API.py` is kept as an optional OpenAI-compatible API backend and writes `_api_llm.csv` outputs instead of overwriting local `_local_llm.csv` outputs.

## Installation

Install PyTorch and PyTorch Geometric according to your CUDA version first. For example, with CUDA 11.8 and PyTorch 2.1:

```bash
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
pip install -r requirements.txt
```

If your CUDA or PyTorch version is different, adjust the PyG wheel URL accordingly.

## Running

The main entry point is:

```bash
python main.py
```

By default, the project runs node classification on Cora. The dataset, task, paths, model settings, and hyperparameters are configured in `config.py`.

Run node classification:

```bash
export KCOT_TASK=nc
python main.py
```

Run link prediction:

```bash
export KCOT_TASK=lp
python main.py
```

## Pipeline Outputs

During training, KCoT creates intermediate files under `dataset/<dataset_name>/`.

Prompt files:

```text
prompt/<dataset>_structural_prompts.csv
prompt/<dataset>_fusion_knn_prompts.csv
prompt/<dataset>_prompts_thought_<thought>.csv
```

LLM outputs:

```text
node_summaries.csv
<epoch>/<dataset>_refined_text_structural_local_llm.csv
<epoch>/<dataset>_refined_text_fusion_knn_local_llm.csv
```

Refined embedding files:

```text
<epoch>/<thought>_thought_embeddings.pt
<epoch>/<dataset>_refined_structural_emb.pt
<epoch>/<dataset>_refined_fusion_knn_emb.pt
```

Long LLM jobs are resumable. Incomplete CSV files use the `.partial` suffix and are finalized automatically when all expected `paper_id` rows are present.

## Configuration

Edit `config.py` to change:

- dataset name and task (`nc` or `lp`);
- dataset, checkpoint, and local LLM paths;
- GCN dimensions and training epochs;
- downstream learning rate and weight decay;
- number of KCoT thoughts;
- local Vicuna generation parameters;
- optional API backend settings.
