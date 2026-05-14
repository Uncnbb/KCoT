import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.utils import negative_sampling
from tqdm import tqdm

from config import (
    CHECKPOINT_DIR,
    CKPT_PATH,
    CONDITION_HIDDEN_DIM,
    DOWNSTREAM_EPOCHS,
    DOWNSTREAM_LR,
    DOWNSTREAM_WEIGHT_DECAY,
    DROPOUT,
    EMBED0_PATH,
    GCN_LAYERS,
    N_H,
    N_IN,
    NEGATIVE_SAMPLE_NUM,
    PRETRAIN_EPOCHS,
    PRETRAIN_LR,
    RANDOM_SEED,
    TASK,
    THOUGHTS,
    UPDATE_THOUGHT_EVERY,
    DATASET_NAME,
)
from dataloader import load_gnn_dataset, load_lp_data_with_test_split
from gcn import PrePrompt, pca_compression, prompt_pretrain_sample
from model import FusionMLP


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main_pretrain(data, n_in, n_h, num_layers, dropout, negative_sample_num, epochs, lr):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    edge_index_np = data.edge_index.cpu().numpy()
    negative_samples = prompt_pretrain_sample(edge_index_np, n=negative_sample_num)

    model = PrePrompt(
        n_in=n_in,
        n_h=n_h,
        num_layers_num=num_layers,
        dropout=dropout,
        sample=negative_samples,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    data = data.to(device)

    best_loss = float("inf")
    best_model_state = None
    best_embed = None
    save_window = min(100, epochs)

    for epoch in tqdm(range(1, epochs + 1), desc="Pretrain"):
        model.train()
        optimizer.zero_grad()
        loss = model(data.x, data.edge_index)
        loss.backward()
        optimizer.step()

        if epoch > epochs - save_window and loss.item() < best_loss:
            best_loss = loss.item()
            best_model_state = model.state_dict()
            best_embed = model.embed(data.x, data.edge_index).detach()

    if best_model_state is not None:
        torch.save(best_model_state, CKPT_PATH)
        torch.save(best_embed, EMBED0_PATH)
        print(f"Saved pretrain checkpoint: {CKPT_PATH}")
        print(f"Saved initial embeddings: {EMBED0_PATH}")

    return model, best_embed


def _link_prediction_loss(model, node_embeddings, data, device):
    negative_base = getattr(data, "all_pos_edge_index", data.edge_index)
    neg_edge = negative_sampling(
        edge_index=negative_base,
        num_nodes=data.num_nodes,
        num_neg_samples=data.edge_index.size(1),
    ).to(device)

    pos_pred = model.predict_links(node_embeddings, data.edge_index)
    neg_pred = model.predict_links(node_embeddings, neg_edge)
    y_pred = torch.cat([pos_pred, neg_pred])
    y_true = torch.cat(
        [
            torch.ones(pos_pred.size(0), device=device),
            torch.zeros(neg_pred.size(0), device=device),
        ]
    )
    return nn.BCEWithLogitsLoss()(y_pred, y_true)


def _evaluate_link_prediction(model, node_embeddings, data):
    val_pos = model.predict_links(node_embeddings, data.val_pos_edge_index).sigmoid()
    val_neg = model.predict_links(node_embeddings, data.val_neg_edge_index).sigmoid()
    test_pos = model.predict_links(node_embeddings, data.test_pos_edge_index).sigmoid()
    test_neg = model.predict_links(node_embeddings, data.test_neg_edge_index).sigmoid()

    val_pred = torch.cat([val_pos, val_neg]).cpu().numpy()
    val_true = np.hstack([np.ones(val_pos.size(0)), np.zeros(val_neg.size(0))])
    test_pred = torch.cat([test_pos, test_neg]).cpu().numpy()
    test_true = np.hstack([np.ones(test_pos.size(0)), np.zeros(test_neg.size(0))])

    return {
        "val_auc": roc_auc_score(val_true, val_pred),
        "val_ap": average_precision_score(val_true, val_pred),
        "test_auc": roc_auc_score(test_true, test_pred),
        "test_ap": average_precision_score(test_true, test_pred),
    }


def main_downstream(
    gcn,
    data,
    n_in,
    n_h,
    hidden_dim,
    dropout,
    num_classes,
    downstream_epochs,
    downstream_lr,
    downstream_weight_decay,
    thoughts,
    update_thought_every,
    task,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)
    model = FusionMLP(
        hidden_dim=hidden_dim,
        n_h=n_h,
        n_in=n_in,
        dropout=dropout,
        num_classes=num_classes,
        num_nodes=data.num_nodes,
        think_layer_num=thoughts,
    ).to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=downstream_lr,
        weight_decay=downstream_weight_decay,
    )

    latest_x = data.x
    update_interval = max(1, update_thought_every)

    for epoch in tqdm(range(1, downstream_epochs + 1), desc="Downstream"):
        model.train()
        optimizer.zero_grad()

        update_thought = (epoch - 1) % update_interval == 0 and (epoch // update_interval) < thoughts
        new_x, logits = model(gcn, data.edge_index, data.x, update_thought, epoch)

        if task == "nc":
            loss = nn.CrossEntropyLoss()(logits[data.train_mask], data.y[data.train_mask])
        elif task == "lp":
            loss = _link_prediction_loss(model, new_x, data, device)
        else:
            raise ValueError(f"Unknown task: {task}")

        loss.backward()
        optimizer.step()
        latest_x = new_x

        if epoch % 10 == 0 or epoch == downstream_epochs:
            model.eval()
            with torch.no_grad():
                cur_x, cur_logits = model(gcn, data.edge_index, data.x, False, epoch)
                latest_x = cur_x

                if task == "nc":
                    val_pred = cur_logits[data.val_mask].argmax(dim=1)
                    val_acc = (val_pred == data.y[data.val_mask]).float().mean().item()
                    test_pred = cur_logits[data.test_mask].argmax(dim=1)
                    test_acc = (test_pred == data.y[data.test_mask]).float().mean().item()
                    print(f"Epoch {epoch:03d} | Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f}")
                elif task == "lp":
                    metrics = _evaluate_link_prediction(model, cur_x, data)
                    print(
                        f"Epoch {epoch:03d} | "
                        f"Val AUC: {metrics['val_auc']:.4f} AP: {metrics['val_ap']:.4f} | "
                        f"Test AUC: {metrics['test_auc']:.4f} AP: {metrics['test_ap']:.4f}"
                    )

    return latest_x


def _load_or_pretrain_gcn(data, device):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    if not os.path.exists(CKPT_PATH):
        print("No GCN pretrain checkpoint found; starting pretraining.")
        gcn_model, _ = main_pretrain(
            data,
            N_IN,
            N_H,
            GCN_LAYERS,
            DROPOUT,
            NEGATIVE_SAMPLE_NUM,
            PRETRAIN_EPOCHS,
            PRETRAIN_LR,
        )
        return gcn_model

    print(f"Loading GCN pretrain checkpoint: {CKPT_PATH}")
    gcn_model = PrePrompt(
        n_in=N_IN,
        n_h=N_H,
        num_layers_num=GCN_LAYERS,
        dropout=DROPOUT,
    ).to(device)
    state_dict = torch.load(CKPT_PATH, map_location=device)
    state_dict.pop("negative_sample", None)
    missing, unexpected = gcn_model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"Checkpoint loaded with missing keys={missing}, unexpected keys={unexpected}")
    return gcn_model


def main():
    set_random_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task = TASK.lower()

    if task not in {"nc", "lp"}:
        raise ValueError("TASK must be either 'nc' or 'lp'.")

    print(f"Dataset: {DATASET_NAME}")
    print(f"Task: {task}")
    print(f"Device: {device}")

    if task == "lp":
        data = load_lp_data_with_test_split(DATASET_NAME)
        num_classes = 1
    else:
        data = load_gnn_dataset(DATASET_NAME, task="nc")
        num_classes = int(torch.unique(data.y).numel())

    data.x = torch.as_tensor(pca_compression(data.x, k=N_IN), dtype=torch.float32)
    gcn_model = _load_or_pretrain_gcn(data, device)

    main_downstream(
        gcn_model.gcn,
        data,
        N_IN,
        N_H,
        CONDITION_HIDDEN_DIM,
        DROPOUT,
        num_classes,
        DOWNSTREAM_EPOCHS,
        DOWNSTREAM_LR,
        DOWNSTREAM_WEIGHT_DECAY,
        THOUGHTS,
        UPDATE_THOUGHT_EVERY,
        task=task,
    )
    print("Process completed.")


if __name__ == "__main__":
    main()
