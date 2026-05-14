import torch
import torch.nn as nn
import torch.nn.functional as F

from config import DATASET_NAME


class ConditionNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, dropout=0.0):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, hidden_dim)
        self.hidden_fc = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 1)]
        )
        self.output_fc = nn.Linear(hidden_dim, output_dim)
        self.dropout_layer = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.input_fc(x)
        for layer in self.hidden_fc:
            x = F.elu(layer(x))
            x = self.dropout_layer(x)
        return self.output_fc(x)


class FusionMLP(nn.Module):
    def __init__(self, hidden_dim, n_h, n_in, dropout, num_classes, num_nodes, think_layer_num=1):
        super().__init__()
        self.think_layer_num = think_layer_num
        self.condition_layers = nn.ModuleList(
            [ConditionNet(768 * 2, hidden_dim, n_h, 1, dropout) for _ in range(think_layer_num)]
        )
        self.classifier = nn.Sequential(
            nn.Linear(n_h, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
        )
        self.cached_thoughts = None
        self.input_proj = nn.Linear(n_in, n_h) if n_in != n_h else nn.Identity()

    def forward(self, gcn, edge_index, x, update_thought, epoch, a=None):
        x = self.input_proj(x)
        origin_x = x.clone()
        current_pass_thoughts = []

        for layer_idx, condition_net in enumerate(self.condition_layers):
            embed_1 = gcn.convs[0](x, edge_index)
            embed_2 = gcn.convs[1](embed_1, edge_index) + embed_1

            if update_thought or self.cached_thoughts is None:
                thought = self.use_thought(embed_2, layer_idx, epoch, a)
                current_pass_thoughts.append(thought.detach())
            elif layer_idx >= len(self.cached_thoughts):
                print(f"Thought cache is incomplete at layer {layer_idx}; recomputing it.")
                thought = self.use_thought(embed_2, layer_idx, epoch, a)
                current_pass_thoughts.append(thought.detach())
            else:
                thought = self.cached_thoughts[layer_idx]
                current_pass_thoughts.append(thought)

            prompt = condition_net(thought)
            x = origin_x + prompt * origin_x

        if update_thought or self.cached_thoughts is None:
            self.cached_thoughts = current_pass_thoughts

        embed = gcn(x, edge_index)
        logits = self.classifier(embed)
        return x, logits

    @staticmethod
    def predict_links(node_embeddings, edge_index):
        src, dst = edge_index
        return (node_embeddings[src] * node_embeddings[dst]).sum(dim=-1)

    def use_thought(self, x, thought_counter, epoch, a):
        num_thoughts = thought_counter + 1
        from utils import create_path, generate_embeddings, generate_prompt, load_thought, use_llm

        print(f"Updating thought #{num_thoughts} at epoch {epoch}")
        create_path(x, num_thoughts, epoch)
        prompt_path = generate_prompt(num_thoughts, True, epoch)
        llm_output_path = use_llm(True, prompt_path, num_thoughts, epoch)
        generate_embeddings(DATASET_NAME, llm_output_path, num_thoughts, epoch)

        emb_fusion, emb_structural = load_thought(DATASET_NAME, x.device, num_thoughts, epoch)
        if emb_fusion is None or emb_structural is None:
            raise RuntimeError(f"Failed to load thought embeddings for thought {num_thoughts} at epoch {epoch}.")

        return torch.cat([emb_fusion, emb_structural], dim=1)
