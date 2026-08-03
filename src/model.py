from typing import Optional, Tuple, List, Union
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import sklearn.linear_model
from tqdm import tqdm
from dataclasses import dataclass, field
from utils.evo.tokenization import Vocab
from utils.evo.metrics import lmdata_compute_precisions
from utils.evo.tensor import symmetrize, apc

from .modules import (
    TransformerLayer,
    PKMLayer,
    AxialTransformerLayer,
    ContactPredictionHead,
    LearnedPositionalEmbedding,
    RobertaLMHead,
    RowSelfAttention,
    ColumnSelfAttention,
)

from .rna_esm.modules import (
    ESM1bLayerNorm,
    RobertaLMHead as esm2_RobertaLMHead,
    TransformerLayer as esm2_TransformerLayer,
    EmbedAdapted,
)
#from product_key_memory import PKM

#from torch.optim import lr_schedulers
from torch.optim.lr_scheduler import StepLR
#from dataset import TRRosettaContactDataset
from pathlib import Path

current_directory = Path(__file__).parent.absolute()
class ESM2(nn.Module):
    def __init__(
        self,
        vocab: Vocab,
        model_config,
        token_dropout: bool = True,
    ):
        super().__init__(
        )
        self.vocab = vocab
        self.model_config = model_config
        self.token_dropout = token_dropout
        self._init_submodules()

    def _init_submodules(self):
        config = self.model_config
        self.embed_scale = 1

        self.embed_tokens = nn.Embedding(
            len(self.vocab),
            config.embed_dim,
            padding_idx=self.vocab.pad_idx,
        )

        self.layers = nn.ModuleList(
            [
                esm2_TransformerLayer(
                    config.embed_dim,
                    4 * config.embed_dim,
                    config.num_attention_heads,
                    add_bias_kv=False,
                    use_esm1b_layer_norm=True,
                    use_rotary_embeddings=True,
                    attention_type="standard",
                    performer_attention_features=256,
                )
                for _ in range(config.num_layers)
            ]
        )

        self.contact_head = ContactPredictionHead(
            config.num_layers * config.num_attention_heads,
            self.vocab.prepend_bos,
            self.vocab.append_eos,
            eos_idx=self.vocab.eos_idx,
        )
        self.emb_layer_norm_after = ESM1bLayerNorm(config.embed_dim)

        self.lm_head = esm2_RobertaLMHead(
            embed_dim=config.embed_dim,
            output_dim=len(self.vocab),
            weight=self.embed_tokens.weight,
        )

    def forward(self, tokens, repr_layers=[12], need_head_weights=False, return_contacts=False):
        if return_contacts:
            need_head_weights = True

        assert tokens.ndim == 2
        padding_mask = tokens.eq(self.vocab.pad_idx)  # B, T

        x = self.embed_scale * self.embed_tokens(tokens)    # B, T, C

        if self.token_dropout:
            x.masked_fill_((tokens == self.vocab.mask_idx).unsqueeze(-1), 0.0)
            # x: B x T x C
            mask_ratio_train = 0.15 * 0.8
            src_lengths = (~padding_mask).sum(-1)
            mask_ratio_observed = (tokens == self.vocab.mask_idx).sum(-1).to(x.dtype) / src_lengths
            x = x * (1 - mask_ratio_train) / (1 - mask_ratio_observed)[:, None, None]

        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

        repr_layers = set(repr_layers)
        hidden_representations = {}
        if 0 in repr_layers:
            hidden_representations[0] = x

        if need_head_weights:
            attn_weights = []

        # (B, T, E) => (T, B, E)    B；batch_size, T:target_length, E: embedding_dim
        x = x.transpose(0, 1)

        if not padding_mask.any():
            padding_mask = None

        for layer_idx, layer in enumerate(self.layers):
            x, attn = layer(
                x,
                self_attn_padding_mask=padding_mask,
                need_head_weights=need_head_weights,
            )
            if (layer_idx + 1) in repr_layers:
                hidden_representations[layer_idx + 1] = x.transpose(0, 1)
            if need_head_weights:
                # (H, B, T, T) => (B, H, T, T)
                attn_weights.append(attn.transpose(1, 0))

        x = self.emb_layer_norm_after(x)
        x = x.transpose(0, 1)  # (T, B, E) => (B, T, E)

        # last hidden representation should have layer norm applied
        if (layer_idx + 1) in repr_layers:
            hidden_representations[layer_idx + 1] = x
        x = self.lm_head(x)

        result = {"logits": x, "representations": hidden_representations}
        if need_head_weights:
            # attentions: B x L x H x T x T
            attentions = torch.stack(attn_weights, 1)
            if padding_mask is not None:
                attention_mask = 1 - padding_mask.type_as(attentions)
                attention_mask = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)
                attentions = attentions * attention_mask[:, None, None, :, :]
            result["attentions"] = attentions
            if return_contacts:
                contacts = self.contact_head(tokens, attentions)
                result["contacts"] = contacts

        return result
