import torch
import torch.nn as nn

from layers.attention import AttentionLayer, FullAttention
from layers.embedding import ChannelEmbedding
from layers.transformer import Encoder, EncoderLayer
from model.s4 import S4Model


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-05):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine_weight = nn.Parameter(torch.ones(num_features))
        self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode: str, mask=None):
        if mode != "norm":
            raise ValueError("RevIN only supports mode='norm' in S4-iTNet")
        self._get_statistics(x)
        return self._normalize(x)

    def _get_statistics(self, x):
        self.mean = torch.mean(x, dim=-1, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=-1, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        return x * self.affine_weight.view(1, 1, -1) + self.affine_bias.view(1, 1, -1)


class S4iTNet(nn.Module):
    """S4-iTNet seizure type classifier."""

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.num_dy_gr = configs.num_dy_gr
        self.enc_embedding = ChannelEmbedding(
            configs.s4_d_model * configs.num_dy_gr, configs.d_model, configs.dropout
        )
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, attention_dropout=configs.dropout),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )
        self.projector1 = nn.Linear(configs.d_model, 1, bias=True)
        self.num_channels = getattr(configs, "num_channels", 22)
        self.revin = RevIN(num_features=configs.seq_len)
        self.s4 = S4Model(
            d_input=1,
            n_layers=configs.n_layers,
            d_model=configs.s4_d_model,
            dropout=configs.s4_dropout,
            l_max=configs.l_max,
        )
        self.num_classes = getattr(configs, "num_classes", 4)
        hidden = max(128, configs.d_model // 2)
        self.cls_head = nn.Sequential(
            nn.LayerNorm(configs.d_model),
            nn.Linear(configs.d_model, hidden),
            nn.GELU(),
            nn.Dropout(configs.dropout),
            nn.Linear(hidden, self.num_classes),
        )
        self.channel_pos_emb = nn.Embedding(self.num_channels, configs.d_model)
        self.pe_dropout = nn.Dropout(configs.dropout)

    def _apply_positional_encoding(self, enc_out: torch.Tensor) -> torch.Tensor:
        (B, N, _) = enc_out.shape
        if N > self.num_channels:
            raise ValueError(
                f"num channels in batch ({N}) > embedding size ({self.num_channels}); set args.num_channels >= {N}"
            )
        channel_ids = torch.arange(N, device=enc_out.device)
        pos = self.channel_pos_emb(channel_ids).unsqueeze(0).expand(B, -1, -1)
        return self.pe_dropout(enc_out + pos)

    def _forward_backbone(self, x_enc, return_logits=False):
        """
        Shared backbone used by different forward variants.
        Returns:
            binary probabilities or raw logits: (B, num_channels)
            multi_logits: (B, num_classes)
            alpha: (B, num_channels)
            pooled: (B, d_model)
        """
        x_enc = x_enc.to(dtype=next(self.parameters()).dtype)
        x_enc = self.revin(x_enc, mode="norm")
        (b, c, h) = (x_enc.shape[0], x_enc.shape[1], x_enc.shape[2])
        x_enc = x_enc.reshape(x_enc.shape[0] * x_enc.shape[1], x_enc.shape[2], 1)
        x_enc = self.s4(x_enc)
        enc_out = x_enc.view(b, c, h, -1)
        num_dynamic_graphs = self.num_dy_gr
        segment_len = h // num_dynamic_graphs
        x_tmp = []
        for t in range(num_dynamic_graphs):
            start = t * segment_len
            stop = start + segment_len
            curr_x = torch.mean(enc_out[:, :, start:stop, :], dim=2)
            x_tmp.append(curr_x)
        enc_out = torch.stack(x_tmp, dim=2)
        del x_tmp
        enc_out = enc_out.reshape(b, c, -1)
        enc_out = self.enc_embedding(enc_out)
        enc_out = self._apply_positional_encoding(enc_out)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        binary_logits = self.projector1(enc_out).squeeze(-1)
        binary_probabilities = torch.sigmoid(binary_logits)
        if binary_probabilities.shape[1] != self.num_channels:
            raise ValueError(
                f"binary_probabilities second dim {binary_probabilities.shape[1]} != configured num_channels {self.num_channels}"
            )
        alpha = torch.softmax(binary_logits.detach(), dim=1)
        pooled = torch.sum(alpha.unsqueeze(-1) * enc_out, dim=1)
        multi_logits = self.cls_head(pooled)
        binary_out = binary_logits if return_logits else binary_probabilities
        return (binary_out, multi_logits, alpha, pooled)

    def forward(self, x_enc):
        binary_probabilities, multi_logits, _, _ = self._forward_backbone(x_enc)
        return binary_probabilities, multi_logits

    def forward_with_logits(self, x_enc):
        (binary_logits, multi_logits, _, _) = self._forward_backbone(
            x_enc, return_logits=True
        )
        return (binary_logits, multi_logits)

    def forward_with_alpha(self, x_enc):
        (binary_probabilities, multi_logits, alpha, _) = self._forward_backbone(x_enc)
        return (binary_probabilities, multi_logits, alpha)

    def freeze_classifier(self):
        for param in self.cls_head.parameters():
            param.requires_grad = False
        self.cls_head.eval()

    def unfreeze_classifier(self):
        for param in self.cls_head.parameters():
            param.requires_grad = True
        self.cls_head.train()

    def enable_finetuning(self):
        for param in self.parameters():
            param.requires_grad = True
        trainable = sum(
            (
                parameter.numel()
                for parameter in self.parameters()
                if parameter.requires_grad
            )
        )
        print(f"[Stage 2] trainable parameters={trainable}", flush=True)
        self.train()
