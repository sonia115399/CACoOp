from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class OfficialStylePromptLearner(nn.Module):
    def __init__(self, clip_model, tokenizer, classnames: list[str], ctx_init: str, n_ctx: int):
        super().__init__()
        self.clip_model = clip_model
        self.n_cls = len(classnames)
        dtype = clip_model.token_embedding.weight.dtype
        ctx_init = ctx_init.replace("_", " ").strip()
        init_tokens = ctx_init.split()
        if init_tokens:
            n_ctx = len(init_tokens)
            tokenized = tokenizer([ctx_init]).to(clip_model.token_embedding.weight.device)
            with torch.no_grad():
                embedding = clip_model.token_embedding(tokenized).to(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :].clone()
            prompt_prefix = ctx_init
        else:
            ctx_dim = clip_model.ln_final.weight.shape[0]
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        self.ctx = nn.Parameter(ctx_vectors)
        prompts = [f"{prompt_prefix} {name}." for name in classnames]
        tokenized_prompts = tokenizer(prompts).to(clip_model.token_embedding.weight.device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).to(dtype)
        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])
        self.register_buffer("tokenized_prompts", tokenized_prompts)

    def forward(self) -> torch.Tensor:
        ctx = self.ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        return torch.cat([self.token_prefix, ctx, self.token_suffix], dim=1)


def encode_prompt_features(model, learner: OfficialStylePromptLearner) -> torch.Tensor:
    prompts = learner()
    tokenized = learner.tokenized_prompts
    cast_dtype = model.transformer.get_cast_dtype()
    x = prompts.to(cast_dtype) + model.positional_embedding.to(cast_dtype)
    x = x.permute(1, 0, 2)
    x = model.transformer(x, attn_mask=model.attn_mask)
    x = x.permute(1, 0, 2)
    x = model.ln_final(x)
    pooled = x[torch.arange(x.shape[0], device=x.device), tokenized.argmax(dim=-1)]
    projection = model.text_projection
    if projection is not None:
        if isinstance(projection, nn.Linear):
            pooled = projection(pooled)
        else:
            pooled = pooled @ projection
    return F.normalize(pooled, dim=-1)
