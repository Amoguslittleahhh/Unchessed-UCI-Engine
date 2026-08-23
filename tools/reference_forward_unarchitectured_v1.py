#!/usr/bin/env python3
"""Ground-truth reference forward pass for UnarchitecturedV1Student, computed from
the real exported UNARCHV1 package, for validating the Rust port against.

Usage:
  python3 tools/reference_forward_unarchitectured_v1.py artifacts/unarchitectured-v1-final.unarchv1
  python3 tools/reference_forward_unarchitectured_v1.py artifacts/unarchitectured-v1-final.unarchv1 --all-exits
"""
import json
import math
import struct
import sys
import zlib
from pathlib import Path

if any(arg in ("-h", "--help") for arg in sys.argv[1:]):
    print(__doc__)
    raise SystemExit(0)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MAGIC = b"UNARCHV1"
HEADER = struct.Struct("<8sHHIQQ16sIIQ")
ENTRY = struct.Struct("<HBBI8IQQfiII128s")
DTYPE_MAP = {1: np.int8, 4: np.float32}
FLAG_METADATA = 1 << 1


def read_package(path):
    data = Path(path).read_bytes()
    magic, version, header_size, section_count, table_bytes, payload_bytes, model_uuid, table_crc, payload_crc, reserved = HEADER.unpack(
        data[:64]
    )
    assert magic == MAGIC
    table = data[64 : 64 + table_bytes]
    payload = data[64 + table_bytes : 64 + table_bytes + payload_bytes]
    sections = {}
    offset = 0
    for _ in range(section_count):
        entry = data[64 + offset : 64 + offset + ENTRY.size]
        (
            name_len, dtype, ndim, flags, s0, s1, s2, s3, s4, s5, s6, s7,
            data_offset, length, scale, zero_point, crc, reserved2, name_bytes,
        ) = ENTRY.unpack(entry)
        name = name_bytes[:name_len].decode("ascii")
        shape = (s0, s1, s2, s3, s4, s5, s6, s7)[:ndim]
        offset += ENTRY.size
        raw = payload[data_offset : data_offset + length]
        if flags & FLAG_METADATA:
            continue
        arr = np.frombuffer(raw, dtype=DTYPE_MAP[dtype]).reshape(shape)
        if dtype == 1:
            arr = arr.astype(np.float32) * scale
        sections[name] = torch.from_numpy(arr.copy())
    return sections


def rmsnorm(values, scale, width):
    normalized = values * torch.rsqrt(values.float().pow(2).mean(-1, keepdim=True) + 1e-6)
    return normalized.to(values.dtype) * scale[:width]


def elastic_block(values, geometric_bias, width, heads, w):
    prefix = w["prefix"]
    normalized = rmsnorm(values, w[f"{prefix}.norm_attention.scale"], width)
    qkv_full = w[f"{prefix}.qkv"]  # (3, W, W)
    qkv_weight = qkv_full[:, :width, :width].reshape(3 * width, width)
    batch, tokens, _ = values.shape
    qkv = F.linear(normalized, qkv_weight).view(batch, tokens, 3, heads, width // heads)
    query, key, value = qkv.unbind(2)
    attended = F.scaled_dot_product_attention(
        query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), attn_mask=geometric_bias
    )
    attended = attended.transpose(1, 2).reshape(batch, tokens, width)
    project = w[f"{prefix}.project"][:width, :width]
    project_bias = w[f"{prefix}.project_bias"][:width]
    values = values + F.linear(attended, project, project_bias)
    normalized = rmsnorm(values, w[f"{prefix}.norm_ffn.scale"], width)
    up_full = w[f"{prefix}.up"][:, :width, :width]
    up_weight = up_full.reshape(2 * width, width)
    up_bias = w[f"{prefix}.up_bias"][:, :width].reshape(2 * width)
    hidden, gate = F.linear(normalized, up_weight, up_bias).chunk(2, dim=-1)
    down = w[f"{prefix}.down"][:width, :width]
    down_bias = w[f"{prefix}.down_bias"][:width]
    values = values + F.linear(F.silu(gate) * hidden, down, down_bias)
    return values


def persona(values, width, policy_kind, history, w, prefix, rank, has_bias):
    weight = w[f"{prefix}.weight"][:width, :width]
    bias = w[f"{prefix}.bias"][:width] if has_bias else None
    base = F.linear(values, weight, bias)
    adapter_input = values + history[:, None, :width]
    a = w[f"{prefix}.adapter_a"][policy_kind, :, :width]
    b = w[f"{prefix}.adapter_b"][policy_kind, :width, :]
    low = torch.einsum("bti,bri->btr", adapter_input, a)
    return base + torch.einsum("btr,bwr->btw", low, b) / rank


def forward(w, batch, config, layers, width):
    d = config["d_model"]
    heads = config["heads"]
    pieces = batch["pieces"]
    batch_size = pieces.shape[0]
    squares = torch.arange(64).expand(batch_size, -1)
    global_state = (
        F.embedding(batch["castling"], w["castling_embedding.weight"])
        + F.embedding(batch["ep_file"], w["ep_embedding.weight"])
        + F.embedding(batch["halfmove_bucket"], w["halfmove_embedding.weight"])
    )[:, None, :width]
    values = (
        F.embedding(pieces, w["piece_embedding.weight"])[:, :, :width]
        + F.embedding(squares, w["square_embedding.weight"])[:, :, :width]
        + global_state
    )

    # GAB context
    token_projection = w["gab.token_projection"][:, :width]
    projected = F.linear(values, token_projection).flatten(1)
    hidden = F.gelu(F.linear(projected, w["gab.compress.weight"], w["gab.compress.bias"]))
    hidden = hidden * torch.rsqrt(hidden.float().pow(2).mean(-1, keepdim=True) + 1e-6)
    context = hidden.to(values.dtype) * w["gab.norm"]

    for layer in range(layers):
        coeff_w = w[f"gab.coefficients.{layer}.weight"]
        templates = w["gab.templates"]
        templates_count = templates.shape[0]
        coefficients = F.linear(context, coeff_w).view(-1, heads, templates_count)
        geometric_bias = torch.einsum("bht,tij->bhij", coefficients, templates)
        wblock = dict(w)
        wblock["prefix"] = f"blocks.{layer}"
        values = elastic_block(values, geometric_bias, width, heads, wblock)

    normalized = rmsnorm(values, w["final_norm.scale"], width)
    pooled = normalized.mean(1)
    evidence = F.softplus(F.linear(pooled, w["value_weight"][:, :width], w["value_bias"]))

    # history context (history_len=0 -> mask all False -> zero contribution besides time/rating)
    history_vec = torch.zeros(batch_size, config["history_width"])
    normalized_rating = ((batch["rating"].float() - 100.0) / 3550.0).clamp(0.0, 1.0)
    history_vec = history_vec + F.embedding(batch["time_class"], w["time_embedding.weight"])
    history_vec = history_vec + normalized_rating[:, None] * w["rating_weight"] + w["rating_bias"]
    history = F.linear(history_vec, w["history_project.weight"], w["history_project.bias"])

    body = F.gelu(persona(normalized, width, batch["policy_kind"], history, w, "policy_body", config["policy_adapter_rank"], True))
    source_values = persona(body, width, batch["policy_kind"], history, w, "policy_source", config["policy_adapter_rank"], False)
    target_values = persona(body, width, batch["policy_kind"], history, w, "policy_target", config["policy_adapter_rank"], False)

    actions = batch["safe_actions"]
    source = actions & 63
    target = (actions >> 6) & 63
    promotion = (actions >> 12).clamp(0, 4)

    def gather_squares(values_, squares_):
        return torch.gather(values_, 1, squares_[:, :, None].expand(-1, -1, values_.shape[-1]))

    source_vectors = gather_squares(source_values, source)
    target_vectors = gather_squares(target_values, target)
    logits = (source_vectors * target_vectors).sum(-1) / math.sqrt(width)
    logits = logits + F.embedding(promotion, w["promotion_bias.weight"]).squeeze(-1)
    logits = logits.masked_fill(~batch["legal_mask"], -1e4)

    regret_source_all = F.linear(normalized, w["regret_from"][:, :width])
    regret_target_all = F.linear(normalized, w["regret_to"][:, :width])
    regret_hidden = torch.tanh(
        gather_squares(regret_source_all, source)
        + gather_squares(regret_target_all, target)
        + F.embedding(promotion, w["regret_promotion.weight"])
    )
    regret_raw = F.linear(regret_hidden, w["regret_output.weight"], w["regret_output.bias"])
    regret_mean = F.softplus(regret_raw[:, :, 0])
    regret_log_scale = regret_raw[:, :, 1].clamp(-8.0, 4.0)

    return {
        "logits": logits,
        "evidence": evidence,
        "regret_mean": regret_mean,
        "regret_log_scale": regret_log_scale,
        "representation": pooled,
    }


def main():
    path = sys.argv[1]
    w = read_package(path)
    config = {
        "d_model": 256, "heads": 8, "history_width": 32, "policy_adapter_rank": 16,
    }

    # Start position, mover = White (no flip needed), 20 legal moves.
    pieces = [0] * 64
    # rank0 (a1..h1) white pieces: R N B Q K B N R
    back = [3 + 1, 1 + 1, 2 + 1, 4 + 1, 5 + 1, 2 + 1, 1 + 1, 3 + 1]  # ROOK=3,KNIGHT=1,BISHOP=2,QUEEN=4,KING=5 -> value = idx+1
    for f in range(8):
        pieces[f] = back[f]
        pieces[8 + f] = 1  # own pawn value = PAWN(0)+1=1
        pieces[48 + f] = 6 + 1  # opp pawn value = 6+PAWN+1=7
        pieces[56 + f] = 6 + back[f]  # opp back rank, value = 6+piece+1
    pieces_t = torch.tensor([pieces], dtype=torch.long)

    # 20 legal moves for the start position (mover=White): 16 pawn moves + 4 knight moves.
    actions = []
    for f in range(8):
        src = f  # rank0 file f -> a2 etc? wait pawns start rank1 (index 8..15)
    actions = []
    for f in range(8):
        src = 8 + f  # a2..h2
        actions.append(src | ((src + 8) << 6) | (0 << 12))  # one step
        actions.append(src | ((src + 16) << 6) | (0 << 12))  # two step
    # knights: b1(1)->a3(16),c3(18); g1(6)->f3(21),h3(23)
    actions.append(1 | (16 << 6))
    actions.append(1 | (18 << 6))
    actions.append(6 | (21 << 6))
    actions.append(6 | (23 << 6))
    actions.sort()
    legal_count = len(actions)
    actions = actions + [0xFFFF] * (218 - legal_count)

    batch = {
        "pieces": pieces_t,
        "castling": torch.tensor([15], dtype=torch.long),
        "ep_file": torch.tensor([8], dtype=torch.long),
        "halfmove_bucket": torch.tensor([0], dtype=torch.long),
        "rating": torch.tensor([2700], dtype=torch.long),
        "time_class": torch.tensor([2], dtype=torch.long),
        "policy_kind": torch.tensor([1], dtype=torch.long),
        "safe_actions": torch.tensor([actions], dtype=torch.long),
        "legal_mask": torch.tensor([[i < legal_count for i in range(218)]]),
    }

    out = forward(w, batch, config, layers=8, width=256)
    legal_logits = out["logits"][0][: legal_count].tolist()
    print("legal_count", legal_count)
    print("logits[:legal_count]", [round(x, 6) for x in legal_logits])
    print("evidence", out["evidence"][0].tolist())
    print("regret_mean[:legal_count]", [round(x, 6) for x in out["regret_mean"][0][:legal_count].tolist()])
    print("representation[:8]", [round(x, 6) for x in out["representation"][0][:8].tolist()])
    best = max(range(legal_count), key=lambda i: legal_logits[i])
    print("best_action_index", best, "action", actions[best], "logit", legal_logits[best])

    if "--all-exits" in sys.argv[2:]:
        for layers, width in ((2, 128), (4, 192), (8, 256)):
            result = forward(w, batch, config, layers=layers, width=width)
            values = result["logits"][0][:legal_count].tolist()
            selected = max(range(legal_count), key=lambda index: values[index])
            print(
                "exit",
                layers,
                width,
                "logits",
                [round(value, 8) for value in values],
            )
            print(
                "exit",
                layers,
                width,
                "evidence",
                [round(value, 8) for value in result["evidence"][0].tolist()],
            )
            print(
                "exit",
                layers,
                width,
                "representation",
                [round(value, 8) for value in result["representation"][0][:8].tolist()],
            )
            print("exit", layers, width, "best", selected, actions[selected])


if __name__ == "__main__":
    main()
