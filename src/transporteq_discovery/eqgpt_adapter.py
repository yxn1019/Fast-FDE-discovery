"""Adapter for the original EqGPT PDE structure generator.

The adapter intentionally follows the public EqGPT code path: autoregressive
token sampling with scientific masks, reward-ranked top structures, and a small
top-k fine-tuning loop. It only generates RHS structures; the fractional
operator evaluation and bounded order correction remain in the local
discoverer.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn

from transporteq_discovery.generated_structures import GeneratedStructure, generated_structure_from_eqgpt_tokens


SPECIAL_TOKENS = {"<pad>", "E", "+", "*", "/", "S"}
DEFAULT_MAX_POS = 50
DEFAULT_D_MODEL = 768
DEFAULT_D_FF = 2048
DEFAULT_D_K = 64
DEFAULT_D_V = 64
DEFAULT_N_LAYERS = 6
DEFAULT_N_HEADS = 8


@dataclass(frozen=True)
class EqGPTSample:
    token_ids: tuple[int, ...]
    equation: str
    structure: GeneratedStructure | None
    invalid_reason: str | None = None


def infer_eqgpt_dictionary_path(checkpoint_path: str | Path, explicit_path: str | Path | None = None) -> Path:
    if explicit_path is not None:
        return Path(explicit_path)
    checkpoint = Path(checkpoint_path)
    root = checkpoint.parent.parent if checkpoint.parent.name == "results_save" else checkpoint.parent
    return root / "code" / "dict_datas_0725.json"


class EqGPTGenerator:
    """Original EqGPT sampler with device-safe PyTorch implementation."""

    def __init__(
        self,
        *,
        checkpoint_path: str | Path,
        dictionary_path: str | Path | None = None,
        seed: int = 7,
        random_exploration_probability: float = 0.2,
        device: str | torch.device | None = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"EqGPT checkpoint not found: {self.checkpoint_path}")
        self.dictionary_path = infer_eqgpt_dictionary_path(self.checkpoint_path, dictionary_path)
        if not self.dictionary_path.exists():
            raise FileNotFoundError(f"EqGPT dictionary not found: {self.dictionary_path}")
        payload = json.loads(self.dictionary_path.read_text(encoding="utf-8"))
        self.word2id = {str(key): int(value) for key, value in payload["word2id"].items()}
        self.id2word = tuple(str(item) for item in payload["id2word"])
        self.vocab_size = len(self.id2word)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.random_exploration_probability = float(random_exploration_probability)
        self._python_rng = random.Random(int(seed))
        self._numpy_rng = np.random.default_rng(int(seed))
        torch.manual_seed(int(seed))
        self.model = _EqGPTModel(vocab_size=self.vocab_size).to(self.device)
        state_dict = self._load_state_dict(self.checkpoint_path)
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "EqGPT checkpoint does not match the expected architecture: "
                f"missing={missing[:5]}, unexpected={unexpected[:5]}"
            )
        self.model.eval()
        self.mask_invalid = self._make_dimension_mask(variables=("x", "t"))

    @staticmethod
    def _load_state_dict(path: Path) -> dict[str, torch.Tensor]:
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            return torch.load(path, map_location="cpu")

    def sample_structures(self, *, count: int, max_terms: int) -> tuple[EqGPTSample, ...]:
        samples: list[EqGPTSample] = []
        for _ in range(max(0, int(count))):
            token_ids = self._sample_token_ids()
            equation = self.decode(token_ids)
            structure = generated_structure_from_eqgpt_tokens(token_ids, self.id2word, max_terms=max_terms)
            samples.append(
                EqGPTSample(
                    token_ids=token_ids,
                    equation=equation,
                    structure=structure,
                    invalid_reason=None if structure is not None else "unsupported_or_empty_rhs",
                )
            )
        return tuple(samples)

    def fine_tune(self, token_sequences: Iterable[Iterable[int]], *, epochs: int = 5, learning_rate: float = 1.0e-5) -> None:
        sequences: list[tuple[int, ...]] = []
        for seq in token_sequences:
            item = tuple(int(token) for token in seq)
            if len(item) >= 3:
                sequences.append(item)
        if not sequences or epochs <= 0:
            return
        dataset = _EqGPTFineTuneDataset(sequences, pad_id=self.word2id["<pad>"])
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=len(dataset),
            shuffle=False,
            collate_fn=dataset.collate,
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=float(learning_rate))
        criterion = nn.CrossEntropyLoss(ignore_index=self.word2id["<pad>"]).to(self.device)
        self.model.train()
        for _ in range(int(epochs)):
            for dec_inputs, dec_outputs in loader:
                dec_inputs = dec_inputs.to(self.device)
                dec_outputs = dec_outputs.to(self.device)
                optimizer.zero_grad()
                logits, _ = self.model(dec_inputs)
                loss = criterion(logits, dec_outputs.reshape(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
        self.model.eval()

    def decode(self, token_ids: Iterable[int]) -> str:
        words = [
            self.id2word[int(token)]
            for token in token_ids
            if 0 <= int(token) < len(self.id2word) and self.id2word[int(token)] not in {"<pad>", "S", "E"}
        ]
        return "".join(words)

    def _sample_token_ids(self) -> tuple[int, ...]:
        sentence = [self.word2id["S"]]
        while len(sentence) < DEFAULT_MAX_POS - 1:
            next_token = self._step(sentence)
            sentence.append(next_token)
            if next_token == self.word2id["E"]:
                break
        return tuple(sentence)

    def _step(self, sentence: list[int]) -> int:
        dec_input = torch.tensor(sentence, dtype=torch.long, device=self.device).unsqueeze(0)
        with torch.no_grad():
            dec_outputs, _ = self.model.decoder(dec_input)
            projected = self.model.projection(dec_outputs)
            probabilities = torch.nn.functional.softmax(projected, dim=2)[0, -1]
        prob_filter = probabilities * self.mask_invalid
        if dec_input.shape[1] % 2 == 0:
            prob_filter[0] = 0.0
            prob_filter[5:] = 0.0
        else:
            prob_filter[0:6] = 0.0
        prob_sum = torch.sum(prob_filter)
        if not torch.isfinite(prob_sum) or float(prob_sum) <= 0.0:
            valid = torch.nonzero(self.mask_invalid > 0, as_tuple=False).reshape(-1).cpu().numpy()
            return int(self._numpy_rng.choice(valid))
        prob_filter = (prob_filter / prob_sum).detach().cpu().numpy()
        if self._python_rng.random() <= (1.0 - self.random_exploration_probability):
            return int(self._numpy_rng.choice(np.arange(self.vocab_size), p=prob_filter.ravel()))
        valid = np.where(prob_filter != 0.0)[0]
        return int(self._numpy_rng.choice(valid))

    def _make_dimension_mask(self, *, variables: tuple[str, ...]) -> torch.Tensor:
        allowed = torch.ones(self.vocab_size, dtype=torch.float32, device=self.device)
        variable_set = set(variables)
        for idx, token in enumerate(self.id2word):
            if "y" in token and "y" not in variable_set:
                allowed[idx] = 0.0
            if "z" in token and "z" not in variable_set:
                allowed[idx] = 0.0
            if ("Laplace" in token or "BiLaplace" in token or "Div" in token) and not {"x", "y"}.issubset(variable_set):
                allowed[idx] = 0.0
        return allowed


class _EqGPTFineTuneDataset(torch.utils.data.Dataset):
    def __init__(self, sequences: list[tuple[int, ...]], *, pad_id: int) -> None:
        self.sequences = sequences
        self.pad_id = int(pad_id)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
        seq = self.sequences[index]
        return seq[:-1], seq[1:]

    def collate(self, batch: list[tuple[tuple[int, ...], tuple[int, ...]]]) -> tuple[torch.Tensor, torch.Tensor]:
        max_in = max(len(item[0]) for item in batch)
        max_out = max(len(item[1]) for item in batch)
        inputs = [list(item[0]) + [self.pad_id] * (max_in - len(item[0])) for item in batch]
        outputs = [list(item[1]) + [self.pad_id] * (max_out - len(item[1])) for item in batch]
        return torch.tensor(inputs, dtype=torch.long), torch.tensor(outputs, dtype=torch.long)


def _attention_pad_mask(seq_q: torch.Tensor, seq_k: torch.Tensor) -> torch.Tensor:
    batch_size, len_q = seq_q.size()
    _, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
    return pad_attn_mask.expand(batch_size, len_q, len_k)


def _attention_subsequence_mask(seq: torch.Tensor) -> torch.Tensor:
    attn_shape = (seq.size(0), seq.size(1), seq.size(1))
    return torch.triu(torch.ones(attn_shape, dtype=torch.bool, device=seq.device), diagonal=1)


class _ScaledDotProductAttention(nn.Module):
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(DEFAULT_D_K)
        scores.masked_fill_(attn_mask, -1.0e9)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, v)
        return context, attn


class _MultiHeadAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.W_Q = nn.Linear(DEFAULT_D_MODEL, DEFAULT_D_K * DEFAULT_N_HEADS, bias=False)
        self.W_K = nn.Linear(DEFAULT_D_MODEL, DEFAULT_D_K * DEFAULT_N_HEADS, bias=False)
        self.W_V = nn.Linear(DEFAULT_D_MODEL, DEFAULT_D_V * DEFAULT_N_HEADS, bias=False)
        self.fc = nn.Linear(DEFAULT_N_HEADS * DEFAULT_D_V, DEFAULT_D_MODEL, bias=False)
        self.layernorm = nn.LayerNorm(DEFAULT_D_MODEL)

    def forward(self, input_Q: torch.Tensor, input_K: torch.Tensor, input_V: torch.Tensor, attn_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        residual, batch_size = input_Q, input_Q.size(0)
        q = self.W_Q(input_Q).view(batch_size, -1, DEFAULT_N_HEADS, DEFAULT_D_K).transpose(1, 2)
        k = self.W_K(input_K).view(batch_size, -1, DEFAULT_N_HEADS, DEFAULT_D_K).transpose(1, 2)
        v = self.W_V(input_V).view(batch_size, -1, DEFAULT_N_HEADS, DEFAULT_D_V).transpose(1, 2)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, DEFAULT_N_HEADS, 1, 1)
        context, attn = _ScaledDotProductAttention()(q, k, v, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1, DEFAULT_N_HEADS * DEFAULT_D_V)
        output = self.fc(context)
        return self.layernorm(output + residual), attn


class _PoswiseFeedForwardNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(DEFAULT_D_MODEL, DEFAULT_D_FF, bias=False),
            nn.ReLU(),
            nn.Linear(DEFAULT_D_FF, DEFAULT_D_MODEL, bias=False),
        )
        self.layernorm = nn.LayerNorm(DEFAULT_D_MODEL)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = inputs
        return self.layernorm(self.fc(inputs) + residual)


class _DecoderLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dec_self_attn = _MultiHeadAttention()
        self.dec_enc_attn = _MultiHeadAttention()
        self.pos_ffn = _PoswiseFeedForwardNet()

    def forward(self, dec_inputs: torch.Tensor, dec_self_attn_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn


class _Decoder(nn.Module):
    def __init__(self, *, vocab_size: int) -> None:
        super().__init__()
        self.tgt_emb = nn.Embedding(vocab_size, DEFAULT_D_MODEL)
        self.pos_emb = nn.Embedding(DEFAULT_MAX_POS, DEFAULT_D_MODEL)
        self.layers = nn.ModuleList([_DecoderLayer() for _ in range(DEFAULT_N_LAYERS)])

    def forward(self, dec_inputs: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        seq_len = dec_inputs.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=dec_inputs.device)
        pos = pos.unsqueeze(0).expand_as(dec_inputs)
        dec_outputs = self.tgt_emb(dec_inputs) + self.pos_emb(pos)
        pad_mask = _attention_pad_mask(dec_inputs, dec_inputs)
        subsequence_mask = _attention_subsequence_mask(dec_inputs)
        dec_self_attn_mask = torch.gt((pad_mask + subsequence_mask), 0)
        dec_self_attns: list[torch.Tensor] = []
        for layer in self.layers:
            dec_outputs, dec_self_attn = layer(dec_outputs, dec_self_attn_mask)
            dec_self_attns.append(dec_self_attn)
        return dec_outputs, dec_self_attns


class _EqGPTModel(nn.Module):
    def __init__(self, *, vocab_size: int) -> None:
        super().__init__()
        self.decoder = _Decoder(vocab_size=vocab_size)
        self.projection = nn.Linear(DEFAULT_D_MODEL, vocab_size)

    def forward(self, dec_inputs: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        dec_outputs, dec_self_attns = self.decoder(dec_inputs)
        dec_logits = self.projection(dec_outputs)
        return dec_logits.view(-1, dec_logits.size(-1)), dec_self_attns
