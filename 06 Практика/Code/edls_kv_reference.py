"""CPU teaching solution for the EDLS Week 8 prefill/decode cache contract.

Adapted from the MIT-licensed course interface at commit
e632aa89ca9e6638d52e1b686095e7442faffbb0. Copyright (c) 2021 course authors.
See /licenses/edls-e632aa89-MIT.txt. This is not an HTTP server or a benchmark.
Requires transformers==4.57.1; full-attention DynamicCache only.
"""
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache


@dataclass
class Request:
    request_id: str
    prompt_ids: tuple[int, ...]
    max_new_tokens: int
    eos_token_id: int | None = None
    generated_tokens: list[int] = field(default_factory=list)
    past_key_values: DynamicCache | None = None
    stop_reason: str | None = None

    @property
    def is_finished(self):
        return self.stop_reason is not None

    def cancel(self):
        if not self.is_finished:
            self.stop_reason = "cancelled"


@dataclass
class BatchResult:
    request_ids: list[str]
    new_tokens: list[list[int]]
    finished: list[bool]


class KVGreedyEngine:
    """Right-padded temporary batches; compact, independently owned request KV."""

    def __init__(self, model, max_context=128, pad_token_id=0):
        if not isinstance(max_context, int) or not 1 <= max_context <= model.config.max_position_embeddings:
            raise ValueError("max_context must fit the configured position limit")
        if not isinstance(pad_token_id, int) or not 0 <= pad_token_id < model.config.vocab_size:
            raise ValueError("pad token outside vocabulary")
        self.model = model.eval()
        self.max_context = max_context
        self.pad_token_id = pad_token_id
        self.device = next(model.parameters()).device

    def _validate(self, requests):
        if len({r.request_id for r in requests}) != len(requests):
            raise ValueError("duplicate request IDs in one batch")
        for r in requests:
            if not isinstance(r.max_new_tokens, int) or r.max_new_tokens < 0:
                raise ValueError("max_new_tokens must be a nonnegative integer")
            if not r.prompt_ids or len(r.prompt_ids) > self.max_context:
                raise ValueError("prompt must fit the context and contain a token")
            if any(not isinstance(t, int) or not 0 <= t < self.model.config.vocab_size
                   for t in r.prompt_ids):
                raise ValueError("prompt token outside vocabulary")
            if r.eos_token_id is not None and not 0 <= r.eos_token_id < self.model.config.vocab_size:
                raise ValueError("EOS token outside vocabulary")

    def _save(self, request, cache, row, positions):
        # index_select copies the selected columns; no view pins another row.
        owned = DynamicCache()
        for layer_idx, (key, value) in enumerate(cache):
            owned.update(key[row:row+1].index_select(2, positions),
                         value[row:row+1].index_select(2, positions), layer_idx)
        request.past_key_values = owned

    def _emit(self, request, token):
        request.generated_tokens.append(int(token))
        if request.eos_token_id is not None and token == request.eos_token_id:
            request.stop_reason = "eos"
        elif len(request.generated_tokens) >= request.max_new_tokens:
            request.stop_reason = "length"
        elif request.past_key_values.get_seq_length() >= self.max_context:
            request.stop_reason = "context"

    @staticmethod
    def _result(requests, tokens):
        return BatchResult([r.request_id for r in requests],
                           [[int(t)] for t in tokens],
                           [r.is_finished for r in requests])

    @torch.no_grad()
    def prefill(self, requests):
        self._validate(requests)
        if any(r.past_key_values is not None and not r.is_finished for r in requests):
            raise ValueError("active request was already prefilled")
        for r in requests:
            if not r.is_finished and r.max_new_tokens == 0:
                r.stop_reason = "length"
        active = [r for r in requests if not r.is_finished]
        if not active:
            return BatchResult([], [], [])
        lengths = torch.tensor([len(r.prompt_ids) for r in active], device=self.device)
        width = int(lengths.max())
        ids = torch.full((len(active), width), self.pad_token_id,
                         dtype=torch.long, device=self.device)
        for row, r in enumerate(active):
            ids[row, :len(r.prompt_ids)] = torch.tensor(r.prompt_ids, device=self.device)
        columns = torch.arange(width, device=self.device)
        mask = columns[None, :] < lengths[:, None]
        output = self.model(input_ids=ids, attention_mask=mask.long(),
                            position_ids=columns[None, :].expand(len(active), -1),
                            use_cache=True)
        rows = torch.arange(len(active), device=self.device)
        # The final physical column may be padding: gather each valid last row.
        tokens = output.logits[rows, lengths - 1].argmax(-1).tolist()
        for row, (r, token) in enumerate(zip(active, tokens)):
            positions = torch.arange(len(r.prompt_ids), device=self.device)
            self._save(r, output.past_key_values, row, positions)
            self._emit(r, token)
        return self._result(active, tokens)

    @torch.no_grad()
    def decode(self, requests):
        self._validate(requests)
        active = [r for r in requests if not r.is_finished]
        if not active:
            return BatchResult([], [], [])
        if any(r.past_key_values is None or not r.generated_tokens for r in active):
            raise ValueError("decode requires prefilled request state")
        lengths = [r.past_key_values.get_seq_length() for r in active]
        width = max(lengths)
        packed = DynamicCache()
        for layer_idx in range(self.model.config.num_hidden_layers):
            keys, values = [], []
            for r, length in zip(active, lengths):
                key, value = r.past_key_values[layer_idx]
                keys.append(F.pad(key, (0, 0, 0, width - length)))
                values.append(F.pad(value, (0, 0, 0, width - length)))
            packed.update(torch.cat(keys), torch.cat(values), layer_idx)
        mask = torch.zeros((len(active), width + 1), dtype=torch.long, device=self.device)
        for row, length in enumerate(lengths):
            mask[row, :length] = 1
        mask[:, width] = 1
        ids = torch.tensor([[r.generated_tokens[-1]] for r in active], device=self.device)
        output = self.model(input_ids=ids, attention_mask=mask,
                            position_ids=torch.tensor(lengths, device=self.device)[:, None],
                            cache_position=torch.tensor([width], device=self.device),
                            past_key_values=packed, use_cache=True)
        tokens = output.logits[:, -1].argmax(-1).tolist()
        for row, (r, length, token) in enumerate(zip(active, lengths, tokens)):
            # New KV was appended at `width`, beyond temporary right-padding.
            positions = torch.tensor([*range(length), width], device=self.device)
            self._save(r, output.past_key_values, row, positions)
            self._emit(r, token)
        return self._result(active, tokens)


@torch.no_grad()
def uncached_greedy(model, prompt_ids, max_new_tokens):
    """Independent oracle: recompute the whole valid prefix, with no cache."""
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=next(model.parameters()).device)
    generated = []
    for _ in range(max_new_tokens):
        token = int(model(input_ids=ids, use_cache=False).logits[0, -1].argmax())
        generated.append(token)
        ids = torch.cat([ids, torch.tensor([[token]], device=ids.device)], dim=1)
    return generated
