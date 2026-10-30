"""Offline tokenizer/checkpoint round-trip through the EDLS neural KV engine.

Creates a tiny random Qwen3 checkpoint, not a pretrained language model.
Requires torch==2.14.1, numpy==2.3.5, transformers==4.57.1.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import torch
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM)
from edls_kv_reference import KVGreedyEngine, Request, uncached_greedy


def main(device, output):
    if device == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('CUDA unavailable: no GPU timing, peak-memory, or NCCL result produced.')
    if device == 'mps' and not torch.backends.mps.is_available():
        raise SystemExit('MPS unavailable: no accelerator result produced.')
    torch.set_num_threads(1)
    torch.manual_seed(8)
    texts = ['the cat reads a book', 'the model predicts the next token',
             'книга объясняет внимание', 'a book explains attention']
    tokenizer = Tokenizer(models.WordLevel(unk_token='[UNK]'))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.train_from_iterator(texts, trainers.WordLevelTrainer(
        special_tokens=['[PAD]', '[UNK]', '[BOS]', '[EOS]']))
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token='[PAD]',
        unk_token='[UNK]', bos_token='[BOS]', eos_token='[EOS]')
    config = Qwen3Config(vocab_size=len(fast), hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=128, attention_dropout=0.,
        pad_token_id=fast.pad_token_id, bos_token_id=fast.bos_token_id,
        eos_token_id=fast.eos_token_id, use_sliding_window=False)
    config._attn_implementation = 'eager'
    original = Qwen3ForCausalLM(config).eval()
    encoded = [fast.encode(text, add_special_tokens=False) for text in texts]
    assert all(fast.unk_token_id not in ids for ids in encoded)
    assert fast.encode('unseen-word', add_special_tokens=False)[0] == fast.unk_token_id
    with tempfile.TemporaryDirectory(prefix='bookvar-tiny-checkpoint-') as directory:
        path = Path(directory)
        fast.save_pretrained(path)
        original.save_pretrained(path, safe_serialization=True)
        restored_tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        restored = AutoModelForCausalLM.from_pretrained(
            path, local_files_only=True, attn_implementation='eager').eval()
        assert [restored_tokenizer.encode(t, add_special_tokens=False) for t in texts] == encoded
        with torch.inference_mode():
            ids = torch.tensor([encoded[0]])
            torch.testing.assert_close(original(ids).logits, restored(ids).logits, rtol=0, atol=0)
        manifest = {p.name: {'bytes': p.stat().st_size,
                           'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                    for p in path.iterdir() if p.is_file()}
        restored = restored.to(device)
        requests = [Request(str(i), tuple(ids), 5, eos_token_id=None)
                    for i, ids in enumerate(encoded)]
        engine = KVGreedyEngine(restored, pad_token_id=fast.pad_token_id)
        engine.prefill(requests)
        while any(not r.is_finished for r in requests):
            engine.decode(list(reversed(requests)))
        rows = []
        for text, request in zip(texts, requests):
            oracle = uncached_greedy(restored, request.prompt_ids, request.max_new_tokens)
            assert request.generated_tokens == oracle
            assert request.past_key_values.get_seq_length() == len(request.prompt_ids) + 4
            rows.append(dict(text=text, input_ids=list(request.prompt_ids),
                output_ids=request.generated_tokens,
                decoded=restored_tokenizer.decode(request.generated_tokens),
                cache_positions=request.past_key_values.get_seq_length()))
        report = dict(device=device, scope='random checkpoint and text/KV correctness; no language-quality or performance claim',
            parameters=sum(p.numel() for p in restored.parameters()), vocabulary=len(fast),
            checkpoint_files=manifest, checkpoint_bytes=sum(v['bytes'] for v in manifest.values()),
            tokenizer_roundtrip=True, checkpoint_logits_exact=True,
            oracle_matches=len(rows), rows=rows,
            cuda_available=torch.cuda.is_available(), mps_available=torch.backends.mps.is_available(),
            torch=torch.__version__, physical_peak_memory='not measured')
        if device == 'mps':
            torch.mps.synchronize()
            report['mps_allocated_bytes_at_end'] = torch.mps.current_allocated_memory()
            report['mps_driver_bytes_at_end'] = torch.mps.driver_allocated_memory()
            report['memory_note'] = 'end-of-run allocation snapshots, not peak memory'
        if device == 'cuda':
            torch.cuda.synchronize()
            report['cuda_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            report['cuda_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
            report['physical_peak_memory'] = 'PyTorch allocator peaks for this tiny fixture only'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(f'PASS {device}: tokenizer + checkpoint + {len(rows)} cached/uncached text requests')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.device, args.output)
