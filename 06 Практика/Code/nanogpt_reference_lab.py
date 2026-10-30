"""Run pinned, unmodified nanoGPT on a tiny local character stream.

uv run --offline --with torch==2.14.1 --with numpy==2.3.5 \
  --with requests==2.32.5 --with tiktoken==0.12.0 \
  python '06 Практика/Code/nanogpt_reference_lab.py' --output /tmp/nanogpt-result.json

No network, pretrained weights, real training corpus, CUDA, DDP or compilation
is used by the experiment. Temporary corpora and checkpoints are deleted.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import pickle
import platform
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import requests
import tiktoken
import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

ROOT = Path(__file__).resolve().parents[2]
PACKET = ROOT / "05 Источники/Originals/nanogpt-2026-10-30"
REVISION = "3adf61e154c3fe3fca428ad6bc3818b27a3b8291"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pinned_module():
    manifest = json.loads((PACKET / "manifest.json").read_text())
    assert manifest["nanoGPTRevision"] == REVISION
    for row in manifest["sources"]:
        assert digest(PACKET / row["file"]) == row["sha256"], row["file"]
    spec = importlib.util.spec_from_file_location("bookvar_pinned_nanogpt", PACKET / "model.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def selected_function(name, scope):
    tree = ast.parse((PACKET / "train.py").read_text())
    definition = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    code = compile(ast.Module(body=[definition], type_ignores=[]), str(PACKET / "train.py"), "exec")
    exec(code, scope)
    return scope[name]


def model_checks(module):
    torch.manual_seed(13)
    cfg = module.GPTConfig(vocab_size=7, block_size=8, n_layer=2,
                           n_head=2, n_embd=16, dropout=0.0, bias=True)
    with contextlib.redirect_stdout(io.StringIO()):
        model = module.GPT(cfg).double().eval()
    ids = torch.randint(0, 7, (4, 8))
    targets = torch.randint(0, 7, (4, 8))
    with sdpa_kernel(backends=[SDPBackend.MATH]):
        logits, loss = model(ids, targets)
        inferred, absent_loss = model(ids)
        torch.testing.assert_close(loss, F.cross_entropy(logits.reshape(-1, 7), targets.reshape(-1)))
        torch.testing.assert_close(inferred, logits[:, [-1], :])
        assert absent_loss is None and tuple(logits.shape) == (4, 8, 7)
        assert sum(p.numel() for p in model.parameters()) == 6832

        changed = ids.clone()
        changed[:, 4:] = (changed[:, 4:] + 1) % 7
        other_logits, _ = model(changed, targets)
        torch.testing.assert_close(logits[:, :4], other_logits[:, :4], rtol=0, atol=0)

        whole, micros = copy.deepcopy(model), copy.deepcopy(model)
        whole(ids, targets)[1].backward()
        for x, y in zip(ids.chunk(2), targets.chunk(2)):
            (micros(x, y)[1] / 2).backward()
        differences = []
        for p, q in zip(whole.parameters(), micros.parameters()):
            torch.testing.assert_close(p.grad, q.grad, rtol=1e-9, atol=1e-12)
            differences.append((p.grad - q.grad).abs().max().item())

        seen = []
        hook = model.register_forward_pre_hook(lambda _m, a: seen.append(a[0].shape[1]))
        generated = model.generate(ids[:1, :6], 5, temperature=1.0, top_k=1)
        hook.remove()
        assert seen == [6, 7, 8, 8, 8]
        assert generated.shape == (1, 11)

    schedule = selected_function("get_lr", dict(math=math, learning_rate=.003,
        min_lr=.0003, warmup_iters=2, lr_decay_iters=6))
    values = [schedule(i) for i in range(8)]
    assert values[:3] == [.001, .002, .003]
    assert math.isclose(values[6], .0003) and math.isclose(values[7], .0003)
    return dict(parameters=6832, training_logits=[4, 8, 7],
        inference_logits=list(inferred.shape), direct_target_loss=loss.item(),
        max_accumulation_gradient_error=max(differences),
        math_backend_forced=True, upstream_flash_attribute=model.transformer.h[0].attn.flash,
        generation_input_lengths=seen, generated_shape=list(generated.shape),
        causal_prefix_equal=True, learning_rates=values)


def run_native(work, filename, args, env):
    run = subprocess.run([sys.executable, filename, *args], cwd=work, env=env,
                         text=True, capture_output=True, timeout=60)
    if run.returncode:
        raise RuntimeError(f"{filename} exited {run.returncode}\n{run.stdout}\n{run.stderr}")
    return run


def native_checks(module):
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", PYTHONHASHSEED="0",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
    # A stale torchrun environment must not silently turn this CPU test into DDP.
    for key in ["RANK", "LOCAL_RANK", "WORLD_SIZE"]:
        env.pop(key, None)
    with tempfile.TemporaryDirectory(prefix="bookvar-nanogpt-") as directory:
        work = Path(directory)
        for name in ["model.py", "train.py", "configurator.py", "sample.py"]:
            shutil.copyfile(PACKET / name, work / name)
        data_dir = work / "data/shakespeare_char"
        data_dir.mkdir(parents=True)
        shutil.copyfile(PACKET / "data--shakespeare_char--prepare.py", data_dir / "prepare.py")
        text = "".join(np.random.default_rng(19).choice(list(" AB\n"), size=2048))
        (data_dir / "input.txt").write_text(text, encoding="utf-8")
        prepared = run_native(work, "data/shakespeare_char/prepare.py", [], env)
        with (data_dir / "meta.pkl").open("rb") as stream:
            meta = pickle.load(stream)
        assert meta["vocab_size"] == 4 and "".join(meta["itos"][i] for i in range(4)) == "\n AB"
        train_ids = np.fromfile(data_dir / "train.bin", dtype=np.uint16)
        val_ids = np.fromfile(data_dir / "val.bin", dtype=np.uint16)
        assert len(train_ids) == 1843 and len(val_ids) == 205
        decoded = "".join(meta["itos"][int(i)] for i in np.concatenate([train_ids, val_ids]))
        assert decoded == text
        scope = dict(np=np, torch=torch, os=os, data_dir=str(data_dir), block_size=8,
                     batch_size=2, device_type="cpu", device="cpu")
        get_batch = selected_function("get_batch", scope)
        x, y = get_batch("train")
        assert x.dtype == torch.int64 and tuple(x.shape) == (2, 8)
        assert torch.equal(x[:, 1:], y[:, :-1])

        common = ["--device=cpu", "--dtype=float32", "--compile=False",
            "--dataset=shakespeare_char", "--batch_size=2", "--block_size=8",
            "--n_layer=2", "--n_head=2", "--n_embd=16", "--bias=True",
            "--dropout=0.0", "--gradient_accumulation_steps=2",
            "--eval_interval=1", "--eval_iters=1", "--log_interval=1",
            "--always_save_checkpoint=True", "--decay_lr=False",
            "--learning_rate=0.001", "--wandb_log=False"]
        uninterrupted = run_native(work, "train.py", common + ["--out_dir=continuous", "--max_iters=2"], env)
        first = run_native(work, "train.py", common + ["--out_dir=resumed", "--max_iters=1"], env)
        first_checkpoint = torch.load(work / "resumed/ckpt.pt", map_location="cpu", weights_only=True)
        assert first_checkpoint["iter_num"] == 1
        assert {state["step"].item() for state in first_checkpoint["optimizer"]["state"].values()} == {1}
        resumed = run_native(work, "train.py", common + ["--out_dir=resumed", "--init_from=resume",
                                                         "--max_iters=2"], env)
        uninterrupted_checkpoint = torch.load(work / "continuous/ckpt.pt", map_location="cpu", weights_only=True)
        resumed_checkpoint = torch.load(work / "resumed/ckpt.pt", map_location="cpu", weights_only=True)
        assert set(resumed_checkpoint) == {"model", "optimizer", "model_args", "iter_num", "best_val_loss", "config"}
        assert resumed_checkpoint["iter_num"] == uninterrupted_checkpoint["iter_num"] == 2
        for checkpoint in [uninterrupted_checkpoint, resumed_checkpoint]:
            assert {s["step"].item() for s in checkpoint["optimizer"]["state"].values()} == {2}
        difference = max((resumed_checkpoint["model"][k] - value).abs().max().item()
                         for k, value in uninterrupted_checkpoint["model"].items())
        assert difference > 0, "this fixture must expose non-identical stochastic continuation"
        assert re.findall(r"^iter (\d+):", uninterrupted.stdout, re.M) == ["0", "1", "2"]
        assert re.findall(r"^iter (\d+):", first.stdout, re.M) == ["0", "1"]
        assert re.findall(r"^iter (\d+):", resumed.stdout, re.M) == ["1", "2"]

        with contextlib.redirect_stdout(io.StringIO()):
            loaded = module.GPT(module.GPTConfig(**resumed_checkpoint["model_args"])).eval()
        loaded.load_state_dict(resumed_checkpoint["model"])
        torch.manual_seed(1337)
        prefix = torch.tensor([[meta["stoi"]["A"]]])
        ids = loaded.generate(prefix, 4, temperature=1.0, top_k=1)
        expected_sample = "".join(meta["itos"][int(i)] for i in ids[0])
        sampled = run_native(work, "sample.py", ["--out_dir=resumed", "--device=cpu",
            "--dtype=float32", "--compile=False", "--num_samples=1",
            "--start=A", "--max_new_tokens=4", "--temperature=1.0", "--top_k=1"], env)
        assert "\n" + expected_sample + "\n---------------" in sampled.stdout
        # Every copied upstream program is still byte-identical after execution.
        for name in ["model.py", "train.py", "configurator.py", "sample.py"]:
            assert digest(work / name) == digest(PACKET / name)
        return dict(corpus="local fixed-seed four-character fixture, not Shakespeare",
            train_tokens=len(train_ids), val_tokens=len(val_ids), vocab_size=4,
            native_programs=["prepare.py", "train.py (scratch twice, resume once)", "sample.py"],
            optimizer_tokens_per_update=32, scratch_logged_iterations=[0, 1, 2],
            final_checkpoint_iteration=2, checkpoint_optimizer_steps=2,
            checkpoint_keys=sorted(resumed_checkpoint),
            non_identical_resume_max_weight_error=difference,
            sampled_token_ids=ids[0].tolist(), sampled_text=expected_sample,
            model_parameters=sum(p.numel() for p in loaded.parameters()),
            checkpoint_bytes=(work / "resumed/ckpt.pt").stat().st_size,
            temporary_artifacts_deleted_on_exit=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    sys.dont_write_bytecode = True
    module = pinned_module()
    checks = model_checks(module)
    native = native_checks(module)
    report = dict(status="passed", upstream_revision=REVISION,
        environment=dict(python=platform.python_version(), torch=torch.__version__,
            numpy=np.__version__, requests=requests.__version__, tiktoken=tiktoken.__version__,
            device="cpu", threads=1),
        model_checks=checks, native_execution=native,
        limits=["No CUDA, DDP, compiler, real dataset, quality result or speed benchmark.",
                "The original program is executed, not replaced by a simplified training loop.",
                "Resume state restoration is not claimed to reproduce the stochastic trajectory."])
    result = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result, encoding="utf-8")
    print(result, end="")


if __name__ == "__main__":
    main()
