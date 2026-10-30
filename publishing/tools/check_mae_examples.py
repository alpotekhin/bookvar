"""Execute MAE reader programs and compare pinned methods with independent cases."""
import ast
import copy
import hashlib
import json
import re
from pathlib import Path
from types import MethodType, SimpleNamespace

import torch

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "00 Учебник/11 Pre-training и Scaling/41h Masked autoencoders — MAE и восстановление патчей.md"
SOURCE = ROOT / "05 Источники/Originals/mae-2026-10-28/mae/models_mae.py"
text = CHAPTER.read_text()
source = SOURCE.read_text()
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == "973c01f31fdf15febb5976baecd639b83a5cc011161af852e54ca68c6c2bafb4"
programs = re.findall(r"^```python\n(.*?)^```", text, re.M | re.S)
assert len(programs) == 6
reader = {}
for program in programs:
    exec(compile(program, str(CHAPTER), "exec"), reader)

source_class = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef))
source_methods = {n.name: n for n in source_class.body if isinstance(n, ast.FunctionDef)}
helpers = ["patchify", "unpatchify", "random_masking", "forward_loss"]
forwards = ["forward_encoder", "forward_decoder", "forward"]

def method_ast(node):
    # The reader normalizes trailing whitespace, including a source docstring
    # line. Preserve every operation and all docstring words in this comparison.
    node = copy.deepcopy(node)
    if (node.body and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)):
        node.body[0].value.value = "\n".join(
            line.rstrip() for line in node.body[0].value.value.split("\n"))
    return ast.dump(node, include_attributes=False)

for program, class_name, names in [(programs[0], "PatchLayout", helpers[:2]),
                                   (programs[1], "MaskedPatches", ["random_masking"]),
                                   (programs[2], "MAEOps", ["forward_loss"]),
                                   (programs[4], "TinyMAE", forwards)]:
    cls = next(n for n in ast.parse(program).body if isinstance(n, ast.ClassDef) and n.name == class_name)
    methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
    for name in names:
        assert method_ast(methods[name]) == method_ast(source_methods[name]), name

# Compile only the already inspected, unchanged source methods. Neither timm,
# the original large constructor, a checkpoint nor a dataset is required.
original_tree = ast.Module(body=[ast.ClassDef(
    name="OriginalMethods", bases=[], keywords=[], decorator_list=[],
    body=[copy.deepcopy(source_methods[n]) for n in helpers + forwards])], type_ignores=[])
original_scope = {"torch": torch}
exec(compile(ast.fix_missing_locations(original_tree), str(SOURCE), "exec"), original_scope)
OriginalMethods = original_scope["OriginalMethods"]

def source_ops(patch_size, normalized=False):
    result = OriginalMethods()
    result.patch_embed = SimpleNamespace(patch_size=(patch_size, patch_size))
    result.norm_pix_loss = normalized
    return result

torch.manual_seed(47)
layout_cases = 0
for batch in [1, 2]:
    for side in [4, 8]:
        for patch_size in [1, 2, 4]:
            ops = source_ops(patch_size)
            images = torch.arange(batch*3*side*side, dtype=torch.float64).reshape(batch, 3, side, side)
            expected = torch.stack([torch.stack([
                images[b, :, y:y+patch_size, x:x+patch_size].permute(1, 2, 0).reshape(-1)
                for y in range(0, side, patch_size) for x in range(0, side, patch_size)
            ]) for b in range(batch)])
            actual = ops.patchify(images)
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(ops.unpatchify(actual), images)
            layout_cases += 1

mask_cases = 0
for batch in [1, 2]:
    for length in [4, 9, 16]:
        for dim in [1, 3]:
            for ratio in [0., .25, .75, 1.]:
                for seed in [0, 2, 9]:
                    x = torch.randn(batch, length, dim, dtype=torch.float64, requires_grad=True)
                    torch.manual_seed(seed)
                    noise = torch.rand(batch, length)
                    order = [sorted(range(length), key=lambda j: noise[b, j].item()) for b in range(batch)]
                    keep = int(length*(1-ratio))
                    expected_ids = torch.tensor([row[:keep] for row in order], dtype=torch.long)
                    expected_restore = torch.tensor([[row.index(j) for j in range(length)] for row in order])
                    expected_mask = torch.ones(batch, length)
                    expected_mask.scatter_(1, expected_ids, 0.)
                    torch.manual_seed(seed)
                    visible, mask, inverse = source_ops(1).random_masking(x, ratio)
                    torch.testing.assert_close(inverse, expected_restore)
                    torch.testing.assert_close(mask, expected_mask)
                    torch.testing.assert_close(visible, x.gather(1, expected_ids[..., None].expand(-1, -1, dim)))
                    visible.sum().backward()
                    torch.testing.assert_close(x.grad, (1-expected_mask)[..., None].expand(-1, -1, dim).double())
                    mask_cases += 1

loss_cases = 0
for dtype in [torch.float32, torch.float64]:
    for normalized in [False, True]:
        for batch in [1, 2]:
            for side in [4, 8]:
                for patch_size in [1, 2, 4]:
                    ops = source_ops(patch_size, normalized)
                    images = torch.randn(batch, 3, side, side, dtype=dtype)
                    targets = ops.patchify(images)
                    if normalized:
                        centered = targets - targets.mean(-1, keepdim=True)
                        variance = centered.square().sum(-1, keepdim=True)/(targets.shape[-1]-1)
                        targets = centered/(variance+1e-6).sqrt()
                    predictions = torch.randn_like(targets, requires_grad=True)
                    mask = (torch.rand(batch, targets.shape[1]) > .4).to(dtype)
                    mask[0, 0] = 1
                    error = predictions-targets
                    expected_loss = (error.square()*mask[..., None]).sum()/(mask.sum()*targets.shape[-1])
                    expected_gradient = 2*error.detach()*mask[..., None]/(mask.sum()*targets.shape[-1])
                    actual = ops.forward_loss(images, predictions, mask)
                    actual.backward()
                    torch.testing.assert_close(actual, expected_loss)
                    torch.testing.assert_close(predictions.grad, expected_gradient)
                    loss_cases += 1

TinyMAE = reader["TinyMAE"]
full_cases = 0
encoder_shapes, decoder_shapes = [], []
for seed in [3, 7]:
    for batch in [1, 2]:
        for ratio in [.25, .75, 1.]:
            torch.manual_seed(seed)
            candidate = TinyMAE().double()
            original = TinyMAE().double()
            original.load_state_dict(candidate.state_dict())
            for name in helpers+forwards:
                setattr(original, name, MethodType(getattr(OriginalMethods, name), original))
            h1 = original.blocks[0].register_forward_pre_hook(lambda module, args: encoder_shapes.append(list(args[0].shape)))
            h2 = original.decoder_blocks[0].register_forward_pre_hook(lambda module, args: decoder_shapes.append(list(args[0].shape)))
            images = torch.randn(batch, 3, 4, 4, dtype=torch.float64)
            torch.manual_seed(23)
            result = candidate(images, ratio)
            torch.manual_seed(23)
            expected = original(images, ratio)
            for a, b in zip(result, expected):
                torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-10)
            result[0].backward()
            expected[0].backward()
            for (name, parameter), (other_name, other) in zip(candidate.named_parameters(), original.named_parameters()):
                assert name == other_name
                if parameter.grad is None:
                    assert other.grad is None
                else:
                    torch.testing.assert_close(parameter.grad, other.grad, rtol=1e-9, atol=1e-9)
            assert encoder_shapes[-1] == [batch, 1+int(4*(1-ratio)), 16]
            assert decoder_shapes[-1] == [batch, 5, 12]
            h1.remove()
            h2.remove()
            full_cases += 1

# A held-out input patch must not enter the encoder through patch projection.
torch.manual_seed(41)
model = TinyMAE().double()
images = torch.randn(1, 3, 4, 4, dtype=torch.float64)
torch.manual_seed(17)
latent, mask, _ = model.forward_encoder(images, .75)
pixel_mask = model.unpatchify(mask[..., None].repeat(1, 1, 12))
torch.manual_seed(17)
changed_latent, changed_mask, _ = model.forward_encoder(images+100*pixel_mask, .75)
torch.testing.assert_close(latent, changed_latent)
torch.testing.assert_close(mask, changed_mask)

print(json.dumps({
    "chapter_sha256": hashlib.sha256(CHAPTER.read_bytes()).hexdigest(),
    "programs_executed_in_order": len(programs),
    "source_method_asts_matching_after_docstring_whitespace_normalization": len(helpers+forwards),
    "independent_layout_roundtrips": layout_cases,
    "independent_mask_index_gradient_cases": mask_cases,
    "independent_loss_gradient_cases": loss_cases,
    "source_forward_and_parameter_gradient_cases": full_cases,
    "hidden_input_encoder_invariance_cases": 1,
    "forward_constructor": "small torch Transformer; not timm ViT-L initialization",
    "dataset_checkpoint_gpu_training": "not executed",
    "torch": torch.__version__,
}, indent=2))
