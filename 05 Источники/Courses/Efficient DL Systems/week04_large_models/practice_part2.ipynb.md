---
title: "Practice Part2"
type: external-resource
status: imported-source
language: original
source_kind: notebook
source_commit: e632aa89ca9e6638d52e1b686095e7442faffbb0
---

> [!note] Original course material
> The material below is preserved in its original language from
> [`week04_large_models/practice_part2.ipynb`](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week04_large_models/practice_part2.ipynb) in *Efficient Deep Learning Systems* at commit
> `e632aa89ca9e6638d52e1b686095e7442faffbb0`. License: [MIT](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/LICENSE). Bookvar changed only the
> publication markup and link paths.



<a href="https://colab.research.google.com/github/mryab/efficient-dl-systems/blob/main/week04_large_models/practice_part2.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

### Efficient DL Practice: Advanced Parallelism (5 points)

In this practice session, we'll cover techniques for training large models in parallel: **Model** and **Sequence Parallelism**.
More precisely, you will implement them, and we will root for you as you go. Good luck, 🥩👜!

```python
# dependencies: the code will likely work with slightly newer/older versions, but may require minimal patching
%pip install -q transformers==4.48.3 peft==0.14.0

import transformers; assert transformers.__version__.startswith("4.48"), transformers.__version__
import peft; assert peft.__version__.startswith("0.14"), peft.__version__
```


```text
[33mWARNING: Ignoring invalid distribution ~vidia-cusparse-cu12 (/usr/local/lib/python3.11/dist-packages)[0m[33m
[0m[33mWARNING: Ignoring invalid distribution ~vidia-cusparse-cu12 (/usr/local/lib/python3.11/dist-packages)[0m[33m
[0m[33mWARNING: Ignoring invalid distribution ~vidia-cusparse-cu12 (/usr/local/lib/python3.11/dist-packages)[0m[33m
[0m
```


__Part 1: Tensor Parallelism (2 points)__
![img](https://pytorch.org/tutorials/_images/megatron_lm.png)

We'll begin by implementing a simple tensor parallelism (also known as the [original](https://papers.nips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html) model parallelism).

Our ultimate objective is to run and fine-tune a Llama 3.x model in tensor-parallel mode. However, it is rather difficult to do that in one go, especially if you take bugs into account. So we'll start simple: __here's a single Llama MLP module:__

`please read the code below carefully, it's a template for the remaining assgnments`.

```python
%%writefile tensor_parallel_mlp.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


class LlamaMLP(nn.Module):  #  based on llama 3.1 8B configuration
    def __init__(self, hidden_size: int = 4096, intermediate_size: int = 14336):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, input):
        return self.down_proj(F.silu(self.gate_proj(input)) * self.up_proj(input))


class ComputeWithAllReduce(torch.autograd.Function):
    @staticmethod  # fun fact: torch.distributed.nn has differentiable all_reduce!
    def forward(ctx, tp_shard: nn.Module, input: torch.Tensor):
        input = input.detach().requires_grad_(input.requires_grad)
        ctx.save_for_backward(input)
        ctx._tp_shard = tp_shard
        output = tp_shard(input)
        dist.all_reduce(output)
        return output
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        with torch.enable_grad():
          output = ctx._tp_shard(ctx.saved_tensors[0])
          output.backward(grad_output)
        dist.all_reduce(ctx.saved_tensors[0].grad)
        return None, ctx.saved_tensors[0].grad


class AllReduceModule(nn.Sequential):
    def forward(self, input: torch.Tensor):
        return ComputeWithAllReduce.apply(super().forward, input)


if __name__ == "__main__":
    dist.init_process_group("gloo")   # use nccl for cuda devices
    torch.manual_seed(1337)           # init weights equally on all ranks
    rank, world_size = dist.get_rank(), dist.get_world_size()

    for active_rank in range(world_size):
      dist.barrier()  # initialize each rank sequentially to save system RAM
      if rank != active_rank: continue

      # we will now implement Tensor Parallelism for the ref_module below:
      ref_module = nn.Sequential(nn.RMSNorm(4096), LlamaMLP())
      # compute reference tensors to test against them later
      input = torch.randn(1, 4096, requires_grad=True)
      ref_output = ref_module(input)
      ref_output.sum().backward()
      ref_input_grad = input.grad.clone()

      # TP step 1: define a module that computes a portion of intermediate units
      intermediate_size = ref_module[1].down_proj.in_features
      local_units = intermediate_size // world_size
      assert intermediate_size % world_size == 0
      tp_module = nn.Sequential(   # assign a portion of units per rank --v
          nn.RMSNorm(4096), AllReduceModule(LlamaMLP(intermediate_size=local_units))
      )   # all-reduce outputs during forward, all-reduce gradients on backward

      with torch.no_grad():  # copy select weights from the reference MLP
        # v-- input norm layer is too small to bother parallelizing - we replicate it!
        tp_module[0].load_state_dict(ref_module[0].state_dict())
        # up and gate projections are sharded across output units
        unit_slice = slice(local_units * rank, local_units * (rank + 1))
        tp_module[1][0].up_proj.weight[...] = ref_module[1].up_proj.weight[unit_slice]
        tp_module[1][0].gate_proj.weight[...] = ref_module[1].gate_proj.weight[unit_slice]
        # down projection is sharded across input units, matching up/gate proj
        tp_module[1][0].down_proj.weight[...] = ref_module[1].down_proj.weight[:, unit_slice]
      print(f"Initialized {rank=}", flush=True)
      del ref_module  # free RAM for next rank

    dist.barrier()  # test 1: forward pass
    tp_input = input.detach().requires_grad_(True)
    tp_output = tp_module(tp_input)
    if rank == 0:
        print(f"\nReference outputs ({rank=}):", ref_output.data, flush=True)
    for i in range(world_size):
        dist.barrier()
        if i != rank: continue
        print(f"TParallel outputs ({rank=}):", tp_output.data, flush=True)
        assert torch.allclose(tp_output, ref_output, atol=1e-6), f"output mismatch on {rank=}"

    dist.barrier()  # test 2: backward w.r.t. inputs
    assert tp_input.grad is None
    tp_output.sum().backward()
    if rank == 0:
        print(f"\nReference input grad ({rank=}):", ref_input_grad, flush=True)
    for i in range(world_size):
        dist.barrier()
        if i != rank: continue
        print(f"TParallel input grad ({rank=}):", tp_input.grad.data, flush=True)
        assert torch.allclose(tp_input.grad, ref_input_grad, atol=1e-6), f"input_grad mismatch on {rank=}"
```


```text
Overwriting tensor_parallel_mlp.py
```


```python
!OMP_NUM_THREADS=1 torchrun --nproc_per_node 4 tensor_parallel_mlp.py
```


```text
Initialized rank=0
Initialized rank=1
Initialized rank=2
Initialized rank=3

Reference outputs (rank=0): tensor([​[-0.1145,  0.0160,  0.0500,  ..., -0.1455,  0.1126, -0.0192]​])
TParallel outputs (rank=0): tensor([​[-0.1145,  0.0160,  0.0500,  ..., -0.1455,  0.1126, -0.0192]​])
TParallel outputs (rank=1): tensor([​[-0.1145,  0.0160,  0.0500,  ..., -0.1455,  0.1126, -0.0192]​])
TParallel outputs (rank=2): tensor([​[-0.1145,  0.0160,  0.0500,  ..., -0.1455,  0.1126, -0.0192]​])
TParallel outputs (rank=3): tensor([​[-0.1145,  0.0160,  0.0500,  ..., -0.1455,  0.1126, -0.0192]​])

Reference input grad (rank=0): tensor([​[ 0.0343, -0.2492, -0.1858,  ..., -0.0541,  0.0388, -0.1529]​])
TParallel input grad (rank=0): tensor([​[ 0.0343, -0.2492, -0.1858,  ..., -0.0541,  0.0388, -0.1529]​])
TParallel input grad (rank=1): tensor([​[ 0.0343, -0.2492, -0.1858,  ..., -0.0541,  0.0388, -0.1529]​])
TParallel input grad (rank=2): tensor([​[ 0.0343, -0.2492, -0.1858,  ..., -0.0541,  0.0388, -0.1529]​])
TParallel input grad (rank=3): tensor([​[ 0.0343, -0.2492, -0.1858,  ..., -0.0541,  0.0388, -0.1529]​])
```


Note that the code above lacks two details:
- it uses a form of checkpointing, but does not save random state, which would be required if you use dropout;
- it replicates RMSNorm, but it is not synchronized. Training would require all-reduce-ing gradients for those layers, e.g. by wrapping them with DDP.

```

```

```

```

```

```


__Task 1 (1 point):__ Implement tensor-parallel multi-head attention.

Like with the MLP module before, you can partition attention across multiple devices. This time, every device is to compute a portion of whole attention **heads** (and not individual units). We exploit the property that an multi-head attention layer can be viewed as a sum of individual head outputs after output projection.

For the sake of formality, this is the computation you need to parallelize:

```python
import torch
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaAttention, LlamaRotaryEmbedding
MODEL_NAME = "unsloth/Llama-3.2-1B"  # for testing (but not grading!), you may want to use Maykeye/TinyLLama-v0
config = LlamaConfig.from_pretrained(MODEL_NAME)
layer = LlamaAttention(config, layer_idx=5)
rotary_emb = LlamaRotaryEmbedding(config)

input = torch.randn(1, 128, config.hidden_size, requires_grad=True)
position_embeddings = rotary_emb(input, position_ids=torch.arange(128)[None])

output, *_etc = layer(input, attention_mask=None, position_embeddings=position_embeddings)
print(f"{output=}")
output.norm().backward()
print(f"{input.grad=}")
```


```text
/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_auth.py:94: UserWarning: 
The secret `HF_TOKEN` does not exist in your Colab secrets.
To authenticate with the Hugging Face Hub, create a token in your settings tab (https://huggingface.co/settings/tokens), set it as secret in your Google Colab and restart your session.
You will be able to reuse this secret in all of your notebooks.
Please note that authentication is recommended but still optional to access public models or datasets.
  warnings.warn(
```



```text
output=tensor([​[[ 0.0096, -0.0238,  0.0242,  ..., -0.0217, -0.0136,  0.0237],
         [-0.0025, -0.0019,  0.0408,  ..., -0.0186,  0.0170,  0.0290],
         [ 0.0110, -0.0258,  0.0225,  ..., -0.0073,  0.0033,  0.0340],
         ...,
         [ 0.0148, -0.0118,  0.0502,  ..., -0.0153, -0.0119,  0.0338],
         [ 0.0071, -0.0040,  0.0366,  ..., -0.0214, -0.0104,  0.0600],
         [-0.0003, -0.0279,  0.0330,  ..., -0.0091, -0.0124,  0.0251]​]],
       grad_fn=<UnsafeViewBackward0>)
input.grad=tensor([​[[ 0.0020,  0.0006, -0.0003,  ..., -0.0004, -0.0011, -0.0018],
         [ 0.0020,  0.0005, -0.0002,  ..., -0.0005, -0.0011, -0.0016],
         [ 0.0020,  0.0006, -0.0002,  ..., -0.0005, -0.0012, -0.0017],
         ...,
         [ 0.0019,  0.0006, -0.0002,  ..., -0.0004, -0.0011, -0.0018],
         [ 0.0019,  0.0007, -0.0002,  ..., -0.0004, -0.0012, -0.0017],
         [ 0.0020,  0.0007, -0.0002,  ..., -0.0004, -0.0011, -0.0018]​]])
```


Same as before, your task is to create a multi-head attention layer, partition it across ranks and verify two things:
- attention outputs on the same inputs (and mask) match with the non-parallel version;
- gradients w.r.t. attention inputs are the same; gradients w.r.t. mask need not be verified.

```python
%%writefile tensor_parallel_attn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


class MyLlamaAttention(nn.Module):
    ...  # please take a reference implementation of Llama attention from Hugging Face transformers:
    # https://github.com/huggingface/transformers/blob/v4.44-release/src/transformers/models/llama/modeling_llama.py#L326-L455
    # You can also directly import transformers.models.llama.modeling_llama.LlamaAttention, as in the reference above.
    # Alternatively, you are welcome to simplify their code or implement your own version.

    # Note: the link above points to an older version of attention with built-in rotary position embeddings (RoPE);
    # If you are using a newer version, please make sure to define extra inputs


# You will likely need to define additional classes below, e.g. a module to perform all-reduce


if __name__ == "__main__":
    dist.init_process_group("gloo")   # use nccl for cuda devices
    torch.manual_seed(1337)           # init weights equally on all ranks
    rank, world_size = dist.get_rank(), dist.get_world_size()

    for active_rank in range(world_size):
      dist.barrier()  # initialize each rank sequentially to save system RAM
      if rank != active_rank: continue

      # we will now implement Tensor Parallelism for the ref_module below:
      ref_module = MyLlamaAttention()
      # ^-- you may need to modify this code, e.g. pass parameters or use transformers LlamaAttention (as above)

      # generate reference tensors to test against them later
      input = torch.randn(1, 128, 4096, requires_grad=True)
      extra_inputs = dict()  # <-- OPTIONAL: either design additional inputs here, as in the reference above

      ref_output = ref_module(input, **extra_inputs)
      ref_output.sum().backward()
      ref_input_grad = input.grad.clone()

      # TP step 1: define a module that computes a portion of attention heads

      tp_module = <YOUR CODE HERE>  # create a tensor-parallel version of the Attention module

      with torch.no_grad():
          <YOUR CODE HERE>  # copy select weights from the reference attention

      print(f"Initialized {rank=}", flush=True)
      del ref_module  # free RAM for next rank

    # TEST AREA: you are free to add additional parameters, but your code *must* run the same tests as below
    dist.barrier()  # test 1: forward pass
    tp_input = input.detach().requires_grad_(True)
    tp_output = tp_module(tp_input, **extra_inputs)
    if rank == 0:
        print(f"\nReference outputs ({rank=}):", ref_output.data, flush=True)
    for i in range(world_size):
        dist.barrier()
        if i != rank: continue
        print(f"TParallel outputs ({rank=}):", tp_output.data, flush=True)
        assert torch.allclose(tp_output, ref_output, atol=1e-5), f"output mismatch on {rank=}"

    dist.barrier()  # test 2: backward w.r.t. inputs
    assert tp_input.grad is None
    tp_output.sum().backward()
    if rank == 0:
        print(f"\nReference input grad ({rank=}):", ref_input_grad, flush=True)
    for i in range(world_size):
        dist.barrier()
        if i != rank: continue
        print(f"TParallel input grad ({rank=}):", tp_input.grad.data, flush=True)
        assert torch.allclose(tp_input.grad, ref_input_grad, atol=1e-4), f"input_grad mismatch on {rank=}"
```

```python
!OMP_NUM_THREADS=1 torchrun --nproc_per_node 2 tensor_parallel_attn.py
# ^-- feel free to modify parameters, as long as there are at least 2 ranks
```

Well done! *(hopefully. If not, go back and, well... do it)*

```

```


```

```


```

```


```

```


```

```


### Full model conversion

Now let's apply this technique to parallelize the actual Llama model. As in, with weights.

__Task 2 (1 point):__ Combine the two previous techniques in one file that parallelizes an actual Llama model and .generates meaningful output. For simplicity, you do not need to partition key-value cache here - only the forward pass itself. We will default to generating tokens with recomputation.

For the sake of formality, your task is to parallelize the following inference code:

```python
import torch
import transformers
MODEL_NAME = "unsloth/Llama-3.2-1B"  # for testing (but not grading!), you may want to use Maykeye/TinyLLama-v0

tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
model = transformers.LlamaForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)  # <-- you are allowed to switch to bf16

prompt = "A quick brown fox"
input_ids = tokenizer(prompt, return_tensors='pt')["input_ids"]
print(end=prompt)
for i in range(5):
  with torch.no_grad():
    new_token = model(input_ids).logits[0, -1].argmax(-1)
    input_ids = torch.cat([input_ids, new_token.view(1, 1)], dim=1)
  print(end=tokenizer.decode(new_token), flush=True)
# pro tip: delete the model or restart session to free RAM for the TP experiments
```


```text
A quick brown fox jumps over the lazy dog
```


**Requirements:** your code must do the following things for the full grade:
- instantiate an actually trained Llama model (Llama 3.2 1B or larger is fine, maykeye is not)
- run forward pass with at least 2 ranks and verify that the logits are close,
- run backward pass w.r.t. non-parallelized input embeddings, verify that the gradients are close,
- perform inference for 10 steps to verify that the model produces meaningful outputs (see below)

You are only required to tensor-parallel-ize the transformer layers. Parallelizing embeddings and logits is optional. If you do choose to parallelize embeddings, we sincerely recommend that you partition across the embedding dim, not across tokens - so that the computation is balanced.

```python
<<A whole lot of your code here>>
```

```python
<<... and a dedicated cell to show off that it works>>
```

```

```

```

```

```

```

```

```

```

```

```

```

### Using [`torch.distributed.tensor`](https://pytorch.org/docs/stable/distributed.tensor.html)

PyTorch has an in-built functionality called [DTensor](https://pytorch.org/docs/stable/distributed.tensor.html), designed to help implementing tensor-level parallelism with various sharding strategies. This includes Tensor parallelism itself, as well as other techniques such as Sequence Parallelism, as they are both, essentially, parallelism across different tensor dimensions.

__Task 3 (1 point):__ Your next task will be to replicate your previous code (llama inference) using DTensor instead of manual AllReduce. We recommend you start by skimming the [documentation for DTensor](https://pytorch.org/docs/stable/distributed.tensor.html) to learn the interface and [the minimal example](https://github.com/pytorch/examples/blob/main/distributed/tensor_parallelism/tensor_parallel_example.py) to learn how to put the pieces together.


We recommend that you dedicate some time to learn and play with it before you proceed to parallelize Llama.

The main objective is the same as in the previous task - run .generate with DTensor - and then compare it against the manual implementation. **Please report at least some speed comparison for forward and backward passes between this and the previous task.** If absolutely impossible (e.g. you don't have multiple gpus), we can accept a fallback assignment of implementing basic training: overfit the model to a single batch (like task 5 below) and demonstrate that it works - if you choose this option, say so in bold, large-font letters somewhere where the grader can see.

But first, here's a quick demo of using DTensor for simple matrix multiplication - meant as a testbed for your experiments.

```python
%%writefile tensor_parallel_mlp_dtensor.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, DeviceMesh, Replicate, Shard
import torch.distributed.tensor.parallel as tp


class LlamaMLP(nn.Module):  # same module, but with smaller dims for quick prototyping
    def __init__(self, hidden_size: int = 1024, intermediate_size: int = 4096):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, input):
        return self.down_proj(F.silu(self.gate_proj(input)) * self.up_proj(input))


if __name__ == "__main__":
    dist.init_process_group("gloo")  # use nccl for cuda devices
    torch.manual_seed(1337)          # init weights equally on all ranks
    rank, world_size = dist.get_rank(), dist.get_world_size()

    # Initialize device mesh for tensor parallelism
    device_mesh = init_device_mesh(device_type="cpu", mesh_shape=(world_size,))  # use "cuda" for GPU

    # Create reference module for comparison
    ref_module = nn.Sequential(nn.RMSNorm(1024), LlamaMLP())

    input = torch.randn(1, 1024, requires_grad=True)
    ref_output = ref_module(input)
    ref_output.sum().backward()
    ref_input_grad = input.grad.clone()

    # Create tensor parallel module (we wrap ref_module instead of copying)
    tp_module = tp.parallelize_module(
        ref_module,
        device_mesh,
        parallelize_plan={  # define parallelism type for each module
            # up_proj and gate_proj are column-wise parallel (sharded across outputs);
            "1.up_proj": tp.ColwiseParallel(),
            "1.gate_proj": tp.ColwiseParallel(),
            # down_proj is row-wise parallel (sharded across input dim)
            "1.down_proj": tp.RowwiseParallel(),
          },  # note: RMSNorm is simply replicated across all devices - hence, we skip it
    )
    if rank == 0:  # Note: no need to copy weight chunks manually: DTensor handles parameter sharding for us
      for name, param in tp_module.named_parameters():
        print(f"{name=},\ttype={type(param.data)}\tglobal shape={param.shape},\tlocal shape={param._local_tensor.shape if hasattr(param, '_local_tensor') else param.shape}")

    dist.barrier()  # Test forward and backward pass with Tensor Parallelism
    tp_input = input.detach().requires_grad_(True)
    tp_output = tp_module(tp_input)
    tp_output.sum().backward()
    tp_output = tp_output.trigger_wait()  # convert from AsyncCollectiveTensor to regular torch tensor
    if rank == 0:
        print(f"\nReference outputs ({rank=}):", ref_output.data, flush=True)
        print(f"TParallel outputs ({rank=}):", tp_output.data, flush=True)
        print(f"\nReference input grad ({rank=}):", ref_input_grad, flush=True)
        print(f"TParallel input grad ({rank=}):", tp_input.grad, flush=True)
    dist.barrier()
    assert torch.allclose(tp_output, ref_output, atol=1e-6), f"output mismatch on {rank=}"
    assert torch.allclose(tp_input.grad, ref_input_grad, atol=1e-6), f"input_grad mismatch on {rank=}"
    print(end=f"Tests passed ({rank=})\n", flush=True); dist.barrier()

# fun fact: 90% of the code above was generated by grok-3 for prompt "Please rewrite the following code using torch.distributed.tensor ```python <paste MLP code here>```"
# the remaining 10% are nasty bugfixes that took 99% of assignment preparation time. Do not trust the shogoths yet :)
```


```text
Overwriting tensor_parallel_mlp_dtensor.py
```


```python
!OMP_NUM_THREADS=1 torchrun --nproc_per_node 2 tensor_parallel_mlp_dtensor.py
```


```text
/usr/local/lib/python3.11/dist-packages/torch/distributed/tensor/_random.py:45: UserWarning: DTensor random operators may not have complete support on cpu device mesh
  warnings.warn(
/usr/local/lib/python3.11/dist-packages/torch/distributed/tensor/_random.py:45: UserWarning: DTensor random operators may not have complete support on cpu device mesh
  warnings.warn(
name='0.weight',	type=<class 'torch.Tensor'>	global shape=torch.Size([1024]),	local shape=torch.Size([1024])
name='1.gate_proj.weight',	type=<class 'torch.distributed.tensor.DTensor'>	global shape=torch.Size([4096, 1024]),	local shape=torch.Size([2048, 1024])
name='1.up_proj.weight',	type=<class 'torch.distributed.tensor.DTensor'>	global shape=torch.Size([4096, 1024]),	local shape=torch.Size([2048, 1024])
name='1.down_proj.weight',	type=<class 'torch.distributed.tensor.DTensor'>	global shape=torch.Size([1024, 4096]),	local shape=torch.Size([1024, 2048])

Reference outputs (rank=0): tensor([​[ 0.0102,  0.0432, -0.0467,  ...,  0.0798, -0.0179,  0.0527]​])
TParallel outputs (rank=0): tensor([​[ 0.0102,  0.0432, -0.0467,  ...,  0.0798, -0.0179,  0.0527]​])

Reference input grad (rank=0): tensor([​[ 0.1543, -0.0858, -0.0882,  ..., -0.2082,  0.0298,  0.2388]​])
TParallel input grad (rank=0): tensor([​[ 0.1543, -0.0858, -0.0882,  ..., -0.2082,  0.0298,  0.2388]​])
Tests passed (rank=0)
Tests passed (rank=1)
```


```python
<<A WHOLE LOT OF YOUR CODE - apply the same strategy to implement tensor-only parallelism for full LLama model>>
```

```python
<<YOUR CODE - run the script, check correctness, then run text generation with tensor-parallel code>>
```

```

```

```

```

```

```

```

```

```

```

```

```

```

```

```

```



### Sequence Parallelism with Ulysses


Now let's parallelize the other way - across the sequence dimension. To showcase why this is necessary, our main task will be to parallelize LLM fine-tuning over a very long sequence. The way you do this, of course, is through Sequence Parallelism. You can implement naive [sequence parallelism](https://arxiv.org/abs/2205.05198), similar to [DeepSpeed Ulysses](https://arxiv.org/pdf/2309.14509) (n.b.: not the first work to do this).

![figure-from-paper](https://ar5iv.labs.arxiv.org/html/2309.14509/assets/figs/image3.png)


Here's the short version:
- All weights are replicated between ranks (optionally: FSDP)
- Each rank holds a subset of sequence tokens
- Embeddings, logits, normalizations, MLP all apply independently to token shards
- The multi-head attention is the only layer that gets special treatment
    - First, apply QKV projections to local tokens, as in data-parallel training;
    - Then re-shard so that each rank holds a **subset of heads** across **all tokens**;
    - Compute the attention ''core'' (RoPE and F.scaled_dot_product_attention) for its chunk of heads independently;
    - Re-shard outputs again so that each rank concatenates **all heads**, but only for its **subset of tokens**;
    - Apply the output ("O") projection to your local tokens again.
- This approach *may* be combined with tensor parallelism, but this is an advanced technique that you don't have to implement.


__You have a choice__ between two options on how to implement it: either manually with torch.distributed like in task 2, or using the DTensor route like in task 3. We provide some tips for both tasks.


**Option A. with raw `torch.distirbuted`:**
- Use [`dist.all_to_all`](https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all) to switch between per-token and per-head sharding without materializing the full tensor on any device;
- Wrap the model with [`DistributedDataParallel`](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) or [`FullyShardedDataParallel`](https://pytorch.org/docs/stable/fsdp.html) so that fine-tuning synchronizes trainable parameters. Note that using FSDP for parameter-efficient fine-tuning can be tricky: we recommend you either wrap **trainable modules** with separate FSDP sub-instances via auto_wrap_policy - or simply use DDP instead of FSDP.

**Option B. with `DTensor`:**
- We recommend you first skim the official [tutorial](https://pytorch.org/tutorials/intermediate/TP_tutorial.html) on applying Tensor Parallelism (sic.) - or browse the [TorchTitan's version](https://github.com/pytorch/torchtitan/blob/82afc842e303e49d1a137fc7ea48291a57f72d5d/torchtitan/models/llama/parallelize_llama.py) of it.
- Note that there is a [`SequenceParallel`](https://pytorch.org/docs/stable/distributed.tensor.parallel.html#torch.distributed.tensor.parallel.SequenceParallel) class in torch.distributed.tensor.parallel` - **however, it does not magick the sequence parallelism for you** - it is only meant for small layers (e.g. normalization). You still need to do the sharding in self-attention!

For the sake of formality, here's an example script you need to parallelize:

```python
import torch
import transformers
import peft
MODEL_NAME = "unsloth/Llama-3.2-1B"  # for testing (but not grading!), you may want to use Maykeye/TinyLLama-v0
SEQUENCE_LENGTH = 128                # IMPORTANT!!! you need to increase this parameter! Look for the maximum sequence length on one and multiple GPUs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
model = transformers.LlamaForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16).to(device)

for param in model.parameters():
  param.required_grad = False
model.gradient_checkpointing_enable()
model.enable_input_require_grads()

model = peft.get_peft_model(model, peft.PromptTuningConfig(task_type=peft.TaskType.CAUSAL_LM, num_virtual_tokens=32))
assert any(param.requires_grad for param in model.parameters()), "No trainable parameters - did you enable PEFT?"

!wget -q https://www.gutenberg.org/cache/epub/4300/pg4300.txt -O ulysses.txt  # ... or use any other text of your choosing
input_ids = tokenizer(open("ulysses.txt").read(), return_tensors='pt')['input_ids']
print(f"Cropping {input_ids.shape[1]=} to {SEQUENCE_LENGTH} tokens")
input_ids, labels = input_ids[:, :SEQUENCE_LENGTH], input_ids[:, 1:SEQUENCE_LENGTH + 1]

trainable_parameters = {p for p in model.parameters() if p.requires_grad}
print(f"Parameters: {sum(map(torch.Tensor.numel, trainable_parameters))} trainable / {sum(map(torch.Tensor.numel, model.parameters()))} total")
opt = torch.optim.Adam(trainable_parameters)
for i in range(10):
  loss = model(input_ids=input_ids.to(device), labels=labels.to(device)).loss
  opt.zero_grad()
  loss.backward()
  opt.step()
  print(f"{i=}\t{loss.item()=}")

# pro tip: delete the model or restart session to free RAM for the TP experiments
```


```text
/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_auth.py:94: UserWarning: 
The secret `HF_TOKEN` does not exist in your Colab secrets.
To authenticate with the Hugging Face Hub, create a token in your settings tab (https://huggingface.co/settings/tokens), set it as secret in your Google Colab and restart your session.
You will be able to reuse this secret in all of your notebooks.
Please note that authentication is recommended but still optional to access public models or datasets.
  warnings.warn(
```


```text
tokenizer_config.json:   0%|          | 0.00/50.6k [00:00<?, ?B/s]
```

```text
tokenizer.json:   0%|          | 0.00/17.2M [00:00<?, ?B/s]
```

```text
special_tokens_map.json:   0%|          | 0.00/459 [00:00<?, ?B/s]
```

```text
config.json:   0%|          | 0.00/935 [00:00<?, ?B/s]
```

```text
model.safetensors:   0%|          | 0.00/2.47G [00:00<?, ?B/s]
```

```text
generation_config.json:   0%|          | 0.00/230 [00:00<?, ?B/s]
```


```text
Token indices sequence length is longer than the specified maximum sequence length for this model (397368 > 131072). Running this sequence through the model will result in indexing errors
```



```text
Cropping input_ids.shape[1]=397368 to 128 tokens
Parameters: 65536 trainable / 1235879936 total
i=0	loss.item()=8.582364082336426
i=1	loss.item()=8.413138389587402
i=2	loss.item()=8.251019477844238
i=3	loss.item()=8.113922119140625
i=4	loss.item()=8.001195907592773
i=5	loss.item()=7.875072002410889
i=6	loss.item()=7.784201145172119
i=7	loss.item()=7.691455364227295
i=8	loss.item()=7.621612548828125
i=9	loss.item()=7.547983169555664
```


__Task 4 (1 point):__ before you do training, let's first parallelize a single forward pass. Implement sharding with the same interface you used in tasks 2 (or 3 if you use DTensor), but this time, parallelize across the sequence dimension. Note: if you are running out of (V)RAM, load the 1B model in half precision and disable gradients for all weights except the first (few) layers.

```python
<<A whole lot of your code here>>
```

```python
<<... and a dedicated cell to show off that it works>>
```

__Task 5 (1 point):__ Now use the script above to parallelize the entire training run. You are free to use other fine-tuning methods (e.g. LoRA or even full fine-tuning), as long as you can demonstrate that the loss goes down.

**Make sure you increase SEQUENCE_LENGTH as much as possible!** Even on a single GPU, you should be able to go into thousands, if not tens of thousands of tokens - and report the maximum sequence length with one and with multiple GPUs respectively.

If you don't have access to multiple GPUs, you may optionally submit a version that does training on a single GPU, but computes attention heads sequentially with gradient checkpointing - but if you do, please announce that you are using this option in bold, capital letters, so that the grader will notice it.

```python
<<A whole lot of your code here>>
```

```python
<<... and a dedicated cell to show off that it works>>
```

```

```

```

```

```

```


### Optional: bonus tasks

There are many routes to further improve the training/inference code. You may (but you don't have to) implement any combination of them for bonus points.

However, please not that the total points for this week's entire assignment (part 1 & 2) are **capped at 14**.

__Bonus task: parallel key-value caching (1 point).__ In tasks 2 and 3, you implement tensor parallelism for attention forward pass and perform inference with re-computation. However, real world inference engines use [KV caching](https://huggingface.co/docs/transformers/main/en/kv_cache) - keeping key and value caches from past tokens and only processing the new token each time.

For this task, you will have to implement this type of parallelism for either torch.distributed or DTensor implementation of attention $-$ simply cache the heads already assigned to each rank. To get the grade, you will need to demonstrate that the model generates a sensible text with any cache (via past_key_values=).

__Bonus task: pipeline parallelism (1-2 points):__ In tasks 1-3, you've implemented symmetric model parallelism, aka Tensor Parallelism. However, there is another way to partition model parameters $-$ assign entire layers to each rank and run them in a pipeline. This can be faster, especially if you are running

For 1 point, check out [torch.distributed.pipelinging](https://pytorch.org/docs/stable/distributed.pipelining.html), [DeepSpeed pipelining](https://deepspeed.readthedocs.io/en/latest/pipeline.html) or [torchgpipe](https://github.com/kakaobrain/torchgpipe) and demonstrate that you can run or fine-tune a model that would not fit into a single GPU (you will need multuple devices for this!).

For 2 points, compare different pipelining schedules in terms of training throughput: use GPipe as a baseline and try ScheduleInterleaved1F1B (or a more advanced pipeline of your choosing).

__Bonus task: better sequence parallelism (2 points).__ In tasks 4 and 5, you implemented basic sequence parallelism. However, there are multiple ways you can improve that technique for further memory savings or better device utilization.

For 1 point, implement combined tensor + sequence parallelism and compare results with naive sequence parallelism.

For 2 points, implement [Ring Attention](https://arxiv.org/abs/2310.01889) *or* integrate computation-communication overlap from [FLUX](https://arxiv.org/abs/2406.06858) and measure the speed and memory trade-offs.
