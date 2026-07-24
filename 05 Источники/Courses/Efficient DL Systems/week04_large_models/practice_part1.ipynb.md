---
title: "to use:"
type: external-resource
status: imported-source
language: original
source_kind: notebook
source_commit: e632aa89ca9e6638d52e1b686095e7442faffbb0
---

> [!note] Original course material
> The material below is preserved in its original language from
> [`week04_large_models/practice_part1.ipynb`](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week04_large_models/practice_part1.ipynb) in *Efficient Deep Learning Systems* at commit
> `e632aa89ca9e6638d52e1b686095e7442faffbb0`. License: [MIT](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/LICENSE). Bookvar changed only the
> publication markup and link paths.



```python
# dependencies: a recent pytorch and %pip install transformers==4.46.3 datasets==2.20.0 accelerate==0.32.1
# the assignment may still work on other recent versions (transformers>4.25 and datasets>2.9), but no promises
```

### Part 1: Memory-efficient training and inference

__Your quest__ is to fine-tune a large language with restricted GPU memory. You can choose one of these two models:

- colab, kaggle or datasphere: choose either [facebook/opt-6.7b](https://huggingface.co/facebook/opt-6.7b), [Qwen/Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B) or Llama-3-8B ([official](https://huggingface.co/meta-llama/Llama-3.1-8B), [unsloth](https://huggingface.co/unsloth/Llama-3.1-8B))
- if you have >64GB disk space: [facebook/opt-iml-30b](https://huggingface.co/facebook/opt-iml-max-30b) or [Qwen/Qwen-32B](https://huggingface.co/Qwen/Qwen2.5-32B)

Both are powerful language models: opt-6.7b is a relatively old open-access GPT3 equivalent and Llama-3 / Qwen 2.5 are state-of-the-art LMs

You can use __up to 10GiB GPU memory__ (as in 3080 or 2080Ti) for 6.7B model and up to 48GB for the 30B one. We deliberately limit GPU memory below and recommend you to check the peak memory usage via: [`torch.cuda.max_memory_allocated()`](https://pytorch.org/docs/stable/generated/torch.cuda.max_memory_allocated.html). We shall also assume that you don't have enough RAM to load the full model on CPU. If your your machine has enough, you may take advantage of it.


Your code should be able to do 3 things:
* run forward pass on a sequence of 2048 tokens
* compute gradients w.r.t. a small subset of parameters: only one layer or similar
* generate an answer to a question using `model.generate` (see below)


Model compression alone will not count for full grade! Please either use a 16-bit model (6-8B) or, if you feel like you want to prune/quantize the model, use the 30B+ version.

```python
import torch
# if your GPU has less than 10GB memory, please remove the code below
# if your GPU has less than 4GB memory, use colab or kaggle instead
max_memory_gib = torch.cuda.get_device_properties('cuda').total_memory / 2 ** 30
torch.cuda.set_per_process_memory_fraction(min(1.0, 10 / max_memory_gib))
print(f"Setting memory limit to {min(1.0, 11 / max_memory_gib) * 100:.2f}%")
```


```text
Setting memory limit to 74.62%
```


For now, we're gonna load a smaller version of the model to show you around.

The large models use the same code, but with more layers & hidden units - so you can debug your code on the smaller model, then switch to the real deal.

```python
import transformers
model_name = "facebook/opt-iml-1.3b"   # full model: 'facebook/opt-6.7b' or see above
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_name, low_cpu_mem_usage=True, torch_dtype=torch.float16).cuda()

model.enable_input_require_grads()  # for gradient checkpointing compatibility, see FAQ
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
tokenizer_config.json:   0%|          | 0.00/682 [00:00<?, ?B/s]
```

```text
config.json:   0%|          | 0.00/596 [00:00<?, ?B/s]
```

```text
vocab.json:   0%|          | 0.00/899k [00:00<?, ?B/s]
```

```text
merges.txt:   0%|          | 0.00/456k [00:00<?, ?B/s]
```

```text
special_tokens_map.json:   0%|          | 0.00/221 [00:00<?, ?B/s]
```

```text
pytorch_model.bin:   0%|          | 0.00/2.63G [00:00<?, ?B/s]
```

### Inference baseline

Here's a simple code that generates some tokens without offloading. You can use this as a reference, to check that your offloading algorithm is correct. Naturally, it will not work on the full 6.7B (or 30B) model.

```python
# here's how the model works: tokenizer converts raw data to pytorch tensors
batch = tokenizer(["A cat sat", "import numpy"], return_tensors='pt')
batch = {name: tensor.cuda() for name, tensor in batch.items()}
print("Batch:", repr(batch)[:70].replace('\n', ' '), ' ...')
```


```text
Batch: {'input_ids': tensor([​[    2,   250,  4758,  4005],         [    2, 41  ...
```


```python
# fun fact: you can use the model to generate text given prefix
generated_ids = model.generate(**batch, max_length=32)
print("Sample A:", tokenizer.decode(generated_ids[0]))
print("Sample B:", tokenizer.decode(generated_ids[1]))
```


```text
Sample A: </s>A cat sat on my lap and I was watching a movie. I was about to fall asleep and the cat jumped up and started licking my face. I
Sample B: </s>import numpy as np

import numpy as np

import numpy as np

import numpy as np

import numpy
```


### Training baseline

Here's some sample data you can use for prototyping -- and demonstrating that your algorithm works.
Then again, you are free to use any dataset you like.

We also provide a very simple fine-tuning example that mimics [BitFit](https://arxiv.org/abs/2106.10199).

```python
from datasets import load_dataset

data = load_dataset("wikitext", "wikitext-2-v1")['train']
tokenizer.pad_token = tokenizer.eos_token

sample_batch = tokenizer(data['text'][:1], max_length=5, padding=True, pad_to_multiple_of=5, return_tensors='pt')

# note: sample_batch has a size of 1x5, you will need a larger batch in the next assignment
# note(2) if you want something more peculiar, https://huggingface.co/datasets/transformersbook/codeparrot
```

```text
Downloading readme:   0%|          | 0.00/10.5k [00:00<?, ?B/s]
```

```text
Downloading data:   0%|          | 0.00/685k [00:00<?, ?B/s]
```

```text
Downloading data:   0%|          | 0.00/6.07M [00:00<?, ?B/s]
```

```text
Downloading data:   0%|          | 0.00/618k [00:00<?, ?B/s]
```

```text
Generating test split:   0%|          | 0/4358 [00:00<?, ? examples/s]
```

```text
Generating train split:   0%|          | 0/36718 [00:00<?, ? examples/s]
```

```text
Generating validation split:   0%|          | 0/3760 [00:00<?, ? examples/s]
```


```text
/usr/local/lib/python3.11/dist-packages/transformers/tokenization_utils_base.py:2852: UserWarning: `max_length` is ignored when `padding`=`True` and there is no truncation strategy. To pad to max length, use `padding='max_length'`.
  warnings.warn(
```


```python
# example: only train bias parameters, as in the BitFitPaper
for name, param in model.named_parameters():
    param.requires_grad = name.endswith("bias")
    if param.requires_grad:
        param.data = param.data.to(torch.float32)
print(f"Total parameters: {sum(p.numel() for p in model.parameters())/1e6:0.2f} million")
print(f"Trained parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:0.2f} million")


opt = torch.optim.Adam(model.parameters(), lr=1e-4)
# model turns those tensors into logits (pre-softmax activations) and loss
# in the example below, logits are available as pred.logits, and loss is pred.loss

for i in range(10):
    sample_batch = {name: tensor.cuda() for name, tensor in sample_batch.items()}
    with torch.cuda.amp.autocast():
        loss = model(**sample_batch, labels=sample_batch['input_ids']).loss / 1000
    loss.backward()
    opt.step()
    print(f"Loss[{i}] = {loss.item():.3f}")

# if all went well, you'll see the loss go down
```


```text
<ipython-input-8-e2c2471f698e>:16: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast():
```



```text
Total parameters: 1315.76 million
Trained parameters: 0.54 million
Loss[0] = 0.011
Loss[1] = 0.010
Loss[2] = 0.009
Loss[3] = 0.008
Loss[4] = 0.007
Loss[5] = 0.007
Loss[6] = 0.006
Loss[7] = 0.005
Loss[8] = 0.005
Loss[9] = 0.004
```


If it looked a bit too easy - that was because you are dealing with a small model that fits into RAM. Once you have something larger, you can no longer simply `.from_pretrained` your model. Instead, you will need to process weights in small groups - the way they are stored in Hugging Face hub.

## Assignment details


Your main objective is to implement parameter offloading and solve two problems: fine-tuning and inference.

__Task 1.1:__ run forward and backward pass, accumulate gradients w.r.t. a subset of model parameters. Use a training batch size of 128 sequences, and sequence length of 1024 tokens. In other words, `input_ids.shape == (128, 1024)`.

You may choose one of these options:
- train only the embedding layer, [similar to this paper](https://arxiv.org/abs/2104.08691)
- train low-rank adapters (LoRA), [like in this paper](https://arxiv.org/abs/2106.09685)
- use [Hugging Face PEFT](https://github.com/huggingface/peft/)

You don't have to train the model to convergence, just show that it can run 10 consecutive forward-backward-step passes and **the loss goes down**. You can even run those forward/backward passes on the same batch!


Please do not use native offloading / quantization libraries for this assignment - you are to implement your own.


__Task 1.2:__ generate a short sequence given a prefix. You may choose any generation task that requires generating at least 25 consecutive tokens. Here's one example from the NLP course (the generated code is in blue)

![img](https://i.imgur.com/a1QhKF7.png)

You may use model.generate (if your code is compatible with that) or write your own inference loop. If you choose to write your own loop, you are free to use sampling, greedy, top-p, top-k or any other [inference mode supported by HF transformers](https://huggingface.co/docs/transformers/main_classes/text_generation).


__Grading (5 points):__

- __+1 point__ you can perform forward pass with offloading on *some* input sequence (any batch size / length)
- __+1 point__ check that forward pass with offloading is `torch.allclose` to forward pass without offloading
    - since you (likely) can't run the full model w/o offloading, test it the 1.3B model from earlier
- __+1 point__ you can perform forward pass on 128x1024 tokens of actual text data (e.g. the sample data above)
- __+1 point__ you can compute gradients with offloading on the same 128x1024 tokens from the real text data
- __+1 point__ you can inference the model - and it generates some human-readable text
- __bonus points:__ we offer two optional assignments:
   - **Selective activation checkpointing (2pt):** there is a gentler version of gradient checkpointing where you don't just remember the layer inputs, but also some activations that are easier to compute - compared to their size. For instance, MLP linear layers are compute-heavy, but the nonlinearity is relatively compute-light for the same amount of memory. You can re-compute only the compute-light operations and keep the compute-heavy ones in memory. There's [a paper](https://arxiv.org/pdf/2205.05198) that describes such an approach in detail (see 'Selective activation checkpointing').
   - **Prefetch offloaded layers (2pt):** optimize your code so that it begins pre-loading the next offloaded layer in the background, while computing the current layer. It can be done with a copy with non_blocking=True, or, for fine-grained control, CUDA streams. To get the full grade for this assignment, please demonstrate that your approach is faster than naive offloading, at least during large batch forward/backward pass. This can be done using a profiler.
   - Please note that the maximum points for this week are **capped at 14**.

__Conditions:__
- using more than 10GiB of GPU memory at any point is forbidden (check with [`torch.cuda.max_memory_allocated()`](https://pytorch.org/docs/stable/generated/torch.cuda.max_memory_allocated.html))
- please keep all model parameters in either float16, bfloat16, or float32 - no quantization for now
   - if you *really* want to show off quantization, evaluate your code with both original and quantized weights
- at least 99% of model's floating point computations should be done on GPU. If you find a server with a ton of RAM and run the model on cpu, it will not count as a solution
- please do **not** use any thrid-party offloading implementations (e.g. from deepspeed or accelerate)
- your solution may be slow - especially when loading from colab disks. This is not your fault :)
   - if you found a way to speed up the code in a non-trivial way (e.g. load i+1st layer in parallel when computing i-th), please attach a short summary of what when submitting the notebook (e.g. anytask/lms) to get bonus points



__FAQ:__

- __My training outputs have .requires_grad == False!__ This may be a side-effect of using gradient checkpointing if all your trainable parameters are inside the checkpoints (e.g. with LoRA). To circumvent this, either set `model.enable_input_require_grads()` or manually ensure that input tensors to each checkpoint have requires_grad=True.

- __I am getting out-of-memory errors for no reason!__
  - it could be because of some leftover tensors from previous cells. To get rid of them, please restart the notebook and only run the code that is relevant to your current task.

- __The forward pass activations are too large, it does not fit!__
   - __Gradient accumulation:__ you probably can't process 128 sequences at once -- but what if you accumulate them over several forward/batckward passes with a smaller batch size.
   - __Gradient checkpointing:__ you can further reduce activation memory by not storing intermediate activations. You can learn how to usa built-in checkpoints [from their docs](https://huggingface.co/docs/transformers/main_classes/model) or build your own using [PyTorch default checkpointing](https://pytorch.org/docs/stable/checkpoint.html).
  
- __My float16 gradients are NaN!__
   - There should be a way to scale your loss function by a constant -- only to un-scale it later. You can use GradScaler from [PyTorch AMP](https://pytorch.org/docs/stable/amp.html) or write your own monstrosity.
   - You can also cast weights to bfloat16 _but you have to demonstrate that bfloat16 model generates the same (or close) output as float16 one!_ As in "you have to write a short report with code and samples."
     
- __I can run forward with no_grad, but running with grad goes out of memory!__
   - If the problem only occurs with large batches, please see "activations are too large" above.
   - If you get OOM errors even with a single training token (a 1x1 batch), but only in training mode,
     maybe you forgot to mark most parameters as `requires_grad=False`? The .grad buffers can be quite large.
     
   - If not, OOM  be because PyTorch autograd remembers the intermediate weight tensors for backprop.
     For example, consider this code:
     
```python
    x = embeddings_and_input_layernorm(input_ids)
    for layer_index in range(num_layers):
        layer = load_from_disk(layer_index)
        x = layer(x)
        del layer  # we no longer need this layer's weights, but PyTorch will keep it in memory for autograd!
```

    If this is your case, you can write an autograd function that loads the necessary weight.
    a look at "[Optional] Suggested Interface" section below.
     
- __I cannot load the full model even in CPU RAM!__
   - This is intended - and a real problem that you often face in production.
     You gotta find a way to prepare your model for offloading without loading the full thing into RAM.
     In the next section, we explain how you can handle checkpoints and initialize the model in google colab.
     Please see the [Optional] sections that mention low RAM.

<details>
    <summary> <h3> <u> [Optional] Suggested Interface with torch.autograd.Function (click to expand) </u> </h3> </summary>

You can assume that offloaded weights do not require grad themselves - but they take part in intermediate computations that *do* require grad.
The problem is, if you load weights naively without `torch.no_grad`, PyTorch will remember them until the end of backward pass. If not addressed, this will keep all model weights in memory and mess up your offloading.


To avoid this, you can implement a custom autograd function that loads weights from ram / disk internally. That way, PyTorch will not keep any gpu tensors except unless you explicitly tell it to. Crucially, __we only need this function for linear layers__ since all other layers can fit on GPU. Though, you may *optionally* offload embedding layers as well.


Here's [some documentation](https://pytorch.org/docs/stable/notes/extending.html#extending-torch-autograd) on writing your own autograd functions. Your solution could look something like this:


```python
class _OffloadedLinearOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, saved_weight_path, bias_or_none):
        weight = you.load_by_name(saved_weight_path)
        ctx._saved_weight_path = saved_weight_path
        ctx._has_bias = bias_or_none is not None
        return torch.nn.functional.linear(input, weight, bias=bias_or_none)

    @staticmethod
    def backward(ctx, grad_output):
        weight = you.load_by_name(ctx._saved_weight_path)
        grad_input = torch.nn.functional.linear(grad_output, weight.t())
        grad_bias = grad_output.flatten(0, -2).sum(0) if ctx._has_bias else None
        return grad_input, None, grad_bias

    
# to use:
# output = _OffloadedLinearOp.apply(input, "my_weight.pth", bias)
# loss(output).backward()  # uses custom backward
```

You can implement this function separately and test it on a single layer to make sure forward and backward passes match. Once you are confident in your code, it's time to apply it to your model. One way to do this is:


```python
class MyOffloadedLinear(torch.nn.Module):
    def __init__(self, saved_weight_path, bias_or_none):
        super().__init__()
        self.saved_weight_path, self.bias_or_none = saved_weight_path, bias_or_none
    def forward(self, input):
        return _OffloadedLinearOp.apply(input, self.saved_weight_path, self.bias_or_none)

for module_that_contains_linear in you.find_these_modules(model):
    linear = you.take_linear_layer_from(module_that_contains_linear)
    saved_weight_path = save_weight_somewhere(linear.weight)
    offloaded_linear = MyOffloadedLinear(saved_weight_path, linear.bias)
    you.replace_that_linear_with(offloaded_linear)
```
    
Please note that this algorithm is "lazy" in the sense that it loads weights just in time. A smarter (and faster!) way to offload the data is to do it in parallel: once you load the first weight, you immediately start loading the second weight from disk in a background thread. You can do this by recording the order in which your model uses the offloaded weights and keeping track of which weight you should load next.

</details>

<details>
    <summary><h3><u>[Optional] How to initialize the model with low RAM (click to expand)</u></h3></summary>
    
    The trick is that you don't initialize all modules at once.
    Instead, you can load *some* modules, prepare them for offloading (e.g. remove some params), then load the next bunch of modules.
    
    Here's one way you can do this:

    ```python
    config = transformers.AutoConfig.from_pretrained("facebook/opt-6.7b")
    actual_hidden_layers = config.num_hidden_layers
    config.num_hidden_layers = 0  # create a model with no hidden layers
    model = transformers.AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16)
    print(f"Total parameters (embeddings only): {sum(p.numel() for p in model.parameters())/1e6:0.2f} million")
    # only 0.21 billion instead of 6.7

    for _ in range(actual_hidden_layers):
        new_layer = transformers.models.opt.modeling_opt.OPTDecoderLayer(config)
        new_layer = you.prepare_for_offloading(new_layer)
        model.model.decoder.layers.append(new_layer)
    config.num_hidden_layers = actual_hidden_layers

    you.load_parameters_that_werent_offloaded(model, preprocessed_checkpoint_chunks)
    ```
    
    If `you.prepare_for_offloading` properly offloads all heavy parameters to the disk, this code will build the full offloaded model without going over 10GB CPU RAM.
    We also recommend that you check that the resulting code works correctly by test-running it on the 1.3B model.

</details>


<details>
    <summary><h3><u>[Optional] Dealing with HuggingFace weights with low RAM (click to expand)</u></h3></summary>


When you download a Hugging Face model, there will be one or more "chunks", holding the data parameters. These chunks can be seen in the model repository, under "Files and versions" tab:
![image.png](https://i.imgur.com/3gZ2KPB.png)

Reference links to "files and versions": [opt-6.7b](https://huggingface.co/facebook/opt-6.7b/tree/main), [opt-iml-30b](https://huggingface.co/facebook/opt-iml-30b/tree/main)

You can download individual chunks of parameters by going clicking on a chunk and copying the "download" url, like this:

![img](https://i.imgur.com/cv9WvYw.png)

Any chunks downloaded this way will contain a `torch.load`-able state dict. Here's how it works:

```python
# example: download one (small) chunk out of OPT-IML-30B
chunk7_download_url = "https://huggingface.co/facebook/opt-iml-30b/resolve/828fabfb08d5d3f81b4d33cd27a64e3a360a5770/pytorch_model-00007-of-00007.bin"
!wget {chunk7_download_url} -O "chunk7.pth"

partial_state_dict = torch.load("chunk7.pth")
print(f"Keys:", partial_state_dict.keys(), '\n')
print(f"Shape of decoder.layers.47.fc1.weight: {partial_state_dict['decoder.layers.47.fc1.weight'].shape}")
# Keys: dict_keys(['decoder.layers.47.fc1.weight', 'decoder.layers.47.fc1.bias', 'decoder.layers.47.fc2.weight', 'decoder.layers.47.fc2.bias', 'decoder.layers.47.final_layer_norm.weight', 'decoder.layers.47.final_layer_norm.bias'])
# Shape of decoder.layers.47.fc1.weight: torch.Size([28672, 7168])
```

</details>

```python
# if it helps, <YOUR CODE HERE>
```
