---
title: "Week05 Fsdp — lecture slides"
type: external-resource
status: imported-source
language: original
source_kind: slides
source_commit: e632aa89ca9e6638d52e1b686095e7442faffbb0
---

> [!note] Original course material
> The material below is preserved in its original language from
> [`week05_fsdp/seminar.pdf`](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week05_fsdp/seminar.pdf) in *Efficient Deep Learning Systems* at commit
> `e632aa89ca9e6638d52e1b686095e7442faffbb0`. License: [MIT](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/LICENSE). Bookvar changed only the
> publication markup and link paths.



The embedded PDF is the primary visual version. The page-separated text below is included for search and quotation; it was extracted mechanically and has not been rewritten.

<iframe class="source-pdf" src="https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week05_fsdp/seminar.pdf?raw=1" title="Week05 Fsdp — lecture slides" loading="lazy"></iframe>

## Extracted slide text

### Page 1

```text
FSDP семинар
```

### Page 2

```text
Plan


 Prerequisites: CUDA streams / events, DeviceMesh, DTensor
 FSDP2: interface, options, internals
 PyTorch DCP, efficient garbage collection
```

### Page 3

```text
CUDA streams and events
all_gather_stream = torch.cuda.Stream()

...

# layer 3 unshard
with torch.cuda.stream(all_gather_stream):
    model.layers[3].all_gather()
    all_gather_event_3 = torch.cuda.Event()
    # or all_gather_stream.record_event()

# layer 2 forward
activations = model.layers[2](activations)

# layer 4 unshard
with torch.cuda.stream(all_gather_stream):
    model.layers[4].all_gather()
    all_gather_event_4 = torch.cuda.Event()

# layer 3 forward
torch.cuda.default_stream().wait_event(all_gather_event_3)
activations = model.layers[3](activations)

...
```

### Page 4

```text
CUDA streams and events
```

### Page 5

```text
DeviceMesh
```

### Page 6

```text
DeviceMesh
from torch.distributed.device_mesh import init_device_mesh

mesh_1d = init_device_mesh("cuda", mesh_shape=(8,), mesh_dim_names=("dp",))
mesh_2d = init_device_mesh("cpu", mesh_shape=(2, 8), mesh_dim_names=("dp", "tp"))
mesh_3d = init_device_mesh(
    "cuda",
    mesh_shape=(2, 2, 8),
    mesh_dim_names=("pp", "dp", "tp"),
)

dp_group = mesh_2d.get_group("dp")
dist.all_gather(..., group=dp_group)

mesh_2d.get_local_rank("tp")

mesh = init_device_mesh("cpu", mesh_shape=(16,), mesh_dim_names=("world",))
mesh_dp_cp = mesh._unflatten(0, mesh_sizes=(4, 4), mesh_dim_names("dp", "cp"))
mesh_host = mesh._unflatten(0, mesh_sizes=(2, 8), mesh_dim_names("inter", "intra"))
```

### Page 7

```text
DTensor
class DTensor:
    _local_tensor: torch.Tensor
    _spec:

@dataclass
class DTensorSpec:
    mesh: DeviceMesh
    placements: tuple[Placement, ...] # (Shard(0), Replicate())

   # tensor meta will only be set during sharding propagation
   tensor_meta: TensorMeta | None = None # (dtype, shape, stride)

   def redistribute(
       placements: Sequence[Placement]
   ) -> DTensor | torch.Tensor: ...

   def __torch_dispatch__(): ...
```

### Page 8

```text
DTensor
from torch.distributed.tensor import DTensor, distribute_tensor

mesh = init_device_mesh("cuda", mesh_shape=(8,), mesh_dim_names=("dp",))
big_tensor = torch.randn(1024, 4096)
placements = (Shard(dim=0),)

dtensor = distribute_tensor(
    big_tensor,
    device_mesh=mesh,
    placements=placements,
)
dtensor._local_tensor
dtensor.to_local() # .shape = (512, 4096)


shard = ... # .shape = (512, 4096)
dtensor = DTensor.from_local(
    shard,
    device_mesh=mesh,
    placements=placements,
) # .shape = (1024, 4096)

dtensor.redistribute(placements=(Replicate(),))
dtensor.full_tensor()
```

### Page 9

```text
DTensor
```

### Page 10

```text
DTensor
```

### Page 11

```text
FSDP2
```

### Page 12

```text
FSDP2
from torch.distributed.fsdp import fully_shard   for step in ...:
                                                     for gas_step in ...:
mesh_2d = init_device_mesh(                              is_last_backward = gas_step == num_gas_steps - 1
    "cuda",                                              # ZeRO-2
    mesh_shape=(2, 8),                                   model.set_reshard_after_backward(is_last_backward)
    mesh_dim_names=("dp", "tp"),                         # ZeRO-1
)                                                        model.requires_gradient_sync(is_last_backward)
model = Model()
                                                    loss = loss_fn(model(inputs), targets)
for layer in model.layers:                          ...
    fully_shard(
        module, # (module1, module2)
        mesh=dp_mesh,
        reshard_after_forward=True, # ZeRO-3
        mp_policy=MixedPrecisionPolicy(
            param_dtype=torch.float16,
            reduce_dtype=torch.float32,
        ),
        offload_policy=CPUOffloadPolicy(),
    )

fully_shard(model, ...)
```

### Page 13

```text
FSDP2
```

### Page 14

```text
FSDP2 — hooks
```

### Page 15

```text
FSDP2 — pre-forward
def pre_forward(module, args):
    module.unshard() # in all-gather stream
    module.wait_for_unshard() # sync compute (default) stream with all-gather stream
    module._register_post_backward_hook(args)
    return args

def unshard(module):
    with torch.cuda.stream(all_gather_stream):
        module.all_gather_result = module.all_gather()
    module.all_gather_event = all_gather_stream.record_event()

def wait_for_unshard(module):
    torch.cuda.default_stream().wait_event(module.all_gather_event)
    module.set_unsharded_params(module.all_gather_result)

def fully_shard(module, ...):
    ...
    module.register_forward_pre_hook(pre_forward)
```

### Page 16

```text
FSDP2 — post-forward
def post_forward(module, args, output):
    module.reshard()
    module._record_post_forward()
    module._register_pre_backward_hook(output)
    return output

def reshard(module):
    module.set_sharded_params() # and free unsharded params

def _record_post_forward(module):
    post_forward_index = len(module.comm_ctx.post_forward_order)
    module.comm_ctx.post_forward_order.append(module)
    module._post_forward_indices.append(post_forward_index)

def fully_shard(module, ...):
    ...
    module.register_forward_hook(post_forward)
```

### Page 17

```text
FSDP2 — pre-backward
def pre_backward(module, *unused):
    module.unshard() # no-op if prefetched
    module.wait_for_unshard()
    module._backward_prefetch()

def _backward_prefetch(module):
    curr_index = module._post_forward_indices.pop()
    target_index = curr_index - 1
    target_module = self.comm_ctx.post_forward_order[target_index]
    target_module.unshard()

def _register_pre_backward_hook(self, output):
    for t in output:
        if torch.is_tensor(t) and t.requires_grad:
            t.register_hook(self._pre_backward)
    return output
```

### Page 18

```text
FSDP2 — post-backward
def post_backward(module, *unused: Any):
    if module.reshard_after_backward:
        module.reshard()
    if module.reduce_grads:
        reduce_scatter_stream.wait_stream(torch.cuda.default_stream())
        with torch.cuda.stream(reduce_scatter_stream):
            module.reduce_scatter_grads()
        reduce_event = reduce_scatter_stream.record_event()

def _register_post_backward_hook(module, args):
    RegisterPostBackwardFunction.apply(self, *args)

class RegisterPostBackwardFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, module, *inputs):
        ctx.module = module
        return inputs

   @staticmethod
   def backward(ctx, *grads):
       module.post_backward()
       return (None,) + grads
```

### Page 19

```text
FSDP2 — memory
```

### Page 20

```text
FSDP2 — memory
```

### Page 21

```text
Computation / communication overlap
 Implicit prefetching
    в pre_forward
 Explicit prefetching

    в pre_backward

    можно задать руками

     module.set_modules_to_forward_prefetch(modules)
     module.set_modules_to_backward_prefetch(modules)
```

### Page 22

```text
Подробнее про работу со стримами
```

### Page 23

```text
Подробнее про работу со стримами — forward
```

### Page 24

```text
Подробнее про работу со стримами — backward
```

### Page 25

```text
ZeRO-2
```

### Page 26

```text
ZeRO-2
```

### Page 27

```text
ZeRO-1
```

### Page 28

```text
ZeRO-1
```

### Page 29

```text
HSDP
mesh_2d = init_device_mesh(
    "cpu",
    mesh_shape=(2, 8),
    mesh_dim_names=("dp_replicate", "dp_shard"),
)

fully_shard(
    module,
    mesh=mesh_2d,
    ...
)



  логика становится заметно сложнее, показывать не буду(
```

### Page 30

```text
CPU offloading
  ZeRO-Offload

with torch.device("cpu"):
    model = Model()

fully_shard(
    module,
    ...
    offload_policy=CPUOffloadPolicy(),
)

def unshard(module):
    sharded_param = sharded_param.to(
        device,
        non_blocking=True,
    )
    ...
    module.all_gather()

def post_backward(module):
    new_sharded_grad = new_sharded_grad.to(
        torch.device("cpu"),
        non_blocking=True
    )
```

### Page 31

```text
CPU offloading
```

### Page 32

```text
hpZ
  ZeRO++

mesh = init_device_mesh(
    "cuda",
    mesh_shape=(16,),
    mesh_dim_names=("dp",),
)
fully_shard(
    module,
    mesh,
    ...
    reshard_after_forward=8,
)
```

### Page 33

```text
hpZ
```

### Page 34

```text
hpZ
```

### Page 35

```text
PyTorch DCP
 два вида state_dict
    SHARDED_STATE_DICT
    FULL_STATE_DICT
 в FSDP2 всегда sharded, но состоит из DTensor-ов
   с помощью .redistribute() можно менять шардирование чекпоинта
 DCP умеет эффективно отгружать чекпоинты с минимальным оверхедом
```

### Page 36

```text
PyTorch DCP
import torch.distributed.checkpoint as dcp
model = Model()
fully_shard(model)
optimizer = Optimizer(model.parameters())

state_dict = {
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict()
}
dcp.state_dict_saver.save(state_dict)
dcp.state_dict_loader.load(state_dict)



  truthfully it’s a bit more complicated
```

### Page 37

```text
PyTorch DCP
```

### Page 38

```text
PyTorch DCP
```

### Page 39

```text
PyTorch DCP
```

### Page 40

```text
PyTorch DCP
```

### Page 41

```text
Garbage collection tuning
gc.disable()
gc.collect(1)

... init

for step in ...:
    if step > 1 and step % _gc_freq == 0:
        gc.collect(1)

   ... step
```

### Page 42

```text
Extras
 SimpleFSDP
 unshard_in_backward
 meta device init
 compile
```
