---
title: "PyTorch Distributed and data-parallel training"
type: external-resource
status: imported-source
language: original
source_kind: notebook
source_commit: e632aa89ca9e6638d52e1b686095e7442faffbb0
---

> [!note] Original course material
> The material below is preserved in its original language from
> [`week03_data_parallel/practice.ipynb`](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week03_data_parallel/practice.ipynb) in *Efficient Deep Learning Systems* at commit
> `e632aa89ca9e6638d52e1b686095e7442faffbb0`. License: [MIT](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/LICENSE). Bookvar changed only the
> publication markup and link paths.



In this notebook, we'll be overview the distributed part of the PyTorch library. We will also see a couple of examples of distributed training using available wrappers.

To find out about the communication pattern of your GPUs, you can use the following command:

```python
!nvidia-smi topo -m
```

Let's import all required libraries and define a function which will create the process group. There are [three](https://pytorch.org/docs/stable/distributed.html#backends-that-come-with-pytorch) communication backends in PyTorch: as a simple rule, use GLOO for CPU communication and NCCL for communication between NVIDIA GPUs.

```python
import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process
import random

def init_process(rank, size, fn, master_port, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(master_port)
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)
    
torch.set_num_threads(1)
```

First, we'll run a very simple function with torch.distributed.barrier. The cell below prints in the first process and then prints in all other processes.

```python
def run(rank, size):
    """ Distributed function to be implemented later. """
    if rank!=0:
        dist.barrier()
    print(f'Started {rank}',flush=True)
    if rank==0:
        dist.barrier()

if __name__ == "__main__":
    size = 4
    processes = []
    port = random.randint(25000, 30000)
    for rank in range(size):
        p = Process(target=init_process, args=(rank, size, run, port))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
```

Let's implement a classical ping-pong application with this paradigm. We have two processes, and the goal is to have P1 output 'ping' and P2 output 'pong' without any race conditions.

```python
def run_pingpong(rank, size, num_iter=10):
    """ Distributed function to be implemented later. """
    
    
    for _ in range(num_iter):
        pass


if __name__ == "__main__":
    size = 2
    processes = []
    port = random.randint(25000, 30000)
    for rank in range(size):
        p = Process(target=init_process, args=(rank, size, run_pingpong, port))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
```

# Point-to-point communication
The functions below show that it's possible to send data from one process to another with `torch.distributed.send/torch.distributed.recv`:

```python
"""Blocking point-to-point communication."""

def run_sendrecv(rank, size):
    tensor = torch.zeros(1)+int(rank==0)
    print('Rank ', rank, ' has data ', tensor[0], flush=True)
    if rank == 0:
        # Send the tensor to process 1
        dist.send(tensor=tensor, dst=1)
    else:
        # Receive tensor from process 0
        dist.recv(tensor=tensor, src=0)
    print('Rank ', rank, ' has data ', tensor[0], flush=True)

if __name__ == "__main__":
    size = 2
    processes = []
    port = random.randint(25000, 30000)
    for rank in range(size):
        p = Process(target=init_process, args=(rank, size, run_sendrecv, port))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
```

Also, these functions have an immediate (asynchronous) version:

```python
"""Non-blocking point-to-point communication."""
import time

def run_isendrecv(rank, size):
    tensor = torch.zeros(1)
    req = None
    if rank == 0:
        tensor += 1
        # Send the tensor to process 1
        req = dist.isend(tensor=tensor, dst=1)
        print('Rank 0 started sending')
    else:
        # Receive tensor from process 0
        req = dist.irecv(tensor=tensor, src=0)
        print('Rank 1 started receiving')
        
    print('Rank ', rank, ' has data ', tensor[0])
    req.wait()
    print('Rank ', rank, ' has data ', tensor[0])
    
if __name__ == "__main__":
    size = 2
    processes = []
    port = random.randint(25000, 30000)
    for rank in range(size):
        p = Process(target=init_process, args=(rank, size, run_isendrecv, port))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
```

Adding an artificial delay shows that the communication is asynchronous:

```python
import time

def run_isendrecv(rank, size):
    tensor = torch.zeros(1)
    req = None
    if rank == 0:
        tensor += 1
        # Send the tensor to process 1
        time.sleep(5)
        req = dist.isend(tensor=tensor, dst=1)
        print('Rank 0 started sending')
    else:
        # Receive tensor from process 0
        req = dist.irecv(tensor=tensor, src=0)
        print('Rank 1 started receiving')
        
    print('Rank ', rank, ' has data ', tensor[0])
    req.wait()
    print('Rank ', rank, ' has data ', tensor[0])
    
if __name__ == "__main__":
    size = 2
    processes = []
    port = random.randint(25000, 30000)
    for rank in range(size):
        p = Process(target=init_process, args=(rank, size, run_isendrecv, port))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
```

# Collective communication and All-Reduce
Now, let's run a simple All-Reduce example which computes the sum across all workers. We'll be running the code with the `!python` magic to avoid issues caused by interaction of Jupyter and multiprocessing.

```python
%%writefile run_allreduce.py
#!/usr/bin/env python
import os
from functools import partial

import torch
import torch.distributed as dist
from torch.multiprocessing import Process

def run_allreduce(rank, size):
    tensor = torch.ones(1)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print('Rank ', rank, ' has data ', tensor[0])
    
def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)
    
if __name__ == "__main__":
    size = 10
    processes = []
    for rank in range(size):
        p = Process(target=init_process, args=(rank, size, run_allreduce))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
```

```python
!python run_allreduce.py
```

The same thing can be done with a simpler [torch.multiprocessing.spawn](https://pytorch.org/docs/stable/multiprocessing.html#torch.multiprocessing.spawn) wrapper:

```python
%%writefile run_allreduce_spawn.py
#!/usr/bin/env python
import os
from functools import partial
import torch
import torch.distributed as dist

def run_allreduce(rank, size):
    tensor = torch.ones(1)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print('Rank ', rank, ' has data ', tensor[0])
    
def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)
    
if __name__ == "__main__":
    size = 10

    fn = partial(init_process, size=size, fn=run_allreduce, backend='gloo')
    torch.multiprocessing.spawn(fn, nprocs=size)
```

```python
!./run_allreduce_spawn.py
```

Let's write our own Butterfly All-Reduce. First, we start with creating 5 random vectors and getting the "true" average, just for comparison:

```python
size = 5
tensors = []

for i in range(size):
    torch.manual_seed(i)
    cur_tensor = torch.randn((size,), dtype=torch.float)
    print(cur_tensor)
    tensors.append(cur_tensor)
    
print("result", torch.stack(tensors).mean(0))
```

Now, let's create a custom implementation below:

```python
%%writefile custom_allreduce.py
import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process
import random

def init_process(rank, size, fn, master_port, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(master_port)
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)

def butterfly_allreduce(send, rank, size):
    """
    Performs Butterfly All-Reduce over the process group.
    Args:
        send: torch.Tensor to be averaged with other processes.
        rank: Current process rank (in a range from 0 to size)
        size: Number of workers
    """
    
    buffer_for_chunk = torch.empty((size,), dtype=torch.float)
    
    send_futures = []
    
    for i, elem in enumerate(send):
        if i!=rank:
            send_futures.append(dist.isend(elem, i))
            
    recv_futures = []
    
    for i, elem in enumerate(buffer_for_chunk):
        if i!=rank:
            recv_futures.append(dist.irecv(elem, i))
        else:
            elem.copy_(send[i])
            
    for future in recv_futures:
        future.wait()
        
    # compute the average
    torch.mean(buffer_for_chunk, dim=0, out=send[rank])
    
    for i in range(size):
        if i!=rank:
            send_futures.append(dist.isend(send[rank], i))
            
    recv_futures = []
    
    for i, elem in enumerate(send):
        if i!=rank:
            recv_futures.append(dist.irecv(elem, i))
    
    for future in recv_futures:
        future.wait()
    for future in send_futures:
        future.wait()
            

def run_allreduce(rank, size):
    """ Simple point-to-point communication. """
    torch.manual_seed(rank)
    tensor = torch.randn((size,), dtype=torch.float)
    print('Rank ', rank, ' has data ', tensor)
    butterfly_allreduce(tensor, rank, size)
    print('Rank ', rank, ' has data ', tensor)
    
if __name__ == "__main__":
    size = 5
    processes = []
    port = random.randint(25000, 30000)
    for rank in range(size):
        p = Process(target=init_process, args=(rank, size, run_allreduce, port))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
```

```python
!python custom_allreduce.py
```

# Distributed training

Now that we have this simple implementation of AllReduce, we can run multi-process distributed training. For now, let's use the model and the dataset from the official MNIST [example](https://github.com/pytorch/examples/blob/master/mnist/main.py), as well as the [torchrun](https://pytorch.org/docs/stable/elastic/run.html?highlight=torchrun) command used to manage processes:

```python
%%writefile custom_allreduce_training.py
import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import MNIST


def init_process(local_rank, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    dist.init_process_group(backend, rank=local_rank)
    size = dist.get_world_size()
    fn(local_rank, size)


torch.set_num_threads(1)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(4608, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        output = self.fc2(x)
        return output


def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size


def run_training(rank, size):
    torch.manual_seed(1234)
    dataset = MNIST('./mnist', download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]))
    loader = DataLoader(dataset, sampler=DistributedSampler(dataset, size, rank), batch_size=16)
    model = Net()
    device = torch.device('cpu')
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=0.01, momentum=0.5)

    num_batches = len(loader)
    steps = 0
    epoch_loss = 0
    for data, target in loader:
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = torch.nn.functional.cross_entropy(output, target)
        epoch_loss += loss.item()
        loss.backward()
        average_gradients(model)
        optimizer.step()
        steps += 1
        if True:
            print(f'Rank {dist.get_rank()}, loss: {epoch_loss / num_batches}')
            epoch_loss = 0


if __name__ == "__main__":
    local_rank = int(os.environ["LOCAL_RANK"])
    init_process(local_rank, fn=run_training, backend='gloo')
```

```python
!torchrun --nproc_per_node 2 custom_allreduce_training.py
```

Now let's use the standard [DistributedDataParallel](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) wrapper (which you should probably use in real-world training anyway):

```python
%%writefile ddp_example.py
import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import MNIST


def init_process(local_rank, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    dist.init_process_group(backend, rank=local_rank)
    size = dist.get_world_size()
    fn(local_rank, size)


torch.set_num_threads(1)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(4608, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        output = self.fc2(x)
        return output

def run_training(rank, size):
    torch.manual_seed(1234)
    dataset = MNIST('./mnist', download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]))
    loader = DataLoader(dataset,
                        sampler=DistributedSampler(dataset, size, rank),
                        batch_size=16)
    model = Net()
    device = torch.device('cuda', rank)
    model.to(device)
    
    model = DistributedDataParallel(model, device_ids=[rank], output_device=rank)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    num_batches = len(loader)
    steps = 0
    epoch_loss = 0
    for data, target in loader:
        data = data.to(device)
        target = target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = torch.nn.functional.cross_entropy(output, target)
        epoch_loss += loss.item()
        loss.backward()
        
        optimizer.step()
        steps += 1
        if True:
            print(f'Rank {dist.get_rank()}, loss: {epoch_loss / num_batches}')
            epoch_loss = 0


if __name__ == "__main__":
    local_rank = int(os.environ["LOCAL_RANK"])
    init_process(local_rank, fn=run_training, backend='gloo')
```

```python
!torchrun --nproc_per_node 2 ddp_example.py
```

That's it for today! For the homework this week, see the `homework` folder.

```python

```
