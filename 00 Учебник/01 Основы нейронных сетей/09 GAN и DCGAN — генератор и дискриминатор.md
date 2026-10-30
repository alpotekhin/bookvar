---
title: GAN и DCGAN — генератор и дискриминатор
type: textbook-chapter
status: canonical
source_language: en
last_updated: 2026-10-28
last_verified: 2026-10-28
primary_sources:
  - https://d2l.ai/chapter_generative-adversarial-networks/gan.html
  - https://d2l.ai/chapter_generative-adversarial-networks/dcgan.html
  - https://arxiv.org/abs/1406.2661
  - https://arxiv.org/abs/1511.06434
---

# 9. GAN и DCGAN — генератор и дискриминатор

A decoder can turn a random vector into an image without providing a tractable
likelihood for that image. A generative adversarial network trains such a
generator using a second network that distinguishes generated examples from
training data. There is no input image to reconstruct and no paired target
image for each noise vector.

## A generator and a learned discriminator

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/gan-dcgan-2026-10-28/gan.svg|D2L original GAN architecture: noise, generator, generated and real inputs, discriminator]]

*Original illustration from Aston Zhang, Zachary C. Lipton, Mu Li and Alexander J. Smola,
[Dive into Deep Learning, GAN](https://d2l.ai/chapter_generative-adversarial-networks/gan.html#fig-gan),
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
Complete SVG, unchanged.*

The two routes into the discriminator use the same parameters. Real examples
come from the dataset; generated examples come from independent noise passed
through the generator. The figure shows a forward computation. During training,
a derivative through the discriminator also tells the generator how changing
its output would change the classification loss.

The discriminator is a binary classifier to distinguish if the input $x$ is real (from real data) or fake (from the generator). Typically, the discriminator outputs a scalar prediction $o\in\mathbb R$ for input $\mathbf x$, such as using a fully connected layer with hidden size 1, and then applies sigmoid function to obtain the predicted probability $D(\mathbf x) = 1/(1+e^{-o})$. Assume the label $y$ for the true data is $1$ and $0$ for the fake data. For one labelled input, the binary cross-entropy is

$$
\begin{aligned}
\ell_D(x,y)&=-y\log D(x)\\
&\quad-(1-y)\log(1-D(x)).
\end{aligned}
$$

For the generator, it first draws some parameter $\mathbf z\in\mathbb R^d$ from a source of randomness, *e.g.*, a normal distribution $\mathbf z \sim \mathcal{N} (0, 1)$. We often call $\mathbf z$ as the latent variable.
It then applies a function to generate $\mathbf x'=G(\mathbf z)$. The goal of the generator is to fool the discriminator to classify $\mathbf x'=G(\mathbf z)$ as true data, *i.e.*, we want $D( G(\mathbf z)) \approx 1$.
The Gaussian used below has independent coordinates, written
$z\sim\mathcal N(0,I)$ for a vector. Its distribution is fixed; the generator's
weights are learned.

For equal real and generated batch sizes, the D2L implementation averages the
two summed classification losses:

$$
\begin{aligned}
L_D={}&-\frac12\sum_{i=1}^B\log D(x_i)\\
      &-\frac12\sum_{i=1}^B\log(1-D(G(z_i))).
\end{aligned}
$$

The original minimax game instead writes an expectation for each population:

$$
\begin{gathered}
\min_G\max_D V(D,G),\\
V(D,G)=\mathbb E_{x\sim p_{\rm data}}\log D(x)\\
{}+\mathbb E_{z\sim p_z}\log(1-D(G(z))).
\end{gathered}
$$

D maximizes classification log-likelihood; minimizing its negative implements
the same direction. G changes the generated distribution to reduce the value.
Multiplying an objective by a positive constant preserves its optima but
rescales the gradients. A sum over a batch and a mean over that batch therefore
need not produce the same optimizer step at the same learning rate.

## Why the generator usually uses a different loss

Let $a$ be the discriminator's logit on one generated example and
$q=\sigma(a)$. The minimax generator term and the commonly used
non-saturating term have derivatives

$$
\begin{aligned}
L_G^{\rm mm}&=\log(1-q),&
\frac{\partial L_G^{\rm mm}}{\partial a}&=-q,\\
L_G^{\rm ns}&=-\log q,&
\frac{\partial L_G^{\rm ns}}{\partial a}&=q-1.
\end{aligned}
$$

When D confidently rejects a generated example, $q$ is near zero. The minimax
derivative then becomes small. At $a=-6$, $q\approx0.00247$: the two derivatives
are approximately $-0.00247$ and $-0.99753$. Gradient descent on either term
tries to increase the logit, but the non-saturating term gives a much larger
signal in this case. The full generator gradient also contains the derivatives
of D with respect to its input and of G with respect to its parameters.

This is the early-training case described by
[Goodfellow et al., §3](https://arxiv.org/html/1406.2661v1#S3).
The D2L paragraph motivating the replacement reverses this case and attributes
the small gradient to the discriminator. Here it is corrected explicitly:
the relevant small gradient is the generator's minimax gradient when D
successfully rejects its samples.


```python
import math
import torch
from torch import nn
from torch.nn import functional as F

torch.set_num_threads(2)
a = torch.tensor([-6., 0., 6.], dtype=torch.float64, requires_grad=True)
q = a.sigmoid()
minimax = -F.softplus(a)          # log(1 - sigmoid(a))
non_saturating = F.softplus(-a)  # -log(sigmoid(a))
grad_mm, = torch.autograd.grad(minimax.sum(), a, retain_graph=True)
grad_ns, = torch.autograd.grad(non_saturating.sum(), a)
torch.testing.assert_close(grad_mm, -q)
torch.testing.assert_close(grad_ns, q - 1)
assert abs(grad_mm[0]) < .003 and abs(grad_ns[0]) > .997
print(torch.stack([a.detach(), q.detach(), grad_mm, grad_ns], dim=1))
```

Passing label 1 for a generated input to binary cross-entropy implements
$-\log D(G(z))$. The label describes what the generator wants D to predict;
it does not turn the generated example into a real dataset observation.
The discriminator's own update still labels that example 0.

## What the idealized equilibrium establishes

For fixed generated density $g(x)$ and data density $p(x)$, D maximizes
$p\log D+g\log(1-D)$ at each point. Its derivative is
$p/D-g/(1-D)$. Setting this to zero yields $p(1-D)=gD$, hence

$$
D^*(x)=\frac{p(x)}{p(x)+g(x)}.
$$

This expression assumes equal real/fake sampling weights and a discriminator
that can attain the pointwise optimum. It is undefined where both densities
vanish, which does not affect these expectations.

For two outcomes with $p=(0.8,0.2)$ and $g=(0.5,0.5)$,
$D^*=(8/13,2/7)$. Write $m=(p+g)/2$; then $D^*=p/(2m)$
and $1-D^*=g/(2m)$. Substituting these expressions into V separates
the constant $-2\log2$ from two relative entropies:

$$
\begin{aligned}
V(D^*,G)&=-\log4+{\rm KL}(p\|m)\\
&\quad+{\rm KL}(g\|m)\\
&=-\log4+2\,{\rm JS}(p\|g).
\end{aligned}
$$

Jensen–Shannon divergence is the average of those two KL terms, each
comparing a distribution with their common mixture. Its non-negativity
and equality condition give the idealized optimum at equal distributions.
At $p=g$, D returns one half on their support and both non-saturating and
half-averaged discriminator losses per example equal $\log2$.
The [paper's §4](https://arxiv.org/html/1406.2661v1#S4) analyzes the
unrestricted-distribution setting. It does not guarantee convergence of
alternating updates of finite neural networks.


```python
p = torch.tensor([.8, .2], dtype=torch.float64)
g = torch.tensor([.5, .5], dtype=torch.float64)
d_star = p / (p + g)
torch.testing.assert_close(d_star, torch.tensor([8/13, 2/7], dtype=p.dtype))
value = (p * d_star.log() + g * torch.log1p(-d_star)).sum()
mixture = (p + g) / 2
js = .5 * ((p * (p / mixture).log()).sum()
           + (g * (g / mixture).log()).sum())
torch.testing.assert_close(value, -torch.tensor(math.log(4.), dtype=p.dtype) + 2*js)
equal_d = p / (p + p)
torch.testing.assert_close(equal_d, torch.full_like(p, .5))
assert abs(-math.log(.5) - math.log(2)) < 1e-15
print("optimal discriminator", d_star, "JS", js.item())
```

An observed discriminator accuracy near 50% is not a test of distributional
equality by itself. An undertrained classifier can perform at chance even
when the two distributions differ. Losses, independent samples, coverage,
and the chosen held-out evaluation procedure need to be examined together.

## Fit a two-dimensional Gaussian first

Since this is going to be the world's lamest example, we simply generate data drawn from a Gaussian.


```python
torch.manual_seed(29)
X = torch.normal(0.0, 1, (1000, 2))
A = torch.tensor([[1., 2.], [-0.1, 0.5]])
b = torch.tensor([1., 2.])
data = torch.matmul(X, A) + b
population_covariance = A.T @ A
torch.testing.assert_close(
    population_covariance, torch.tensor([[1.01, 1.95], [1.95, 4.25]]))
assert data.shape == (1000, 2)
print("population mean", b, "covariance", population_covariance)
```

Each row of X is an independent sample with covariance I. For the row-vector
transformation $x=XA+b$, the population mean is b and the covariance is
$A^\mathsf T A$. A finite sample's empirical mean and covariance fluctuate;
the assertions above concern the population transformation, not exact equality
of sample moments.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/gan-dcgan-2026-10-28/gaussian-data.svg|Original D2L scatter plot of the two-dimensional Gaussian training data]]

*Original illustration from Aston Zhang, Zachary C. Lipton, Mu Li and Alexander J. Smola,
[Dive into Deep Learning, GAN](https://d2l.ai/chapter_generative-adversarial-networks/gan.html#generate-some-real-data),
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
Complete SVG, unchanged.*

Our generator network will be the simplest network possible - a single layer linear model. This is since we will be driving that linear network with a Gaussian data generator. Hence, it literally only needs to learn the parameters to fake things perfectly.
```python
net_G = nn.Sequential(nn.Linear(2, 2))
```

For row-vector input, a PyTorch linear layer computes
$zW^\mathsf T+c$, so the output covariance is $WW^\mathsf T$.
It can represent this Gaussian with $c=b$ and $W=A^\mathsf T$.
The same distribution can have other factorizations; fitting its moments
does not identify A uniquely.

For the discriminator we will be a bit more discriminating: we will use an MLP with 3 layers to make things a bit more interesting.
```python
net_D = nn.Sequential(
    nn.Linear(2, 5), nn.Tanh(),
    nn.Linear(5, 3), nn.Tanh(),
    nn.Linear(3, 1))
```

The discriminator ends with one unconstrained logit. Its final layer has no
sigmoid because BCEWithLogitsLoss combines sigmoid and cross-entropy using
stable arithmetic. Adding a sigmoid before that loss would change its input
from logits to probabilities and apply the sigmoid twice.

## The two source update functions

The following PyTorch functions are retained from D2L without changing their
bodies. The discriminator step detaches the generated tensor before using it
as a classifier input.


```python
def update_D(X, Z, net_D, net_G, loss, trainer_D):
    """Update discriminator."""
    batch_size = X.shape[0]
    ones = torch.ones((batch_size,), device=X.device)
    zeros = torch.zeros((batch_size,), device=X.device)
    trainer_D.zero_grad()
    real_Y = net_D(X)
    fake_X = net_G(Z)
    # Do not need to compute gradient for `net_G`, detach it from
    # computing gradients.
    fake_Y = net_D(fake_X.detach())
    loss_D = (loss(real_Y, ones.reshape(real_Y.shape)) +
              loss(fake_Y, zeros.reshape(fake_Y.shape))) / 2
    loss_D.backward()
    trainer_D.step()
    return loss_D
```

Detaching prevents the discriminator loss from backpropagating into G.
It does not detach D's parameters: D still needs their gradients to learn.
The generator step recomputes D's prediction after D's parameters have changed.


```python
def update_G(Z, net_D, net_G, loss, trainer_G):
    """Update generator."""
    batch_size = Z.shape[0]
    ones = torch.ones((batch_size,), device=Z.device)
    trainer_G.zero_grad()
    # We could reuse `fake_X` from `update_D` to save computation
    fake_X = net_G(Z)
    # Recomputing `fake_Y` is needed since `net_D` is changed
    fake_Y = net_D(fake_X)
    loss_G = loss(fake_Y, ones.reshape(fake_Y.shape))
    loss_G.backward()
    trainer_G.step()
    return loss_G
```

During the generator step, detaching the generated tensor or placing the
whole D forward pass under no_grad would remove the path needed to train G.
D's parameters need not be updated, but its input derivative must remain
available. The source computes gradients for D in this backward pass too;
only trainer_G.step applies a parameter update. The next discriminator call
clears D's accumulated gradients.

### Check parameter ownership in an actual alternating step


```python
torch.manual_seed(30)
for parameter in list(net_D.parameters()) + list(net_G.parameters()):
    nn.init.normal_(parameter, 0., .02)
loss = nn.BCEWithLogitsLoss(reduction="sum")
trainer_D = torch.optim.Adam(net_D.parameters(), lr=.05)
trainer_G = torch.optim.Adam(net_G.parameters(), lr=.005)
real_batch = data[:8]
noise_batch = torch.randn(8, 2)
saved_G = [p.detach().clone() for p in net_G.parameters()]
saved_D = [p.detach().clone() for p in net_D.parameters()]
loss_D = update_D(real_batch, noise_batch, net_D, net_G, loss, trainer_D)
assert all(p.grad is None for p in net_G.parameters())
assert all(torch.equal(p, old) for p, old in zip(net_G.parameters(), saved_G))
assert any(not torch.equal(p, old) for p, old in zip(net_D.parameters(), saved_D))
saved_D = [p.detach().clone() for p in net_D.parameters()]
saved_G = [p.detach().clone() for p in net_G.parameters()]
loss_G = update_G(noise_batch, net_D, net_G, loss, trainer_G)
assert all(torch.equal(p, old) for p, old in zip(net_D.parameters(), saved_D))
assert any(not torch.equal(p, old) for p, old in zip(net_G.parameters(), saved_G))
assert all(p.grad is not None and torch.isfinite(p.grad).all()
           for p in net_G.parameters())
print("D/G summed losses", loss_D.item(), loss_G.item())
```

These checks establish which optimizer changes which parameters. They do not
establish a useful generated distribution after one step.

### Run the Gaussian experiment without plotting dependencies

D2L trains with batch size 8, 20 epochs, learning rates 0.05 for D and 0.005
for G. This CPU loop keeps those choices and the source update functions.
It replaces D2L's animation/accumulator utilities with a loss ledger, so no
D2L package, image dataset, accelerator, or network download is required.


```python
torch.manual_seed(31)
for parameter in list(net_D.parameters()) + list(net_G.parameters()):
    nn.init.normal_(parameter, 0., .02)
trainer_D = torch.optim.Adam(net_D.parameters(), lr=.05)
trainer_G = torch.optim.Adam(net_G.parameters(), lr=.005)
loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(data), batch_size=8, shuffle=True)
gaussian_history = []
for epoch in range(20):
    total_D = total_G = 0.
    visits = 0
    for (real_batch,) in loader:
        noise_batch = torch.randn(len(real_batch), 2)
        total_D += update_D(real_batch, noise_batch, net_D, net_G,
                            loss, trainer_D).item()
        total_G += update_G(noise_batch, net_D, net_G, loss, trainer_G).item()
        visits += len(real_batch)
    gaussian_history.append((total_D / visits, total_G / visits))
with torch.no_grad():
    generated = net_G(torch.randn(10_000, 2))
assert torch.isfinite(generated).all()
print("final mean", generated.mean(0))
print("final covariance", torch.cov(generated.T))
print("final per-example losses", gaussian_history[-1])
```

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/gan-dcgan-2026-10-28/gaussian-training.svg|Original D2L Gaussian GAN training curves and real/generated samples]]

*Original illustration from Aston Zhang, Zachary C. Lipton, Mu Li and Alexander J. Smola,
[Dive into Deep Learning, GAN](https://d2l.ai/chapter_generative-adversarial-networks/gan.html#training),
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
Complete SVG, unchanged.*

This is the original book output, not a plot produced by the local loop.
Its first panel shows D's half-averaged loss and G's non-saturating loss
approaching roughly 0.69 in that run. The second panel compares samples.
A similar pair of losses alone would not rule out missing modes or a weak D.
Random initialization, shuffling, and numerical versions can change the
trajectory; the local loop prints its own moments rather than borrowing
the book's result.

In the executed CPU run above, the final mean was approximately $(0.925,1.812)$
and the covariance was approximately

$$
\begin{pmatrix}
0.631&1.364\\
1.364&3.243
\end{pmatrix},
$$

while both recorded losses were close to 0.693. The covariance still differs
from the target. These are observations of this seeded toy run, not a claim
that those loss values certify convergence. Different seeds and versions need
their own sample comparison.

## DCGAN: turn a noise vector into an RGB image

The generator needs to map the noise variable $\mathbf z\in\mathbb R^d$, a length-$d$ vector, to a RGB image with width and height to be $64\times 64$ . Transposed convolutions enlarge the spatial maps. The basic block of the generator contains a transposed convolution layer followed by the batch normalization and ReLU activation.

The layer changes the spatial size through the transpose of the corresponding
convolution operator. It is not an inverse that recovers discarded pixels.
The preceding
[[02 Areas/ML & DL/00 Учебник/01 Основы нейронных сетей/08 Autoencoder и VAE|autoencoder chapter]]
works through that distinction with explicit matrices.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/autoencoder-2026-10-26/transpose-convolution.gif|Original Dumoulin and Visin animation of padding, stride and transposed convolution]]

*Vincent Dumoulin and Francesco Visin,
[Convolution arithmetic: transposed convolution](https://github.com/vdumoulin/conv_arithmetic/tree/af6f818b0bb396c26da79899554682a8a499101d),
[MIT](/licenses/conv-arithmetic-af6f818b-MIT.txt).
Complete original animation, also used in the autoencoder chapter.*

For dilation 1 and output_padding 0, one spatial dimension has size

$$
h_{\rm out}=(h_{\rm in}-1)s-2p+k.
$$

With kernel 4, stride 2, and padding 1, this becomes $2h_{\rm in}$.
The initial block instead uses stride 1 and no padding: an input of size 1
becomes size 4. The following D2L blocks and networks retain the original
PyTorch code.


```python
class G_block(nn.Module):
    def __init__(self, out_channels, in_channels=3, kernel_size=4, strides=2,
                 padding=1, **kwargs):
        super(G_block, self).__init__(**kwargs)
        self.conv2d_trans = nn.ConvTranspose2d(in_channels, out_channels,
                                kernel_size, strides, padding, bias=False)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, X):
        return self.activation(self.batch_norm(self.conv2d_trans(X)))
```

```python
n_G = 64
net_G = nn.Sequential(
    G_block(in_channels=100, out_channels=n_G*8,
            strides=1, padding=0),                  # Output: (64 * 8, 4, 4)
    G_block(in_channels=n_G*8, out_channels=n_G*4), # Output: (64 * 4, 8, 8)
    G_block(in_channels=n_G*4, out_channels=n_G*2), # Output: (64 * 2, 16, 16)
    G_block(in_channels=n_G*2, out_channels=n_G),   # Output: (64, 32, 32)
    nn.ConvTranspose2d(in_channels=n_G, out_channels=3,
                       kernel_size=4, stride=2, padding=1, bias=False),
    nn.Tanh())  # Output: (3, 64, 64)
```

The four G_block objects produce spatial sizes 4, 8, 16, and 32; the final
ConvTranspose2d produces 64. Tanh changes the range, not the shape.

| Stage | Channels × height × width |
| --- | --- |
| Noise | 100 × 1 × 1 |
| First G_block | 512 × 4 × 4 |
| Second G_block | 256 × 8 × 8 |
| Third G_block | 128 × 16 × 16 |
| Fourth G_block | 64 × 32 × 32 |
| Output convolution and tanh | 3 × 64 × 64 |

The complete generator has five transposed convolutions: four in the basic
blocks and the output layer. Counting only the blocks would miss the final
spatial expansion.

## The convolutional discriminator

The discriminator is a normal convolutional network except that it uses a leaky ReLU as its activation function. Given $\alpha \in[0, 1]$, its definition is

$$
{\rm LeakyReLU}(x)=
\begin{cases}
x,&x>0,\\
\alpha x,&x\le0.
\end{cases}
$$

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/gan-dcgan-2026-10-28/leaky-relu.svg|Original D2L comparison of ReLU and leaky ReLU slopes]]

*Original illustration from Aston Zhang, Zachary C. Lipton, Mu Li and Alexander J. Smola,
[Dive into Deep Learning, DCGAN](https://d2l.ai/chapter_generative-adversarial-networks/dcgan.html#discriminator),
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
Complete SVG, unchanged.*

At a negative input, the input derivative is $\alpha$ rather than zero.
The plotted curve for 0.2 corresponds to the default in the following block.
This preserves a local derivative for a negative preactivation; it is not a
guarantee that every gradient elsewhere in a GAN will be useful.


```python
class D_block(nn.Module):
    def __init__(self, out_channels, in_channels=3, kernel_size=4, strides=2,
                padding=1, alpha=0.2, **kwargs):
        super(D_block, self).__init__(**kwargs)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size,
                                strides, padding, bias=False)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(alpha, inplace=True)

    def forward(self, X):
        return self.activation(self.batch_norm(self.conv2d(X)))
```

```python
n_D = 64
net_D = nn.Sequential(
    D_block(n_D),  # Output: (64, 32, 32)
    D_block(in_channels=n_D, out_channels=n_D*2),  # Output: (64 * 2, 16, 16)
    D_block(in_channels=n_D*2, out_channels=n_D*4),  # Output: (64 * 4, 8, 8)
    D_block(in_channels=n_D*4, out_channels=n_D*8),  # Output: (64 * 8, 4, 4)
    nn.Conv2d(in_channels=n_D*8, out_channels=1,
              kernel_size=4, bias=False))  # Output: (1, 1, 1)
```

A kernel-4, stride-2, padding-1 convolution halves each even spatial dimension:
64 becomes 32, then 16, 8, and 4. The last kernel-4 convolution has no padding
and reduces 4 to 1. The resulting shape is $[B,1,1,1]$, one logit per image,
not a class-probability vector. There are four blocks plus that final
convolution.

The D2L teaching network includes BatchNorm in its first discriminator block.
[Radford et al., §3](https://arxiv.org/html/1511.06434v2#S3) exclude BatchNorm
at the discriminator input layer and generator output. The code above is
therefore identified as D2L's variant, rather than presented as an exact
reproduction of the paper architecture.

## Range, training state, and a complete image-shaped step

D2L's image example resizes RGB images to $64\times64$, converts pixels to
$[0,1]$, then applies $(x-0.5)/0.5$ to obtain $[-1,1]$, matching the generator's
tanh output. Display conversion is $x/2+0.5$; it does not standardize the
images to a unit-variance Gaussian. A discriminator could otherwise exploit
a simple range mismatch instead of learning useful image structure.

The original dataset example uses Pokemon sprites from
[pokemondb](https://pokemondb.net/sprites). Its loader and full plotting run
remain available in [D2L §20.2](https://d2l.ai/chapter_generative-adversarial-networks/dcgan.html#the-pokemon-dataset).
They are not silently downloaded by the following code. This local check uses
assigned random RGB tensors to verify the full forward/backward path; it is
not an image-quality demonstration or a reproduction of dataset training.


```python
torch.manual_seed(32)
rgb_batch = torch.rand(2, 3, 64, 64) * 2 - 1
image_noise = torch.randn(2, 100, 1, 1)
g_shapes = []
with torch.no_grad():
    intermediate = image_noise
    for layer in net_G:
        intermediate = layer(intermediate)
        g_shapes.append(tuple(intermediate.shape[1:]))
assert g_shapes == [(512, 4, 4), (256, 8, 8), (128, 16, 16),
                    (64, 32, 32), (3, 64, 64), (3, 64, 64)]
assert intermediate.abs().max() <= 1
assert net_D(rgb_batch).shape == (2, 1, 1, 1)
# Reset only the counters used to demonstrate subsequent forward visits.
for model in [net_G, net_D]:
    for layer in model.modules():
        if isinstance(layer, nn.BatchNorm2d):
            layer.num_batches_tracked.zero_()
image_D_optimizer = torch.optim.Adam(net_D.parameters(), lr=.0002, betas=(.5, .999))
image_G_optimizer = torch.optim.Adam(net_G.parameters(), lr=.0002, betas=(.5, .999))
for model in [net_G, net_D]:
    model.zero_grad(set_to_none=True)
saved_G = [p.detach().clone() for p in net_G.parameters()]
image_loss_D = update_D(rgb_batch, image_noise, net_D, net_G,
                        loss, image_D_optimizer)
assert all(p.grad is None for p in net_G.parameters())
assert all(torch.equal(p, old) for p, old in zip(net_G.parameters(), saved_G))
assert net_G[0].batch_norm.num_batches_tracked.item() == 1
saved_D = [p.detach().clone() for p in net_D.parameters()]
image_loss_G = update_G(image_noise, net_D, net_G, loss, image_G_optimizer)
assert all(torch.equal(p, old) for p, old in zip(net_D.parameters(), saved_D))
assert net_G[0].batch_norm.num_batches_tracked.item() == 2
assert net_D[0].batch_norm.num_batches_tracked.item() == 3
assert all(p.grad is not None and torch.isfinite(p.grad).all()
           for p in net_G.parameters())
print("generator shapes", g_shapes)
print("image-shaped D/G losses", image_loss_D.item(), image_loss_G.item())
```

An unchanged parameter tensor does not mean an unchanged complete module state.
During D's update, G's forward pass can still update its BatchNorm buffers,
even though its output is detached afterward. D sees the real batch, the
detached fake batch, then the generator-step fake batch: three BatchNorm visits.
Freezing parameter gradients and choosing train/eval behavior are separate
decisions.

The step uses Adam with learning rate 0.0002 and $\beta_1=0.5$, as in
[Radford et al., §4](https://arxiv.org/html/1511.06434v2#S4).
D2L's short dataset demonstration instead sets learning rate 0.005 and trains
for 20 epochs. Those are distinct recipes, not interchangeable measurements.
One successful backward pass establishes neither stability over many updates
nor realistic image generation.

## Sampling, coverage, and downstream labels

After training, generating an image needs noise and G; it does not need D or
a reconstruction encoder. With BatchNorm, eval mode selects the stored
statistics and prevents sampling from changing them. The learned generator
can still depend on its training distribution and can memorize examples.

A generator that emits only a few convincing outputs can miss most of that
distribution. This is mode collapse, distinct from the recursive
synthetic-training model collapse discussed in the
[[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/53 Синтетические данные и учебные программы|synthetic-data chapter]].
The discriminator's loss alone does not measure that missing coverage.

For example, suppose the data assigns half its probability to each of two
separated groups, A and B. A generator that always emits A can produce sharp
individual samples while assigning zero probability to B. Under the ideal
pointwise calculation above, D would return $1/3$ on A and 1 on B, so this
generator is not the global distribution-matching solution. A finite,
alternately trained discriminator may fail to supply a useful signal for
the missing group. Checking only the appearance of a few A samples would
not expose the missing half of the distribution.

This DCGAN is unconditional: a real/fake training target is not a semantic
class label for each generated image. Using generated images to train a
classifier also requires a way to obtain valid class labels and a held-out
real-data comparison. The
[[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/53a Синтетические изображения и перенос из симуляции|synthetic-image chapter]]
develops label provenance, domain differences, and real-data evaluation.
Generation by itself does not close those requirements.

## Sources and reuse

The source English and PyTorch bodies are retained from Aston Zhang,
Zachary C. Lipton, Mu Li and Alexander J. Smola,
[Dive into Deep Learning, GAN](https://d2l.ai/chapter_generative-adversarial-networks/gan.html)
and [DCGAN](https://d2l.ai/chapter_generative-adversarial-networks/dcgan.html),
archived at [23d7a5a](https://github.com/d2l-ai/d2l-en/tree/23d7a5aecceee57d1292c56e90cce307f183bb0a/chapter_generative-adversarial-networks).
This adapted chapter is distributed under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
The source plotting utilities, repeated framework variants, dated introductions,
summaries, and exercises are omitted. The loss motivation is corrected,
shapes and gradient ownership are worked through, and dataset training is
separated from executed CPU examples. No new English translation was authored.

The primary papers support the mechanisms and the stated limits:

- Ian Goodfellow et al., [Generative Adversarial Nets, §§3–4](https://arxiv.org/abs/1406.2661).
- Alec Radford, Luke Metz and Soumith Chintala,
  [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks, §§3–4](https://arxiv.org/abs/1511.06434).

---

**Предыдущая глава:** [[02 Areas/ML & DL/00 Учебник/01 Основы нейронных сетей/08 Autoencoder и VAE|Autoencoder и VAE]]

**Следующая глава:** [[02 Areas/ML & DL/00 Учебник/16 Multimodal Models/64g Диффузионные и flow-модели изображений|Diffusion, latent diffusion и flow matching]]

