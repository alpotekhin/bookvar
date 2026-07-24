---
title: "Deep Learning Tips and Tricks"
type: external-resource
status: imported-source
language: original
source_kind: slides
source_commit: 4653bc01297b269edb19e844b01127ba13de59df
---

> [!note] Original source material
> This page preserves [`en/cheatsheet-deep-learning-tips-tricks.pdf`](https://github.com/afshinea/stanford-cs-230-deep-learning/blob/4653bc01297b269edb19e844b01127ba13de59df/en/cheatsheet-deep-learning-tips-tricks.pdf) from
> *Stanford CS230 Deep Learning Cheatsheets* at commit `4653bc01297b269edb19e844b01127ba13de59df`. License:
> [MIT](https://github.com/afshinea/stanford-cs-230-deep-learning/blob/4653bc01297b269edb19e844b01127ba13de59df/LICENSE). Bookvar changed only
> publication markup, link paths, and characters required for safe rendering.



The embedded PDF is the primary visual document. The page-separated text below was extracted mechanically for search and has not been rewritten.

<iframe class="source-pdf" src="https://github.com/afshinea/stanford-cs-230-deep-learning/blob/4653bc01297b269edb19e844b01127ba13de59df/en/cheatsheet-deep-learning-tips-tricks.pdf?raw=1" title="Deep Learning Tips and Tricks" loading="lazy"></iframe>

## Extracted document text

### Page 1

```text
CS 230 – Deep Learning                                                                                                                                        https://stanford.edu/~shervine


                 VIP Cheatsheet: Tips and Tricks
                                                                                                                                                     xi − µ B
                                                                                                                                             xi ←− γ p        +β
                                                                                                                                                        2 +
                                                                                                                                                       σB
                        Afshine Amidi and Shervine Amidi
                                   November 26, 2018                                                    It is usually done after a fully connected/convolutional layer and before a non-linearity layer and
                                                                                                        aims at allowing higher learning rates and reducing the strong dependence on initialization.


Data processing                                                                                         Training a neural network
r Data augmentation – Deep learning models usually need a lot of data to be properly trained.
It is often useful to get more data from the existing ones using data augmentation techniques.          r Epoch – In the context of training a model, epoch is a term used to refer to one iteration
The main ones are summed up in the table below. More precisely, given the following input               where the model sees the whole training set to update its weights.
image, here are the techniques that we can apply:
                                                                                                        r Mini-batch gradient descent – During the training phase, updating weights is usually not
                                                                                                        based on the whole training set at once due to computation complexities or one data point due
      Original                   Flip                     Rotation             Random crop              to noise issues. Instead, the update step is done on mini-batches, where the number of data
                                                                                                        points in a batch is a hyperparameter that we can tune.

                                                                                                        r Loss function – In order to quantify how a given model performs, the loss function L is
                                                                                                        usually used to evaluate to what extent the actual outputs y are correctly predicted by the
                                                                                                        model outputs z.

                                                                                                        r Cross-entropy loss – In the context of binary classification in neural networks, the cross-
                                                                                                        entropy loss L(z,y) is commonly used and is defined as follows:
                                                                                                                                               h                               i
                                                                                                                                    L(z,y) = − y log(z) + (1 − y) log(1 − z)
                                                                              - Random focus
                         - Flipped with respect     - Rotation with           on one part of
  - Image without        to an axis for which       a slight angle            the image
                         the meaning of the         - Simulates incorrect     - Several random          r Backpropagation – Backpropagation is a method to update the weights in the neural network
  any modification                                                                                      by taking into account the actual output and the desired output. The derivative with respect
                         image is preserved         horizon calibration       crops can be
                                                                                                        to each weight w is computed using the chain rule.
                                                                              done in a row

     Color shift          Noise addition           Information loss          Contrast change




                                                                                                        Using this method, each weight is updated with the rule:

                                                                                                                                                            ∂L(z,y)
                                                                                                                                              w ←− w − α
                                                                                                                                                              ∂w
  - Nuances of RGB
                         - Addition of noise      - Parts of image          - Luminosity changes
  is slightly changed
                         - More tolerance to      ignored                   - Controls difference
  - Captures noise                                                                                      r Updating weights – In a neural network, weights are updated as follows:
                         quality variation of     - Mimics potential        in exposition due
  that can occur
                         inputs                   loss of parts of image    to time of day
  with light exposure                                                                                      • Step 1: Take a batch of training data and perform forward propagation to compute the
                                                                                                             loss.

r Batch normalization – It is a step of hyperparameter γ, β that normalizes the batch {xi }.               • Step 2: Backpropagate the loss to get the gradient of the loss with respect to each weight.
By noting µB , σB
                2 the mean and variance of that we want to correct to the batch, it is done as

follows:                                                                                                   • Step 3: Use the gradients to update the weights of the network.


Stanford University                                                                                 1                                                                                    Winter 2019
```

### Page 2

```text
CS 230 – Deep Learning                                                                                                                                                 https://stanford.edu/~shervine


                                                                                                               Method                 Explanation                      Update of w                 Update of b
                                                                                                                             - Dampens oscillations
                                                                                                              Momentum       - Improvement to SGD                        w − αvdw                     b − αvdb
                                                                                                                             - 2 parameters to tune
                                                                                                                             - Root Mean Square propagation
                                                                                                                                                                              dw                             db
                                                                                                               RMSprop       - Speeds up learning algorithm             w − α√                   b ←− b − α √
                                                                                                                                                                              sdw                             sdb
                                                                                                                             by controlling oscillations
                                                                                                                             - Adaptive Moment estimation
                                                                                                                                                                                 vdw                             vdb
Parameter tuning                                                                                                Adam         - Most popular method                     w − α√                  b ←− b − α √
                                                                                                                                                                                sdw +                          sdb + 
                                                                                                                             - 4 parameters to tune
r Xavier initialization – Instead of initializing the weights in a purely random manner, Xavier
initialization enables to have initial weights that take into account characteristics that are unique       Remark: other methods include Adadelta, Adagrad and SGD.
to the architecture.

r Transfer learning – Training a deep learning model requires a lot of data and more impor-
tantly a lot of time. It is often useful to take advantage of pre-trained weights on huge datasets          Regularization
that took days/weeks to train, and leverage it towards our use case. Depending on how much
data we have at hand, here are the different ways to leverage this:                                         r Dropout – Dropout is a technique used in neural networks to prevent overfitting the training
                                                                                                            data by dropping out neurons with probability p > 0. It forces the model to avoid relying too
                                                                                                            much on particular sets of features.


    Training size                   Illustration                          Explanation


                                                                   Freezes all layers,
        Small
                                                                   trains weights on softmax

                                                                                                            Remark: most deep learning frameworks parametrize dropout through the ’keep’ parameter 1−p.

                                                                                                            r Weight regularization – In order to make sure that the weights are not too large and that
                                                                   Freezes most layers,                     the model is not overfitting the training set, regularization techniques are usually performed on
       Medium                                                      trains weights on last                   the model weights. The main ones are summed up in the table below:
                                                                   layers and softmax

                                                                                                                        LASSO                           Ridge                                 Elastic Net

                                                                   Trains weights on layers                  - Shrinks coefficients to 0                                         Tradeoff between variable
                                                                                                                                              Makes coefficients smaller
        Large                                                      and softmax by initializing               - Good for variable selection                                       selection and small coefficients
                                                                   weights on pre-trained ones




r Learning rate – The learning rate, often noted α or sometimes η, indicates at which pace the
weights get updated. It can be fixed or adaptively changed. The current most popular method
is called Adam, which is a method that adapts the learning rate.

r Adaptive learning rates – Letting the learning rate vary when training a model can reduce                                                                                               h                          i
the training time and improve the numerical optimal solution. While Adam optimizer is the                             ... + λ||θ||1                   ... + λ||θ||22             ... + λ (1 − α)||θ||1 + α||θ||22
most commonly used technique, others can also be useful. They are summed up in the table                                  λ∈R                             λ∈R
below:                                                                                                                                                                                        λ ∈ R,α ∈ [0,1]


Stanford University                                                                                     2                                                                                            Winter 2019
```

### Page 3

```text
CS 230 – Deep Learning                                                                                      https://stanford.edu/~shervine


r Early stopping – This regularization technique stops the training process as soon as the
validation loss reaches a plateau or starts to increase.




Good practices
r Overfitting small batch – When debugging a model, it is often useful to make quick tests
to see if there is any major issue with the architecture of the model itself. In particular, in order
to make sure that the model can be properly trained, a mini-batch is passed inside the network
to see if it can overfit on it. If it cannot, it means that the model is either too complex or not
complex enough to even overfit on a small batch, let alone a normal-sized training set.
r Gradient checking – Gradient checking is a method used during the implementation of
the backward pass of a neural network. It compares the value of the analytical gradient to the
numerical gradient at given points and plays the role of a sanity-check for correctness.

                          Numerical gradient                        Analytical gradient
                       df       f (x + h) − f (x − h)                    df
   Formula                (x) ≈                                             (x) = f 0 (x)
                       dx                2h                              dx
                   - Expensive; loss has to be
                   computed two times per dimension
                                                              - ’Exact’ result
                   - Used to verify correctness
  Comments         of analytical implementation               - Direct computation
                   -Trade-off in choosing h
                   not too small (numerical instability)      - Used in the final implementation
                   nor too large (poor gradient approx.)



                                             ?   ?    ?




Stanford University                                                                                     3                    Winter 2019
```
