---
title: "Deep Learning Cheatsheet — Complete Compilation"
type: external-resource
status: imported-source
language: original
source_kind: slides
source_commit: 4653bc01297b269edb19e844b01127ba13de59df
---

> [!note] Original source material
> This page preserves [`en/super-cheatsheet-deep-learning.pdf`](https://github.com/afshinea/stanford-cs-230-deep-learning/blob/4653bc01297b269edb19e844b01127ba13de59df/en/super-cheatsheet-deep-learning.pdf) from
> *Stanford CS230 Deep Learning Cheatsheets* at commit `4653bc01297b269edb19e844b01127ba13de59df`. License:
> [MIT](https://github.com/afshinea/stanford-cs-230-deep-learning/blob/4653bc01297b269edb19e844b01127ba13de59df/LICENSE). Bookvar changed only
> publication markup, link paths, and characters required for safe rendering.



The embedded PDF is the primary visual document. The page-separated text below was extracted mechanically for search and has not been rewritten.

<iframe class="source-pdf" src="https://github.com/afshinea/stanford-cs-230-deep-learning/blob/4653bc01297b269edb19e844b01127ba13de59df/en/super-cheatsheet-deep-learning.pdf?raw=1" title="Deep Learning Cheatsheet — Complete Compilation" loading="lazy"></iframe>

## Extracted document text

### Page 1

```text
CS 230 – Deep Learning                                                                                                                           Shervine Amidi & Afshine Amidi



          Super VIP Cheatsheet: Deep Learning                                                1     Convolutional Neural Networks


                    Afshine Amidi and Shervine Amidi                                         1.1    Overview

                               November 25, 2018
                                                                                             r Architecture of a traditional CNN – Convolutional neural networks, also known as CNNs,
                                                                                             are a specific type of neural networks that are generally composed of the following layers:
Contents


1 Convolutional Neural Networks                                                     2
  1.1 Overview . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .    2
  1.2 Types of layer . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .    2
  1.3 Filter hyperparameters . . . . . . . . . . . . . . . . . . . . . . . . . .    2
  1.4 Tuning hyperparameters . . . . . . . . . . . . . . . . . . . . . . . . .      3
  1.5 Commonly used activation functions . . . . . . . . . . . . . . . . . . .      3
  1.6 Object detection . . . . . . . . . . . . . . . . . . . . . . . . . . . . .    4 The convolution layer and the pooling layer can be fine-tuned with respect to hyperparameters
      1.6.1 Face verification and recognition . . . . . . . . . . . . . . . . .     5 that are described in the next sections.
      1.6.2 Neural style transfer . . . . . . . . . . . . . . . . . . . . . . .     5
      1.6.3 Architectures using computational tricks . . . . . . . . . . . .        6

2 Recurrent Neural Networks                                                          7
  2.1 Overview . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .     7
  2.2 Handling long term dependencies . . . . . . . . . . . . . . . . . . . .        8
                                                                                             1.2    Types of layer
  2.3 Learning word representation . . . . . . . . . . . . . . . . . . . . . .       9
      2.3.1 Motivation and notations . . . . . . . . . . . . . . . . . . .           9
      2.3.2 Word embeddings . . . . . . . . . . . . . . . . . . . . . . .            9
                                                                                             r Convolutional layer (CONV) – The convolution layer (CONV) uses filters that perform
  2.4 Comparing words . . . . . . . . . . . . . . . . . . . . . . . . . . . .        9       convolution operations as it is scanning the input I with respect to its dimensions. Its hyperpa-
  2.5 Language model . . . . . . . . . . . . . . . . . . . . . . . . . . . . .      10       rameters include the filter size F and stride S. The resulting output O is called feature map or
                                                                                             activation map.
  2.6 Machine translation . . . . . . . . . . . . . . . . . . . . . . . . . . .     10
  2.7 Attention . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .   10

3 Deep Learning Tips and Tricks                                                     11
  3.1 Data processing . . . . . . . . . . . . . . . . . . . . . . . . . . . . .     11
  3.2 Training a neural network . . . . . . . . . . . . . . . . . . . . . . . .     12
      3.2.1 Definitions . . . . . . . . . . . . . . . . . . . . . . . . . . . .     12
      3.2.2 Finding optimal weights . . . . . . . . . . . . . . . . . . . . .       12
  3.3 Parameter tuning . . . . . . . . . . . . . . . . . . . . . . . . . . . .      12
                                                                                             Remark: the convolution step can be generalized to the 1D and 3D cases as well.
      3.3.1 Weights initialization . . . . . . . . . . . . . . . . . . . . . .      12
      3.3.2 Optimizing convergence . . . . . . . . . . . . . . . . . . . . .        12
                                                                                             r Pooling (POOL) – The pooling layer (POOL) is a downsampling operation, typically applied
  3.4 Regularization . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .    13       after a convolution layer, which does some spatial invariance. In particular, max and average
  3.5 Good practices . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .    13       pooling are special kinds of pooling where the maximum and average value is taken, respectively.


Stanford University                                                                      1                                                                                  Winter 2019
```

### Page 2

```text
CS 230 – Deep Learning                                                                                                                                          Shervine Amidi & Afshine Amidi


                              Max pooling                         Average pooling
                    Each pooling operation selects the     Each pooling operation averages
      Purpose
                    maximum value of the current view      the values of the current view
                                                                                                       r Zero-padding – Zero-padding denotes the process of adding P zeroes to each side of the
                                                                                                       boundaries of the input. This value can either be manually specified or automatically set through
                                                                                                       one of the three modes detailed below:

                                                                                                                               Valid                           Same                          Full
   Illustration                                                                                                                                            j                   k
                                                                                                                                                                  I e−I+F −S
                                                                                                                                                               Sd S
                                                                                                                                                Pstart =              2              Pstart ∈ [​[0,F − 1]​]
                                                                                                             Value             P =0                        l      I e−I+F −S
                                                                                                                                                                               m
                                                                                                                                                               Sd S
                                                                                                                                                Pend =                                  Pend = F − 1
                                                                                                                                                                      2
                    - Preserves detected features          - Downsamples feature map
      Comments
                    - Most commonly used                   - Used in LeNet

                                                                                                        Illustration
r Fully Connected (FC) – The fully connected layer (FC) operates on a flattened input where
each input is connected to all neurons. If present, FC layers are usually found towards the end
of CNN architectures and can be used to optimize objectives such as class scores.
                                                                                                                                                                                   - Maximum padding
                                                                                                                         - No padding          - Padding such that feature
                                                                                                                                                                      l m          such that end
                                                                                                                                               map size has size          I
                                                                                                                                                                                   convolutions are
                                                                                                                         - Drops last                                     S
                                                                                                          Purpose                                                                  applied on the limits
                                                                                                                         convolution if        - Output size is
                                                                                                                                                                                   of the input
                                                                                                                         dimensions do not     mathematically convenient
                                                                                                                         match                                                     - Filter ’sees’ the input
                                                                                                                                               - Also called ’half’ padding
                                                                                                                                                                                   end-to-end


                                                                                                       1.4    Tuning hyperparameters
                                                                                                       r Parameter compatibility in convolution layer – By noting I the length of the input
1.3    Filter hyperparameters                                                                          volume size, F the length of the filter, P the amount of zero padding, S the stride, then the
                                                                                                       output size O of the feature map along that dimension is given by:
The convolution layer contains filters for which it is important to know the meaning behind its
hyperparameters.                                                                                                                             I − F + Pstart + Pend
                                                                                                                                        O=                         +1
r Dimensions of a filter – A filter of size F × F applied to an input containing C channels is                                                        S
a F × F × C volume that performs convolutions on an input of size I × I × C and produces an
output feature map (also called activation map) of size O × O × 1.




Remark: the application of K filters of size F × F results in an output feature map of size
O × O × K.

r Stride – For a convolutional or a pooling operation, the stride S denotes the number of pixels       Remark: often times, Pstart = Pend , P , in which case we can replace Pstart + Pend by 2P in
by which the window moves after each operation.                                                        the formula above.


Stanford University                                                                                2                                                                                       Winter 2019
```

### Page 3

```text
CS 230 – Deep Learning                                                                                                                                        Shervine Amidi & Afshine Amidi


r Understanding the complexity of the model – In order to assess the complexity of a                                     ReLU                        Leaky ReLU                        ELU
model, it is often useful to determine the number of parameters that its architecture will have.
In a given layer of a convolutional neural network, it is done as follows:                                                                         g(z) = max(z,z)           g(z) = max(α(ez − 1),z)
                                                                                                                    g(z) = max(0,z)
                                                                                                                                                      with   1                    with α  1
                           CONV                         POOL                      FC



  Illustration



   Input size             I ×I ×C                      I ×I ×C                    Nin
                                                                                                               Non-linearity complexities     Addresses dying ReLU
                                                                                                                                                                              Differentiable everywhere
  Output size            O×O×K                        O×O×C                      Nout                          biologically interpretable     issue for negative values
  Number of
                    (F × F × C + 1) · K                       0            (Nin + 1) × Nout
  parameters                                                                                             r Softmax – The softmax step can be seen as a generalized logistic function that takes as input
                                                                                                         a vector of scores x ∈ Rn and outputs a vector of output probability p ∈ Rn through a softmax
                                                                         - Input is flattened            function at the end of the architecture. It is defined as follows:
                   - One bias parameter
                                              - Pooling operation        - One bias parameter
                   per filter
                                              done channel-wise          per neuron                                                          p1 
      Remarks      - In most cases, S < F                                                                                                     ..                       e xi
                                                                         - The number of FC                                            p=             where    pi =
                   - A common choice          - In most cases, S = F                                                                           .                      n
                                                                         neurons is free of                                                  pn                       X
                   for K is 2C                                                                                                                                              e xj
                                                                         structural constraints
                                                                                                                                                                      j=1


r Receptive field – The receptive field at layer k is the area denoted Rk × Rk of the input
that each pixel of the k-th activation map can ’see’. By calling Fj the filter size of layer j and       1.6    Object detection
Si the stride value of layer i and with the convention S0 = 1, the receptive field at layer k can
be computed with the formula:                                                                            r Types of models – There are 3 main types of object recognition algorithms, for which the
                                             k                j−1                                        nature of what is predicted is different. They are described in the table below:
                                             X                Y
                                  Rk = 1 +         (Fj − 1)         Si
                                                                                                                                            Classification
                                             j=1              i=0                                          Image classification                                                      Detection
                                                                                                                                            w. localization
In the example below, we have F1 = F2 = 3 and S1 = S2 = 1, which gives R2 = 1+2 · 1+2 · 1 =
5.




                                                                                                           - Classifies a picture     - Detects object in a picture    - Detects up to several objects
                                                                                                                                      - Predicts probability of        in a picture
                                                                                                           - Predicts probability     object and where it is           - Predicts probabilities of objects
                                                                                                           of object                  located                          and where they are located
                                                                                                           Traditional CNN            Simplified YOLO, R-CNN           YOLO, R-CNN
1.5     Commonly used activation functions
r Rectified Linear Unit – The rectified linear unit layer (ReLU) is an activation function g             r Detection – In the context of object detection, different methods are used depending on
that is used on all elements of the volume. It aims at introducing non-linearities to the network.       whether we just want to locate the object or detect a more complex shape in the image. The
Its variants are summarized in the table below:                                                          two main ones are summed up in the table below:


Stanford University                                                                                  3                                                                                      Winter 2019
```

### Page 4

```text
CS 230 – Deep Learning                                                                                                                                            Shervine Amidi & Afshine Amidi


             Bounding box detection                       Landmark detection
                                                 - Detects a shape or characteristics of
        Detects the part of the image where
                                                 an object (e.g. eyes)
        the object is located
                                                 - More granular




                                                                                                        r YOLO – You Only Look Once (YOLO) is an object detection algorithm that performs the
                                                                                                        following steps:

                                                                                                           • Step 1: Divide the input image into a G × G grid.

                                                                                                           • Step 2: For each grid cell, run a CNN that predicts y of the following form:
        Box of center (bx ,by ), height bh
                                                 Reference points (l1x ,l1y ), ...,(lnx ,lny )                                                                              T
        and width bw                                                                                                           y = pc ,bx ,by ,bh ,bw ,c1 ,c2 ,...,cp ,...        ∈ RG×G×k×(5+p)
                                                                                                                                        |           {z              }
                                                                                                                                             repeated k times

r Intersection over Union – Intersection over Union, also known as IoU, is a function that
quantifies how correctly positioned a predicted bounding box Bp is over the actual bounding                   where pc is the probability of detecting an object, bx ,by ,bh ,bw are the properties of the
box Ba . It is defined as:                                                                                    detected bouding box, c1 ,...,cp is a one-hot representation of which of the p classes were
                                                                                                              detected, and k is the number of anchor boxes.
                                                      Bp ∩ Ba
                                     IoU(Bp ,Ba ) =                                                        • Step 3: Run the non-max suppression algorithm to remove any potential duplicate over-
                                                      Bp ∪ Ba
                                                                                                             lapping bounding boxes.




                                                                                                        Remark: when pc = 0, then the network does not detect any object. In that case, the corre-
                                                                                                        sponding predictions bx , ..., cp have to be ignored.
Remark: we always have IoU ∈ [0,1]. By convention, a predicted bounding box Bp is considered
as being reasonably good if IoU(Bp ,Ba ) ⩾ 0.5.                                                         r R-CNN – Region with Convolutional Neural Networks (R-CNN) is an object detection algo-
                                                                                                        rithm that first segments the image to find potential relevant bounding boxes and then run the
r Anchor boxes – Anchor boxing is a technique used to predict overlapping bounding boxes.               detection algorithm to find most probable objects in those bounding boxes.
In practice, the network is allowed to predict more than one box simultaneously, where each box
prediction is constrained to have a given set of geometrical properties. For instance, the first
prediction can potentially be a rectangular box of a given form, while the second will be another
rectangular box of a different geometrical form.
r Non-max suppression – The non-max suppression technique aims at removing duplicate
overlapping bounding boxes of a same object by selecting the most representative ones. After
having removed all boxes having a probability prediction lower than 0.6, the following steps are
repeated while there are boxes remaining:

   • Step 1: Pick the box with the largest prediction probability.
                                                                                                        Remark: although the original algorithm is computationally expensive and slow, newer archi-
   • Step 2: Discard any box having an IoU ⩾ 0.5 with the previous box.                                 tectures enabled the algorithm to run faster, such as Fast R-CNN and Faster R-CNN.


Stanford University                                                                                 4                                                                                          Winter 2019
```

### Page 5

```text
CS 230 – Deep Learning                                                                                                                                                   Shervine Amidi & Afshine Amidi


1.6.1    Face verification and recognition

r Types of models – Two main types of model are summed up in table below:


             Face verification                           Face recognition
        - Is this the correct person?      - Is this one of the K persons in the database?
        - One-to-one lookup                - One-to-many lookup

                                                                                                          r Activation – In a given layer l, the activation is noted a[l] and is of dimensions nH × nw × nc

                                                                                                          r Content cost function – The content cost function Jcontent (C,G) is used to determine how
                                                                                                          the generated image G differs from the original content image C. It is defined as follows:

                                                                                                                                                               1 [l](C)
                                                                                                                                         Jcontent (C,G) =        ||a    − a[l](G) ||2
                                                                                                                                                               2


                                                                                                          r Style matrix – The style matrix G[l] of a given layer l is a Gram matrix where each of its
                                                                                                                     [l]
                                                                                                          elements Gkk0 quantifies how correlated the channels k and k0 are. It is defined with respect to
r One Shot Learning – One Shot Learning is a face verification algorithm that uses a limited              activations a[l] as follows:
training set to learn a similarity function that quantifies how different two given images are. The                                                          [l]   [l]
similarity function applied to two images is often noted d(image 1, image 2).                                                                            n nw
                                                                                                                                                         H X
                                                                                                                                                         X
                                                                                                                                                 [l]          [l]              [l]
                                                                                                                                                Gkk0 =                   aijk aijk0
r Siamese Network – Siamese Networks aim at learning how to encode images to then quantify
                                                                                                                                                         i=1 j=1
how different two images are. For a given input image x(i) , the encoded output is often noted
as f (x(i) ).
                                                                                                          Remark: the style matrix for the style image and the generated image are noted G[l](S) and
r Triplet loss – The triplet loss ` is a loss function computed on the embedding representation           G[l](G) respectively.
of a triplet of images A (anchor), P (positive) and N (negative). The anchor and the positive
example belong to a same class, while the negative example to another one. By calling α ∈ R+              r Style cost function – The style cost function Jstyle (S,G) is used to determine how the
the margin parameter, this loss is defined as follows:                                                    generated image G differs from the style S. It is defined as follows:

                           `(A,P,N ) = max (d(A,P ) − d(A,N ) + α,0)                                                                                                                     nc 
                                                                                                                                 1                                        1              X                           2
                                                                                                             [l]                                                                              [l](S)        [l](G)
                                                                                                           Jstyle (S,G) =                 ||G[l](S) − G[l](G) ||2F =                              Gkk0   − Gkk0
                                                                                                                            (2nH nw nc )2                            (2nH nw nc )2
                                                                                                                                                                                        k,k0 =1




                                                                                                          r Overall cost function – The overall cost function is defined as being a combination of the
                                                                                                          content and style cost functions, weighted by parameters α,β, as follows:

                                                                                                                                         J(G) = αJcontent (C,G) + βJstyle (S,G)

                                                                                                          Remark: a higher value of α will make the model care more about the content while a higher
                                                                                                          value of β will make it care more about the style.


                                                                                                          1.6.3    Architectures using computational tricks
1.6.2    Neural style transfer                                                                            r Generative Adversarial Network – Generative adversarial networks, also known as GANs,
                                                                                                          are composed of a generative and a discriminative model, where the generative model aims at
r Motivation – The goal of neural style transfer is to generate an image G based on a given               generating the most truthful output that will be fed into the discriminative which aims at
content C and a given style S.                                                                            differentiating the generated and true image.


Stanford University                                                                                   5                                                                                              Winter 2019
```

### Page 6

```text
CS 230 – Deep Learning                                                                                                                                         Shervine Amidi & Afshine Amidi


                                                                                                       2     Recurrent Neural Networks

                                                                                                       2.1    Overview
                                                                                                       r Architecture of a traditional RNN – Recurrent neural networks, also known as RNNs,
                                                                                                       are a class of neural networks that allow previous outputs to be used as inputs while having
                                                                                                       hidden states. They are typically as follows:




Remark: use cases using variants of GANs include text to image, music generation and syn-
thesis.
r ResNet – The Residual Network architecture (also called ResNet) uses residual blocks with a
high number of layers meant to decrease the training error. The residual block has the following
characterizing equation:
                                    a[l+2] = g(a[l] + z [l+2] )                                        For each timestep t, the activation a<t> and the output y <t> are expressed as follows:

r Inception Network – This architecture uses inception modules and aims at giving a try
at different convolutions in order to increase its performance. In particular, it uses the 1 × 1               a<t> = g1 (Waa a<t−1> + Wax x<t> + ba )          and   y <t> = g2 (Wya a<t> + by )
convolution trick to lower the burden of computation.
                                                                                                       where Wax , Waa , Wya , ba , by are coefficients that are shared temporally and g1 , g2 activation
                                                                                                       functions
                                            ?   ?    ?




                                                                                                       The pros and cons of a typical RNN architecture are summed up in the table below:

                                                                                                                             Advantages                                     Drawbacks
                                                                                                             - Possibility of processing input of any length    - Computation being slow
                                                                                                             - Model size not increasing with size of input     - Difficulty of accessing information
                                                                                                             - Computation takes into account                   from a long time ago
                                                                                                             historical information                             - Cannot consider any future input
                                                                                                             - Weights are shared across time                   for the current state


                                                                                                       r Applications of RNNs – RNN models are mostly used in the fields of natural language
                                                                                                       processing and speech recognition. The different applications are summed up in the table below:


Stanford University                                                                                6                                                                                    Winter 2019
```

### Page 7

```text
CS 230 – Deep Learning                                                                                                                                       Shervine Amidi & Afshine Amidi


     Type of RNN                    Illustration                            Example
                                                                                                                                                         T
                                                                                                                                             ∂L(T )   X ∂L(T )
                                                                                                                                                    =
        One-to-one                                                                                                                            ∂W         ∂W
                                                                                                                                                                    (t)
                                                                                                                                                      t=1
                                                                    Traditional neural network
       Tx = Ty = 1

                                                                                                         2.2    Handling long term dependencies
                                                                                                         r Commonly used activation functions – The most common activation functions used in
       One-to-many                                                                                       RNN modules are described below:
                                                                    Music generation
      Tx = 1, Ty > 1
                                                                                                                         Sigmoid                      Tanh                      RELU

                                                                                                                                  1                      ez − e−z
                                                                                                                      g(z) =                    g(z) =                     g(z) = max(0,z)
                                                                                                                               1 + e−z                   ez + e−z
       Many-to-one
                                                                    Sentiment classification
      Tx > 1, Ty = 1




      Many-to-many
                                                                    Name entity recognition
         Tx = Ty                                                                                         r Vanishing/exploding gradient – The vanishing and exploding gradient phenomena are
                                                                                                         often encountered in the context of RNNs. The reason why they happen is that it is difficult
                                                                                                         to capture long term dependencies because of multiplicative gradient that can be exponentially
                                                                                                         decreasing/increasing with respect to the number of layers.

      Many-to-many                                                                                       r Gradient clipping – It is a technique used to cope with the exploding gradient problem
                                                                                                         sometimes encountered when performing backpropagation. By capping the maximum value for
                                                                    Machine translation                  the gradient, this phenomenon is controlled in practice.
         Tx 6= Ty




r Loss function – In the case of a recurrent neural network, the loss function L of all time
steps is defined based on the loss at every time step as follows:

                                             Ty
                                             X                                                           r Types of gates – In order to remedy the vanishing gradient problem, specific gates are used
                                    b,y) =
                                  L(y                b<t> ,y<t> )
                                                   L(y                                                   in some types of RNNs and usually have a well-defined purpose. They are usually noted Γ and
                                             t=1                                                         are equal to:

                                                                                                                                         Γ = σ(W x<t> + U a<t−1> + b)

r Backpropagation through time – Backpropagation is done at each point in time. At                       where W, U, b are coefficients specific to the gate and σ is the sigmoid function. The main ones
timestep T , the derivative of the loss L with respect to weight matrix W is expressed as follows:       are summed up in the table below:


Stanford University                                                                                  7                                                                                 Winter 2019
```

### Page 8

```text
CS 230 – Deep Learning                                                                                                                                     Shervine Amidi & Afshine Amidi


              Type of gate                       Role                        Used in                  2.3     Learning word representation
           Update gate Γu        How much past should matter now?          GRU, LSTM                  In this section, we note V the vocabulary and |V | its size.
          Relevance gate Γr          Drop previous information?            GRU, LSTM
              Forget gate Γf              Erase a cell or not?                LSTM                    2.3.1    Motivation and notations
              Output gate Γo        How much to reveal of a cell?             LSTM                    r Representation techniques – The two main ways of representing words are summed up in
                                                                                                      the table below:
r GRU/LSTM – Gated Recurrent Unit (GRU) and Long Short-Term Memory units (LSTM)
deal with the vanishing gradient problem encountered by traditional RNNs, with LSTM being                             1-hot representation                           Word embedding
a generalization of GRU. Below is a table summing up the characterizing equations of each
architecture:
                         Gated Recurrent Unit                Long Short-Term Memory
                                (GRU)                                (LSTM)

      c̃<t>          tanh(Wc [Γr ? a<t−1> ,x<t> ] + bc )    tanh(Wc [Γr ? a<t−1> ,x<t> ] + bc )

      c<t>            Γu ? c̃<t> + (1 − Γu ) ? c<t−1>            Γu ? c̃<t> + Γf ? c<t−1>

      a<t>                         c<t>                                   Γo ? c<t>

                                                                                                            - Noted ow                                      - Noted ew
                                                                                                            - Naive approach, no similarity information     - Takes into account words similarity


  Dependencies                                                                                        r Embedding matrix – For a given word w, the embedding matrix E is a matrix that maps
                                                                                                      its 1-hot representation ow to its embedding ew as follows:
                                                                                                                                                 ew = Eow

                                                                                                      Remark: learning the embedding matrix can be done using target/context likelihood models.

Remark: the sign ? denotes the element-wise multiplication between two vectors.
r Variants of RNNs – The table below sums up the other commonly used RNN architectures:               2.3.2    Word embeddings
                       Bidirectional                              Deep                                r Word2vec – Word2vec is a framework aimed at learning word embeddings by estimating the
                         (BRNN)                                                                       likelihood that a given word is surrounded by other words. Popular models include skip-gram,
                                                                 (DRNN)                               negative sampling and CBOW.




                                                                                                      r Skip-gram – The skip-gram word2vec model is a supervised learning task that learns word
                                                                                                      embeddings by assessing the likelihood of any given target word t happening with a context
                                                                                                      word c. By noting θt a parameter associated with t, the probability P (t|c) is given by:


Stanford University                                                                               8                                                                                  Winter 2019
```

### Page 9

```text
CS 230 – Deep Learning                                                                                                                                             Shervine Amidi & Afshine Amidi



                                                       exp(θtT ec )
                                    P (t|c) =
                                                     |V |
                                                     X
                                                            exp(θjT ec )
                                                     j=1

Remark: summing over the whole vocabulary in the denominator of the softmax part makes
this model computationally expensive. CBOW is another word2vec model using the surrounding
words to predict a given word.
r Negative sampling – It is a set of binary classifiers using logistic regressions that aim at         2.5    Language model
assessing how a given context and a given target words are likely to appear simultaneously, with
the models being trained on sets of k negative examples and 1 positive example. Given a context        r Overview – A language model aims at estimating the probability of a sentence P (y).
word c and a target word t, the prediction is expressed by:
                                                                                                       r n-gram model – This model is a naive approach aiming at quantifying the probability that
                                        P (y = 1|c,t) = σ(θtT ec )                                     an expression appears in a corpus by counting its number of appearance in the training data.
Remark: this method is less computationally expensive than the skip-gram model.                        r Perplexity – Language models are commonly assessed using the perplexity metric, also
                                                                                                       known as PP, which can be interpreted as the inverse probability of the dataset normalized by
r GloVe – The GloVe model, short for global vectors for word representation, is a word em-             the number of words T . The perplexity is such that the lower, the better and is defined as
bedding technique that uses a co-occurence matrix X where each Xi,j denotes the number of              follows:
times that a target i occurred with a context j. Its cost function J is as follows:                                                                                            ! T1
                                                                                                                                              T
                                 |V |
                                                                                                                                              Y                   1
                               1 X                                                                                                    PP =
                      J(θ) =       f (Xij )(θiT ej + bi + b0j − log(Xij ))2
                                                                                                                                                     P|V |        (t)    (t)
                                                                                                                                                                 yj · y
                               2                                                                                                              t=1        j=1
                                                                                                                                                                      bj
                                i,j=1
                                                                                                       Remark: PP is commonly used in t-SNE.
here f is a weighting function such that Xi,j = 0 =⇒ f (Xi,j ) = 0.
                                                                              (final)
Given the symmetry that e and θ play in this model, the final word embedding ew       is given
by:                                                                                                    2.6    Machine translation
                                           (final)          e w + θw                                   r Overview – A machine translation model is similar to a language model except it has an
                                          ew         =
                                                                2                                      encoder network placed before. For this reason, it is sometimes referred as a conditional language
                                                                                                       model. The goal is to find a sentence y such that:
Remark: the individual components of the learned word embeddings are not necessarily inter-
pretable.                                                                                                                          y=      arg max           P (y <1> ,...,y <Ty > |x)
                                                                                                                                        y <1> ,...,y <Ty >



2.4    Comparing words                                                                                 r Beam search – It is a heuristic search algorithm used in machine translation and speech
                                                                                                       recognition to find the likeliest sentence y given an input x.
r Cosine similarity – The cosine similarity between words w1 and w2 is expressed as follows:
                                                                                                          • Step 1: Find top B likely words y <1>
                                            w1 · w2
                            similarity =                  = cos(θ)
                                          ||w1 || ||w2 ||                                                 • Step 2: Compute conditional probabilities y <k> |x,y <1> ,...,y <k−1>

                                                                                                          • Step 3: Keep top B combinations x,y <1> ,...,y <k>
Remark: θ is the angle between words w1 and w2 .




r t-SNE – t-SNE (t-distributed Stochastic Neighbor Embedding) is a technique aimed at re-
ducing high-dimensional embeddings into a lower dimensional space. In practice, it is commonly
used to visualize word vectors in the 2D space.


Stanford University                                                                                9                                                                                     Winter 2019
```

### Page 10

```text
CS 230 – Deep Learning                                                                                                                                        Shervine Amidi & Afshine Amidi


Remark: if the beam width is set to 1, then this is equivalent to a naive greedy search.               Remark: the attention scores are commonly used in image captioning and machine translation.
r Beam width – The beam width B is a parameter for beam search. Large values of B yield
to better result but with slower performance and increased memory. Small values of B lead to
worse results but is less computationally intensive. A standard value for B is around 10.
r Length normalization – In order to improve numerical stability, beam search is usually ap-
plied on the following normalized objective, often called the normalized log-likelihood objective,
defined as:
                                         Ty
                                      1 X
                                                h                                     i
                    Objective =           log p(y <t> |x,y <1> , ..., y <t−1> )
                                     Tyα
                                         t=1


Remark: the parameter α can be seen as a softener, and its value is usually between 0.5 and 1.

r Error analysis – When obtaining a predicted translation y  b that is bad, one can wonder why         r Attention weight – The amount of attention that the output y <t> should pay to the
                                                                                                                     0                   0
we did not get a good translation y ∗ by performing the following error analysis:                      activation a<t > is given by α<t,t > computed as follows:
                                                                                                                                                                  0
                                                                                                                                            0           exp(e<t,t > )
                  Case           P (y ∗ |x) > P (y
                                                 b|x)               P (y ∗ |x) ⩽ P (y
                                                                                    b|x)                                               α<t,t > =
                                                                                                                                                       Tx
                                                                                                                                                                      00
                                                                                                                                                       X
              Root cause        Beam search faulty                       RNN faulty                                                                         exp(e<t,t > )
                                                              - Try different architecture                                                         t00 =1

               Remedies        Increase beam width            - Regularize                             Remark: computation complexity is quadratic with respect to Tx .
                                                              - Get more data

                                                                                                                                                   ?    ?     ?
r Bleu score – The bilingual evaluation understudy (bleu) score quantifies how good a machine
translation is by computing a similarity score based on n-gram precision. It is defined as follows:
                                                               n
                                                                     !
                                                            1X
                                 bleu score = exp              pk
                                                            n
                                                              k=1

where pn is the bleu score on n-gram only defined as follows:
                                           X
                                                     countclip (n-gram)
                                        n-gram∈y
                                pn =
                                                b
                                               X
                                                         count(n-gram)
                                          n-gram∈y   b
Remark: a brevity penalty may be applied to short predicted translations to prevent an artificially
inflated bleu score.


2.7    Attention
r Attention model – This model allows an RNN to pay attention to specific parts of the input
that is considered as being important, which improves the performance of the resulting model
                             0
in practice. By noting α<t,t > the amount of attention that the output y <t> should pay to the
               0
activation a <t  > and c <t> the context at time t, we have:
                                           0     0                           0
                               X                                    X
                      c<t> =         α<t,t > a<t >          with         α<t,t > = 1
                                t0                                  t0



Stanford University                                                                               10                                                                             Winter 2019
```

### Page 11

```text
CS 230 – Deep Learning                                                                                                                                       Shervine Amidi & Afshine Amidi


3     Deep Learning Tips and Tricks                                                                       3.2     Training a neural network

3.1     Data processing                                                                                   3.2.1    Definitions

r Data augmentation – Deep learning models usually need a lot of data to be properly trained.             r Epoch – In the context of training a model, epoch is a term used to refer to one iteration
It is often useful to get more data from the existing ones using data augmentation techniques.            where the model sees the whole training set to update its weights.
The main ones are summed up in the table below. More precisely, given the following input
                                                                                                          r Mini-batch gradient descent – During the training phase, updating weights is usually not
image, here are the techniques that we can apply:
                                                                                                          based on the whole training set at once due to computation complexities or one data point due
                                                                                                          to noise issues. Instead, the update step is done on mini-batches, where the number of data
        Original                  Flip                     Rotation             Random crop               points in a batch is a hyperparameter that we can tune.
                                                                                                          r Loss function – In order to quantify how a given model performs, the loss function L is
                                                                                                          usually used to evaluate to what extent the actual outputs y are correctly predicted by the
                                                                                                          model outputs z.
                                                                                                          r Cross-entropy loss – In the context of binary classification in neural networks, the cross-
                                                                                                          entropy loss L(z,y) is commonly used and is defined as follows:
                                                                                                                                                h                              i
                                                                                                                                    L(z,y) = − y log(z) + (1 − y) log(1 − z)

                                                                               - Random focus
                          - Flipped with respect     - Rotation with           on one part of
    - Image without       to an axis for which       a slight angle            the image
                                                                                                          3.2.2    Finding optimal weights
    any modification      the meaning of the         - Simulates incorrect     - Several random
                          image is preserved         horizon calibration       crops can be               r Backpropagation – Backpropagation is a method to update the weights in the neural network
                                                                               done in a row              by taking into account the actual output and the desired output. The derivative with respect
                                                                                                          to each weight w is computed using the chain rule.

       Color shift         Noise addition           Information loss          Contrast change




                                                                                                          Using this method, each weight is updated with the rule:
                                                                                                                                                            ∂L(z,y)
                                                                                                                                               w ←− w − α
    - Nuances of RGB                                                                                                                                          ∂w
                          - Addition of noise      - Parts of image          - Luminosity changes
    is slightly changed
                          - More tolerance to      ignored                   - Controls difference
    - Captures noise                                                                                      r Updating weights – In a neural network, weights are updated as follows:
                          quality variation of     - Mimics potential        in exposition due
    that can occur
                          inputs                   loss of parts of image    to time of day                  • Step 1: Take a batch of training data and perform forward propagation to compute the
    with light exposure
                                                                                                               loss.
                                                                                                             • Step 2: Backpropagate the loss to get the gradient of the loss with respect to each weight.
r Batch normalization – It is a step of hyperparameter γ, β that normalizes the batch {xi }.
                                                                                                             • Step 3: Use the gradients to update the weights of the network.
By noting µB , σB
                2 the mean and variance of that we want to correct to the batch, it is done as

follows:
                                              xi − µB
                                      xi ←− γ p       +β
                                                 2 +
                                                σB

It is usually done after a fully connected/convolutional layer and before a non-linearity layer and
aims at allowing higher learning rates and reducing the strong dependence on initialization.


Stanford University                                                                                  11                                                                                 Winter 2019
```

### Page 12

```text
CS 230 – Deep Learning                                                                                                                                          Shervine Amidi & Afshine Amidi


3.3     Parameter tuning                                                                                    Method                 Explanation                      Update of w                 Update of b
                                                                                                                          - Dampens oscillations
3.3.1     Weights initialization                                                                           Momentum       - Improvement to SGD                        w − αvdw                     b − αvdb
                                                                                                                          - 2 parameters to tune

r Xavier initialization – Instead of initializing the weights in a purely random manner, Xavier                           - Root Mean Square propagation
                                                                                                                                                                           dw                             db
initialization enables to have initial weights that take into account characteristics that are unique       RMSprop       - Speeds up learning algorithm             w − α√                   b ←− b − α √
to the architecture.                                                                                                      by controlling oscillations
                                                                                                                                                                           sdw                             sdb

r Transfer learning – Training a deep learning model requires a lot of data and more impor-                               - Adaptive Moment estimation
tantly a lot of time. It is often useful to take advantage of pre-trained weights on huge datasets                                                                            vdw                             vdb
                                                                                                               Adam       - Most popular method                     w − α√                  b ←− b − α √
that took days/weeks to train, and leverage it towards our use case. Depending on how much                                                                                   sdw +                          sdb + 
data we have at hand, here are the different ways to leverage this:                                                       - 4 parameters to tune


                                                                                                         Remark: other methods include Adadelta, Adagrad and SGD.

    Training size                   Illustration                          Explanation
                                                                                                         3.4    Regularization

                                                                                                         r Dropout – Dropout is a technique used in neural networks to prevent overfitting the training
                                                                   Freezes all layers,                   data by dropping out neurons with probability p > 0. It forces the model to avoid relying too
         Small
                                                                   trains weights on softmax             much on particular sets of features.




                                                                   Freezes most layers,
        Medium                                                     trains weights on last
                                                                   layers and softmax

                                                                                                         Remark: most deep learning frameworks parametrize dropout through the ’keep’ parameter 1−p.
                                                                   Trains weights on layers              r Weight regularization – In order to make sure that the weights are not too large and that
        Large                                                      and softmax by initializing           the model is not overfitting the training set, regularization techniques are usually performed on
                                                                   weights on pre-trained ones           the model weights. The main ones are summed up in the table below:


                                                                                                                      LASSO                          Ridge                                 Elastic Net
                                                                                                          - Shrinks coefficients to 0                                         Tradeoff between variable
                                                                                                                                           Makes coefficients smaller
                                                                                                          - Good for variable selection                                       selection and small coefficients




3.3.2     Optimizing convergence

r Learning rate – The learning rate, often noted α or sometimes η, indicates at which pace the
weights get updated. It can be fixed or adaptively changed. The current most popular method
is called Adam, which is a method that adapts the learning rate.

r Adaptive learning rates – Letting the learning rate vary when training a model can reduce
                                                                                                                                                                                       h                          i
the training time and improve the numerical optimal solution. While Adam optimizer is the                          ... + λ||θ||1                   ... + λ||θ||22             ... + λ (1 − α)||θ||1 + α||θ||22
most commonly used technique, others can also be useful. They are summed up in the table                               λ∈R                             λ∈R
below:                                                                                                                                                                                     λ ∈ R,α ∈ [0,1]


Stanford University                                                                                 12                                                                                            Winter 2019
```

### Page 13

```text
CS 230 – Deep Learning                                                                                       Shervine Amidi & Afshine Amidi


r Early stopping – This regularization technique stops the training process as soon as the
validation loss reaches a plateau or starts to increase.




3.5    Good practices
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




Stanford University                                                                                     13                     Winter 2019
```
