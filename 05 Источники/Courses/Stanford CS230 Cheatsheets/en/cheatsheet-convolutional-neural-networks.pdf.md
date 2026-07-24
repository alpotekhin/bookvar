---
title: "Convolutional Neural Networks"
type: external-resource
status: imported-source
language: original
source_kind: slides
source_commit: 4653bc01297b269edb19e844b01127ba13de59df
---

> [!note] Original source material
> This page preserves [`en/cheatsheet-convolutional-neural-networks.pdf`](https://github.com/afshinea/stanford-cs-230-deep-learning/blob/4653bc01297b269edb19e844b01127ba13de59df/en/cheatsheet-convolutional-neural-networks.pdf) from
> *Stanford CS230 Deep Learning Cheatsheets* at commit `4653bc01297b269edb19e844b01127ba13de59df`. License:
> [MIT](https://github.com/afshinea/stanford-cs-230-deep-learning/blob/4653bc01297b269edb19e844b01127ba13de59df/LICENSE). Bookvar changed only
> publication markup, link paths, and characters required for safe rendering.



The embedded PDF is the primary visual document. The page-separated text below was extracted mechanically for search and has not been rewritten.

<iframe class="source-pdf" src="https://github.com/afshinea/stanford-cs-230-deep-learning/blob/4653bc01297b269edb19e844b01127ba13de59df/en/cheatsheet-convolutional-neural-networks.pdf?raw=1" title="Convolutional Neural Networks" loading="lazy"></iframe>

## Extracted document text

### Page 1

```text
CS 230 – Deep Learning                                                                                                                                       https://stanford.edu/~shervine


                                                                                                                                      Max pooling                         Average pooling
  VIP Cheatsheet: Convolutional Neural Networks                                                              Purpose
                                                                                                                            Each pooling operation selects the     Each pooling operation averages
                                                                                                                            maximum value of the current view      the values of the current view



                       Afshine Amidi and Shervine Amidi
                                                                                                           Illustration

                                   November 26, 2018

                                                                                                                            - Preserves detected features          - Downsamples feature map
                                                                                                            Comments
                                                                                                                            - Most commonly used                   - Used in LeNet
Overview

r Architecture of a traditional CNN – Convolutional neural networks, also known as CNNs,                r Fully Connected (FC) – The fully connected layer (FC) operates on a flattened input where
are a specific type of neural networks that are generally composed of the following layers:             each input is connected to all neurons. If present, FC layers are usually found towards the end
                                                                                                        of CNN architectures and can be used to optimize objectives such as class scores.




The convolution layer and the pooling layer can be fine-tuned with respect to hyperparameters
that are described in the next sections.
                                                                                                        Filter hyperparameters
                                                                                                        The convolution layer contains filters for which it is important to know the meaning behind its
Types of layer                                                                                          hyperparameters.

r Convolutional layer (CONV) – The convolution layer (CONV) uses filters that perform                   r Dimensions of a filter – A filter of size F × F applied to an input containing C channels is
convolution operations as it is scanning the input I with respect to its dimensions. Its hyperpa-       a F × F × C volume that performs convolutions on an input of size I × I × C and produces an
rameters include the filter size F and stride S. The resulting output O is called feature map or        output feature map (also called activation map) of size O × O × 1.
activation map.




                                                                                                        Remark: the application of K filters of size F × F results in an output feature map of size
                                                                                                        O × O × K.
Remark: the convolution step can be generalized to the 1D and 3D cases as well.                         r Stride – For a convolutional or a pooling operation, the stride S denotes the number of pixels
                                                                                                        by which the window moves after each operation.
r Pooling (POOL) – The pooling layer (POOL) is a downsampling operation, typically applied
after a convolution layer, which does some spatial invariance. In particular, max and average
pooling are special kinds of pooling where the maximum and average value is taken, respectively.


Stanford University                                                                                 1                                                                                 Winter 2019
```

### Page 2

```text
CS 230 – Deep Learning                                                                                                                                              https://stanford.edu/~shervine


r Zero-padding – Zero-padding denotes the process of adding P zeroes to each side of the                                               CONV                         POOL                      FC
boundaries of the input. This value can either be manually specified or automatically set through
one of the three modes detailed below:

                        Valid                           Same                          Full                    Illustration
                                                    j      I e−I+F −S
                                                                        k
                                                        Sd S
                                         Pstart =              2              Pstart ∈ [​[0,F − 1]​]
     Value              P =0                        l      I e−I+F −S
                                                                        m
                                                        Sd S
                                         Pend =                                  Pend = F − 1                  Input size             I ×I ×C                      I ×I ×C                    Nin
                                                               2
                                                                                                              Output size            O×O×K                        O×O×C                      Nout
                                                                                                              Number of
                                                                                                                                (F × F × C + 1) · K                       0            (Nin + 1) × Nout
                                                                                                              parameters
 Illustration
                                                                                                                               - One bias parameter                                  - Input is flattened
                                                                                                                               per filter                 - Pooling operation        - One bias parameter
                                                                                                                                                          done channel-wise          per neuron
                                                                                                                Remarks        - In most cases, S < F
                                                                                                                                                                                     - The number of FC
                                                                            - Maximum padding                                  - A common choice
                  - No padding          - Padding such that feature                                                                                       - In most cases, S = F     neurons is free of
                                                                            such that end                                      for K is 2C                                           structural constraints
                                                               l m
                                        map size has size          I
                                                                            convolutions are
                  - Drops last                                     S
   Purpose                                                                  applied on the limits
                  convolution if        - Output size is
                                                                            of the input
                  dimensions do not     mathematically convenient
                  match                                                     - Filter ’sees’ the input       r Receptive field – The receptive field at layer k is the area denoted Rk × Rk of the input
                                        - Also called ’half’ padding                                        that each pixel of the k-th activation map can ’see’. By calling Fj the filter size of layer j and
                                                                            end-to-end
                                                                                                            Si the stride value of layer i and with the convention S0 = 1, the receptive field at layer k can
                                                                                                            be computed with the formula:

Tuning hyperparameters                                                                                                                                    k               j−1
                                                                                                                                                         X                Y
r Parameter compatibility in convolution layer – By noting I the length of the input                                                          Rk = 1 +         (Fj − 1)         Si
volume size, F the length of the filter, P the amount of zero padding, S the stride, then the                                                            j=1              i=0
output size O of the feature map along that dimension is given by:

                                                                                                            In the example below, we have F1 = F2 = 3 and S1 = S2 = 1, which gives R2 = 1+2 · 1+2 · 1 =
                                      I − F + Pstart + Pend                                                 5.
                                 O=                         +1
                                               S




Remark: often times, Pstart = Pend , P , in which case we can replace Pstart + Pend by 2P in                Commonly used activation functions
the formula above.
r Understanding the complexity of the model – In order to assess the complexity of a                        r Rectified Linear Unit – The rectified linear unit layer (ReLU) is an activation function g
model, it is often useful to determine the number of parameters that its architecture will have.            that is used on all elements of the volume. It aims at introducing non-linearities to the network.
In a given layer of a convolutional neural network, it is done as follows:                                  Its variants are summarized in the table below:


Stanford University                                                                                     2                                                                                   Winter 2019
```

### Page 3

```text
CS 230 – Deep Learning                                                                                                                                         https://stanford.edu/~shervine


                ReLU                        Leaky ReLU                       ELU                                     Bounding box detection                       Landmark detection
                                          g(z) = max(z,z)          g(z) = max(α(ez − 1),z)                                                              - Detects a shape or characteristics of
           g(z) = max(0,z)                                                                                      Detects the part of the image where
                                             with   1                   with α  1                                                                     an object (e.g. eyes)
                                                                                                                the object is located
                                                                                                                                                         - More granular




      Non-linearity complexities     Addresses dying ReLU
                                                                    Differentiable everywhere
      biologically interpretable     issue for negative values
                                                                                                                Box of center (bx ,by ), height bh
                                                                                                                                                         Reference points (l1x ,l1y ), ...,(lnx ,lny )
r Softmax – The softmax step can be seen as a generalized logistic function that takes as input                 and width bw
a vector of scores x ∈ Rn and outputs a vector of output probability p ∈ Rn through a softmax
function at the end of the architecture. It is defined as follows:

                                    p1                                                                r Intersection over Union – Intersection over Union, also known as IoU, is a function that
                                     ..                       exi                                       quantifies how correctly positioned a predicted bounding box Bp is over the actual bounding
                              p=             where    pi =                                              box Ba . It is defined as:
                                      .                      n
                                    pn                       X
                                                                   exj                                                                                        Bp ∩ Ba
                                                             j=1
                                                                                                                                             IoU(Bp ,Ba ) =
                                                                                                                                                              Bp ∪ Ba


Object detection
r Types of models – There are 3 main types of object recognition algorithms, for which the
nature of what is predicted is different. They are described in the table below:

                                   Classification
  Image classification                                                     Detection
                                   w. localization




                                                                                                        Remark: we always have IoU ∈ [0,1]. By convention, a predicted bounding box Bp is considered
                                                                                                        as being reasonably good if IoU(Bp ,Ba ) ⩾ 0.5.

                                                                                                        r Anchor boxes – Anchor boxing is a technique used to predict overlapping bounding boxes.
                                                                                                        In practice, the network is allowed to predict more than one box simultaneously, where each box
  - Classifies a picture     - Detects object in a picture    - Detects up to several objects           prediction is constrained to have a given set of geometrical properties. For instance, the first
                                                              in a picture                              prediction can potentially be a rectangular box of a given form, while the second will be another
                             - Predicts probability of
                                                                                                        rectangular box of a different geometrical form.
  - Predicts probability     object and where it is           - Predicts probabilities of objects
  of object                  located                          and where they are located                r Non-max suppression – The non-max suppression technique aims at removing duplicate
                                                                                                        overlapping bounding boxes of a same object by selecting the most representative ones. After
  Traditional CNN            Simplified YOLO, R-CNN           YOLO, R-CNN                               having removed all boxes having a probability prediction lower than 0.6, the following steps are
                                                                                                        repeated while there are boxes remaining:

r Detection – In the context of object detection, different methods are used depending on                  • Step 1: Pick the box with the largest prediction probability.
whether we just want to locate the object or detect a more complex shape in the image. The
two main ones are summed up in the table below:                                                            • Step 2: Discard any box having an IoU ⩾ 0.5 with the previous box.


Stanford University                                                                                 3                                                                                       Winter 2019
```

### Page 4

```text
CS 230 – Deep Learning                                                                                                                                         https://stanford.edu/~shervine


                                                                                                         Face verification and recognition

                                                                                                         r Types of models – Two main types of model are summed up in table below:


                                                                                                                      Face verification                           Face recognition
                                                                                                                 - Is this the correct person?      - Is this one of the K persons in the database?
                                                                                                                 - One-to-one lookup                - One-to-many lookup

r YOLO – You Only Look Once (YOLO) is an object detection algorithm that performs the
following steps:

   • Step 1: Divide the input image into a G × G grid.

   • Step 2: For each grid cell, run a CNN that predicts y of the following form:
                                                                    T
                       y = pc ,bx ,by ,bh ,bw ,c1 ,c2 ,...,cp ,...        ∈ RG×G×k×(5+p)
                                |           {z              }
                                     repeated k times


      where pc is the probability of detecting an object, bx ,by ,bh ,bw are the properties of the
      detected bouding box, c1 ,...,cp is a one-hot representation of which of the p classes were        r One Shot Learning – One Shot Learning is a face verification algorithm that uses a limited
      detected, and k is the number of anchor boxes.                                                     training set to learn a similarity function that quantifies how different two given images are. The
                                                                                                         similarity function applied to two images is often noted d(image 1, image 2).
   • Step 3: Run the non-max suppression algorithm to remove any potential duplicate over-
     lapping bounding boxes.                                                                             r Siamese Network – Siamese Networks aim at learning how to encode images to then quantify
                                                                                                         how different two images are. For a given input image x(i) , the encoded output is often noted
                                                                                                         as f (x(i) ).

                                                                                                         r Triplet loss – The triplet loss ` is a loss function computed on the embedding representation
                                                                                                         of a triplet of images A (anchor), P (positive) and N (negative). The anchor and the positive
                                                                                                         example belong to a same class, while the negative example to another one. By calling α ∈ R+
                                                                                                         the margin parameter, this loss is defined as follows:

                                                                                                                                    `(A,P,N ) = max (d(A,P ) − d(A,N ) + α,0)



Remark: when pc = 0, then the network does not detect any object. In that case, the corre-
sponding predictions bx , ..., cp have to be ignored.

r R-CNN – Region with Convolutional Neural Networks (R-CNN) is an object detection algo-
rithm that first segments the image to find potential relevant bounding boxes and then run the
detection algorithm to find most probable objects in those bounding boxes.




                                                                                                         Neural style transfer

Remark: although the original algorithm is computationally expensive and slow, newer archi-              r Motivation – The goal of neural style transfer is to generate an image G based on a given
tectures enabled the algorithm to run faster, such as Fast R-CNN and Faster R-CNN.                       content C and a given style S.


Stanford University                                                                                  4                                                                                    Winter 2019
```

### Page 5

```text
CS 230 – Deep Learning                                                                                                                                                     https://stanford.edu/~shervine




r Activation – In a given layer l, the activation is noted a[l] and is of dimensions nH × nw × nc
r Content cost function – The content cost function Jcontent (C,G) is used to determine how                       Remark: use cases using variants of GANs include text to image, music generation and syn-
the generated image G differs from the original content image C. It is defined as follows:                        thesis.

                                                   1 [l](C)                                                       r ResNet – The Residual Network architecture (also called ResNet) uses residual blocks with a
                               Jcontent (C,G) =      ||a    − a[l](G) ||2                                         high number of layers meant to decrease the training error. The residual block has the following
                                                   2                                                              characterizing equation:
                                                                                                                                                      a[l+2] = g(a[l] + z [l+2] )
r Style matrix – The style matrix G[l] of a given layer l is a Gram matrix where each of its
           [l]
elements Gkk0 quantifies how correlated the channels k and k0 are. It is defined with respect to
                                                                                                                  r Inception Network – This architecture uses inception modules and aims at giving a try
activations a[l] as follows:                                                                                      at different convolutions in order to increase its performance. In particular, it uses the 1 × 1
                                                [l]
                                                                                                                  convolution trick to lower the burden of computation.
                                               n      [l]
                                                 nw
                                               H X
                                       [l]
                                               X    [l]           [l]
                                      Gkk0 =                aijk aijk0                                                                                        ?   ?    ?
                                               i=1 j=1


Remark: the style matrix for the style image and the generated image are noted G[l](S) and
G[l](G) respectively.
r Style cost function – The style cost function Jstyle (S,G) is used to determine how the
generated image G differs from the style S. It is defined as follows:

                                                                             nc 
                       1                                        1            X                           2
   [l]                                                                            [l](S)        [l](G)
 Jstyle (S,G) =                 ||G[l](S) − G[l](G) ||2F =                            Gkk0   − Gkk0
                  (2nH nw nc )2                            (2nH nw nc )2
                                                                            k,k0 =1




r Overall cost function – The overall cost function is defined as being a combination of the
content and style cost functions, weighted by parameters α,β, as follows:

                               J(G) = αJcontent (C,G) + βJstyle (S,G)

Remark: a higher value of α will make the model care more about the content while a higher
value of β will make it care more about the style.


Architectures using computational tricks
r Generative Adversarial Network – Generative adversarial networks, also known as GANs,
are composed of a generative and a discriminative model, where the generative model aims at
generating the most truthful output that will be fed into the discriminative which aims at
differentiating the generated and true image.


Stanford University                                                                                           5                                                                                 Winter 2019
```
