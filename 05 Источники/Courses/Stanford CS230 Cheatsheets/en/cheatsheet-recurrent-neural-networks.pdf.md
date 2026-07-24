---
title: "Recurrent Neural Networks"
type: external-resource
status: imported-source
language: original
source_kind: slides
source_commit: 4653bc01297b269edb19e844b01127ba13de59df
---

> [!note] Original source material
> This page preserves [`en/cheatsheet-recurrent-neural-networks.pdf`](https://github.com/afshinea/stanford-cs-230-deep-learning/blob/4653bc01297b269edb19e844b01127ba13de59df/en/cheatsheet-recurrent-neural-networks.pdf) from
> *Stanford CS230 Deep Learning Cheatsheets* at commit `4653bc01297b269edb19e844b01127ba13de59df`. License:
> [MIT](https://github.com/afshinea/stanford-cs-230-deep-learning/blob/4653bc01297b269edb19e844b01127ba13de59df/LICENSE). Bookvar changed only
> publication markup, link paths, and characters required for safe rendering.



The embedded PDF is the primary visual document. The page-separated text below was extracted mechanically for search and has not been rewritten.

<iframe class="source-pdf" src="https://github.com/afshinea/stanford-cs-230-deep-learning/blob/4653bc01297b269edb19e844b01127ba13de59df/en/cheatsheet-recurrent-neural-networks.pdf?raw=1" title="Recurrent Neural Networks" loading="lazy"></iframe>

## Extracted document text

### Page 1

```text
CS 230 – Deep Learning                                                                                                                                          https://stanford.edu/~shervine


                                                                                                                                Advantages                                  Drawbacks
     VIP Cheatsheet: Recurrent Neural Networks                                                                - Possibility of processing input of any length   - Computation being slow
                                                                                                              - Model size not increasing with size of input    - Difficulty of accessing information
                                                                                                              - Computation takes into account                  from a long time ago
                       Afshine Amidi and Shervine Amidi                                                       historical information                            - Cannot consider any future input
                                                                                                              - Weights are shared across time                  for the current state

                                   November 26, 2018
                                                                                                         r Applications of RNNs – RNN models are mostly used in the fields of natural language
                                                                                                         processing and speech recognition. The different applications are summed up in the table below:


Overview                                                                                                      Type of RNN                    Illustration                        Example

r Architecture of a traditional RNN – Recurrent neural networks, also known as RNNs,
are a class of neural networks that allow previous outputs to be used as inputs while having                    One-to-one
hidden states. They are typically as follows:
                                                                                                                                                                        Traditional neural network
                                                                                                                Tx = Ty = 1




                                                                                                               One-to-many
                                                                                                                                                                        Music generation
                                                                                                               Tx = 1, Ty > 1

For each timestep t, the activation a<t> and the output y <t> are expressed as follows:

                                                                                                                Many-to-one
        a<t> = g1 (Waa a<t−1> + Wax x<t> + ba )       and    y <t> = g2 (Wya a<t> + by )                                                                                Sentiment classification
                                                                                                               Tx > 1, Ty = 1
where Wax , Waa , Wya , ba , by are coefficients that are shared temporally and g1 , g2 activation
functions


                                                                                                               Many-to-many
                                                                                                                                                                        Name entity recognition
                                                                                                                  Tx = Ty




                                                                                                               Many-to-many
                                                                                                                                                                        Machine translation
                                                                                                                  Tx 6= Ty




The pros and cons of a typical RNN architecture are summed up in the table below:                        r Loss function – In the case of a recurrent neural network, the loss function L of all time


Stanford University                                                                                  1                                                                                  Winter 2019
```

### Page 2

```text
CS 230 – Deep Learning                                                                                                                                            https://stanford.edu/~shervine


steps is defined based on the loss at every time step as follows:                                        r Types of gates – In order to remedy the vanishing gradient problem, specific gates are used
                                                                                                         in some types of RNNs and usually have a well-defined purpose. They are usually noted Γ and
                                             Ty
                                             X                                                           are equal to:
                                    b,y) =
                                  L(y                b<t> ,y<t> )
                                                   L(y
                                             t=1
                                                                                                                                           Γ = σ(W x<t> + U a<t−1> + b)

r Backpropagation through time – Backpropagation is done at each point in time. At
timestep T , the derivative of the loss L with respect to weight matrix W is expressed as follows:
                                                                                                         where W, U, b are coefficients specific to the gate and σ is the sigmoid function. The main ones
                                                                                                         are summed up in the table below:
                                                 T
                                     ∂L(T )   X ∂L(T )
                                            =
                                      ∂W         ∂W
                                               t=1            (t)
                                                                                                                        Type of gate                       Role                       Used in
                                                                                                                    Update gate Γu         How much past should matter now?        GRU, LSTM
                                                                                                                   Relevance gate Γr           Drop previous information?          GRU, LSTM
Handling long term dependencies
                                                                                                                        Forget gate Γf              Erase a cell or not?               LSTM
r Commonly used activation functions – The most common activation functions used in
RNN modules are described below:                                                                                        Output gate Γo        How much to reveal of a cell?            LSTM


                 Sigmoid                      Tanh                      RELU
                                                                                                         r GRU/LSTM – Gated Recurrent Unit (GRU) and Long Short-Term Memory units (LSTM)
                         1                         ez − e−z
             g(z) =                     g(z) =                      g(z) = max(0,z)                      deal with the vanishing gradient problem encountered by traditional RNNs, with LSTM being
                      1 + e−z                      ez + e−z                                              a generalization of GRU. Below is a table summing up the characterizing equations of each
                                                                                                         architecture:



                                                                                                                                   Gated Recurrent Unit                Long Short-Term Memory
                                                                                                                                          (GRU)                                (LSTM)

                                                                                                                c̃<t>          tanh(Wc [Γr ? a<t−1> ,x<t> ] + bc )    tanh(Wc [Γr ? a<t−1> ,x<t> ] + bc )

                                                                                                                c<t>            Γu ? c̃<t> + (1 − Γu ) ? c<t−1>            Γu ? c̃<t> + Γf ? c<t−1>

r Vanishing/exploding gradient – The vanishing and exploding gradient phenomena are                            a<t>                          c<t>                                 Γo ? c<t>
often encountered in the context of RNNs. The reason why they happen is that it is difficult
to capture long term dependencies because of multiplicative gradient that can be exponentially
decreasing/increasing with respect to the number of layers.

r Gradient clipping – It is a technique used to cope with the exploding gradient problem
sometimes encountered when performing backpropagation. By capping the maximum value for                    Dependencies
the gradient, this phenomenon is controlled in practice.




                                                                                                         Remark: the sign ? denotes the element-wise multiplication between two vectors.

                                                                                                         r Variants of RNNs – The table below sums up the other commonly used RNN architectures:


Stanford University                                                                                  2                                                                                    Winter 2019
```

### Page 3

```text
CS 230 – Deep Learning                                                                                                                                             https://stanford.edu/~shervine


                      Bidirectional                             Deep
                        (BRNN)                                 (DRNN)




                                                                                                   r Skip-gram – The skip-gram word2vec model is a supervised learning task that learns word
                                                                                                   embeddings by assessing the likelihood of any given target word t happening with a context
                                                                                                   word c. By noting θt a parameter associated with t, the probability P (t|c) is given by:

                                                                                                                                                          exp(θtT ec )
                                                                                                                                       P (t|c) =
                                                                                                                                                        |V |
                                                                                                                                                        X
                                                                                                                                                               exp(θjT ec )
                                                                                                                                                        j=1
Learning word representation
                                                                                                   Remark: summing over the whole vocabulary in the denominator of the softmax part makes
In this section, we note V the vocabulary and |V | its size.                                       this model computationally expensive. CBOW is another word2vec model using the surrounding
r Representation techniques – The two main ways of representing words are summed up in             words to predict a given word.
the table below:
                                                                                                   r Negative sampling – It is a set of binary classifiers using logistic regressions that aim at
                                                                                                   assessing how a given context and a given target words are likely to appear simultaneously, with
                                                                                                   the models being trained on sets of k negative examples and 1 positive example. Given a context
                1-hot representation                           Word embedding                      word c and a target word t, the prediction is expressed by:

                                                                                                                                           P (y = 1|c,t) = σ(θtT ec )

                                                                                                   Remark: this method is less computationally expensive than the skip-gram model.

                                                                                                   r GloVe – The GloVe model, short for global vectors for word representation, is a word em-
                                                                                                   bedding technique that uses a co-occurence matrix X where each Xi,j denotes the number of
                                                                                                   times that a target i occurred with a context j. Its cost function J is as follows:

                                                                                                                                    |V |
                                                                                                                                  1 X
     - Noted ow                                       - Noted ew                                                         J(θ) =       f (Xij )(θiT ej + bi + b0j − log(Xij ))2
                                                                                                                                  2
     - Naive approach, no similarity information      - Takes into account words similarity                                        i,j=1


                                                                                                   here f is a weighting function such that Xi,j = 0 =⇒ f (Xi,j ) = 0.
r Embedding matrix – For a given word w, the embedding matrix E is a matrix that maps              Given the symmetry that e and θ play in this model, the final word embedding ew
                                                                                                                                                                                       (final)
                                                                                                                                                                                                 is given
its 1-hot representation ow to its embedding ew as follows:                                        by:
                                           ew = Eow
                                                                                                                                              (final)          ew + θw
                                                                                                                                             ew         =
                                                                                                                                                                  2
Remark: learning the embedding matrix can be done using target/context likelihood models.

r Word2vec – Word2vec is a framework aimed at learning word embeddings by estimating the           Remark: the individual components of the learned word embeddings are not necessarily inter-
likelihood that a given word is surrounded by other words. Popular models include skip-gram,       pretable.
negative sampling and CBOW.


Stanford University                                                                            3                                                                                    Winter 2019
```

### Page 4

```text
CS 230 – Deep Learning                                                                                                                                           https://stanford.edu/~shervine


Comparing words                                                                                          r Beam search – It is a heuristic search algorithm used in machine translation and speech
                                                                                                         recognition to find the likeliest sentence y given an input x.
r Cosine similarity – The cosine similarity between words w1 and w2 is expressed as follows:
                                            w1 · w2                                                         • Step 1: Find top B likely words y <1>
                            similarity =                  = cos(θ)
                                          ||w1 || ||w2 ||
                                                                                                            • Step 2: Compute conditional probabilities y <k> |x,y <1> ,...,y <k−1>

Remark: θ is the angle between words w1 and w2 .                                                            • Step 3: Keep top B combinations x,y <1> ,...,y <k>




r t-SNE – t-SNE (t-distributed Stochastic Neighbor Embedding) is a technique aimed at re-
ducing high-dimensional embeddings into a lower dimensional space. In practice, it is commonly
used to visualize word vectors in the 2D space.

                                                                                                         Remark: if the beam width is set to 1, then this is equivalent to a naive greedy search.
                                                                                                         r Beam width – The beam width B is a parameter for beam search. Large values of B yield
                                                                                                         to better result but with slower performance and increased memory. Small values of B lead to
                                                                                                         worse results but is less computationally intensive. A standard value for B is around 10.
                                                                                                         r Length normalization – In order to improve numerical stability, beam search is usually ap-
                                                                                                         plied on the following normalized objective, often called the normalized log-likelihood objective,
                                                                                                         defined as:
                                                                                                                                                 Ty
                                                                                                                                            1 X
                                                                                                                                                        h                                i
Language model                                                                                                               Objective =        log p(y <t> |x,y <1> , ..., y <t−1> )
                                                                                                                                           Tyα
                                                                                                                                                t=1
r Overview – A language model aims at estimating the probability of a sentence P (y).
r n-gram model – This model is a naive approach aiming at quantifying the probability that               Remark: the parameter α can be seen as a softener, and its value is usually between 0.5 and 1.
an expression appears in a corpus by counting its number of appearance in the training data.
                                                                                                         r Error analysis – When obtaining a predicted translation y  b that is bad, one can wonder why
r Perplexity – Language models are commonly assessed using the perplexity metric, also                   we did not get a good translation y ∗ by performing the following error analysis:
known as PP, which can be interpreted as the inverse probability of the dataset normalized by
the number of words T . The perplexity is such that the lower, the better and is defined as
follows:                                                                                                                   Case           P (y ∗ |x) > P (y
                                                                                                                                                          b|x)          P (y ∗ |x) ⩽ P (y
                                                                                                                                                                                        b|x)
                                                                        ! T1
                                        T                                                                              Root cause        Beam search faulty                RNN faulty
                                       Y                   1
                               PP =           P|V |                                                                                                               - Try different architecture
                                                           (t)    (t)
                                                          yj · y
                                                                                                                                        Increase beam width       - Regularize
                                       t=1        j=1
                                                               bj                                                       Remedies
                                                                                                                                                                  - Get more data
Remark: PP is commonly used in t-SNE.

                                                                                                         r Bleu score – The bilingual evaluation understudy (bleu) score quantifies how good a machine
Machine translation                                                                                      translation is by computing a similarity score based on n-gram precision. It is defined as follows:
r Overview – A machine translation model is similar to a language model except it has an                                                                          n
                                                                                                                                                                         !
encoder network placed before. For this reason, it is sometimes referred as a conditional language                                                               1X
                                                                                                                                          bleu score = exp          pk
model. The goal is to find a sentence y such that:                                                                                                               n
                                                                                                                                                                  k=1
                            y=      arg max           P (y <1> ,...,y <Ty > |x)
                                 y <1> ,...,y <Ty >                                                      where pn is the bleu score on n-gram only defined as follows:


Stanford University                                                                                  4                                                                                         Winter 2019
```

### Page 5

```text
CS 230 – Deep Learning                                                                                    https://stanford.edu/~shervine



                                             X
                                                         countclip (n-gram)
                                        n-gram∈y
                                pn =
                                                  b
                                                 X
                                                             count(n-gram)
                                          n-gram∈y       b
Remark: a brevity penalty may be applied to short predicted translations to prevent an artificially
inflated bleu score.


Attention
r Attention model – This model allows an RNN to pay attention to specific parts of the input
that is considered as being important, which improves the performance of the resulting model
                              0
in practice. By noting α<t,t > the amount of attention that the output y <t> should pay to the
                0
activation a <t   > and c <t> the context at time t, we have:
                                             0       0                                   0
                               X                                           X
                      c<t> =          α<t,t > a<t >             with                 α<t,t > = 1
                                t0                                              t0

Remark: the attention scores are commonly used in image captioning and machine translation.




r Attention weight – The amount of attention that the output y <t> should pay to the
              0                   0
activation a<t > is given by α<t,t > computed as follows:
                                                                       0
                                         0               exp(e<t,t > )
                                     α<t,t > =
                                                     Tx
                                                                           00
                                                     X
                                                              exp(e<t,t > )
                                                  t00 =1

Remark: computation complexity is quadratic with respect to Tx .

                                                 ?        ?     ?




Stanford University                                                                                   5                    Winter 2019
```
