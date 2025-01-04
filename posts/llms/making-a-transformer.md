---
title: "LLMs Part 3: Making a Transformer block"
author: "Rahul Dubey"
date: "2024-12-28"
categories: [ml, deep-learning, llm]
---
### A Transformer block
In part 2 we saw the Multi-head attention module which combines multiple attention heads and computes them in parallel. Though that is a key part of what makes a Transformer, there are other key components which are neat inventions or tricks from the past re-used to make Transformers as effective as possible. Note here that I am focusing on GPT-like decoder architecture.

In addition to Multi-head attention, we have 3 other key pieces:
1. Layer Normalization
2. Feedforward network with GeLU
3. Skip connections

Let's go over them 1 by 1 and in the end assemble them into the Transformer block.

#### Layer Normalization
Layer normalization is quite literal in the sense it normalizes a layer in a neural network. When we say normalize a layer, we mean normalize the activations of the layer. Imagine a layer y = f(x), the output y here needs to be normalized so that it has mean of 0 and variance of 1. We can imagine that we have a batch of y, let's call it Y. Layer normalization normalizes each y_i in Y independently to have mean of 0 and variance of 1. Note that layer normalization normalizes across the feature dimension so that all features in a single y_i are used to compute mean and variance, which is then used to normalize y_i.

The primary reason to do this is to stabilize training and avoid exploding or vanishing gradients because we know neural networks like the activations to be in a nice small range but not-too-small or not-too-big! 

##### Layer Normalization v/s Batch Normalization
One might wonder, don't we have a batch normalization technique? Why don't we use that instead of creating a new normalization technique? First, as the name suggests, batch normalization normalizes each input in the entire batch to have mean of 0 and variance of 1, whereas layer normalization normalizes each input independently. Say, we have a batch Y of y_i, batch size=N. Each y_i has say J features, then batch normalization gathers each feature f_j from the entire batch, giving us N values of feature f_j. We then compute mean/variance and normalize f_j. There is a subtle difference in this process v/s layer normalization.

When we have to use varying batch sizes (distributed training) or small batch sizes (in case of LLMs which are very big, sometimes we cannot fit a large number of examples in a batch due to resource constraints), layer normalization provides a better performance because on small batch sizes. A good heuristic/practice in industry is to use Batch normalization for CNNs and when one can fit a large number of samples in a batch and use Layer normalization for RNNs/Transformers where batch size may vary or batch size is small.

##### Scale and Shift
Another thing to mention are the scale and shift parameters. Scale and shift are learnable parameters that change the output of layer normalization. Say N(x) is output of a layer once it is normalized , γ=scale and β=shift, then the final output of layer normalization L(x) = γ * N(x) + β. 

Scale is initialized as 1 and Shift is initialized as 0, but it may change during the learning process. It's a fair question to ask: if we are normalizing the activations, why screw up that normalization by scaling and shifting? 

The answer is that the new parametrization can represent the same family of functions of the input as the old parametrization, but the new parametrization has different learning dynamics. In the old parametrization, the mean of X was determined by a complicated interaction between the parameters in the layers below X. In the new parametrization, the mean of γ*N(x)+β is determined solely by β. The new parametrization is much easier to learn with gradient descent. Read more in [this chapter](https://www.deeplearningbook.org/contents/optimization.html) of the Deep Learning book.

##### Where to normalize?
In the original transformer model, layer normalization was applied after the self-attention and feed forward networks, but it has become more common to apply it before attention and feedforward layer as well, which can lead to more stable training dynamics.

#### Feedforward with GeLU
After the inputs go through Multi-head attention and Layer Normalization, we feed them through feedforward layers with non-linearity. There are 2 main reasons to do it:
1. Position-based transformations: Attention blocks learns relationship between input tokens, but does not focus as much on each token independently. Feedforward layers do that. They process and transform each token independently and does not explicitly consider their relationship to each other during linear transformations.
2.  Non-linearity: Attention blocks do not perform any non linearity, they essentially do large matrix multiplications with Linear layers, but no non-linearity is introduced. Feedforward layers introduce non-linearity with techniques such as GeLU.
3.  Increasing model capacity: Feedforward layers also have much larger hidden dimensions compared to the embedding dimension that our LLM is trying to learn, so we get increased model capacity

**GeLU**
ReLU is pretty straightforward to understand. It squishes all negative values to 0. 

`ReLU(x) = x if x>0 else 0`

Dropout on the other hand stochastically squishes neurons to 0.

The idea behind GeLU (Gaussian Error Linear Unit) is to do something between Dropout and ReLU. It does not squish all negative values to 0 and also does not randomly drop neurons. 
GeLU uses the fact that inputs are distributed normally (due to normalization techniques seen above). It then uses a CDF of this normal distribution and the value of the input to stochastically determine whether to drop a keep this input.

GeLU has a higher probability of dropping a neuron (multiplying by 0) while x decreases since CDF(x) will be small for smaller values. That's how we get a combination of dropout and ReLU. Below is the exact mathematical form of [GeLU](https://paperswithcode.com/method/gelu).

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mtext>GELU</mtext>
  <mrow data-mjx-texclass="INNER">
    <mo data-mjx-texclass="OPEN">(</mo>
    <mi>x</mi>
    <mo data-mjx-texclass="CLOSE">)</mo>
  </mrow>
  <mo>=</mo>
  <mi>x</mi>
  <mrow>
    <mi>P</mi>
  </mrow>
  <mrow data-mjx-texclass="INNER">
    <mo data-mjx-texclass="OPEN">(</mo>
    <mi>X</mi>
    <mo>&#x2264;</mo>
    <mrow>
      <mi>x</mi>
    </mrow>
    <mo data-mjx-texclass="CLOSE">)</mo>
  </mrow>
  <mo>=</mo>
  <mi>x</mi>
  <mi mathvariant="normal">&#x3A6;</mi>
  <mrow data-mjx-texclass="INNER">
    <mo data-mjx-texclass="OPEN">(</mo>
    <mi>x</mi>
    <mo data-mjx-texclass="CLOSE">)</mo>
  </mrow>
  <mo>=</mo>
  <mi>x</mi>
  <mo>&#x22C5;</mo>
  <mfrac>
    <mn>1</mn>
    <mn>2</mn>
  </mfrac>
  <mrow data-mjx-texclass="INNER">
    <mo data-mjx-texclass="OPEN">[</mo>
    <mn>1</mn>
    <mo>+</mo>
    <mtext>erf</mtext>
    <mo stretchy="false">(</mo>
    <mi>x</mi>
    <mrow>
      <mo>/</mo>
    </mrow>
    <msqrt>
      <mn>2</mn>
    </msqrt>
    <mo stretchy="false">)</mo>
    <mo data-mjx-texclass="CLOSE">]</mo>
  </mrow>
  <mo>,</mo>
</math>
<math xmlns="http://www.w3.org/1998/Math/MathML">
  <mi>X</mi>
  <mo>&#x223C;</mo>
  <mrow>
    <mi data-mjx-variant="-tex-calligraphic" mathvariant="script">N</mi>
  </mrow>
  <mo stretchy="false">(</mo>
  <mn>0</mn>
  <mo>,</mo>
  <mn>1</mn>
  <mo stretchy="false">)</mo>
</math>

In practice, we might not want to compute CDF so we use an [approximation of GeLU](https://paperswithcode.com/method/gelu):

<math xmlns="http://www.w3.org/1998/Math/MathML">
  <mn>0.5</mn>
  <mi>x</mi>
  <mrow data-mjx-texclass="INNER">
    <mo data-mjx-texclass="OPEN">(</mo>
    <mn>1</mn>
    <mo>+</mo>
    <mi>tanh</mi>
    <mo data-mjx-texclass="NONE">&#x2061;</mo>
    <mrow data-mjx-texclass="INNER">
      <mo data-mjx-texclass="OPEN">[</mo>
      <msqrt>
        <mn>2</mn>
        <mrow>
          <mo>/</mo>
        </mrow>
        <mi>&#x3C0;</mi>
      </msqrt>
      <mrow data-mjx-texclass="INNER">
        <mo data-mjx-texclass="OPEN">(</mo>
        <mi>x</mi>
        <mo>+</mo>
        <mn>0.044715</mn>
        <msup>
          <mi>x</mi>
          <mrow>
            <mn>3</mn>
          </mrow>
        </msup>
        <mo data-mjx-texclass="CLOSE">)</mo>
      </mrow>
      <mo data-mjx-texclass="CLOSE">]</mo>
    </mrow>
    <mo data-mjx-texclass="CLOSE">)</mo>
  </mrow>
</math>

::: {.callout-note}
PyTorch provides both the exact and approximation with tanh version in its implementation
:::
So, our feedforward part of transformer becomes a regular MLP with non-linearities introduced by GeLU.

#### Skip connections



