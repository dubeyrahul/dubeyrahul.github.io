---
title: "LLMs Part 2: Attention"
author: "Rahul Dubey"
date: "2024-12-28"
categories: [ml, deep-learning, llm]
---
In this post, I'll dive into the attention mechanism that is one of the key feature of modern LLMs. We'll go over some of the shortcomings of pre-LLM Neural language models such as RNNs and its variants, how attention solves these shortcomings, and how it is implemented in practice. Lastly, we'll discuss what are some computational infrastructure implication of attention mechanism that allows large scale training.

### Neural language models
Language models are nothing but an ML model tasked with `modeling the language`, simply put we want to learn a probability distribution over the language. We want to be able to predict `P(word | context-words)`. The initial approaches were doing so were very specific to textual data that leveraged the grammar and structure of the language. Next, came statistical approaches such as Naive Bayes / Bag-of-words model. Then we got one of the most impressive and cited paper: [`A Neural Probabilistic Language Model`](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf) where the authors demonstrated how to leverage MLPs to model the language. Next, we got word embedding based model but after MLP the next big advancement came in [`Sequence to Sequence Learning with Neural Networks`](https://arxiv.org/pdf/1409.3215) which proposed LSTMs with encoder-decoder setup to encode and decode sequences and were particularly impressive in tasks such as language translation. For quite a long period, these RNN-variants were the state of the art approach but they suffered from the following issues:
- unable to model long range dependency
- slow training due to serial processing of text
- vanishing and exploding gradients

To solve some of these problems, the next paper [`Attention is all you need`](https://arxiv.org/pdf/1706.03762) created a new architecture called `Transformer` which leveraged `Attention`. The concept of attention was not introduced in this paper but in another paper: [`Neural Machine Translation by Jointly Learning to Align and Translate`](https://arxiv.org/pdf/1409.0473).

### Problems solved by Attention mechanism
The way RNN based models work is they process one token at a time and as they move forward in time, they create a hidden representation of context seen so far. It does not refer to older tokens directly but only relies on this learned hidden representation. Since, they are using a single vector to encode the entire historical text, it is difficult for them to encode long range dependency. 

Attention mechanism gets around this problem by basically doing a "smart" brute-force approach. Rather than summarizing all the historical context into a single vector, we let each to-be-generated token to go back and look at all the historical tokens directly! Why restrict it to a compressed representation when I can look at all the previous tokens and their representation? Moreover, during training it learns how much "attention" to pay each historical token to generate the current token: hence the "smart" brute-force.

By thinking this way, we can see how incredibly more powerful attention mechanism can be: it has access to more information directly compared to traditional RNN based approaches. Now, one may think this brute force approach is great but must be super slow compared to RNN right? RNN is only using one historical vector whereas attention is using so many! To get around this problem, we use parallelization. Since attention works by doing a bunch of vector multiplication between current token and historical token, you can set it up as a matrix multiplication and leverage GPUs specialized for parallelizing matrix multiplication operations! So, in theory, yes we are doing a lot more computation than RNNs but we are able to do it really fast because the operations can be parallelized, whereas in RNN, where each token is processed sequentially, we cannot parallelize it.

Now, the gradient problem with RNN was due to gradient being propagated from the last token to the first token via a bunch of multiplication steps. When a bunch of small numbers or large numbers are multiplied, they either reach 0 or a really large number, both of which are terrible for back propagation. In attention, we have direct connection between tokens so the information (gradient) can directly propagate from the last token to the first token: no multiplication-hops required! To be fair, in addition to this attention mechanism, some nice techniques such as Layer Normalization and Residual connections were added to the Transformer architecture to handle the vanishing/exploding gradient problem.

Now that we know that attention mechanism is awesome and helps us build Transformers, which are backbone of all modern LLMs (at least in 2024), let's dive into the details of attention mechanism and its variants.

### Attention mechanism
At its core, attention mechanism allows a model to build a contextual representation of each token by incorporating information from surrounding tokens. Attention really means which surrounding tokens the model should pay more attention to. It can be described as analogous to how we read a text: as we read English text from left-to-right, we remember certain words more than the other to understand the meaning of the sentence.

Attention mechanism requires us to understand 3 concepts: Queries, Keys, and Values
1. Query: Query is like a search term and represents the current token that the model is trying to understand
2. Key: Key is like a database index used to index and search a database. Query is compared to Key to find which tokens to pay more attention to
3. Values: Value represents the actual input. After determening which keys to pay attention to, we retrieve the values of those tokens in proportion to how relevant their keys were to the query.

In essence, attention mechanism figures out how relevant each key is to the query, and then uses this weight to values to compute contextual representation for the query token.

#### Simple Self-attention
Self-attention simply means that the token in the sequence attends to the tokens of the same sequence. It is called `self-attention` because the first definition of attention in the literature was in the context of encoder-decoder architecture of sequence translation where the decoder was allowed to "attend" to the state of encoder at each token step, therefore the decoder was not attending on itself but on the encoder.
In self-attention, the decoder (or encoder) attends to its own state and the sequence input to itself.

To understand self-attention, let's start with a simplified self-attention. Say we have a sequence of token input embeddings `[X1, X2, X3,..XT]`. Recall from part-1, that these embeddings are based on vocabulary and position, i.e. they are not contextual. Both the sequences: "An apple today", and "An apple iphone" will have same `X` for `apple`. To create a contextual embedding, we want to compare `apple` to its surrounding words and compute an attention score w.r.t each surrounding token.

Let's call these `Attention-score(X1) = [W1, W2, W3,...WT]`. For now, let's assume that we can just take dot-product between `X1` and all other `X_i` to get `W_i`. Next, we normalize this `W` vector with the softmax function to get `W_norm`. Now, we can compute contextual vector `XC_1` for `X1` by multiplying it with normalized attention scores in `W_norm` and adding up the vectors. 

Therefore, `XC_1` = `W_norm` * `X`. We just take a weighted-sum of X to get XC.

In this naive attention scoring, we have simply taken a dot product of input X with itself (X @ X.T) to get attention scores W, then we softmax normalize it for each token, and lastly multiply this back with X.

#### Self-attention
Now, let's understand the actual self-attention mechanism with weights that LLMs learn. It is quite similar to the above simplified approach, the only difference being (1) how W is calculated and (2) how W is utilized to get the contextual vector. The rest of the operations remain the same: (1) compute attention scores (2) normalize attention scores (3) weight the input per normalized attention scores to get contextual representation.

**Computing Attention weight**

We ask the LLM to learn 3 matrices: W_Q: Query, W_K: Keys, W_V: Values and these matrices drive the computation of attention scores. Say, we have a token embedding `X1`. First thing we do is, we bring it into `Q's` space by multiplying it with W_Q to get `X1_Q`. Note that to carry out this multiplication, the shape of W_Q should align with `X1`. So if `X1` is of dimension `d`, then `W_Q`'s first (or both) dimension has to be `d`. We will use this `X1_Q`: query vector to search among the `Keys` (similar to a database where a query is evaluated against index keys).

To do this, we compute `keys = X @ W_K` where `X` is input matrix of shape (n_tokens, d) and `W_K` is our Keys matrix of shape (d, d). So we get `keys` matrix of shape (n_tokens, d). Now, we will compare `X1_Q` to each of the rows in `keys` matrix to figure out which other tokens should we pay attention to, which gives us `Attention-scores(X1)` of shape (1, n_tokens), so a score for each token.

`Attention-scores(X1) = X1 dot keys`

We essentially did a weighted brute-force search across all tokens in the sequence to find which other tokens should our token-1 attend to.

Now, we softmax normalize this. However, the trick is to take `Softmax(Attention-scores(X1)/ sqrt(d))`. This is called `scaled-dot-product attention` due to the scaling by `sqrt(d)`. Let's call this `Attention-scores-norm(X1)`.

```
Reason behind scaling: Say we are taking softmax over 2 elements z1 and z2 and z1 >>> z2. Now, when we calculate a softmax e^z1 >>>>>> e^z2. Therefore, softmax for z1 will tend to become 1 and softmax for z2 will tend to become 0. This leads to softmax function becoming a step function whose gradients are not well-defined and are nearly close to 0.

Now imagine our context-vectors of 1000s of dimensions whose dot product can grow very large. These large dot products run into the same issue as mentioned above. So, to avoid this learning problem during training, we divide the attention scores (dot products) by sqrt(d) (I am guessing a heuristic) and then take a softmax.
```
Ok, let's get back! We have the attention-scores normalized. Now, all we have to do is multiply this with "something" to get contextual representation of `X1`.

To compute this "something", we use the W_V matrix. We compute `values = X @ W_V` to get a matrix of shape (n_tokens, d). These represent the value of each token in d-dimensional space. Now, we can multiply the attention scores with values:

`X1_C = Attention-scores-norm(X1) @ values` (matrix shape math: (1, n_tokens) * (n_tokens, d) => (1,d))

We computed the context vector of just `X1` but we can compute this for all tokens at once using matrix multiplication. These computations can also be parallelized and sped up on GPUs, making attention/transformers such an attractive architecture in modern DL systems.

```
Let X be our input token sequence of shape (N, d1). We can initialize W_K, W_Q, W_V of shape (d1, d2). Our context vector will be of dimension d2.
Step 1: Project into K,V,Q:

keys = X @ W_K                                                  # (N, d2)
queries = X @ W_Q                                               # (N, d2)
values = X @ W_V                                                # (N, d2)

Step 2: Compute scaled dot-product attention scores
attention_scores = queries @ keys.T                             # (N, N)
attention_scores_norm = softmax(attention_scores / sqrt(d1))    # (N, N)

Step 3: Compute contextual embedding using values
context_x = attention_scores_norm @ values                      # (N, d2)
```

#### Causal attention
Now, let's create a type of attention that is used in decoder-only architecture: these are models used in text-generation where at a certain point in time, the model can only look back at past-tokens rather than all tokens in a sequence. We can continue using the same scaled dot product attention mechanism with a small twist: Just zero-out the tokens after the current token!

This is called `Causal attention` or `Masked attention`. Causal because we are only relying on previous tokens to predict next tokens so we are saying that the previous tokens causes the next token. I am not sure whether this can be really be called causal from a `causal inference` standpoint. It is also called Masked because we are masking tokens appearing after the current token so that we only attend to tokens occurring before the current token.

We can create a 2D matrix which looks like this:
```
[1,0,0,0]
[1,1,0,0]
[1,1,1,0]
[1,1,1,1]
```
It's a lower-triangular matrix (diagonal and below are 1, rest are 0). It is obvious how this works as a mask. If we multiply this with a matrix, the 0s will 0 out those elements.

So, we carry out our attention computation like before, only before we multiply scores with values, we apply this mask so that all the attention-scores after our current token are zeroed-out. We then normalize this masked-attention-score and multiply it with values, and voila! We have causal attention scores for each token!

```
Let M be the mask matrix of shape (n, n) where all elements on and below the diagonal are 1, rest are 0.

Step 2: Compute scaled dot-product attention scores and mask it
attention_scores = queries @ keys.T                                         # (N, N)
attenion_scores_masked = M @ attention_scores                               # (N, N)

# Normalize the attenion_scores_masked so that rows sum to 1
attention_scores_norm = attenion_scores_masked / sum_of_rows                # (N, N)
attention_scores_norm_scaled = softmax(attention_scores_norm / sqrt(d1))    # (N, N)

Step 3: same as before
context_x = attention_scores_norm_scaled @ values                           # (N, d2)
...
```
A slightly better approach would be to think about what softmax does. It performs e^x for each x and divides by sum of each e^x. So, if we set x=-inf, then e^x will automatically be 0. Hence, instead of creating a maxk of 1s and 0s, we can creating a mask of 1s and -inf
```
Step 2: Compute scaled dot-product attention scores and mask it
M = torch.tril(torch.ones(n, n))
attenion_scores_masked = attention_scores.masked_fill(~mask.bool(), -torch.inf) # replace the 0s (with ~mask.bool) with -inf
attention_scores_norm_scaled = softmax(attention_scores_norm / sqrt(d1))    # (N, N)

Step 3: same as before
context_x = attention_scores_norm_scaled @ values                           # (N, d2)
```
Note:
1. We mask before we normalize, because we want to ensure that attention-scores that are multiplied with values sum to 1.
2. We are still doing things for the entire sequence in parallel by leveraging matrix-multiplication: we still have the same advantages of parallelization that we had in self-attention.

Another common operation that is done at this point is applying Dropout to introduce regularization. So the process becomes:
1. Compute attention scores
2. Apply causal mask
4. Softmax with scaling based on d_out
5. **Apply dropout** (attention weights get scaled by 1 / (1 - dropout_rate) to ensures that rows sum to ~1. For instance, if a particular attention weight is 0.3 before dropout, and dropout is 0.2, the new value after applying dropout will be 0 if that attention weight is dropped, or 0.3 * 1.25 = 0.375 if it is not dropped. The sum of the weights over all inputs may not be exactly 1. This scaling is implemented in Dropout layer and is not specific to Attention, that's just how Dropout works during training, so that during inference we don't have to do any scaling.)
6. Compute attention-weighted values

#### Multi-headed attention
Conceptually, multi-headed attention is just the above attention mechanism split into "multiple heads". Imagine that we want to create a d_out dimensional context vector. We can split this d_out into n_heads where each head is an attention block of d_head = d_out // n_heads. The intuition is that we will train each head **independently** and the model will learn specific and unique features in each head. It is kind of similar to CNN filters where the idea is that each filter is trained independently and each filter learns something specific and unique about the images. In CNNs, we get one filter that learns edges, another may learn gradients, and so on. In LLMs with multiple-attention heads, we may get one head focusing on syntactic structure, another focus on semantic relationship, and so on. Check out the [BertViz](https://github.com/jessevig/bertviz) tool for a visualization of attention heads.

Note that, multiple heads is not merely a single massive attention split into multiple heads, i.e. it is really important that each head is trained independently, otherwise we are not learning independent features in each head but just training a single attention head in parallel. This becomes crucial in implementation because we have to lay out and reshape the matrices in such a way that allows for such independent training.

One may ask, why not stack attention heads vertically on top of each other rather than next to each other? One neat advantage is that we can achieve similar learning capacity with less cost by laying them out horizontally since it is a single W matrix to learn instead of sequentially learning separate W matrices for each layer.

Implementation wise, one can essentially create multiple Causal Attention modules and put them in a list such as and compute each head sequentially:
```
MultiHeadedAttentionList = concatenate([CA1, CA2, CA3]) # 3 headed causal-attention (CA=causal attention)
```
However, we can again parallelize this and leverage GPUs to speed things up. Trick to parallelization: stuff it in a matrix such that each head still operates independently but gets computed in parallel.
```
MultiHeadedAttention = [CA1_CA2_CA3] # where each CA is laid out next to each other
```
The implementation below is taken from [LLM from Scratch](https://github.com/rasbt/LLMs-from-scratch/blob/bb31de89993441224e9005926dedad95395bb058/ch03/01_main-chapter-code/multihead-attention.ipynb)
In addition to having multiple-heads, we will also introduce a batch dimension so that we are not passing 1 sequence at a time but a batch of sequences. This way we can fully utilize the GPUs!

```python
class MultiHeadedAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.num_heads = num_heads
        if self.d_out % self.num_heads != 0:
            raise("Num heads should be perfectly divisible by output dimension")
        self.head_dim = self.d_out // self.num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        # x @ W.T -> (b, num_tokens, d_in) @ (d_in, d_out) -> (b, num_tokens, d_out)
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # Now, we will split the matrix implicitly by number of heads. We will use view() to do this
        # now each of keys, queries, values are of shape (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)

        # As discussed above, we want to compute each attention head in parallel, so we will move the dimension around
        # to make it so -> (b, num_heads, num_tokens, head_dim). We are swapping dimension 1 and 2.
        # Note that if we ignore the first 2 dimension, we get our simple single headed single sequence attention matrices of size(num-tokens, d_out)

        # now each of keys, queries, values are of shape (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1,2)
        queries = queries.transpose(1,2)
        values = values.transpose(1,2)

        # now we compute attention scores for each head and sequence in parallel
        # we want to multiple queries and keys with matrix multiplication and we want attention scores for each batch, head, and token
        # desired output dimension is (b, num_heads, num_tokens, num_tokens). So, let's prepare our data to get this.
        # we will adjust keys so that we get keys shape as (b, num_heads, head_dim, num_tokens) and queries shape is (b, num_heads, num_tokens, head_dim), so the product will have (b, num_heads, num_tokens, num_tokens), i.e. attention for each token with another for each batch, for each head.

        attn_scores = queries @ keys.transpose(2, 3) # we get (b, num_heads, num_tokens, num_tokens)

        # masking
        mask_bool = self.mask.bool()[:num_tokens,:num_tokens] # because num_tokens can be <  context_length
        # fill the masked area with -inf
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # now we can take softmax and see we divide it by sqrt(head_dim)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # now we compute context-vector which has to be of shape: (b, num_tokens, num_heads, head_dim)
        # attn_weights shape is (b, num_heads, num_tokens, num_tokens)
        # values shape is (b, num_heads, num_tokens, head_dim)
        # We still want to keep computation of context-vector for each head independent so we will do the multiplication as it is
        # attn_weights @ values will be of shape (b, num_heads, num_tokens, head_dim)
        # now we can reshape this: (b, num_tokens, num_heads, head_dim)

        context_vec = (attn_weights @ values).transpose(1,2)

        # now that each head has produced context vector independently, we can combine them by stacking horizontally
        # remember that self.d_out = num_heads * head_dim
        # we need to do a contiguous operation to lay this out in memory in a contiguous manner so that the view works
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)

        # the projection multplication is (b, num_tokens, d_out) @ (d_out, d_out) -> (b, num_tokens, d_out)
        context_vec = self.out_proj(context_vec) # optional projection
        
        # hence, we get context_vec of shape (b, num_tokens, d_out)
        return context_vec
```

### Summary
- Attention mechanism allows us to focus on each and every token independently and directly
- Attention scores quantify how much each token should be paid attention to and allows us to "weigh" the token value accordingly
- Causal attention is simply a way to prevent LLM from cheating and looking into the future tokens and we implement it via masking
- Multi-headed attention is several attention heads that are trained independently
- By leveraging smart layout of the attention heads, we can train each head independently and in parallel at the same time, thereby leveraging GPUs to speed up the code with matrix multiplication instead of sequentially training each head via for loops
- We can also add Dropout to drop certain attention scores before computing values as a means of regularization.
- Batched Matrix Multiplication can get complicated to get the head around but we can use the following tricks:
  - Write down the dimensions of your inputs and desired outputs. Print them if needed while developing modules.
  - Remember the following rules of matmul so that we are computing the right output while leveraging parallelism: PyTorch's torch.matmul (and the @ operator) performs batched matrix multiplication when the tensors have more than two dimensions. The rules are as follows:
    - If both tensors are 2D: This is standard matrix multiplication.
    - If one tensor is ND and the other is 2D: The 2D tensor is treated as a batch of matrices, and the multiplication is performed for each batch element of the ND tensor.
    - If both tensors are ND: The multiplication is performed over the last two dimensions of each tensor, with the leading dimensions being treated as batch dimensions.



        

