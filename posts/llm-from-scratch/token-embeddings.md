---
title: "LLMs Part 1: Token Embeddings"
author: "Rahul Dubey"
date: "2024-12-25"
categories: [ml, deep-learning, llm]
---
In this post, I am diving into token embeddings. LLMs like any other ML model cannot process raw-text directly. The process with which we convert raw-text into acceptable format (i.e. vector of numbers) is loosely called tokenization. Once tokenized, LLMs learn contextual embeddings for these tokens. We learn embeddings as part of LLM pre-training because doing so makes embeddings optimized for the specific data and learning task. The same concept applies to other data formats: audio, video, etc.

We perform this conversion in following steps:
1. Tokenization
2. Mapping to Token IDs
3. Adding special context tokens
4. Byte-pair encoding
5. Generate token dataset for GPT-style model
6. Token Embedding
7. Position Embedding

### Tokenization
A simple way to do this is to use some kind of regex expression to split your text into individual words. A lot of design decisions come into play here, for example:
- do we keep text case sensitive? 
  - Might be good if your LLM is supposed to generate case sensitive text
- do we keep and encode whitespace characters?
  - Might be good if your LLM is used for applications where whitespace generation is important: Python code generation
- do we want to keep punctuation?
  - Might be good if your LLM is used for expressive text generation but maybe not if you are doing simpler tasks like text classification

If we use both cases, whitespaces, punctuation, etc. we increase computational complexity but get more expressability in our tokens. In most modern NLP approaches (with LLMs), we keep all characters (rather we algorithmically decide using sub-words: more details later in this post)

After we tokenize the data, we typically end up with more words than if we had just split by whitespaces. Ok, so now instead of a single string we have a list of strings, but this is not acceptable to LLMs either, so we now generate an identifier for each token.

### Mapping to Token IDs
Essentially, we find all the unique tokens and assign them an integer ID to generate a `map` from token (str) -> tokenID (int). We also want to keep a `reverse-map` of this so that we can map IDs back into tokens. The size of this mapping defines the "vocabulary size". Vocabulary size is the number of unique tokens that your LLM can generate.
Tokenizer is now capable of 1) encoding: converting a string to a list of tokenIDs using a vocabulary and 2) decoding: converting a list of tokenIDs into list of tokens and subsequently a string.
Since, we are working with `map`, one obvious question is what happens if we a Key does not exist in the map. What this means in our tokenization context is what if we get a string that our Tokenizer has not seen before? It won't be able to convert it to TokenID, and hence it will fail to tokenize any text with words/strings it has not seen before. Similar to how we handle unknown keys in `map`, we can come up with some `default` value for unknown keys.

### Special context tokens
Special context tokens are essentially a catch-all to cover our corner cases (like mentioned above) and to provide LLM further assistance in text generation and understanding. 
For example, we can add an `<unk>` token to our vocabulary which acts as as a default value when our tokenizer seens an unknown token. Another common token is `<eos>` to denote end of sequence (or rather to indicate to LLM that it can stop generating text) It is useful when giving LLMs multiple sentences/documents that are unrelated. Some other common special context tokens are: `<bos>`: beginning of a new sequence and `<pad>` to denote that these are just padding to fit the batch size during training.

We basically add these special tokens to our vocabulary `map` and `reverse-map` and now our tokenizer is ready to tokenize any string! The downside of our existing tokenizer is the following:
- It is terrible at unseen words. For example, if "come" is in our vocabulary but "coming" is not, there's no way for our tokenizer to encode "coming" even though it is very closely related to a known token "come". So all unseen words become `<unk>`, even if we have closely related words in the vocabulary
- Our tokenizer relies on splitting over whitespace which works fine for English but not for all languages

### Byte pair encoding
Byte pair encoding, commonly called BPE is a subword-based tokenization scheme. On a high level, this is what it does:
1. Split words into individual characters (unigrams)
2. Merge commonly occurring pair of adjacent characters into new tokens
3. Repeat 2 until some stop condition like size of vocabulary or frequency cut-off
So, instead of word-based tokenization which is what we discussed earlier, we get "sub-word" based tokenization. Conceptually: `character < sub-word < word`.
The subword based tokenization does not split frequently occurring words but instead splits rarely occurring words. So "toy" may remain as a token but "toys" is split into "toy" and "s".
So, now when our tokenizer seens an unknown word while encoding, it splits into into tokens it has seen before. For e.g. if we are trying to tokenize a word: "absobloodylutely", it might break it up into <abso>, <bloody>, <lutely> subword tokens (assuming these 3 are present in the vocabularly generated by BPE tokenizer). 

TODO: I'll dive into BPE and sub-word tokenization is discussed in a separate blog post.

### Generate token dataset for GPT-style model
Before we go about creating embeddings for token, we need a way to create a "task" that will serve to create these embeddings. How GPT-style decoders typically go about doing so is some sort of variation of next-token prediction. So, we have a sentence: "I am a writer" in our text corpus. We can create training samples with <data, label> pair like below:
- `[I]`, `am`
- `[I, am]`, `a`
- `[I, am, a]`, `writer`
So essentially, we have to employ a sliding window approach through our corpus and create these pairs. One common design choice is what is the maximum length of my feature-vector. This is called the context-length. Higher context-length means more memory/compute needed during training as well as inference. However, the LLM can process and reason through a much longer piece of text, i.e. it can understand more context. Shorter context length could work if you are creating very targeted use-case like sentiment analysis of product reviews where reviews are short and the label generated is also just a sentiment label. Most GPT-style/decoder-style LLM models come with massive context length so that they are are as powerful as possible.
To do this in practice, we can create a Dataset and DataLoader class in PyTorch. Dataset would convert our text into tokens using a tokenizer and create input_ids and target_ids by shifting the sliding window till it reaches a size of `context_length`.
So, now our dataset looks like a row of tensors where our input_ids look like: `[30,23,1000,12]` and target_ids look like: `[23,1000,12,6]` (see how we set stride of 1 here).

### Generate Token Embeddings
To go from a list of tokenIDs to a list of token embeddings, we have to create an Embedding matrix of shape [vocab_size, embedding_size] where embedding_size is a design choice. Larger embedding_size means more compute/memory required but we also get a more powerful model. If we look at the shape of the Embedding matrix, we can say that each token is represented by a vector of size `embedding_size` and this Embedding matrix is basically a look-up table to go from token-id to a token-embedding. But instead of doing a naive one-hot encoding lookup, we do a lookup over a continuous space of `embedding_size`. This matrix is initialized randomly and is learned during the process of LLM pre-training.

We should be done now, right? We started with a piece of text, cleaned it, tokenized it into subword based tokens, and now we have a vector representation that can be fed into a neural network. But, there's a crucial piece missing. Modern transformer based LLMs are based on attention-mechanism. This attention mechanism works on tokens independently as in it does not have a notion of position of tokens in a piece of text. So, if a token 22 appears in text, it will always be mapped to the same embedding irrespective of where it appeared in the sequence.

To get around this shortcoming, `position embeddings` were introduced.

### Generate Position Embeddings
There are 2 categories of position embeddings: absolute and relative. Absolute position embeddings are defined by exact position of the token. So, any token at first position always gets the same position embedding irrespective of how long is the sequence. Relative position embeddings depend on a token's position w.r.t other tokens in the sequence, i.e. distance between tokens in a sequence. So the model learns how far apart are certain tokens rather than where exactly a token is situated. So relative works pretty well with varying size sequences and sequence lengths that are unseen in training. For absolute position embedding, the shape of this matrix would be `[context_length, embedding_size]`. Note that the input to position embedding layer is essentially a placeholder vector containing each position from 0 to `context_length`. Position embeddings can also be learned during training process, this is how it works in GPT models. T5 model uses relative position embedding whereas GPT/LLama use a Rotation based absolute position embedding.

Finally, to create the input embeddings we add token embeddings and position embeddings!
```
text -> token IDs -> token_embedding
[0,...,context_length] -> position_embedding
input_embedding = token_embedding + position_embedding
```

### Summary
- LLMs like any other NNs need numbers to work with, so we need to convert text to numbers
- We can split the text into subword based tokens
- Next, we learn a continuous representation for these tokens and call it token embedding
- Since attention based LLMs are position agnostic, we add a position embedding to token embedding to obtain the input embedding of a token