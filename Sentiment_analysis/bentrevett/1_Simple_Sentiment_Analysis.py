# conda activate py36gputorch100 && \
# cd /mnt/1T-5e7/mycodehtml/NLP/Sentiment_analysis/bentrevett && \
# rm e.l && python 1_Simple_Sentiment_Analysis.py \
# 2>&1 | tee -a e.l && code e.l

# ================================================================================
# 1 - Simple Sentiment Analysis

# @@ In this series we'll be building a machine learning model to detect sentiment (i.e. detect if a sentence is positive or negative) using PyTorch and TorchText. 

# @@ This will be done on movie reviews, using the IMDb dataset <http://ai.stanford.edu/~amaas/data/sentiment>

# @@ In this first notebook, we'll start very simple to understand the general concepts whilst not really caring about good results. 

# @@ Further notebooks will build on this knowledge and we'll actually get good results.

# ================================================================================
# Introduction

# @@ We'll be using a **recurrent neural network** (RNN) as they are commonly used in analysing sequences. An RNN takes in sequence of words, $X=\{x_1, ..., x_T\}$, one at a time, and produces a _hidden state_, $h$, for each word. We use the RNN _recurrently_ by feeding in the current word $x_t$ as well as the hidden state from the previous word, $h_{t-1}$, to produce the next hidden state, $h_t$. 

# @@ $$h_t = \text{RNN}(x_t, h_{t-1})$$

# @@ Once we have our final hidden state, $h_T$, (from feeding in the last word in the sequence, $x_T$) we feed it through a linear layer, $f$, (also known as a fully connected layer), to receive our predicted sentiment, $\hat{y} = f(h_T)$.

# Below shows an example sentence, with the RNN predicting zero, which indicates a negative sentiment. The RNN is shown in orange and the linear layer shown in silver. Note that we use the same RNN for every word, i.e. it has the same parameters. The initial hidden state, $h_0$, is a tensor initialized to all zeros. 
# <https://raw.githubusercontent.com/bentrevett/pytorch-sentiment-analysis/808aaafa6f0e8b40e1f3832f12605b44a2a503ad/assets/sentiment1.png>

# @@ some layers and steps have been omitted from the diagram, but these will be explained later.

# ================================================================================
# Preparing Data

# One of the main concepts of TorchText is the `Field`.
# These define how your data should be processed.
# In our sentiment classification task the data consists of both the raw string of the review and the sentiment, either "pos" or "neg".

# The parameters of a `Field` specify how the data should be processed.

# We use the `TEXT` field to define how the review should be processed, and the `LABEL` field to process the sentiment.

# Our `TEXT` field has `tokenize='spacy'` as an argument.
# This defines that the "tokenization" (the act of splitting the string into discrete "tokens") should be done using the spaCy<https://spacy.io> tokenizer.
# If no `tokenize` argument is passed, the default is simply splitting the string on spaces.

# `LABEL` is defined by a `LabelField`, a special subset of the `Field` class specifically used for handling labels.
#  We will explain the `dtype` argument later.

# For more on `Fields`, go here
# <https://github.com/pytorch/text/blob/master/torchtext/data/field.py>

# We also set the random seeds for reproducibility. 

# ================================================================================
import time
import random
import torch
from torchtext import data
from torchtext import datasets
import torch.nn as nn
import torch.optim as optim

# ================================================================================
SEED=1234

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic=True

# c TEXT: data.Field object
TEXT=data.Field(tokenize='spacy')
# print("TEXT",TEXT)
# <torchtext.data.field.Field object at 0x7f1288f6e5c0>

# c LABEL: data.LabelField object
LABEL=data.LabelField(dtype=torch.float)
# print("LABEL",LABEL)
# <torchtext.data.field.LabelField object at 0x7f1239d64080>

# ================================================================================
# Another handy feature of TorchText is that it has support for common datasets used in natural language processing (NLP). 

# The following code automatically downloads the IMDb dataset and splits it into the canonical train/test splits as `torchtext.datasets` objects. 

# It process the data using the `Fields` we have previously defined. 

# The IMDb dataset consists of 50,000 movie reviews, each marked as being a positive or negative review.

# ================================================================================
train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
# print("train_data",type(train_data))
# <class 'torchtext.datasets.imdb.IMDB'>

# iter_train_data=iter(train_data)
# for one_data in iter_train_data:
#     # print("one_data",one_data)
#     # print("one_data",dir(one_data))
#     # print("one_data",one_data.text)
#     # ['If', 'I', 'did', "n't", 'know', 'any', 'better', ',', 'it', 'almost', 'seems', 'like', 'it', 'was', 'staged', ',', 'but', 'it', 'was', "n't", '.', 'It', 'was', 'set', 'up', 'perfectly', ',', 'and', 'how', 'they', 'got', 'all', 'of', 'that', 'footage', 'is', 'amazing', '!', 'The', 'unfortunate', 'events', 'of', 'September', '11', ',', '2001', 'are', 'put', 'together', 'well', 'in', 'this', 'documentary', 'and', 'the', 'classic', 'footage', 'that', 'they', 'got', 'made', 'this', 'an', 'unfortunate', 'classic', '.', 'Just', 'the', 'history', 'in', 'the', 'footage', 'alone', 'should', 'make', 'it', 'a', 'MUST', 'see', 'for', 'any', 'american', 'or', 'person', 'touched', 'by', 'the', 'tragedy', 'of', 'September', '11', '.']
#     # print("one_data",one_data.label)
#     # pos

# ================================================================================
# one_data=train_data.examples[0]
# print("one_data",one_data)
# <torchtext.data.example.Example object at 0x7f52065f57f0>

# one_data=vars(one_data)
# print("one_data",one_data)
# {'text': ['If', 'I', 'did', "n't", 'know', 'any', 'better', ',', 'it', 'almost', 'seems', 'like', 'it', 'was', 'staged', ',', 'but', 'it', 'was', "n't", '.', 'It', 'was', 'set', 'up', 'perfectly', ',', 'and', 'how', 'they', 'got', 'all', 'of', 'that', 'footage', 'is', 'amazing', '!', 'The', 'unfortunate', 'events', 'of', 'September', '11', ',', '2001', 'are', 'put', 'together', 'well', 'in', 'this', 'documentary', 'and', 'the', 'classic', 'footage', 'that', 'they', 'got', 'made', 'this', 'an', 'unfortunate', 'classic', '.', 'Just', 'the', 'history', 'in', 'the', 'footage', 'alone', 'should', 'make', 'it', 'a', 'MUST', 'see', 'for', 'any', 'american', 'or', 'person', 'touched', 'by', 'the', 'tragedy', 'of', 'September', '11', '.'], 'label': 'pos'}

# ================================================================================
# @@ The IMDb dataset only has train/test splits, so we need to create a validation set. 

# @@ We can do this with the `.split()` method. 

# @@ By default this splits 70/30, however by passing a `split_ratio` argument, we can change the ratio of the split, i.e. a `split_ratio` of 0.8 would mean 80% of the examples make up the training set and 20% make up the validation set. 

# Default: 70/30
# data.split(split_ratio=0.8): 80 train/20 validataion

# ================================================================================
# @@ We also pass our random seed to the `random_state` argument, ensuring that we get the same train/validation split each time.

# Same seed: same random number

train_data,valid_data=train_data.split(random_state=random.seed(SEED))

# ================================================================================
# print("train_data",len(train_data))
# 17500

# print("valid_data",len(valid_data))
# 7500

# print("test_data",len(test_data))
# 25000

# ================================================================================
# Next, we have to build a _vocabulary_.

# This is a effectively a look up table where every unique word in your data set has a corresponding _index_ (an integer).

# @@ We do this as our machine learning model cannot operate on strings, only numbers.

# Each _index_ is used to construct a _one-hot_ vector for each word.

# @@ A one-hot vector is a vector where all of the elements are 0, except one, which is 1, and dimensionality is the total number of unique words in your vocabulary, commonly denoted by $V$.

# V: dimensionality of vector for each word
# V: total number of unique words in your vocabulary

# The number of unique words in our training set is over 100,000, which means that our one-hot vectors will have over 100,000 dimensions! 

# This will make training slow and possibly won't fit onto your GPU (if you're using one).

# ================================================================================
# There are two ways effectively cut down our vocabulary, we can either only take the top $n$ most common words or ignore words that appear less than $m$ times.

# We'll do the former, only keeping the top 25,000 words.

# ================================================================================
# What do we do with words that appear in examples but we have cut from the vocabulary? 

# We replace them with a special _unknown_ or `<unk>` token.

# For example, if the sentence was "This film is great and I love it" but the word "love" was not in the vocabulary, it would become "This film is great and I `<unk>` it".

# ================================================================================
# The following builds the vocabulary, only keeping the most common `max_size` tokens.

# @ MAX_VOCAB_SIZE: create 25000 vocabulary by using some criterion
MAX_VOCAB_SIZE=25_000

TEXT.build_vocab(train_data,max_size=MAX_VOCAB_SIZE)

LABEL.build_vocab(train_data)

# ================================================================================
# Why do we only build the vocabulary on the training set? 

# When testing any machine learning system you do not want to look at the test set in any way. 

# We do not include the validation set as we want it to reflect the test set as much as possible.

# ================================================================================
# print("Unique tokens in TEXT vocabulary",len(TEXT.vocab))
# 25002
# 25000 vocabulary + <unk> + <pad>

# print("Unique tokens in LABEL vocabulary",len(LABEL.vocab))
# 2

# ================================================================================
# Why is the vocab size 25002 and not 25000? One of the addition tokens is the `<unk>` token and the other is a `<pad>` token.

# When we feed sentences into our model, we feed a _batch_ of them at a time, i.e. more than one at a time, and all sentences in the batch need to be the same size. Thus, to ensure each sentence in the batch is the same size, any shorter than the longest within the batch are padded.

# ![](assets/sentiment6.png)

# ================================================================================
# We can also view the most common words in the vocabulary and their frequencies.

most_freq_20words=TEXT.vocab.freqs.most_common(20)
# print("most_freq_20words",most_freq_20words)
# [('the', 202502), (',', 193115), ('.', 165478), ('a', 109008), ('and', 108941), ('of', 100551), ('to', 93425), ('is', 76354), ('in', 60974), ('I', 54422), ('it', 53460), ('that', 49013), ('"', 44187), ("'s", 43188), ('this', 42213), ('-', 36728), ('/><br', 35482), ('was', 35034), ('as', 30219), ('movie', 29911)]

# ================================================================================
# print("TEXT.vocab",TEXT.vocab)
# <torchtext.vocab.Vocab object at 0x7f7a1e2aee80>
# <https://torchtext.readthedocs.io/en/latest/vocab.html#torchtext.vocab.Vocab>
entire_vocab=TEXT.vocab.itos

# ================================================================================
# @ integer to string method

int_to_str=TEXT.vocab.itos[:10]
# print("int_to_str",int_to_str)
# ['<unk>', '<pad>', 'the', ',', '.', 'a', 'and', 'of', 'to', 'is']

# ================================================================================
# @ string to integer method

label_str_to_int=LABEL.vocab.stoi
# print("label_str_to_int",label_str_to_int)
# defaultdict(<function _default_unk_index at 0x7f2e8fe2cae8>, {'neg': 0, 'pos': 1})

# ================================================================================
# @@ The final step of preparing the data is creating the iterators.

# @@ We iterate over these in the training/evaluation loop, and they return a batch of examples (indexed and converted into tensors) at each iteration.

# @@ We'll use a `BucketIterator` which is a special type of iterator that will return a batch of examples where each example is of a similar length, minimizing the amount of padding per example.

# @@ We also want to place the tensors returned by the iterator on the GPU (if you're using one).

# @@ PyTorch handles this using `torch.device`, we then pass this device to the iterator.

# ================================================================================
BATCH_SIZE = 64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data), 
    batch_size = BATCH_SIZE,
    device = device)

# ================================================================================
# Build the Model

# @@ The next stage is building the model that we'll eventually train and evaluate.

# @@ There is a small amount of boilerplate code when creating models in PyTorch, note how our `RNN` class is a sub-class of `nn.Module` and the use of `super`.

# Within the `__init__` we define the _layers_ of the module.
# our three layers are an _embedding_ layer, our RNN, and a _linear_ layer.

# @@ All layers have their parameters initialized to random values, unless explicitly specified.

# @@ The embedding layer is used to transform our sparse one-hot vector (sparse as most of the elements are 0) into a dense embedding vector (dense as the dimensionality is a lot smaller and all the elements are real numbers).

# embedding_vector=embedding_layer(sparse_one_hot_vector)

# res=dim(embedding_vector)<dim(sparse_one_hot_vector)
# res: True

# res=Is_element_real_number(embedding_vector)
# res: True

# @@ This embedding layer is simply a single fully connected layer.

# def embedding layer(sparse_one_hot_vectors):
#     fully_connected_layer(sparse_one_hot_vectors)

# @@ As well as reducing the dimensionality of the input to the RNN, there is the theory that words which have similar impact on the sentiment of the review are mapped close together in this dense vector space.

# embedding_vectors=embedding_layer(sparse_one_hot_vectors)

# scope: embedding_vector[0] similar to embedding_vectors[1]
#     res=distance_in_vector_space(embedding_vector[0],embedding_vectors[1])
#     res: close

# ================================================================================
# For more information about word embeddings, see here 
# <https://monkeylearn.com/blog/word-embeddings-transform-text-numbers>

# ================================================================================
# @@ The RNN layer is our RNN which takes in our dense vector and the previous hidden state $h_{t-1}$, which it uses to calculate the next hidden state, $h_t$.

# <https://raw.githubusercontent.com/bentrevett/pytorch-sentiment-analysis/808aaafa6f0e8b40e1f3832f12605b44a2a503ad/assets/sentiment7.png>

# ================================================================================
# @@ Finally, the linear layer takes the final hidden state and feeds it through a fully connected layer, $f(h_T)$, transforming it to the correct output dimension.

# final_hidden_output=RNN(input)
# vector_whose_dimension_is_number_of_classes=fully_connected_layer(final_hidden_output)

# ================================================================================
# @@ The `forward` method is called when we feed examples into our model.

# Each batch, `text`, is a tensor of size [sentence length, batch size]

# That is a batch of sentences, each having each word converted into a one-hot vector.

# ================================================================================
# You may notice that this tensor should have another dimension due to the one-hot vectors, however PyTorch conveniently stores a one-hot vector as it's index value, i.e.
# the tensor representing a sentence is just a tensor of the indexes for each token in that sentence.

# ================================================================================
# @@ The act of converting a list of tokens into a list of indexes is commonly called numericalizing.

# list_of_indexes=numericalizing(list_of_tokens)

# ================================================================================
# @@ The input batch is then passed through the embedding layer to get `embedded`, which gives us a dense vector representation of our sentences.

# `embedded` is a tensor of size [sentence length, batch size, embedding dim].

# ================================================================================
# @@ `embedded` is then fed into the RNN.

# @@ In some frameworks you must feed the initial hidden state, $h_0$, into the RNN, however in PyTorch, if no initial hidden state is passed as an argument it defaults to a tensor of all zeros.

# ================================================================================
# The RNN returns 2 tensors, `output` of size [sentence length, batch size, hidden dim] and `hidden` of size [1, batch size, hidden dim].

# `output` is the concatenation of the hidden state from every time step, whereas `hidden` is simply the final hidden state.

# We verify this using the `assert` statement.

# Note the `squeeze` method, which is used to remove a dimension of size 1.

# ================================================================================
# @@ Finally, we feed the last hidden state, `hidden`, through the linear layer, `fc`, to produce a prediction.

# ================================================================================
class RNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):

        super().__init__()

        # print("input_dim",input_dim)
        # 25002
        # print("embedding_dim",embedding_dim)
        # 100
        # print("hidden_dim",hidden_dim)
        # 256
        # print("output_dim",output_dim)
        # 1

        # ================================================================================
        self.embedding = nn.Embedding(input_dim, embedding_dim)

        self.rnn = nn.RNN(embedding_dim, hidden_dim)

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        # print("text",text.shape)
        # [968, 64]

        # 968: max length of each review text
        # 64: 64 number of review text

        # ================================================================================
        embedded = self.embedding(text)
        # print("embedded",embedded)
        # tensor([[[-0.2222,  0.8854,  0.1469,  ..., -0.9625,  0.0565, -0.4714],
        #          [ 1.1721, -0.7754,  1.5951,  ...,  0.7601,  0.1505,  0.1308],

        # print("embedded",embedded.shape)
        # torch.Size([968, 64, 100])
        # /mnt/1T-5e7/mycodehtml/NLP/Sentiment_analysis/bentrevett/pics/2019_05_26_12:57:38.png

        # ================================================================================
        # @ Pass embedding vector into RNN

        output, hidden = self.rnn(embedded)
        # print("output",output)
        # tensor([[[ 1.7660e-01,  1.3046e-02, -3.3094e-01,  ..., -4.0187e-01,
        #            9.4054e-02, -3.1611e-01],
        # print("output",output.shape)
        # torch.Size([968, 64, 256])

        # print("hidden",hidden)
        # tensor([[[ 0.5799,  0.5206,  0.2181,  ..., -0.0136,  0.4553, -0.0208],
        #          [ 0.5799,  0.5206,  0.2181,  ..., -0.0136,  0.4553, -0.0208],
        # print("hidden",hidden.shape)
        # torch.Size([1, 64, 256])

        # ================================================================================
        #output = [text length, batch size, hidden dim]
        #hidden = [1, batch size, hidden dim]
        
        # ================================================================================
        assert torch.equal(output[-1,:,:], hidden.squeeze(0))
        
        # ================================================================================
        out_from_fc=self.fc(hidden.squeeze(0))
        # print("out_from_fc",out_from_fc)
        # tensor([[ 0.2215],
        #         [ 0.2215],

        # print("out_from_fc",out_from_fc.shape)
        # [64, 1]

        return out_from_fc

# ================================================================================
# @@ We now create an instance of our RNN class.

# @@ The input dimension is the dimension of the one-hot vectors, which is equal to the vocabulary size.

# input_dimension
# = dimension_of_one_hot_vector
# = number of vocabulary

# ================================================================================
# @@ The embedding dimension is the size of the dense word vectors.

# @@ This is usually around 50-250 dimensions, but depends on the size of the vocabulary.

# dimension_of_embedding_vector 
# = 50 to 250, depending on number of entire vocabulary

# ================================================================================
# @@ The hidden dimension is the size of the hidden states.

# @@ This is usually around 100-500 dimensions, but also depends on factors such as on the vocabulary size, the size of the dense vectors and the complexity of the task.

# hidden dimension
# = usually around 100-500 dimensions

# depending on
# - number of entire vocabulary
# - dimension of embedding vector
# - complexity of the task

# ================================================================================
# @@ The output dimension is usually the number of classes, however in the case of only 2 classes the output value is between 0 and 1 and thus can be 1-dimensional, i.e. a single scalar real number.

# ================================================================================
INPUT_DIM = len(TEXT.vocab)
# print("INPUT_DIM",INPUT_DIM)
# 25002

# c EMBEDDING_DIM: dimension of embedding vector
EMBEDDING_DIM = 100

# c HIDDEN_DIM: dimension of hidden state vector
HIDDEN_DIM = 256

OUTPUT_DIM = 1

# ================================================================================
model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)

# ================================================================================
# @@ Let's also create a function that will tell us how many trainable parameters our model has so we can compare the number of parameters across different models.

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

num_of_trainable_params=count_parameters(model)
# print("num_of_trainable_params",num_of_trainable_params)
# 2592105

# ================================================================================
# Train the Model

# @@ Now we'll set up the training and then train the model.

# @@ First, we'll create an optimizer. 

# @@ This is the algorithm we use to update the parameters of the module. 

# @@ Here, we'll use _stochastic gradient descent_ (SGD). 

# @@ The first argument is the parameters will be updated by the optimizer, the second is the learning rate, i.e. how much we'll change the parameters by when we do a parameter update.

optimizer = optim.SGD(model.parameters(), lr=1e-3)

# ================================================================================
# @@ Next, we'll define our loss function. 

# @@ In PyTorch this is commonly called a criterion. 

# @@ The loss function here is _binary cross entropy with logits_. 

# Our model currently outputs an unbound real number. 

# As our labels are either 0 or 1, we want to restrict the predictions to a number between 0 and 1. 

# We do this using the _sigmoid_ or _logit_ functions. 

# ================================================================================
# We then use this this bound scalar to calculate the loss using binary cross entropy. 

# @@ The `BCEWithLogitsLoss` criterion carries out both the sigmoid and the binary cross entropy steps.

# def BCEWithLogitsLoss():
#     sigmoid()
#     cross_entropy()

criterion = nn.BCEWithLogitsLoss()

# ================================================================================
# @@ Using `.to`, we can place the model and the criterion on the GPU (if we have one). 

model = model.to(device)
criterion = criterion.to(device)

# ================================================================================
# @@ Our criterion function calculates the loss, however we have to write our function to calculate the accuracy. 

# @@ This function first feeds the predictions through a sigmoid layer, squashing the values between 0 and 1, we then round them to the nearest integer. This rounds any value greater than 0.5 to 1 (a positive sentiment) and the rest to 0 (a negative sentiment).

# @@ We then calculate how many rounded predictions equal the actual labels and average it across the batch.

# ================================================================================
def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division 
    acc = correct.sum() / len(correct)
    return acc

# ================================================================================
# @@ The `train` function iterates over all examples, one batch at a time.

# ================================================================================
# `model.train()` is used to put the model in "training mode", which turns on _dropout_ and _batch normalization_.

# Although we aren't using them in this model, it's good practice to include it.

# ================================================================================
# @@ For each batch, we first zero the gradients.

# Each parameter in a model has a `grad` attribute which stores the gradient calculated by the `criterion`.

# PyTorch does not automatically remove (or "zero") the gradients calculated from the last gradient calculation, so they must be manually zeroed.

# ================================================================================
# We then feed the batch of sentences, `batch.text`, into the model.

# @@ Note, you do not need to do `model.forward(batch.text)`, simply calling the model works.

# The `squeeze` is needed as the predictions are initially size [batch size, 1], and we need to remove the dimension of size 1 as PyTorch expects the predictions input to our criterion function to be of size [batch size].

# The loss and accuracy are then calculated using our predictions and the labels, `batch.label`, with the loss being averaged over all examples in the batch.

# @@ We calculate the gradient of each parameter with `loss.backward()`, and then update the parameters using the gradients and optimizer algorithm with `optimizer.step()`.

# The loss and accuracy is accumulated across the epoch, the `.item()` method is used to extract a scalar from a tensor which only contains a single value.

# Finally, we return the loss and accuracy, averaged across the epoch.
# The `len` of an iterator is the number of batches in the iterator.

# You may recall when initializing the `LABEL` field, we set `dtype=torch.float`.

# This is because TorchText sets tensors to be `LongTensor`s by default, however our criterion expects both inputs to be `FloatTensor`s.

# Setting the `dtype` to be `torch.float`, did this for us.

# The alternative method of doing this would be to do the conversion inside the `train` function by passing `batch.label.float()` instad of `batch.label` to the criterion.


# ================================================================================
def idx_word_to_str_word(vectors):
    # print("vectors",vectors.shape)
    # torch.Size([968, 64])

    str_all_setences=[]
    for i in range(vectors.shape[1]):
        one_sentence=vectors[:,i]
        # print("one_sentence",one_sentence)
        # tensor([   11,    98,   946,    13,    11,    86,    33,   226,    16,    21,
        #            91,     7,     5,   618,     4,    11,   852,   208,   448,    29,
        
        str_one_setence=[]
        len_of_one_sentence=one_sentence.shape[0]
        for j in range(len_of_one_sentence):
            idx_of_one_char=one_sentence[j]
            str_of_one_char=entire_vocab[idx_of_one_char]

            str_one_setence.append(str_of_one_char)

        str_all_setences.append(str_one_setence)
    
    # print("str_one_setence",str_one_setence)
    # ['This', 'film', 'seemed', 'way', 'too', 'long', 'even', 'at', 'only', '75', 'minutes', '.', 'The', 'problem', 'with', 'jungle', 'horror', 'films', 'is', 'that', 'there', 'is', 'always', 'way', 'too', 'much', 'footage', 'of', 'people', 'walking', '(', 'through', 'the', 'jungle', ',', 'up', 'a', 'rocky', 'cliff', ',', 'near', 'a', 'river', 'or', 'lake', ')', 'to', 'pad', 'out', 'the', 'running', 'time', '.', 'The', 'film', 'is', 'worth', 'seeing', 'for', 'the', 'laughable', 'and', 'naked', 'native', 'zombie', 'with', 'big', 'bulging', ',', 'bloody', 'eyes', 'which', 'is', 'always', 'accompanied', 'on', 'the', 'soundtrack', 'with', 'heavy', 'breathing', 'and', 'lots', 'of', '<unk>', '.', '<unk>', 'fans', 'will', 'be', 'plenty', 'entertained', 'by', 'the', 'bad', 'English', 'dubbing', ',', 'gratuitous', 'female', 'flesh', 'and', 'very', 'silly', 'makeup', 'jobs', 'on', 'the', 'monster', 'and', 'native', 'extras', '.', 'For', 'a', 'zombie', '/', 'cannibal', 'flick', 'this', 'was', 'pretty', 'light', 'on', 'the', 'gore', 'but', 'then', 'I', 'probably', 'did', "n't", 'see', 'an', 'uncut', 'version', '.', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>',..., '<pad>', '<pad>']
    
def train(model, iterator, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in iterator:
        
        optimizer.zero_grad()
        
        # ================================================================================
        batch_text_data=batch.text
        # print("batch_text_data",batch_text_data)
        # tensor([[  11,   66, 6722,  ..., 3402,  158,   66],
        #         [  98,    9,   17,  ...,    8,    5,   24],
        #         [ 946,  238, 3154,  ..., 9077,  740,  476],
        #         ...,
        #         [   1,    1,    1,  ...,    1,    1,    1],
        #         [   1,    1,    1,  ...,    1,    1,    1],
        #         [   1,    1,    1,  ...,    1,    1,    1]], device='cuda:0')
        
        # ================================================================================
        idx_word_to_str_word(batch.text)

        # ================================================================================
        predictions = model(batch_text_data).squeeze(1)
        
        loss = criterion(predictions, batch.label)
        
        acc = binary_accuracy(predictions, batch.label)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

# ================================================================================
# @@ `evaluate` is similar to `train`, with a few modifications as you don't want to update the parameters when evaluating.

# `model.eval()` puts the model in "evaluation mode", this turns off _dropout_ and _batch normalization_. Again, we are not using them in this model, but it is good practice to include them.

# No gradients are calculated on PyTorch operations inside the `with no_grad()` block. This causes less memory to be used and speeds up computation.

# @@ The rest of the function is the same as `train`, with the removal of `optimizer.zero_grad()`, `loss.backward()` and `optimizer.step()`, as we do not update the model's parameters when evaluating.

# ================================================================================
def evaluate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
    
        for batch in iterator:

            predictions = model(batch.text).squeeze(1)
            
            loss = criterion(predictions, batch.label)
            
            acc = binary_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

# ================================================================================
# @@ We'll also create a function to tell us how long an epoch takes to compare training times between models.

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

# ================================================================================
# @@ We then train the model through multiple epochs, an epoch being a complete pass through all examples in the training and validation sets.

# @@ At each epoch, if the validation loss is the best we have seen so far, we'll save the parameters of the model and then after training has finished we'll use that model on the test set.

# ================================================================================
N_EPOCHS = 5

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):

    start_time = time.time()
    
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    # afaf 1: train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
    
    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut1-model.pt')
    
    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

# ================================================================================
# You may have noticed the loss is not really decreasing and the accuracy is poor. This is due to several issues with the model which we'll improve in the next notebook.
# 
# Finally, the metric we actually care about, the test loss and accuracy, which we get from our parameters that gave us the best validation loss.

model.load_state_dict(torch.load('tut1-model.pt'))

test_loss, test_acc = evaluate(model, test_iterator, criterion)

print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')


# Next Steps
# 
# In the next notebook, the improvements we will make are:
# - packed padded sequences
# - pre-trained word embeddings
# - different RNN architecture
# - bidirectional RNN
# - multi-layer RNN
# - regularization
# - a different optimizer
# 
# This will allow us to achieve ~84% accuracy.
