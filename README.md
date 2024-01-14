# Sequence to sequence model for machine translation

In this repository, an encoder-decoder model with attention is implmented to perform machine translation task on the tensorflow Spanish to English dataset. The model is implemented and trained using the pytorch module. 

## Model architechture

The model is mainly composed of a RNN encoder and a RNN decoder. The RNN encoder consists of a word embedding layer with embedding dimension of 128. After the embedding, the data is then passed into a GRU layer with hidden size of 128. Dropout with probability 0.1 is then applied during training. 

The encoder output is then passed into the attention class, which is implmented based on the classic Bahdanau attention mechanism. The context and the attention weights are then returned. 

In the decoder unit, the target sentence is passed into the embedding layer with dimension 128. The result is then concatenated with the context vector from the attention unit. This is then passed into a GRU unit and a linear unit with log softmax activation to output the logits. The logits is then used for word prediction. 

<div align="center">
  <img src="Graphics/Arch.png" width="30%">
</div>


## Preprocessing

Both the input and target sentence is normalized to unicode and lowercased. At the beginning and end of each sentence, a start of sentence token '<sos>' and an end of sentence token '<eos>' is inserted. The list of preprocessed sentences is then fitted to the 'Vocab' class, which creates a list of dictionaries. Each word is only added to the dictionary with the number of occurence in the sentences is greater than 2. When each patch of data is retrieved, each sentence is then padded to the same length. 

## Training

For training, 50000 sentences are used, with train, validation, and test ratio being 0.6, 0.2, 0.2, respectively. In Google collab, T4 GPU is used to increase the speed of training. The adam optimizer is used with fixed learning rate 0.001, and the negative log-likelihood loss function is used. It is experimented using learning rate of 0.01, however, the validation loss quickly overshoots the minimum within the first few epochs. Different hidden state size/embedding size is also experiemented (32, 64, 128, 256, 512), and it is confirmed that hidden size and embedding size of 128 yields the optimum model. The training and validation curve is shown below

From the graph, we see that after 10-15 epochs, the validation loss reaches the minimum. The model parameters at epoch 13 is used for testing. The validation loss is given by 0.742. 

## Model evaluation



