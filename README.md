# Sentiment Analysis with Simple RNN in TensorFlow

This project demonstrates how to build, train, and evaluate a basic Recurrent Neural Network (RNN) using TensorFlow/Keras for sentiment analysis on the IMDB movie reviews dataset. The goal is to classify movie reviews as either positive or negative.

## Project Overview

The Python script covers the following essential steps:

1.  **Dataset Loading and Preprocessing**: Loads the IMDB movie reviews dataset, which consists of pre-tokenized text data, and pads sequences to a uniform length.
2.  **RNN Model Building**: Defines a sequential RNN model using `Embedding`, `SimpleRNN`, and `Dense` layers.
3.  **Model Compilation**: Configures the model with an optimizer, loss function, and metrics suitable for binary classification.
4.  **Model Training**: Trains the RNN model on the preprocessed training data with validation.
5.  **Model Evaluation**: Assesses the trained model's performance on the unseen test dataset.

## Dataset

The **IMDB movie reviews dataset** is a classic dataset for binary sentiment classification. It contains 50,000 highly polarized reviews (25,000 for training, 25,000 for testing), labeled as either positive (1) or negative (0). The dataset is pre-processed, with reviews already converted into sequences of integers, where each integer represents a specific word.

### Data Preprocessing

* **Vocabulary Size (`vocab_size`)**: Set to 10,000, meaning only the 10,000 most frequent words in the dataset will be considered. Less frequent words are discarded.
* **Maximum Sequence Length (`max_len`)**: Set to 200. All movie review sequences are either truncated or padded with zeros to this fixed length.
    * `padding="post"`: Zeros are added at the end of sequences shorter than `max_len`. Longer sequences are truncated from the beginning.
* **Loading Data**: `imdb.load_data(num_words=vocab_size)` directly loads the pre-processed integer sequences.

## RNN Model Architecture

The RNN model is built using `tf.keras.models.Sequential` and includes the following layers:

1.  **`Embedding(input_dim=vocab_size, output_dim=128)`**:
    * **Purpose**: This layer is crucial for handling textual data. It converts positive integer indices (representing words) into dense vectors of fixed size.
    * `input_dim=vocab_size`: The size of the vocabulary (10,000 unique words).
    * `output_dim=128`: The dimensionality of the dense embedding. Each word will be represented by a 128-dimensional vector. This layer effectively learns a numerical representation for each word.
2.  **`SimpleRNN(128, activation='tanh', return_sequences=False)`**:
    * **Purpose**: This is the core recurrent layer. A `SimpleRNN` processes sequences one element at a time, maintaining an internal state that captures information from previous elements.
    * `128`: The number of units (neurons) in the RNN layer. This determines the dimensionality of the output space of the recurrent layer.
    * `activation='tanh'`: The hyperbolic tangent activation function, commonly used in RNNs.
    * `return_sequences=False`: This is important. When `False`, the RNN layer returns only the output of the *last* time step for each input sequence. This is suitable for classification tasks where a single prediction is made per sequence. If `True`, it would return the output for each time step, which is useful for sequence-to-sequence tasks.
3.  **`Dense(1, activation='sigmoid')`**:
    * **Purpose**: This is the output layer for binary classification.
    * `1`: A single neuron, as it's a binary classification problem.
    * `activation='sigmoid'`: The sigmoid activation function, which squashes the output to a value between 0 and 1, representing the probability of the positive class.

The `model.summary()` output provides a concise overview of each layer, including its output shape and the number of trainable parameters.

## Training and Evaluation

### Compilation

The model is compiled with:

* **Optimizer**: `'adam'` (Adaptive Moment Estimation), a popular and efficient optimization algorithm for deep learning.
* **Loss Function**: `'binary_crossentropy'`, which is the standard loss function for binary classification problems.
* **Metrics**: `['accuracy']` to monitor the classification accuracy during training and evaluation.

### Training Process

The model is trained for `5 epochs` using a `batch_size` of 32. A `validation_split` of 0.2 means 20% of the training data is automatically set aside by Keras to monitor the model's performance on unseen data during training, helping to detect overfitting.

### Evaluation

After training, the model's final performance is evaluated on the separate `X_test` and `y_test` datasets to get an unbiased assessment of its generalization capability. The test loss and accuracy are printed.

## Setup and Usage

### Prerequisites

* Python 3.x
* `tensorflow` (including Keras)
* `numpy` (usually installed as a dependency)

You can install the required libraries using pip:

```bash
pip install tensorflow numpy