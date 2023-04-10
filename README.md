# Neural Networks from Scratch in Python | Full Machine Learning Tutorial for Beginners

Neural networks are one of the most popular machine learning models used for classification tasks. In this tutorial, we will learn how to write a neural network model in PyTorch to classify the famous Iris dataset. The Iris dataset is a classic dataset used for classification tasks, and it consists of 150 samples with three categories/classes/labels and four features (length/width of sepals and petals).

We will first preprocess the dataset using standardization and one-hot encoding, and then split it into train and test sets. Next, we will define the neural network model using PyTorch, which consists of two linear layers with ReLU and softmax activations. We will then train the model using the train set and evaluate its performance on the test set.

By the end of this tutorial, you will have a good understanding of how to write a neural network model using PyTorch for classification tasks and how to preprocess and split datasets to train and test machine learning models.

Here's the YouTube video tutorial: [Neural Networks from Scratch in Python | Full Machine Learning Tutorial for Beginners](https://youtu.be/Cc9mpMuk4sY).

<a href="https://www.youtube.com/watch?v=Cc9mpMuk4sY" target="_blank">
  <img src="https://img.youtube.com/vi/Cc9mpMuk4sY/0.jpg" alt="Neural Networks from Scratch in Python | Full Machine Learning Tutorial for Beginners" width="560" height="315" border="0"/>
</a>

## Step 1: Loading the Iris dataset
The first step is to load the Iris dataset using the load_iris() function from the sklearn.datasets module. This function returns an object containing the data and target labels for the dataset. The dataset consists of 150 samples, each with 4 features: length and width of sepals and petals.

```python
from sklearn.datasets import load_iris

iris_dataset = load_iris()
X_data = iris_dataset.data
y_labels = iris_dataset.target
```

## Step 2: Preprocessing the dataset
The next step is to preprocess the dataset. We use two preprocessing techniques:

1. StandardScaler - It scales the data by subtracting the mean and dividing by the standard deviation of each feature, so that each feature has a mean of 0 and a standard deviation of 1.
2. OneHotEncoder - It encodes the target labels using one-hot encoding, which is a process of converting categorical data (such as the target labels) into numerical data.
```python
from sklearn.preprocessing import StandardScaler, OneHotEncoder

scaler = StandardScaler()
encoder = OneHotEncoder()

X_data_scaled = scaler.fit_transform(X_data)
y_labels_encoded = encoder.fit_transform(y_labels.reshape(-1, 1)).toarray()
```

## Step 3: Splitting the data into train and test sets
The next step is to split the preprocessed data into training and testing sets. We use the train_test_split() function from the `sklearn.model_selection` module to randomly split the data into training and testing sets. The test set size is set to 0.25, which means 25% of the data is used for testing and 75% of the data is used for training.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test  = train_test_split(X_data_scaled, y_labels_encoded, test_size=0.25)
```

## Step 4: Defining the neural network model
The next step is to define the neural network model. We define a simple neural network model with two fully connected layers (`nn.Linear()`), each followed by a ReLU activation function (`torch.relu()`) except for the final layer, which is followed by a softmax activation function (`torch.softmax()`) since we are doing multi-class classification. The model takes in 4 features as input and outputs a probability distribution over the 3 classes.

```python
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        self.fc1 = nn.Linear(4, 10) # inputs = 4, outputs = 10
        self.fc2 = nn.Linear(10, 3) # inputs = 10, outputs = 3

    def forward(self, x):
        # first layer
        x = self.fc1(x) # x.size() = 10
        x = torch.relu(x) # x.size() = 10
        
        # second layer
        x = self.fc2(x) # x.size() = 3
        x = torch.softmax(x, dim=1)

        return x

model = Model()
```

## Step 5: Training the model
The next step is to train the neural network model. We use the `nn.CrossEntropyLoss()` function as the loss function and the `torch.optim.Adam()` optimizer with a learning rate of `lr=0.01`. We run the training loop for 10 epochs, where in each epoch we compute the loss, compute gradients, and update the network parameters based on the computed gradients.

```python
# Train the model
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

n_epochs = 10
for epoch in range(n_epochs):
    optimizer.zero_grad() # avoid existing grad values in the optimizer
    y_labels = torch.max(torch.Tensor(y_train), 1)[1]

    outputs = model(torch.Tensor(X_train)) # predictions/outputs of the model
    loss = criterion(outputs, y_labels) # computes the loss
    loss.backward() # compute gradients
    optimizer.step() # update the parameters

    print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
```

## Step 06: Evaluating the Model
After training the model, the next step is to evaluate the model using the test dataset. For that, we first turn off gradient computation using the `torch.no_grad()` function to speed up computation. We then compute the predicted outputs of the model using the test dataset and compare them to the actual labels to calculate the accuracy of the model.
```python
# Evaluate the model
with torch.no_grad(): # do not compute the gradients
    outputs = model(torch.Tensor(X_test))
    _, predictions = torch.max(outputs.data, 1)
    
    y_labels = torch.max(torch.Tensor(y_test), 1)[1]
    accuracy = (predictions == y_labels).sum().item() / len(y_test)
    print(f"The accuracy on the test set is {accuracy*100:.2f}%")
```
The accuracy on the test set is then printed to the console.

# Conclusion
In this tutorial, we built a basic neural network model for machine learning. We learned how to preprocess the Iris dataset using standardization and one-hot encoding, split it into train and test sets, define a neural network model using PyTorch, train the model using the train set, and evaluate the model using the test set. This is an amazing beginner-level example of how to use PyTorch to train a classification model. With some modifications, you can use this code as an ignition point for training more complex models on different datasets with different real-world scenarios.

The full code can be found from this repo in the file `basic-neural-network.py`. Happpy coding!

