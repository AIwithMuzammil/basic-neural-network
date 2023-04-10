from sklearn.datasets import load_iris

# loading the dataset
# 150 samples, 3 categories/class/labels, 4 features (length/width of sepals and petals)
iris_dataset = load_iris()

X_data = iris_dataset.data
y_labels = iris_dataset.target

# preprocess the dataset
from sklearn.preprocessing import StandardScaler, OneHotEncoder
scaler = StandardScaler()
encoder = OneHotEncoder()

X_data_scaled = scaler.fit_transform(X_data) # delivers scaled data
y_labels_encoded = encoder.fit_transform(y_labels.reshape(-1, 1)).toarray() # provides OHE labels

# splitting the data into test/train, X mean data, y means labels
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test  = train_test_split(X_data_scaled, y_labels_encoded, test_size=0.25)

# define the model
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


# Evaluate the model
with torch.no_grad(): # do not compute the gradients
    outputs = model(torch.Tensor(X_test))
    _, predictions = torch.max(outputs.data, 1)
    
    y_labels = torch.max(torch.Tensor(y_test), 1)[1]
    accuracy = (predictions == y_labels).sum().item() / len(y_test)
    print(f"The accuracy on the test set is {accuracy*100:.2f}%")
