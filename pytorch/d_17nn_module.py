import torch
import torch.nn as nn

# class LinearRegressionModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.layer1 = nn.Linear(in_features=1, out_features=1)

#     def forward(self, x):
#         out = self.layer1(x)
#         return out
    
# # Example usage:
# model = LinearRegressionModel()
# print(model)

# x = torch.tensor([[2.0], [4.0], [6.0], [8.0]])
# y_true = torch.tensor([[4.0], [8.0], [12.0], [16.0]])

# print(f"Initial weights: {model.layer1.weight.item()}, bias: {model.layer1.bias.item()}")

# print("parameters")

# for name, param in model.named_parameters():
#     print(f"{name}, \n{param.data},\n{param.shape}, \n{param.item()}")
# y_pred = model(x)

# # Calculate loss (Mean Squared Error)
# loss = nn.MSELoss()(y_pred, y_true)
# print(f"Loss: {loss.item()}")

# # Compare predictions with true values
# print(f"True values: \n{y_true}")
# print(f"Predicted values: \n{y_pred}")
# print(f"Difference: \n{y_true - y_pred}")
# # Set the number of epochs
# num_epochs = 1000
# learning_rate = 0.01

# # Define the optimizer
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# # Training loop
# for epoch in range(num_epochs):
#     model.train()  # Set the model to training mode
#     optimizer.zero_grad()  # Zero the gradients
#     y_pred = model(x)  # Forward pass
#     loss = nn.MSELoss()(y_pred, y_true)  # Calculate loss
#     loss.backward()  # Backward pass
#     optimizer.step()  # Update weights

    # if (epoch + 1) % 100 == 0:  # Print every 100 epochs
    #     print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# class MyNetwork(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.layer = nn.Linear(1, 1)

#     def forward(self, x):
#         out = self.layer(x)
#         return out
    

# # Example usage:
# output = MyNetwork()
# print(output)
# x = torch.tensor([[2.0]])
# y = torch.tensor([[6.0]])

# y_pred = output(x)
# loss = (y_pred - y) ** 2
# print(loss)
# loss.backward()
# print(output.layer.weight.grad)
# print(output.layer.bias.grad)
# print(y_pred.item())
import numpy as np
import pandas as pd 
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
x_train, x_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df['target'], test_size=0.2)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.transform(y_test)

x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)
x_test = torch.from_numpy(x_test)
y_test = torch.from_numpy(y_test)

class MyNN(nn.Module):
    def __init__(self, x):
        self.weight = torch.rand(x.shape[1], 1, dtype=torch.float32, requires_grad=True)
        self.bias = torch.rand(1, dtype=torch.float32, requires_grad=True)

    def forward(self, x):
        z = torch.matmul(x, self.weight) + self.bias
        y_pred = torch.sigmoid(z)
        return y_pred
    def loss(self, y_pred, y):
        epsilon = 1e-7
        y_pred = torch.clamp(y_pred, epsilon, 1 - epsilon)
        loss = -torch.mean(y_train * torch.log(y_pred) + (1 - y) * torch.log(1 - y_pred))
        return loss

learning_rate = 0.01
num_epochs = 100

model = MyNN(x_train)
model.weight
model.bias
for epoch in range(num_epochs):
    y_pred = model.forward(x_train.float())
    loss = model.loss(y_pred, y_train.float())
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    loss.backward()
    with torch.no_grad():
        model.weight -= learning_rate * model.weight.grad
        model.bias -= learning_rate * model.bias.grad

        model.weight.grad.zero_()
        model.bias.grad.zero_()

    # model evaluation
    with torch.no_grad():
        y_test_pred = model.forward(x_test.float())
        y_test_pred_class = (y_test_pred >= 0.5).float()
        accuracy = accuracy_score(y_test.float(), y_test_pred_class)
        print(f"Epoch {epoch+1}, Test Accuracy: {accuracy:.4f}")
# 