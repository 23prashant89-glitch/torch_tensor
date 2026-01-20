import torch
import torch.nn as nn

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(in_features=1, out_features=1)

    def forward(self, x):
        out = self.layer1(x)
        return out
    
# Example usage:
model = LinearRegressionModel()
print(model)

x = torch.tensor([[2.0], [4.0], [6.0], [8.0]])
y_true = torch.tensor([[4.0], [8.0], [12.0], [16.0]])

print(f"Initial weights: {model.layer1.weight.item()}, bias: {model.layer1.bias.item()}")

print("parameters")

for name, param in model.named_parameters():
    print(f"{name}, \n{param.data},\n{param.shape}, \n{param.item()}")
y_pred = model(x)

# Calculate loss (Mean Squared Error)
loss = nn.MSELoss()(y_pred, y_true)
print(f"Loss: {loss.item()}")

# Compare predictions with true values
print(f"True values: \n{y_true}")
print(f"Predicted values: \n{y_pred}")
print(f"Difference: \n{y_true - y_pred}")
# Set the number of epochs
num_epochs = 1000
learning_rate = 0.01

# Define the optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    optimizer.zero_grad()  # Zero the gradients
    y_pred = model(x)  # Forward pass
    loss = nn.MSELoss()(y_pred, y_true)  # Calculate loss
    loss.backward()  # Backward pass
    optimizer.step()  # Update weights

    if (epoch + 1) % 100 == 0:  # Print every 100 epochs
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

