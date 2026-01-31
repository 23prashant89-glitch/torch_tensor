import torch 
import torch.nn as nn
import torch.optim as optim

# x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
# y = torch.tensor([[5.0], [8.0], [11.0], [14.0]])

# class LinearNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.linear = nn.Linear(1, 1)

#     def forward(self, x):
#         output = self.linear(x)
#         return output

# def train_model(optimizer_name, learning_rate = 1.0):
#     model = LinearNet()
#     torch.manual_seed(42)

#     criterion = nn.MSELoss()

#     if optimizer_name == 'SGD':
#         optimizer = optim.SGD(model.parameters(), lr = learning_rate)
#     else:
#         optimizer_name == 'Adam'
#         optimizer = optim.Adam(model.parameters(), lr = learning_rate)

#     print(f"\nTraining with {optimizer_name} optimizer:")
    
#     for epoch in range(100):
#         y_pred = model(x)
#         loss = criterion(y_pred, y)
#         optimizer.zero_grad()

#         loss.backward()
#         optimizer.step()

#         if (epoch+1) % 10 == 0:
#             print(f"Epoch [{epoch+1}/100], Loss: {loss.item():.4f}")
#     return model    
# model_sgd = train_model('SGD', learning_rate=0.01)
# model_adam = train_model('Adam', learning_rate=0.01)


# x = torch.tensor([[10.0], [20.0], [80.0], [90.0]])
# y = torch.tensor([[0.0], [0.0], [1.0], [1.0]])

# class ClassificationNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.linear = nn.Linear(1, 1)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         output = self.linear(x)
#         output = self.sigmoid(output)
#         return output
    
# def train_classification_model(optimizer_name, learning_rate = 0.1):
#     model = ClassificationNet()
#     torch.manual_seed(42)

#     criterion = nn.BCELoss()
#     if optimizer_name == 'SGD':
#         optimizer = optim.SGD(model.parameters(), lr = learning_rate)
#     else:
#         optimizer_name == 'Adam'
#         optimizer = optim.Adam(model.parameters(), lr = learning_rate)

#     print(f"\nTraining Classification with {optimizer_name} optimizer:")
    
#     for epoch in range(100):
#         y_pred = model(x)
#         loss = criterion(y_pred, y)
#         optimizer.zero_grad()

#         loss.backward()
#         optimizer.step()

#         if (epoch+1) % 10 == 0:
#             print(f"Epoch [{epoch+1}/100], Loss: {loss.item():.4f}")
#     return model

# model_sgd = train_classification_model('SGD', learning_rate=0.01)
# model_adam = train_classification_model('Adam', learning_rate=0.01)

# loss_fn = nn.MSELoss()

# predictions = torch.tensor([[2.5], [0.0], [2.0], [8.0]])
# targets = torch.tensor([[3.0], [-0.5], [2.0], [7.0]])

# loss = loss_fn(predictions, targets)
# print(f"Mean Squared Error Loss: {loss.item():.4f}")

# loss_fn = nn.CrossEntropyLoss()
# predictions = torch.tensor([[0.2, 2.0, 0.1],
#                             [1.0, 0.1, 0.2],
#                             [0.05, 0.95, 0.0]])
# targets = torch.tensor([1, 0, 1])  # Correct class indices
# loss = loss_fn(predictions, targets)
# print(f"Cross Entropy Loss: {loss.item():.4f}")

# model = nn.Linear(1, 1)
# loss_fn = nn.MSELoss()

# optimizer = optim.SGD(model.parameters(), lr=0.01)

# x = torch.tensor([[2.0]])
# y = torch.tensor([[6.0]])

# for epoch in range(20):
#     y_pred = model(x)
#     loss = loss_fn(y_pred, y)

#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

#     print(f"Epoch [{epoch+1}/20], Loss: {loss.item():.4f}")

# model = nn.Linear(1, 1)

# x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
# y = torch.tensor([[5.0], [8.0], [11.0], [14.0]])

# y_pred = model(x)
# loss_fn = nn.MSELoss()

# loss = loss_fn(y_pred, y)
# print(f"Initial Loss: {loss.item():.4f}")

# optimizer = optim.SGD(model.parameters(), lr=0.001)

# for epoch in range(100):
#     y_pred = model(x)
#     loss = loss_fn(y_pred, y)

#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

#     if (epoch+1) % 10 == 0:
#         print(f"Epoch [{epoch+1}/100], Loss: {loss.item():.4f}")

# print(f"Trained Weights: {model.weight.item()}, Bias: {model.bias.item()}")
# Define a single model for comparison
# model = nn.Linear(1, 1)

# # Data for training
# x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
# y = torch.tensor([[5.0], [8.0], [11.0], [14.0]])

# # Function to train the model with a specified optimizer
# def train_and_compare(optimizer_name, learning_rate=0.01):
#     model = nn.Linear(1, 1)  # Reinitialize model for each optimizer
#     loss_fn = nn.MSELoss()
#     optimizer = optim.SGD(model.parameters(), lr=learning_rate) if optimizer_name == 'SGD' else optim.Adam(model.parameters(), lr=learning_rate)

#     for epoch in range(20):
#         y_pred = model(x)
#         loss = loss_fn(y_pred, y)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#     return loss.item()

# # Train with SGD
# final_loss_sgd = train_and_compare('SGD', learning_rate=0.01)
# print(f"Final Loss with SGD: {final_loss_sgd:.4f}")

# # Train with Adam
# final_loss_adam = train_and_compare('Adam', learning_rate=0.01)
# print(f"Final Loss with Adam: {final_loss_adam:.4f}")

# # Train with learning rate scheduler
# def train_with_scheduler(optimizer_name, learning_rate=0.01, epochs=30):
#     model = nn.Linear(1, 1)  # Reinitialize model for each optimizer
#     loss_fn = nn.MSELoss()
#     optimizer = optim.SGD(model.parameters(), lr=learning_rate) if optimizer_name == 'SGD' else optim.Adam(model.parameters(), lr=learning_rate)
    
#     # Learning Rate Scheduler
#     scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

#     losses = []
#     for epoch in range(epochs):
#         y_pred = model(x)
#         loss = loss_fn(y_pred, y)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#         # Step the scheduler
#         scheduler.step()

#         losses.append(loss.item())
#         if (epoch + 1) % 10 == 0:
#             print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}, Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

#     return losses

# # Train with SGD and learning rate scheduler
# losses_sgd = train_with_scheduler('SGD', learning_rate=0.01)
# print(f"Final Loss with SGD: {losses_sgd[-1]:.4f}")

# # Train with Adam and learning rate scheduler
# losses_adam = train_with_scheduler('Adam', learning_rate=0.01)
# print(f"Final Loss with Adam: {losses_adam[-1]:.4f}")

# # Plotting the loss curves
# import matplotlib.pyplot as plt

# plt.plot(losses_sgd, label='SGD Loss')
# plt.plot(losses_adam, label='Adam Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.title('Loss Curve')
# plt.legend()
# plt.show()
# with open ("training_loop.py", "w") as file:
    # print(file)