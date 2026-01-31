import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np    

# x_train = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float32)
# y_train = torch.tensor([[2.0], [4.0], [6.0], [8.0]], dtype=torch.float32)


# x_val = torch.tensor([[5.0], [6.0]], dtype=torch.float32)
# y_val = torch.tensor([[10.0], [12.0]], dtype=torch.float32)

# class SimpleNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.linear = nn.Linear(1, 1)

#     def forward(self, x):
#         out = self.linear(x)
#         return out
    
# model = SimpleNet()
# criterion = nn.MSELoss()
# optimizer = optim.SGD(model.parameters(), lr=0.01)
# num_epochs = 100
# train_losses = []
# best_val_loss = float('inf')
# print("Starting training...")

# for epoch in range(num_epochs):
#     model.train()
#     predictions = model(x_train)
#     loss = criterion(predictions, y_train)
#     optimizer.zero_grad()
#     optimizer.step()

#     train_losses.append(loss.item())

#     model.eval()

#     with torch.no_grad():
#         val_predictions = model(x_val)
#         val_loss = criterion(val_predictions, y_val)


#     if val_loss.item() < best_val_loss:
#         best_val_loss = val_loss.item()
#         torch.save(model.state_dict(), 'best_model.pth')

#     if (epoch+1) % 10 == 0:
#         print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

#     # load model
# model.load_state_dict(torch.load('best_model.pth'))
# print("Training complete. Best model loaded.")  

# import matplotlib.pyplot as plt
# plt.plot(range(num_epochs), train_losses, label='Training Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.grid()
# plt.xlim(0, num_epochs)
# plt.ylim(0, max(train_losses))
# plt.show()

# class SipleNet(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(SipleNet, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(hidden_size, output_size)

#     def forword(self, x):
#         out = self.fc1(x)
#         out = self.relu(out)
#         out = self.fc2(out)
#         return out
    
# def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
# model = nn.Linear(1, 1)
# loss = nn.MSELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# x = torch.randn(100, 1)
# y = 3 * x + 2 + 0.1 * torch.randn(100)


# x_val = torch.randn(20, 1)
# y_val = 3 * x_val + 2 + 0.1 * torch.randn(20)

# for epoch in range(20):
#     model.train()
#     optimizer.zero_grad()


#     y_pred = model(x)
#     train_loss = loss(y_pred, y)
#     train_loss.backward()
#     optimizer.step()

# model.eval()
# with torch.no_grad():
#     y_val_pred = model(x_val)
#     val_loss = loss(y_val_pred, y_val)
# print(f'Epoch {epoch+1}, Validation Loss: {val_loss.item():.4f}')


# torch.save(model.state_dict(), 'linear_model.pth')
# print("Model saved.")

# model = nn.Linear(1, 1)
# model.load_state_dict(torch.load('linear_model.pth'))
# model.eval()

# print(model.state_dict())
# class SimpleNet(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(SimpleNet, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(hidden_size, output_size)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         return x
# # train the model
# def train_model(model, train_loader, criterion, optimizer, device):
#     model.train()
#     running_loss = 0.0
#     correct = 0
#     total = 0
#     for inputs, labels in train_loader:
#         inputs, labels = inputs.to(device), labels.to(device)

#         optimizer.zero_grad()

#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()
#         _, predicted = torch.max(outputs, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#     epoch_loss = running_loss / len(train_loader)
#     epoch_acc = correct / total
#     return epoch_loss, epoch_acc
# # validate the model
# def validate(model, val_loader, criterion, device):
#     model.eval()
#     running_loss = 0.0
#     correct = 0
#     total = 0

#     with torch.no_grad():
#         for inputs, labels in val_loader:
#             inputs, labels = inputs.to(device), labels.to(device)

#             outputs = model(inputs)
#             loss = criterion(outputs, labels)

#             running_loss += loss.item()
#             _, predicted = torch.max(outputs, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()

#     val_loss = running_loss / len(val_loader)
#     val_acc = 100*correct / total
#     return val_loss, val_acc

# # complete training loop

# def training_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
#     best_val_acc = 0.0

#     for epoch in range(num_epochs):
#         train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)


#         val_loss, val_acc = validate(model, val_loader, criterion, device)

#         print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
#         print('-' * 50)

#         if val_acc > best_val_acc:
#             best_val_acc = val_acc
#             torch.save({
#                 'epoch': epoch + 1,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'val_acc': best_val_acc,
#                 'val_loss': val_loss,
#                 'train_loss': train_loss,
#             }, 'best_model.pth')
#             print("Training complete. Best model saved.€--->> €{best_val_acc:.4f}%")

# # load the best model
# def load_best_model(model, optimizer, device):
#     checkpoint = torch.load('best_model.pth', map_location=device)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     epoch = checkpoint['epoch']
#     val_acc = checkpoint['val_acc']
#     val_loss = checkpoint['val_loss']
#     train_loss = checkpoint['train_loss']
#     print(f'Loaded best model from epoch {epoch+1} with Val Acc: {val_acc:.4f}, Val Loss: {val_loss:.4f}, Train Loss: {train_loss:.4f}')
#     return model, optimizer

# # Example usage
# if __name__ == "__main__":
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f'Using device: {device}')
#     x_train = torch.randn(1000, 10)
#     y_train = torch.randint(0, 3, (1000,))
#     x_val = torch.randn(200, 10)
#     y_val = torch.randint(0, 3, (200,))

#     train_dataset = TensorDataset(x_train, y_train)
#     val_dataset = TensorDataset(x_val, y_val)

#     train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
#     val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)

#     model = SimpleNet(input_size=10, hidden_size=64, output_size=3).to(device)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001)

#     num_epochs = 20
#     training_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)

#     model, optimizer = load_best_model(model, optimizer, device)
#     model.eval()
#     print("training complete. Best model loaded.")

# Example usage
# if __name__ == "__main__":
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f'Using device: {device}')
    
#     x_train = torch.randn(1000, 10)
#     y_train = torch.randint(0, 3, (1000,))
#     x_val = torch.randn(200, 10)
#     y_val = torch.randint(0, 3, (200,))

#     train_dataset = TensorDataset(x_train, y_train)
#     val_dataset = TensorDataset(x_val, y_val)

#     train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
#     val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)

#     model = SimpleNet(input_size=10, hidden_size=64, output_size=3).to(device)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001)

#     num_epochs = 30
    
#     for epoch in range(num_epochs):
#         train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
#         val_loss, val_acc = validate(model, val_loader, criterion, device)
        
#         print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create synthetic data
    x_train = torch.randn(1000, 10)
    y_train = torch.randint(0, 3, (1000,))
    x_val = torch.randn(200, 10)
    y_val = torch.randint(0, 3, (200,))

    # Create data loaders
    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Simple model
    model = nn.Linear(10, 3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 30
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        # Save model at epoch 20
        if epoch + 1 == 20:
            torch.save(model.state_dict(), 'model_epoch_20.pth')
            print("Model saved at epoch 20")

    # Reload and test
    model_loaded = nn.Linear(10, 3).to(device)
    model_loaded.load_state_dict(torch.load('model_epoch_20.pth'))
    model_loaded.eval()

    # Test with same input
    test_input = x_val[:5].to(device)
    with torch.no_grad():
        original_output = model(test_input)
        loaded_output = model_loaded(test_input)

    print("\nOriginal model output:")
    print(original_output)
    print("\nLoaded model output:")
    print(loaded_output)
    print("\nOutputs match:", torch.allclose(original_output, loaded_output))
    # Validation WITHOUT model.eval()
    print("\n" + "="*50)
    print("Validation WITHOUT model.eval()")
    print("="*50)
    model.train()  # Keep model in training mode
    val_loss_train_mode = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss_train_mode += loss.item()
    val_loss_train_mode /= len(val_loader)
    print(f"Val Loss (in train mode): {val_loss_train_mode:.4f}")
    
    # Validation WITH model.eval()
    print("\n" + "="*50)
    print("Validation WITH model.eval()")
    print("="*50)
    model.eval()  # Set to evaluation mode
    val_loss_eval_mode = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss_eval_mode += loss.item()
    val_loss_eval_mode /= len(val_loader)
    print(f"Val Loss (in eval mode): {val_loss_eval_mode:.4f}")
    
    # Comparison
    print("\n" + "="*50)
    print("DIFFERENCE OBSERVED:")
    print("="*50)
    print(f"Loss difference: {abs(val_loss_train_mode - val_loss_eval_mode):.6f}")
    print("Note: Dropout and BatchNorm behave differently in train vs eval mode")
    print("In eval mode: Dropout is disabled, BatchNorm uses running statistics")
    print("In train mode: Dropout is active, BatchNorm uses batch statistics")