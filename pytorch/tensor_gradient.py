
import torch
"""
PyTorch Tensor Operations and Gradient Computation Module
This module demonstrates fundamental PyTorch operations including:
1. Tensor Creation and Initialization:
  - Creating tensors from lists, numpy arrays, and random initialization
  - Tensor properties: shape, dtype, device
2. Tensor Operations:
  - Element-wise operations: addition, subtraction, multiplication, division
  - Dot product: torch.dot(a, b) computes the scalar result of a·b
    Example: [1,2]·[3,4] = (1*3) + (2*4) = 11
  - Matrix multiplication: torch.matmul() for matrix-vector and matrix-matrix products
3. Tensor Reshaping:
  - reshape(): Changes tensor dimensions without changing data
    Example: tensor shape (2,) reshaped to (2,1) for matrix multiplication
    a.reshape(2,1) converts row vector to column vector for proper matmul operation
4. Automatic Differentiation:
  - Gradient computation using .backward()
  - Gradient accumulation with requires_grad=True
  - Manual vs PyTorch gradient verification
  - Chain rule application for composite functions
5. Device Management:
  - GPU/CPU tensor allocation using .to(device)
  - Device availability checking with torch.cuda.is_available()
6. NumPy Integration:
  - Converting NumPy arrays to PyTorch tensors using torch.from_numpy()
  - Seamless interoperability between NumPy and PyTorch
"""
# import subprocess
# import sys
# # Check if a specific library is installed
# result = subprocess.check_output([sys.executable, "-m", "pip", "list"])
# print(result.decode())
# # Check if PyTorch is installed
# try:
#     print(f"PyTorch is installed. Version: {torch.__version__}")
# except ImportError:
#     print("PyTorch is not installed")

# a = torch.tensor(5)
# # a.requires_grad_(True)
# # b = a ** 2
# # b.backward()
# # print(a.grad)  # Should print tensor(10)
# b = torch.tensor([3,5,7])
# c = torch.tensor([[2,4],[6,8]])
# print( f"\n{b} ,\n{c} , \n{a}")

  # Should print tensor([10., 14., 18.])
# x = torch.zeros(3,3)  
# print(x)
# y = torch.ones(2,4)
# print(y)
# z = torch.rand(5,2)
# print(z)
# w = torch.randn(4,4)    
# print(w)

# x = torch.tensor([1,2,3])
# y = torch.tensor([7,8,9])

# print (x + y)
# print(x*y)
# print(x-y)
# print(x/y)
# print(torch.dot(x,y.T))  # Matrix multiplication
# print(torch.matmul(x,y.T))  # Matrix multiplication
# print(x.shape)
# print(x.dtype)
# print(x.device)

# x = torch.tensor(2.0, requires_grad=True)
# y = x ** 2 + 3 * x + 1

# y.backward()
# print(x.grad)  # Should print tensor([5., 7., 9.])

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")
# a = torch.tensor(3.0, requires_grad=True).to(device)
# b = a ** 3 + 2 * a ** 2 + a
# b.backward()
# print(a.grad)  # Should print tensor(35.)
# a = torch.rand(3,4)
# print(a)
# b = torch.transpose(a, 0,1)
# print(b)
# print(f"Original shape: {a.shape}")
# print(f"Transposed shape: {b.shape}")
# print(f"Are they equal? {torch.equal(a.T, b)}")
# # Transpose swaps rows and columns
# # Original: shape (3,3) → Transposed: shape (3,3)
# # Element at position [i,j] moves to [j,i]
# c = a.reshape(4, 3)
# d = torch.transpose(c, 0, 2)  # Swap dimensions 0 and 2
# print(f"Reshaped: {c.shape}")
# print(f"Transposed: {d.shape}")
# a = torch.tensor(5.0, requires_grad=True)
# b = a ** 2
# b.backward()
# print(a.grad)  # Should print tensor(10.)
# Task 1: Manual vs PyTorch Gradient
# For f(x) = x², derivative is f'(x) = 2x

# Manual calculation
# x_val = 3.0
# manual_derivative = 2 * x_val
# print(f"Manual derivative of x² at x={x_val}: {manual_derivative}")

# # PyTorch verification
# x = torch.tensor(x_val, requires_grad=True)
# y = x ** 2
# y.backward()
# print(f"PyTorch gradient of x² at x={x_val}: {x.grad}")
# print(f"Match: {manual_derivative == x.grad.item()}")
# w = torch.tensor(1.0, requires_grad=True)
# x = torch.tensor(2.0)
# y = torch.tensor(4.0)
# y_pred = w * x 
# loss = (y_pred - y) ** 2
# loss.backward()
# print(f"Gradient of loss w.r.t w: {w.grad}")  # Should print tensor(-12.0)
# for i in range(10):
#     x = torch.tensor(float(i), requires_grad=True)
#     y = x ** 3 + 2 * x ** 2 + x
#     y.backward()
#     print(f"At x={i}, dy/dx={x.grad}")
import numpy as np

# x = torch.tensor([1, 2, 3])
# y = torch.ones(3,3)
# z = torch.rand(2,2)

# print(x)
# print(y.shape)
# print(z)

# np_arr = np.array([5,10,15])
# tensor_from_np = torch.from_numpy(np_arr)
# print(tensor_from_np)

# a = torch.tensor([1,2])
# b = torch.tensor([3,4])
# # print(a+b)
# # print(a*b)
# print(torch.dot(a,b))
# print(torch.matmul(a.reshape(2,1), b.reshape(1,2)))
# # print(a-b)
# # print(a/b)
# # reshape() changes tensor dimensions without changing the data
# # Example: a has shape (2,) - a 1D tensor with 2 elements

# # Reshaping a to (2,1) - converts row vector to column vector
# a_reshaped = a.reshape(2, 1)
# print(f"Original shape: {a.shape}")  # (2,)
# print(f"Reshaped shape: {a_reshaped.shape}")  # (2, 1)
# print(f"Original data: {a}")
# print(f"Reshaped data:\n{a_reshaped}")

# # This creates a NEW tensor with the same data but different dimensions
# # reshape() returns a new view or copy depending on memory layout
# # Use .clone() if you need a completely independent copy
# a_copy = a.reshape(2, 1).clone()
# t = torch.rand(4,4)
# print(t.shape)
# print(t)

# flat = t.view(16)
# print(flat.shape)
# print(flat)
# new_shape = flat.reshape(2,8)
# print(new_shape.shape)
# print(new_shape)

# if torch.cuda.is_available():
#     device = torch.device("cuda")
#     print("Using GPU")
    
# else:
#     device = torch.device("cpu")
#     print("Using CPU")

# x = torch.tensor(3.0, requires_grad=True).to(device)
# y = x ** 3 + 2 * x ** 2 + x
# y.backward()
# print(x.grad)  # Should print tensor(38.)
# print(f"Device of x: {x.device}")
# a = torch.empty(3,3)
# print(a)
# b = torch.zeros(2,4)
# print(b)
# c = torch.rand(5,2)
# print(c)
# d = torch.randn(4,4)
# print(d)
# x = torch.manual_seed(10)
# print(x)
# y = torch.rand(2,3)
# print(y)
# a = torch.arange(0, 10, 2)
# print(a)

# a = torch.tensor([[1,2,3], [3,4,6]])
# print(a)
# # z_SHAPE = (3,2)
# # b = a.reshape(z_SHAPE)

# y = torch.empty_like(a)
# print(y)
# z = a.dtype
# print(z)
# Linear Equation: y = w * x + b
# x = torch.tensor(10.0)
# w = torch.tensor(2.0, requires_grad=True)
# b = torch.tensor(3.0, requires_grad=True)
# y_true = torch.tensor(30.0)

# # Forward pass
# y_pred = w * x + b

# # Calculate loss
# loss = (y_pred - y_true) ** 2

# # Backward pass
# loss.backward()

# # Print gradients
# print(f"y_pred: {y_pred.item()}")
# print(f"Loss: {loss.item()}")
# print(f"Gradient of w: {w.grad}")
# print(f"Gradient of b: {b.grad}")
