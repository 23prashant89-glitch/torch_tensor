import torch
# x = torch.tensor(4.0, requires_grad=True)
# y = x ** 2 + 3 * x 
# y.backward()
# a = x.grad
# print(a)

# x = torch.tensor(5.0, requires_grad=True)
# y = x ** 2 
# z = y + 2 * x + 1
# z.backward()
# a = x.grad
# print(a)
# a = torch.rand(3,3)
# print(a)
# b = torch.transpose(a, 0, 1)
# print(b)
# print(b.shape)
# print(a.shape)
# print(torch.equal(a, b))
# Manual derivative of x^2
# dy/dx = 2*x

# Verify with PyTorch
# x = torch.tensor(2.0, requires_grad=True)  # Choose a value for x
# y = x ** 2
# y.backward()
# manual_derivative = 2 * x.item()
# pytorch_derivative = x.grad.item()

# print(f"Manual derivative: {manual_derivative}")
# print(f"PyTorch derivative: {pytorch_derivative}")
# x = torch.tensor([1.0, 2.0, 3.0, 4.0])
# y_true = torch.tensor([2.0, 4.0, 6.0, 8.0])

# w = torch.tensor(2.0, requires_grad= True)

# print(f"Initial weightf: {w.item()}")

# for epoch in range(3):
#     y_pred = w * x
#     loss =(y_pred - y_true).pow(2).mean()

#     loss.backward()
# print(f"\nEpoch {epoch+1}:")
# print(f"loss: {loss.item():.4f}")
# print(f"Gradient of loss w.r.t w: {w.grad.item()}")

# with torch.no_grad():
#     w -= 0.1 * w.grad
#     w.grad.zero_()
#     print(f"Updated weight: {w.item()}")

# print("-" * 20)
# print(f"Final weight after training: {w.item():.4f}")   

# x = torch.tensor(2.0, requires_grad = True)
# y1 = x ** 2
# y1.backward()
# print(f"Gradient of y1 w.r.t x: {x.grad.item()}")  # Should print 4.0
# x.grad.zero_()
# y2 = x ** 3
# y2.backward()
# print(f"Gradient of y2 w.r.t x: {x.grad.item()}")

# w = torch.tensor(1.0, requires_grad = True)
# x = torch.tensor(2.0)
# y = torch.tensor(6.0)

# y_prd = w * x 
# loss = (y_prd - y ) ** 2
# loss.backward()

# print("gradient", w.grad)
# print("loss", loss)

# x = torch.tensor(6.7, requires_grad = True)
# y = torch.tensor(0.0, requires_grad = True)

# w = torch.tensor(-0.5, requires_grad = True)
# b = torch.tensor(0.0, requires_grad = True)

# z = w * x + b
# y_pred = torch.sigmoid(z)

# loss = binary_cross_entropy = - (y * torch.log(y_pred) + (1 - y) * torch.log(1 - y_pred))

# loss.backward()
# print("Gradient of loss w.r.t w:", w.grad)
# print("Gradient of loss w.r.t b:", b.grad)
# a = torch.tensor(3.0, requires_grad=True)
# y = a ** 2
# z = torch.sin(y)
# z.backward()
# print(a.grad)  # Should print the gradient of z w.r.t a
# x = torch.tensor(4.0, requires_grad = True)
# y = x ** 2 + 3 * x + 2
# y.backward()
# print(x.grad)  # Should print 11.0
# # Verify: dy/dx = 6x + 2, at x=4 → 6(4) + 2 = 26
# x2 = torch.tensor(4.0, requires_grad=True)
# y2 = 3 * x2 ** 2 + 2 * x2
# y2.backward()
# print(x2.grad)  # Should print 26.0
# x = torch.tensor(5.0, requires_grad=True)
# y = x ** 4
# y.backward()
# print(x.grad)  # Should print tensor(500.0) because dy/dx =
# x = torch.tensor(2.0, requires_grad=True)
# y = x * 3

# z = y ** 2
# z.backward()
# print(x.grad)  # Should print tensor(36.0) because dz/dx = 18x, at x=2 → 18*2 = 36
# w = torch.tensor(0.5, requires_grad=True)
# x = torch.tensor(3.0)
# y = torch.tensor(9.0)

# for epoch in range(5):
#     y_pred = w * x
#     loss = (y_pred - y) ** 2

#     loss.backward()
#     print(f"\nEpoch {epoch+1}:")
#     print(f"loss: {loss.item():.4f}")
#     print(f"Gradient of loss w.r.t w: {w.grad.item()}")

#     with torch.no_grad():
#         w -= 0.1 * w.grad
#         w.grad.zero_()
#         print(f"Updated weight: {w.item()}")