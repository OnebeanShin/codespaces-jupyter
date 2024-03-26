import torch
x_train = torch.Tensor([[1,1], [2,2], [3,3]])
y_train = torch.Tensor([[10], [20], [30]])
W = torch.randn([2,1], requires_grad =True)
b = torch.randn([1], requires_grad =True)
optimizer = torch.optim.SGD([W,b], lr=0.01)

def H(x):
    return torch.matmul(x,W)+b

for step in range(2000):
    cost = torch.mean((H(x_train) - y_train) ** 2)
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

x_test = torch.Tensor([4,4])
print(H(x_test))
