import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.optim import SGD

df = pd.DataFrame({
    "area": [120, 180, 150, 210, 106],
    "age": [5, 2, 1, 2, 1],
    "price": [30, 90, 100, 180, 85]
})

normalize = lambda x: pd.DataFrame(MinMaxScaler(feature_range=(-1, 1)).fit_transform(x), columns=x.columns)
df = normalize(df)
print(df)

x = torch.tensor(df[["area", "age"]].values, dtype=torch.float32)  # 5 2
y = torch.tensor(df[["price"]].values, dtype=torch.float32)  # 5 1
print("x shape:", x.shape, "y shape:", y.shape)
w = torch.randn(size=(2, 1), requires_grad=True)
b = torch.randn(1, requires_grad=True)

optimizer = SGD(params=(w, b), lr=0.2)

for i in range(10):
    predict = x @ w + b
    loss = (y - predict) ** 2
    loss = loss.mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(f"step {i} loss: {loss.item()}")
