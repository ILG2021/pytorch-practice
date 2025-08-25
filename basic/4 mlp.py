import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn, ops, optim
from torch.nn import MSELoss

model = nn.Sequential(nn.Linear(2, 128),
                      nn.ReLU(),
                      nn.Linear(128, 128),
                      nn.ReLU(),
                      nn.Linear(128, 1))

df = pd.DataFrame({
    "area": [120, 180, 150, 210, 106],
    "age": [5, 2, 1, 2, 1],
    "price": [30, 90, 100, 180, 85]
})

df = pd.DataFrame(MinMaxScaler(feature_range=(-1, 1)).fit_transform(df), columns=df.columns)
print(df)
x = torch.tensor(df[["area", "age"]].values, dtype=torch.float32)
y = torch.tensor(df[["price"]].values, dtype=torch.float32)

loss_fn = MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.2)

for i in range(10):
    predict = model(x)
    loss = loss_fn(predict, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(f"step {i} loss: {loss.item()}")
