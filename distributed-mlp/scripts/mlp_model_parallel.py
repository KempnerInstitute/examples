import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from random_dataset import RandomTensorDataset

class MLP(nn.Module):
  def __init__(self, in_feature, hidden_units, out_feature):
    super().__init__()
    torch.manual_seed(12345)
    self.hidden_layer = nn.Linear(in_feature, hidden_units).to(0)
    self.output_layer = nn.Linear(hidden_units, out_feature).to(1)
  
  def forward(self, x):
    x = self.hidden_layer(x.to(0))
    x = self.output_layer(x.to(1))
    return x

assert (
   torch.cuda.device_count() > 1
), f'Requires two GPUs on the same node but {torch.cuda.device_count()} GPUs are available'

# model construction
layer_1_units = 6
layer_2_units = 4
layer_3_units = 2
model = MLP(
  in_feature=layer_1_units,
  hidden_units=layer_2_units,
  out_feature=layer_3_units
)

# W1, b1 tensors on GPU0 and W2, b2 on GPU1
# Total number of the model parameters on GPU0 is equal to: 28 ==> W1(6*4) + b1(1*4)
# Total number of the model parameters on GPU1 is equal to: 10 ==> W2(4*2) + b1(1*2)
for p in model.parameters():
  print(p)
total_params_0 = sum(p.numel() for p in model.hidden_layer.parameters())
print(f"[GPU0] Total number of parameters in the model: {total_params_0}")
total_params_1 = sum(p.numel() for p in model.output_layer.parameters())
print(f"[GPU1] Total number of parameters in the model: {total_params_1}")

loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(),lr=0.01)

# dataset construction
num_samples = 1024
batch_size=32
dataset = RandomTensorDataset(
  num_samples=num_samples,
  in_shape=layer_1_units,
  out_shape=layer_3_units
  )

dataloader = DataLoader(
  dataset,
  batch_size=batch_size,
  pin_memory=True,
  shuffle=True
  )

max_epochs = 1
for i in range(max_epochs):
  print(f"Epoch {i} | Batchsize: {len(next(iter(dataloader))[0])} | Steps: {len(dataloader)}")
  for x, y in dataloader:
    y = y.to(1)
    
    # Forward Pass 
    out = model(x)

    # Calculate loss
    loss = loss_fn(out, y)

    # Zero grad
    optimizer.zero_grad(set_to_none=True)

    # Backward Pass
    loss.backward()

    # Update Model
    optimizer.step()