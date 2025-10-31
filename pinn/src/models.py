import torch
import torch.nn as nn

'''
Implementation of PINNs. 

Source: https://github.com/AdityaLab/pinnsformer/blob/main/model/pinn.py
'''
class PINN(nn.Module):
  def __init__(self, in_dim, hidden_dim, out_dim, num_layer, dtype=torch.float32):
    super(PINN, self).__init__()

    layers = []
    for i in range(num_layer-1):
      if i == 0:
        layers.append(nn.Linear(in_features=in_dim, out_features=hidden_dim, dtype=dtype))
        layers.append(nn.Tanh())
      else:
        layers.append(nn.Linear(in_features=hidden_dim, out_features=hidden_dim, dtype=dtype))
        layers.append(nn.Tanh())

    layers.append(nn.Linear(in_features=hidden_dim, out_features=out_dim, dtype=dtype))

    self.linear = nn.Sequential(*layers)

  def forward(self, x, t):
    src = torch.cat((x,t), dim=-1)
    return self.linear(src)