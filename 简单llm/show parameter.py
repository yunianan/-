import torch
from model import Model


model = Model(max_token_value=4055)
state_dict = torch.load('model-scifi.pt')
model.load_state_dict(state_dict)

# Step 2: Calculate the number of parameters
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"模型参数数量为: {total_params:,}")



