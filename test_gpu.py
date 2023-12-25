import torch 

device = torch.device('cuda')
arr = torch.zeros(5,5)
arr.to(device)