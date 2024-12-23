import torch
print(f"{torch.__version__=}")
print(f"{torch.version.cuda=}")

print(f"{torch.cuda.device_count()=}")
print(f"{torch.cuda.is_available()=}")

device = torch.device("cuda")
data = torch.tensor([1, 2, 3, 4, 5]).to(device)
print(data)