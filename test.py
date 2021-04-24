import torch

# device = torch.device("cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")
a = torch.cuda.is_available()
print(a)
