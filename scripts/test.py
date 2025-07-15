import torch
print('Pytorch Version:', torch.__version__)
print('CUDA Available:', torch.cuda.is_available())
print('CUDA Device:', torch.cuda.device_count())