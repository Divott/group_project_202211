import torch
import numpy as np
a = np.arange(8)
a = a.cuda()
print(torch.cuda.is_available())
