import numpy as np
import matplotlib.pyplot as plt
import torch
import os

from objectives import prot_builder

f = prot_builder()
# x = torch.rand(1,9)

x = torch.tensor([1.51799321e+00, 7.51994286e-01, 4.04369888e-01, 1.31945073e-01,
 1.36680050e+04, 2.29911665e+03, 1.17216536e+00, 1.41004617e+02,
 5.55049282e+00]).reshape(1, -1)

x = torch.tensor([0.4072, 0.7533, 0.8986, 0.0911, 0.1367, 0.1196, 0.6465, 0.2600, 0.2886]).reshape(1, -1)

print(f(x))
