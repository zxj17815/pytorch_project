#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File        :   main.py
@Description :   None
@DateTime    :   2024-04-03 13:37:32
@Author      :   JayZhang 
'''

# here put the import lib
import torch


# here put the main code
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

x = torch.ones((1024 * 12, 1024 * 12), dtype=torch.float32, device=device)
print(x)
print(x.device)