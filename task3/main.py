import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T

import os

"""
********************  FILE STRUCTURE  ********************
---- $Current directory (path)
   |
   |---- main.py
   |---- dataset
       |
       |---- *.txt (Labels)
       |
       |---- food
            |
            |---- *.jpg (Images)
   
"""


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path = os.getcwd()
    path_label = os.path.join(path, 'dataset')
    path_img = os.path.join(path_label, 'food')
    
