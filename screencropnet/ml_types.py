from typing import NewType

import numpy as np
import torch

ImageNdarrayBGR = NewType("ImageBGR", np.ndarray)
ImageNdarrayHWC = NewType("ImageHWC", np.ndarray)
TensorCHW = NewType("TensorCHW", torch.Tensor)
