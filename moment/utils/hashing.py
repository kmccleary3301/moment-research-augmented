import numpy as np
from torch import Tensor
import hashlib

def sha256_of_array(arr: np.ndarray) -> str:
    # Ensure the array is C-contiguous (if it isn’t, this will make a copy)
    arr = np.ascontiguousarray(arr)
    # Create a memory view of the array data and compute the hash
    return hashlib.sha256(memoryview(arr)).hexdigest()

def sha256_of_tensor(tensor: Tensor) -> str:
    # Ensure the tensor is on CPU and contiguous
    tensor = tensor.cpu().contiguous()
    # Convert to NumPy and create a memory view for hashing
    return hashlib.sha256(memoryview(tensor.numpy())).hexdigest()

def sha256_of_string(string: str) -> str:
    return hashlib.sha256(string.encode()).hexdigest()
