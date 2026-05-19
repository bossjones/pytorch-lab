# SOURCE: https://github.com/socialhourmobile/SD-hassan-ns/blob/3b6b266b17e0fd0a9b17374cd2afbf4c59b7c245/modules/devices.py
import argparse
import contextlib

import torch
from icecream import ic

from screennet import errors


# has_mps is only available in nightly pytorch (for now) and MasOS 12.3+.
# check `getattr` and try it for compatibility
def has_mps() -> bool:
    """Check whether the Apple MPS (Metal Performance Shaders) backend is usable.

    Returns ``False`` if MPS is not available, and additionally verifies it
    works by attempting to move a tensor onto the ``mps`` device.

    Returns:
        ``True`` if a tensor can be allocated on the MPS device, ``False``
        otherwise.
    """
    if not torch.backends.mps.is_available():
        return False
    try:
        torch.zeros(1).to(torch.device("mps"))
        return True
    except Exception:
        return False


def extract_device_id(args, name):
    """Extract the value following a named flag from a list of CLI arguments.

    Scans ``args`` for the first element containing ``name`` and returns the
    element immediately after it (e.g. the device id following ``--device``).

    Args:
        args: A sequence of command-line argument strings.
        name: The flag name to search for within each argument.

    Returns:
        The argument value immediately following the matched flag, or ``None``
        if the flag is not present.
    """
    for x in range(len(args)):
        if name in args[x]:
            return args[x + 1]

    return None


def get_optimal_device(args: argparse.Namespace):
    """Select the best available compute device.

    Prefers CUDA (optionally pinned to ``args.gpu`` if set), then falls back to
    Apple MPS, and finally to CPU.

    Args:
        args: Parsed arguments expected to expose a ``gpu`` attribute holding an
            optional CUDA device id.

    Returns:
        A :class:`torch.device` for the highest-priority available backend.
    """
    if torch.cuda.is_available():
        # from modules import shared
        device_id: int | None | None
        device_id = args.gpu

        if device_id is not None:
            cuda_device = f"cuda:{device_id}"
            return torch.device(cuda_device)
        else:
            return torch.device("cuda")

    if has_mps():
        return torch.device("mps")

    return cpu


def torch_gc():
    """Release cached GPU memory.

    When CUDA is available, empties the allocator cache and performs an
    inter-process CUDA collection. No-op on MPS/CPU.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def enable_tf32():
    """Enable TF32 precision for CUDA matmul and cuDNN operations.

    Has no effect on MPS or CPU since TF32 is a CUDA-only feature.
    """
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


errors.run(enable_tf32, "Enabling TF32")

cpu = torch.device("cpu")
device = device_interrogate = device_gfpgan = device_swinir = device_esrgan = (
    device_scunet
) = device_codeformer = None
dtype = torch.float16
dtype_vae = torch.float16


def randn(seed: int, shape: int) -> torch.Tensor:
    """Generate a seeded random-normal tensor on the active device.

    On MPS the tensor is generated on the CPU with a seeded generator and then
    moved to the device, working around PyTorch's incorrect seeding behavior on
    the Metal backend.

    Args:
        seed: The manual seed used for reproducible sampling.
        shape: The shape of the tensor to generate.

    Returns:
        A random-normal :class:`torch.Tensor` on the active device.
    """
    # Pytorch currently doesn't handle setting randomness correctly when the metal backend is used.
    if device.type == "mps":
        generator = torch.Generator(device=cpu)
        generator.manual_seed(seed)
        noise = torch.randn(shape, generator=generator, device=cpu).to(device)
        return noise

    torch.manual_seed(seed)
    return torch.randn(shape, device=device)


def randn_without_seed(shape: int):
    """Generate an unseeded random-normal tensor on the active device.

    On MPS the tensor is generated on the CPU and then moved to the device,
    working around PyTorch's randomness handling on the Metal backend.

    Args:
        shape: The shape of the tensor to generate.

    Returns:
        A random-normal :class:`torch.Tensor` on the active device.
    """
    # Pytorch currently doesn't handle setting randomness correctly when the metal backend is used.
    if device.type == "mps":
        generator = torch.Generator(device=cpu)
        noise = torch.randn(shape, generator=generator, device=cpu).to(device)
        return noise

    return torch.randn(shape, device=device)


# SOURCE: https://github.com/socialhourmobile/SD-hassan-ns/blob/3b6b266b17e0fd0a9b17374cd2afbf4c59b7c245/modules/shared.py#L42
def autocast(disable=False, precision: str = "autocast"):
    """_summary_

    Args:
        precision (str): Options include ["full", "autocast"]
        disable (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    # from modules import shared

    if disable:
        return contextlib.nullcontext()

    if dtype == torch.float32 or precision == "full":
        return contextlib.nullcontext()

    if torch.cuda.is_available():
        return torch.autocast("cuda")
    # MPS/CPU: torch.autocast("cuda") only warns and disables itself here.
    return contextlib.nullcontext()


# MPS workaround for https://github.com/pytorch/pytorch/issues/79383
def mps_contiguous(input_tensor: torch.Tensor, device: torch.device):
    """Returns a contiguous in memory tensor containing the same data as self tensor. If self tensor is already in the specified memory format, this function returns the self tensor.

    Args:
        input_tensor (torch.Tensor): _description_
        device (torch.device): _description_

    Returns:
        _type_: _description_
    """
    return input_tensor.contiguous() if device.type == "mps" else input_tensor


def mps_contiguous_to(input_tensor: torch.Tensor, device: torch.device):
    """Make a tensor contiguous (if on MPS) and move it to the target device.

    Combines the MPS contiguity workaround with a device transfer in one call.

    Args:
        input_tensor: The tensor to (optionally) make contiguous and move.
        device: The target device to move the tensor to.

    Returns:
        The tensor on ``device``, made contiguous first when ``device`` is MPS.
    """
    return mps_contiguous(input_tensor, device).to(device)


def mps_check():
    """Diagnose Apple MPS availability and exercise the device.

    Prints an explanatory message when MPS is unavailable (distinguishing a
    PyTorch build without MPS from an unsupported macOS/device). When MPS is
    available, allocates tensors directly on the ``mps`` device and runs a
    trivial operation to confirm the backend works.
    """
    # Check that MPS is available
    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print(
                "MPS not available because the current PyTorch install was not "
                "built with MPS enabled."
            )
        else:
            print(
                "MPS not available because the current MacOS version is not 12.3+ "
                "and/or you do not have an MPS-enabled device on this machine."
            )

    else:
        ic(torch.backends.mps.is_available())
        if torch.backends.mps.is_available():
            mps_device = torch.device("mps")
            x = torch.ones(1, device=mps_device)
            print(x)
        else:
            print("MPS device not found.")

        mps_device = torch.device("mps")

        # Create a Tensor directly on the mps device
        x = torch.ones(5, device=mps_device)
        # Or
        x = torch.ones(5, device="mps")

        # Any operation happens on the GPU (exercises the MPS device)
        _ = x * 2


# SOURCE: https://github.com/pytorch/pytorch/issues/77988
def seed_everything(seed: int):
    """Seed all relevant RNGs for reproducible runs.

    Seeds Python's ``random``, the ``PYTHONHASHSEED`` environment variable,
    NumPy, and PyTorch (CPU and CUDA), and enables deterministic cuDNN.

    Args:
        seed: The seed value applied across all random number generators.
    """
    # Ref: https://gist.github.com/ihoromi4/b681a9088f348942b01711f251e5f964
    import os
    import random

    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
