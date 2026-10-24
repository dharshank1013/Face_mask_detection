"""Hardware verification and device selection utility for Face Mask Detection.

Strict device selection priority:
1. xpu  (Intel Arc / Intel GPU via native PyTorch XPU)
2. cuda (NVIDIA GPU)
3. cpu  (safe universal fallback)
"""

from __future__ import annotations

import sys
import time
from typing import Dict, Any, Tuple
import torch


def is_xpu_available() -> bool:
    """Check if Intel Arc / native PyTorch XPU is available."""
    try:
        return hasattr(torch, "xpu") and torch.xpu.is_available()
    except Exception:
        return False


def is_cuda_available() -> bool:
    """Check if NVIDIA CUDA is available."""
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


def get_device(force_cpu: bool = False) -> torch.device:
    """Select compute device following strict priority: XPU -> CUDA -> CPU.

    Args:
        force_cpu: If True, forces CPU selection regardless of GPU availability.

    Returns:
        torch.device instance.
    """
    if force_cpu:
        return torch.device("cpu")

    if is_xpu_available():
        return torch.device("xpu")
    elif is_cuda_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def get_hardware_info() -> Dict[str, Any]:
    """Collect comprehensive hardware information."""
    xpu_avail = is_xpu_available()
    cuda_avail = is_cuda_available()

    xpu_name = "Not Available"
    if xpu_avail:
        try:
            xpu_name = torch.xpu.get_device_name(0)
        except Exception as e:
            xpu_name = f"Available (query error: {e})"

    cuda_name = "Not Available"
    if cuda_avail:
        try:
            cuda_name = torch.cuda.get_device_name(0)
        except Exception as e:
            cuda_name = f"Available (query error: {e})"

    selected = get_device()

    return {
        "pytorch_version": torch.__version__,
        "selected_device": str(selected),
        "xpu_available": xpu_avail,
        "xpu_device_name": xpu_name,
        "cuda_available": cuda_avail,
        "cuda_device_name": cuda_name,
    }


def run_benchmark(device: torch.device, matrix_size: int = 512) -> Tuple[bool, float, str]:
    """Execute a matrix multiplication test on the target device."""
    try:
        start_time = time.perf_counter()
        a = torch.randn(matrix_size, matrix_size, device=device, dtype=torch.float32)
        b = torch.randn(matrix_size, matrix_size, device=device, dtype=torch.float32)
        c = torch.matmul(a, b)
        if device.type == "xpu" and hasattr(torch.xpu, "synchronize"):
            torch.xpu.synchronize()
        elif device.type == "cuda" and hasattr(torch.cuda, "synchronize"):
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        # Verification check
        assert c.shape == (matrix_size, matrix_size)
        return True, elapsed_ms, f"Passed ({matrix_size}x{matrix_size} matmul: {elapsed_ms:.2f} ms)"
    except Exception as e:
        return False, 0.0, f"Failed: {str(e)}"


def print_hardware_report() -> bool:
    """Print standard verification output required by project specification."""
    info = get_hardware_info()
    device = torch.device(info["selected_device"])

    print("=" * 65)
    print("   FACE MASK DETECTION - HARDWARE VERIFICATION")
    print("=" * 65)
    print(f"PyTorch Version   : {info['pytorch_version']}")
    print(f"XPU Available     : {info['xpu_available']}")
    print(f"XPU Device Name   : {info['xpu_device_name']}")
    print(f"CUDA Available    : {info['cuda_available']}")
    if info["cuda_available"]:
        print(f"CUDA Device Name  : {info['cuda_device_name']}")
    print("-" * 65)
    print(f"Selected Device   : {info['selected_device'].upper()}")
    print("-" * 65)

    print("Running matrix operation test on selected device...")
    success, elapsed, msg = run_benchmark(device)
    print(f"Matrix Test       : {msg}")
    print("=" * 65)

    if info["selected_device"] == "xpu":
        print(">> Intel Arc GPU is ACTIVE via native PyTorch XPU.")
    elif info["selected_device"] == "cuda":
        print(">> NVIDIA CUDA GPU is ACTIVE.")
    else:
        print(">> Running on CPU fallback.")
        print(">> Note: To enable Intel Arc GPU, install PyTorch with XPU support:")
        print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/xpu")
    print("=" * 65)

    return success


if __name__ == "__main__":
    ok = print_hardware_report()
    sys.exit(0 if ok else 1)
