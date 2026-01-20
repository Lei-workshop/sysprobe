#!/usr/bin/env python3

# pyright: strict

import datetime
import json
import os
import platform
import re
import socket
import subprocess
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Self

if TYPE_CHECKING:
    from unittest.mock import MagicMock

KEY_PACKAGE_REGEX = "|".join(
    [
        "accelerate",
        "deepspeed",
        "transformers",
        "tensorflow",
        "torch",
        r"nvidia-cublas-cu\d+",
        r"nvidia-nccl-cu\d+",
        r"nvidia-nvshmem-cu\d+",
        "megatron",
        "vllm",
        "sglang",
        "triton",
        "flash-attn",
        "deep_ep",
    ]
)

KEY_ENV_REGEX = "|".join(
    [
        "PYTHONPATH",
        "LD_LIBRARY_PATH",
        "CUDA_HOME",
        "CUDA_PATH",
        "CUDA_VERSION",
        "NVIDIA_DRIVER_CAPABILITIES",
        "NVIDIA_PYTORCH_VERSION",
        "NVIDIA_VISIBLE_DEVICES",
        "NCCL_IB_HCA",
        "NCCL_IB_DISABLE",
        "NCCL_IB_TIMEOUT",
    ]
)


def run(cmd: str, timeout: int = 5) -> str:
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        if result.returncode != 0:
            return ""
        return result.stdout.rstrip()
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return ""


@dataclass
class PyTorchDevice:
    device_id: int
    name: str
    total_memory_gb: float
    multi_processor_count: int
    major: int
    minor: int


@dataclass
class CPUInfo:
    os: str
    os_release: str
    architecture: str
    model_name: str

    @classmethod
    def probe(cls) -> Self:
        return cls(
            os=f"{platform.system()} {platform.release()}",
            os_release=run(
                "cat /etc/os-release | grep PRETTY_NAME | cut -d'=' -f2 | tr -d '\"'"
            ),
            architecture=platform.machine(),
            model_name=run(
                "cat /proc/cpuinfo | grep 'model name' | head -1 | cut -d':' -f2"
            ).strip(),
        )


@dataclass
class GPUInfo:
    driver_version: str
    cuda_version: str
    devices: list[str]
    gpu_topo: str | None

    @staticmethod
    def query(result: list[str], key: str) -> list[str]:
        return [line.split(":")[-1].strip() for line in result if key in line]

    @classmethod
    def probe(cls) -> Self | None:
        result = run("nvidia-smi -q").splitlines()
        if not result:
            return None
        return cls(
            driver_version=cls.query(result, "Driver Version")[0],
            cuda_version=cls.query(result, "CUDA Version")[0],
            devices=cls.query(result, "Product Name"),
            gpu_topo=run(
                r"nvidia-smi topo -m 2>/dev/null | sed -n '1,/^Legend:/p' | grep -v 'Legend:'"
            ),
        )


@dataclass
class PyTorchInfo:
    pytorch_version: str | None
    pytorch_cuda_version: str | None = None
    pytorch_device_count: int | None = None
    pytorch_cuda_nccl_version: str | None = None

    @classmethod
    def probe(cls) -> Self | None:
        try:
            if TYPE_CHECKING:
                torch = MagicMock()
            else:
                import torch

            pytorch_cuda_available: bool = torch.cuda.is_available()
            if not pytorch_cuda_available:
                return cls(torch.__version__)

            return cls(
                pytorch_version=torch.__version__,
                pytorch_cuda_version=torch.version.cuda,
                pytorch_device_count=torch.cuda.device_count(),
                pytorch_cuda_nccl_version=".".join(map(str, torch.cuda.nccl.version())),
            )

        except ImportError:
            pass


def parse_pip_list() -> dict[str, str] | None:
    output = run("python -m pip freeze") or run("uv pip freeze")
    if not output:
        return None
    packages = dict[str, str]()
    for line in output.splitlines():
        if "==" in line and not line.startswith("#"):
            package, version = line.split("==", 1)
            packages[package] = version
    return packages


def get_comm_device_info() -> list[str]:
    direct_rdma_info = run(r"""
        for f in /sys/class/infiniband/mlx5_*/ports/*/rate; do
        d=${f#/sys/class/infiniband/}
        d=${d%/ports/*/rate}
        p=${f##*ports/}
        p=${p%/rate}
        r=$(cat "$f")
        s=$(cat "${f%/rate}/state" 2>/dev/null || echo "?")
        ll=$(cat "${f%/rate}/link_layer" 2>/dev/null || echo "?")
        echo "$d:$p -> ${r} ($s, $ll)"
        done 2>/dev/null
    """)
    return direct_rdma_info.splitlines()


@dataclass
class SystemInfoBase:
    hostname: str
    python_version: str
    cpu: CPUInfo
    gpu: GPUInfo | None
    comm_device: list["str"]
    pytorch: PyTorchInfo | None


@dataclass
class SystemInfoFull(SystemInfoBase):
    packages: dict[str, str] | None
    envs: dict[str, str]

    @classmethod
    def probe(cls):
        return cls(
            hostname=socket.gethostname(),
            python_version=platform.python_version(),
            cpu=CPUInfo.probe(),
            gpu=GPUInfo.probe(),
            comm_device=get_comm_device_info(),
            pytorch=PyTorchInfo.probe(),
            packages=parse_pip_list(),
            envs=dict(os.environ),
        )


@dataclass
class SystemInfoAbstract(SystemInfoBase):
    key_packages: dict[str, str] | None
    key_envs: dict[str, str]

    def __init__(self, full: SystemInfoFull):
        super().__init__(
            **{
                k: v
                for k, v in asdict(full).items()
                if k in SystemInfoBase.__dataclass_fields__.keys()
            },
        )
        self.key_packages = (
            {
                k: v
                for k, v in full.packages.items()
                if re.fullmatch(KEY_PACKAGE_REGEX, k)
            }
            if full.packages
            else None
        )
        self.key_envs = {
            k: v for k, v in full.envs.items() if re.fullmatch(KEY_ENV_REGEX, k)
        }


def main():
    full = SystemInfoFull.probe()
    abstract = SystemInfoAbstract(full)
    print(json.dumps(asdict(abstract), indent=4))


if __name__ == "__main__":
    main()
