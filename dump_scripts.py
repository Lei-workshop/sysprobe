#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
信息采集
"""
from dataclasses import dataclass, field, fields, MISSING
import datetime
import os
import platform
from typing import Any, Optional
import socket
import subprocess
import sys

KEY_PACKAGES = [
    "accelerate",
    "deepspeed",
    "transformers",
    "tensorflow",
    "torch",
    "nvidia-cublas-cu12",
    "nvidia-nccl-cu12 ",
    "nvidia-nvshmem-cu12",
    "megatron",
    "vllm", 
    "sglang",
    "triton",
    "flash-attn",
]

IMPORTANT_ENV_VARS = [
    # "PATH", 
    "PYTHONPATH", 
    "LD_LIBRARY_PATH", 
    "CUDA_HOME", 
    "CUDA_PATH", 
    "CUDA_VERSION",
    "NVIDIA_DRIVER_CAPABILITIES", 
    "NVIDIA_PYTORCH_VERSION",
    "NVIDIA_VISIBLE_DEVICES", 
    # "MASTER_ADDR", 
    # "MASTER_PORT", 
    # "WORLD_SIZE", 
    # "LOCAL_RANK",
    "NCCL_IB_HCA",
    "NCCL_IB_DISABLE",
    "NCCL_IB_TIMEOUT",
]

def run(cmd: str, timeout: int = 5) -> str:
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            return ""
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return ""

@dataclass
class FieldTag:
    name: str
    format_spec: str

def tagged(name: str, *, format: str = "{}", default_factory: Any = MISSING) -> Any:
    metadata = {"tag": FieldTag(name=name, format_spec=format)}
    if default_factory is MISSING:
        return field(metadata=metadata)
    else:
        return field(default_factory=default_factory, metadata=metadata)

def alias(name: str):
    def decorator(cls):
        cls._alias_ = name
        return cls
    return decorator

def format_dict_table_with_all_keys(data: dict[str, str], key_order: list[str], indent: int = 2) -> str:
    if not key_order:
        return "{}"
    indent_str = "  " * indent
    lines = ["{"]
    for key in key_order:
        value = data.get(key, "")
        lines.append(f"{indent_str}{key}: {value}")
    lines.append("  " * (indent - 1) + "}")
    return "\n".join(lines)

def format_key_packages(packages: dict[str, str], indent: int = 2) -> str:
    return format_dict_table_with_all_keys(packages, KEY_PACKAGES, indent)

def format_env_vars(env_vars: dict[str, str], indent: int = 2) -> str:
    return format_dict_table_with_all_keys(env_vars, IMPORTANT_ENV_VARS, indent)

def print_info(obj, indent: int = 0) -> str:
    lines = []
    alias_name = getattr(type(obj), '_alias_', type(obj).__name__)
    prefix = "  " * indent
    lines.append(f"{prefix}{alias_name}:")

    for f in fields(obj):
        tag = f.metadata.get("tag")
        if not tag:
            continue

        value = getattr(obj, f.name)
        disp_name = f"{prefix}  {tag.name}:"

        if f.name == "packages" and isinstance(value, dict):
            formatted = format_key_packages(value)
            lines.append(f"{disp_name} {formatted}")
        elif f.name == "env_vars" and isinstance(value, dict):
            formatted = format_env_vars(value)
            lines.append(f"{disp_name} {formatted}")
        elif hasattr(value, '__dataclass_fields__'):
            nested_str = print_info(value, indent + 1)
            lines.append(nested_str)
        elif isinstance(value, list):
            if not value:
                lines.append(f"{disp_name} []")
            elif hasattr(value[0], '__dataclass_fields__'):
                lines.append(disp_name)
                for item in value:
                    item_str = print_info(item, indent + 2)
                    lines.append(item_str)
            else:
                lines.append(f"{disp_name} {value}")
        elif isinstance(value, dict):
            if not value:
                lines.append(f"{disp_name} {{}}")
            else:
                pairs = [f"{k}: {v}" for k, v in value.items()]
                lines.append(f"{disp_name} {{{', '.join(pairs)}}}")
        else:
            if value is None:
                formatted = "null"
            else:
                try:
                    formatted = tag.format_spec.format(value)
                except (ValueError, TypeError, KeyError):
                    formatted = str(value)
            lines.append(f"{disp_name} {formatted}")

    return "\n".join(lines)

@dataclass
class PyTorchDevice:
    device_id: int = tagged("device_id")
    name: str = tagged("name")
    total_memory_gb: float = tagged("total_memory_gb", format="{:.1f}GB")
    multi_processor_count: int = tagged("multi_processor_count")
    major: int = tagged("major")
    minor: int = tagged("minor")

@alias("cpu_info")
@dataclass
class CPUInfo:
    os:str = tagged("os", format="{:10s}")
    os_release: str = tagged("os_release", format="{:10s}")
    architecture: str = tagged("architecture", format="{:10s}")
    model_name: str = tagged("model_name", format="{:10s}")


@alias("gpu_info")
@dataclass
class GPUInfo:
    nvidia_smi_path: str = tagged("nvidia_smi_path", format="{:10s}")
    driver_version: str = tagged("driver_version")
    cuda_version: str = tagged("cuda_version")
    pytorch_available: bool = tagged("pytorch_available", default_factory=lambda: False)
    pytorch_version: Optional[str] = tagged("pytorch_version", default_factory=lambda: None)
    pytorch_cuda_available: Optional[bool] = tagged("pytorch_cuda_available", default_factory=lambda: None)
    pytorch_cuda_version: Optional[str] = tagged("pytorch_cuda_version", default_factory=lambda: None)
    pytorch_cuda_nccl_version: Optional[str] = tagged("pytorch_cuda_nccl_version", default_factory=lambda: None)
    pytorch_device_count: Optional[int] = tagged("pytorch_device_count", default_factory=lambda: None)
    pytorch_current_device: Optional[int] = tagged("pytorch_current_device", default_factory=lambda: None)
    pytorch_devices: Optional[list['PyTorchDevice']] = tagged("pytorch_devices", default_factory=lambda: None)
    gpu_topo: Optional[str] = tagged("topology_matrix", default_factory=lambda: None)


@alias("comm_device_info")
@dataclass
class CommDeviceInfo:
    direct_rdma_info: str = tagged("direct_rdma_info", default_factory=str)


@alias("env_info")
@dataclass
class EnvInfo:
    ngc_info: Optional[tuple] = tagged("ngc_info", default_factory=lambda: None)
    env_vars: dict[str, str] = tagged("env_vars", default_factory=dict)

@alias("system")
@dataclass
class SystemInfo:
    hostname: str = tagged("hostname", format="{:10s}")
    python_version: str = tagged("python_version", format="{:10s}")
    cpu_info: Optional[CPUInfo] = tagged("cpu_info")
    gpu_info: Optional[GPUInfo] = tagged("gpu_info")
    comm_device_info: Optional[CommDeviceInfo] = tagged("comm_device_info")
    packages: dict[str, str] = tagged("python_packages")
    env_info: Optional[EnvInfo] = tagged("env_info")

def collect_system_info() -> SystemInfo:
    hostname = socket.gethostname()
    python_version = platform.python_version()
    cpu_info = get_cpu_info()
    gpu_info = get_gpu_info()
    comm_device_info = get_comm_device_info()
    env_info = get_env_info()
    get_pytorch_info(gpu_info) if gpu_info else None
    packages = parse_pip_list()

    system_info = SystemInfo(
        hostname=hostname,
        python_version=python_version,
        cpu_info=cpu_info,
        gpu_info=gpu_info,
        comm_device_info=comm_device_info,
        packages=packages,
        env_info=env_info
    )
    return system_info

def get_env_info() -> EnvInfo:
    env_vars = {}
    for var in IMPORTANT_ENV_VARS:
        value = os.getenv(var)
        if value:
            env_vars[var] = value
    ngc_pytorch_version = os.environ.get('NVIDIA_PYTORCH_VERSION')
    if ngc_pytorch_version is not None:
        ngc_info = (True, ngc_pytorch_version)
    else:
        ngc_info = (False, None)
    return EnvInfo(
        ngc_info=ngc_info,
        env_vars=env_vars
    )

def get_pytorch_info(gpu_info: GPUInfo) -> None:
    try:
        import torch
        gpu_info.pytorch_available = True
        gpu_info.pytorch_version = torch.__version__
        gpu_info.pytorch_cuda_available = torch.cuda.is_available()
        gpu_info.gpu_topo = run(r"nvidia-smi topo -m 2>/dev/null | sed -n '1,/^Legend:/p' | grep -v 'Legend:'")
        if gpu_info.pytorch_cuda_available:
            gpu_info.pytorch_cuda_version = torch.version.cuda
            gpu_info.pytorch_device_count = torch.cuda.device_count()
            gpu_info.pytorch_current_device = torch.cuda.current_device()
            try:
                from torch.cuda import nccl
                gpu_info.pytorch_cuda_nccl_version = ".".join(map(str, nccl.version()))
            except:
                pass

            pytorch_devices = []
            for i in range(1):
                # show one device
                props = torch.cuda.get_device_properties(i)
                pytorch_devices.append(PyTorchDevice(
                    device_id=i,
                    name=torch.cuda.get_device_name(i),
                    total_memory_gb=props.total_memory / (1024**3),
                    multi_processor_count=props.multi_processor_count,
                    major=props.major,
                    minor=props.minor
                ))
            gpu_info.pytorch_devices = pytorch_devices

    except ImportError:
        pass
    except Exception as e:
        print(f"[WARN] PyTorch error: {e}", file=sys.stderr)

def get_cpu_info() -> Optional[CPUInfo]:
    os_info = f"{platform.system()} {platform.release()}"
    os_release = run("cat /etc/os-release | grep PRETTY_NAME | cut -d'=' -f2 | tr -d '\"'")
    architecture = platform.machine()
    model_name = run("cat /proc/cpuinfo | grep 'model name' | head -1 | cut -d':' -f2").strip()
    return CPUInfo(
        os=os_info,
        os_release=os_release,
        architecture=architecture,
        model_name=model_name
    )

def get_gpu_info() -> Optional[GPUInfo]:
    nvidia_smi_path = run("which nvidia-smi")
    if not nvidia_smi_path:
        return None 
    driver_version = run("nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1")
    cuda_version = run("nvcc --version 2>/dev/null | grep 'release' | awk '{print $NF}'")
    return GPUInfo(
        nvidia_smi_path=nvidia_smi_path,
        driver_version=driver_version,
        cuda_version=cuda_version,
    )

def get_comm_device_info() -> Optional[CommDeviceInfo]:
    direct_rdma_info = run(r'''
        for f in /sys/class/infiniband/mlx5_*/ports/*/rate; do
        d=${f#/sys/class/infiniband/}
        d=${d%/ports/*/rate}
        p=${f##*ports/}
        p=${p%/rate}
        r=$(cat "$f")
        s=$(cat "${f%/rate}/state" 2>/dev/null || echo "?")
        ll=$(cat "${f%/rate}/link_layer" 2>/dev/null || echo "?")
        echo "$d:$p → ${r}G ($s, $ll)"
        done 2>/dev/null || echo 'No InfiniBand devices'
    ''')
    return CommDeviceInfo(
        direct_rdma_info=direct_rdma_info
    )

def parse_pip_list() -> dict[str, str]:
    output = run("python -m pip list --format=freeze")
    packages = {}
    if output:
        for line in output.split("\n"):
            if "==" in line and not line.startswith("#"):
                try:
                    name, ver = line.split("==", 1)
                    packages[name] = ver
                except ValueError:
                    continue
    return packages


def main():
    print("🔍 正在收集系统与环境信息...")
    try:
        system = collect_system_info()
    except Exception as e:
        print(f"❌ 收集失败: {e}", file=sys.stderr)
        sys.exit(1)

    report = print_info(system)

    print("\n" + "=" * 70)
    print("📊 环境信息完整报告")
    print(timestamp:=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 70)
    print(report)

    output_file = "environment_summary.txt"
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("=" * 70 + "\n")
            f.write("环境信息完整报告\n")
            f.write(f"生成时间: {timestamp}\n")
            f.write("=" * 70 + "\n")
            f.write(report)
            f.write("\n\n" + "=" * 70 + "\n")
            f.write("完整包列表 (python -m pip list --format=freeze)\n")
            f.write("=" * 70 + "\n")
            pip_output = run("python -m pip list --format=freeze")
            f.write(pip_output if pip_output else "No pip output")
            f.write("\n\n" + "=" * 70 + "\n")
            f.write("完整环境变量 (export)\n")
            f.write("=" * 70 + "\n")
            env_output = "\n".join([f"{k}={v}" for k, v in sorted(os.environ.items())])
            f.write(env_output)
        print(f"\n✅ 报告已保存至: {os.path.abspath(output_file)}")
        installed_key = [pkg for pkg in KEY_PACKAGES if system.packages.get(pkg)]
        print(f"📦 关键包: {len(installed_key)}/{len(KEY_PACKAGES)} 已安装")
        if installed_key:
            print("   已安装: " + ", ".join(installed_key))
        if system.env_info:
            found_vars = [var for var in IMPORTANT_ENV_VARS if system.env_info.env_vars.get(var)]
            print(f"🌍 环境变量: {len(found_vars)}/{len(IMPORTANT_ENV_VARS)} 已设置")
            if found_vars:
                print("   已设置: " + ", ".join(found_vars))
    except Exception as e:
        print(f"⚠️  保存文件失败: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
