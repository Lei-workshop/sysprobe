# Information-Collection

## 简介

该工具用于快速收集和报告系统环境信息

## 功能

- **系统信息**：主机名、IP、Python 版本
- **CPU 信息**：操作系统、架构、CPU 型号
- **GPU 信息**：NVIDIA 驱动、CUDA 版本、GPU 拓扑
- **PyTorch 信息**：版本、CUDA 支持、设备详情、通信设备信息
- **关键包版本**：关键包的安装情况
- **环境变量**：关键环境变量的设置情况
- **NGC 环境**：判断是否是NVIDIA NGC 容器环境及版本

## 项目结构

- `Information-Collection/`：
  - `env_collector.py`：环境信息收集脚本
  - `README.md`：说明文件
  - `environment_summary.txt`：生成的完整信息报告

## 使用方法

```bash
python env_collector.py
```
## 输出示例

- 运行脚本后生成展示信息，简单示例如下：

```
================================================
环境信息完整报告
2026-01-19
================================================
system:
  hostname: xxx
  ... 
  cpu_info:
    os: xxx 
    ... 
  gpu_info:
    driver_version: xxx
    ...
  comm_device_info:
    direct_rdma_info: xxx 
  python_packages: {
    accelerate: xxx
    ...
  }
  env_info:
    ngc_info: xxx
  }

================================================
================================================
```
- 同时生成`environment_summary.txt`文件，包含完整的环境信息报告

## 运行要求
- NVIDIA GPU
- Python 3.9 及以上版本
- Linux 操作系统
