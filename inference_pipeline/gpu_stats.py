import pynvml

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)


def get_gpu_stats():
    """
    Returns a snapshot of GPU utilization and memory stats.
    """
    # GPU and memory utilization
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)  # returns gpu and memory %
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)  # returns used, total, free memory in bytes

    # Return GPU Utilization, Memory Utilization , Used Memory (GB), Total Memory (GB)
    return {
        "gpu_util_percent": util.gpu,
        "memory_util_percent": util.memory,
        "memory_used_gb": mem_info.used / 1024 ** 3,
        "memory_total_gb": mem_info.total / 1024 ** 3
    }
