"""
Profile Measurement Script for PIE-UNet Ablation Study

Measures:
- #Params
- GFLOPs (using thop)
- CPU Latency
- Peak RAM (CPU memory)
"""

import torch
import time
import tracemalloc
import sys
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from thop import profile, clever_format
except ImportError:
    print("Installing thop...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "thop"])
    from thop import profile, clever_format

from ablation.profile.config import PROFILE_CONFIGS, MEASURE_CONFIG
from ablation.profile.pie_unet import PIE_UNet


def count_parameters(model):
    """Count total and trainable parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def measure_flops(model, input_tensor):
    """
    Measure GFLOPs using thop - EXACTLY same as scripts/measure_flops.py
    """
    model.eval()
    with torch.no_grad():
        macs, params = profile(model, inputs=(input_tensor,), verbose=False)
    
    # Convert to GFLOPs (1 MAC = 2 FLOPs typically)
    # thop returns MACs on the scale of raw numbers
    formatted_macs, formatted_params = clever_format([macs, params], "%.3f")
    
    g_macs = macs / 1e9
    g_flops = g_macs * 2
    
    return macs, g_flops, formatted_macs, formatted_params


def get_cpu_info():
    """Get CPU information for documentation."""
    import platform
    import multiprocessing
    
    info = {
        "processor": platform.processor() or "Unknown",
        "machine": platform.machine(),
        "num_cores": multiprocessing.cpu_count(),
    }
    
    # Try to get more detailed CPU info on Linux
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if "model name" in line:
                    info["model_name"] = line.split(":")[1].strip()
                    break
    except:
        pass
    
    return info


def measure_cpu_latency(model, input_tensor, num_warmup=10, num_runs=100, verbose=False):
    """
    Measure average CPU inference latency.
    
    IMPORTANT: 
    - Model and input are moved to CPU
    - CUDA is explicitly disabled during measurement
    - PyTorch threads are set for consistent measurement
    """
    import os
    
    # Disable CUDA for this measurement
    original_cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
    # Set consistent thread count for reproducibility
    num_threads = 4  # Use fixed thread count for fair comparison
    torch.set_num_threads(num_threads)
    
    model.eval()
    model_cpu = model.cpu()
    input_cpu = input_tensor.cpu()
    
    if verbose:
        cpu_info = get_cpu_info()
        print(f"  CPU Info: {cpu_info.get('model_name', cpu_info['processor'])}")
        print(f"  Threads: {num_threads}, Cores: {cpu_info['num_cores']}")
    
    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model_cpu(input_cpu)
    
    # Measure
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.perf_counter()
            _ = model_cpu(input_cpu)
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)  # ms
    
    avg_latency = sum(latencies) / len(latencies)
    std_latency = (sum((l - avg_latency) ** 2 for l in latencies) / len(latencies)) ** 0.5
    
    # Restore CUDA_VISIBLE_DEVICES
    if original_cuda_visible is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda_visible
    elif 'CUDA_VISIBLE_DEVICES' in os.environ:
        del os.environ['CUDA_VISIBLE_DEVICES']
    
    return avg_latency, std_latency


def measure_peak_ram(model, input_tensor):
    """
    Measure peak RSS (Resident Set Size) during CPU inference.
    
    Uses resource.getrusage() to get accurate peak RSS.
    This measures the ACTUAL max RAM usage when running 1 sample.
    """
    import resource
    import gc
    
    # Force CPU-only
    model = model.cpu()
    model.eval()
    input_cpu = input_tensor.cpu()
    
    # Clear memory before measurement
    gc.collect()
    
    # Get baseline RSS (in KB on Linux)
    baseline_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    
    # Run inference to trigger memory allocation
    with torch.no_grad():
        _ = model(input_cpu)
    
    # Get peak RSS after inference
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    
    # Convert to MB (rusage returns KB on Linux)
    peak_mb = peak_rss / 1024
    
    return peak_mb


def measure_profile(profile_name, n_classes=4, verbose=True):
    """
    Measure all metrics for a given profile.
    
    Returns:
        dict with keys: params, gflops, cpu_latency_ms, peak_ram_mb
    """
    config = PROFILE_CONFIGS[profile_name]
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Measuring Profile: {config['name']}")
        print(f"  n_channels: {config['n_channels']}, depth: {config['depth']}")
        print(f"{'='*60}")
    
    # Create model
    model = PIE_UNet(
        n_channels=config["n_channels"],
        n_classes=n_classes,
        depth=config["depth"],
        base_filters=config["base_filters"],
        deep_supervision=True
    )
    
    # Create dummy input
    input_size = MEASURE_CONFIG["input_size"]
    dummy_input = torch.randn(1, config["n_channels"], input_size, input_size)
    
    # Measure parameters
    total_params, trainable_params = count_parameters(model)
    if verbose:
        print(f"  #Params: {total_params:,} ({total_params/1e6:.2f}M)")
    
    # Measure GFLOPs (same as scripts/measure_flops.py)
    macs, gflops, formatted_macs, formatted_params = measure_flops(model, dummy_input)
    if verbose:
        print(f"  MACs: {formatted_macs}, GFLOPs: {gflops:.3f}")
    
    # Measure CPU latency (with CUDA disabled)
    num_warmup = MEASURE_CONFIG["num_warmup_runs"]
    num_runs = MEASURE_CONFIG["num_measure_runs"]
    avg_latency, std_latency = measure_cpu_latency(model, dummy_input, num_warmup, num_runs, verbose=verbose)
    if verbose:
        print(f"  CPU Latency: {avg_latency:.2f} ± {std_latency:.2f} ms (CUDA disabled, {num_runs} runs)")
    
    # Measure Peak RAM (RSS peak during CPU inference)
    peak_ram_mb = measure_peak_ram(model, dummy_input)
    if verbose:
        print(f"  Peak RAM (RSS): {peak_ram_mb:.1f} MB")
    
    return {
        "profile": profile_name,
        "name": config["name"],
        "n_channels": config["n_channels"],
        "depth": config["depth"],
        "params": total_params,
        "params_m": total_params / 1e6,
        "macs": macs,
        "g_macs": macs / 1e9,
        "gflops": gflops,
        "cpu_latency_ms": avg_latency,
        "cpu_latency_std": std_latency,
        "peak_ram_mb": peak_ram_mb
    }


def measure_all_profiles(n_classes=4):
    """Measure all profiles and return results."""
    results = []
    for profile_name in PROFILE_CONFIGS.keys():
        result = measure_profile(profile_name, n_classes=n_classes)
        results.append(result)
    return results


def print_results_table(results):
    """Print results as a formatted table."""
    print("\n" + "=" * 90)
    print("ABLATION STUDY: PIE-UNet Profile Comparison")
    print("=" * 90)
    print(f"{'Profile':<12} {'C_in':<6} {'Depth':<6} {'#Params':<12} {'GFLOPs':<10} {'CPU Latency':<15} {'Peak RAM':<10}")
    print("-" * 90)
    
    for r in results:
        print(f"{r['name']:<12} {r['n_channels']:<6} {r['depth']:<6} "
              f"{r['params_m']:.2f}M{'':<6} {r['gflops']:<10.3f} "
              f"{r['cpu_latency_ms']:.2f}ms{'':<8} {r['peak_ram_mb']:.1f}MB")
    
    print("=" * 90)


def save_results_csv(results, output_path):
    """Save results to CSV file."""
    import csv
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure PIE-UNet profile metrics")
    parser.add_argument("--profile", type=str, choices=list(PROFILE_CONFIGS.keys()),
                        help="Specific profile to measure (default: all)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV file path")
    args = parser.parse_args()
    
    if args.profile:
        results = [measure_profile(args.profile)]
    else:
        results = measure_all_profiles()
    
    print_results_table(results)
    
    if args.output:
        save_results_csv(results, args.output)
    else:
        # Save to default location
        from ablation.profile.config import OUTPUT_CONFIG
        output_path = OUTPUT_CONFIG["results_dir"] / "profile_metrics.csv"
        save_results_csv(results, output_path)
