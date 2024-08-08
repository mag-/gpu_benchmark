#!/usr/bin/env python

"""

This is Maximum Achievable Matmul FLOPS (MAMF) Finder

For discussion and multiple important nuances please refer to
https://github.com/stas00/ml-engineering/tree/master/compute/accelerator/benchmarks#maximum-achievable-matmul-flops-finder

Credits:
- Parts of this benchmark have been derived from https://github.com/EleutherAI/cookbook/tree/main/benchmarks/sizing (highly recommended!)
- Imtiaz Sajwani: HPU porting

"""

from pathlib import Path

import argparse
import datetime
import termios
import tty
import numpy as np
import os
import platform
import re
import shlex
import signal
import sys
import time
import torch
import optuna

from optuna.storages import RDBStorage
import warnings
warnings.filterwarnings("ignore", message="set_metric_names is experimental*")

has_hpu = False
try:
    import habana_frameworks.torch as ht
    if torch.hpu.is_available():
        has_hpu = True
except ModuleNotFoundError:
    pass

file_dir = os.path.abspath(os.path.dirname(__file__))



### Architecture specific helper classes ###

class Arch:
    def __init__(self):
        self.arch = "unknown"

    def __repr__(self):
        return self.arch

class CUDAArch(Arch):
    """ shared with CUDA and ROCm: NVIDIA + AMD """
    def __init__(self):
        if torch.version.hip is not None:
            self.arch = "rocm"
        else:
            self.arch = "cuda"

    def device(self):
        return torch.device('cuda:0')

    def name(self):
        return self.arch

    def device_info(self):
        return torch.cuda.get_device_properties(device)

    def compute_info(self):
        if self.arch == "rocm":
            return f"hip={torch.version.hip}, cuda={torch.version.cuda}"
        else:
            return f"cuda={torch.version.cuda}"

    def event(self, enable_timing=True):
        return torch.cuda.Event(enable_timing)

    def synchronize(self):
        torch.cuda.synchronize()

class HPUArch(Arch):
    """ Intel Gaudi* """
    def __init__(self):
        self.arch = "hpu"

    def device(self):
        return torch.device('hpu')

    def name(self):
        return self.arch

    def device_info(self):
        return torch.hpu.get_device_properties(device)

    def compute_info(self):
        return f"hpu={torch.version.hpu}"

    def event(self, enable_timing=True):
        return ht.hpu.Event(enable_timing)

    def synchronize(self):
        ht.hpu.synchronize()


def get_accelerator_arch():
    """
    returns: CUDAArch or HPUArch object
    """
    # cuda / rocm
    if torch.cuda.is_available():
        return CUDAArch()

    # hpu
    if has_hpu:
        return HPUArch()

    raise ValueError("Currently only cuda, rocm and hpu are supported")

arch = get_accelerator_arch()



### Helper classes ###

class Tee(object):
    def __init__(self, filename, verbose):
        Path(filename).resolve().parent.mkdir(parents=True, exist_ok=True)
        self.file = open(filename, "w")
        self.verbose = verbose
        if self.verbose:
            self.stdout = sys.stdout

    def write(self, message):

        if self.verbose:
            self.stdout.write(message)
        # replace `\r` and `033\[K` which are nice in the console, but we don't want those in the log file
        message = re.sub(r"(\r|\033\[K)", "\n", message)
        self.file.write(message)

    def flush(self):
        self.file.flush()
        if self.verbose:
            self.stdout.flush()


def print_benchmark_header(dtype, device, notes="None"):

    device_info = arch.device_info()
    compute_info = arch.compute_info()

    print(f"""
Benchmark started on {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}

** Command line:
{sys.executable} {" ".join(map(shlex.quote, sys.argv))}

** Dtype: {dtype}

** Platform/Device info:
{" ".join(platform.uname())}
{device_info}

** Critical software versions:
torch={torch.__version__}
{compute_info}

** Additional notes:
{notes}

{"-" * 80}

""")

def getch():
  fd = sys.stdin.fileno()
  old_settings = termios.tcgetattr(fd)
  try:
    tty.setraw(sys.stdin.fileno())
    ch = sys.stdin.read(1)
  finally:
    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
  return ch

# Benchmark of a basic GEMM
def benchmark_mm(m, n, k, dtype, device, num_iterations, num_warmup_iterations):
    start = arch.event(enable_timing=True)
    end = arch.event(enable_timing=True)

    A = torch.randn(m, n, dtype=dtype, device=device)
    B = torch.randn(n, k, dtype=dtype, device=device)
    C = torch.empty(m, k, dtype=dtype, device=device)

    times = np.zeros(num_iterations+num_warmup_iterations)
    for i in range(num_warmup_iterations + num_iterations):
        with torch.no_grad():
            start.record()
            torch.mm(A, B, out=C)
            end.record()
        arch.synchronize()
        times[i] = start.elapsed_time(end)
    times = times[num_warmup_iterations:]
    elapsed_time = np.amin(times)/1000 # want the fastest
    tflops = (2 * m * n * k) / (elapsed_time * 10**12)
    return tflops


def objective(trial):
    M = trial.suggest_int('M', args.m_range[0], args.m_range[1], step=args.m_range[2])
    N = trial.suggest_int('N', args.n_range[0], args.n_range[1], step=args.n_range[2])
    K = trial.suggest_int('K', args.k_range[0], args.k_range[1], step=args.k_range[2])
    
    tflops = benchmark_mm(M, N, K, dtype, device, args.num_iterations, args.num_warmup_iterations)
    return tflops

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--m_range", nargs=3, type=int, default=[1024, 18432, 128], help="The first dimension of the GEMM, [start,stop,step]")
    parser.add_argument("--n_range", nargs=3, type=int, default=[1024, 18432, 128], help="The shared dimension of the GEMM, [start,stop,step]")
    parser.add_argument("--k_range", nargs=3, type=int, default=[1024, 18432, 128], help="The last dimension of the GEMM, [start,stop,step]")
    parser.add_argument("--num_iterations", type=int, default=100, help='The number of iterations used to benchmark each GEMM')
    parser.add_argument("--num_warmup_iterations", type=int, default=50, help='The number of warmup iterations')
    parser.add_argument("--cuda_device", type=int, default=0, help="The cuda device to run the benchmark on")
    parser.add_argument("--output_file", type=str, default=f"{file_dir}/out.txt")
    parser.add_argument("--notes", type=str, default="", help="benchmark-specific notes to add to the output_file's header")
    parser.add_argument("--verbose", default=True, action=argparse.BooleanOptionalAction, help='log to stdout besides output_file?')
    parser.add_argument("--n_trials", type=int, default=1000, help="Number of trials for Optuna")
    parser.add_argument("--study_name", type=str, default="mamf_study", help="Name of the Optuna study")
    args = parser.parse_args()

    dtype = torch.bfloat16
    device = arch.device()

    sys.stdout = Tee(args.output_file, args.verbose)
    print_benchmark_header(dtype, device, args.notes)

    # Create a SQLite storage
    storage = RDBStorage(
        url=f"sqlite:///optuna.db",
        engine_kwargs={"connect_args": {"timeout": 30}}
    )

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        load_if_exists=True,
        direction="maximize"
    )
    study.set_metric_names(["TFLOPS"])

    start_time = time.time()

    def sigkill_handler(signum, frame):
        finish()
        sys.exit(1)

    signal.signal(signal.SIGINT, sigkill_handler)

    def finish():
        time_delta = time.time() - start_time
        time_str = str(datetime.timedelta(seconds=time_delta)).split(".")[0]
        print("", end="\033[K")
        best_trial = study.best_trial
        best_tflops = best_trial.value
        best_config = f"{best_trial.params['M']}x{best_trial.params['N']}x{best_trial.params['K']} (MxNxK)"
        print(f"The best outcome was {best_tflops:.1f}TFLOPS @ {best_config} (tried {len(study.trials)} shapes)")
        print(f"Elapsed time: {time_str}")
        # Ask user if they want to upload the data
        print("Do you want to upload the benchmark results to the API? (y/n): ", end='', flush=True)
        upload_response = getch().lower()
        print(upload_response)
        if upload_response == 'y':
            upload_to_api(best_tflops, best_config)

    def upload_to_api(tflops, config):
        import urllib.request
        import json
        data = json.dumps({
            'tflops': tflops,
            'config': config,
            'device_info': str(arch.device_info()),
            'compute_info': arch.compute_info(),
        }).encode('utf-8')
        req = urllib.request.Request('https://rafalkwasny.com/log_gpu_benchmark', data=data, headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req) as response:
                if response.getcode() == 200:
                    print("Data successfully uploaded.")
                else:
                    print(f"Failed to upload data. Status code: {response.getcode()}")
        except urllib.error.URLError as e:
            print(f"Failed to upload data. Error: {e.reason}")

    def print_progress(study, trial):
        print(f"Trial {trial.number:>6} | {trial.value:6.1f} TFLOPS @ {trial.params['M']}x{trial.params['N']}x{trial.params['K']:<20} | best: {study.best_value:6.1f} TFLOPS", end="\r")

    # Add known best shapes to the study
    known_best_shapes = [
        (6912, 16384, 2048),  # NVIDIA A100 SXM
        (2304, 5120, 1536),   # NVIDIA A100 PCIe
        (6144, 17920, 2816),  # NVIDIA H100 SXM
        (14336, 4096, 4096),  # NVIDIA RTX 4090
        (4352, 13568, 3840),  # AMD MI300X
    ]

    for m, n, k in known_best_shapes:
        study.enqueue_trial({'M': m, 'N': n, 'K': k})

    study.optimize(objective, n_trials=args.n_trials, callbacks=[print_progress])
    finish()