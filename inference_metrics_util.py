"""
Shared inference-benchmark metrics recorder for UrbanFloodBench solutions.
==========================================================================

Framework-agnostic (works with PyTorch or JAX; degrades gracefully with
neither).  Each solution's inference.py imports this, records the metrics
that ARE derivable from dummy/synthetic inference, and writes:

    <solution_folder>/inference_metrics.json
    <solution_folder>/inference_metrics.txt

Metrics captured (all feasible with random weights + synthetic inputs):
  - device / framework / hardware (GPU name, total memory, versions)
  - model parameter counts and on-disk size
  - per-phase timing: preprocessing (synthetic input build), inference
    (forward / AR rollout), postprocessing (denorm / output assembly)
  - CPU process time AND wall time for every phase
  - peak GPU memory (allocated + reserved) and peak CPU RSS
  - batch / rollout-length sensitivity sweep (throughput per config)

Explicitly left NULL (cannot come from dummy inference — fill externally):
  - hecras_runtime_s        (from the reference HEC-RAS simulation)
  - speedup_vs_hecras       (= hecras_runtime_s / model inference time)
  - training_cost_gpu_hours (from training logs / estimates)
"""

import os
import sys
import json
import time
import platform
import datetime
import resource
import statistics


# ---------------------------------------------------------------------------
# framework / hardware probes
# ---------------------------------------------------------------------------
def _torch():
    try:
        import torch
        return torch
    except Exception:
        return None


def _cuda_available():
    t = _torch()
    return t is not None and t.cuda.is_available()


def gpu_sync():
    """Block until queued GPU work finishes (so timings are accurate)."""
    t = _torch()
    if t is not None and t.cuda.is_available():
        t.cuda.synchronize()


def reset_gpu_peak():
    t = _torch()
    if t is not None and t.cuda.is_available():
        t.cuda.reset_peak_memory_stats()


def gpu_peak_alloc_mb():
    """Peak GPU memory in MB (torch first, then JAX, else None)."""
    t = _torch()
    if t is not None and t.cuda.is_available():
        return t.cuda.max_memory_allocated() / 1e6
    try:
        import jax
        st = jax.devices()[0].memory_stats()
        if st and "peak_bytes_in_use" in st:
            return st["peak_bytes_in_use"] / 1e6
    except Exception:
        pass
    return None


def _peak_cpu_rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is bytes on macOS, kilobytes on Linux
    return r / 1e6 if sys.platform == "darwin" else r / 1e3


# ---------------------------------------------------------------------------
# timing context manager
# ---------------------------------------------------------------------------
class Phase:
    """Context manager timing one pipeline phase (wall + CPU + GPU peak)."""

    def __init__(self, recorder, name, reset_gpu=True):
        self.rec = recorder
        self.name = name
        self.reset_gpu = reset_gpu

    def __enter__(self):
        if self.reset_gpu:
            reset_gpu_peak()
        gpu_sync()
        self._w0 = time.perf_counter()
        self._c0 = time.process_time()
        return self

    def __exit__(self, *exc):
        gpu_sync()
        wall = time.perf_counter() - self._w0
        cpu = time.process_time() - self._c0
        self.rec.data["phases"][self.name] = {
            "wall_s": wall,
            "cpu_s": cpu,
            "gpu_peak_alloc_mb": gpu_peak_alloc_mb(),
        }
        return False


# ---------------------------------------------------------------------------
# recorder
# ---------------------------------------------------------------------------
class MetricsRecorder:
    def __init__(self, solution, framework, device):
        self.data = {
            "solution": solution,
            "framework": framework,
            "device": str(device),
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "hardware": {},
            "model": {},
            "phases": {},
            "timing": {},
            "throughput": {},
            "batch_sensitivity": [],
            "memory": {},
            "external_reference_required": {
                "hecras_runtime_s": None,
                "speedup_vs_hecras": None,
                "training_cost_gpu_hours": None,
                "_note": (
                    "Not derivable from dummy inference. Fill hecras_runtime_s "
                    "from the reference HEC-RAS simulation, speedup_vs_hecras = "
                    "hecras_runtime_s / inference wall time, and "
                    "training_cost_gpu_hours from training logs/estimates."
                ),
            },
        }
        t = _torch()
        if t is not None:
            self.data["hardware"]["torch"] = t.__version__
            if t.cuda.is_available():
                p = t.cuda.get_device_properties(0)
                self.data["hardware"]["gpu_name"] = t.cuda.get_device_name(0)
                self.data["hardware"]["gpu_total_mem_mb"] = p.total_memory / 1e6
                self.data["hardware"]["cuda"] = t.version.cuda
        try:
            import jax
            self.data["hardware"]["jax"] = jax.__version__
            self.data["hardware"]["jax_devices"] = [str(d) for d in jax.devices()]
        except Exception:
            pass

    # ---- phase timing ----
    def phase(self, name, reset_gpu=True):
        return Phase(self, name, reset_gpu)

    # ---- model info ----
    @staticmethod
    def count_params(model):
        """Parameter count for a torch or keras model (best effort)."""
        try:  # torch
            return int(sum(p.numel() for p in model.parameters()))
        except Exception:
            pass
        try:  # keras
            return int(model.count_params())
        except Exception:
            return None

    @staticmethod
    def param_size_mb(model, bytes_per_param=4):
        n = MetricsRecorder.count_params(model)
        return None if n is None else n * bytes_per_param / 1e6

    def set_model(self, name, n_params=None, size_mb=None, **extra):
        self.data["model"][name] = {
            "n_params": None if n_params is None else int(n_params),
            "size_mb": size_mb,
            **extra,
        }

    # ---- timing statistics from a list of per-run times ----
    def set_timing(self, label, times):
        arr = [float(x) for x in times]
        if not arr:
            return
        self.data["timing"][label] = {
            "runs": len(arr),
            "mean_s": statistics.mean(arr),
            "std_s": statistics.pstdev(arr) if len(arr) > 1 else 0.0,
            "median_s": statistics.median(arr),
            "min_s": min(arr),
            "max_s": max(arr),
        }
        return self.data["timing"][label]

    def set_throughput(self, label, items, mean_s):
        self.data["throughput"][label] = {
            "items": items,
            "mean_s": mean_s,
            "items_per_s": (items / mean_s) if mean_s else None,
        }

    # ---- batch / rollout sensitivity ----
    def add_batch_point(self, config, mean_s, items=None, gpu_peak_mb=None, **extra):
        self.data["batch_sensitivity"].append({
            "config": config,
            "mean_s": mean_s,
            "items": items,
            "items_per_s": (items / mean_s) if (items and mean_s) else None,
            "gpu_peak_alloc_mb": gpu_peak_mb,
            **extra,
        })

    def set(self, key, value):
        self.data[key] = value

    # ---- finalize + write ----
    def _finalize_memory(self):
        self.data["memory"]["peak_cpu_rss_mb"] = _peak_cpu_rss_mb()
        if _cuda_available():
            t = _torch()
            self.data["memory"]["gpu_max_allocated_mb"] = t.cuda.max_memory_allocated() / 1e6
            self.data["memory"]["gpu_max_reserved_mb"] = t.cuda.max_memory_reserved() / 1e6
        else:
            gp = gpu_peak_alloc_mb()
            if gp is not None:
                self.data["memory"]["gpu_peak_alloc_mb"] = gp

    def _pretty(self):
        d = self.data
        L = []
        L.append("=" * 64)
        L.append(f"Inference metrics — {d['solution']}")
        L.append("=" * 64)
        L.append(f"framework : {d['framework']}")
        L.append(f"device    : {d['device']}")
        L.append(f"timestamp : {d['timestamp']}")
        L.append(f"platform  : {d['platform']}  (python {d['python']})")
        if d["hardware"]:
            L.append("hardware  : " + ", ".join(f"{k}={v}" for k, v in d["hardware"].items()))
        L.append("")
        if d["model"]:
            L.append("-- model --")
            for k, v in d["model"].items():
                L.append(f"  {k}: params={v.get('n_params')}, size_mb={v.get('size_mb')}")
            L.append("")
        if d["phases"]:
            L.append("-- phase timing (wall / cpu seconds; gpu peak MB) --")
            for k, v in d["phases"].items():
                L.append(f"  {k:<16} wall={v['wall_s']:.5f}  cpu={v['cpu_s']:.5f}  gpu_peak={v['gpu_peak_alloc_mb']}")
            L.append("")
        if d["timing"]:
            L.append("-- benchmark timing --")
            for k, v in d["timing"].items():
                L.append(f"  {k:<16} mean={v['mean_s']:.5f}s  std={v['std_s']:.5f}  median={v['median_s']:.5f}  (n={v['runs']})")
            L.append("")
        if d["batch_sensitivity"]:
            L.append("-- batch / rollout sensitivity --")
            for b in d["batch_sensitivity"]:
                L.append(f"  {b['config']:<20} mean={b['mean_s']:.5f}s  items/s={b['items_per_s']}  gpu_peak={b['gpu_peak_alloc_mb']}")
            L.append("")
        if d["memory"]:
            L.append("-- memory --")
            for k, v in d["memory"].items():
                L.append(f"  {k}: {v}")
            L.append("")
        L.append("-- external reference required (NOT from dummy inference) --")
        for k, v in d["external_reference_required"].items():
            L.append(f"  {k}: {v}")
        return "\n".join(L) + "\n"

    def save(self, folder=None, stem="inference_metrics"):
        if folder is None:
            folder = os.path.dirname(os.path.abspath(sys.argv[0])) or "."
        self._finalize_memory()
        json_path = os.path.join(folder, stem + ".json")
        txt_path = os.path.join(folder, stem + ".txt")
        with open(json_path, "w") as f:
            json.dump(self.data, f, indent=2)
        with open(txt_path, "w") as f:
            f.write(self._pretty())
        print(f"[metrics] wrote {json_path}")
        print(f"[metrics] wrote {txt_path}")
        return json_path, txt_path
