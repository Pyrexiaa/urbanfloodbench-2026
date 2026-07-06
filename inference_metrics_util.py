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
        self.rec._section()["phases"][self.name] = {
            "wall_s": wall,
            "cpu_s": cpu,
            "gpu_peak_alloc_mb": gpu_peak_alloc_mb(),
        }
        return False


# ---------------------------------------------------------------------------
# recorder
# ---------------------------------------------------------------------------
class MetricsRecorder:
    def __init__(self, solution, framework, device=None):
        self.data = {
            "solution": solution,
            "framework": framework,
            "device": None,
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "hardware": {},
            # Top-level buckets are used when no device is set (single-device /
            # backward-compatible use). When set_device() is called, metrics are
            # written into a per-device section under "devices" instead.
            "model": {},
            "phases": {},
            "timing": {},
            "throughput": {},
            "batch_sensitivity": [],
            "memory": {},
            "devices": {},
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

        # Current per-device section (None => write to top-level buckets).
        self._cur_device = None
        if device is not None:
            self.set_device(device)

    # ---- device selection ----
    @staticmethod
    def _new_section():
        return {
            "model": {},
            "phases": {},
            "timing": {},
            "throughput": {},
            "batch_sensitivity": [],
            "memory": {},
        }

    def _section(self):
        """The dict metrics are currently written into (device or top-level)."""
        if self._cur_device is None:
            return self.data
        return self.data["devices"][self._cur_device]

    def set_device(self, device):
        """Direct subsequent metrics into a section for `device`.

        Lets a single recorder capture separate GPU and CPU sections in one
        JSON. Returns the device so callers can do `dev = rec.set_device(dev)`.
        """
        key = str(device)
        self.data["device"] = key
        self._cur_device = key
        if key not in self.data["devices"]:
            self.data["devices"][key] = self._new_section()
        return device

    def devices_to_run(self):
        """Devices to benchmark: [cuda, cpu] if a GPU is visible, else [cpu]."""
        t = _torch()
        if t is not None:
            if t.cuda.is_available():
                return [t.device("cuda"), t.device("cpu")]
            return [t.device("cpu")]
        # No torch: fall back to string labels (JAX callers build their own list)
        return ["cpu"]

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
        self._section()["model"][name] = {
            "n_params": None if n_params is None else int(n_params),
            "size_mb": size_mb,
            **extra,
        }

    # ---- timing statistics from a list of per-run times ----
    def set_timing(self, label, times):
        arr = [float(x) for x in times]
        if not arr:
            return
        self._section()["timing"][label] = {
            "runs": len(arr),
            "mean_s": statistics.mean(arr),
            "std_s": statistics.pstdev(arr) if len(arr) > 1 else 0.0,
            "median_s": statistics.median(arr),
            "min_s": min(arr),
            "max_s": max(arr),
        }
        return self._section()["timing"][label]

    def set_throughput(self, label, items, mean_s):
        self._section()["throughput"][label] = {
            "items": items,
            "mean_s": mean_s,
            "items_per_s": (items / mean_s) if mean_s else None,
        }

    # ---- batch / rollout sensitivity ----
    def add_batch_point(
        self, config, times=None, mean_s=None, items=None, gpu_peak_mb=None, **extra
    ):
        """Record one batch/rollout sweep point.

        Accepts either a list of per-run `times` (mean/std are derived) or a
        precomputed `mean_s`.
        """
        std_s = None
        if times is not None:
            arr = [float(x) for x in times]
            if arr:
                mean_s = statistics.mean(arr)
                std_s = statistics.pstdev(arr) if len(arr) > 1 else 0.0
        self._section()["batch_sensitivity"].append(
            {
                "config": config,
                "mean_s": mean_s,
                "std_s": std_s,
                "items": items,
                "items_per_s": (items / mean_s) if (items and mean_s) else None,
                "gpu_peak_alloc_mb": gpu_peak_mb,
                **extra,
            }
        )

    def set(self, key, value):
        self.data[key] = value

    # ---- finalize + write ----
    def _finalize_memory(self):
        mem = self._section()["memory"]
        mem["peak_cpu_rss_mb"] = _peak_cpu_rss_mb()
        if _cuda_available():
            t = _torch()
            mem["gpu_max_allocated_mb"] = t.cuda.max_memory_allocated() / 1e6
            mem["gpu_max_reserved_mb"] = t.cuda.max_memory_reserved() / 1e6
        else:
            gp = gpu_peak_alloc_mb()
            if gp is not None:
                mem["gpu_peak_alloc_mb"] = gp

    @staticmethod
    def _pretty_section(sec, L, indent=""):
        if sec.get("model"):
            L.append(indent + "-- model --")
            for k, v in sec["model"].items():
                L.append(
                    indent
                    + f"  {k}: params={v.get('n_params')}, size_mb={v.get('size_mb')}"
                )
            L.append("")
        if sec.get("phases"):
            L.append(indent + "-- phase timing (wall / cpu seconds; gpu peak MB) --")
            for k, v in sec["phases"].items():
                L.append(
                    indent
                    + f"  {k:<16} wall={v['wall_s']:.5f}  cpu={v['cpu_s']:.5f}  gpu_peak={v['gpu_peak_alloc_mb']}"
                )
            L.append("")
        if sec.get("timing"):
            L.append(indent + "-- benchmark timing --")
            for k, v in sec["timing"].items():
                L.append(
                    indent
                    + f"  {k:<16} mean={v['mean_s']:.5f}s  std={v['std_s']:.5f}  median={v['median_s']:.5f}  (n={v['runs']})"
                )
            L.append("")
        if sec.get("batch_sensitivity"):
            L.append(indent + "-- batch / rollout sensitivity --")
            for b in sec["batch_sensitivity"]:
                _m = b["mean_s"]
                _mean = f"{_m:.5f}s" if isinstance(_m, (int, float)) else str(_m)
                L.append(
                    indent
                    + f"  {str(b['config']):<20} mean={_mean}  items/s={b['items_per_s']}  gpu_peak={b['gpu_peak_alloc_mb']}"
                )
            L.append("")
        if sec.get("memory"):
            L.append(indent + "-- memory --")
            for k, v in sec["memory"].items():
                L.append(indent + f"  {k}: {v}")
            L.append("")

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
            L.append(
                "hardware  : " + ", ".join(f"{k}={v}" for k, v in d["hardware"].items())
            )
        L.append("")
        # Top-level section (used when no device was set).
        self._pretty_section(d, L)
        # Per-device sections.
        for dev, sec in d.get("devices", {}).items():
            L.append("#" * 64)
            L.append(f"# DEVICE: {dev}")
            L.append("#" * 64)
            self._pretty_section(sec, L)
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
