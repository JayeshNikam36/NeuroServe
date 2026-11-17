"""GPU state during a benchmark run, sampled with NVML (nvidia-ml-py).

Clocks, temperature and power change under sustained load (a laptop GPU throttles when it gets
hot or reaches its power limit), and a throttled run is slower for reasons that have nothing to do
with the code being measured. Every benchmark run records these samples, so such a run is visible
instead of silently skewing a comparison.

    with GpuSampler(device_index=0) as sampler:
        ...                       # the run
    trace = sampler.result()      # {"available": True, "samples": [...], "summary": {...}}
"""

from __future__ import annotations

import statistics
import threading
import time
from typing import Any

# Clock event ("throttle") reason bits, as defined by NVML (nvml.h).
SW_POWER_CAP = 0x4
HW_SLOWDOWN = 0x8             # thermal or power-brake slowdown, set by the hardware
SW_THERMAL_SLOWDOWN = 0x20
HW_THERMAL_SLOWDOWN = 0x40
HW_POWER_BRAKE_SLOWDOWN = 0x80
THERMAL_REASONS = HW_SLOWDOWN | SW_THERMAL_SLOWDOWN | HW_THERMAL_SLOWDOWN
POWER_REASONS = SW_POWER_CAP | HW_POWER_BRAKE_SLOWDOWN


class GpuSampler:
    """Samples one GPU every interval_s seconds in a background thread.

    If NVML is not available (no nvidia-ml-py, no driver, no GPU) the run goes ahead and the result
    says why there are no samples. A field the driver does not support is recorded as None.
    """

    def __init__(self, device_index: int = 0, interval_s: float = 0.2, nvml: Any = None) -> None:
        self._device_index = device_index
        self._interval_s = interval_s
        self._nvml = nvml               # tests pass a fake; None: import pynvml
        self._handle: Any = None
        self._name: str | None = None
        self._reason: str | None = None  # why there are no samples
        self._samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._t0 = 0.0

    def __enter__(self) -> "GpuSampler":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def start(self) -> None:
        if self._nvml is None:
            try:
                import pynvml
            except ImportError:
                self._reason = "nvidia-ml-py is not installed"
                return
            self._nvml = pynvml
        try:
            self._nvml.nvmlInit()
            self._handle = self._nvml.nvmlDeviceGetHandleByIndex(self._device_index)
            name = self._nvml.nvmlDeviceGetName(self._handle)
            self._name = name.decode() if isinstance(name, bytes) else name
        except self._nvml.NVMLError as exc:
            self._reason = f"NVML unavailable: {exc}"
            self._handle = None
            return
        self._t0 = time.monotonic()
        self._samples.append(self._sample())  # one at the very start
        self._thread = threading.Thread(target=self._loop, name="gpu-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is not None:
            self._stop.set()
            self._thread.join()
            self._thread = None
            self._samples.append(self._sample())  # and one at the very end
        if self._handle is not None:
            self._handle = None
            try:
                self._nvml.nvmlShutdown()
            except self._nvml.NVMLError:
                pass

    def result(self) -> dict[str, Any]:
        return {"available": self._reason is None and bool(self._samples), "reason": self._reason,
                "device_index": self._device_index, "name": self._name, "interval_s": self._interval_s,
                "summary": summarize_samples(self._samples), "samples": list(self._samples)}

    def _loop(self) -> None:
        while not self._stop.wait(self._interval_s):
            self._samples.append(self._sample())

    def _read(self, fn_name: str, *args: Any) -> Any:
        fn = getattr(self._nvml, fn_name, None)
        if fn is None:
            return None
        try:
            return fn(self._handle, *args)
        except self._nvml.NVMLError:
            return None  # not supported by this driver or GPU (some fields are not, under WSL)

    def _sample(self) -> dict[str, Any]:
        nvml = self._nvml
        power_mw = self._read("nvmlDeviceGetPowerUsage")
        util = self._read("nvmlDeviceGetUtilizationRates")
        reasons = self._read("nvmlDeviceGetCurrentClocksEventReasons")  # the newer name, then the older
        if reasons is None:
            reasons = self._read("nvmlDeviceGetCurrentClocksThrottleReasons")
        return {
            "t_s": round(time.monotonic() - self._t0, 3),
            "sm_clock_mhz": self._read("nvmlDeviceGetClockInfo", nvml.NVML_CLOCK_SM),
            "mem_clock_mhz": self._read("nvmlDeviceGetClockInfo", nvml.NVML_CLOCK_MEM),
            "temp_c": self._read("nvmlDeviceGetTemperature", nvml.NVML_TEMPERATURE_GPU),
            "power_w": round(power_mw / 1000.0, 2) if power_mw is not None else None,
            "util_pct": util.gpu if util is not None else None,
            "reasons": reasons,
        }


def summarize_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Medians and extremes over a run (None where no sample had the field)."""

    def values(key: str) -> list[float]:
        return [s[key] for s in samples if s.get(key) is not None]

    def fraction(mask: int) -> float | None:
        reasons = values("reasons")
        return round(sum(1 for r in reasons if r & mask) / len(reasons), 3) if reasons else None

    sm, temp, power, util = values("sm_clock_mhz"), values("temp_c"), values("power_w"), values("util_pct")
    return {
        "samples": len(samples),
        "sm_clock_mhz_median": statistics.median(sm) if sm else None,
        "sm_clock_mhz_min": min(sm) if sm else None,
        "temp_c_max": max(temp) if temp else None,
        "power_w_mean": round(statistics.fmean(power), 2) if power else None,
        "util_pct_mean": round(statistics.fmean(util), 1) if util else None,
        "thermal_slowdown_frac": fraction(THERMAL_REASONS),  # share of samples with a thermal reason set
        "power_cap_frac": fraction(POWER_REASONS),
    }
