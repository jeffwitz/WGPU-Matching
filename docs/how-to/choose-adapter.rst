Choose a WebGPU adapter
=======================

The package uses a native WebGPU backend. `device_type="auto"` selects a hardware adapter when one is available; `"gpu"` requires hardware, and `"cpu"` requests a software CPU adapter that implements WebGPU. The latter still executes WebGPU shaders; it is not a CPU implementation of HORG or DeepMatching.

.. code-block:: python

   from wgpu_matching import WGPUMatching

   with WGPUMatching(device_type="gpu", power_preference="high-performance") as matcher:
       print(matcher.device_name)
       # Call match_horg(...) or match_deepmatching(...)

To prefer an integrated adapter, set `power_preference="low-power"`. `matcher.device_name` reports the backend and adapter name exposed by the driver; record it with performance measurements.

If device creation fails:

1. Check that the driver exposes WebGPU through Vulkan or another backend supported by `wgpu-py`.
2. Run `python -c 'import wgpu; print(wgpu.gpu.enumerate_adapters_sync())'` to list exposed adapters.
3. Use `device_type="auto"` if you do not need to require a hardware GPU.

See :class:`wgpu_matching.WGPUMatching` for supported values and errors.
