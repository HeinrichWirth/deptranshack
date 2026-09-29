"""Build all native dependencies from source using strict floating-point rules."""

import subprocess, sysconfig
from pathlib import Path
import pybind11

output = Path("/compiled")
(output / "native").mkdir(parents=True)
for stage in ("performance_final", "performance_final2", "c4_v2", "realtime_final"):
    subprocess.run(
        [
            "cmake",
            "-S",
            f"/native/{stage}",
            "-B",
            f"/build/{stage}",
            "-DCMAKE_BUILD_TYPE=Release",
            f"-Dpybind11_DIR={pybind11.get_cmake_dir()}",
        ],
        check=True,
    )
    subprocess.run(["cmake", "--build", f"/build/{stage}", "--parallel", "2"], check=True)
    subprocess.run(
        ["cmake", "--install", f"/build/{stage}", "--prefix", "/compiled/app"], check=True
    )
for source, module in [("envelope", "_envelope"), ("obstacle_envelope", "_obstacle_envelope")]:
    subprocess.run(
        [
            "g++",
            "-O3",
            "-shared",
            "-std=c++17",
            "-fPIC",
            "-fno-fast-math",
            "-ffp-contract=off",
            "-I" + pybind11.get_include(),
            "-I" + sysconfig.get_paths()["include"],
            f"/native/{source}.cpp",
            "-o",
            str(output / "native" / (module + sysconfig.get_config_var("EXT_SUFFIX"))),
        ],
        check=True,
    )
