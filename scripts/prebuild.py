import shutil
import requests


fps_cu_to_cpp = [
    "Detector",
]
fps_pyx = ["pycppdetector"]

for fp in fps_cu_to_cpp:
    shutil.copy("lisatools/cutils/src/" + fp + ".cu", "lisatools/cutils/src/" + fp + ".cpp")

for fp in fps_pyx:
    shutil.copy("lisatools/cutils/src/" + fp + ".pyx", "lisatools/cutils/src/" + fp + "_cpu.pyx")


# setup version file
with open("README.md", "r") as fh:
    lines = fh.readlines()

