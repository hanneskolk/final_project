import time
from ultralytics import YOLO
from pathlib import Path
import numpy as np

images = list(Path("dataset/test/images").glob("*.jpg"))[:100]
results = {}

for backend, path in [("PyTorch", "best.pt"),
                       ("ONNX",    "best.onnx"),
                       ("TensorRT","best.engine")]:
    model  = YOLO(path)
    times  = []
    for img in images:
        t0 = time.perf_counter()
        model(str(img), verbose=False)
        times.append((time.perf_counter() - t0) * 1000)
    results[backend] = times
    print(f"{backend}: mean={np.mean(times):.1f}ms  "
          f"std={np.std(times):.1f}ms  "
          f"min={np.min(times):.1f}ms  "
          f"max={np.max(times):.1f}ms")