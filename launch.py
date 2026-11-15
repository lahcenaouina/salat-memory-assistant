"""
Run:  python launch.py
Needs one of these files in the same folder:
  salat.onnx / salat_best.onnx   (already exported)
  salat.pt   / salat_best.pt     (exported automatically on first run)
"""
import http.server
import json
import os
import socketserver
import sys
import threading
import webbrowser

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
PORT = 8000
DEFAULT_CLASSES = ["qiyam", "ruku", "sitting", "sujud"]

def first_existing(names):
    return next((n for n in names if os.path.exists(n)), None)

onnx = first_existing(["salat.onnx", "salat_best.onnx"])
pt = first_existing(["salat.pt", "salat_best.pt"])
classes = None

# Export from .pt if there is no .onnx yet
if onnx is None:
    if pt is None:
        sys.exit("Put salat.pt (or salat.onnx) in this folder first.")
    print("Exporting model to ONNX (first run only)...")
    from ultralytics import YOLO  # pip install ultralytics
    m = YOLO(pt)
    onnx = os.path.basename(m.export(format="onnx", imgsz=640, opset=12, simplify=True))
    classes = [m.names[i] for i in sorted(m.names)]
elif pt is not None:
    # Read class names from the .pt if ultralytics is available
    try:
        from ultralytics import YOLO
        m = YOLO(pt)
        classes = [m.names[i] for i in sorted(m.names)]
    except Exception:
        pass

if classes is None:
    classes = DEFAULT_CLASSES
    print("Warning: using default class names:", classes)

with open("config.json", "w") as f:
    json.dump({"model": onnx, "classes": classes}, f)
print("Model:", onnx, "| Classes:", classes)

class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map,
                      ".onnx": "application/octet-stream",
                      ".wasm": "application/wasm"}
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()
    def log_message(self, *args):
        pass

socketserver.TCPServer.allow_reuse_address = True
with socketserver.TCPServer(("192.168.1.9", PORT), Handler) as server:
    url = f"http://192.168.1.9:{PORT}"
    print(f"Open {url}  (Ctrl+C to stop)")
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")