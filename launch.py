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
IMGSZ = 320   # model input size: 320 = fast (~4x faster, target 5 fps), 640 = slower but more accurate
DEFAULT_CLASSES = ["qiyam", "ruku", "sitting", "sujud"]

def first_existing(names):
    return next((n for n in names if os.path.exists(n)), None)

old = {}
try:
    with open("config.json") as f:
        old = json.load(f)
except Exception:
    pass

onnx = first_existing(["salat.onnx", "salat_best.onnx"])
pt = first_existing(["salat.pt", "salat_best.pt"])
classes = None
size = old.get("size", 640)   # size of an .onnx exported earlier

# (Re)export from .pt when there is no .onnx yet, or it was exported at another size
if pt is not None and (onnx is None or old.get("size") != IMGSZ):
    try:
        print(f"Exporting model to ONNX at {IMGSZ}px (first run only)...")
        from ultralytics import YOLO  # pip install ultralytics
        m = YOLO(pt)
        onnx = os.path.basename(m.export(format="onnx", imgsz=IMGSZ, opset=12, simplify=True))
        classes = [m.names[i] for i in sorted(m.names)]
        size = IMGSZ
    except Exception as e:
        if onnx is None:
            sys.exit(f"Export failed: {e}")
        print("Warning: export failed, keeping existing", onnx, "-", e)
elif onnx is None:
    sys.exit("Put salat.pt (or salat.onnx) in this folder first.")
elif pt is not None:
    try:
        from ultralytics import YOLO
        m = YOLO(pt)
        classes = [m.names[i] for i in sorted(m.names)]
    except Exception:
        pass

if classes is None:
    classes = old.get("classes", DEFAULT_CLASSES)
    print("Warning: using class names:", classes)

with open("config.json", "w") as f:
    json.dump({"model": onnx, "classes": classes, "size": size}, f)
print("Model:", onnx, "| Size:", size, "| Classes:", classes)

class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map,
                      ".onnx": "application/octet-stream",
                      ".wasm": "application/wasm"}
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        # Lets the browser use several CPU threads for the model (faster)
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        super().end_headers()
    def log_message(self, *args):
        pass

socketserver.TCPServer.allow_reuse_address = True
with socketserver.TCPServer(("127.0.0.1", PORT), Handler) as server:
    url = f"http://localhost:{PORT}"
    print(f"Open {url}  (Ctrl+C to stop)")
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")