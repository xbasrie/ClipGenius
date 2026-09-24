import sys
import os
import webbrowser
import threading
import time
import socket
import subprocess
import multiprocessing
import uvicorn

# Set root directory for assets
from pathlib import Path
_app_cfg = Path(os.environ.get("LOCALAPPDATA") or os.path.expanduser("~/.local/share")) / "ClipGenius" / "workspace.txt"
if _app_cfg.exists():
    try:
        _saved = _app_cfg.read_text("utf-8").strip()
        if _saved and Path(_saved).exists():
            os.environ.setdefault("CLIPGENIUS_HOME", _saved)
    except Exception:
        pass
os.environ.setdefault("CLIPGENIUS_HOME", r"D:\clipgenius_data")
os.environ.setdefault("CLIPGENIUS_PORT", "8089")
os.environ.setdefault("CLIPGENIUS_TOKEN", "dev-local-token")

def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0

def open_ui(port: int):
    time.sleep(1.2)
    url = f"http://127.0.0.1:{port}/"
    edge_paths = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ]
    for p in edge_paths:
        if os.path.exists(p):
            try:
                subprocess.Popen([p, f"--app={url}"])
                return
            except Exception:
                pass
    webbrowser.open(url)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    port = int(os.environ.get("CLIPGENIUS_PORT", 8089))
    
    if is_port_in_use(port):
        open_ui(port)
        sys.exit(0)

    t = threading.Thread(target=open_ui, args=(port,), daemon=True)
    t.start()
    
    from pipeline.server import app
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")

