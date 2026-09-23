import sys
import os
import webbrowser
import threading
import time
import uvicorn

# Set root directory for assets
os.environ.setdefault("CLIPGENIUS_HOME", r"D:\clipgenius_data")
os.environ.setdefault("CLIPGENIUS_PORT", "8089")
os.environ.setdefault("CLIPGENIUS_TOKEN", "dev-local-token")

def open_browser():
    time.sleep(1.8)
    webbrowser.open("http://127.0.0.1:8089/")

if __name__ == "__main__":
    t = threading.Thread(target=open_browser, daemon=True)
    t.start()
    
    from pipeline.server import app
    port = int(os.environ.get("CLIPGENIUS_PORT", 8089))
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")
