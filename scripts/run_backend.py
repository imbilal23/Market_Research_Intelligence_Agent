import os
import sys
from pathlib import Path
import uvicorn

root_dir = Path(__file__).resolve().parent.parent
backend_dir = root_dir / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

if __name__ == "__main__":
    from app.core.config import settings

    host = os.getenv("HOST", settings.host)
    port = int(os.getenv("PORT", str(settings.port)))
    reload = os.getenv("RELOAD", "false").lower() in ("true", "1")
    uvicorn.run("app.main:app", host=host, port=port, reload=reload)
