import os

from fastapi import FastAPI, Response
from fastapi.responses import RedirectResponse

from app.api.v1.health import check_system_health
from app.api.v1.router import api_v1_router
from app.core.config import settings
from app.core.lifespan import lifespan
from app.core.middleware import register_middleware

version = "v1"
version_prefix = f"/api/{version}"

app = FastAPI(
    title="Market Research Intelligence Agent API",
    description="Market Research Intelligence Agent API",
    version=version,
    openapi_url=f"{version_prefix}/openapi.json",
    docs_url=f"{version_prefix}/docs",
    redoc_url=f"{version_prefix}/redoc",
    lifespan=lifespan,
    license_info={"name": "MIT License", "url": "https://opensource.org/license/mit"},
)

register_middleware(app)


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url=f"{version_prefix}/docs")


@app.get("/docs", include_in_schema=False)
async def docs_redirect():
    return RedirectResponse(url=f"{version_prefix}/docs")


@app.get("/health", include_in_schema=True, tags=["HEALTH"])
async def root_health(response: Response):
    """Root health probe for platform health checks (e.g. Railway, container runners)."""
    return await check_system_health(response)


app.include_router(api_v1_router, prefix=version_prefix)

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", settings.host)
    port = int(os.getenv("PORT", str(settings.port)))
    uvicorn.run("app.main:app", host=host, port=port)
