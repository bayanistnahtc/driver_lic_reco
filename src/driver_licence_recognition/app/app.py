import uvicorn
import logging
from api.base_api import router as base_router
from api.v1.routes import router as predict_router
from app_settings import settings
from contextlib import asynccontextmanager
from fastapi import FastAPI
from starlette_exporter import PrometheusMiddleware, handle_metrics

from app_settings import settings


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(name)s: %(message)s")
logger = logging.getLogger("uvicorn")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Check dependencies")
    logger.info(f"settings: {settings.triton_host} : {settings.triton_port_http}")
    logger.info(f"config: {settings.config}")

    try:
        from dependencies.pipelines import get_cv_detection_pipeline, get_ocr_pipelines
        detection_pipeline = get_cv_detection_pipeline()
        ocr_pipelines = get_ocr_pipelines()

    except Exception:
        triton_config = settings.config.get("triton_config")
        raise ValueError(
            f'Can\'t resolve host "{triton_config.get("host")}:{triton_config.get("port")}" or model with name "{triton_config.get("model_name")}"'
        )

    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(PrometheusMiddleware, app_name="name service")
app.add_route("/metrics", handle_metrics)

app.include_router(base_router, prefix="/api/v1")
app.include_router(predict_router, prefix="/api/v1")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
