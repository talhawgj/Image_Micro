import logging
from fastapi import FastAPI
from mangum import Mangum
from routes import image_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="GIS Image Generation Microservice",
    description="Stateless rendering engine",
)

app.include_router(image_router)


@app.get("/health", tags=["Health Check"])
async def health_check():
    return {"status": "healthy", "architecture": "serverless"}


# Single Lambda entry-point – Lambda Function URL
# api_gateway_base_path is intentionally omitted (None = no prefix stripping).
# Function URLs forward the path as-is: /image/parcel, /image/flood, etc.
handler = Mangum(app, lifespan="off")