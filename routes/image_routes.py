# routes/image_routes.py
import json
import logging
import boto3
from fastapi import APIRouter, HTTPException, File, Form, UploadFile
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from schemas import ImageRequestPayload
from services import image_service
from config import config
logger = logging.getLogger(__name__)
router = APIRouter(prefix="/image", tags=["Images"])


AWS_REGION = config.AWS_REGION
SQS_QUEUE_URL = config.SQS_REGEN_QUEUE_URL
sqs_client = boto3.client("sqs", region_name=AWS_REGION) if SQS_QUEUE_URL else None

def _format_response(result):
    """Helper to consistently format the service responses."""
    if isinstance(result, str):
        return JSONResponse(content={"image_url": result})
    return JSONResponse(content=jsonable_encoder(result))

@router.post("/parcel", summary="Get Aerial Parcel Image")
async def get_parcel_image(payload: ImageRequestPayload):
    result = await image_service.get_parcel_image(
        gid=payload.parcel_gid, geom_input=payload.parcel_geojson, regenerate=payload.regenerate
    )
    return _format_response(result)

@router.post("/road-frontage", summary="Get Road Frontage Image")
async def get_road_image(payload: ImageRequestPayload):
    result = await image_service.get_road_frontage_image(
        gid=payload.parcel_gid, geom_input=payload.parcel_geojson, 
        features=payload.overlay_features, regenerate=payload.regenerate
    )
    return _format_response(result)

@router.post("/flood", summary="Get Flood Hazard Image")
async def get_flood_image(payload: ImageRequestPayload):
    result = await image_service.get_flood_image(
        gid=payload.parcel_gid, geom_input=payload.parcel_geojson, 
        features=payload.overlay_features, regenerate=payload.regenerate
    )
    return _format_response(result)

@router.post("/tree", summary="Get Tree Coverage Image")
async def get_tree_image(payload: ImageRequestPayload):
    result = await image_service.get_tree_image(
        gid=payload.parcel_gid, geom_input=payload.parcel_geojson, regenerate=payload.regenerate
    )
    return _format_response(result)

@router.post("/contour", summary="Get Elevation Contours Image")
async def get_contour_image(payload: ImageRequestPayload):
    result = await image_service.get_contour_image(
        gid=payload.parcel_gid, geom_input=payload.parcel_geojson, 
        features=payload.overlay_features, regenerate=payload.regenerate
    )
    return _format_response(result)

@router.post("/water", summary="Get Ponds, Creeks & Wetlands Image")
async def get_water_image(payload: ImageRequestPayload):
    result = await image_service.get_water_image(
        gid=payload.parcel_gid, geom_input=payload.parcel_geojson, 
        features=payload.overlay_features, regenerate=payload.regenerate
    )
    return _format_response(result)

@router.post("/pipeline", summary="Get Gas Pipelines Image")
async def get_pipeline_image(payload: ImageRequestPayload):
    result = await image_service.get_pipeline_image(
        gid=payload.parcel_gid, geom_input=payload.parcel_geojson, 
        features=payload.overlay_features, regenerate=payload.regenerate
    )
    return _format_response(result)

@router.post("/well", summary="Get Water Wells Image")
async def get_well_image(payload: ImageRequestPayload):
    result = await image_service.get_well_image(
        gid=payload.parcel_gid, geom_input=payload.parcel_geojson, 
        features=payload.overlay_features, regenerate=payload.regenerate
    )
    return _format_response(result)

@router.post("/gas-transmission", summary="Get Combined Gas & Electric Transmission Image")
async def get_gas_and_transmission_image(payload: ImageRequestPayload):
    result = await image_service.get_gas_transmission_image(
        gid=payload.parcel_gid, geom_input=payload.parcel_geojson, 
        features=payload.overlay_features, regenerate=payload.regenerate
    )
    return _format_response(result)

@router.post("/transmission", summary="Get Transmission Lines Image")
async def get_transmission_image(payload: ImageRequestPayload):
    result = await image_service.get_transmission_image(
        gid=payload.parcel_gid,
        geom_input=payload.parcel_geojson,
        features=payload.overlay_features,
        regenerate=payload.regenerate,
    )
    return _format_response(result)

@router.post("/electric", summary="Get Electric Transmission Lines Image")
async def get_electric_image(payload: ImageRequestPayload):
    result = await image_service.get_electric_image(
        gid=payload.parcel_gid,
        geom_input=payload.parcel_geojson,
        features=payload.overlay_features,
        regenerate=payload.regenerate,
    )
    return _format_response(result)

@router.post("/ponds-creeks", summary="Get Ponds and Creeks Image")
async def get_ponds_creeks_image(payload: ImageRequestPayload):
    result = await image_service.get_ponds_creeks_image(
        gid=payload.parcel_gid,
        geom_input=payload.parcel_geojson,
        features=payload.overlay_features,
        regenerate=payload.regenerate,
    )
    return _format_response(result)

@router.post("/county", summary="Get County & Surrounding Cities Image")
async def get_county_image(payload: ImageRequestPayload):
    result = await image_service.get_county_image(
        gid=payload.parcel_gid, geom_input=payload.parcel_geojson, 
        overlay_features=payload.overlay_features, regenerate=payload.regenerate
    )
    return _format_response(result)

@router.post("/upload", summary="Upload a custom image")
async def upload_custom_image(file: UploadFile = File(...), gid: int = Form(...)):
    if not gid:
        raise HTTPException(status_code=400, detail="gid (parcel ID) is required")

    try:
        file_bytes = await file.read()
    except Exception as exc:
        logger.error("Could not read uploaded file: %s", exc)
        raise HTTPException(status_code=400, detail="Could not read the uploaded file.")

    try:
        image_url = await image_service.upload_user_image(
            file_bytes=file_bytes,
            filename=file.filename,
            gid=gid,
            content_type=file.content_type or "application/octet-stream",
        )
        return JSONResponse(content={"status": "success", "image_url": image_url, "filename": file.filename})
    except Exception as exc:
        logger.error("Upload failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to upload image")

@router.post("/async-regen/{image_type}", summary="Queue Image for Background Generation")
async def queue_async_regeneration(image_type: str, payload: ImageRequestPayload):
    """Pushes payload to SQS for the Async Worker Lambda."""
    alias_map = {
        "gas_and_transmission": "gas_transmission",
        "road-frontage": "road_frontage",
        "ponds-creeks": "ponds_creeks",
    }
    image_type = alias_map.get(image_type, image_type)
    valid_types = [
        "parcel", "road_frontage", "flood", "tree", "contour",
        "water", "pipeline", "well", "gas_transmission", "county",
        "transmission", "electric", "ponds_creeks"
    ]
    if image_type not in valid_types:
        raise HTTPException(status_code=400, detail=f"Invalid type. Must be one of: {valid_types}")
        
    if not sqs_client or not SQS_QUEUE_URL:
        raise HTTPException(status_code=500, detail="SQS is not configured in this environment.")
        
    try:
        message_body = {"task_type": image_type, "payload": payload.model_dump()}
        message_body = jsonable_encoder(message_body)
        sqs_client.send_message(QueueUrl=SQS_QUEUE_URL, MessageBody=json.dumps(message_body))
        
        return JSONResponse(status_code=202, content={
            "status": "queued", 
            "message": f"Regeneration task for {image_type} sent to SQS worker."
        })
    except Exception as e:
        logger.error(f"Failed to queue SQS task: {e}")
        raise HTTPException(status_code=500, detail="Failed to queue background task.")