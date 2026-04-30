# workers/sqs_handler.py
"""
SQS Retry Worker
================
Deployed as a separate Lambda function triggered by the SQS queue.

When Image_Micro fails to render an image it pushes the job payload to SQS.
This handler receives those messages, re-invokes the correct image generator,
and uploads the result to S3 – guaranteeing eventual image delivery.

AWS SAM / CDK wiring example
-----------------------------
  EventSourceMapping:
    EventSourceArn: !GetAtt ImageRegenQueue.Arn
    FunctionName:   !Ref SqsRetryWorkerFunction
    BatchSize:      1   # process one job at a time to respect Chrome memory

Lambda entry-point: ``workers.sqs_handler.handler``
"""

import asyncio
import json
import logging
from typing import Any, Dict, List

from services.image import ImageService

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# One ImageService instance per container (warm Lambda reuse)
_svc = ImageService()

# Map the folder_name / task_type string to the matching ImageService method
_TASK_MAP = {
    "aerial":                   _svc.get_parcel_image,
    "parcel":                   _svc.get_parcel_image,
    "road_frontage":            _svc.get_road_frontage_image,
    "flood_hazard":             _svc.get_flood_image,
    "flood":                    _svc.get_flood_image,
    "tree_coverage":            _svc.get_tree_image,
    "tree":                     _svc.get_tree_image,
    "contour":                  _svc.get_contour_image,
    "water_features":           _svc.get_water_image,
    "water":                    _svc.get_water_image,
    "gas_pipelines":            _svc.get_pipeline_image,
    "pipeline":                 _svc.get_pipeline_image,
    "gas_transmission_lines":   _svc.get_gas_transmission_image,
    "gas_transmission":         _svc.get_gas_transmission_image,
    "water_wells":              _svc.get_well_image,
    "well":                     _svc.get_well_image,
    "ponds_creeks":             _svc.get_ponds_creeks_image,
    "county_boundary":          _svc.get_county_image,
    "county":                   _svc.get_county_image,
    "electric_lines":           _svc.get_electric_image,
    "electric":                 _svc.get_electric_image,
    "transmission_lines":       _svc.get_transmission_image,
    "transmission":             _svc.get_transmission_image,
}

# Image types that accept overlay_features
_NEEDS_FEATURES = {
    "road_frontage", "road-frontage",
    "flood_hazard", "flood",
    "contour",
    "water_features", "water",
    "gas_pipelines", "pipeline",
    "gas_transmission_lines", "gas_transmission",
    "water_wells", "well",
    "ponds_creeks",
    "county_boundary", "county",
    "electric_lines", "electric",
    "transmission_lines", "transmission",
}


async def _process_record(record: Dict[str, Any]) -> None:
    """Process a single SQS message record."""
    try:
        body = json.loads(record["body"])
    except (KeyError, json.JSONDecodeError) as exc:
        logger.error("Malformed SQS message body: %s | error: %s", record.get("body"), exc)
        return

    gid: int = body.get("parcel_gid")
    task_type: str = body.get("task_type", "").strip()
    geom: str = body.get("parcel_geojson", "")
    features: List[Dict] = body.get("overlay_features", [])
    regenerate: bool = body.get("regenerate", True)

    if not gid or not geom:
        logger.error("SQS message missing parcel_gid or parcel_geojson: %s", body)
        return

    func = _TASK_MAP.get(task_type)
    if func is None:
        logger.error("Unknown task_type '%s' in SQS message for GID %s", task_type, gid)
        return

    logger.info("Retrying %s for GID %s (regenerate=%s)", task_type, gid, regenerate)

    try:
        if task_type in _NEEDS_FEATURES:
            result = await func(gid=gid, geom_input=geom, features=features, regenerate=regenerate)
        else:
            result = await func(gid=gid, geom_input=geom, regenerate=regenerate)

        if isinstance(result, str):
            logger.info("Retry SUCCESS for GID %s / %s → %s", gid, task_type, result)
        else:
            # If it's still a dict (no_data or another failure) we just log —
            # do NOT re-enqueue to avoid infinite retry loops.
            logger.warning(
                "Retry for GID %s / %s returned non-URL: %s", gid, task_type, result
            )
    except Exception as exc:
        # Let the exception propagate so SQS can apply its redrive policy
        # (move to DLQ after maxReceiveCount attempts).
        logger.error(
            "Retry handler raised for GID %s / %s: %s", gid, task_type, exc, exc_info=True
        )
        raise


def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Lambda entry-point invoked by the SQS event source mapping.
    Each invocation receives a batch of records (BatchSize=1 recommended).
    """
    records: List[Dict] = event.get("Records", [])
    logger.info("SQS retry worker received %d record(s)", len(records))

    failed: List[Dict] = []

    for record in records:
        try:
            asyncio.run(_process_record(record))
        except Exception:
            # Report as partial batch failure so SQS retries only this message
            failed.append({"itemIdentifier": record["messageId"]})

    if failed:
        # Partial batch response — only failed messages go back to the queue
        return {"batchItemFailures": failed}

    return {"batchItemFailures": []}
