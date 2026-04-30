
from pydantic import BaseModel, Field, model_validator
from typing import Optional, List, Dict, Any

class OverlayFeature(BaseModel):
    """Represents an intersecting feature like a flood zone, road, or city."""
    geojson: Optional[Any] = None
    geometry: Optional[Any] = None
    properties: Optional[Dict[str, Any]] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def normalize_geometry_payload(cls, data: Any):
        if isinstance(data, dict):
            if "geojson" not in data and "geometry" in data:
                data = {**data, "geojson": data.get("geometry")}
        return data

class ImageRequestPayload(BaseModel):
    parcel_gid: int
    parcel_geojson: str
    regenerate: bool = False
    overlay_features: Optional[List[OverlayFeature]] = Field(default_factory=list)