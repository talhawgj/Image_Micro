from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    """
    Configuration for the Stateless Serverless Image Generation Microservice.
    Loads from environment variables or a local .env file.
    """
    AWS_REGION: str = "us-east-2"
    AWS_BUCKET_IMAGES: str = "radcorp-images1"
    SQS_REGEN_QUEUE_URL: Optional[str] = None

    # --- Amazon EFS Mount Paths ---
    ELEVATION_FILE: str = "/mnt/land200/gis-data/tx_terrain/Texas_DEM.vrt"
    TREE_COVERAGE_PATH: str = "/mnt/land200/gis-data/tx_treecoverage"

    # --- External GIS API Keys ---
    GOOGLE_MAPS_API_KEY: str = ""
    VEXCEL_API_KEY: str = ""
    VEXCEL_LAYER: str = "urban"
    MAPBOX_TOKEN: str = ""
    
    model_config = SettingsConfigDict(env_file=".env", extra='ignore')

config = Settings()


#aws ecr get-login-password --region us-east-2 | docker login --username AWS --password-stdin 525133621869.dkr.ecr.us-east-2.amazonaws.com
# docker tag gis-image-service:latest 525133621869.dkr.ecr.us-east-2.amazonaws.com/gis-image-service:latest

# docker push 525133621869.dkr.ecr.us-east-2.amazonaws.com/gis-image-service:latest