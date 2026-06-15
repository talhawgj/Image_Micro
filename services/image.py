# services/image.py
import os
import json
import asyncio
import base64
import shutil
import tempfile
import math
from collections.abc import Callable
from io import BytesIO
from typing import Dict, Optional, Union, Tuple, List, Any
import rasterio.features
import boto3
from botocore.exceptions import ClientError
import numpy as np
import rasterio
from rasterio.mask import mask
from PIL import Image,ImageFilter
import folium
import geopandas as gpd
from shapely.geometry import shape, mapping, Polygon
from shapely import wkt
from config import config
from utils import get_chrome_driver
from scipy.ndimage import zoom
import logging

logger = logging.getLogger(__name__)

class ImageService:
    """
    Centralized Serverless service for generating GIS analysis images.
    Optimized for high-resolution (2048x1536) rendering per Cartography Style Guide Rev.0.
    Stateless implementation: receives geometries directly from payloads.
    """

    def __init__(self):
        self.s3_bucket = config.AWS_BUCKET_IMAGES
        self.s3_region = config.AWS_REGION
        self.s3_base_url = f"https://{self.s3_bucket}.s3.{self.s3_region}.amazonaws.com"
        self.dem_path = config.ELEVATION_FILE
        self.tree_path = config.TREE_COVERAGE_PATH
        self._s3_client = boto3.client("s3", region_name=self.s3_region)
        self.STYLE_COLOR = "#FFEB3B"  
        self.STYLE_CASING = "#FFFFFF"  
    def _km_to_degree_deltas(self, km: float, latitude: float = 32.0) -> Tuple[float, float]:
        safe_lat = max(min(latitude, 85.0), -85.0)
        cos_lat = max(abs(math.cos(math.radians(safe_lat))), 0.2)
        delta_lat = km / 111.0
        delta_lon = km / (111.0 * cos_lat)
        return delta_lon, delta_lat

    def _feature_properties(self, feature: Any) -> Dict[str, Any]:
        if hasattr(feature, "properties") and isinstance(feature.properties, dict):
            return feature.properties
        if isinstance(feature, dict):
            props = feature.get("properties")
            if isinstance(props, dict):
                return props
        return {}

    def _feature_geometry(self, feature: Any) -> Any:
        """Parse geometry from model or dict payloads.

        Supports:
        - OverlayFeature(geojson='{"type":"LineString", ...}')
        - OverlayFeature(geojson={...})
        - Feature-like dict with {"geometry": {...}, "properties": {...}}
        """
        geo_value = getattr(feature, "geojson", None)
        if geo_value is None and isinstance(feature, dict):
            geo_value = feature.get("geojson", feature.get("geometry"))

        if geo_value is None:
            raise ValueError("Feature has no geometry payload")

        if isinstance(geo_value, str):
            parsed = json.loads(geo_value)
        elif isinstance(geo_value, dict):
            parsed = geo_value
        else:
            raise ValueError("Unsupported geometry payload type")

        if parsed.get("type") == "Feature":
            parsed = parsed.get("geometry") or {}

        return shape(parsed)

    def _classify_utility_feature(self, props: Dict[str, Any]) -> str:
        """Classify utility feature as gas/electric/unknown from heterogeneous payloads."""
        text_blob = " ".join(str(v) for v in props.values() if v is not None).lower()

        electric_markers = ["ac", "overhead", "underground", "kv", "volt", "transmission", "electric", "power"]
        gas_markers = ["gas", "pipeline", "oil", "commodity", "cmdty", "lng", "nng"]

        has_electric = any(marker in text_blob for marker in electric_markers)
        has_gas = any(marker in text_blob for marker in gas_markers)

        if has_electric and not has_gas:
            return "electric"
        if has_gas and not has_electric:
            return "gas"

        t = str(props.get("type") or props.get("TYPE") or "").lower()
        if t in {"electric", "power", "transmission", "ac; overhead", "ac; underground", "overhead"}:
            return "electric"
        if t in {"gas", "pipeline", "oil"}:
            return "gas"

        return "unknown"

    def _get_upload_s3_key(self, gid: int, filename: str) -> str:
        """Generate S3 key for user-uploaded images under parcels/{gid}/upload/."""
        return f"parcels/{gid}/upload/{filename}"

    async def upload_user_image(
        self,
        file_bytes: bytes,
        filename: str,
        gid: int,
        content_type: str = "application/octet-stream",
    ) -> str:
        """Upload a user-provided image directly to S3 and return a public URL."""
        s3_key = self._get_upload_s3_key(gid, filename)
        file_obj = BytesIO(file_bytes)
        await asyncio.to_thread(
            self._s3_client.upload_fileobj,
            file_obj,
            self.s3_bucket,
            s3_key,
            ExtraArgs={"ContentType": content_type},
        )
        return f"{self.s3_base_url}/{s3_key}"

    async def _handle_cache_or_generate(
        self,
        gid: int,
        folder_name: str,
        generate_func: Callable[..., Any],
        geom_input: str,
        regenerate: bool = False,
        **kwargs,
    ) -> Union[str, Dict]:
        s3_key = f"parcels/{gid}/{folder_name}_{gid}.png"

        if regenerate:
            try:
                await asyncio.to_thread(
                    self._s3_client.delete_object, Bucket=self.s3_bucket, Key=s3_key
                )
            except Exception:
                pass
        else:
            try:
                await asyncio.to_thread(
                    self._s3_client.head_object, Bucket=self.s3_bucket, Key=s3_key
                )
                logger.info("Cache hit: %s", s3_key)
                return f"{self.s3_base_url}/{s3_key}"
            except ClientError as exc:
                if exc.response["Error"]["Code"] not in ("404", "NoSuchKey"):
                    raise
            except Exception:
                pass

        try:
            result = await generate_func(gid, geom_input, **kwargs)
        except Exception as exc:
            error_msg = str(exc)
            logger.error("Render failed for GID %s / %s: %s", gid, folder_name, error_msg)
            return {"status": "failed", "error": error_msg}

        # generate_func returns a dict when there is no data to render
        if isinstance(result, dict):
            return result

        try:
            await asyncio.to_thread(
                self._s3_client.upload_fileobj,
                result,
                self.s3_bucket,
                s3_key,
                ExtraArgs={"ContentType": "image/png"},
            )
            return f"{self.s3_base_url}/{s3_key}"
        except Exception as exc:
            error_msg = str(exc)
            logger.error("S3 upload failed for GID %s / %s: %s", gid, folder_name, error_msg)
            return {"status": "failed", "error": error_msg}

    def _get_geometry_and_bounds(self, geom_input: str, buffer_km: float = 0.01) -> Tuple[Any, List[float]]:
        try:
            data = json.loads(geom_input) if geom_input.strip().startswith("{") else None
            shapely_geom = shape(data) if data else wkt.loads(geom_input)
        except Exception as e:
            raise ValueError(f"Invalid geometry input: {e}")

        minx, miny, maxx, maxy = shapely_geom.bounds
        center_lat = (miny + maxy) / 2
        parcel_span_lon = maxx - minx
        parcel_span_lat = maxy - miny
        parcel_span_km = max(parcel_span_lon, parcel_span_lat) * 111.0  # rough degrees→km
        adaptive_buffer_km = max(buffer_km, parcel_span_km * 0.20)

        delta_lon, delta_lat = self._km_to_degree_deltas(adaptive_buffer_km, latitude=center_lat)
        
        return shapely_geom, [minx - delta_lon, miny - delta_lat, maxx + delta_lon, maxy + delta_lat]
    def _create_base_map(self, bounds: List[float], padding: int = 0) -> folium.Map:
        minx, miny, maxx, maxy = bounds
        center_lat = (miny + maxy) / 2
        center_lon = (minx + maxx) / 2

        m = folium.Map(
            location=[center_lat, center_lon],
            tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
            attr="Google Satellite",
            max_zoom=21,         
            zoom_control=False,
            attribution_control=False,
            control_scale=True,  
            no_touch=True,
            zoomSnap=0,      
            zoomDelta=0.1    
        )
        m.fit_bounds([[miny, minx], [maxy, maxx]], padding=(padding, padding))
        
        north_arrow_html = """
        <div style="position: absolute; bottom: 30px; left: 20px; z-index: 1000; text-align: center;">
            <div style="color: white; font-family: Arial, sans-serif; font-size: 18px; font-weight: bold; text-shadow: 1px 1px 3px black, -1px -1px 3px black, 1px -1px 3px black, -1px 1px 3px black; margin-bottom: 2px;">N</div>
            <svg width="22" height="35" viewBox="0 0 24 35" xmlns="http://www.w3.org/2000/svg">
                <polygon points="12,0 24,35 12,25 0,35" fill="rgba(255,255,255,0.95)" stroke="#333" stroke-width="1"/>
            </svg>
        </div>
        """

        m.get_root().html.add_child(folium.Element(north_arrow_html + """
            <style>
                /* Hide the first child which is the metric (m/km) scale */
                .leaflet-control-scale-line:first-child { display: none !important; border-bottom: none !important; }
                
                /* Ensure the imperial (ft/mi) scale looks correct */
                .leaflet-control-scale-line:last-child { border-top: 2px solid white !important; }

                .leaflet-tile-pane { opacity: 0.8 !important; }
                /* ... [keep your existing CSS here] ... */
            </style>
        """))
        return m

    def _apply_parcel_style(self, m: folium.Map, geom: Any, darken_exterior: bool = False):
        if darken_exterior:
            world_bounds = Polygon([(-180, -90), (180, -90), (180, 90), (-180, 90), (-180, -90)])
            exterior_mask = world_bounds.difference(geom)
            folium.GeoJson(
                exterior_mask,
                style_function=lambda x: {'fillColor': '#000000', 'fillOpacity': 0.5, 'color': 'none', 'weight': 0}
            ).add_to(m)
            
        glow_levels = [(16, 0.08), (10, 0.15), (5, 0.25)]
        for weight, opacity in glow_levels:
            folium.GeoJson(geom, style_function=lambda x, w=weight, op=opacity: {
                'color': self.STYLE_COLOR, 'weight': w, 'opacity': op, 'fillOpacity': 0, 'lineCap': 'round', 'lineJoin': 'round'
            }).add_to(m)
            
        folium.GeoJson(geom, style_function=lambda x: {
            'color': self.STYLE_CASING, 'weight': 4, 'opacity': 0.9, 'fillOpacity': 0, 'lineCap': 'round', 'lineJoin': 'round'
        }).add_to(m)
        
        folium.GeoJson(geom, style_function=lambda x: {
            'color': self.STYLE_COLOR, 'weight': 2, 'fillColor': self.STYLE_COLOR, 'fillOpacity': 0, 'opacity': 1, 'lineCap': 'round', 'lineJoin': 'round'
        }).add_to(m)

    def _add_legend(self, m: folium.Map, items: List[str]):
        legend_content = "<br>".join(items)
        html = f"""
        <div style="position: fixed; bottom: 25px; right: 25px; background-color: rgba(0, 0, 0, 0.65); color: white;
            border: 1px solid rgba(255, 255, 255, 0.4); border-radius: 5px; padding: 12px; font-size: 14px; z-index: 1000;
            font-family: Arial, sans-serif; text-shadow: 1px 1px 2px black;">
            <strong>Map Legend</strong><br>{legend_content}
        </div>
        """
        m.get_root().html.add_child(folium.Element(html))
        m.get_root().html.add_child(folium.Element("<style>.leaflet-control-container .leaflet-control-attribution { display: none !important; }</style>"))

    async def _render_and_screenshot(self, m: folium.Map) -> BytesIO:
        html_content = m.get_root().render()
        fd, temp_path = tempfile.mkstemp(suffix=".html", dir="/tmp")
        chrome_temp_dirs = []
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            driver,chrome_temp_dirs = get_chrome_driver()
            try:
                await asyncio.to_thread(driver.set_window_size, 1600, 1200)
                await asyncio.to_thread(driver.get, f"file://{temp_path}")
                await asyncio.sleep(2.5) 
                png_bytes = await asyncio.to_thread(driver.get_screenshot_as_png)
                
                buf = BytesIO()
                Image.open(BytesIO(png_bytes)).convert("RGB").save(buf, format="PNG", optimize=True)
                buf.seek(0)
                return buf
            finally:
                await asyncio.to_thread(driver.quit)
                for d in chrome_temp_dirs:
                    try:
                        shutil.rmtree(d,ignore_errors=True)
                    except Exception as cleanup_error:
                        logger.warning(f"Failed to clean up Chrome temp directory {d}: {cleanup_error}")
        finally:
            for d in chrome_temp_dirs:
                try:
                    shutil.rmtree(d,ignore_errors=True)
                except Exception as cleanup_error:
                    logger.warning(f"Failed to clean up Chrome temp directory {d}: {cleanup_error}")
            if os.path.exists(temp_path):
                os.remove(temp_path)

    # ==========================================
    # ANALYSIS RENDERERS
    # ==========================================

    async def get_parcel_image(self, gid: int, geom_input: str, regenerate: bool = False):
        return await self._handle_cache_or_generate(gid, "aerial", self._gen_parcel, geom_input, regenerate)

    async def _gen_parcel(self, gid: int, geom_input: str) -> BytesIO:
        shapely_geom, bounds = self._get_geometry_and_bounds(geom_input, buffer_km=0.05)
        m = self._create_base_map(bounds, padding=40)
        self._apply_parcel_style(m, shapely_geom, darken_exterior=False)
        self._add_legend(m, [f"<span style='color:{self.STYLE_COLOR};'>▬</span> Property Boundary"])
        return await self._render_and_screenshot(m)

    async def get_road_frontage_image(self, gid: int, geom_input: str, features: list, regenerate: bool = False):
        return await self._handle_cache_or_generate(
            gid, 
            "road_frontage", 
            self._gen_road, 
            geom_input, 
            regenerate, 
            features=features
        )

    async def _gen_road(self, gid: int, geom_input: str, features: list) -> Union[BytesIO, Dict]:
        if isinstance(features, dict):
            ext_feats = features.get("_extended_features", [])
            ren_feats = features.get("_render_features", [])
            int_feats = features.get("interior_roads", {}).get("features", [])
        else:
            # Graceful fallback if the old flat list is passed
            ext_feats = features if isinstance(features, list) else []
            ren_feats = []
            int_feats = []

        if not (ext_feats or ren_feats or int_feats):
            return {"message": "No road frontage detected within bounds.", "status": "no_data"}

        shapely_geom, bounds = self._get_geometry_and_bounds(geom_input, buffer_km=0.15)
        m = self._create_base_map(bounds, padding=60)

        # 1. Target Parcel Styling (Custom Blue Override)
        folium.GeoJson(
            mapping(shapely_geom),
            name="Target Parcel",
            style_function=lambda x: {
                "fillColor": "#3388ff",
                "color": "#0044ff",
                "weight": 3,
                "fillOpacity": 0.15,
            },
        ).add_to(m)
        for feat in ext_feats:
            try:
                folium.GeoJson(
                    self._feature_geometry(feat),
                    name="Contributing Road",
                    style_function=lambda x: {"color": "#cc00cc", "weight": 4, "opacity": 0.75},
                ).add_to(m)
            except Exception as e:
                logger.warning(f"Skipping malformed contributing road: {e}")

        # 3. Frontage Highlight (Orange, thick)
        for feat in ren_feats:
            try:
                folium.GeoJson(
                    self._feature_geometry(feat),
                    name="Frontage Highlight",
                    style_function=lambda x: {"color": "#ffaa00", "weight": 8, "opacity": 0.95},
                ).add_to(m)
            except Exception as e:
                logger.warning(f"Skipping malformed frontage highlight: {e}")

        # 4. Interior Roads (Orange, dashed)
        for feat in int_feats:
            try:
                folium.GeoJson(
                    self._feature_geometry(feat),
                    name="Interior Road",
                    style_function=lambda x: {
                        "color": "#ff6600",
                        "weight": 4,
                        "opacity": 0.9,
                        "dashArray": "6, 6",
                    },
                ).add_to(m)
            except Exception as e:
                logger.warning(f"Skipping malformed interior road: {e}")
        self._add_legend(m, [
            "<span style='color:#0044ff; background-color:rgba(51, 136, 255, 0.15); border: 1px solid #0044ff; padding: 0 3px;'>▬</span> Target Parcel",
            "<span style='color:#ffaa00; font-weight:bold;'>▬</span> Frontage Highlight",
            "<span style='color:#cc00cc;'>▬</span> Contributing Roads",
            "<span style='color:#ff6600; border-bottom: 2px dashed #ff6600; height: 0; display: inline-block; width: 15px; margin-bottom: 3px;'></span> Interior Roads",
        ])

        return await self._render_and_screenshot(m)
    
    async def get_flood_image(self, gid: int, geom_input: str, features: list, regenerate: bool = False):
        return await self._handle_cache_or_generate(gid, "flood_hazard", self._gen_flood, geom_input, regenerate, features=features)

    async def _gen_flood(self, gid: int, geom_input: str, features: list) -> Union[BytesIO, Dict]:
        hazardous = [
            f for f in features
            if self._feature_properties(f).get("fld_zone") not in ["AREA NOT INCLUDED", "OPEN WATER"]
        ]
        if not hazardous:
            return {"message": "No flood hazards detected on property.", "status": "no_data"}

        shapely_geom, bounds = self._get_geometry_and_bounds(geom_input, buffer_km=0.04)
        m = self._create_base_map(bounds)

        sld_mapping = {
            'A':  {'fill': '#42A5F5', 'op': 0.35, 'stroke': '#1565C0', 'w': 1.5},
            'AE': {'fill': '#0D47A1', 'op': 0.45, 'stroke': '#002171', 'w': 1.5},
            'V':  {'fill': '#E53935', 'op': 0.35, 'stroke': '#B71C1C', 'w': 1.5},
            'VE': {'fill': '#E53935', 'op': 0.35, 'stroke': '#B71C1C', 'w': 1.5},
            'AH': {'fill': '#AB47BC', 'op': 0.35, 'stroke': '#6A1B9A', 'w': 1.5},
            'AO': {'fill': '#66BB6A', 'op': 0.35, 'stroke': '#2E7D32', 'w': 1.5},
            'X':  {'fill': '#EEEEEE', 'op': 0.20, 'stroke': '#9E9E9E', 'w': 0.5},
        }

        for feat in hazardous:
            props = self._feature_properties(feat)
            zone = props.get("fld_zone", "X")
            style = sld_mapping.get(zone, {'fill': '#808080', 'op': 0.3, 'stroke': '#333333', 'w': 1})
            poly_geom = self._feature_geometry(feat)

            folium.GeoJson(
                poly_geom,
                style_function=lambda x, s=style: {'fillColor': s['fill'], 'color': s['stroke'], 'weight': s['w'], 'fillOpacity': s['op']}
            ).add_to(m)

            if zone in ['A', 'AE', 'V', 'VE']:
                centroid = poly_geom.centroid
                folium.map.Marker(
                    [centroid.y, centroid.x],
                    icon=folium.DivIcon(html=f"""<div style="font-family: DejaVu Sans; font-size: 11pt; color: white; font-weight: bold; text-shadow: -1px -1px 0 {style['stroke']}, 1px -1px 0 {style['stroke']}, -1px 1px 0 {style['stroke']}, 1px 1px 0 {style['stroke']};">{zone}</div>""")
                ).add_to(m)

        self._apply_parcel_style(m, shapely_geom, darken_exterior=False)
        self._add_legend(m, [
            f"<span style='color:{self.STYLE_COLOR};'>▬</span> Property Boundary", 
            "<span style='color:#1565C0;'>■</span> High Risk (A/AE)",
            "<span style='color:#B71C1C;'>■</span> Coastal Risk (V/VE)",
            "<span style='color:#6A1B9A;'>■</span> Other Flood Areas (AH/AO)"
        ])
        return await self._render_and_screenshot(m)

    async def get_tree_image(self, gid: int, geom_input: str, regenerate: bool = False):
        return await self._handle_cache_or_generate(gid, "tree_coverage", self._gen_tree, geom_input, regenerate)

    async def _gen_tree(self, gid: int, geom_input: str) -> Union[BytesIO, Dict]:
        shapely_geom, bounds = self._get_geometry_and_bounds(geom_input, buffer_km=0.03)
        has_trees = False
        m = self._create_base_map(bounds)
        
        if os.path.exists(self.tree_path):
            try:
                gdf = gpd.GeoDataFrame(geometry=[shapely_geom], crs="EPSG:4326").to_crs(epsg=3857)
                proj_geom = gdf.geometry[0]
                bbox = proj_geom.bounds
                for f in os.listdir(self.tree_path):
                    if f.endswith(".tif"):
                        fpath = os.path.join(self.tree_path, f)
                        with rasterio.open(fpath) as src:
                            if not (src.bounds.right < bbox[0] or src.bounds.left > bbox[2] or src.bounds.bottom > bbox[3] or src.bounds.top < bbox[1]):
                                try:
                                    # Note: We now capture 'out_transform' to correctly map the pixels to real-world coordinates
                                    out_image, out_transform = mask(src, [mapping(proj_geom)], crop=True, nodata=0)
                                    data = out_image[0]
                                    
                                    if np.any(data > 0.1):
                                        has_trees = True
                                        
                                        # 1. Create a binary mask of where trees exist
                                        tree_mask = (data > 0.1).astype('uint8')
                                        
                                        # 2. Extract Vector Polygons from the Raster pixels
                                        shapes = rasterio.features.shapes(tree_mask, transform=out_transform)
                                        polygons = [shape(geom) for geom, val in shapes if val == 1]
                                        
                                        if polygons:
                                            # 3. Load into GeoPandas (using your projected CRS from earlier)
                                            tree_gdf = gpd.GeoDataFrame(geometry=polygons, crs="EPSG:3857")
                                            
                                            # 4. The Vector Smoothing Magic: 
                                            # Buffer outwards (5 meters) to round outer pixel corners, 
                                            # then buffer inwards (-5 meters) to return to original size while rounding inner corners.
                                            tree_gdf['geometry'] = tree_gdf.geometry.buffer(5).buffer(-5)
                                            
                                            # 5. Convert to lat/long for Folium
                                            tree_gdf = tree_gdf.to_crs(epsg=4326)
                                            
                                            # 6. Add to map as a Vector GeoJson Layer (resolution independent)
                                            folium.GeoJson(
                                                tree_gdf,
                                                style_function=lambda feature: {
                                                    'fillColor': '#1B5E20',  # Dark green fill
                                                    'color': '#7CB342',      # Lighter green border
                                                    'weight': 1.5,           # Border thickness
                                                    'fillOpacity': 0.85,
                                                    'smoothFactor': 1.0      # Tells Leaflet to smooth the vector lines
                                                }
                                            ).add_to(m)
                                            break
                                except Exception: continue
            except Exception: pass

        if not has_trees:
            return {"message": "No significant tree coverage detected.", "status": "no_data"}

        self._apply_parcel_style(m, shapely_geom, darken_exterior=False)
        
        # 5. Legend Update
        self._add_legend(m, [
            f"<span style='color:{self.STYLE_COLOR};'>▬</span> Property Boundary", 
            "<span style='background: linear-gradient(to right, #7CB342, #1B5E20); -webkit-background-clip: text; color: transparent;'>■</span> Tree Coverage Density"
        ])
        
        return await self._render_and_screenshot(m)

    async def get_contour_image(self, gid: int, geom_input: str, features: list, regenerate: bool = False):
        return await self._handle_cache_or_generate(gid, "contour", self._gen_contour, geom_input, regenerate, features=features)

    async def _gen_contour(self, gid: int, geom_input: str, features: list) -> Union[BytesIO, Dict]:
        if not features:
            return {"message": "No contour lines detected.", "status": "no_data"}

        shapely_geom, bounds = self._get_geometry_and_bounds(geom_input, buffer_km=0.03)
        m = self._create_base_map(bounds)
        for feat in features:
            c_geom = self._feature_geometry(feat)
            props = self._feature_properties(feat)
            elevation = props.get("elevation")
            folium.GeoJson(c_geom, style_function=lambda x: {'color': '#FFFF00', 'weight': 3, 'opacity': 0.8}).add_to(m)
            
            if elevation is not None:
                mid_point = c_geom.interpolate(0.5, normalized=True)
                folium.map.Marker(
                    [mid_point.y, mid_point.x],
                    icon=folium.DivIcon(html=f"""<div style="font-family: Arial, sans-serif; font-size: 8pt; color: #FFFFFF; font-weight: bold; white-space: nowrap; text-shadow: -1.5px -1.5px 0 #4E342E, 1.5px -1.5px 0 #4E342E, -1.5px 1.5px 0 #4E342E, 1.5px 1.5px 0 #4E342E;">{int(elevation)}</div>""")
                ).add_to(m)

        self._apply_parcel_style(m, shapely_geom, darken_exterior=False)
        self._add_legend(m, [
            f"<span style='color:{self.STYLE_COLOR};'>▬</span> Property Boundary", 
            "<span style='color:#FFFF00; background-color:#4E342E; padding: 0 2px;'>▬</span> Elevation Contour"
        ])
        return await self._render_and_screenshot(m)

    async def get_water_image(self, gid: int, geom_input: str, features: list, regenerate: bool = False):
        return await self._handle_cache_or_generate(gid, "water_features", self._gen_water, geom_input, regenerate, features=features)

    async def _gen_water(self, gid: int, geom_input: str, features: list) -> Union[BytesIO, Dict]:
        if not features:
            return {"message": "No major water features detected.", "status": "no_data"}
            
        shapely_geom, bounds = self._get_geometry_and_bounds(geom_input, buffer_km=0.024)
        m = self._create_base_map(bounds)
        
        WATER_FILL = '#1E88E5'
        WATER_STROKE = '#0D47A1'
        WATER_LINE = '#1976D2'
        WETLAND_FILL = '#4DB6AC'
        WETLAND_STROKE = '#00695C'
        WATER_OPACITY = 0.5

        has_streams = False
        has_polygons = False
        has_wetlands = False

        for feat in features:
            feat_geom = self._feature_geometry(feat)
            props = self._feature_properties(feat)
            f_type = (props.get("type") or props.get("strm_type") or "").lower()
            w_type = (props.get("wetland_type") or "").lower()

            # Render linear features (streams, creeks, rivers) and riverine
            if f_type in {"stream", "river", "creek"} or w_type == "riverine":
                has_streams = True
                folium.GeoJson(feat_geom, style_function=lambda x: {'color': WATER_LINE, 'weight': 2.5, 'lineCap': 'round', 'lineJoin': 'round', 'opacity': 1.0}).add_to(m)
                name = props.get("name")
                if name:
                    mid = feat_geom.interpolate(0.5, normalized=True) if feat_geom.geom_type in ['LineString', 'MultiLineString'] else feat_geom.centroid
                    folium.map.Marker([mid.y, mid.x], icon=folium.DivIcon(html=f"""<div style="font-family: Arial, sans-serif; font-size: 11pt; color: white; font-style: italic; font-weight: bold; white-space: nowrap; text-shadow: -2px -2px 0 {WATER_STROKE}, 2px -2px 0 {WATER_STROKE}, -2px 2px 0 {WATER_STROKE}, 2px 2px 0 {WATER_STROKE};">{name}</div>""")).add_to(m)

            # Render standard wetlands
            elif f_type == "wetland":
                has_wetlands = True
                folium.GeoJson(feat_geom, style_function=lambda x: {'fillColor': WETLAND_FILL, 'color': WETLAND_STROKE, 'fillOpacity': 0.4, 'weight': 1.5}).add_to(m)

            # Render polygonal water features (ponds, lakes, sea)
            else:
                has_polygons = True
                folium.GeoJson(feat_geom, style_function=lambda x: {'fillColor': WATER_FILL, 'color': WATER_STROKE, 'fillOpacity': WATER_OPACITY, 'weight': 1.5}).add_to(m)

        self._apply_parcel_style(m, shapely_geom, darken_exterior=False)
        
        legend_items = [f"<span style='color:{self.STYLE_COLOR};'>▬</span> Property Boundary"]
        if has_streams:
            legend_items.append(f"<span style='color:{WATER_LINE};'>▬</span> Rivers/Creeks")
        if has_polygons:
            legend_items.append(f"<span style='color:{WATER_FILL};'>■</span> Ponds/Lakes/Open Water")
        if has_wetlands:
            legend_items.append(f"<span style='color:{WETLAND_FILL};'>■</span> Wetlands")

        self._add_legend(m, legend_items)
        return await self._render_and_screenshot(m)

    async def get_pipeline_image(self, gid: int, geom_input: str, features: list, regenerate: bool = False):
        return await self._handle_cache_or_generate(gid, "gas_pipelines", self._gen_pipeline, geom_input, regenerate, features=features)

    async def _gen_pipeline(self, gid: int, geom_input: str, features: list) -> Union[BytesIO, Dict]:
        if not features:
            return {"message": "No gas pipelines detected.", "status": "no_data"}

        shapely_geom, bounds = self._get_geometry_and_bounds(geom_input, buffer_km=0.034)
        m = self._create_base_map(bounds)
        
        for feat in features:
            pipeline_geom = self._feature_geometry(feat)
            folium.GeoJson(pipeline_geom, style_function=lambda x: {'color': '#FDD835', 'weight': 4, 'opacity': 1.0, 'lineCap': 'round', 'lineJoin': 'round'}).add_to(m)
            folium.GeoJson(pipeline_geom, style_function=lambda x: {'color': '#FF6D00', 'weight': 2, 'dashArray': '12, 6', 'opacity': 1.0, 'lineCap': 'round', 'lineJoin': 'round'}).add_to(m)
        
        self._apply_parcel_style(m, shapely_geom, darken_exterior=False)
        self._add_legend(m, [
            f"<span style='color:{self.STYLE_COLOR};'>▬</span> Property Boundary", 
            "<span style='color:#FDD835; text-shadow: 0 0 2px #FF6D00; font-weight: bold;'>▬ ▬</span> Industrial Gas Pipeline"
        ])
        return await self._render_and_screenshot(m)

    async def get_gas_transmission_image(self, gid: int, geom_input: str, features: list, regenerate: bool = False):
        return await self._handle_cache_or_generate(gid, "gas_transmission_lines", self._gen_gas_and_transmission, geom_input, regenerate, features=features)

    async def get_gas_and_transmission_image(self, gid: int, geom_input: str, features: list, regenerate: bool = False):
        # Backward-compatible alias for older callers.
        return await self.get_gas_transmission_image(gid, geom_input, features, regenerate)

    async def _gen_gas_and_transmission(self, gid: int, geom_input: str, features: list) -> Union[BytesIO, Dict]:
        if not features:
            return {"message": "No utility lines detected.", "status": "no_data"}

        shapely_geom, bounds = self._get_geometry_and_bounds(geom_input, buffer_km=0.03)
        m = self._create_base_map(bounds)
        
        has_gas = False
        has_electric = False
        has_any_line = False

        for feat in features:
            try:
                feat_geom = self._feature_geometry(feat)
            except Exception:
                continue

            props = self._feature_properties(feat)
            classification = self._classify_utility_feature(props)

            if classification == "electric":
                has_electric = True
                has_any_line = True
                folium.GeoJson(feat_geom, style_function=lambda x: {'color': '#E040FB', 'weight': 4, 'opacity': 0.8, 'lineCap': 'round'}).add_to(m)
                folium.GeoJson(feat_geom, style_function=lambda x: {'color': '#FFFFFF', 'weight': 1, 'dashArray': '4, 8', 'opacity': 0.9}).add_to(m)

            elif classification == "gas":
                has_gas = True
                has_any_line = True
                folium.GeoJson(feat_geom, style_function=lambda x: {'color': '#FDD835', 'weight': 4, 'opacity': 1.0, 'lineCap': 'round'}).add_to(m)
                folium.GeoJson(feat_geom, style_function=lambda x: {'color': '#FF6D00', 'weight': 2, 'dashArray': '12, 6', 'opacity': 1.0}).add_to(m)

            else:
                has_any_line = True
                folium.GeoJson(feat_geom, style_function=lambda x: {'color': '#F2F2F2', 'weight': 3, 'opacity': 0.9, 'lineCap': 'round'}).add_to(m)

        if not has_any_line:
            return {"message": "No utility lines detected.", "status": "no_data"}

        self._apply_parcel_style(m, shapely_geom, darken_exterior=False)
        
        legend_items = [f"<span style='color:{self.STYLE_COLOR};'>▬</span> Property Boundary"]
        if has_gas:
            legend_items.append("<span style='color:#FDD835; text-shadow: 0 0 2px #FF6D00; font-weight: bold;'>▬ ▬</span> Gas/Oil Pipeline")
        if has_electric:
            legend_items.append("<span style='color:#E040FB; font-weight: bold;'>▬ ▬</span> Electric Transmission")

        self._add_legend(m, legend_items)
        return await self._render_and_screenshot(m)

    async def get_well_image(self, gid: int, geom_input: str, features: list, regenerate: bool = False):
        return await self._handle_cache_or_generate(gid, "water_wells", self._gen_well, geom_input, regenerate, features=features)

    async def _gen_well(self, gid: int, geom_input: str, features: list) -> Union[BytesIO, Dict]:
        if not features:
            return {"message": "No water wells detected on property.", "status": "no_data"}
        
        shapely_geom, bounds = self._get_geometry_and_bounds(geom_input, buffer_km=0.03)
        m = self._create_base_map(bounds)

        pin_svg = """<svg viewBox="0 0 384 512" width="28" height="40" style="filter: drop-shadow(2px 2px 3px rgba(0,0,0,0.6));" xmlns="http://www.w3.org/2000/svg">
            <path fill="#cb2b27" stroke="white" stroke-width="15" d="M172.268 501.67C26.97 291.031 0 269.413 0 192 0 85.961 85.961 0 192 0s192 85.961 192 192c0 77.413-26.97 99.031-172.268 309.67-9.535 13.774-29.93 13.773-39.464 0zM192 272c44.183 0 80-35.817 80-80s-35.817-80-80-80-80 35.817-80 80 35.817 80 80 80z"/>
        </svg>"""

        for feat in features:
            point_geom = self._feature_geometry(feat)
            folium.Marker(
                location=[point_geom.y, point_geom.x],
                icon=folium.DivIcon(
                    icon_size=(28, 40),
                    icon_anchor=(14, 40),   # tip of the pin touches the exact coordinate
                    html=pin_svg,
                ),
                tooltip="Water Well",
            ).add_to(m)

        self._apply_parcel_style(m, shapely_geom, darken_exterior=False)
        self._add_legend(m, [
            f"<span style='color:{self.STYLE_COLOR};'>▬</span> Property Boundary",
            "<svg width='14' height='20' viewBox='0 0 384 512' xmlns='http://www.w3.org/2000/svg'>"
            "<path fill='#cb2b27' stroke='white' stroke-width='15' d='M172.268 501.67C26.97 291.031 0 269.413 0 192 0 85.961 85.961 0 192 0s192 85.961 192 192c0 77.413-26.97 99.031-172.268 309.67-9.535 13.774-29.93 13.773-39.464 0zM192 272c44.183 0 80-35.817 80-80s-35.817-80-80-80-80 35.817-80 80 35.817 80 80 80z'/>"
            "</svg> Water Well",
        ])
        return await self._render_and_screenshot(m)

    async def get_electric_image(self, gid: int, geom_input: str, features: list, regenerate: bool = False):
        return await self._handle_cache_or_generate(gid, "electric_lines", self._gen_electric, geom_input, regenerate, features=features)

    async def _gen_electric(self, gid: int, geom_input: str, features: list) -> Union[BytesIO, Dict]:
        if not features:
            return {"message": "No electric transmission lines detected.", "status": "no_data"}

        shapely_geom, bounds = self._get_geometry_and_bounds(geom_input, buffer_km=0.03)
        m = self._create_base_map(bounds)

        sld_styles = {
            "AC; UNDERGROUND": {"color": "#FF0033", "weight": 3.5, "dash": "12, 6"},
            "AC; OVERHEAD": {"color": "#FF0033", "weight": 3.5, "dash": None},
            "OVERHEAD": {"color": "#FF5A5F", "weight": 3, "dash": None},
            "NOT AVAILABLE": {"color": "#B0BEC5", "weight": 2.5, "dash": None},
        }

        for feat in features:
            line_geom = self._feature_geometry(feat)
            props = self._feature_properties(feat)
            line_type = props.get("TYPE") or props.get("type") or "OVERHEAD"
            voltage = props.get("VOLTAGE") or props.get("voltage")
            style = sld_styles.get(line_type, {"color": "#FF5A5F", "weight": 2, "dash": None})

            folium.GeoJson(
                line_geom,
                style_function=lambda x, s=style: {
                    "color": s["color"],
                    "weight": s["weight"],
                    "dashArray": s["dash"],
                    "lineCap": "round",
                    "lineJoin": "round",
                },
            ).add_to(m)

            if voltage and str(voltage) != "999999":
                label_text = str(voltage)
                if not label_text.lower().endswith("kv"):
                    label_text = f"{label_text} kV"

                mid_point = line_geom.interpolate(0.5, normalized=True)
                folium.map.Marker(
                    [mid_point.y, mid_point.x],
                    icon=folium.DivIcon(
                        html=f"""<div style="font-family: DejaVu Sans, sans-serif; font-size: 10pt; color: white; font-weight: bold; white-space: nowrap; text-shadow: -2px -2px 0 #212121, 2px -2px 0 #212121, -2px 2px 0 #212121, 2px 2px 0 #212121;">{label_text}</div>"""
                    ),
                ).add_to(m)

        self._apply_parcel_style(m, shapely_geom, darken_exterior=False)
        self._add_legend(m, [
            f"<span style='color:{self.STYLE_COLOR};'>▬</span> Property Boundary",
            "<span style='color:#FF0033;'>▬</span> Transmission Line (AC)",
            "<span style='color:#FF0033; border-bottom: 2px dashed #FF0033; height: 0; display: inline-block; width: 15px; margin-bottom: 3px;'></span> Underground Line",
        ])
        return await self._render_and_screenshot(m)

    async def get_transmission_image(self, gid: int, geom_input: str, features: list, regenerate: bool = False):
        return await self._handle_cache_or_generate(gid, "transmission_lines", self._gen_transmission, geom_input, regenerate, features=features)

    async def _gen_transmission(self, gid: int, geom_input: str, features: list) -> Union[BytesIO, Dict]:
        if not features:
            return {"message": "No transmission lines detected.", "status": "no_data"}

        shapely_geom, bounds = self._get_geometry_and_bounds(geom_input, buffer_km=0.03)
        m = self._create_base_map(bounds)

        for feat in features:
            line_geom = self._feature_geometry(feat)
            folium.GeoJson(
                line_geom,
                style_function=lambda x: {
                    "color": "#FF0033",
                    "weight": 3,
                    "lineCap": "round",
                    "lineJoin": "round",
                },
            ).add_to(m)

        self._apply_parcel_style(m, shapely_geom, darken_exterior=False)
        self._add_legend(m, [
            f"<span style='color:{self.STYLE_COLOR};'>▬</span> Property Boundary",
            "<span style='color:#FF0033;'>▬</span> Transmission Line",
        ])
        return await self._render_and_screenshot(m)

    async def get_ponds_creeks_image(self, gid: int, geom_input: str, features: list, regenerate: bool = False):
        return await self._handle_cache_or_generate(gid, "ponds_creeks", self._gen_ponds_creeks, geom_input, regenerate, features=features)

    async def _gen_ponds_creeks(self, gid: int, geom_input: str, features: list) -> Union[BytesIO, Dict]:
        if not features:
            return {"message": "No ponds or creeks detected.", "status": "no_data"}

        shapely_geom, bounds = self._get_geometry_and_bounds(geom_input, buffer_km=0.03)
        m = self._create_base_map(bounds)

        has_streams = False
        has_ponds = False
        WATER_FILL = '#1E88E5'
        WATER_STROKE = '#0D47A1'
        WATER_LINE = '#1976D2'
        for feat in features:
            feat_geom = self._feature_geometry(feat)
            props = self._feature_properties(feat)
            feat_type = (props.get("type") or "").lower()
            name = props.get("name") or props.get("strm_nm")

            if feat_type in {"stream", "creek", "river"}:
                has_streams = True
                folium.GeoJson(
                    feat_geom,
                    style_function=lambda x: {
                        "color": WATER_LINE,
                        "weight": 2.5,
                        "lineCap": "round",
                        "lineJoin": "round",
                        "opacity": 1.0,
                    },
                ).add_to(m)
                if name:
                    centroid = feat_geom.centroid
                    folium.Marker(
                        location=[centroid.y, centroid.x],
                        icon=folium.DivIcon(
                            html=f"""<div style="font-family: DejaVu Sans, sans-serif; font-size: 11px; font-style: italic; font-weight: bold; color: #FFFFFF; text-shadow: 2px 2px 0px #01579B, -2px -2px 0px #01579B, 2px -2px 0px #01579B, -2px 2px 0px #01579B; white-space: nowrap;">{name}</div>"""
                        ),
                    ).add_to(m)
            else:
                has_ponds = True
                folium.GeoJson(
                    feat_geom,
                    style_function=lambda x: {
                        "fillColor": WATER_FILL,
                        "color": WATER_STROKE,
                        "weight": 1,
                        "fillOpacity": 0.6,
                    },
                ).add_to(m)

        if not has_streams and not has_ponds:
            return {"message": "No ponds or creeks detected.", "status": "no_data"}

        self._apply_parcel_style(m, shapely_geom, darken_exterior=False)
        legend_items = [f"<span style='color:{self.STYLE_COLOR};'>▬</span> Property Boundary"]
        if has_streams:
            legend_items.append(f"<span style='color:{WATER_LINE}; font-weight:bold;'>▬</span> Rivers/Creeks")
        if has_ponds:
            legend_items.append(f"<span style='display:inline-block; width:12px; height:12px; background:{WATER_FILL}; border:1px solid {WATER_STROKE};'></span> Ponds/Lakes/Wetlands")
        self._add_legend(m, legend_items)
        return await self._render_and_screenshot(m)

    async def get_county_image(self, gid: int, geom_input: str, overlay_features: list, regenerate: bool = False):
        return await self._handle_cache_or_generate(gid, "county_boundary", self._gen_county, geom_input, regenerate, features=overlay_features)

    async def _gen_county(self, gid: int, geom_input: str, features: list) -> Union[BytesIO, Dict]:
        shapely_geom, bounds = self._get_geometry_and_bounds(geom_input, buffer_km=32)
        m = self._create_base_map(bounds)
        
        legend_items = [f"<span style='color:{self.STYLE_COLOR};'>📍</span> Property Location"]
        has_county = any(self._feature_properties(f).get("type") == "county" for f in features)
        has_city = any(self._feature_properties(f).get("type") == "city" for f in features)
        
        legend_items.append("<span style='color: #fc03e3;'>▬</span> County Boundary" if has_county else "<span style='color:gray;'>ℹ</span> No County Boundary within 20 miles")
        if has_city: legend_items.append("<span style='color:#03fcf0;'>▬</span> City Limits")
        else: legend_items.append("<span style='color:gray;'>ℹ</span> No Cities within 20 miles")

        for feat in features:
            props = self._feature_properties(feat)
            feat_type = props.get("type")
            name = props.get("name", "")
            feat_geom = self._feature_geometry(feat)
            
            if feat_type == "county":
                folium.GeoJson(feat_geom, style_function=lambda x: {'color': '#fc03e3', 'weight': 2.5, 'opacity': 0.7, 'fillOpacity': 0}).add_to(m)
                if name:
                    centroid = feat_geom.centroid
                    folium.Marker(location=[centroid.y, centroid.x], icon=folium.DivIcon(icon_size=(200, 36), icon_anchor=(100, 18), html=f'<div style="font-size: 20pt; font-weight: bold; color: black; text-shadow: 3px 3px 4px white, -3px -3px 4px white, 3px -3px 4px white, -3px 3px 4px white, 0px 3px 4px white, 0px -3px 4px white, 3px 0px 4px white, -3px 0px 4px white; text-align: center; opacity: 0.9;">{name.upper()}</div>')).add_to(m)
                    
            elif feat_type == "city":
                folium.GeoJson(feat_geom, style_function=lambda x: {'color': '#03fcf0', 'weight': 2, 'opacity': 0.6, 'fillOpacity': 0.05, 'fillColor': '#03fcf0'}).add_to(m)
                if name:
                    centroid = feat_geom.centroid
                    folium.Marker(location=[centroid.y, centroid.x], icon=folium.DivIcon(icon_size=(200, 50), icon_anchor=(80, 25), html=f'<div style="font-size: 16pt; font-weight: bold; color: black; text-shadow: 3px 3px 4px white, -3px -3px 4px white, 3px -3px 4px white, -3px 3px 4px white, 0px 3px 4px white, 0px -3px 4px white, 3px 0px 4px white, -3px 0px 4px white; text-align: center;margin-top: 40px;">{name}</div>')).add_to(m)

        prop_centroid = shapely_geom.centroid
        pin_svg = """<svg viewBox="0 0 384 512" width="35" height="50" style="filter: drop-shadow(3px 3px 3px rgba(0,0,0,0.6));" xmlns="http://www.w3.org/2000/svg"><path fill="#cb2b27" stroke="white" stroke-width="15" d="M172.268 501.67C26.97 291.031 0 269.413 0 192 0 85.961 85.961 0 192 0s192 85.961 192 192c0 77.413-26.97 99.031-172.268 309.67-9.535 13.774-29.93 13.773-39.464 0zM192 272c44.183 0 80-35.817 80-80s-35.817-80-80-80-80 35.817-80 80 35.817 80 80 80z"/></svg>"""
        folium.Marker(location=[prop_centroid.y, prop_centroid.x], icon=folium.DivIcon(icon_size=(45, 60), icon_anchor=(22, 60), html=pin_svg)).add_to(m)
        
        self._add_legend(m, legend_items)
        return await self._render_and_screenshot(m)