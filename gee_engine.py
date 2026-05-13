import ee
import os
from datetime import date, timedelta
from dotenv import load_dotenv

load_dotenv()

_EE_INITIALIZED = False

def initialize_ee():
    """Initialize Earth Engine safely (works locally + avoids permission issues)."""
    global _EE_INITIALIZED

    if _EE_INITIALIZED:
        return

    try:
        # ✅ Use default authentication 
        ee.Initialize(project=os.getenv("GEE_PROJECT"))
        _EE_INITIALIZED = True

    except ee.EEException as exc:
        raise RuntimeError(
            "Failed to initialize Google Earth Engine. "
            "Run 'earthengine authenticate' and try again."
        ) from exc

def _validate_coordinates(lat, lon):
    if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
        raise TypeError('Latitude and longitude must be numeric.')
    if not -90 <= lat <= 90:
        raise ValueError('Latitude must be between -90 and 90.')
    if not -180 <= lon <= 180:
        raise ValueError('Longitude must be between -180 and 180.')


def analyze_location(
    lat,
    lon,
    start_date=None,
    end_date=None,
    cloud_threshold=20,
    buffer_meters=50,
):
    """
    Takes a latitude and longitude,
    returns groundwater analysis results
    """

    initialize_ee()
    _validate_coordinates(lat, lon)

# DYNAMIC DRY SEASON LOGIC HERE:
    if start_date is None or end_date is None:
        today = date.today()
        
        # Scenario 1: We are IN the dry season right now (Feb to May)
        if 2 <= today.month <= 5:
            start_date = f"{today.year}-02-01"
            end_date = today.isoformat()  # Pulls data right up to today!
            
        # Scenario 2: It's January (Dry season hasn't started yet)
        elif today.month == 1:
            start_date = f"{today.year - 1}-02-01"
            end_date = f"{today.year - 1}-05-31"
            
        # Scenario 3: It's post-monsoon (June to December)
        else:
            start_date = f"{today.year}-02-01"
            end_date = f"{today.year}-05-31"

    # Create a point from farmer's location
    point = ee.Geometry.Point([lon, lat])

    # Use the requested buffer consistently for filtering and sampling.
    region = point.buffer(buffer_meters).bounds()

    # -----------------------------------------------
    # STEP 1: Get Sentinel-2 Satellite Image
    # -----------------------------------------------
    collection = (
        ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(region)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_threshold))
    )

    # -----------------------------------------------
    # STEP 2: Cloud Masking (removes bad pixels)
    # -----------------------------------------------
    def mask_clouds(img):
        scl = img.select('SCL')  # Scene Classification Layer
        # Keep only vegetation, bare soil, water and unclassified pixels.
        allowed = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(7))
        mask = allowed
        return img.updateMask(mask)

    image_count = collection.size().getInfo()
    if image_count == 0:
        raise ValueError(
            f'No Sentinel-2 images found for {start_date} to {end_date} '
            f'with cloud threshold {cloud_threshold}%.'
        )

    image = collection.map(mask_clouds).median()

    # -----------------------------------------------
    # STEP 3: Compute Spectral Indices
    # -----------------------------------------------

    # NDVI - how green/vegetated the land is
    # High NDVI = good vegetation = possible water nearby
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')

    # NDWI - detects surface water presence
    # High NDWI = more water content
    ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')

    # NDMI - detects soil/vegetation moisture
    # High NDMI = moist soil = possible groundwater
    ndmi = image.normalizedDifference(['B8', 'B11']).rename('NDMI')

    # -----------------------------------------------
    # STEP 4: Get Elevation and Slope
    # -----------------------------------------------

    # SRTM = NASA elevation data (free)
    dem = ee.Image('USGS/SRTMGL1_003')

    # Elevation at this location (in meters)
    elevation = dem.select('elevation')

    # Slope - groundwater collects in flat/low slope areas
    slope = ee.Terrain.slope(dem)

    # -----------------------------------------------
    # STEP 5: Extract actual values at farmer's point
    # -----------------------------------------------
    combined = ndvi.addBands(ndwi).addBands(ndmi)\
                   .addBands(elevation).addBands(slope)

    values = combined.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=point.buffer(buffer_meters),
        scale=10,
        bestEffort=True,
        maxPixels=1e8,
    ).getInfo()

    return values
