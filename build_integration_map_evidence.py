from __future__ import annotations

import json
import math
import tempfile
from gzip import open as gzip_open
from datetime import date, datetime, timezone
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import rasterio
from netCDF4 import Dataset, num2date
from PIL import Image, ImageDraw, ImageFont
from rasterio.windows import from_bounds


BOUNDARIES = Path("data/boundaries/uganda_districts_prepared.geojson")
CHIRPS = Path("data/climate/raw/chirps/2025/chirps-v2.0.2025.05.12.tif.gz")
ERA5 = Path("data/climate/raw/era5_land/temperature/era5_land_temperature_2025_05.nc")
OUTPUT_DIR = Path("docs/presentation/integration-map-evidence")
DISTRICT_ID = "UGA_KAPCHORWA"
DISPLAY_DATE = date(2025, 5, 12)

NAVY = (18, 53, 91)
BLUE = (23, 105, 170)
TEAL = (8, 127, 122)
GREEN = (20, 122, 85)
AMBER = (183, 121, 31)
MAGENTA = (203, 24, 99)
INK = (20, 32, 51)
MUTED = (89, 105, 130)
PALE = (245, 248, 252)
GRID = (210, 221, 234)


def font(size: int, bold: bool = False):
    candidates = [
        Path("C:/Windows/Fonts/aptosbd.ttf" if bold else "C:/Windows/Fonts/aptos.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def load_features() -> list[dict]:
    data = json.loads(BOUNDARIES.read_text(encoding="utf-8"))
    return data["features"]


def geometry_rings(geometry: dict) -> list[list[list[float]]]:
    if geometry["type"] == "Polygon":
        return [geometry["coordinates"][0]]
    if geometry["type"] == "MultiPolygon":
        return [polygon[0] for polygon in geometry["coordinates"]]
    raise ValueError(f"Unexpected geometry type: {geometry['type']}")


def district_feature(features: list[dict]) -> dict:
    return next(
        feature
        for feature in features
        if feature["properties"].get("district_id") == DISTRICT_ID
    )


def bounds(feature: dict) -> tuple[float, float, float, float]:
    points = [point for ring in geometry_rings(feature["geometry"]) for point in ring]
    longitudes = [point[0] for point in points]
    latitudes = [point[1] for point in points]
    return min(longitudes), min(latitudes), max(longitudes), max(latitudes)


def project(lon, lat, extent, box):
    west, south, east, north = extent
    x0, y0, x1, y1 = box
    x = x0 + ((lon - west) / (east - west)) * (x1 - x0)
    y = y1 - ((lat - south) / (north - south)) * (y1 - y0)
    return round(x), round(y)


def draw_header(image: Image.Image, title: str, subtitle: str):
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 104), fill=NAVY)
    draw.rectangle((0, 104, image.width, 112), fill=TEAL)
    draw.text((44, 24), title, fill="white", font=font(31, True))
    draw.text((46, 68), subtitle, fill=(225, 240, 252), font=font(16))


def color_scale(values: np.ndarray, low: tuple[int, int, int], high: tuple[int, int, int]):
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        raise ValueError("No finite climate values in selected map extent.")
    lower = float(np.quantile(valid, 0.05))
    upper = float(np.quantile(valid, 0.95))
    if math.isclose(lower, upper):
        upper = lower + 1.0
    scaled = np.clip((values - lower) / (upper - lower), 0, 1)
    output = np.zeros((*values.shape, 3), dtype=np.uint8)
    for channel in range(3):
        output[..., channel] = np.where(
            np.isfinite(values),
            np.round(low[channel] + scaled * (high[channel] - low[channel])),
            224,
        )
    return output, lower, upper


def add_footer(draw, width: int, source: str):
    draw.text((44, 936), source, fill=MUTED, font=font(13))
    draw.text(
        (44, 964),
        "Boundary outline: prepared GeoJSON | Magenta outline: UGA_KAPCHORWA | Climate grid values are shown only to demonstrate spatial overlap.",
        fill=MUTED,
        font=font(12),
    )


def create_uganda_location_map(features: list[dict], kapchorwa: dict):
    image = Image.new("RGB", (1600, 1000), PALE)
    draw_header(
        image,
        "Kapchorwa location in the prepared Uganda district GeoJSON",
        "This map shows the district boundary used to link surveillance and climate data.",
    )
    draw = ImageDraw.Draw(image)
    all_points = [
        point
        for feature in features
        for ring in geometry_rings(feature["geometry"])
        for point in ring
    ]
    longs = [point[0] for point in all_points]
    lats = [point[1] for point in all_points]
    extent = (min(longs), min(lats), max(longs), max(lats))
    map_box = (275, 145, 1045, 895)
    draw.rounded_rectangle(map_box, radius=18, fill="white", outline=GRID, width=2)
    for feature in features:
        for ring in geometry_rings(feature["geometry"]):
            points = [project(lon, lat, extent, map_box) for lon, lat in ring]
            draw.polygon(points, fill=(231, 240, 250), outline=BLUE)
    for ring in geometry_rings(kapchorwa["geometry"]):
        points = [project(lon, lat, extent, map_box) for lon, lat in ring]
        draw.polygon(points, fill=(244, 202, 224), outline=MAGENTA)
        draw.line(points, fill=MAGENTA, width=5, joint="curve")
    west, south, east, north = bounds(kapchorwa)
    label_point = project((west + east) / 2, (south + north) / 2, extent, map_box)
    draw.ellipse((label_point[0] - 8, label_point[1] - 8, label_point[0] + 8, label_point[1] + 8), fill=MAGENTA)
    draw.line((label_point[0], label_point[1], 1180, 368), fill=MAGENTA, width=3)
    draw.rounded_rectangle((1080, 250, 1510, 492), radius=18, fill=(255, 247, 251), outline=MAGENTA, width=2)
    draw.text((1110, 280), "CROSS-REFERENCE DISTRICT", fill=MAGENTA, font=font(16, True))
    draw.text((1110, 325), "KAPCHORWA", fill=INK, font=font(27, True))
    draw.text((1110, 372), "district_id: UGA_KAPCHORWA", fill=INK, font=font(17, True))
    draw.text((1110, 410), f"longitude: {west:.6f} to {east:.6f}", fill=INK, font=font(15))
    draw.text((1110, 438), f"latitude: {south:.6f} to {north:.6f}", fill=INK, font=font(15))
    draw.text((1110, 466), "GeoJSON coordinates: longitude, latitude", fill=MUTED, font=font(13))
    add_footer(draw, image.width, "Source: data/boundaries/uganda_districts_prepared.geojson")
    image.save(OUTPUT_DIR / "01_kapchorwa_location_in_prepared_geojson.png")


def draw_grid_map(
    title: str,
    subtitle: str,
    values: np.ndarray,
    west: float,
    south: float,
    east: float,
    north: float,
    kapchorwa: dict,
    low_color: tuple[int, int, int],
    high_color: tuple[int, int, int],
    unit_label: str,
    source_text: str,
    output_name: str,
):
    image = Image.new("RGB", (1600, 1000), PALE)
    draw_header(image, title, subtitle)
    draw = ImageDraw.Draw(image)
    extent = (west, south, east, north)
    map_box = (90, 150, 1120, 890)
    draw.rounded_rectangle(map_box, radius=16, fill="white", outline=GRID, width=2)
    rgb_values, lower, upper = color_scale(values, low_color, high_color)
    flipped = np.flipud(rgb_values)
    raster = Image.fromarray(flipped, mode="RGB").resize((map_box[2] - map_box[0], map_box[3] - map_box[1]), Image.Resampling.NEAREST)
    image.paste(raster, (map_box[0], map_box[1]))
    grid_draw = ImageDraw.Draw(image)
    rows, cols = values.shape
    for row in range(rows + 1):
        y = map_box[1] + round((row / rows) * (map_box[3] - map_box[1]))
        grid_draw.line((map_box[0], y, map_box[2], y), fill=(255, 255, 255), width=1)
    for col in range(cols + 1):
        x = map_box[0] + round((col / cols) * (map_box[2] - map_box[0]))
        grid_draw.line((x, map_box[1], x, map_box[3]), fill=(255, 255, 255), width=1)
    for ring in geometry_rings(kapchorwa["geometry"]):
        points = [project(lon, lat, extent, map_box) for lon, lat in ring]
        grid_draw.line(points, fill=MAGENTA, width=6, joint="curve")
    kap_west, kap_south, kap_east, kap_north = bounds(kapchorwa)
    grid_draw.rounded_rectangle((1170, 164, 1535, 492), radius=18, fill="white", outline=GRID, width=2)
    grid_draw.text((1200, 196), "WHAT THIS PROVES", fill=NAVY, font=font(18, True))
    text = [
        "The magenta outline is the", "Kapchorwa GeoJSON boundary.", "", "The coloured rectangles are", "climate grid cells from the", "original source file.", "", "The code uses every cell", "inside the polygon, then", "stores one district-week value.", "", f"Displayed range: {lower:.2f} to {upper:.2f}", unit_label,
    ]
    y = 240
    for line in text:
        grid_draw.text((1200, y), line, fill=INK if line else MUTED, font=font(15, bool(line and line.startswith("The magenta"))))
        y += 23
    grid_draw.text((92, 902), f"Map extent: {west:.4f} to {east:.4f} E; {south:.4f} to {north:.4f} N", fill=MUTED, font=font(13))
    add_footer(grid_draw, image.width, source_text)
    image.save(OUTPUT_DIR / output_name)


def create_chirps_map(kapchorwa: dict):
    kap_west, kap_south, kap_east, kap_north = bounds(kapchorwa)
    padding = 0.15
    west, south, east, north = kap_west - padding, kap_south - padding, kap_east + padding, kap_north + padding
    with tempfile.TemporaryDirectory(prefix="chirps_map_") as temp_dir:
        tif_path = Path(temp_dir) / "chirps_daily.tif"
        with gzip_open(CHIRPS, "rb") as compressed, tif_path.open("wb") as extracted:
            extracted.write(compressed.read())
        with rasterio.open(tif_path) as dataset:
            window = from_bounds(west, south, east, north, dataset.transform).round_offsets().round_lengths()
            values = dataset.read(1, window=window, masked=True).filled(np.nan).astype("float64")
            transform = dataset.window_transform(window)
            actual_west = transform.c
            actual_north = transform.f
            actual_east = actual_west + transform.a * values.shape[1]
            actual_south = actual_north + transform.e * values.shape[0]
    draw_grid_map(
        "CHIRPS daily rainfall cells within Kapchorwa",
        "Daily CHIRPS file for 12 May 2025. The cells do not contain district names; the polygon gives them district meaning.",
        values,
        actual_west,
        actual_south,
        actual_east,
        actual_north,
        kapchorwa,
        (232, 246, 251),
        (8, 81, 156),
        "millimetres of rainfall on 12 May 2025",
        "Source file: data/climate/raw/chirps/2025/chirps-v2.0.2025.05.12.tif.gz",
        "02_kapchorwa_chirps_grid_overlay_2025-05-12.png",
    )


def create_era5_map(kapchorwa: dict):
    kap_west, kap_south, kap_east, kap_north = bounds(kapchorwa)
    with tempfile.TemporaryDirectory(prefix="era5_map_") as temp_dir:
        with ZipFile(ERA5) as archive:
            archive.extract("data_0.nc", temp_dir)
        with Dataset(Path(temp_dir) / "data_0.nc") as dataset:
            latitudes = np.asarray(dataset.variables["latitude"][:], dtype="float64")
            longitudes = np.asarray(dataset.variables["longitude"][:], dtype="float64")
            times = num2date(dataset.variables["valid_time"][:], dataset.variables["valid_time"].units)
            time_indices = [
                index
                for index, value in enumerate(times)
                if value.year == DISPLAY_DATE.year and value.month == DISPLAY_DATE.month and value.day == DISPLAY_DATE.day
            ]
            if not time_indices:
                raise ValueError(f"ERA5 file has no values for {DISPLAY_DATE.isoformat()}.")
            raw = np.asarray(dataset.variables["t2m"][time_indices, :, :], dtype="float64")
            daily_celsius = np.nanmean(raw, axis=0) - 273.15
    padding = 0.22
    row_indices = np.where((latitudes >= kap_south - padding) & (latitudes <= kap_north + padding))[0]
    col_indices = np.where((longitudes >= kap_west - padding) & (longitudes <= kap_east + padding))[0]
    selected_latitudes = latitudes[row_indices]
    selected_longitudes = longitudes[col_indices]
    values = daily_celsius[np.ix_(row_indices, col_indices)]
    latitude_step = abs(float(np.diff(latitudes).mean()))
    longitude_step = abs(float(np.diff(longitudes).mean()))
    west = float(selected_longitudes.min() - longitude_step / 2)
    east = float(selected_longitudes.max() + longitude_step / 2)
    south = float(selected_latitudes.min() - latitude_step / 2)
    north = float(selected_latitudes.max() + latitude_step / 2)
    draw_grid_map(
        "ERA5-Land temperature cells within Kapchorwa",
        "Daily mean of ERA5-Land 2 m temperature on 12 May 2025. The same polygon defines the district extraction area.",
        values,
        west,
        south,
        east,
        north,
        kapchorwa,
        (255, 247, 214),
        (205, 73, 42),
        "daily mean degrees Celsius on 12 May 2025",
        "Source file: data/climate/raw/era5_land/temperature/era5_land_temperature_2025_05.nc (archive contains data_0.nc)",
        "03_kapchorwa_era5_land_grid_overlay_2025-05-12.png",
    )


def main():
    for path in (BOUNDARIES, CHIRPS, ERA5):
        if not path.exists():
            raise FileNotFoundError(path)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    features = load_features()
    kapchorwa = district_feature(features)
    create_uganda_location_map(features, kapchorwa)
    create_chirps_map(kapchorwa)
    create_era5_map(kapchorwa)
    print(OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
