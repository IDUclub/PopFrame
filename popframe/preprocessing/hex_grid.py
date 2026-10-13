"""Construct the teaching grid from H3 addresses, retaining the original IDs."""

import json
from importlib.resources import files

import geopandas as gpd
import h3
from shapely.geometry import Polygon


def create_leningrad_hexagons(crs=4326) -> gpd.GeoDataFrame:
    """Generate the 2,148 hexagons of the Leningrad teaching dataset.

    The packaged list stores H3 resolution-6 addresses in the original
    ``LO_hex.geojson`` ID order. H3 constructs polygon coordinates at runtime;
    no external GeoJSON is needed. The returned ``id`` values remain 0–2147,
    including the original study territory 1632. ``h3_id`` is its global address.

    Parameters
    ----------
    crs : CRS-like, default 4326
        Output coordinate reference system, e.g. the prepared Region's CRS.
    """
    cell_ids = json.loads(files(__package__).joinpath("resources/leningrad_hex_ids.json").read_text(encoding="utf-8"))
    geometries = [Polygon([(lng, lat) for lat, lng in h3.cell_to_boundary(cell)]) for cell in cell_ids]
    return gpd.GeoDataFrame({"id": range(len(cell_ids)), "h3_id": cell_ids}, geometry=geometries, crs=4326).to_crs(crs)
