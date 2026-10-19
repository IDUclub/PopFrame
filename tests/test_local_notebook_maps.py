"""The local notebook overlays its model boundary without changing map data."""

import json
from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import pandas as pd
import pytest
import requests
from folium import GeoJson
from geopandas.testing import assert_geodataframe_equal
from matplotlib.colors import to_rgba
from shapely.geometry import Point, box, shape


@pytest.mark.parametrize("cell_id", ["6d6f5582", "6ea85d93", "b8cbc1b6", "1248c45f"])
@pytest.mark.parametrize("notebook_name", ["PopFrame_local.ipynb", "PopFrame.ipynb"])
def test_interactive_maps_include_model_boundary(cell_id, notebook_name):
    notebook = json.loads((Path(__file__).resolve().parents[1] / notebook_name).read_text(encoding="utf-8"))
    source = next("".join(cell["source"]) for cell in notebook["cells"] if cell["id"] == cell_id)
    boundary = gpd.GeoDataFrame(geometry=[box(29, 59, 32, 61)], crs=4326).to_crs(32636)
    original = boundary.copy()
    hexagons = gpd.GeoDataFrame({"id": [1632]}, geometry=[box(30.1, 60.1, 30.2, 60.2)], crs=4326).to_crs(32636)
    territory = hexagons.assign(name="Гексагон 1632")
    towns = gpd.GeoDataFrame(
        {"name": ["Первый"], "population": [1000]}, geometry=[Point(30.15, 60.15)], crs=4326
    ).to_crs(32636)
    namespace = {
        "boundary": boundary,
        "hexagons": hexagons,
        "territory": territory,
        "territory_scored": territory.assign(score=2),
        "towns": towns,
        "pd": pd,
        "display": lambda value: None,
        "evaluation": SimpleNamespace(evaluate_territory_location=lambda grid: [{"score": 2}] * len(grid)),
    }
    exec(compile(source, cell_id, "exec"), namespace)
    map_view = namespace["m"]
    layers = [layer for layer in map_view._children.values() if isinstance(layer, GeoJson)]
    feature = layers[-1].data["features"][0]
    assert shape(feature["geometry"]).equals_exact(boundary.to_crs(4326).geometry.iloc[0], tolerance=1e-9)
    style = layers[-1].style_function(feature)
    assert style["fill"] is False
    assert style["color"] == "#28486d"
    assert len(layers) == (3 if cell_id == "b8cbc1b6" else 2)
    west, south, east, north = hexagons.to_crs(4326).total_bounds
    assert map_view.location == pytest.approx([(south + north) / 2, (west + east) / 2])
    assert "L.geoJson" in map_view.get_root().render()
    assert_geodataframe_equal(boundary, original)
    assert "score" not in hexagons


def test_landuse_map_uses_downloaded_data_and_shows_legend_and_hexagon(monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError("Rendering the downloaded land-use data must not make network requests")

    monkeypatch.setattr(requests.sessions.Session, "request", no_network)
    notebook = json.loads((Path(__file__).resolve().parents[1] / "PopFrame_local.ipynb").read_text(encoding="utf-8"))
    source = next("".join(cell["source"]) for cell in notebook["cells"] if cell.get("id") == "landuse-map")
    landuse = gpd.GeoDataFrame(
        {
            "Название хранимое": ["Процент земель промышленного назначения", "Территории смежного назначения"],
            "Значение": [25.0, 75.0],
        },
        geometry=[box(30.1, 59.3, 30.15, 59.4), box(30.15, 59.3, 30.3, 59.4)],
        crs=4326,
    )
    territory = gpd.GeoDataFrame(
        {"id": [1632], "name": ["Гексагон 1632"]}, geometry=[box(30.1, 59.3, 30.3, 59.4)], crs=4326
    ).to_crs(32636)
    original_landuse, original_territory = landuse.copy(), territory.copy()
    namespace = {"landuse_data": landuse, "territory": territory}

    exec(compile(source, "landuse-map", "exec"), namespace)

    map_view = namespace["m"]
    layers = [layer for layer in map_view._children.values() if isinstance(layer, GeoJson)]
    assert len(layers) == 2
    assert len(layers[0].data["features"]) == 2
    for feature in layers[0].data["features"]:
        label = feature["properties"]["Категория"]
        assert label in {"Промышленные земли", "Территории смежного назначения"}
        assert feature["properties"]["Значение"] in {25.0, 75.0}
        assert to_rgba(layers[0].style_function(feature)["fillColor"]) == to_rgba(
            "gray" if label == "Промышленные земли" else "white"
        )
    feature = layers[1].data["features"][0]
    assert layers[1].style_function(feature)["fill"] is False
    assert shape(feature["geometry"]).equals_exact(territory.to_crs(4326).geometry.iloc[0], tolerance=1e-9)
    html = map_view.get_root().render()
    for text in ["Промышленные земли", "Территории смежного назначения", "Доля площади, %"]:
        assert text in html or json.dumps(text, ensure_ascii=True)[1:-1] in html
    assert "Сельскохозяйственные земли" not in html
    assert json.dumps("Сельскохозяйственные земли", ensure_ascii=True)[1:-1] not in html
    assert_geodataframe_equal(landuse, original_landuse)
    assert_geodataframe_equal(territory, original_territory)
