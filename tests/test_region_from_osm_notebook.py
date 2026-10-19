"""Execute the short notebook with mocked OSM sources and verify one model output."""

import ast
import json
from pathlib import Path
from unittest.mock import Mock

import geopandas as gpd
import networkx as nx
import osmnx as ox
import pandas as pd
import requests
from shapely.geometry import Point, box

from popframe.models.region import Region
from popframe.preprocessing import data_loader


def test_notebook_executes_to_one_pickle(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "examples/region_from_osm.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    monkeypatch.chdir(tmp_path)
    boundary = gpd.GeoDataFrame(geometry=[box(30.05, 60.05, 30.3, 60.3)], crs=4326)
    places = gpd.GeoDataFrame(
        {"name": ["Первый", "Второй"], "population": ["1000", "200"]},
        index=pd.MultiIndex.from_tuples([("node", 101), ("node", 205)], names=["element_type", "osmid"]),
        geometry=[Point(30.1, 60.1), Point(30.2, 60.2)],
        crs=4326,
    )
    graph = nx.MultiDiGraph(crs="EPSG:4326")
    graph.add_node(10, x=30.1, y=60.1)
    graph.add_node(20, x=30.2, y=60.2)
    graph.add_edge(10, 20, length=12000.0, highway="residential", maxspeed="60")
    graph.add_edge(20, 10, length=12000.0, highway="residential", maxspeed="30")
    sources = [Mock(return_value=boundary), Mock(return_value=places), Mock(return_value=graph)]
    monkeypatch.setattr(ox, "geocode_to_gdf", sources[0])
    monkeypatch.setattr(data_loader, "_download_place_nodes", sources[1])
    monkeypatch.setattr(ox, "graph_from_polygon", sources[2])
    monkeypatch.setattr(requests.sessions.Session, "request", Mock(side_effect=AssertionError("No live requests")))
    data = tmp_path / "data"
    data.mkdir()
    original = data / "Ленинградская область.pickle"
    original.write_bytes(b"original model must not be loaded or overwritten")
    namespace = {}
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            assert not any(isinstance(node, (ast.FunctionDef, ast.ClassDef)) for node in ast.walk(ast.parse(source)))
            exec(compile(source, f"{path}:{cell['id']}", "exec"), namespace)
    output = data / "Ленинградская область_osm.pickle"
    assert set(data.iterdir()) == {original, output}
    assert original.read_bytes() == b"original model must not be loaded or overwritten"
    restored = Region.from_pickle(output)
    assert restored.get_towns_gdf().index.tolist() == [101, 205]
    assert restored.get_towns_gdf().population.tolist() == [1000, 200]
    assert restored.accessibility_matrix.loc[101, 205] == 12.0
    assert restored.accessibility_matrix.loc[205, 101] == 24.0
    pd.testing.assert_frame_equal(restored.accessibility_matrix, namespace["region_model"].accessibility_matrix)
    assert all(source.call_count == 1 for source in sources)
