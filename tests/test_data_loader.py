"""Resumable Overpass loading, isolated settings, routing and one safe pickle."""

import json
from pathlib import Path
from unittest.mock import Mock

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import pytest
import requests
from geopandas.testing import assert_geodataframe_equal
from osmnx._errors import InsufficientResponseError
from shapely.geometry import Point, Polygon, box

from popframe.models.region import Region
from popframe.preprocessing import data_loader
from popframe.preprocessing.data_loader import (
    RegionDataLoader,
    apply_population_updates,
    build_region,
    calculate_accessibility,
    prepare_boundary,
    prepare_towns,
    save_region,
)


@pytest.fixture
def osm_sources(monkeypatch):
    boundary = gpd.GeoDataFrame(geometry=[box(30.05, 60.05, 30.3, 60.3)], crs=4326)
    places = gpd.GeoDataFrame(
        {"name": ["Первый", "Второй"], "population": ["1000", "200"], "place": ["town", "village"]},
        index=pd.MultiIndex.from_tuples([("node", 101), ("node", 205)], names=["element_type", "osmid"]),
        geometry=[Point(30.1, 60.1), Point(30.2, 60.2)],
        crs=4326,
    )
    graph = nx.MultiDiGraph(crs="EPSG:4326")
    graph.add_node(10, x=30.1, y=60.1)
    graph.add_node(20, x=30.2, y=60.2)
    graph.add_edge(10, 20, length=12000.0, highway="residential", maxspeed="60")
    graph.add_edge(20, 10, length=12000.0, highway="residential", maxspeed="30")
    sources = {
        "boundary": Mock(side_effect=lambda query: boundary.copy()),
        "places": Mock(side_effect=lambda geometry: places.copy()),
        "graph": Mock(side_effect=lambda geometry, **kwargs: graph.copy()),
    }
    monkeypatch.setattr(ox, "geocode_to_gdf", sources["boundary"])
    monkeypatch.setattr(data_loader, "_download_place_nodes", sources["places"])
    monkeypatch.setattr(ox, "graph_from_polygon", sources["graph"])
    monkeypatch.setattr(requests.sessions.Session, "request", Mock(side_effect=AssertionError("No live requests")))
    return sources


@pytest.fixture
def loader(tmp_path):
    return RegionDataLoader({"state": "Тестовая область", "country": "Россия"}, cache_dir=tmp_path / "cache")


def test_loader_reuses_cache_and_builds_one_pickle(loader, osm_sources, tmp_path):
    boundary = loader.load_boundary()
    towns = loader.load_towns()
    graph = loader.load_graph()
    before_boundary, before_towns, before_graph = boundary.copy(), towns.copy(), graph.copy()
    result = build_region(boundary, towns, graph)
    output = save_region(result, tmp_path / "output/model.pickle")
    restored = Region.from_pickle(output)
    assert restored.get_towns_gdf().index.tolist() == [101, 205]
    assert restored.accessibility_matrix.loc[101, 205] == 12.0
    assert restored.accessibility_matrix.loc[205, 101] == 24.0
    pd.testing.assert_frame_equal(restored.accessibility_matrix, result.accessibility_matrix)
    assert list(output.parent.iterdir()) == [output]
    assert_geodataframe_equal(boundary, before_boundary)
    assert_geodataframe_equal(towns, before_towns)
    assert nx.utils.graphs_equal(graph, before_graph)
    counts = {name: source.call_count for name, source in osm_sources.items()}
    assert counts == {"boundary": 1, "places": 1, "graph": 1}
    resumed = RegionDataLoader(loader.place_query, cache_dir=loader.cache_dir)
    resumed.load_boundary()
    resumed.load_towns()
    resumed.load_graph()
    assert {name: source.call_count for name, source in osm_sources.items()} == counts


def test_cache_rejects_a_different_region_or_buffer(loader):
    with pytest.raises(ValueError, match="Параметры выборки изменились"):
        RegionDataLoader("Другой регион", cache_dir=loader.cache_dir)
    with pytest.raises(ValueError, match="Параметры выборки изменились"):
        RegionDataLoader(loader.place_query, cache_dir=loader.cache_dir, routing_buffer_m=100)


def test_partial_tiles_resume_and_deduplicate(loader, monkeypatch):
    boundary = gpd.GeoDataFrame(geometry=[box(30.1, 60.1, 31.9, 60.9)], crs=4326)
    monkeypatch.setattr(loader, "load_boundary", lambda: boundary.copy())
    first = gpd.GeoDataFrame(
        {"name": ["На границе"], "population": ["1000"]},
        index=pd.MultiIndex.from_tuples([("node", 101)], names=["element_type", "osmid"]),
        geometry=[Point(31, 60.5)],
        crs=4326,
    )
    second = gpd.GeoDataFrame(
        {"name": ["На границе", "Второй"], "population": ["1000", "200"]},
        index=pd.MultiIndex.from_tuples([("node", 101), ("node", 205)], names=["element_type", "osmid"]),
        geometry=[Point(31, 60.5), Point(31.5, 60.5)],
        crs=4326,
    )
    download = Mock(side_effect=[first, ValueError("broken tile"), second])
    monkeypatch.setattr(data_loader, "_download_place_nodes", download)
    with pytest.raises(ValueError, match="broken tile"):
        loader.load_towns()
    assert (loader.cache_dir / "place_tiles/30_60.parquet").exists()
    assert not (loader.cache_dir / "places_osm.parquet").exists()
    towns = loader.load_towns()
    assert towns.index.tolist() == [101, 205]
    assert download.call_count == 3
    assert download.call_args_list[0].args[0].bounds == (30.1, 60.1, 31.0, 60.9)
    assert download.call_args_list[2].args[0].bounds == (31.0, 60.1, 31.9, 60.9)
    loader.load_towns()
    assert download.call_count == 3


def http_response(status, payload=None):
    response = requests.Response()
    response.status_code = status
    response.url = "https://example.test/api/interpreter"
    response._content = json.dumps(payload or {"elements": []}).encode()
    response._content_consumed = True
    return response


@pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 504])
def test_transient_http_errors_retry_and_restore_settings(loader, monkeypatch, status):
    failed = http_response(status)
    if status == 429:
        failed.headers["Retry-After"] = "90"
    responses = Mock(side_effect=[failed, http_response(200)])
    original_kwargs = ox.settings.requests_kwargs
    original_endpoint = ox.settings.overpass_endpoint
    sleep = Mock()
    monkeypatch.setattr(data_loader.time, "sleep", sleep)

    def operation():
        assert ox.settings.overpass_endpoint == loader.overpass_endpoint
        return ox.settings.requests_kwargs["hooks"]["response"](responses())

    assert loader._retry("Граф", operation).status_code == 200
    assert responses.call_count == 2
    sleep.assert_called_once_with(90 if status == 429 else 30)
    assert ox.settings.requests_kwargs is original_kwargs
    assert ox.settings.overpass_endpoint == original_endpoint


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_access_errors_are_not_repeated(loader, monkeypatch, status):
    response = http_response(status)
    operation = Mock(side_effect=lambda: response.raise_for_status())
    sleep = Mock()
    monkeypatch.setattr(data_loader.time, "sleep", sleep)
    with pytest.raises(RuntimeError, match=f"HTTP {status}"):
        loader._retry("Пункты", operation)
    operation.assert_called_once()
    sleep.assert_not_called()


def test_timeouts_stop_after_four_attempts_with_original_cause(loader, monkeypatch):
    error = requests.Timeout("slow server")
    operation = Mock(side_effect=error)
    pauses = []
    monkeypatch.setattr(data_loader.time, "sleep", pauses.append)
    with pytest.raises(RuntimeError, match="4 попытки") as raised:
        loader._retry("Пункты", operation)
    assert raised.value.__cause__ is error
    assert operation.call_count == 4
    assert pauses == [30, 60, 60]


def test_malformed_json_retries_but_partial_data_is_rejected(loader, monkeypatch):
    error = InsufficientResponseError("HTML instead of JSON")
    error.__cause__ = requests.exceptions.JSONDecodeError("Not JSON", "<html>", 0)
    monkeypatch.setattr(data_loader.time, "sleep", Mock())
    operation = Mock(side_effect=[error, 42])
    assert loader._retry("Пункты", operation) == 42
    partial = InsufficientResponseError("Overpass: Query timed out")
    operation = Mock(side_effect=partial)
    with pytest.raises(InsufficientResponseError):
        loader._retry("Пункты", operation)
    operation.assert_called_once()


def test_place_request_filters_holes_and_accepts_empty_tile(monkeypatch):
    boundary = box(30, 60, 31, 61).difference(box(30.4, 60.4, 30.6, 60.6))
    response = http_response(
        200,
        {
            "elements": [
                {
                    "type": "node",
                    "id": 10,
                    "lon": 30.1,
                    "lat": 60.1,
                    "tags": {"place": "town", "name": "Первый", "population": "1 000"},
                },
                {"type": "node", "id": 20, "lon": 30.5, "lat": 60.5, "tags": {"place": "town"}},
                {"type": "node", "id": 30, "lon": 32, "lat": 62, "tags": {"place": "town"}},
            ]
        },
    )
    post = Mock(side_effect=[response, http_response(200)])
    monkeypatch.setattr(requests, "post", post)
    places = data_loader._download_place_nodes(boundary)
    assert places.index.tolist() == [("node", 10)]
    assert places.loc[("node", 10), "population"] == "1 000"
    query = post.call_args.kwargs["data"]["data"]
    assert 'node["place"="town"]' in query and "way" not in query
    assert data_loader._download_place_nodes(boundary).empty


def test_overpass_remark_rejects_an_incomplete_response(monkeypatch):
    response = http_response(200, {"elements": [], "remark": "Query timed out"})
    monkeypatch.setattr(requests, "post", Mock(return_value=response))
    with pytest.raises(InsufficientResponseError, match="неполный ответ"):
        data_loader._download_place_nodes(box(30, 60, 31, 61))


def test_boundary_repair_is_explicit_and_preserves_source():
    boundary = gpd.GeoDataFrame(geometry=[Polygon([(30, 60), (31, 61), (30, 61), (31, 60), (30, 60)])], crs=4326)
    original = boundary.copy()
    with pytest.raises(ValueError, match="корректных"):
        prepare_boundary(boundary)
    repaired = prepare_boundary(boundary, repair_geometry=True)
    assert repaired.geometry.is_valid.all() and repaired.crs.is_projected
    assert_geodataframe_equal(boundary, original)


def test_invalid_population_blocks_routing(loader, osm_sources, monkeypatch):
    towns = loader.load_towns()
    towns.loc[101, "population"] = pd.NA
    matrix = Mock(side_effect=AssertionError("No routing on incomplete population"))
    monkeypatch.setattr(data_loader, "calculate_accessibility", matrix)
    with pytest.raises(ValueError, match="ID:.*101"):
        build_region(loader.load_boundary(), towns, loader.load_graph())
    matrix.assert_not_called()


def test_remote_attachments_are_rejected_in_metres(loader, osm_sources):
    towns = loader.load_towns().to_crs(4326)
    towns.loc[101, "geometry"] = Point(30.29, 60.29)
    with pytest.raises(ValueError, match="далёкая привязка.*ID:.*101"):
        build_region(loader.load_boundary(), towns, loader.load_graph())


def test_pickle_does_not_overwrite_and_failed_write_is_not_published(loader, osm_sources, tmp_path, monkeypatch):
    model = build_region(loader.load_boundary(), loader.load_towns(), loader.load_graph())
    output = save_region(model, tmp_path / "result.pickle")
    saved = output.read_bytes()
    with pytest.raises(FileExistsError):
        save_region(model, output)
    assert output.read_bytes() == saved

    def failed_write(path):
        Path(path).write_bytes(b"partial")
        raise OSError("write interrupted")

    monkeypatch.setattr(model, "to_pickle", failed_write)
    with pytest.raises(OSError, match="write interrupted"):
        save_region(model, tmp_path / "failed.pickle")
    assert not (tmp_path / "failed.pickle").exists()
    assert not list(tmp_path.glob(".popframe-*"))


def osm_places(populations):
    return gpd.GeoDataFrame(
        {
            "name": ["Первый", "Второй", "Полигон", ""],
            "name:ru": ["Первый по-русски", None, None, None],
            "population": populations,
            "place": ["town", "village", "village", "hamlet"],
        },
        index=pd.MultiIndex.from_tuples(
            [("node", 205), ("node", 101), ("way", 205), ("node", 303)],
            names=["element_type", "osmid"],
        ),
        geometry=[Point(30.1, 60.1), Point(30.2, 60.2), box(30.1, 60.1, 30.3, 60.3), Point(30.3, 60.3)],
        crs=4326,
    )


def test_prepare_towns_preserves_osm_identity_and_missing_population():
    places = osm_places(["1\u00a0234", "~100", "500", "20"])
    original = places.copy(deep=True)
    towns = prepare_towns(places, 32636)
    assert towns.index.tolist() == [101, 205]
    assert towns["id"].tolist() == [101, 205]
    assert pd.isna(towns.loc[101, "population"])
    assert towns.loc[205, "population"] == 1234
    assert towns.loc[205, "name"] == "Первый по-русски"
    assert towns.crs.to_epsg() == 32636
    assert_geodataframe_equal(places, original)


@pytest.mark.parametrize("population", [None, "", "0", "-3", "10.5", "inf", "2020;2021"])
def test_invalid_osm_population_stays_missing(population):
    towns = prepare_towns(osm_places([population, "25", "500", "20"]), 32636)
    assert pd.isna(towns.loc[205, "population"])


def test_population_updates_require_valid_ids_values_and_sources():
    towns = prepare_towns(osm_places(["100", None, "500", "20"]), 32636)
    updates = pd.DataFrame({"population": [250], "source": ["Перепись, 2021"]}, index=[101])
    result = apply_population_updates(towns, updates)
    assert result.loc[101, "population"] == 250
    assert result.loc[101, "population_source"] == "Перепись, 2021"
    assert pd.isna(towns.loc[101, "population"])
    invalid_updates = [
        updates.rename(index={101: 999}),
        pd.concat([updates, updates]),
        updates.assign(population=-1),
        updates.assign(population=2.5),
        updates.assign(population=np.inf),
        updates.assign(source=" "),
    ]
    for invalid in invalid_updates:
        with pytest.raises(ValueError):
            apply_population_updates(towns, invalid)


def test_matrix_uses_fastest_directed_edge_minutes_and_town_order():
    graph = nx.MultiDiGraph()
    graph.add_edge(30, 10, travel_time=120.0)
    graph.add_edge(30, 10, travel_time=60.0)
    graph.add_edge(10, 30, travel_time=180.0)
    graph.add_node(99)
    town_nodes = pd.Series([10, 30, 10, 99], index=pd.Index([205, 101, 303, 404], name="id"))
    matrix = calculate_accessibility(graph, town_nodes)
    expected = pd.DataFrame(
        [[0.0, 3.0, 0.0, np.inf], [1.0, 0.0, 1.0, np.inf], [0.0, 3.0, 0.0, np.inf], [np.inf, np.inf, np.inf, 0.0]],
        index=town_nodes.index,
        columns=town_nodes.index,
    )
    pd.testing.assert_frame_equal(matrix, expected)
    assert graph.number_of_edges() == 3


@pytest.mark.parametrize("weight", [None, -1, np.nan, np.inf])
def test_matrix_rejects_invalid_edge_weights(weight):
    graph = nx.MultiDiGraph()
    graph.add_edge(10, 20, **({} if weight is None else {"travel_time": weight}))
    with pytest.raises(ValueError, match="travel_time"):
        calculate_accessibility(graph, pd.Series([10, 20], index=[101, 205]))
