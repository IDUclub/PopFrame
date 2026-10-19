"""Anchor maps use population colors and preserve separate time scenarios."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pytest
from geopandas.testing import assert_geodataframe_equal
from matplotlib.colors import to_rgba
from shapely.geometry import Point, box

from popframe.visualization import plot_anchor_settlements


@pytest.fixture
def map_data():
    points = gpd.GeoDataFrame(
        {
            "name": ["Первый", "Второй", "Третий"],
            "population": [1000, 15000, 100000],
            "level": ["Нет уровня"] * 3,
            "is_anchor_settlement": [True] * 3,
        },
        geometry=[Point(500000, 6600000), Point(501000, 6600000), Point(500500, 6601000)],
        crs=32636,
    )
    zones = gpd.GeoDataFrame(geometry=[box(499000, 6599000, 502000, 6602000)], crs=points.crs)
    region = SimpleNamespace(region=zones.copy(), crs=points.crs)
    yield region, zones, points
    plt.close("all")


@pytest.mark.parametrize("colored", [False, True])
def test_population_colors_and_thin_boundaries_preserve_data(map_data, colored):
    region, zones, points = map_data
    original_zones, original_points = zones.copy(), points.copy()

    ax = plot_anchor_settlements(region, zones, points, color_by_population=colored)

    expected = ["#FFD700", "#CD5C5C", "#DC143C"] if colored else ["green"]
    np.testing.assert_allclose(ax.collections[-1].get_facecolors(), [to_rgba(color, alpha=0.9) for color in expected])
    assert len(ax.collections[-1].get_offsets()) == 3
    assert ax.collections[0].get_linewidths()[0] == 0.4
    assert ax.collections[1].get_linewidths()[0] == 0.4
    labels = [text.get_text() for text in ax.get_legend().texts]
    if colored:
        assert labels == ["Зоны опорных пунктов", "201–1 000 чел.", "5 001–50 000 чел.", "50 001–100 000 чел."]
    else:
        assert labels == ["Зоны опорных пунктов", "Опорные пункты"]
    assert_geodataframe_equal(zones, original_zones)
    assert_geodataframe_equal(points, original_points)


@pytest.mark.parametrize("base_time", [50, 70])
@pytest.mark.parametrize("notebook_name", ["PopFrame_local.ipynb", "PopFrame.ipynb"])
def test_notebook_scenario_adds_fifteen_minutes_without_replacing_base(map_data, base_time, monkeypatch, notebook_name):
    region, base, points = map_data
    original_base, original_points = base.copy(), points.copy()
    scenario = base.copy()
    scenario.geometry = scenario.buffer(1000)
    builder = SimpleNamespace(get_anchor_settlement_boundaries=Mock(return_value=scenario))
    factory = Mock(return_value=builder)
    notebook = json.loads((Path(__file__).resolve().parents[1] / notebook_name).read_text(encoding="utf-8"))
    source = next("".join(cell["source"]) for cell in notebook["cells"] if cell.get("id") == "anchor-time-scenario")
    namespace = {
        "ANCHOR_TIME": base_time,
        "AnchorSettlementBuilder": factory,
        "region_model": region,
        "anchor_towns": points,
        "anchor_boundaries": base,
        "plot_anchor_settlements": plot_anchor_settlements,
        "plt": plt,
    }
    monkeypatch.setattr(plt, "show", lambda: None)

    exec(compile(source, "anchor-time-scenario", "exec"), namespace)

    factory.assert_called_once_with(region=region)
    kwargs = builder.get_anchor_settlement_boundaries.call_args.kwargs
    assert kwargs["towns"] is points
    assert kwargs["time"] == base_time + 15
    assert namespace["anchor_boundaries_scenario"] is scenario
    assert namespace["anchor_boundaries"] is base
    assert_geodataframe_equal(base, original_base)
    assert_geodataframe_equal(points, original_points)
