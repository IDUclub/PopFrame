"""Location results include only the settlement fields used by each assessment."""

import geopandas as gpd
import pandas as pd
import pytest
from geopandas.testing import assert_geodataframe_equal
from shapely.geometry import Point, box

from popframe.method.territory_evaluation import TerritoryEvaluation
from popframe.models.region import Region


@pytest.fixture
def evaluation():
    towns = gpd.GeoDataFrame(
        {"name": ["Западный", "Восточный"], "population": [20000, 30000], "level": ["Малый город"] * 2},
        geometry=[Point(-20000, 0), Point(20000, 0)],
        index=[101, 205],
        crs=3857,
    )
    region = Region(
        region=gpd.GeoDataFrame(geometry=[box(-100000, -100000, 100000, 100000)], crs=towns.crs),
        towns=towns,
        accessibility_matrix=pd.DataFrame([[0.0, 30.0], [30.0, 0.0]], index=towns.index, columns=towns.index),
    )
    return TerritoryEvaluation(region=region)


@pytest.mark.parametrize(
    "geometry, score, settlements",
    [
        (box(19900, -100, 20100, 100), 5, {"closest_settlement": "Восточный"}),
        (
            box(-100, -100, 100, 100),
            1,
            {"closest_settlement1": "Западный", "closest_settlement2": "Восточный"},
        ),
        (box(99000, 99000, 100000, 100000), 0, {}),
    ],
)
def test_location_table_has_only_relevant_settlement_columns(evaluation, geometry, score, settlements):
    territory = gpd.GeoDataFrame({"name": ["Участок"]}, geometry=[geometry], crs=3857)

    results = evaluation.evaluate_territory_location(territory)

    assert len(results) == 1
    assert results[0]["territory"] == "Участок"
    assert results[0]["score"] == score
    assert {key: value for key, value in results[0].items() if key.startswith("closest_settlement")} == settlements
    assert set(pd.DataFrame(results).columns) == {"territory", "score", "interpretation"} | settlements.keys()


def test_mixed_location_results_keep_order_scores_and_input_data(evaluation):
    territories = gpd.GeoDataFrame(
        {"name": ["Рядом", "Между", "Вдали"]},
        geometry=[box(19900, -100, 20100, 100), box(-100, -100, 100, 100), box(99000, 99000, 100000, 100000)],
        index=[9, 2, 7],
        crs=3857,
    ).to_crs(4326)
    original_territories = territories.copy()
    original_towns = evaluation.region.get_towns_gdf()

    results = evaluation.evaluate_territory_location(territories)

    assert [result["territory"] for result in results] == ["Рядом", "Между", "Вдали"]
    assert [result["score"] for result in results] == [5, 1, 0]
    assert [len(result) for result in results] == [4, 5, 3]
    assert_geodataframe_equal(territories, original_territories)
    assert_geodataframe_equal(evaluation.region.get_towns_gdf(), original_towns)
