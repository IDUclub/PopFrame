"""Population scenarios preserve geometry and leave the base region unchanged."""

import geopandas as gpd
import pandas as pd
from geopandas.testing import assert_geodataframe_equal
from pandas.testing import assert_frame_equal
from shapely.geometry import Point, box

from popframe.method.agglomeration import AgglomerationBuilder
from popframe.method.popuation_frame import PopulationFrame
from popframe.models.region import Region


def test_population_scenario_builds_agglomerations_and_circles():
    towns = gpd.GeoDataFrame(
        {
            "name": ["Первый", "Второй"],
            "population": [60000, 1000],
            "level": ["Средний город", "Большое сельское поселение"],
        },
        index=[101, 205],
        geometry=[Point(500000, 6600000), Point(501000, 6600000)],
        crs=32636,
    )
    boundary = gpd.GeoDataFrame(geometry=[box(490000, 6590000, 510000, 6610000)], crs=towns.crs)
    matrix = pd.DataFrame([[0.0, 12.0], [18.0, 0.0]], index=towns.index, columns=towns.index)
    region = Region(region=boundary, towns=towns, accessibility_matrix=matrix)
    original_towns = region.get_towns_gdf()
    original_boundary = region.region.copy()
    original_matrix = region.accessibility_matrix.copy()
    base = AgglomerationBuilder(region=region).get_agglomerations(time=80)
    base_copy = base.copy()
    update_df = pd.DataFrame({"population": [1_000_000]}, index=[101])
    original_update = update_df.copy()

    scenario_towns = region.get_update_towns_gdf(update_df)
    builder = AgglomerationBuilder(region=region)
    scenario = builder.get_agglomerations(update_df=update_df, time=80)
    status = builder.evaluate_city_agglomeration_status(scenario_towns, scenario)
    circles = PopulationFrame(region=region).build_circle_frame(update_df)

    assert not scenario.empty
    assert scenario.geometry.is_valid.all()
    assert scenario.crs == region.crs
    assert base.population.tolist() == [61000]
    assert scenario.population.tolist() == [1_001_000]
    assert scenario_towns.index.tolist() == [101, 205]
    assert scenario_towns.loc[101, "population"] == 1_000_000
    assert scenario_towns.loc[101, "level"] == "Крупный город"
    assert_geodataframe_equal(scenario_towns[["geometry"]], original_towns[["geometry"]])
    assert status.loc[101, "agglomeration_status"] == "Центр агломерации"
    assert circles.geometry.geom_type.eq("Polygon").all()
    assert circles.crs == region.crs
    assert_geodataframe_equal(region.get_towns_gdf(), original_towns)
    assert_geodataframe_equal(region.region, original_boundary)
    assert_frame_equal(region.accessibility_matrix, original_matrix)
    assert_frame_equal(update_df, original_update)
    assert_geodataframe_equal(base, base_copy)
