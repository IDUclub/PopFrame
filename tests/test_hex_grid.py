"""Preserve teaching IDs, geography, and the map-to-calculation selection."""

import json
import unittest

import geopandas as gpd
import h3
import matplotlib.pyplot as plt
import pandas as pd
from folium import GeoJson
from geopandas.testing import assert_geodataframe_equal
from matplotlib.colors import to_rgba
from shapely.geometry import Point, Polygon, box

from popframe.method.territory_evaluation import TerritoryEvaluation
from popframe.models.region import Region
from popframe.preprocessing.hex_grid import create_leningrad_hexagons
from popframe.visualization import explore_territory_scores, plot_settlement_scheme


class HexGridTests(unittest.TestCase):
    def test_original_grid_ids_and_default_geometry(self):
        grid = create_leningrad_hexagons()
        self.assertEqual(grid.id.tolist(), list(range(2148)))
        self.assertTrue(grid.h3_id.is_unique)
        self.assertTrue(grid.is_valid.all())
        self.assertTrue(grid.h3_id.map(h3.get_resolution).eq(6).all())
        # Independently copied from the original LO_hex.geojson, ID 1632.
        original = Polygon(
            [
                (30.04028664932525, 59.353443067762704),
                (30.077111744900783, 59.32963592202586),
                (30.13293261543692, 59.3368411489111),
                (30.152000952287747, 59.36787341312169),
                (30.115160822025704, 59.39170048869122),
                (30.05926733680958, 59.384475355800944),
            ]
        )
        self.assertLess(grid.loc[1632].geometry.hausdorff_distance(original), 1e-10)
        projected = create_leningrad_hexagons(crs=32636)
        self.assertEqual(projected.crs.to_epsg(), 32636)
        assert_geodataframe_equal(projected.to_crs(4326), grid, check_less_precise=True)

    def test_explore_and_numeric_id_selection(self):
        grid = create_leningrad_hexagons(crs=32636)
        source = grid.copy()
        towns = gpd.GeoDataFrame(
            {
                "name": ["Первый", "Второй"],
                "population": [1000, 2000],
                "level": ["Среднее сельское поселение", "Большое сельское поселение"],
            },
            geometry=[Point(30.1, 59.36), Point(30.3, 59.4)],
            crs=4326,
        ).to_crs(grid.crs)
        region = Region(
            region=gpd.GeoDataFrame(geometry=[box(29, 59, 32, 61)], crs=4326).to_crs(grid.crs),
            towns=towns,
            accessibility_matrix=pd.DataFrame([[0.0, 10.0], [12.0, 0.0]]),
        )
        map_view = grid.explore(tiles="OpenStreetMap", tooltip=["id"], popup=["id"])
        layer = next(layer for layer in map_view._children.values() if isinstance(layer, GeoJson))
        self.assertEqual(len(layer.data["features"]), 2148)
        self.assertEqual(layer.data["features"][1632]["properties"]["id"], 1632)
        self.assertIn("L.geoJson", map_view.get_root().render())
        selected_id = 1632
        selected = grid.loc[grid["id"].eq(selected_id)].copy().reset_index(drop=True)
        self.assertEqual(selected.id.tolist(), [1632])
        selected["name"] = f"Гексагон {selected_id}"
        result = TerritoryEvaluation(region=region).evaluate_territory_location(selected)[0]
        self.assertEqual(result["territory"], "Гексагон 1632")
        self.assertEqual(result["score"], 2)
        ax = region.plot(territories=selected.to_crs(4326))
        self.assertEqual(
            [text.get_text() for text in ax.get_legend().texts], ["Населённые пункты", "Выбранная территория"]
        )
        assert_geodataframe_equal(grid, source)
        plt.close(ax.figure)

        circles = towns.copy()
        circles.geometry = circles.geometry.buffer(100)
        areas = region.region.assign(agglomeration_level=4)
        ax = plot_settlement_scheme(region, circles, areas)
        self.assertEqual(tuple(ax.collections[1].get_facecolors()[0]), to_rgba("#FFFF99"))
        self.assertEqual(tuple(ax.collections[2].get_facecolors()[0]), to_rgba("#FFD700"))
        plt.close(ax.figure)

    def test_score_map_zero_is_black_and_positive_scores_use_old_palette(self):
        grid = create_leningrad_hexagons().iloc[:3].copy()
        grid["score"] = [0, 1, 6]
        source = grid.copy()
        map_view = explore_territory_scores(grid)
        layer = next(layer for layer in map_view._children.values() if isinstance(layer, GeoJson))
        colors = [layer.style_function(feature)["fillColor"] for feature in layer.data["features"]]
        self.assertEqual(colors, ["black", "#a50026", "#006837"])
        self.assertIn(json.dumps("Оценка расположения территории"), map_view.get_root().render())
        assert_geodataframe_equal(grid, source)


if __name__ == "__main__":
    unittest.main()
