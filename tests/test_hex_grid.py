"""Preserve teaching IDs, geography, and the map-to-calculation selection."""

import unittest

import geopandas as gpd
import h3
import matplotlib.pyplot as plt
import pandas as pd
from geopandas.testing import assert_geodataframe_equal
from shapely.geometry import Point, Polygon, box

from popframe.interactive import HexagonSelector
from popframe.method.territory_evaluation import TerritoryEvaluation
from popframe.models.region import Region
from popframe.preprocessing.hex_grid import create_leningrad_hexagons


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

    def test_click_and_dropdown_pass_numeric_id_to_calculation(self):
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
        picker = HexagonSelector(grid, region)
        self.assertEqual(picker.value, 1632)
        feature = next(f for f in picker.grid_layer.data["features"] if f["properties"]["id"] == 25)
        # Exercise the same event dispatch used by the frontend, not just the callback.
        picker.grid_layer._handle_mouse_events(None, {"event": "click", "feature": feature}, [])
        self.assertEqual(picker.value, 25)
        self.assertEqual(picker.selection_layer.data["features"][0]["properties"]["id"], 25)
        self.assertIn("25", picker.caption.value)
        picker.dropdown.value = 1632
        selected = grid.loc[grid["id"].eq(picker.value)].copy().reset_index(drop=True)
        self.assertEqual(selected.id.tolist(), [1632])
        self.assertIs(type(picker.value), int)
        selected["name"] = f"Гексагон {picker.value}"
        result = TerritoryEvaluation(region=region).evaluate_territory_location(selected)[0]
        self.assertEqual(result["territory"], "Гексагон 1632")
        self.assertEqual(result["score"], 2)
        ax = region.plot(territories=selected.to_crs(4326))
        self.assertEqual(
            [text.get_text() for text in ax.get_legend().texts], ["Населённые пункты", "Выбранная территория"]
        )
        assert_geodataframe_equal(grid, source)
        plt.close(ax.figure)
        picker.close()


if __name__ == "__main__":
    unittest.main()
