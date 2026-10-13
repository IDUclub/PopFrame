"""Checks for preparing a loaded region without modifying its source data."""

import unittest
from copy import deepcopy

import geopandas as gpd
import numpy as np
import pandas as pd
from geopandas.testing import assert_geodataframe_equal
from pandas.testing import assert_frame_equal
from shapely.geometry import Point, Polygon, box

from popframe.models.region import Region


class RegionPreparationTests(unittest.TestCase):
    def setUp(self):
        towns = gpd.GeoDataFrame(
            {"name": ["Первый", "Второй"], "population": [60000, 1000], "level": ["Нет уровня"] * 2},
            index=[101, 205],
            geometry=[Point(30.3, 59.9), Point(30.6, 60.0)],
            crs=4326,
        )
        boundary = gpd.GeoDataFrame(geometry=[box(30, 59, 32, 61)], crs=4326)
        matrix = pd.DataFrame([[0.0, 12.0], [18.0, 0.0]], index=towns.index, columns=towns.index)
        self.region = Region(region=boundary, towns=towns, accessibility_matrix=matrix)

    def test_prepare_aligns_projects_and_preserves_source(self):
        # Separate row/column orders must not change the meaning of directed trips.
        self.region.accessibility_matrix = self.region.accessibility_matrix.loc[[205, 101], [101, 205]]
        source_towns = self.region.get_towns_gdf().copy()
        source_boundary = self.region.region.copy()
        source_matrix = self.region.accessibility_matrix.copy()

        result = self.region.prepare_for_analysis()

        self.assertIsNot(result, self.region)
        self.assertTrue(result.crs.is_projected)
        self.assertEqual(result.crs.axis_info[0].unit_name, "metre")
        self.assertEqual(result.region.crs, result.crs)
        self.assertEqual(result.accessibility_matrix.index.tolist(), [101, 205])
        self.assertEqual(result.accessibility_matrix.columns.tolist(), [101, 205])
        self.assertEqual(result.accessibility_matrix.loc[101, 205], 12)
        self.assertEqual(result.accessibility_matrix.loc[205, 101], 18)
        self.assertEqual(result.get_towns_gdf().loc[101, "level"], "Средний город")
        assert_geodataframe_equal(result.get_towns_gdf().to_crs(4326)[["geometry"]],
                                  source_towns[["geometry"]], check_less_precise=True)
        assert_geodataframe_equal(self.region.get_towns_gdf(), source_towns)
        assert_geodataframe_equal(self.region.region, source_boundary)
        assert_frame_equal(self.region.accessibility_matrix, source_matrix)
        result.accessibility_matrix.loc[101, 205] = 99
        assert_frame_equal(self.region.accessibility_matrix, source_matrix)

    def test_unreachable_pairs_are_preserved(self):
        self.region.accessibility_matrix.loc[101, 205] = np.inf
        result = self.region.prepare_for_analysis()
        self.assertTrue(np.isposinf(result.accessibility_matrix.loc[101, 205]))

    def test_invalid_matrices(self):
        base = self.region.accessibility_matrix
        cases = [
            (base.rename(index={101: 999}), "идентификаторы|Идентификаторы"),
            (base.rename(columns={101: 999}), "идентификаторы|Идентификаторы"),
            (base.rename(index={101: 205}), "повторяются"),
            (base.rename(columns={101: 205}), "повторяются"),
            (pd.DataFrame([[0, np.nan], [18, 0]], index=base.index, columns=base.columns), "пропуски"),
            (pd.DataFrame([[0, -1], [18, 0]], index=base.index, columns=base.columns), "отрицательные"),
            (pd.DataFrame([[1, 12], [18, 0]], index=base.index, columns=base.columns), "нулевым"),
            (pd.DataFrame([[0, "bad"], [18, 0]], index=base.index, columns=base.columns), "числа"),
        ]
        for matrix, message in cases:
            with self.subTest(matrix=matrix):
                region = deepcopy(self.region)
                region.accessibility_matrix = matrix
                with self.assertRaisesRegex(ValueError, message):
                    region.prepare_for_analysis()

    def test_missing_crs(self):
        self.region.crs = None
        with self.assertRaisesRegex(ValueError, "система координат"):
            self.region.prepare_for_analysis()

    def test_invalid_boundary(self):
        for geometry in [Polygon(), Polygon([(30, 59), (32, 61), (32, 59), (30, 61), (30, 59)])]:
            with self.subTest(geometry=geometry):
                self.region.region = gpd.GeoDataFrame(geometry=[geometry], crs=4326)
                with self.assertRaisesRegex(ValueError, "Граница региона"):
                    self.region.prepare_for_analysis()

    def test_invalid_town_geometry_and_population(self):
        for field, value, message in [
            ("geometry", Point(), "геометрии"),
            ("population", 0, "Население"),
            ("population", np.inf, "Население"),
        ]:
            with self.subTest(field=field, value=value):
                region = deepcopy(self.region)
                setattr(region._towns[101], field, value)
                with self.assertRaisesRegex(ValueError, message):
                    region.prepare_for_analysis()

    def test_too_few_towns(self):
        for towns in [{}, {101: self.region._towns[101]}]:
            with self.subTest(count=len(towns)):
                self.region._towns = towns
                with self.assertRaisesRegex(ValueError, "минимум два"):
                    self.region.prepare_for_analysis()


if __name__ == "__main__":
    unittest.main()
