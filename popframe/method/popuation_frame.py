import geopandas as gpd
from ..utils.const import METERS_PER_DEGREE
from shapely.geometry import Point
import pandas as pd

from .base_method import BaseMethod


class PopulationFrame(BaseMethod):
    def _create_circle(self, center, size):
        """
        Create a circular buffer around a point.

        Parameters
        ----------
        center : shapely.geometry.Point
            The center point of the circle.
        size : float
            The radius of the circle.

        Returns
        -------
        shapely.geometry.Polygon
            The resulting circular polygon.
        """
        return center.buffer(size)

    def _size_from_population(self, population, level):
        """
        Calculate the size of the circle based on population and settlement level.

        Parameters
        ----------
        population : int or float
            The population of the settlement.
        level : str
            The level of the settlement.

        Returns
        -------
        float
            The calculated size for the circle.
        """
        if level in ["Малое сельское поселение", "Среднее сельское поселение", "Большое сельское поселение"]:
            return 0.0001 * (population**0.5)  # Logarithmic scale for small settlements
        elif level == "Сверхкрупный город":
            return 6e-5 * (population**0.5)  # Reduced linear scale for very large cities
        return 0.0001 * (population**0.5)  # Linear scale for large settlements

    def _convert_points_to_circles(self, gdf):
        """
        Convert point geometries to circles based on population and level.

        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            GeoDataFrame with point geometries and population data.

        Returns
        -------
        geopandas.GeoDataFrame
            GeoDataFrame with circular geometries.
        """
        gdf["size"] = gdf.apply(lambda row: self._size_from_population(row["population"], row["level"]), axis=1)
        gdf["size_in_meters"] = gdf["size"] * METERS_PER_DEGREE
        gdf["geometry"] = gdf.apply(
            lambda row: self._create_circle(row["geometry"], row["size_in_meters"])
            if isinstance(row["geometry"], Point)
            else row["geometry"],
            axis=1,
        )
        gdf = gdf.drop(columns=["size", "size_in_meters"])
        return gdf

    def build_circle_frame(self, update_df: pd.DataFrame | None = None) -> gpd.GeoDataFrame:
        """
        Build a GeoDataFrame of circles representing settlements based on population and level.

        Parameters
        ----------
        update_df : pandas.DataFrame or None, optional
            Optional DataFrame to update the towns data.

        Returns
        -------
        geopandas.GeoDataFrame
            GeoDataFrame with circular geometries for settlements.
        """
        towns = self.region.get_update_towns_gdf(update_df)
        gdf = self._convert_points_to_circles(towns)
        return gdf
