from functools import singledispatchmethod
import dill as pickle
import pyproj
import geopandas as gpd
import pandas as pd
from .town import Town
import matplotlib.pyplot as plt
from popframe.preprocessing.level_filler import LevelFiller


DISTRICTS_PLOT_COLOR = "#28486d"
SETTLEMENTS_PLOT_COLOR = "#ddd"
TOWNS_PLOT_COLOR = "#333333"
TERRITORIES_PLOT_COLOR = "#893434"


class Region:
    """
    A class representing a geographical region that includes districts, settlements, towns, and optionally territories.
    Provides methods for validating and visualizing spatial data, as well as for calculating accessibility between towns.

    Attributes
    ----------
    crs : pyproj.CRS
        Coordinate Reference System of the region.
    region : gpd.GeoDataFrame
        GeoDataFrame representing the boundaries of the region.
    districts : gpd.GeoDataFrame
        GeoDataFrame containing information about districts.
    settlements : gpd.GeoDataFrame
        GeoDataFrame containing information about settlements.
    _towns : dict
        Dictionary containing `Town` objects indexed by their IDs.
    accessibility_matrix : pd.DataFrame
        DataFrame containing accessibility data between towns.
    _territories : dict
        Dictionary containing `Territory` objects indexed by their IDs (optional).
    """

    def __init__(
        self,
        region: gpd.GeoDataFrame,
        towns: gpd.GeoDataFrame,
        accessibility_matrix: pd.DataFrame,
    ):
        """
        Initializes the Region object with GeoDataFrames for region, districts, settlements, and towns.
        Optionally includes territories and an accessibility matrix to model transportation between towns.

        Parameters
        ----------
        region : gpd.GeoDataFrame
            GeoDataFrame representing the boundaries of the region.
        districts : gpd.GeoDataFrame
            GeoDataFrame containing information about districts.
        settlements : gpd.GeoDataFrame
            GeoDataFrame containing information about settlements.
        towns : gpd.GeoDataFrame
            GeoDataFrame containing information about towns.
        accessibility_matrix : pd.DataFrame
            DataFrame containing accessibility data between towns.
        territories : gpd.GeoDataFrame, optional
            GeoDataFrame containing information about territories (default is None).

        Raises
        ------
        AssertionError
            If the CRS or indices between the towns and the accessibility matrix do not match.
        """
        region = self.validate_region(region)
        towns = self.validate_towns(towns)
        accessibility_matrix = self.validate_accessibility_matrix(accessibility_matrix)

        assert (
            accessibility_matrix.index == towns.index
        ).all(), "Accessibility matrix indices and towns indices don't match"
        assert region.crs == towns.crs, "CRS should match everywhere"

        self.crs = towns.crs
        self.region = region
        self._towns = Town.from_gdf(towns)

        self.accessibility_matrix = accessibility_matrix

    @staticmethod
    def validate_towns(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Validates the towns GeoDataFrame.
        """
        assert isinstance(gdf, gpd.GeoDataFrame), "Towns should be instance of gpd.GeoDataFrame"
        return gdf

    @staticmethod
    def validate_region(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Validates the region GeoDataFrame to ensure it has the correct structure and data types.
        """
        assert isinstance(gdf, gpd.GeoDataFrame), "Region should be instance of gpd.GeoDataFrame"
        assert gdf.geom_type.isin(
            ["Polygon", "MultiPolygon"]
        ).all(), "District geometry should be Polygon or MultiPolygon"
        return gdf[["geometry"]]

    @staticmethod
    def validate_accessibility_matrix(df: pd.DataFrame) -> pd.DataFrame:
        """
        Validates the accessibility matrix, ensuring it has non-negative float values
        and matching row and column indices.
        """
        assert pd.api.types.is_float_dtype(df.values), "Accessibility matrix values should be float"
        assert (df.values >= 0).all(), "Accessibility matrix values should be greater or equal 0"
        assert (df.index == df.columns).all(), "Accessibility matrix indices and columns don't match"
        return df

    @property
    def towns(self) -> list[Town]:
        """
        Returns a list of all towns in the region.

        Returns
        -------
        list[Town]
            List of Town objects.
        """
        return self._towns.values()

    def get_update_towns_gdf(self, update_df: pd.DataFrame | None = None):
        gdf = self.get_towns_gdf()
        if update_df is not None:
            # Обновляем значения населения в gdf из update_df
            gdf.update(update_df[["population"]])

            level_filler = LevelFiller(towns=gdf)
            gdf = level_filler.fill_levels()
        return gdf

    def get_towns_gdf(self) -> gpd.GeoDataFrame:
        """
        Returns a GeoDataFrame representing all towns in the region, including their relationships with settlements and districts.

        Returns
        -------
        gpd.GeoDataFrame
            GeoDataFrame with town data.
        """
        data = [town.to_dict() for town in self.towns]
        gdf = gpd.GeoDataFrame(data, crs=self.crs)
        gdf.set_index("id", inplace=True, drop=False)
        gdf = gdf.rename_axis(None)
        return gdf.fillna(0)

    @singledispatchmethod
    def __getitem__(self, arg):
        """
        Overloaded subscript operator to access a town or accessibility data based on the argument type.

        Parameters
        ----------
        arg : int or tuple
            Integer to access a town by its ID, or tuple to retrieve accessibility data between two towns.

        Raises
        ------
        NotImplementedError
            If the argument type is unsupported.
        """
        raise NotImplementedError(f"Can't access object with such argument type {type(arg)}")

    # Make city_model subscriptable, to access block via ID like city_model[123]
    @__getitem__.register(int)
    def _(self, town_id):
        if not town_id in self._towns:
            raise KeyError(f"Can't find town with such id: {town_id}")
        return self._towns[town_id]

    @__getitem__.register(tuple)
    def _(self, towns):
        (town_a, town_b) = towns
        if isinstance(town_a, Town):
            town_a = town_a.id
        if isinstance(town_b, Town):
            town_b = town_b.id
        return self.accessibility_matrix.loc[town_a, town_b]

    @staticmethod
    def from_pickle(file_path: str):
        """
        Load a Region object from a .pickle file.

        Parameters
        ----------
        file_path : str
            Path to the .pickle file.

        Returns
        -------
        Region
            The loaded Region object.
        """
        state = None
        with open(file_path, "rb") as f:
            state = pickle.load(f)
        return state

    def to_pickle(self, file_path: str):
        """
        Save the Region object to a .pickle file.

        Parameters
        ----------
        file_path : str
            Path to the .pickle file.
        """
        with open(file_path, "wb") as f:
            pickle.dump(self, f)
