from statistics import median
from pydantic import BaseModel, model_validator, field_validator, Field, InstanceOf
from ..models.geodataframe import GeoDataFrame, BaseRow
import shapely
import pandas as pd


class UnitRow(BaseRow):
    """
    A class representing a unit of geographic data with a polygon or multipolygon geometry and a population count.

    Attributes
    ----------
    geometry : shapely.Polygon | shapely.MultiPolygon
        Polygonal geometry representing the unit's boundaries.
    population : int
        The population residing within the geographic unit.
    """

    geometry: shapely.Polygon | shapely.MultiPolygon
    population: int


class TownRow(BaseRow):
    """
    A class representing a town's data row with point geometry, name, administrative level, and city status.

    Attributes
    ----------
    geometry : shapely.Point
        The geographic location of the town.
    name : str
        The name of the town.
    level : str
        The administrative level of the town.
    is_city : bool
        A boolean indicating whether the town is a city.
    """

    geometry: shapely.Point
    name: str
    level: str
    is_city: bool


class PopulationFiller(BaseModel):
    """
    A class for filling in population data for towns based on their proximity to geographic units and an adjacency matrix.

    Attributes
    ----------
    units : GeoDataFrame[UnitRow]
        A GeoDataFrame containing geographic units with population data.
    towns : GeoDataFrame[TownRow]
        A GeoDataFrame containing town data.
    adjacency_matrix : pd.DataFrame
        A DataFrame representing the adjacency matrix between towns, used to calculate median travel times.
    city_multiplier : float, optional
        A multiplier applied to towns that are cities for population distribution, defaults to 10.

    Methods
    -------
    validate_units(cls, gdf) -> GeoDataFrame[UnitRow]
        Validates that the units input is a valid GeoDataFrame of UnitRow type.

    validate_towns(cls, gdf) -> GeoDataFrame[TownRow]
        Validates that the towns input is a valid GeoDataFrame of TownRow type.

    validate_adjacency_matrix(cls, df) -> pd.DataFrame
        Validates that the adjacency matrix is square and matches the town index.

    validate_model(self) -> PopulationFiller
        Validates that the CRS of towns and units match and that the adjacency matrix matches the town indices.

    _get_median_time(self, town_id) -> float
        Computes the median time from the adjacency matrix for a given town.

    fill(self) -> GeoDataFrame[TownRow]
        Fills in population data for the towns based on their proximity to geographic units and the adjacency matrix.
    """

    units: GeoDataFrame[UnitRow]
    towns: GeoDataFrame[TownRow]
    adjacency_matrix: InstanceOf[pd.DataFrame]
    city_multiplier: float = Field(gt=0, default=10)

    @field_validator("units", mode="before")
    @classmethod
    def validate_units(cls, gdf):
        """
        Validates that the input is a GeoDataFrame of UnitRow type.

        Parameters
        ----------
        gdf : GeoDataFrame
            The input GeoDataFrame to be validated.

        Returns
        -------
        GeoDataFrame[UnitRow]
            A validated GeoDataFrame of UnitRow type.
        """
        if not isinstance(gdf, GeoDataFrame[UnitRow]):
            gdf = GeoDataFrame[UnitRow](gdf)
        return gdf

    @field_validator("towns", mode="before")
    @classmethod
    def validate_towns(cls, gdf):
        """
        Validates that the input is a GeoDataFrame of TownRow type.

        Parameters
        ----------
        gdf : GeoDataFrame
            The input GeoDataFrame to be validated.

        Returns
        -------
        GeoDataFrame[TownRow]
            A validated GeoDataFrame of TownRow type.
        """
        if not isinstance(gdf, GeoDataFrame[TownRow]):
            gdf = GeoDataFrame[TownRow](gdf)
        return gdf

    @field_validator("adjacency_matrix", mode="after")
    @classmethod
    def validate_adjacency_matrix(cls, df):
        """
        Validates the adjacency matrix to ensure it is square and that its index matches the columns.

        Parameters
        ----------
        df : pd.DataFrame
            The adjacency matrix to be validated.

        Returns
        -------
        pd.DataFrame
            A validated adjacency matrix.

        Raises
        ------
        AssertionError
            If the matrix index and columns do not match.
        """
        assert all(df.index == df.columns), "Matrix index and columns don't match"
        return df

    @model_validator(mode="after")
    def validate_model(self):
        """
        Validates that the coordinate reference systems (CRS) of the towns and units match, and that the adjacency matrix matches the town indices.

        Returns
        -------
        PopulationFiller
            The validated PopulationFiller instance.

        Raises
        ------
        AssertionError
            If the CRS of the towns and units do not match, or if the matrix indices and town indices do not match.
        """
        adj_mx = self.adjacency_matrix
        towns = self.towns.copy()
        units = self.units
        assert towns.crs == units.crs, "Units and towns CRS don't match"
        assert all(adj_mx.index == towns.index), "Matrix index and towns index don't match"
        return self

    def _get_median_time(self, town_id) -> float:
        """
        Calculates the median time from the adjacency matrix for a given town.

        Parameters
        ----------
        town_id : int
            The ID of the town for which to calculate the median time.

        Returns
        -------
        float
            The median travel time for the town.
        """
        return median(self.adjacency_matrix.loc[town_id])

    def fill(self) -> GeoDataFrame[TownRow]:
        """
        Distributes the population from geographic units to towns based on the proximity and population distribution model.

        Returns
        -------
        GeoDataFrame[TownRow]
            A GeoDataFrame with updated population data for the towns.
        """
        towns = self.towns.copy()
        towns["median_time"] = towns.apply(lambda x: self._get_median_time(x.name), axis=1)
        for i in self.units.index:
            geometry = self.units.loc[i, "geometry"]
            population = self.units.loc[i, "population"]
            unit_towns = towns.loc[towns.within(geometry)].copy()
            unit_towns["coef"] = unit_towns.apply(
                lambda x: (self.city_multiplier if x["is_city"] else 1) / x["median_time"], axis=1
            )
            coef_sum = unit_towns["coef"].sum()
            unit_towns["coef_norm"] = unit_towns["coef"] / coef_sum
            unit_towns["population"] = population * unit_towns["coef_norm"]
            for j in unit_towns.index:
                towns.loc[j, "coef"] = unit_towns.loc[j, "coef"]
                towns.loc[j, "coef_norm"] = unit_towns.loc[j, "coef_norm"]
                towns.loc[j, "population"] = round(unit_towns.loc[j, "population"])
        return towns
