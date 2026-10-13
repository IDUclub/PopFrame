import geopandas as gpd
from pydantic import BaseModel, Field, field_validator
from shapely.geometry import Point
from typing import ClassVar
from ..models.geodataframe import GeoDataFrame, BaseRow


class TownRow(BaseRow):
    """
    A model representing a town's data row in a GeoDataFrame.

    Attributes
    ----------
    geometry : shapely.geometry.Point
        The geographic location of the town.
    name : str
        The name of the town.
    population : int
        The population of the town. Must be greater than zero.
    level : str, optional
        The administrative level of the town, defaults to "Нет уровня" (no level).
    """

    id: int
    geometry: Point
    name: str
    population: int = Field(gt=0)
    level: str = Field(default="Нет уровня")


class LevelFiller(BaseModel):
    """
    A class for automatically assigning administrative levels to towns based on population thresholds.

    Attributes
    ----------
    towns : GeoDataFrame[TownRow]
        A GeoDataFrame containing town data that includes name, population, and level.
    population_thresholds : dict
        A class-level attribute defining population ranges for different administrative levels.

    Methods
    -------
    _assign_level(row) -> str
        A static method that assigns the correct administrative level to a town based on its population.

    validate_towns(gdf)
        A Pydantic validator that ensures town levels are correctly assigned before processing the GeoDataFrame.

    fill_levels() -> GeoDataFrame[TownRow]
        Fills in the levels for all towns in the GeoDataFrame based on population and returns the updated GeoDataFrame.
    """

    towns: GeoDataFrame[TownRow]
    population_thresholds: ClassVar[dict[str, tuple[int, int]]] = {
        "Сверхкрупный город": (3000000, float("inf")),
        "Крупнейший город": (1000000, 3000000),
        "Крупный город": (250000, 1000000),
        "Большой город": (100000, 250000),
        "Средний город": (50000, 100000),
        "Малый город": (5000, 50000),
        "Крупное сельское поселение": (3000, 5000),
        "Большое сельское поселение": (1000, 3000),
        "Среднее сельское поселение": (200, 1000),
        "Малое сельское поселение": (0, 200),
    }

    @staticmethod
    def _assign_level(row) -> str:
        """
        Assigns a level to a town based on its population.

        Parameters
        ----------
        row : pd.Series
            A row from the GeoDataFrame containing town data.

        Returns
        -------
        str
            The administrative level of the town.
        """
        population = row["population"]

        for level, (lower_bound, upper_bound) in LevelFiller.population_thresholds.items():
            if lower_bound < population <= upper_bound:
                return level
        return "Неизвестный уровень"  # If the population does not fit in any of the ranges

    @field_validator("towns", mode="before")
    @classmethod
    def validate_towns(cls, gdf):
        """
        Validates and assigns levels to towns in the provided GeoDataFrame.

        Parameters
        ----------
        gdf : GeoDataFrame
            A GeoDataFrame containing town data.

        Returns
        -------
        GeoDataFrame[TownRow]
            A validated GeoDataFrame with assigned administrative levels.
        """
        gdf["id"] = gdf.index
        gdf["level"] = gdf.apply(cls._assign_level, axis=1)
        return GeoDataFrame[TownRow](gdf)

    def fill_levels(self) -> GeoDataFrame[TownRow]:
        """
        Fills in the administrative levels for the towns based on their population.

        Returns
        -------
        GeoDataFrame[TownRow]
            An updated GeoDataFrame with filled levels for each town.
        """
        return self.towns
