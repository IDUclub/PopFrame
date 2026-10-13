from pydantic import BaseModel, InstanceOf, Field
import shapely
import geopandas as gpd


class Town(BaseModel):
    """
    A class representing a town with an ID, name, administrative level, population, and geometry.

    Attributes
    ----------
    id : int
        Unique identifier for the town.
    name : str
        Name of the town.
    level : str
        Administrative level of the town (e.g., city, village).
    population : int
        Population of the town. Must be greater than zero.
    geometry : shapely.Point
        Shapely Point object representing the town's geographic location.

    Methods
    -------
    to_dict() -> dict
        Converts the Town object to a dictionary format.

    from_gdf(cls, gdf: gpd.GeoDataFrame) -> dict
        Class method to create a dictionary of Town instances from a GeoDataFrame.
    """

    id: int
    name: str
    level: str
    population: int = Field(gt=0)
    geometry: InstanceOf[shapely.Point]

    def to_dict(self) -> dict:
        """
        Converts the Town instance into a dictionary.

        Returns
        -------
        dict
            A dictionary containing the 'id', 'name', 'population', 'level', and 'geometry' of the town.
        """
        res = {
            "id": self.id,
            "name": self.name,
            "population": self.population,
            "level": self.level,
            "geometry": self.geometry,
        }
        return res

    @classmethod
    def from_gdf(cls, gdf: gpd.GeoDataFrame) -> dict:
        """
        Creates a dictionary of Town objects from a GeoDataFrame.

        Parameters
        ----------
        gdf : gpd.GeoDataFrame
            GeoDataFrame containing data to initialize Town instances.
            The GeoDataFrame should have 'id', 'name', 'level', 'population', and 'geometry' columns.

        Returns
        -------
        dict
            A dictionary where the keys are the indices of the GeoDataFrame and the values are Town instances.
        """
        return {i: cls(**{**gdf.loc[i].to_dict(), "id": i}) for i in gdf.index}
