from abc import ABC
from typing import Generic, TypeVar

import geopandas as gpd
from pydantic import BaseModel, ConfigDict
from shapely.geometry.base import BaseGeometry


T = TypeVar("T")


class BaseRow(BaseModel, ABC):
    """
    Abstract base class for data validation in a GeoDataFrame, designed to be inherited by other classes.

    Attributes
    ----------
    model_config : ConfigDict
        Configuration for allowing arbitrary types in Pydantic validation.
    geometry : BaseGeometry
        Geometry attribute for storing shapely geometries.
    index : int
        Index attribute, can be overridden, but shouldn't be set by default.

    Notes
    -----
    Inheriting classes can be configured to provide default column values to avoid None and NaN.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    geometry: BaseGeometry
    index: int


class GeoDataFrame(gpd.GeoDataFrame, BaseModel, Generic[T]):
    """
    Custom GeoDataFrame class that extends geopandas.GeoDataFrame and supports data validation with Pydantic's BaseModel.
    This class allows for the automatic validation of data on initialization using a generic class T inherited from BaseRow.

    Attributes
    ----------
    generic : property
        A property to access the generic class used in the GeoDataFrame.

    Methods
    -------
    __init__(self, data, *args, **kwargs)
        Initializes a GeoDataFrame with validated rows based on the provided generic class.

    Parameters
    ----------
    data : Any
        The input data for creating the GeoDataFrame. It can be any format that GeoDataFrame accepts.
    *args : tuple
        Additional arguments to pass to the GeoDataFrame.
    **kwargs : dict
        Additional keyword arguments for configuring the GeoDataFrame.

    Notes
    -----
    - The provided generic class (T) must be a subclass of BaseRow for validation.
    - If the input data is not already a GeoDataFrame, it will be converted into one.
    - The "index" column is managed separately to avoid conflicts with the GeoDataFrame's index.
    - The coordinate reference system (CRS) is either provided in the kwargs or inherited from the input data.
    """

    @property
    def generic(self):
        """
        Returns the generic class type used in the GeoDataFrame. This is needed to ensure Pydantic validation is performed correctly.

        Returns
        -------
        T : Type
            The type of the generic class used for row validation.
        """
        return self.__pydantic_generic_metadata__["args"][0]

    def __init__(self, data, *args, **kwargs):
        """
        Initializes the GeoDataFrame with data validation and optional CRS.

        Parameters
        ----------
        data : Any
            The data used to initialize the GeoDataFrame. It could be a DataFrame, dictionary, or list of records.
        *args : tuple
            Additional positional arguments passed to GeoDataFrame.
        **kwargs : dict
            Additional keyword arguments, including CRS (Coordinate Reference System) for spatial data.

        Raises
        ------
        AssertionError
            If the provided generic class does not inherit from BaseRow.
        """
        generic_class = self.generic
        assert issubclass(generic_class, BaseRow), "Generic should be inherited from BaseRow"

        # Convert data to GeoDataFrame if it isn't one already
        if not isinstance(data, gpd.GeoDataFrame):
            data = gpd.GeoDataFrame(data, *args, **kwargs)

        # Create a list of dicts from BaseRow inherited class for validation
        rows: list[dict] = []
        for i in data.index:
            dict = data.loc[i].to_dict()
            rows.append(generic_class(index=i, **dict).__dict__)

        # Initialize the GeoDataFrame with validated data
        super().__init__(rows)

        # Restore the original index and drop the "index" column if it exists
        if "index" in self.columns:
            self.index = self["index"]
            self.drop(columns=["index"], inplace=True)

        # Restore the original index name and set geometry and CRS
        index_name = data.index.name
        self.index.name = index_name
        self.set_geometry("geometry", inplace=True)

        # Set CRS (Coordinate Reference System)
        self.crs = kwargs["crs"] if "crs" in kwargs else data.crs
