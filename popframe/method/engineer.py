import geopandas as gpd
from shapely.geometry.base import BaseGeometry
from typing import Dict, Any, Set
from popframe.utils.const import RADIUS_NPP_M, RADIUS_HPP_M, RADIUS_DEFAULT_INFRA_M


class InfrastructureAnalyzer:
    def __init__(self, infrastructure_gdf: gpd.GeoDataFrame, assessment_areas_gdf: gpd.GeoDataFrame) -> None:
        """
        Initialize the InfrastructureAnalyzer with infrastructure and assessment areas.

        Parameters
        ----------
        infrastructure_gdf : geopandas.GeoDataFrame
            GeoDataFrame containing infrastructure objects.
        assessment_areas_gdf : geopandas.GeoDataFrame
            GeoDataFrame containing assessment areas.
        """
        # Convert coordinate systems to metric (EPSG:3857) for calculations
        self.infrastructure_gdf = infrastructure_gdf.to_crs(epsg=3857)
        self.assessment_areas_gdf = assessment_areas_gdf.to_crs(epsg=3857)

        # Initialize columns and start analysis
        self.assessment_areas_gdf["score"] = 0
        self.assessment_areas_gdf["types_in_radius"] = None
        self._analyze_infrastructure()

    @staticmethod
    def get_radius(physical_object_type: Dict[str, Any]) -> float:
        """
        Determines the radius based on the physical object type.

        Parameters
        ----------
        physical_object_type : dict
            Dictionary containing information about the physical object type.

        Returns
        -------
        float
            The radius in meters for the given object type.
        """
        name = physical_object_type.get("name", "")
        if "Атомная электростанция" in name:
            return float(RADIUS_NPP_M)
        if "Гидроэлектростанция" in name:
            return float(RADIUS_HPP_M)
        return float(RADIUS_DEFAULT_INFRA_M)

    def _analyze_infrastructure(self) -> None:
        """
        Analyzes infrastructure for each territory and adds assessment attributes.

        Returns
        -------
        None
        """
        # For each assessment area, check the presence of unique object types within the radius
        for index, area in self.assessment_areas_gdf.iterrows():
            # Create a temporary set of unique types for this assessment area
            unique_types_in_radius: Set[str] = set()

            # Check each object in infrastructure_gdf for inclusion in the buffer
            for _, obj in self.infrastructure_gdf.iterrows():
                # Extract `physical_object_type` dictionary to determine the radius
                physical_object_info: Dict[str, Any] = obj["physical_object_type"]
                buffer_distance: float = self.get_radius(physical_object_info)

                # Create a buffer for the current assessment area
                area_buffer: BaseGeometry = area.geometry.buffer(buffer_distance)

                # If the object is within the buffer, add its `type` to the set
                if obj.geometry.within(area_buffer):
                    unique_types_in_radius.add(obj["type"])

            # Count unique types and assign them to 'score' and 'types_in_radius' attributes
            self.assessment_areas_gdf.at[index, "score"] = len(unique_types_in_radius)
            self.assessment_areas_gdf.at[index, "types_in_radius"] = list(unique_types_in_radius)

    def get_results(self) -> gpd.GeoDataFrame:
        """
        Returns the result with columns 'id', 'score', 'types_in_radius', and 'geometry' in CRS 4326.

        Returns
        -------
        geopandas.GeoDataFrame
            GeoDataFrame with the results in CRS 4326.
        """
        # Convert back to CRS 4326 before returning
        return self.assessment_areas_gdf[["score", "types_in_radius", "geometry"]].to_crs(epsg=4326)
