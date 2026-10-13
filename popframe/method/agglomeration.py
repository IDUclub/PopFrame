from .base_method import BaseMethod
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPolygon
import pandas as pd
from shapely.ops import unary_union
from typing import Dict, List, Optional

from popframe.utils.const import (
    TIME_TO_METERS_FACTOR,
    MIN_CITY_POPULATION_FOR_AGGLO,
)

CITY_LEVELS = [
    "Малый город",
    "Средний город",
    "Большой город",
    "Крупный город",
    "Крупнейший город",
    "Сверхкрупный город",
]


class AgglomerationBuilder(BaseMethod):
    """
    Build urban agglomerations from a towns layer and a travel-time matrix.

    Notes
    -----
    The algorithm:
    1) Iterates city levels from largest to smallest and selects candidate centers.
    2) For each center, takes all towns reachable within a time threshold and
       buffers them by the remaining time (converted to meters).
    3) Unifies buffers to produce proto-agglomerations.
    4) Merges intersecting proto-agglomerations and aggregates attributes.
    5) Clips to the region boundary and simplifies MultiPolygons.

    Attributes
    ----------
    radius_m_per_min : int
        Conversion factor from minutes of travel time to meters of buffer radius.
    min_population : int
        Minimal population for a town to be considered as a center or member.
    _in_agglomeration : Dict[int, bool]
        Registry of nodes already assigned to an agglomeration.
    """

    radius_m_per_min: int = TIME_TO_METERS_FACTOR
    min_population: int = MIN_CITY_POPULATION_FOR_AGGLO
    _in_agglomeration: Dict[int, bool] = {}

    def _build_agglomeration(self, towns: gpd.GeoDataFrame, time: int) -> gpd.GeoDataFrame:
        """
        Build proto-agglomerations for eligible centers by level and time threshold.

        Parameters
        ----------
        towns : GeoDataFrame
            Towns with columns: ``id``, ``name``, ``population``, ``level``, ``geometry`` (Points).
            CRS must match ``self.region.crs``.
        time : int
            Global time budget in minutes. Effective per-level budget decreases by 10 minutes
            per level step when iterating from largest to smallest.

        Returns
        -------
        GeoDataFrame
            Columns: ``name``, ``geometry``. Geometry is Polygon or MultiPolygon.
            Always has active geometry and CRS.

        Notes
        -----
        Centers are filtered by ``level in CITY_LEVELS`` and ``population >= min_population``.
        Remaining-time buffers are computed as
        ``(time_center - distance_minutes) * radius_m_per_min``.
        """
        node_population = towns.set_index("id")["population"]
        node_names = towns.set_index("id")["name"]
        agglomerations = []

        for level_index, level in enumerate(reversed(CITY_LEVELS)):
            max_time = time - 10 * level_index
            level_nodes = towns[towns["level"] == level].sort_values(by="population", ascending=False)

            for node, population in level_nodes[["id", "population"]].itertuples(index=False):
                if node in self._in_agglomeration or population < self.min_population:
                    continue

                agglomeration = self._get_agglomeration_around_node(node, max_time, towns)

                if agglomeration:
                    agglomeration["name"] = node_names.get(node)
                    agglomerations.append(agglomeration)

                    for member_node in agglomeration["nodes_in_agglomeration"]:
                        if node_population.get(member_node) < self.min_population:
                            self._in_agglomeration[member_node] = True

        if agglomerations:
            agglomeration_gdf = gpd.GeoDataFrame(
                agglomerations,
                geometry="geometry",
                crs=self.region.crs,
            )[["name", "geometry"]]
        else:
            agglomeration_gdf = gpd.GeoDataFrame(
                columns=["name", "geometry"],
                geometry="geometry",
                crs=self.region.crs,
            )

        return agglomeration_gdf

    def _get_agglomeration_around_node(
        self, start_node: int, max_time: int, towns: gpd.GeoDataFrame
    ) -> Optional[dict]:
        """
        Build a single proto-agglomeration around a candidate center.

        Parameters
        ----------
        start_node : int
            ID of the center town.
        max_time : int
            Maximum travel time (minutes) from ``start_node`` to include towns.
        towns : GeoDataFrame
            Towns table aligned with the accessibility matrix index.

        Returns
        -------
        dict or None
            ``{"geometry": Polygon|MultiPolygon, "nodes_in_agglomeration": List[int]}``
            or ``None`` if no towns are reachable within ``max_time``.

        Notes
        -----
        Uses ``self.region.accessibility_matrix`` with minutes as units.
        Guarantees point geometry for towns by coercing to centroid if needed.
        """
        accessibility_matrix = self.region.accessibility_matrix

        distances_from_start = accessibility_matrix.loc[start_node]
        within_time_nodes = distances_from_start[distances_from_start <= max_time].index

        if len(within_time_nodes) == 0:
            return None

        nodes_data = towns.set_index("id").loc[within_time_nodes].copy()

        # ensure point-like geometry
        nodes_data["geometry"] = nodes_data["geometry"].apply(
            lambda geom: Point(geom.x, geom.y) if isinstance(geom, Point) else geom.centroid
        )

        nodes_gdf = gpd.GeoDataFrame(nodes_data, geometry="geometry", crs=self.region.crs)

        # remaining buffer distance in meters
        distance = {node: (max_time - distances_from_start[node]) * self.radius_m_per_min for node in within_time_nodes}
        nodes_gdf["left_distance"] = nodes_gdf.index.map(distance)

        agglomeration_geom = nodes_gdf.buffer(nodes_gdf["left_distance"]).unary_union

        return {"geometry": agglomeration_geom, "nodes_in_agglomeration": list(within_time_nodes)}

    def _merge_intersecting_agglomerations(
        self, gdf: gpd.GeoDataFrame, towns: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        """
        Merge intersecting proto-agglomerations and compute attributes.

        Parameters
        ----------
        gdf : GeoDataFrame
            Proto-agglomerations with columns ``name`` and ``geometry``.
        towns : GeoDataFrame
            Towns layer used to aggregate population.

        Returns
        -------
        GeoDataFrame
            Columns: ``geometry``, ``type``, ``core_cities``, ``population``, ``agglomeration_level``.
            Always has active geometry and CRS.

        Notes
        -----
        Intersection is tested geometrically. Geometries are unioned via ``unary_union``.
        Invalid geometries are repaired with ``buffer(0)``.
        Agglomeration level is derived from total population thresholds.
        """
        if gdf.empty:
            return gpd.GeoDataFrame(
                columns=["geometry", "type", "core_cities", "population", "agglomeration_level"],
                geometry="geometry",
                crs=towns.crs if getattr(towns, "crs", None) is not None else self.region.crs,
            )

        merged_geometries = []
        processed_indices = set()

        for i, row_i in gdf.iterrows():
            if i in processed_indices:
                continue

            geometry = row_i["geometry"]
            merged_names = {row_i["name"]}

            for j, row_j in gdf.iterrows():
                if i != j and j not in processed_indices:
                    if geometry.intersects(row_j["geometry"]):
                        geometry = unary_union([geometry, row_j["geometry"]])
                        merged_names.add(row_j["name"])
                        processed_indices.add(j)

            still_intersecting = True
            while still_intersecting:
                still_intersecting = False
                for j, row_j in gdf.iterrows():
                    if j not in processed_indices and geometry.intersects(row_j["geometry"]):
                        geometry = unary_union([geometry, row_j["geometry"]])
                        merged_names.add(row_j["name"])
                        processed_indices.add(j)
                        still_intersecting = True

            if not geometry.is_valid:
                geometry = geometry.buffer(0)

            towns_in_agglomeration = towns[towns.intersects(geometry)]
            population_from_towns = towns_in_agglomeration["population"].sum()

            if population_from_towns <= 250_000:
                agglomeration_level = 1
            elif population_from_towns <= 500_000:
                agglomeration_level = 2
            elif population_from_towns <= 1_000_000:
                agglomeration_level = 3
            elif population_from_towns <= 5_000_000:
                agglomeration_level = 4
            else:
                agglomeration_level = 5

            merged_agglomeration = {
                "geometry": geometry,
                "type": "Polycentric" if len(merged_names) > 1 else "Monocentric",
                "core_cities": ", ".join(sorted(merged_names)),
                "population": population_from_towns,
                "agglomeration_level": agglomeration_level,
            }
            merged_geometries.append(merged_agglomeration)

            processed_indices.add(i)

        return gpd.GeoDataFrame(merged_geometries, geometry="geometry", crs=gdf.crs)

    def _simplify_multipolygons(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Keep only the largest polygon for MultiPolygon geometries.

        Parameters
        ----------
        gdf : GeoDataFrame
            Input geometries.

        Returns
        -------
        GeoDataFrame
            Copy with simplified geometries. Empty input is returned unchanged.
        """
        if gdf.empty:
            return gdf
        gdf = gdf.copy()
        gdf["geometry"] = gdf["geometry"].apply(
            lambda geom: max(geom.geoms, key=lambda g: g.area) if isinstance(geom, MultiPolygon) else geom
        )
        return gdf

    def evaluate_city_agglomeration_status(
        self, towns: gpd.GeoDataFrame, agglomeration_gdf: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        """
        Label towns by agglomeration membership and level.

        Parameters
        ----------
        towns : GeoDataFrame
            Towns layer with point geometries and columns ``name`` and ``geometry``.
        agglomeration_gdf : GeoDataFrame
            Agglomerations with columns ``geometry``, ``core_cities``, ``agglomeration_level``.

        Returns
        -------
        GeoDataFrame
            ``towns`` copy with two columns added:
            - ``agglomeration_status``: {"Вне агломерации","В агломерации","Центр агломерации"}
            - ``agglomeration_level``: int in {0..5}
        """
        agglomeration_status = []
        agglomeration_level = []

        for _, town in towns.iterrows():
            town_point = town["geometry"]
            town_name = town["name"]

            in_agglomeration = False
            is_core_city = False
            current_agglomeration_level = 0

            for _, agg in agglomeration_gdf.iterrows():
                if town_point.intersects(agg["geometry"]):
                    in_agglomeration = True
                    current_agglomeration_level = agg["agglomeration_level"]
                    core_cities = agg["core_cities"].split(", ")
                    if town_name in core_cities:
                        is_core_city = True
                        break

            if is_core_city:
                agglomeration_status.append("Центр агломерации")
                agglomeration_level.append(current_agglomeration_level)
            elif not in_agglomeration:
                agglomeration_status.append("Вне агломерации")
                agglomeration_level.append(0)
            else:
                agglomeration_status.append("В агломерации")
                agglomeration_level.append(current_agglomeration_level)

        towns = towns.copy()
        towns["agglomeration_status"] = agglomeration_status
        towns["agglomeration_level"] = agglomeration_level
        return towns

    def get_agglomerations(
        self,
        update_df: Optional[pd.DataFrame] = None,
        time: int = 80,
        raise_on_empty: bool = True,
    ) -> gpd.GeoDataFrame:
        """
        Build, merge, clip, and finalize urban agglomerations.

        Parameters
        ----------
        update_df : DataFrame, optional
            If provided, used by ``self.region.get_update_towns_gdf`` to update towns.
        time : int, default 80
            Global time budget in minutes. Values < 50 are coerced to 50.
        raise_on_empty : bool, default True
            If True, raise an error when no eligible towns or no agglomerations are produced.

        Returns
        -------
        GeoDataFrame
            Final agglomerations after merge, clip, and simplification.

        Raises
        ------
        ValueError
            If there are fewer than two towns or fewer than two towns with valid population.
        RuntimeError
            If no towns pass level/population filters or no agglomerations can be built
            and ``raise_on_empty`` is True.
        RuntimeError
            On any unexpected error inside the pipeline, with original exception chained.

        Notes
        -----
        The function validates candidates by
        ``towns['level'].isin(CITY_LEVELS)`` and
        ``towns['population'] >= self.min_population`` before building.
        """
        try:
            towns = self.region.get_update_towns_gdf(update_df)
            if towns is None or len(towns) < 2:
                raise ValueError("Для построения агломерации требуется минимум два города.")

            valid_pop = towns["population"].notnull() & (towns["population"] > 0)
            if valid_pop.sum() < 2:
                raise ValueError("Требуются данные о населении минимум у двух разных городов.")

            if time < 50:
                print("Минимально допустимое значение параметра 'time' — 50 минут. Заменяю на 50.")
                time = 50

            # preliminary eligibility check
            mask_level = towns["level"].isin(CITY_LEVELS)
            mask_pop = towns["population"] >= self.min_population
            eligible = towns[mask_level & mask_pop]
            if eligible.empty and raise_on_empty:
                levels_sample = list(towns["level"].value_counts().head(5).items())
                raise RuntimeError(
                    "No towns pass level and population criteria: "
                    f"min_population={self.min_population}, allowed levels={CITY_LEVELS}. "
                    f"Levels distribution (top5): {levels_sample}"
                )

            region_boundary = self.region.region

            # 1) build proto-agglomerations
            agglomeration_gdf = self._build_agglomeration(towns, time)

            # explicit error if nothing produced
            if agglomeration_gdf.empty and raise_on_empty:
                lvl_ok = towns[mask_level]
                pop_ok = towns[mask_pop]
                msg_parts = []
                if lvl_ok.empty:
                    msg_parts.append("no towns with allowed levels (CITY_LEVELS)")
                if pop_ok.empty:
                    msg_parts.append("no towns reaching min_population")
                if not msg_parts:
                    msg_parts.append("accessibility produced no reachable sets at given time")
                raise RuntimeError("Agglomerations not built: " + "; ".join(msg_parts))

            # 2) simplify
            agglomeration_gdf = self._simplify_multipolygons(agglomeration_gdf)

            # 3) merge overlapping and compute attributes
            agglomeration_gdf = self._merge_intersecting_agglomerations(agglomeration_gdf, towns)

            # 4) clip to region boundary
            if not agglomeration_gdf.empty:
                agglomeration_gdf = gpd.overlay(agglomeration_gdf, region_boundary, how="intersection")

            # 5) final simplify
            agglomeration_gdf = self._simplify_multipolygons(agglomeration_gdf)
            return agglomeration_gdf

        except Exception as e:
            raise RuntimeError(f"Ошибка при построении агломераций: {e}") from None
