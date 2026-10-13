from .base_method import BaseMethod
import geopandas as gpd
import pandas as pd
import numpy as np
from typing import Optional
from shapely.geometry import Polygon, MultiPolygon, LineString
from shapely.ops import unary_union, split
from geovoronoi import voronoi_regions_from_coords

from popframe.utils.const import TIME_TO_METERS_FACTOR


class AnchorSettlementBuilder(BaseMethod):
    """Build boundaries for anchor (опорные) settlements using travel-time buffers."""

    radius_m_per_min: int = TIME_TO_METERS_FACTOR

    def _build_anchor_settlement_boundaries(self, towns: gpd.GeoDataFrame, time: int) -> gpd.GeoDataFrame:
        """
        Builds boundaries for anchor settlements iteratively, starting from the closest town based on travel time.

        Parameters
        ----------
        towns : geopandas.GeoDataFrame
            GeoDataFrame of towns.
        time : int
            Maximum travel time.

        Returns
        -------
        geopandas.GeoDataFrame
            GeoDataFrame of the anchor settlement boundaries.
        """
        anchor_towns = towns[towns["is_anchor_settlement"] == True]
        boundaries = []

        accessibility_matrix = self.region.accessibility_matrix

        for node in anchor_towns["id"]:
            boundary = self._get_boundary_around_node(node, time, towns, accessibility_matrix)

            if boundary:
                boundary["name"] = towns.loc[towns["id"] == node, "name"].values[0]
                boundaries.append(boundary)

        if boundaries:
            boundary_gdf = gpd.GeoDataFrame(boundaries, columns=["name", "geometry"]).set_geometry("geometry")
            boundary_gdf.set_crs(towns.crs, inplace=True)
        else:
            boundary_gdf = gpd.GeoDataFrame(columns=["name", "geometry"])

        return boundary_gdf

    def _get_boundary_around_node(
        self,
        start_node: int,
        max_time: int,
        towns: gpd.GeoDataFrame,
        accessibility_matrix: pd.DataFrame,
    ) -> Optional[dict]:
        """
        Get the boundary around a node within a maximum travel time.

        Parameters
        ----------
        start_node : int or str
            The starting node (town id).
        max_time : int
            Maximum travel time.
        towns : geopandas.GeoDataFrame
            GeoDataFrame of towns.
        accessibility_matrix : pandas.DataFrame
            Matrix of travel times between towns.

        Returns
        -------
        dict or None
            Dictionary with geometry and nodes in boundary, or None if no nodes found.
        """
        distances_from_start = accessibility_matrix.loc[start_node]
        within_time_nodes = distances_from_start[distances_from_start <= max_time].index

        if within_time_nodes.empty:
            return None

        nodes_gdf = towns.set_index("id").loc[within_time_nodes]

        distance = {node: (max_time - distances_from_start[node]) * self.radius_m_per_min for node in within_time_nodes}
        # distance = {node: (max_time) * RADIUS for node in within_time_nodes}
        nodes_gdf["left_distance"] = nodes_gdf.index.map(distance)
        boundary_geom = nodes_gdf.buffer(nodes_gdf["left_distance"]).unary_union

        return {"geometry": boundary_geom, "nodes_in_boundary": list(within_time_nodes)}

    def _simplify_multipolygons(self, gdf: gpd.GeoDataFrame, towns: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Simplify MultiPolygon geometries, keeping only those intersecting with anchor towns.

        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            GeoDataFrame with geometries to simplify.
        towns : geopandas.GeoDataFrame
            GeoDataFrame of towns.

        Returns
        -------
        geopandas.GeoDataFrame
            Simplified GeoDataFrame.
        """
        anchor_towns = towns[towns["is_anchor_settlement"] == True]

        def process_geometry(geom):
            if isinstance(geom, MultiPolygon):
                polygons = list(geom.geoms)

                # Фильтруем только те полигоны, которые пересекаются с anchor_towns
                filtered_polygons = [p for p in polygons if any(p.intersects(at) for at in anchor_towns.geometry)]

                if filtered_polygons:
                    return MultiPolygon(filtered_polygons) if len(filtered_polygons) > 1 else filtered_polygons[0]
                else:
                    return None  # Удаляем геометрию, если ни один полигон не пересекается

            return geom if any(geom.intersects(at) for at in anchor_towns.geometry) else None

        gdf["geometry"] = gdf["geometry"].apply(process_geometry)
        gdf = gdf.dropna(subset=["geometry"]).reset_index(drop=True)  # Удаляем строки без геометрии
        return gdf

    def _merge_intersecting_boundaries(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Merge intersecting boundaries into single geometries.

        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            GeoDataFrame with boundaries to merge.

        Returns
        -------
        geopandas.GeoDataFrame
            GeoDataFrame with merged boundaries.
        """
        merged_geometries = []
        processed_indices = set()

        for i, row_i in gdf.iterrows():
            if i in processed_indices:
                continue

            overlapping_boundaries = [row_i]
            geometry = row_i["geometry"]
            merged_names = {row_i["name"]}

            for j, row_j in gdf.iterrows():
                if i != j and j not in processed_indices:
                    if geometry.intersects(row_j["geometry"]):
                        overlapping_boundaries.append(row_j)
                        geometry = unary_union([geometry, row_j["geometry"]]).buffer(0)
                        merged_names.add(row_j["name"])
                        processed_indices.add(j)

            still_merging = True
            while still_merging:
                still_merging = False
                for j, row_j in gdf.iterrows():
                    if j not in processed_indices and geometry.intersects(row_j["geometry"]):
                        overlapping_boundaries.append(row_j)
                        geometry = unary_union([geometry, row_j["geometry"]]).buffer(0)
                        merged_names.add(row_j["name"])
                        processed_indices.add(j)
                        still_merging = True

            if not geometry.is_valid:
                geometry = geometry.buffer(0)

            merged_boundary = {
                "geometry": geometry,
                "type": "Merged" if len(merged_names) > 1 else "Single",
                "anchor_settlements": ", ".join(merged_names),
            }
            merged_geometries.append(merged_boundary)

            processed_indices.add(i)

        return gpd.GeoDataFrame(merged_geometries, crs=gdf.crs)

    def split_multipolygons(self, geometry):
        """
        Split MultiPolygon into individual polygons, marking all but the largest as 'created'.

        Parameters
        ----------
        geometry : shapely.geometry.MultiPolygon or Polygon
            The geometry to split.

        Returns
        -------
        list of tuple
            List of (polygon, label) tuples.
        """
        if isinstance(geometry, MultiPolygon):
            sorted_polygons = sorted(geometry.geoms, key=lambda p: p.area, reverse=True)
            return [(sorted_polygons[0], "оригинал")] + [(poly, "создан") for poly in sorted_polygons[1:]]
        return [(geometry, "оригинал")]

    def voronoi_polygons_within_boundaries(
        self, boundary_polygon: Polygon, anchor_towns: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        """
        Create a Voronoi diagram within the given boundary polygon for anchor towns.

        Parameters
        ----------
        boundary_polygon : shapely.geometry.Polygon
            The boundary polygon.
        anchor_towns : geopandas.GeoDataFrame
            GeoDataFrame of anchor towns.

        Returns
        -------
        geopandas.GeoDataFrame
            GeoDataFrame with Voronoi polygons and their sources.
        """

        # Фильтруем точки, оставляя только те, которые внутри границы
        points = [p for p in anchor_towns.geometry if boundary_polygon.contains(p)]

        if len(points) < 2:
            return gpd.GeoDataFrame(columns=["geometry", "source"], crs=anchor_towns.crs)

        if len(points) == 2:
            # Если только 2 точки, делаем разрез границы на две части
            p1, p2 = points

            # Находим середину отрезка
            mid_x, mid_y = (p1.x + p2.x) / 2, (p1.y + p2.y) / 2

            # Направление перпендикуляра (90 градусов к прямой P1-P2)
            dx, dy = p2.x - p1.x, p2.y - p1.y
            perp_dx, perp_dy = -dy, dx  # Вектор перпендикуляра

            # Границы полигона
            min_x, min_y, max_x, max_y = boundary_polygon.bounds

            # Определяем длинную разрезающую линию, которая точно пересекает полигон
            cut_line = LineString(
                [
                    (mid_x - perp_dx * (max_x - min_x), mid_y - perp_dy * (max_y - min_y)),
                    (mid_x + perp_dx * (max_x - min_x), mid_y + perp_dy * (max_y - min_y)),
                ]
            )

            # Проверяем, что cut_line остается LineString
            if cut_line.geom_type == "MultiLineString":
                cut_line = list(cut_line.geoms)[0]  # Берем первый LineString

            # Разрезаем границу полигона
            split_polys = split(boundary_polygon, cut_line)

            if len(split_polys.geoms) == 2:
                return gpd.GeoDataFrame(
                    {"geometry": list(split_polys.geoms), "source": ["оригинал", "оригинал"]}, crs=anchor_towns.crs
                )
            else:
                print("Ошибка разрезания границы на две части.")
                return gpd.GeoDataFrame(columns=["geometry", "source"], crs=anchor_towns.crs)

        # Если точек больше 2, используем стандартную диаграмму Вороного
        points_array = np.array([(p.x, p.y) for p in points])
        region_polys, _ = voronoi_regions_from_coords(points_array, boundary_polygon)

        polygons = []
        sources = []
        for poly in region_polys.values():
            clipped_poly = poly.intersection(boundary_polygon)  # Обрезаем полигоны Вороного
            split_polys = self.split_multipolygons(clipped_poly)
            polygons.extend([p[0] for p in split_polys])
            sources.extend([p[1] for p in split_polys])

        return gpd.GeoDataFrame({"geometry": polygons, "source": sources}, crs=anchor_towns.crs)

    def find_largest_intersection(self, target_poly: Polygon, gdf: gpd.GeoDataFrame):
        """
        Find the polygon in gdf with the largest intersection area with the target polygon.

        Parameters
        ----------
        target_poly : shapely.geometry.Polygon
            The target polygon.
        gdf : geopandas.GeoDataFrame
            GeoDataFrame to search for intersections.

        Returns
        -------
        tuple
            (index, geometry) of the polygon with the largest intersection, or (None, None).
        """
        max_intersection = None
        max_area = 0
        max_index = None
        for idx, row in gdf.iterrows():
            intersection = target_poly.intersection(row.geometry)
            if not intersection.is_empty and intersection.area > max_area:
                max_intersection = row.geometry
                max_area = intersection.area
                max_index = idx
        return max_index, max_intersection

    def cluster_and_merge_created_polygons(self, created_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Cluster and merge all intersecting 'created' polygons.

        Parameters
        ----------
        created_gdf : geopandas.GeoDataFrame
            GeoDataFrame of created polygons.

        Returns
        -------
        geopandas.GeoDataFrame
            GeoDataFrame with merged clusters.
        """
        merged_polygons = []
        processed = set()

        for idx, row in created_gdf.iterrows():
            if idx in processed:
                continue

            cluster = created_gdf[created_gdf.geometry.intersects(row.geometry)].geometry.tolist()
            merged_polygons.append(unary_union(cluster))
            processed.update(created_gdf[created_gdf.geometry.intersects(row.geometry)].index)

        return gpd.GeoDataFrame({"geometry": merged_polygons, "source": "создан"}, crs=created_gdf.crs)

    def merge_created_polygons(self, voronoi_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Merge 'created' polygons and then merge with the 'original' polygon with the largest intersection.

        Parameters
        ----------
        voronoi_gdf : geopandas.GeoDataFrame
            GeoDataFrame with Voronoi polygons.

        Returns
        -------
        geopandas.GeoDataFrame
            GeoDataFrame with merged polygons.
        """
        created_gdf = voronoi_gdf[voronoi_gdf["source"] == "создан"].copy()
        original_gdf = voronoi_gdf[voronoi_gdf["source"] == "оригинал"].copy()

        # Кластеризация созданных полигонов
        clustered_created_gdf = self.cluster_and_merge_created_polygons(created_gdf)

        merged_geometries = []
        original_to_remove = set()

        for _, row in clustered_created_gdf.iterrows():
            buffered_geom = row.geometry.buffer(30)  # Буфер перед объединением с оригиналом
            max_index, max_intersection_poly = self.find_largest_intersection(buffered_geom, original_gdf)

            if max_intersection_poly:
                merged_geometry = unary_union([buffered_geom, max_intersection_poly])
                original_to_remove.add(max_index)
            else:
                merged_geometry = buffered_geom

            merged_geometries.append(merged_geometry)

        # Удаляем оригиналы, которые были объединены
        original_gdf = original_gdf.drop(index=original_to_remove)

        merged_gdf = gpd.GeoDataFrame({"geometry": merged_geometries, "source": "объединён"}, crs=voronoi_gdf.crs)

        return gpd.GeoDataFrame(pd.concat([original_gdf, merged_gdf], ignore_index=True))

    def final_simplification(self, merged_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Shrink geometries by a buffer and merge intersecting polygons into one.

        Parameters
        ----------
        merged_gdf : geopandas.GeoDataFrame
            GeoDataFrame with merged polygons.

        Returns
        -------
        geopandas.GeoDataFrame
            Final simplified GeoDataFrame.
        """
        shrunk_geometries = merged_gdf.geometry.buffer(-30)
        final_geometry = unary_union(shrunk_geometries)

        # Создаем GeoSeries и применяем explode()
        exploded_geometry = gpd.GeoSeries(final_geometry).explode(index_parts=False)
        exploded_geometry = exploded_geometry.buffer(30)
        # Создаем GeoDataFrame с разбиением MultiPolygon на отдельные Polygon
        final_gdf = gpd.GeoDataFrame({"geometry": exploded_geometry, "source": "финальный"}, crs=merged_gdf.crs)

        # Сброс индекса после разбиения
        final_gdf = final_gdf.reset_index(drop=True)

        return final_gdf

    def merge_gdfs_excluding_intersections(
        self, boundaries_gdf: gpd.GeoDataFrame, final_gdf: gpd.GeoDataFrame, anchor_towns: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        """
        Merge boundaries_gdf and final_gdf, excluding intersecting polygons, and assign names from anchor_towns.

        Parameters
        ----------
        boundaries_gdf : geopandas.GeoDataFrame
            Initial boundaries GeoDataFrame.
        final_gdf : geopandas.GeoDataFrame
            Final polygons GeoDataFrame.
        anchor_towns : geopandas.GeoDataFrame
            GeoDataFrame of anchor towns.

        Returns
        -------
        geopandas.GeoDataFrame
            Merged GeoDataFrame with 'geometry' and 'anchor_name'.
        """

        # Удаляем полигоны из boundaries_gdf, которые пересекаются с final_gdf
        filtered_boundaries_gdf = boundaries_gdf[~boundaries_gdf.intersects(final_gdf.unary_union)]

        # Объединяем оставшиеся полигоны с final_gdf
        merged_gdf = gpd.GeoDataFrame(pd.concat([filtered_boundaries_gdf, final_gdf], ignore_index=True))

        # Создаем новый столбец 'name', который будет содержать имя города из anchor_towns
        merged_gdf["anchor_name"] = None

        # Для каждого полигона ищем, какие объекты из anchor_towns находятся внутри
        for idx, row in merged_gdf.iterrows():
            # Выбираем все точки, которые содержатся в полигоне
            towns_in_poly = anchor_towns[anchor_towns.within(row.geometry)]
            if not towns_in_poly.empty:
                # Присваиваем имя первого найденного города
                merged_gdf.at[idx, "anchor_name"] = towns_in_poly.iloc[0]["name"]

        # Оставляем только столбцы 'geometry' и 'name'
        merged_gdf = merged_gdf[["geometry", "anchor_name"]]

        return merged_gdf

    def get_anchor_settlement_boundaries(
        self, towns: gpd.GeoDataFrame, update_df: Optional[pd.DataFrame] = None, time: int = 50
    ) -> gpd.GeoDataFrame:
        """
        Main function to orchestrate the creation, merging, and finalization of anchor settlement boundaries.

        Parameters
        ----------
        towns : geopandas.GeoDataFrame
            GeoDataFrame of towns.
        update_df : pandas.DataFrame or None, optional
            Optional DataFrame for updates.
        time : int, optional
            Maximum travel time. Default is 50.

        Returns
        -------
        geopandas.GeoDataFrame
            GeoDataFrame with finalized anchor settlement boundaries.
        """
        if update_df is not None:
            towns = self.region.get_update_towns_gdf(update_df)

        region_boundary = self.region.region

        local_crs = self.region.region.crs
        towns = towns.to_crs(local_crs)

        boundary_gdf = self._build_anchor_settlement_boundaries(towns, time)

        boundary_gdf = self._simplify_multipolygons(boundary_gdf, towns)

        boundary_gdf = self._merge_intersecting_boundaries(boundary_gdf)

        boundary_gdf = boundary_gdf.explode(index_parts=False)

        # Сбрасываем индекс (если необходимо)
        boundary_gdf = boundary_gdf.reset_index(drop=True)

        boundary_gdf = gpd.overlay(boundary_gdf, region_boundary, how="intersection")

        boundary_gdf = self._simplify_multipolygons(boundary_gdf, towns)

        boundary_gdf["geometry"] = boundary_gdf["geometry"].apply(
            lambda geom: Polygon(geom.exterior) if geom.is_valid else geom
        )

        anchor_towns = towns[towns["is_anchor_settlement"] == True]  # Опорные города

        voronoi_gdfs = []

        for boundary_geom in boundary_gdf.geometry:  # Берем только геометрию границ
            voronoi_gdf = self.voronoi_polygons_within_boundaries(boundary_geom, anchor_towns)
            voronoi_gdfs.append(voronoi_gdf)

        # Объединяем все полученные геодатафреймы в один
        voronoi_gdf = gpd.GeoDataFrame(pd.concat(voronoi_gdfs, ignore_index=True))

        # Объединение полигонов
        merged_voronoi_gdf = self.merge_created_polygons(voronoi_gdf)

        # Финальная обработка
        final_gdf = self.final_simplification(merged_voronoi_gdf)

        result_gdf = self.merge_gdfs_excluding_intersections(boundary_gdf, final_gdf, anchor_towns)

        return result_gdf
