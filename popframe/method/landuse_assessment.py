import osmnx as ox
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from concurrent.futures import ThreadPoolExecutor, as_completed
from retrying import retry
import pandas as pd
import matplotlib.pyplot as plt
import contextily as ctx
from matplotlib.lines import Line2D
from .base_method import BaseMethod
import math
from tqdm import tqdm
from typing import Dict, Optional

from popframe.utils.const import LANDUSE_TAGS, LANDUSE_COLORS, LANDUSE_MAPPING


class LandUseAssessment(BaseMethod):
    @retry(stop_max_attempt_number=5, wait_fixed=2000)
    def fetch_osm_data(self, polygon, tags):
        """
        Fetch OSM data for a given polygon and tags.

        Parameters
        ----------
        polygon : shapely.geometry.Polygon
            The polygon to fetch data for.
        tags : dict
            Dictionary of OSM tags to filter features.

        Returns
        -------
        geopandas.GeoDataFrame
            GeoDataFrame with OSM features.
        """
        return ox.features_from_polygon(polygon, tags=tags)

    def get_landuse_data(
        self, territories: Optional[gpd.GeoDataFrame] = None, landuse_tags: Optional[Dict[str, list]] = None
    ) -> gpd.GeoDataFrame:
        """
        Retrieve and process land use data for given territories.

        Parameters
        ----------
        territories : geopandas.GeoDataFrame
            GeoDataFrame of territories.

        Returns
        -------
        geopandas.GeoDataFrame
            GeoDataFrame with land use indicators and geometries.
        """
        territories_gdf = territories if territories is not None else self.region.get_territories_gdf()
        landuse_tags = landuse_tags or LANDUSE_TAGS

        unique_tags = set(tag for tags in landuse_tags.values() for tag in tags)
        tag_filters = (
            [{"landuse": tag} for tag in unique_tags if not tag.startswith("place_")]
            + [{"place": "city"}, {"place": "town"}]
            + [{"natural": "water"}, {"natural": "wood"}, {"natural": "grassland"}]
        )

        def process_polygon(polygon):
            """
            Process a single polygon to extract land use data.

            Parameters
            ----------
            polygon : shapely.geometry.Polygon
                The polygon to process.

            Returns
            -------
            list of dict
                List of dictionaries with indicator and geometry.
            """
            unique_gdfs = {}
            with ThreadPoolExecutor() as executor:
                future_to_tag_filter = {
                    executor.submit(self.fetch_osm_data, polygon, tag_filter): tag_filter for tag_filter in tag_filters
                }
                for future in tqdm(
                    as_completed(future_to_tag_filter), total=len(tag_filters), desc="Processing landuse tags"
                ):
                    tag_filter = future_to_tag_filter[future]
                    try:
                        gdf = future.result()
                        if not gdf.empty:
                            gdf = gdf.set_geometry("geometry")
                            if gdf.crs is None:
                                gdf.set_crs(epsg=4326, inplace=True)
                            unique_gdfs[frozenset(tag_filter.items())] = gdf
                    except Exception:
                        pass

            combined_geometries = []
            for category, tags in landuse_tags.items():
                category_geoms = []
                for tag in tags:
                    if tag.startswith("place_"):
                        tag_filter = {"place": "city" if tag == "place_city" else "town"}
                    else:
                        tag_filter = {"landuse": tag} if tag not in ["wood", "water", "grassland"] else {"natural": tag}

                    gdf = unique_gdfs.get(frozenset(tag_filter.items()))
                    if gdf is not None:
                        category_geoms.extend(gdf.geometry)

                if category_geoms:
                    valid_geoms = [geom for geom in category_geoms if geom.is_valid]
                    valid_geoms = [geom.buffer(0) for geom in valid_geoms]
                    category_union = unary_union(valid_geoms)
                    category_intersect = category_union.intersection(polygon)
                    if isinstance(category_intersect, (Polygon, MultiPolygon)):
                        combined_geometries.append({"indicator": category, "geometry": category_intersect})
                    elif isinstance(category_intersect, GeometryCollection):
                        for geom in category_intersect.geoms:
                            if isinstance(geom, (Polygon, MultiPolygon)):
                                combined_geometries.append({"indicator": category, "geometry": geom})

            return combined_geometries

        all_combined_geometries = []
        with ThreadPoolExecutor() as executor:
            future_to_polygon = {
                executor.submit(process_polygon, polygon): polygon for polygon in territories_gdf.geometry
            }
            for future in as_completed(future_to_polygon):
                all_combined_geometries.extend(future.result())

        landuse_gdf = gpd.GeoDataFrame(all_combined_geometries, columns=["indicator", "geometry"], crs="EPSG:4326")

        def adjust_geometries(main_indicator, other_gdf, landuse_gdf):
            """
            Adjust geometries for a main indicator by removing overlaps with others.

            Parameters
            ----------
            main_indicator : str
                The main indicator to adjust.
            other_gdf : geopandas.GeoDataFrame
                GeoDataFrame of other indicators.
            landuse_gdf : geopandas.GeoDataFrame
                Full land use GeoDataFrame.

            Returns
            -------
            geopandas.GeoDataFrame
                Adjusted GeoDataFrame.
            """
            main_gdf = landuse_gdf[landuse_gdf["indicator"] == main_indicator]
            if not main_gdf.empty:
                other_union = unary_union(other_gdf.geometry)
                adjusted_geometries = [
                    geom.difference(other_union)
                    for geom in main_gdf.geometry
                    if not geom.difference(other_union).is_empty
                ]
                if adjusted_geometries:
                    main_gdf["geometry"] = adjusted_geometries
                    return pd.concat([other_gdf, main_gdf], ignore_index=True)
            return landuse_gdf

        landuse_gdf = adjust_geometries(
            "1.3.5 Процент земель специального назначения",
            landuse_gdf[~landuse_gdf["indicator"].isin(["1.3.5 Процент земель специального назначения"])],
            landuse_gdf,
        )
        landuse_gdf = adjust_geometries(
            "1.3.1 Процент застройки жилищным строительством",
            landuse_gdf[~landuse_gdf["indicator"].isin(["1.3.1 Процент застройки жилищным строительством"])],
            landuse_gdf,
        )
        landuse_gdf = adjust_geometries(
            "1.3.6 Процент земель населенных пунктов",
            landuse_gdf[~landuse_gdf["indicator"].isin(["1.3.6 Процент земель населенных пунктов"])],
            landuse_gdf,
        )

        region_area_km2 = territories_gdf.to_crs(territories_gdf.estimate_utm_crs()).geometry.area.sum() / 1e6
        landuse_gdf["urbanization"] = (
            landuse_gdf.to_crs(territories_gdf.estimate_utm_crs()).geometry.area / 1e6 / region_area_km2 * 100
        )

        all_landuse_union = unary_union(landuse_gdf.geometry)
        total_polygon = unary_union(territories_gdf.geometry)
        other_landuse = total_polygon.difference(all_landuse_union)

        if not other_landuse.is_empty:
            other_landuse_gdf = gpd.GeoDataFrame(
                [{"indicator": "1.3.9 Территории смежного назначения", "geometry": other_landuse}], crs="EPSG:4326"
            )
            other_landuse_gdf["urbanization"] = (
                other_landuse_gdf.to_crs(territories_gdf.estimate_utm_crs()).geometry.area / 1e6 / region_area_km2 * 100
            )
            landuse_gdf = pd.concat([landuse_gdf, other_landuse_gdf], ignore_index=True)

        grouped_landuse_gdf = (
            landuse_gdf.groupby("indicator")
            .agg({"geometry": lambda x: unary_union(x), "urbanization": "sum"})
            .reset_index()
        )

        results = []
        total_value = grouped_landuse_gdf["urbanization"].sum()

        normalization_factor = 100 / total_value if total_value > 100 else 1

        for index, row in grouped_landuse_gdf.iterrows():
            key = row["indicator"]
            value = row["urbanization"] * normalization_factor
            rounded_value = math.floor(value * 1000) / 1000.0
            results.append(
                {
                    "№ п/п": key.split(" ")[0],
                    "Название хранимое": " ".join(key.split(" ")[1:]) if " " in key else key,
                    "ед.изм.": "%",
                    "Значение": rounded_value,
                    "Источник": "modeled",
                    "Период": 2024,
                    "geometry": row["geometry"],
                }
            )
        results_gdf = gpd.GeoDataFrame(results, crs="EPSG:4326")

        return results_gdf

    def plot_landuse(
        self,
        region_gdf: gpd.GeoDataFrame,
        landuse_gdf: gpd.GeoDataFrame,
        *,
        colors: Optional[Dict[str, str]] = None,
        landuse_mapping: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Plot the land use data on a map with region boundaries and legend.

        Parameters
        ----------
        region_gdf : geopandas.GeoDataFrame
            GeoDataFrame of the region boundary.
        landuse_gdf : geopandas.GeoDataFrame
            GeoDataFrame with land use data.

        Returns
        -------
        None
        """
        if region_gdf.crs is None:
            region_gdf.set_crs(epsg=4326, inplace=True)
        crs = region_gdf.estimate_utm_crs()
        region_gdf_utm = region_gdf.to_crs(crs)  # Преобразование региона в UTM

        fig, ax = plt.subplots(figsize=(10, 10))  # Увеличим размер фигуры
        ax.set_axis_off()  # Отключение осей

        region_gdf_utm.boundary.plot(ax=ax, linewidth=1, color="black", label="Граница региона")

        colors = colors or LANDUSE_COLORS
        landuse_mapping = landuse_mapping or LANDUSE_MAPPING

        for key, label in landuse_mapping.items():
            gdf = landuse_gdf[landuse_gdf["Название хранимое"] == key]
            if not gdf.empty:
                if gdf.crs is None:
                    gdf.set_crs(epsg=4326, inplace=True)
                gdf_utm = gdf.to_crs(crs)  # Преобразование каждого типа земельного использования в UTM
                gdf_utm.plot(ax=ax, color=colors.get(label, "gray"), alpha=0.5, label=label)

        # Добавление подложки карты
        ctx.add_basemap(ax, crs=crs, source=ctx.providers.CartoDB.Positron)

        # Создание пользовательских меток для легенды
        legend_elements = [
            Line2D([0], [0], marker="o", color="w", label=label, markerfacecolor=color, markersize=10)
            for label, color in colors.items()
        ]

        ax.legend(
            handles=legend_elements, loc="upper left", bbox_to_anchor=(1, 1)
        )  # Перемещение легенды в верхнюю часть справа
        plt.show()
