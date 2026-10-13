import geopandas as gpd
import re
from typing import Optional, Tuple, Dict
from popframe.method.base_method import BaseMethod


class SpatialInequalityCalculator(BaseMethod):
    """
    Calculator for spatial inequality metrics.
    Contains methods for transferring inequality metrics from points to polygons,
    finding the optimal territory, and determining the "best" group.
    """

    def transfer_inequality_metrics_to_polygons(
        self, gdf_cities: gpd.GeoDataFrame, gdf_polygons: gpd.GeoDataFrame, inequality_keyword: str = "Неравенство"
    ) -> Tuple[gpd.GeoDataFrame, Dict[str, Dict[str, float]]]:
        """
        Transfer all columns containing inequality metrics from points (cities) to polygons (agglomerations)
        by averaging over the territory.

        Parameters
        ----------
        gdf_cities : geopandas.GeoDataFrame
            GeoDataFrame with city points and inequality metrics.
        gdf_polygons : geopandas.GeoDataFrame
            GeoDataFrame with polygons (agglomerations).
        inequality_keyword : str, optional
            Keyword to identify inequality metric columns. Default is "Нерaвенство".

        Returns
        -------
        tuple
            (gdf_polygons_with_metrics, stats)
            gdf_polygons_with_metrics : geopandas.GeoDataFrame
                Polygons with new metric columns.
            stats : dict
                Dictionary with mean values inside and outside polygons.
        """
        # 1) Колонки с нужным ключевым словом
        pattern = re.compile(rf".*\b{re.escape(inequality_keyword)}\b.*", re.IGNORECASE)
        metric_cols = [c for c in gdf_cities.columns if isinstance(c, str) and pattern.match(c)]
        if not metric_cols:
            raise KeyError(f"Не найдено колонок с '{inequality_keyword}'")
        # 2) Гео‑объединение точек и полигонов
        cities_with_idx = gpd.sjoin(
            gdf_cities[metric_cols + ["geometry"]], gdf_polygons[["geometry"]], how="left", predicate="within"
        )
        # 3) Усреднение по индексам полигонов
        grouped = cities_with_idx.groupby("index_right")[metric_cols].mean().rename_axis("poly_index")
        # 4) Присоединяем к полигонам
        gdf_polygons_with_metrics = (
            gdf_polygons.reset_index()
            .rename(columns={"index": "poly_index"})
            .merge(grouped.reset_index(), on="poly_index", how="left")
            .set_index("poly_index")
        )
        # 5) Статистика внутри/вне
        inside = cities_with_idx.dropna(subset=["index_right"])
        outside = cities_with_idx[cities_with_idx["index_right"].isna()]
        mean_within = inside[metric_cols].mean().to_dict()
        mean_outside = outside[metric_cols].mean().to_dict()
        stats = {"mean_within": mean_within, "mean_outside": mean_outside}
        return gdf_polygons_with_metrics, stats

    def get_best_territory(
        self,
        gdf: gpd.GeoDataFrame,
        group_name: Optional[str] = None,
        spatial_suffix: str = " - Неравенство",
        default_col: str = "Пространственное неравенство",
        top_n: int = 5,
    ) -> gpd.GeoDataFrame:
        """
        Return up to `top_n` territories (rows) with the minimum spatial inequality value.

        If a group is specified, selects only:
        - all columns with "Неравенство" for this group
        - all other columns NOT containing "Неравенство"
        If no group is specified, returns the first `top_n` by the overall metric.

        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            GeoDataFrame containing geometry and columns with the spatial_suffix.
        group_name : str or None, optional
            Name of the social group without suffix.
        spatial_suffix : str, optional
            Suffix for searching the main metric. Default is " - Неравенство".
        default_col : str, optional
            Name of the overall inequality column. Default is "Пространственное неравенство".
        top_n : int, optional
            How many best territories to return (by ascending metric). Default is 5.

        Returns
        -------
        geopandas.GeoDataFrame
            Up to `top_n` rows with the minimum values.
        """
        if top_n < 1:
            raise ValueError("Параметр top_n должен быть >= 1")
        if group_name:
            primary_col = f"{group_name.strip()}{spatial_suffix}"
            if primary_col not in gdf.columns:
                raise KeyError(f"В GeoDataFrame нет колонки «{primary_col}»")
            top_df = gdf.sort_values(primary_col, ascending=True).head(top_n).copy()
            pattern = re.compile(rf"^{re.escape(group_name.strip())}.*\bНеравенство\b", re.IGNORECASE)
            group_metrics = [c for c in top_df.columns if pattern.match(c)]
            non_metrics = [c for c in top_df.columns if "Неравенство" not in c]
            keep_cols = non_metrics + group_metrics
            return top_df[keep_cols]
        else:
            if default_col not in gdf.columns:
                raise KeyError(f"В GeoDataFrame нет колонки «{default_col}»")
            return gdf.sort_values(default_col, ascending=True).head(top_n).copy()

    def get_best_group_for_territory(
        self,
        gdf: gpd.GeoDataFrame,
        suffix: str = " - Неравенство",
        new_col: str = "Наименьшее неравенство для соц‑группы",
    ) -> gpd.GeoDataFrame:
        """
        Add a column with the name of the social group with the minimum inequality metric.

        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            GeoDataFrame with group inequality metrics.
        suffix : str, optional
            Suffix for group metric columns. Default is " - Неравенство".
        new_col : str, optional
            Name of the new column to add. Default is "Наименьшее неравенство для соц‑группы".

        Returns
        -------
        geopandas.GeoDataFrame
            GeoDataFrame with an added column for the group with the minimum inequality.
        """
        all_cols = [
            c
            for c in gdf.columns
            if isinstance(c, str) and c.endswith(suffix) and c[: -len(suffix)].strip().lower() != "итоговое"
        ]
        if not all_cols:
            raise KeyError(f"Колонки с суффиксом '{suffix}' не найдены")
        df_vals = gdf[all_cols]
        best_col = df_vals.idxmin(axis=1)
        pattern = re.escape(suffix) + r"$"
        best_group = best_col.str.replace(pattern, "", regex=True).str.strip()
        result = gdf.copy()
        result[new_col] = best_group
        return result
