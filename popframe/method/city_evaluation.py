import geopandas as gpd
import pandas as pd
import json


class CityPopulationScorer:
    """
    Calculates population-based scores for hexagonal grid cells based on municipal data.

    This class takes two GeoDataFrames:
      - gdf_mo: municipal polygons with population attributes.
      - gdf_hex: hexagonal grid cells with unique identifiers.

    The scoring is performed by:
      1. Computing area and density for each municipality.
      2. Assigning each hexagon to the municipality it overlaps the most.
      3. Normalizing population and density values across all hexagons.
      4. Aggregating normalized values into a score between 1 and 5.
      5. Providing textual interpretations for each score.
    """

    INTERPRETATIONS = {
        0: "Территория имеет нулевые показатели численности и плотности населения, что затрудняет развитие инфраструктуры и экономики.",
        1: "Территория имеет низкиие показателяи численности и плотности населения, что ограничивает возможности развития инфраструктуры и экономики.",
        2: "Территория отличается относительно низкой численностью и плотностью населения, что ограничивает возможности развитие инфраструктуры и экономики.",
        3: "Территория имеет средние показатели численности и плотности населения, что создаёт возможность развитие инфраструктуры и экономики.",
        4: "Территория имеет высокие показатели численности и плотности населения, что способствует развитие инфраструктуры и экономики.",
        5: "Территория имеет очень высокими показателями численности и плотности, что указывает на высокий потенциал развития инфраструктуры и экономики.",
    }

    def __init__(self, gdf_mo: gpd.GeoDataFrame, gdf_hex: gpd.GeoDataFrame, target_crs: int = 3857):
        """
        Initializes the HexPopulationScorer.

        Args:
            gdf_mo (gpd.GeoDataFrame): GeoDataFrame of municipal polygons. Must contain columns
                'geometry', 'territory_id', and 'population'.
            gdf_hex (gpd.GeoDataFrame): GeoDataFrame of hexagonal grid cells. Must contain column
                'hexagon_id'.
            target_crs (int, optional): EPSG code for the metric CRS used to calculate areas.
                Defaults to 3857.
        """
        # Copy inputs to avoid modifying external data
        self.gdf_mo = gdf_mo.copy()
        self.gdf_hex = gdf_hex.copy()
        self.target_crs = target_crs
        self.output = None

    def compute_mo_density(self):
        """
        Computes area and population density for each municipality.

        Converts gdf_mo to a metric CRS, calculates polygon areas in square meters
        (and then in square kilometers), and computes population density (people per km²).

        Modifies:
            self.gdf_mo: Adds columns 'area_m2', 'area_km2', and 'density_mo'.
        """
        self.gdf_mo = self.gdf_mo.to_crs(epsg=self.target_crs)
        self.gdf_mo["area_m2"] = self.gdf_mo.geometry.area
        self.gdf_mo["area_km2"] = self.gdf_mo["area_m2"] / 1_000_000
        self.gdf_mo["density_mo"] = self.gdf_mo["population"] / self.gdf_mo["area_km2"]

    def assign_hex_to_mo(self):
        """
        Assigns each hexagon to the municipality it overlaps the most.

        Steps:
          1. Reprojects hexagons to the same metric CRS as municipalities.
          2. Computes the geometric intersection between each hexagon and municipality.
          3. For each hexagon, selects the municipality with the largest intersection area.
          4. Merges population and density data from the selected municipality back into gdf_hex.

        Modifies:
            self.gdf_hex: Adds columns 'territory_id', 'population', and 'density'.
        """
        self.gdf_hex = self.gdf_hex.to_crs(epsg=self.target_crs)

        mo_small = self.gdf_mo[["geometry", "territory_id", "population", "density_mo"]]

        hex_mo = gpd.overlay(self.gdf_hex[["geometry", "hexagon_id"]], mo_small, how="intersection")
        hex_mo["inter_area"] = hex_mo.geometry.area

        idx = hex_mo.groupby("hexagon_id")["inter_area"].idxmax()
        hex_max = hex_mo.loc[idx, ["hexagon_id", "territory_id", "population", "density_mo"]]

        self.gdf_hex = self.gdf_hex.merge(hex_max, on="hexagon_id", how="left")
        self.gdf_hex = self.gdf_hex.rename(columns={"density_mo": "density"})

    def normalize_and_score(self):
        """
        Нормализует population и density, считает combined_norm,
        присваивает score = 1–5 для ячеек с данными и 0 для ячеек без данных.
        """
        # 1. Нормируем population
        pop_min, pop_max = self.gdf_hex["population"].min(), self.gdf_hex["population"].max()
        if pop_max == pop_min:
            self.gdf_hex["norm_pop"] = 0.5
        else:
            self.gdf_hex["norm_pop"] = (self.gdf_hex["population"] - pop_min) / (pop_max - pop_min)

        # 2. Нормируем density
        dens_min, dens_max = self.gdf_hex["density"].min(), self.gdf_hex["density"].max()
        if dens_max == dens_min:
            self.gdf_hex["norm_dens"] = 0.5
        else:
            self.gdf_hex["norm_dens"] = (self.gdf_hex["density"] - dens_min) / (dens_max - dens_min)

        # Заменяем NaN (в том числе от ячеек без данных) на 0
        self.gdf_hex["norm_pop"] = self.gdf_hex["norm_pop"].fillna(0)
        self.gdf_hex["norm_dens"] = self.gdf_hex["norm_dens"].fillna(0)

        # 3. Вычисляем сырое значение
        self.gdf_hex["combined_raw"] = (self.gdf_hex["norm_pop"] + self.gdf_hex["norm_dens"]) / 2

        # 4. Мин–макс нормируем combined_raw
        raw_min, raw_max = (self.gdf_hex["combined_raw"].min(), self.gdf_hex["combined_raw"].max())
        if raw_max == raw_min:
            self.gdf_hex["combined_norm"] = 0.5
        else:
            self.gdf_hex["combined_norm"] = (self.gdf_hex["combined_raw"] - raw_min) / (raw_max - raw_min)

        self.gdf_hex["combined_norm"] = self.gdf_hex["combined_norm"].fillna(0)

        # 5. Присваиваем оценку:
        #    – для ячеек с данными: масштабируем в [1;5]
        #    – для ячеек без данных (вода): оставляем 0
        # 5.1 Сначала проставляем всем 1–5
        self.gdf_hex["score"] = (self.gdf_hex["combined_norm"] * 4 + 1).round().clip(lower=1, upper=5).astype(int)

        # 5.2 Выставляем 0 там, где нет ни population, ни density
        mask_no_data = self.gdf_hex["population"].isna() & self.gdf_hex["density"].isna()
        self.gdf_hex.loc[mask_no_data, "score"] = 0

    def assign_interpretations(self):
        """
        Assigns textual interpretations to each hexagon's score.

        For each hexagon:
          - If 'score' is not NaN, retrieves the corresponding interpretation string.
          - If 'score' is NaN, sets interpretation to None.

        Modifies:
            self.gdf_hex: Adds column 'interpretation'.
        """
        self.gdf_hex["interpretation"] = self.gdf_hex["score"].apply(
            lambda v: CityPopulationScorer.INTERPRETATIONS[int(v)] if pd.notna(v) else None
        )

    def generate_output(self) -> list:
        """
        Constructs the final output list of dictionaries for JSON serialization.

        Each dictionary contains:
          - 'hexagon_id' (unique identifier of the hexagon)
          - 'project' (always None)
          - 'average_population_density' (rounded to one decimal place, or None)
          - 'total_population' (integer population, or None)
          - 'score' (float score in [1, 5], or None)
          - 'interpretation' (string or None)

        Returns:
            list: A list of dictionaries ready for JSON export.
        """
        output_list = []
        for _, row in self.gdf_hex.iterrows():
            output_list.append(
                {
                    "hexagon_id": row["hexagon_id"],
                    "project": None,
                    "average_population_density": (round(row["density"], 1) if pd.notna(row["density"]) else None),
                    "total_population": (int(row["population"]) if pd.notna(row["population"]) else None),
                    "score": (float(row["score"]) if pd.notna(row["score"]) else None),
                    "interpretation": row["interpretation"],
                }
            )
        self.output = output_list
        return output_list

    def run(self) -> list:
        """
        Executes the full scoring workflow and returns the result.

        Steps executed in order:
          1. compute_mo_density
          2. assign_hex_to_mo
          3. normalize_and_score
          4. assign_interpretations
          5. generate_output

        Returns:
            list: The final list of dictionaries for JSON serialization.
        """
        self.compute_mo_density()
        self.assign_hex_to_mo()
        self.normalize_and_score()
        self.assign_interpretations()
        return self.generate_output()
