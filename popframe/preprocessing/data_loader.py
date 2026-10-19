"""Load OSM data with resumable Overpass caching and build one Region pickle."""

import json
import os
import re
import time
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import requests
from osmnx._errors import InsufficientResponseError
from scipy.sparse.csgraph import dijkstra
from shapely import Point, Polygon, box, get_parts, union_all
from tqdm import tqdm

from popframe.models.region import Region
from popframe.preprocessing.level_filler import LevelFiller


def _write_new_file(destination, write):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Файл уже существует; выберите другое имя: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent, prefix=".popframe-") as directory:
        temporary = Path(directory) / destination.name
        write(temporary)
        os.link(temporary, destination)
    return destination


PLACE_TYPES = ("city", "town", "village", "hamlet", "isolated_dwelling")
HWY_SPEEDS_KPH = {
    "motorway": 100,
    "motorway_link": 50,
    "trunk": 90,
    "trunk_link": 40,
    "primary": 70,
    "primary_link": 40,
    "secondary": 60,
    "secondary_link": 30,
    "tertiary": 50,
    "tertiary_link": 30,
    "unclassified": 40,
    "residential": 30,
    "living_street": 20,
    "service": 20,
}


def _check_http_response(response, **kwargs):
    # Handle errors before OSMnx 1.6 starts its recursive HTTP retries.
    response.raise_for_status()
    return response


def _download_place_nodes(boundary):
    west, south, east, north = boundary.bounds
    selectors = "".join(f'node["place"={json.dumps(value)}]({south},{west},{north},{east});' for value in PLACE_TYPES)
    response = requests.post(
        ox.settings.overpass_endpoint.rstrip("/") + "/interpreter",
        data={"data": f"[out:json][timeout:{ox.settings.timeout}];({selectors});out body;"},
        headers={"User-Agent": ox.settings.default_user_agent},
        timeout=ox.settings.timeout,
        **ox.settings.requests_kwargs,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("remark"):
        raise InsufficientResponseError(f"Overpass вернул неполный ответ: {payload['remark']}")
    records = [
        {**element.get("tags", {}), "osmid": element["id"], "geometry": Point(element["lon"], element["lat"])}
        for element in payload["elements"]
        if element["type"] == "node"
    ]
    if records:
        places = gpd.GeoDataFrame(records, geometry="geometry", crs=4326)
    else:
        places = gpd.GeoDataFrame({"osmid": pd.Series(dtype="int64")}, geometry=[], crs=4326)
    places["element_type"] = "node"
    places = places.set_index(["element_type", "osmid"])
    return places.loc[places.geometry.covered_by(boundary)].copy()


class RegionDataLoader:
    """Download a region's boundary, named place nodes and directed drive graph.

    Completed downloads and place tiles are reused from cache_dir. Cache
    parameters are checked to prevent mixing regions. OSMnx's global settings
    are restored after each network operation, including failed operations.
    Use sequentially: OSMnx itself configures requests through global settings.
    """

    def __init__(
        self,
        place_query: str | dict,
        *,
        cache_dir: str | Path = "cache/osm",
        overpass_endpoint: str = "https://maps.mail.ru/osm/tools/overpass/api",
        attempts: int = 4,
        timeout: int = 300,
        routing_buffer_m: float = 20_000,
    ):
        if not isinstance(place_query, (str, dict)) or not place_query:
            raise ValueError("Нужен непустой запрос региона для Nominatim.")
        if not isinstance(attempts, int) or attempts < 1 or not np.isfinite(timeout) or timeout <= 0:
            raise ValueError("Нужны положительное число попыток и положительный таймаут.")
        if not np.isfinite(routing_buffer_m) or routing_buffer_m < 0:
            raise ValueError("Буфер дорог должен быть конечным неотрицательным числом в метрах.")
        self.place_query = place_query.copy() if isinstance(place_query, dict) else place_query
        self.cache_dir = Path(cache_dir)
        self.overpass_endpoint = overpass_endpoint
        self.attempts, self.timeout, self.routing_buffer_m = attempts, timeout, routing_buffer_m
        parameters = {
            "place_query": self.place_query,
            "place_types": list(PLACE_TYPES),
            "routing_buffer_m": routing_buffer_m,
            "hwy_speeds_kph": HWY_SPEEDS_KPH,
            "fallback_speed_kph": 40,
        }
        parameters_path = self.cache_dir / "download_parameters.json"
        if parameters_path.is_file():
            if json.loads(parameters_path.read_text(encoding="utf-8")) != parameters:
                raise ValueError("Параметры выборки изменились. Используйте другую папку кэша.")
        else:
            _write_new_file(
                parameters_path,
                lambda path: path.write_text(json.dumps(parameters, ensure_ascii=False, indent=2), encoding="utf-8"),
            )

    @contextmanager
    def _network_settings(self):
        settings = {
            "use_cache": True,
            "cache_folder": str(self.cache_dir / "requests"),
            "timeout": self.timeout,
            "overpass_endpoint": self.overpass_endpoint,
            "overpass_rate_limit": True,
            "default_user_agent": "PopFrame (https://github.com/Mvin8/PopFrame/)",
            "requests_kwargs": {**ox.settings.requests_kwargs, "hooks": {"response": _check_http_response}},
        }
        previous = {name: getattr(ox.settings, name) for name in settings}
        try:
            for name, value in settings.items():
                setattr(ox.settings, name, value)
            yield
        finally:
            for name, value in previous.items():
                setattr(ox.settings, name, value)

    def _retry(self, label, download):
        for attempt in range(1, self.attempts + 1):
            print(f"{label}: попытка {attempt}/{self.attempts}")
            try:
                with self._network_settings():
                    return download()
            except (
                requests.Timeout,
                requests.ConnectionError,
                requests.HTTPError,
                requests.exceptions.JSONDecodeError,
                json.JSONDecodeError,
                InsufficientResponseError,
            ) as exc:
                if isinstance(exc, InsufficientResponseError) and not isinstance(
                    exc.__cause__, (requests.exceptions.JSONDecodeError, json.JSONDecodeError)
                ):
                    raise
                response = getattr(exc, "response", None)
                status = response.status_code if response is not None else None
                detail = (
                    " ".join(re.sub(r"<[^>]+>", " ", response.text).split())[:300]
                    if response is not None
                    else str(exc)[:300]
                )
                reason = f"HTTP {status}: {detail}" if status is not None else f"{type(exc).__name__}: {detail}"
                if status is not None and status not in {408, 429, 500, 502, 503, 504}:
                    raise RuntimeError(f"{label}: {reason}. Проверьте адрес и доступ к сервису.") from exc
                if attempt == self.attempts:
                    raise RuntimeError(
                        f"{label}: не удалось загрузить данные за {self.attempts} попытки. Последняя ошибка: {reason}. "
                        f"Повторите шаг позже; готовые файлы остаются в {self.cache_dir}."
                    ) from exc
                delay = min(30 * 2 ** (attempt - 1), 60)
                if response is not None and response.headers.get("Retry-After", "").isdigit():
                    delay = max(delay, int(response.headers["Retry-After"]))
                print(f"{reason}; повтор через {delay} с.")
                time.sleep(delay)

    def load_boundary(self) -> gpd.GeoDataFrame:
        """Load the OSM administrative boundary and return a valid copy in local UTM."""
        path = self.cache_dir / "boundary_osm.geojson"
        if path.is_file():
            boundary = gpd.read_file(path)
        else:
            boundary = self._retry("Граница", lambda: ox.geocode_to_gdf(self.place_query))
            prepared = prepare_boundary(boundary, repair_geometry=True)
            _write_new_file(path, lambda file: boundary.to_crs(4326).to_file(file, driver="GeoJSON"))
            return prepared
        return prepare_boundary(boundary, repair_geometry=True)

    def load_towns(self) -> gpd.GeoDataFrame:
        """Load place nodes by tiles; preserve unknown population as missing values."""
        region = self.load_boundary()
        boundary = region.to_crs(4326).geometry.union_all()
        path = self.cache_dir / "places_osm.parquet"
        if path.is_file():
            places = gpd.read_parquet(path)
        else:
            west, south, east, north = boundary.bounds
            chunks = []
            tiles = [
                (lon, lat)
                for lon in range(int(np.floor(west)), int(np.ceil(east)))
                for lat in range(int(np.floor(south)), int(np.ceil(north)))
                if box(lon, lat, lon + 1, lat + 1).intersects(boundary)
                and not box(lon, lat, lon + 1, lat + 1).touches(boundary)
            ]
            for lon, lat in tqdm(tiles, desc="Участки OSM"):
                tile_path = self.cache_dir / "place_tiles" / f"{lon}_{lat}.parquet"
                if tile_path.is_file():
                    chunk = gpd.read_parquet(tile_path)
                else:
                    clipped = boundary.intersection(box(lon, lat, lon + 1, lat + 1))
                    chunk = self._retry(f"Населённые пункты [{lon}, {lat}]", lambda: _download_place_nodes(clipped))
                    _write_new_file(tile_path, chunk.to_parquet)
                chunks.append(chunk)
            places = gpd.GeoDataFrame(pd.concat(chunks), crs=4326)
            places = places.loc[~places.index.duplicated()].copy()
            places = places.loc[places.geometry.covered_by(boundary)].copy()
            _write_new_file(path, places.to_parquet)
        towns = prepare_towns(places, region.crs)
        if len(towns) < 2:
            raise ValueError("Нужны минимум два именованных населённых пункта внутри региона.")
        return towns

    def load_graph(self) -> nx.MultiDiGraph:
        """Load a drive graph, filling boundary holes and adding a metric buffer."""
        path = self.cache_dir / "drive.graphml"
        if path.is_file():
            graph = ox.load_graphml(path)
        else:
            region = self.load_boundary()
            geometry = union_all([Polygon(part.exterior) for part in get_parts(region.geometry.union_all())])
            area = gpd.GeoSeries([geometry.buffer(self.routing_buffer_m)], crs=region.crs).to_crs(4326).iloc[0]
            graph = self._retry(
                "Дорожный граф",
                lambda: ox.graph_from_polygon(
                    area,
                    network_type="drive",
                    simplify=True,
                    retain_all=True,
                    truncate_by_edge=True,
                ),
            )
        graph = ox.add_edge_speeds(graph, hwy_speeds=HWY_SPEEDS_KPH, fallback=40)
        graph = ox.add_edge_travel_times(graph)
        if not path.is_file():
            _write_new_file(path, lambda file: ox.save_graphml(graph, filepath=file))
        return graph


def save_region(region: Region, destination: str | Path) -> Path:
    """Atomically save one Region pickle; never overwrite an existing file."""
    return _write_new_file(destination, lambda path: region.to_pickle(str(path)))


def prepare_boundary(boundary: gpd.GeoDataFrame, *, repair_geometry: bool = False) -> gpd.GeoDataFrame:
    """Return an independent boundary in local UTM; optionally repair polygons.

    When repair is requested, line and point remnants of make_valid are
    discarded, matching Region.prepare_for_analysis's polygon repair.
    """
    if not isinstance(boundary, gpd.GeoDataFrame) or boundary.crs is None or boundary.empty:
        raise ValueError("Нужна непустая GeoDataFrame границы с заданной CRS.")
    boundary = boundary[["geometry"]].copy()
    if repair_geometry and boundary.geom_type.isin(["Polygon", "MultiPolygon"]).all():
        boundary.geometry = boundary.geometry.make_valid().map(
            lambda geom: union_all([part for part in get_parts(geom) if part.geom_type in {"Polygon", "MultiPolygon"}])
        )
    if (
        not boundary.geom_type.isin(["Polygon", "MultiPolygon"]).all()
        or not boundary.geometry.is_valid.all()
        or boundary.geometry.is_empty.any()
    ):
        raise ValueError("Граница должна состоять из корректных непустых полигонов.")
    local_crs = boundary.estimate_utm_crs()
    if local_crs is None or local_crs.axis_info[0].unit_name != "metre":
        raise ValueError("Не удалось выбрать метрическую проекцию региона.")
    return boundary[["geometry"]].to_crs(local_crs).copy()


def prepare_towns(features, local_crs):
    """Copy named OSM place nodes into local CRS; retain missing population.

    Features use the OSMnx (element_type, osmid) index. OSM node IDs become town IDs.
    """
    places = features.reset_index()
    towns = places.loc[places["element_type"].eq("node") & places.geom_type.eq("Point")].copy()
    names = towns.get("name:ru", pd.Series(index=towns.index, dtype="string"))
    names = names.astype("string").str.strip().replace("", pd.NA)
    towns["name"] = names.fillna(towns.get("name", pd.Series(index=towns.index, dtype="string")))
    towns["name"] = towns["name"].astype("string").str.strip().replace("", pd.NA)
    towns = towns.dropna(subset=["name"]).set_index("osmid").sort_index()
    towns.index.name = "id"
    if not towns.index.is_unique:
        raise ValueError("Повторяются ID узлов OSM.")
    population = towns.get("population", pd.Series(index=towns.index, dtype="string"))
    population = pd.to_numeric(population.astype("string").str.replace(r"\s+", "", regex=True), errors="coerce")
    valid = np.isfinite(population) & population.gt(0) & population.mod(1).eq(0)
    towns["population"] = population.where(valid).astype("Int64")
    towns["population_source"] = pd.NA
    towns.loc[towns.population.notna(), "population_source"] = "OSM: population"
    towns["id"] = towns.index
    return towns.to_crs(local_crs)


def apply_population_updates(towns, updates):
    """Copy towns and apply positive integer population with a named source.

    Updates are indexed by town ID and contain population and source columns.
    """
    if not updates.index.is_unique or not updates.index.isin(towns.index).all():
        raise ValueError("ID в CSV должны быть уникальны и присутствовать в towns.")
    updates = updates.loc[updates["population"].notna()].copy()
    values = pd.to_numeric(updates["population"], errors="raise")
    if not (np.isfinite(values) & values.gt(0) & values.mod(1).eq(0)).all():
        raise ValueError("Население в CSV должно быть положительным целым числом.")
    sources = updates["source"].astype("string").str.strip()
    if sources.isna().any() or sources.eq("").any():
        raise ValueError("Для дополненного населения укажите источник и год в source.")
    result = towns.copy()
    result.loc[updates.index, "population"] = values.astype("Int64")
    result.loc[updates.index, "population_source"] = sources
    return result


def calculate_accessibility(graph, town_nodes):
    """Return directed minutes between town IDs from road-node assignments.

    Edge travel_time is in seconds; unreachable pairs remain positive infinity.
    """
    if not town_nodes.index.is_unique or not town_nodes.isin(list(graph.nodes)).all():
        raise ValueError("Каждому уникальному ID пункта должен соответствовать узел графа.")
    weights = np.array([data.get("travel_time", np.nan) for _, _, data in graph.edges(data=True)])
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Каждое ребро должно иметь конечное неотрицательное travel_time в секундах.")
    # Параллельные дороги выбираются по минимальному времени, а не суммируются.
    directed = ox.utils_graph.get_digraph(graph, weight="travel_time")
    node_order = list(directed.nodes)
    positions = pd.Series(np.arange(len(node_order)), index=node_order)
    targets = positions.loc[town_nodes.to_numpy()].to_numpy()
    adjacency = nx.to_scipy_sparse_array(directed, nodelist=node_order, weight="travel_time", dtype=float, format="csr")
    matrix = pd.DataFrame(np.inf, index=town_nodes.index, columns=town_nodes.index, dtype=float)
    # Один вектор расстояний за раз: не создаём матрицу всех дорожных узлов.
    for source in tqdm(pd.unique(town_nodes), desc="Исходные узлы"):
        distances = dijkstra(adjacency, directed=True, indices=int(positions.loc[source]))
        matrix.loc[town_nodes.eq(source), :] = distances[targets] / 60.0
    return matrix


def build_region(
    boundary: gpd.GeoDataFrame,
    towns: gpd.GeoDataFrame,
    graph: nx.MultiDiGraph,
    *,
    max_snap_distance_m: float = 2_000,
) -> Region:
    """Validate population, snap towns and build a Region with directed minutes.

    Inputs are not modified. Missing population and remote road attachments
    raise ValueError with town IDs. Unreachable pairs remain positive infinity.
    Travel times exclude congestion and the approach from a town to its road
    node; different towns attached to the same node may have zero travel time.
    """
    region = prepare_boundary(boundary)
    if not isinstance(towns, gpd.GeoDataFrame) or towns.crs is None or len(towns) < 2:
        raise ValueError("Нужны минимум два населённых пункта с заданной CRS.")
    if (
        not towns.index.is_unique
        or not towns.geom_type.eq("Point").all()
        or not towns.geometry.is_valid.all()
        or towns.geometry.is_empty.any()
    ):
        raise ValueError("Нужны уникальные ID и корректные непустые точки населённых пунктов.")
    if not pd.api.types.is_integer_dtype(towns.index.dtype):
        raise ValueError("Идентификаторы населённых пунктов должны быть целыми числами.")
    towns = towns.to_crs(region.crs).copy()
    if not towns.geometry.covered_by(region.geometry.union_all()).all():
        raise ValueError("Населённые пункты должны находиться внутри границы региона.")
    population = pd.to_numeric(towns["population"], errors="coerce")
    valid = (np.isfinite(population) & population.gt(0) & population.mod(1).eq(0)).fillna(False)
    if not valid.all():
        ids = towns.index[~valid].tolist()
        raise ValueError(f"Дополните положительное целое население для {len(ids)} пунктов; ID: {ids[:10]}")
    if towns["name"].isna().any() or towns["name"].astype(str).str.strip().eq("").any():
        raise ValueError("У каждого населённого пункта должно быть имя.")
    if not np.isfinite(max_snap_distance_m) or max_snap_distance_m < 0:
        raise ValueError("Порог привязки должен быть конечным неотрицательным числом в метрах.")
    if (
        not isinstance(graph, nx.MultiDiGraph)
        or "crs" not in graph.graph
        or graph.number_of_nodes() == 0
        or graph.number_of_edges() == 0
    ):
        raise ValueError("Нужен непустой направленный дорожный граф с заданной CRS.")
    towns["population"] = population.astype("int64")
    towns = LevelFiller(towns=towns.copy()).fill_levels()
    projected = ox.project_graph(graph.copy(), to_crs=region.crs)
    nearest, distances = ox.distance.nearest_nodes(
        projected,
        X=towns.geometry.x.to_numpy(),
        Y=towns.geometry.y.to_numpy(),
        return_dist=True,
    )
    too_far = np.asarray(distances) > max_snap_distance_m
    if too_far.any():
        ids = towns.index[too_far].tolist()
        raise ValueError(f"Слишком далёкая привязка к дорогам у {len(ids)} пунктов; ID: {ids[:10]}")
    town_nodes = pd.Series(nearest, index=towns.index)
    matrix = calculate_accessibility(graph, town_nodes)
    if matrix.isna().any().any() or not np.allclose(np.diag(matrix), 0):
        raise ValueError("Расчёт вернул некорректную матрицу времени.")
    return Region(region=region, towns=towns, accessibility_matrix=matrix)
