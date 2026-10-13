"""Notebook map for selecting a territory by its original numeric ID."""

import json
from html import escape

from ipyleaflet import GeoJSON, LayersControl, Map, ScaleControl, basemaps
from ipywidgets import HTML, Dropdown, Layout, VBox


class HexagonSelector(VBox):
    """Select a grid cell by clicking the map or choosing its ID in the list.

    ``value`` returns the selected numeric ID for filtering the original grid.
    Selection updates the map; rerun downstream notebook cells to recalculate.
    Input geometries are not modified. Requires the ``popframe[notebook]`` extra.
    """

    def __init__(self, hexagons, region, selected_id=1632):
        try:
            from google.colab import output
        except ImportError:
            pass
        else:
            output.enable_custom_widget_manager()

        if hexagons.empty or not hexagons["id"].is_unique:
            raise ValueError("Нужна непустая сетка с уникальными ID.")
        self._hexagons = hexagons[["id", "geometry"]].to_crs(4326).copy()
        if selected_id not in self._hexagons["id"].values:
            raise ValueError(f"Гексагон {selected_id} отсутствует в сетке.")

        self.dropdown = Dropdown(
            options=sorted(int(value) for value in self._hexagons["id"]),
            value=int(selected_id),
            description="ID гекса:",
            layout=Layout(width="280px"),
        )
        self.caption = HTML()
        self.hover_info = HTML(value="Наведите курсор на гекс или населённый пункт.")
        self.grid_layer = GeoJSON(
            data=json.loads(self._hexagons.to_json()),
            name="Гексагональная сетка",
            style={"color": "#697c8e", "weight": 1, "fillColor": "#e5edf4", "fillOpacity": 0.12},
            hover_style={"color": "#28486d", "weight": 2, "fillOpacity": 0.3},
        )
        self.selection_layer = GeoJSON(
            name="Выбранный гексагон",
            style={"color": "#893434", "weight": 3, "fillColor": "#893434", "fillOpacity": 0.65},
        )
        boundary = region.region.to_crs(4326)
        west, south, east, north = boundary.total_bounds
        self.map = Map(
            center=((south + north) / 2, (west + east) / 2),
            zoom=7,
            basemap=basemaps.OpenStreetMap.Mapnik,
            scroll_wheel_zoom=True,
            layout=Layout(width="100%", height="580px"),
        )
        self.map.add(self.grid_layer)
        # Simplification affects the display only, keeping widget messages small.
        boundary = boundary.set_geometry(boundary.geometry.simplify(0.001, preserve_topology=True))
        self.map.add(
            GeoJSON(
                data=json.loads(boundary.to_json()),
                name="Граница региона",
                style={"color": "#28486d", "weight": 2, "fillOpacity": 0, "interactive": False},
            )
        )
        towns = region.get_towns_gdf()[["name", "population", "geometry"]].to_crs(4326)
        town_layer = GeoJSON(
            data=json.loads(towns.to_json()),
            name="Населённые пункты",
            point_style={"radius": 3, "color": "#333333", "weight": 0, "fillColor": "#333333", "fillOpacity": 0.6},
        )
        town_layer.on_hover(self._hover_town)
        self.map.add(town_layer)
        self.map.add(self.selection_layer)
        self.map.add(LayersControl(position="topright"))
        self.map.add(ScaleControl(position="bottomleft"))
        self.grid_layer.on_click(self._select_feature)
        self.grid_layer.on_hover(self._hover_hexagon)
        self.selection_layer.on_hover(self._hover_hexagon)
        self.dropdown.observe(self._update_selection, names="value")
        self._update_selection()
        super().__init__([self.dropdown, self.caption, self.map, self.hover_info])

    @property
    def value(self) -> int:
        """The selected numeric ID, ready for ``hexagons['id'].eq(value)``."""
        return int(self.dropdown.value)

    def _select_feature(self, feature, **kwargs):
        self.dropdown.value = int(feature["properties"]["id"])

    def _update_selection(self, change=None):
        selected = self._hexagons.loc[self._hexagons["id"].eq(self.value)]
        self.selection_layer.data = json.loads(selected.to_json())
        self.caption.value = f"<b>Выбран гексагон {self.value}</b>. Выполните следующие ячейки для расчёта."

    def _hover_hexagon(self, feature, **kwargs):
        self.hover_info.value = f"Гексагон <b>{int(feature['properties']['id'])}</b> — нажмите для выбора."

    def _hover_town(self, feature, **kwargs):
        properties = feature["properties"]
        self.hover_info.value = f"<b>{escape(str(properties['name']))}</b>: {int(properties['population']):,} чел."
