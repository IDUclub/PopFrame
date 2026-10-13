"""Teaching maps using the palettes from PopFrame_old and the anchor example."""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


AGGLOMERATION_COLORS = {1: "#ADD8E6", 2: "#D8BFD8", 3: "#808000", 4: "#FFFF99", 5: "#006400"}
TOWN_COLORS = {
    "Малое сельское поселение": "#F0E68C",
    "Среднее сельское поселение": "#FFD700",
    "Большое сельское поселение": "#FFA07A",
    "Крупное сельское поселение": "#FF8C00",
    "Малый город": "#CD5C5C",
    "Средний город": "#DC143C",
    "Большой город": "#B22222",
    "Крупный город": "#8B0000",
    "Крупнейший город": "#800080",
    "Сверхкрупный город": "#4B0082",
}


def plot_settlement_scheme(region, circles, agglomerations=None, *, title="Схема расселения", ax=None, legend=True):
    """Draw population circles and agglomerations with the original category colors.

    ``circles`` is the result of PopulationFrame.build_circle_frame(), including
    scenario population updates. All layers are displayed in the region CRS.
    Returns the Matplotlib axes.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(20, 10))
    region.region.plot(ax=ax, color="gray", edgecolor="black", alpha=0.3)
    if agglomerations is not None and not agglomerations.empty:
        areas = agglomerations.to_crs(region.crs).sort_values("agglomeration_level", kind="stable")
        areas.plot(
            ax=ax,
            color=areas["agglomeration_level"].map(AGGLOMERATION_COLORS),
            edgecolor="black",
            linewidth=0.6,
        )
    level_order = {level: rank for rank, level in enumerate(TOWN_COLORS)}
    circles = circles.to_crs(region.crs).sort_values("level", key=lambda levels: levels.map(level_order), kind="stable")
    circles.plot(
        ax=ax,
        color=circles["level"].map(TOWN_COLORS),
        edgecolor="black",
        linewidth=0.25,
    )
    if legend:
        if agglomerations is not None:
            agglomeration_legend = ax.legend(
                handles=[Patch(color=color, label=f"Уровень {level}") for level, color in AGGLOMERATION_COLORS.items()],
                loc="upper left",
                bbox_to_anchor=(1, 1),
                title="Агломерации",
            )
            ax.add_artist(agglomeration_legend)
        ax.legend(
            handles=[Patch(color=color, label=level) for level, color in TOWN_COLORS.items()],
            loc="lower left",
            bbox_to_anchor=(1, 0),
            title="Населённые пункты",
        )
    ax.set_title(title)
    ax.set_axis_off()
    return ax


def plot_anchor_settlements(region, zones, towns, *, title="Зоны опорных пунктов"):
    """Draw light-blue zones and green anchor points as in the library example."""
    _, ax = plt.subplots(figsize=(20, 10))
    region.region.plot(ax=ax, color="gray", edgecolor="black", alpha=0.3)
    zones.to_crs(region.crs).plot(ax=ax, facecolor="#ADD8E6", edgecolor="black", linewidth=1)
    points = towns.to_crs(region.crs)
    other = points.loc[~points["is_anchor_settlement"]]
    anchors = points.loc[points["is_anchor_settlement"]]
    if not other.empty:
        other.plot(ax=ax, color="#333333", markersize=8, alpha=0.5)
    anchors.plot(ax=ax, color="green", markersize=50, alpha=0.9, edgecolor="black")
    for row in anchors.itertuples():
        ax.annotate(row.name, (row.geometry.x, row.geometry.y), xytext=(5, 5), textcoords="offset points")
    ax.legend(
        handles=[
            Patch(facecolor="#ADD8E6", edgecolor="black", label="Зоны опорных пунктов"),
            Line2D([], [], marker="o", color="green", markeredgecolor="black", linestyle="", label="Опорные пункты"),
            Line2D([], [], marker="o", color="#333333", linestyle="", label="Населённые пункты"),
        ],
        loc="upper left",
        bbox_to_anchor=(1, 1),
    )
    ax.set_title(title)
    ax.set_axis_off()
    return ax


def explore_territory_scores(hexagons):
    """Explore numeric location scores: RdYlGn for positive scores, black for 0."""
    return hexagons.explore(
        column="score",
        cmap="RdYlGn",
        vmin=1,
        vmax=max(2, hexagons["score"].max()),
        tiles="OpenStreetMap",
        tooltip=["id", "score"],
        popup=["id", "score"],
        legend_kwds={"caption": "Оценка расположения территории"},
        style_kwds={
            "color": "#cccccc",
            "weight": 0.8,
            "fillOpacity": 0.8,
            "style_function": lambda feature: {"fillColor": "black"} if feature["properties"]["score"] == 0 else {},
        },
    )
