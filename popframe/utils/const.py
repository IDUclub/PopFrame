SQUARE_METERS_IN_HECTARE = 10_000

# Geographic and unit conversion constants
METERS_PER_DEGREE = 111_320  # Approx. meters per 1 degree latitude

# Default parameters for time-to-distance conversion and thresholds
# Used by agglomeration and anchor settlement builders
TIME_TO_METERS_FACTOR = 400  # meters per minute of travel time
MIN_CITY_POPULATION_FOR_AGGLO = 15_000

# Infrastructure analysis radii (in meters)
RADIUS_NPP_M = 100_000
RADIUS_HPP_M = 10_000
RADIUS_DEFAULT_INFRA_M = 1_000

# Territory evaluation parameters
BUFFER_NEARBY_SETTLEMENT_M = 10_000
BUFFER_BETWEEN_SETTLEMENTS_M = 30_000
BETWEEN_SETTLEMENT_RATIO = 1.2

# Population density scoring table for TerritoryEvaluation._assess_territory
# Each row defines range thresholds and the resulting score
DENSITY_SCORE_TABLE = [
    {"min_dens": 0, "max_dens": 10, "min_pop": 0, "max_pop": 1_000, "score": 1},
    {"min_dens": 0, "max_dens": 10, "min_pop": 1_000, "max_pop": 5_000, "score": 2},
    {"min_dens": 0, "max_dens": 10, "min_pop": 5_000, "max_pop": float("inf"), "score": 3},
    {"min_dens": 10, "max_dens": 50, "min_pop": 0, "max_pop": 1_000, "score": 2},
    {"min_dens": 10, "max_dens": 50, "min_pop": 1_000, "max_pop": 5_000, "score": 3},
    {"min_dens": 10, "max_dens": 50, "min_pop": 5_000, "max_pop": float("inf"), "score": 4},
    {"min_dens": 50, "max_dens": float("inf"), "min_pop": 0, "max_pop": 1_000, "score": 3},
    {"min_dens": 50, "max_dens": float("inf"), "min_pop": 1_000, "max_pop": 5_000, "score": 4},
    {"min_dens": 50, "max_dens": float("inf"), "min_pop": 5_000, "max_pop": float("inf"), "score": 5},
]

# Land-use dictionaries derived from OpenStreetMap tagging schema
LANDUSE_TAGS = {
    "1.3.1 Процент застройки жилищным строительством": ["residential", "apartments", "detached", "construction"],
    "1.3.2 Процент земель сельскохозяйственного назначения": [
        "farmland",
        "farmyard",
        "orchard",
        "vineyard",
        "greenhouse_horticulture",
        "meadow",
        "plant_nursery",
        "aquaculture",
        "animal_keeping",
        "breeding",
        "grassland",
    ],
    "1.3.3 Процент земель промышленного назначения": ["industrial", "quarry", "landfill"],
    "1.3.4 Процент земель, занятых лесными массивами": ["forest", "wood"],
    "1.3.5 Процент земель специального назначения": ["military", "railway", "cemetery", "landfill", "brownfield"],
    "1.3.6 Процент земель населенных пунктов": ["place_city", "place_town"],
    "1.3.7 Процент земель, занятых особо охраняемыми природными территориями": [
        "national_park",
        "protected_area",
        "nature_reserve",
        "conservation",
    ],
    "1.3.8 Процент земель, занятых водным фондом": ["basin", "reservoir", "water", "salt_pond"],
}

LANDUSE_COLORS = {
    "Застройка жилищным строительством": "blue",
    "Сельскохозяйственные земли": "yellow",
    "Промышленные земли": "gray",
    "Лесные массивы": "green",
    "Земли специального назначения": "brown",
    "Земли населенных пунктов": "orange",
    "Особо охраняемые природные территории": "purple",
    "Водный фонд": "cyan",
    "Территории смежного назначения": "white",
}

LANDUSE_MAPPING = {
    "Процент застройки жилищным строительством": "Застройка жилищным строительством",
    "Процент земель сельскохозяйственного назначения": "Сельскохозяйственные земли",
    "Процент земель промышленного назначения": "Промышленные земли",
    "Процент земель, занятых лесными массивами": "Лесные массивы",
    "Процент земель специального назначения": "Земли специального назначения",
    "Процент земель населенных пунктов": "Земли населенных пунктов",
    "Процент земель, занятых особо охраняемыми природными территориями": "Особо охраняемые природные территории",
    "Процент земель, занятых водным фондом": "Водный фонд",
    "Территории смежного назначения": "Территории смежного назначения",
}
