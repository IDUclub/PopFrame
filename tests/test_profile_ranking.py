"""Detailed profile rankings explain unmet criteria and treat tied results equally."""

from copy import deepcopy

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point, box

from popframe.method.territory_evaluation import TerritoryEvaluation
from popframe.models.region import Region


@pytest.fixture
def evaluation():
    towns = gpd.GeoDataFrame(
        {"name": ["Пункт"], "population": [1000], "level": ["Большое сельское поселение"]},
        geometry=[Point(0, 0)],
        crs=3857,
    )
    return TerritoryEvaluation(
        region=Region(
            region=gpd.GeoDataFrame(geometry=[box(-1000, -1000, 1000, 1000)], crs=towns.crs),
            towns=towns,
            accessibility_matrix=pd.DataFrame([[0.0]]),
        )
    )


def test_detailed_ranking_explains_deficits_and_ties(evaluation):
    criteria = {"Население": 4, "Транспорт": 4, "Экология": 3, "Соц-об": 3, "Инж инф": 2}
    original = deepcopy(criteria)

    results = evaluation.calculate_potential(criteria, detailed=True)
    by_name = {result["profile"]: result for result in results}

    assert len(results) == 10
    assert {result["profile"] for result in results[:2]} == {"Специального назначения", "Транспортная инженерная"}
    assert [result["rank"] for result in results[:3]] == [1, 1, 3]
    assert results[2]["profile"] == "Рекреационная"
    assert results[2]["missing_criteria"] == {"Экология": 1}
    assert results[2]["total_deficit"] == 1
    assert by_name["Промышленная"]["satisfied_criteria"] == 4
    assert by_name["Промышленная"]["missing_criteria"] == {"Инж инф": 2}
    assert by_name["Рекреационная"]["rank"] < by_name["Промышленная"]["rank"]
    assert by_name["Сельско-хозяйственная"]["rank"] < by_name["Жилая застройка - Малоэтажная"]["rank"]
    for result in results:
        assert result["total_criteria"] == 5
        assert result["satisfied_criteria"] == 5 - len(result["missing_criteria"])
        assert result["total_deficit"] == sum(result["missing_criteria"].values())
    assert criteria == original


def test_all_satisfied_profiles_share_first_place(evaluation):
    criteria = dict.fromkeys(["Население", "Транспорт", "Экология", "Соц-об", "Инж инф"], 10)

    results = evaluation.calculate_potential(criteria, detailed=True)

    assert all(result["rank"] == 1 and result["satisfied_criteria"] == 5 for result in results)
    assert all(result["missing_criteria"] == {} and result["total_deficit"] == 0 for result in results)


def test_more_transport_points_reduce_the_explained_deficit(evaluation):
    criteria = {"Население": 4, "Транспорт": 4, "Экология": 3, "Соц-об": 3, "Инж инф": 2}
    before = {row["profile"]: row for row in evaluation.calculate_potential(criteria, detailed=True)}
    criteria["Транспорт"] = 5

    after = {row["profile"]: row for row in evaluation.calculate_potential(criteria, detailed=True)}

    assert "Транспорт" in before["Общественно-деловая"]["missing_criteria"]
    assert "Транспорт" not in after["Общественно-деловая"]["missing_criteria"]
    assert after["Общественно-деловая"]["satisfied_criteria"] == before["Общественно-деловая"]["satisfied_criteria"] + 1
    assert after["Общественно-деловая"]["total_deficit"] == before["Общественно-деловая"]["total_deficit"] - 1


def test_default_result_keeps_its_existing_tuple_format(evaluation):
    criteria = {"Население": 4, "Транспорт": 4, "Экология": 3, "Соц-об": 3, "Инж инф": 2}

    results = evaluation.calculate_potential(criteria)

    assert len(results) == 10
    assert all(isinstance(row, tuple) and len(row) == 2 and len(row[1]) == 2 for row in results)
    assert dict(results)["Специального назначения"] == (5, 7)


@pytest.mark.parametrize("invalid", [{}, {"Транспорт": 4}, {"Транспортт": 4}])
def test_detailed_ranking_requires_the_five_named_criteria(evaluation, invalid):
    with pytest.raises(ValueError, match="критери"):
        evaluation.calculate_potential(invalid, detailed=True)
