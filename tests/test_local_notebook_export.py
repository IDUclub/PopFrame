"""The export cell writes data directly into one ZIP without service files."""

import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from zipfile import ZipFile

import geopandas as gpd
import pandas as pd
import pytest
from geopandas.testing import assert_geodataframe_equal
from shapely.geometry import box, shape


@pytest.mark.parametrize("notebook_name", ["PopFrame_local.ipynb", "PopFrame.ipynb"])
def test_export_creates_only_one_archive_with_layers_and_tables(tmp_path, notebook_name, monkeypatch):
    notebook = json.loads((Path(__file__).resolve().parents[1] / notebook_name).read_text(encoding="utf-8"))
    source = next("".join(cell["source"]) for cell in notebook["cells"] if cell.get("id") == "popframe-042")
    layer = gpd.GeoDataFrame({"name": ["Территория"]}, geometry=[box(30.1, 59.3, 30.3, 59.4)], crs=4326).to_crs(32636)
    original = layer.copy()
    table = pd.DataFrame({"Население": [1234]}, index=[1874])
    layer_names = {
        "agglomerations_base": "base_agglomerations",
        "agglomerations_population": "population_agglomerations",
        "agglomerations_time": "time_agglomerations",
        "anchor_boundaries": "anchor_boundaries",
        "anchor_boundaries_time": "anchor_boundaries_scenario",
        "territory_location": "territory_scored",
        "hexagons": "hexagons_scored",
    }
    table_names = {
        "scenario_summary": "scenario_summary",
        "town_status": "status_comparison",
        "territory_location": "location_results",
        "development_profiles": "profile_table",
    }
    displayed = []
    downloaded = []
    monkeypatch.setitem(sys.modules, "google.colab", SimpleNamespace(files=SimpleNamespace(download=downloaded.append)))
    namespace = {"DATA_DIR": tmp_path}
    namespace.update({name: layer for name in layer_names.values()})
    namespace.update({name: table for name in table_names.values()})

    # Redirect the imported notebook display function while keeping the real FileLink.
    with patch("IPython.display.display", side_effect=displayed.append):
        exec(compile(source, "archive-cell", "exec"), namespace)

    archive_path = tmp_path / "PopFrame_results.zip"
    assert list(tmp_path.iterdir()) == [archive_path]
    if notebook_name == "PopFrame.ipynb":
        assert downloaded == [str(archive_path)]
        assert displayed == []
    else:
        assert len(displayed) == 1
        assert displayed[0].path == str(archive_path)
        assert downloaded == []
    with ZipFile(archive_path) as archive:
        assert archive.testzip() is None
        assert set(archive.namelist()) == {f"{name}.geojson" for name in layer_names} | {
            f"{name}.csv" for name in table_names
        }
        for name in layer_names:
            feature = json.loads(archive.read(f"{name}.geojson"))["features"][0]
            assert feature["properties"]["name"] == "Территория"
            assert shape(feature["geometry"]).equals_exact(layer.to_crs(4326).geometry.iloc[0], tolerance=1e-9)
        for name in table_names:
            content = archive.read(f"{name}.csv")
            assert content.startswith(b"\xef\xbb\xbf")
            restored = pd.read_csv(io.BytesIO(content), encoding="utf-8-sig")
            assert restored["Население"].tolist() == [1234]
            if name == "town_status":
                assert restored["town_id"].tolist() == [1874]
            if name in {"territory_location", "development_profiles"}:
                assert restored.columns.tolist() == ["Население"]
    assert_geodataframe_equal(layer, original)
