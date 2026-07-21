from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]


def test_release_metadata_is_complete():
    with (ROOT / "pyproject.toml").open("rb") as stream:
        data = tomllib.load(stream)

    project = data["project"]
    assert data["build-system"]["requires"] == ["setuptools>=77.0.3"]
    assert project["version"] == "0.1.0rc1"
    assert project["license"] == "MIT"
    assert project["license-files"] == ["LICENSE"]
    assert project["authors"] == [
        {"name": "Oscar Sanchez", "email": "oscarfasanchez@gmail.com"},
        {"name": "rhugman"},
    ]
    assert project["maintainers"] == [
        {"name": "Oscar Sanchez", "email": "oscarfasanchez@gmail.com"}
    ]
    assert project["dependencies"] == [
        "numpy>=1.24",
        "pandas>=1.5",
        "geopandas>=0.13",
        "shapely>=2.0",
        "scipy>=1.10",
        "gmsh>=4.11",
    ]
    assert "License :: OSI Approved :: MIT License" not in project["classifiers"]


def test_source_fallback_is_not_a_duplicate_release_version():
    source = (ROOT / "src" / "vorflow" / "__init__.py").read_text(
        encoding="utf-8"
    )
    assert '__version__ = "0+unknown"' in source
    assert '__version__ = "0.0.2"' not in source
