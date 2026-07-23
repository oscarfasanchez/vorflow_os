"""Validate vorflow wheel/sdist contents and release metadata."""

from __future__ import annotations

import argparse
from email.parser import BytesParser
from email.policy import default
from pathlib import Path, PurePosixPath
import tarfile
import zipfile


FORBIDDEN_DIRECTORIES = {".conda", "benchmarks", "docs", "__pycache__"}
EXPECTED_REQUIREMENTS = {
    "numpy>=1.24",
    "pandas>=1.5",
    "geopandas>=0.13",
    "shapely>=2.0",
    "scipy>=1.10",
    "gmsh>=4.11",
}


def version_from_tag(tag: str) -> str:
    if not tag.startswith("v") or "rc" not in tag:
        raise ValueError(f"expected a release-candidate tag, received {tag!r}")
    return tag[1:]


def forbidden_members(names: list[str]) -> list[str]:
    result = []
    for name in names:
        parts = PurePosixPath(name).parts
        if (
            any(part in FORBIDDEN_DIRECTORIES for part in parts)
            or name.endswith(".code-workspace")
            or name.endswith((".pyc", ".pyo"))
        ):
            result.append(name)
    return result


def _one(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"expected one {pattern} in {directory}, found {matches}")
    return matches[0]


def _require_suffix(names: list[str], suffix: str) -> None:
    if not any(name.endswith(suffix) for name in names):
        raise ValueError(f"archive is missing required path ending in {suffix!r}")


def _normalized_requirement(value: str) -> str:
    return value.replace(" ", "")


def validate_wheel(wheel: Path, expected_version: str) -> None:
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        bad = forbidden_members(names)
        if bad:
            raise ValueError(f"wheel contains forbidden members: {bad}")
        _require_suffix(names, "vorflow/__init__.py")
        _require_suffix(names, ".dist-info/licenses/LICENSE")
        metadata_name = next(
            (name for name in names if name.endswith(".dist-info/METADATA")),
            None,
        )
        if metadata_name is None:
            raise ValueError("wheel has no .dist-info/METADATA")
        metadata = BytesParser(policy=default).parsebytes(archive.read(metadata_name))

    if metadata["Name"].lower().replace("_", "-") != "vorflow":
        raise ValueError(f"unexpected project name: {metadata['Name']}")
    if metadata["Version"] != expected_version:
        raise ValueError(
            f"wheel version {metadata['Version']} does not match {expected_version}"
        )
    if metadata["Requires-Python"] != ">=3.10":
        raise ValueError(f"unexpected Requires-Python: {metadata['Requires-Python']}")
    requirements = {
        _normalized_requirement(value)
        for value in metadata.get_all("Requires-Dist", [])
        if "extra==" not in _normalized_requirement(value)
    }
    if requirements != EXPECTED_REQUIREMENTS:
        raise ValueError(f"unexpected runtime requirements: {requirements}")
    if metadata["License-Expression"] != "MIT":
        raise ValueError("wheel does not declare the MIT SPDX expression")
    if "LICENSE" not in metadata.get_all("License-File", []):
        raise ValueError("wheel metadata does not declare LICENSE")
    if "Oscar Sanchez" not in (metadata["Author-email"] or ""):
        raise ValueError("primary author is missing from wheel metadata")
    if "rhugman" not in (metadata["Author"] or ""):
        raise ValueError("original author is missing from wheel metadata")
    if "Oscar Sanchez" not in (metadata["Maintainer-email"] or ""):
        raise ValueError("maintainer is missing from wheel metadata")
    project_urls = set(metadata.get_all("Project-URL", []))
    expected_urls = {
        "Repository, https://github.com/oscarfasanchez/vorflow_os",
        "Issues, https://github.com/oscarfasanchez/vorflow_os/issues",
        "Changelog, https://github.com/oscarfasanchez/vorflow_os/blob/main/CHANGELOG.md",
    }
    if not expected_urls.issubset(project_urls):
        raise ValueError(f"wheel is missing project URLs: {expected_urls - project_urls}")


def validate_sdist(sdist: Path, expected_version: str) -> None:
    with tarfile.open(sdist, "r:gz") as archive:
        names = archive.getnames()
    bad = forbidden_members(names)
    if bad:
        raise ValueError(f"sdist contains forbidden members: {bad}")
    root = f"vorflow-{expected_version}/"
    for required in (
        "pyproject.toml",
        "README.md",
        "LICENSE",
        "src/vorflow/__init__.py",
    ):
        if root + required not in names:
            raise ValueError(f"sdist is missing {root + required}")


def validate_dist(directory: Path, expected_version: str) -> None:
    wheel = _one(directory, "*.whl")
    sdist = _one(directory, "*.tar.gz")
    validate_wheel(wheel, expected_version)
    validate_sdist(sdist, expected_version)
    print(f"Validated {wheel.name} and {sdist.name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    version_group = parser.add_mutually_exclusive_group(required=True)
    version_group.add_argument("--expected-version")
    version_group.add_argument("--expected-tag")
    args = parser.parse_args()
    expected = (
        version_from_tag(args.expected_tag)
        if args.expected_tag
        else args.expected_version
    )
    validate_dist(args.directory, expected)


if __name__ == "__main__":
    main()
