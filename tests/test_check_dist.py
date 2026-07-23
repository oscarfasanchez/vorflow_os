import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_dist.py"
SPEC = importlib.util.spec_from_file_location("check_dist", SCRIPT)
check_dist = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(check_dist)


def test_version_from_tag():
    assert check_dist.version_from_tag("v0.1.0rc1") == "0.1.0rc1"


def test_version_from_tag_rejects_production_tag():
    try:
        check_dist.version_from_tag("v0.1.0")
    except ValueError as error:
        assert "release-candidate" in str(error)
    else:
        raise AssertionError("production tag was accepted")


def test_forbidden_members_are_reported():
    members = [
        "vorflow-0.1.0rc1/src/vorflow/__init__.py",
        "vorflow-0.1.0rc1/docs/private-plan.md",
        "vorflow-0.1.0rc1/src/vorflow/vorflow.code-workspace",
    ]
    assert check_dist.forbidden_members(members) == [members[1], members[2]]
