"""Portable GNC rows preserve corrected heads and flows without model mutation."""

from copy import deepcopy

import numpy as np
import pytest

from vorflow._gnc_export import GncCorrections


def _records():
    return [
        dict(
            n=0,
            m=1,
            link=7,
            contributors=[2],
            weights=[0.3214567890123456],
            target=[0.0, 0.25, 0.5],
            residual=0.0,
            kind="horizontal",
        ),
        dict(
            n=1,
            m=0,
            link=7,
            contributors=[3, 4, 5],
            weights=[0.12, 0.23, 0.08],
            target=[1.0, 0.25, 0.5],
            residual=0.0,
            kind="horizontal",
        ),
    ]


def _ghost(n, contributors, weights, heads):
    return heads[n] + np.dot(weights, heads[contributors] - heads[n])


@pytest.mark.parametrize("engine", ["mf6", "mfusg"])
def test_padding_preserves_ghost_heads_and_two_endpoint_corrected_flow(engine):
    records = _records()
    correction = GncCorrections(records=records, nodes=6)
    props = getattr(correction, f"get_gridprops_{engine}_gnc")()
    assert props["numalphaj"] == 3
    assert props["numgnc"] == correction.record_count == 2
    assert correction.corrected_connections == 1
    heads = np.array([10.0, 8.0, 12.0, 4.0, 13.0, 6.0])
    expected, actual = {}, {}
    for record, row in zip(records, props["gncdata"]):
        n, m, *values = row
        contributors, weights = values[:3], values[3:]
        if engine == "mf6":
            n, m = n[0], m[0]
            contributors = [j[0] for j in contributors]
        expected[n] = _ghost(n, record["contributors"], record["weights"], heads)
        actual[n] = _ghost(n, contributors, weights, heads)
        assert sum(weights) == pytest.approx(sum(record["weights"]))
        assert (n, m) == (record["n"], record["m"])
    np.testing.assert_allclose(
        list(actual.values()), list(expected.values()), rtol=0, atol=2e-14
    )
    assert (2 / 3) * (actual[0] - actual[1]) == pytest.approx(
        (2 / 3) * (expected[0] - expected[1]), rel=0, abs=2e-14
    )


def test_collection_copies_inputs_and_repeated_exports_are_independent():
    records = _records()
    before = deepcopy(records)
    unsupported = [dict(n=3, m=4, reason="no support")]
    diagnostics = {"counts": {"aligned": 2}}
    correction = GncCorrections(records, unsupported, diagnostics, nodes=6)
    for _ in range(2):
        mf6 = correction.get_gridprops_mf6_gnc()
        correction.get_gridprops_mfusg_gnc()
        assert correction.records == records == before
    mf6["gncdata"].clear()
    assert len(correction.get_gridprops_mf6_gnc()["gncdata"]) == 2
    records[0]["weights"][0] = 0.1
    unsupported[0]["reason"] = "changed"
    diagnostics["counts"]["aligned"] = 99
    assert correction.records == before
    assert correction.unsupported[0]["reason"] == "no support"
    assert correction.diagnostics["counts"]["aligned"] == 2


def test_empty_collection_skips_packages_and_does_not_overwrite_files(tmp_path):
    correction = GncCorrections(nodes=4)
    assert correction.get_gridprops_mf6_gnc() is None
    assert correction.get_gridprops_mfusg_gnc() is None
    path = tmp_path / "existing.gnc"
    path.write_text("retain existing input")
    assert correction.write_mfusg_gnc(path) is None
    assert path.read_text() == "retain existing input"
    assert correction.write_mfusg_gnc(tmp_path / "absent.gnc") is None
    assert not (tmp_path / "absent.gnc").exists()
    assert correction.corrected_connections == correction.record_count == 0


@pytest.mark.parametrize(
    "change",
    [
        {"n": -1},
        {"m": 6},
        {"n": 0.0},
        {"n": True},
        {"m": 0},
        {"contributors": [6]},
        {"contributors": [2.5]},
        {"contributors": []},
        {"contributors": [[2]]},
        {"weights": []},
        {"weights": [[0.2]]},
        {"weights": [0.2, 0.1]},
        {"weights": [-0.1]},
        {"weights": [float("nan")]},
        {"weights": [float("inf")]},
        {"weights": [0.1 + 0.2j]},
        {"weights": [1.0]},
        {"weights": [1.1]},
        {"contributors": [2, 3], "weights": [0.4, 0.6]},
        {"contributors": [2] * 6, "weights": [0.05] * 6},
    ],
)
def test_invalid_records_are_rejected(change):
    record = _records()[0]
    record.update(change)
    with pytest.raises(ValueError):
        GncCorrections(records=[record], nodes=6)


def test_exports_revalidate_modified_public_records():
    correction = GncCorrections(records=_records(), nodes=6)
    correction.records[0]["weights"] = [1.0]
    with pytest.raises(ValueError, match="positive host"):
        correction.get_gridprops_mf6_gnc()


def test_metadata_and_engine_explicit_defaults():
    correction = GncCorrections(records=_records(), nodes=6, method="k_aware")
    assert correction.experimental and correction.rings == 1
    assert not GncCorrections().experimental
    assert correction.get_gridprops_mf6_gnc()["explicit"] is False
    assert correction.get_gridprops_mf6_gnc(explicit=True)["explicit"] is True
    assert correction.get_gridprops_mfusg_gnc()["isymgncn"] == 1
    assert correction.get_gridprops_mfusg_gnc(explicit=False)["isymgncn"] == 0


def test_native_usg_writer_preserves_double_fractions_and_one_based_ids(tmp_path):
    fraction = 0.12345678901234567
    correction = GncCorrections(
        records=[dict(n=0, m=1, contributors=[2], weights=[fraction])], nodes=3
    )
    path = tmp_path / "m.gnc"
    assert correction.write_mfusg_gnc(path) == path
    lines = path.read_text().splitlines()
    assert lines[1].split() == ["0", "0", "1", "1", "0", "1", "0"]
    n, m, contributor, alpha = lines[2].split()
    assert (n, m, contributor) == ("1", "2", "3")
    assert float(alpha) == fraction
    assert float(np.float32(fraction)) != float(alpha)
    correction.write_mfusg_gnc(path, explicit=False)
    assert path.read_text().splitlines()[1].split()[5] == "0"


def test_optional_flopy_constructor_smoke_for_both_engines(tmp_path):
    flopy = pytest.importorskip("flopy")
    correction = GncCorrections(records=_records(), nodes=6)
    sim = flopy.mf6.MFSimulation(sim_ws=tmp_path / "mf6")
    gwf = flopy.mf6.ModflowGwf(sim, modelname="m")
    mf6 = flopy.mf6.ModflowGwfgnc(gwf, **correction.get_gridprops_mf6_gnc())
    usg = flopy.mfusg.MfUsg(structured=False, model_ws=tmp_path / "usg")
    mfusg = flopy.mfusg.MfUsgGnc(usg, **correction.get_gridprops_mfusg_gnc())
    assert mf6.numgnc.get_data() == mfusg.numgnc == 2
    assert mf6.numalphaj.get_data() == mfusg.numalphaj == 3
