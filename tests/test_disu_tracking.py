"""Standalone DISV input, point lookup and the cell map file."""

import numpy as np
import pandas as pd
import pytest

from vorflow import disu
from vorflow.utils import grid_to_disv

from test_disu_full import _two_columns
from test_disu_grid import _grid


def test_from_disv_matches_building_from_the_grid():
    grid = _grid(nx=4, ny=3, widths=[1, 2, 1.5, 3])
    rng = np.random.default_rng(5)
    botm = np.array([6.0, 3.0, 0.0])[:, None] + rng.uniform(-1, 1, (3, len(grid)))
    idomain = np.ones(botm.shape, dtype=int)
    idomain[1, 4] = -1
    planar = grid_to_disv(grid)
    direct = disu.DisuGrid(grid, 10.0, botm, idomain).build().get_gridprops_disu6()
    standalone = disu.DisuGrid.from_disv(planar["vertices"], planar["cell2d"], 10.0, botm, idomain)
    props = standalone.build().get_gridprops_disu6()
    for key in ("iac", "ja", "ihc", "cl12", "hwva", "angldegx", "top", "bot", "area"):
        np.testing.assert_allclose(props[key], direct[key], err_msg=key)
    assert list(standalone.cellmap.node_id[:4]) == [0, 1, 2, 3]


def test_locate_and_column_follow_the_collapsed_cells():
    # Column A: 20 | 12 | 4; column B: 20 | 16 | 0, with B's first layer passed through.
    grid = _grid(nx=2, ny=1)
    d = disu.DisuGrid(grid, 20.0, [[12, 16], [4, 0]], [[1, -1], [1, 1]]).build()
    a1, a2, b2 = d.nodemap[0, 0], d.nodemap[1, 0], d.nodemap[1, 1]
    assert d.locate(0.5, 0.5, 15.0) == a1
    assert d.locate(0.5, 0.5, 5.0) == a2
    assert d.locate(1.5, 0.5, 18.0) == b2            # B's pass-through interval went to the cell below.
    assert d.locate(0.5, 0.5, 2.0) == -1             # Below the model.
    assert d.locate(5.0, 0.5, 10.0) == -1            # Outside the grid.
    np.testing.assert_array_equal(d.locate([0.5, 1.5], [0.5, 0.5], [15.0, 1.0]), [a1, b2])
    np.testing.assert_array_equal(d.column(0.5, 0.5), [a1, a2])
    np.testing.assert_array_equal(d.column(1.5, 0.5), [b2])


def test_cellmap_csv_round_trip(tmp_path):
    d = _two_columns(np.array([20.0, 20.0]), [12, 4], [16, 0])
    path = d.save_cellmap(tmp_path / "cells.csv")
    back = pd.read_csv(path)
    assert list(back.columns[:5]) == ["node", "layer", "icpl", "grid", "column"]
    np.testing.assert_array_equal(back.node, d.cellmap.node)
    assert list(back.status) == list(d.cellmap.status.astype(str))
    with pytest.raises(RuntimeError, match="build"):
        disu.DisuGrid(_grid(), 10.0, [0.0]).locate(0, 0, 0)

