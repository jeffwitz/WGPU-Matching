"""Coordinate and masking checks for the external-data benchmark, without GPU."""

import importlib.util
from pathlib import Path

import numpy as np


def load_module(name):
    path = Path(__file__).resolve().parents[1] / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_quantized_speckle_is_not_a_material_hole():
    module = load_module("hudge_strains_data")
    image = np.full((100, 100), 51, np.uint8)
    image[10:50, 20:60] = 100
    image[20, 30] = 51
    image[25:40, 30:45] = 51
    mask = module.material_mask(image)
    assert mask[20, 30]
    assert not mask[30, 35]
    assert not mask[0, 0]


def test_silhouette_proposal_maps_pixel_edges():
    module = load_module("hudge_strains_data")
    reference = np.full((100, 100), 51, np.uint8)
    reference[10:50, 20:60] = 100
    target = np.full((150, 150), 51, np.uint8)
    target[20:100, 40:120] = 100
    centres = np.array([[19.5, 9.5], [59.5, 49.5]])
    parameters, _ = module.silhouette_proposal(reference, target, centres)
    np.testing.assert_allclose(centres + parameters[:, [0, 3]], [[39.5, 19.5], [119.5, 99.5]])
    np.testing.assert_allclose(parameters[:, [1, 5]], 1)
    np.testing.assert_allclose(parameters[:, [2, 4]], 0)


def test_final_photometric_check_uses_native_half_pixel_centres():
    module = load_module("summarize_hudge_strains")
    reference = np.random.default_rng(2).random((64, 64)).astype(np.float32)
    parameters = np.zeros((1, 6))
    correlation, bounds, remap = module.photometric_check(
        reference, reference, np.array([[31.5, 31.5]]), parameters, np.array([True]), 64
    )
    np.testing.assert_allclose(correlation, 1, atol=1e-6)
    np.testing.assert_allclose(remap, reference)
    assert bounds.all()
    parameters[:, 0] = 1
    correlation, bounds, _ = module.photometric_check(
        reference, reference, np.array([[31.5, 31.5]]), parameters, np.array([True]), 64
    )
    assert not bounds.any()
    assert np.isnan(correlation).all()


def test_triangular_truth_uses_explicit_coordinates_and_preserves_void(tmp_path):
    module = load_module("hudge_strains_data")
    # u = [2X + 3Y + 1, -X + 4Y + 2] on one triangle only.
    path = tmp_path / "field.vtu"
    path.write_text('''<VTKFile><UnstructuredGrid><Piece>
      <Points><DataArray format="ascii" NumberOfComponents="3">0 0 0  1 0 0  0 1 0</DataArray></Points>
      <PointData><DataArray format="ascii" Name="DEPL" NumberOfComponents="2">1 2  3 1  4 6</DataArray></PointData>
      <Cells><DataArray Name="connectivity" format="ascii">0 1 2</DataArray>
      <DataArray Name="offsets" format="ascii">3</DataArray>
      <DataArray Name="types" format="ascii">5</DataArray></Cells>
      </Piece></UnstructuredGrid></VTKFile>''')
    matrix = np.diag([10., -20.])
    origin = np.array([.5, 100.5])
    points = np.array([[.25, .25], [.8, .8]])
    parameters, valid = module.evaluate_vtu_truth(path, points @ matrix.T + origin, matrix, origin)
    assert valid.tolist() == [True, False]
    gradient = matrix @ np.array([[2., 3.], [-1., 4.]]) @ np.linalg.inv(matrix)
    np.testing.assert_allclose(parameters[0, [0, 3]], matrix @ np.array([2.25, 2.75]))
    np.testing.assert_allclose(parameters[0, [1, 2, 4, 5]].reshape(2,2), gradient)
    assert np.isnan(parameters[1]).all()
