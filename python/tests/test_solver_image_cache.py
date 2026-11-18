"""Test exact image-cache round trips and corruption detection."""

import numpy as np
import pytest
from solver_image_cache import read_solver_images, write_solver_images


def test_image_cache_roundtrip_and_archive_binding(tmp_path):
    archive = tmp_path / "case.npz"
    archive.write_bytes(b"window data")
    ref = np.arange(63, dtype=np.uint8).reshape(7, 9)
    target = np.flip(ref, axis=1).copy()
    cache = tmp_path / "images"
    write_solver_images(cache, archive, ref, target)
    actual_ref, actual_target = read_solver_images(cache, archive)
    np.testing.assert_array_equal(actual_ref, ref)
    np.testing.assert_array_equal(actual_target, target)
    archive.write_bytes(b"changed data")
    with pytest.raises(ValueError, match="fingerprint"):
        read_solver_images(cache, archive)


def test_image_cache_rejects_corrupted_pixels_and_overwrite(tmp_path):
    archive = tmp_path / "case.npz"
    archive.write_bytes(b"window data")
    image = np.zeros((2, 2), dtype=np.uint8)
    cache = tmp_path / "images"
    write_solver_images(cache, archive, image, image)
    with pytest.raises(FileExistsError):
        write_solver_images(cache, archive, image, image)
    (cache / "case_target.png").write_bytes(b"bad image")
    with pytest.raises(ValueError, match="fingerprint"):
        read_solver_images(cache, archive)


def test_image_cache_detects_changed_external_source(tmp_path):
    archive = tmp_path / "case.npz"
    source, target = tmp_path / "source.tiff", tmp_path / "target.tiff"
    archive.write_bytes(b"window data")
    source.write_bytes(b"source")
    target.write_bytes(b"target")
    image = np.zeros((2, 2), dtype=np.uint8)
    cache = tmp_path / "images"
    write_solver_images(cache, archive, image, image, source_files=(source, target))
    read_solver_images(cache, archive, source_files=(source, target))
    source.write_bytes(b"new source")
    with pytest.raises(ValueError, match="source fingerprint"):
        read_solver_images(cache, archive, source_files=(source, target))
    with pytest.raises(ValueError, match="source inputs"):
        read_solver_images(cache, archive)
