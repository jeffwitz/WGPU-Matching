"""Test identity of synthetic pairs sharing random seeds across families."""

import json

import pytest
from qualify_image_seed_pilot import (
    check_lazy_cache,
    file_sha256,
    rotations_for_pair,
    select_archive_windows,
    source_model_size,
)


def test_rotation_family_is_part_of_pair_identity():
    entries = [{"family": "rigid", "texture_seed": 7, "warp_seed": 9, "windows": "rigid_x.npz"},
               {"family": "discontinuous", "texture_seed": 7, "warp_seed": 9,
                "windows": "discontinuous_x.npz"},
               {"texture_seed": 7, "warp_seed": 9, "windows": "discontinuous_legacy.npz"}]
    assert rotations_for_pair(entries, 7, 9, "rigid") == entries[:1]
    assert rotations_for_pair(entries, 7, 9, "discontinuous") == entries[1:]


def test_lazy_cache_requires_matching_fingerprint(tmp_path):
    data = tmp_path / "weights"
    data.write_bytes(b"weights-v1")
    fingerprint = {"weights_sha256": file_sha256(data)}
    cache = tmp_path / "rotation-cache"
    marker = check_lazy_cache(cache, fingerprint)
    cache.mkdir()
    marker.write_text(json.dumps(fingerprint))
    assert check_lazy_cache(cache, fingerprint) == marker
    data.write_bytes(b"weights-v2")
    with pytest.raises(ValueError, match="incompatible"):
        check_lazy_cache(cache, {"weights_sha256": file_sha256(data)})
    marker.unlink()
    (cache / "report.json").write_text("{}")
    with pytest.raises(ValueError, match="no input fingerprint"):
        check_lazy_cache(cache, fingerprint)


def test_lazy_rotation_preserves_nonstandard_model_raster():
    assert source_model_size({"model_input_px": [192, 146]}) == [192, 146]
    with pytest.raises(ValueError, match="dimensions"):
        source_model_size({"model_input_px": [192, 0]})


def test_archive_selection_respects_family_seed_and_texture(tmp_path):
    first = tmp_path / "windows_seed24000000" / "translation_texture24051000_warp1.npz"
    second = tmp_path / "windows_seed24000000" / "heterogeneous_texture24051001_warp2.npz"
    third = tmp_path / "windows_seed23000000" / "translation_texture23051000_warp3.npz"
    files = [first, second, third]
    assert select_archive_windows(files, ["translation"], [24000000], [24051000]) == [first]
    shear = tmp_path / "windows_seed24000000" / "shear_band_texture24051005_warp3.npz"
    assert select_archive_windows([first, shear], ["shear_band"], [24000000]) == [shear]
    legacy = {"texture_seed": 7, "warp_seed": 9, "windows": "shear_band_texture7_warp9.npz"}
    assert rotations_for_pair([legacy], 7, 9, "shear_band") == [legacy]
