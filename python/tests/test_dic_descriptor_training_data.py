from types import SimpleNamespace

import numpy as np
from train_dic_cnn_descriptor import support_is_valid, target_support_is_valid


def test_descriptor_support_must_be_fully_inside_both_validity_masks():
    shape = (160, 160)
    flow = np.zeros((*shape, 2), dtype=np.float32)
    source_valid = np.ones(shape, dtype=bool)
    target_valid = np.ones(shape, dtype=bool)
    source_valid[:, 78:83] = False
    target_valid[78:83, :] = False
    pair = SimpleNamespace(
        reference=np.zeros(shape, dtype=np.float32),
        target=np.zeros(shape, dtype=np.float32),
        flow_xy=flow,
        visible=source_valid,
        target_visible=target_valid,
    )

    points = np.asarray([[40.0, 40.0], [80.0, 80.0], [8.0, 80.0]], dtype=np.float32)
    assert support_is_valid(pair, points).tolist() == [True, False, False]
    assert target_support_is_valid(pair, points).tolist() == [True, False, False]


def test_source_support_checks_mapped_target_boundary():
    shape = (160, 160)
    flow = np.zeros((*shape, 2), dtype=np.float32)
    flow[..., 0] = 28.0
    pair = SimpleNamespace(
        reference=np.zeros(shape, dtype=np.float32),
        target=np.zeros(shape, dtype=np.float32),
        flow_xy=flow,
        visible=np.ones(shape, dtype=bool),
        target_visible=np.ones(shape, dtype=bool),
    )
    points = np.asarray([[80.0, 80.0], [120.0, 80.0]], dtype=np.float32)
    assert support_is_valid(pair, points).tolist() == [True, False]
