"""Check connected-component counting used in the archive audit."""

import numpy as np
from analyze_anchor_availability import components


def test_visible_components_and_anchor_counts():
    centres = np.array([(x, y) for y in range(3) for x in range(5)])
    visible = np.array([x != 2 for x, _ in centres])
    anchors = np.array([(x, y) in {(0, 0), (4, 2)} for x, y in centres])
    found = components(centres, visible, anchors)
    assert found == [{"windows": 6, "correct_anchors": 1}] * 2
