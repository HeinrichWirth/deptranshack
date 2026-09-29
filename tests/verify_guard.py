import sys
from pathlib import Path

R = Path(__file__).resolve().parent
sys.path[:0] = [str(R.parent / "engine/runtime")]
import numpy as np
from extension_guard import guard

y = np.arange(0, 28.01, 0.1)
pair = np.stack(
    (np.column_stack((y * 0 - 0.792, -y, y * 0)), np.column_stack((y * 0 + 0.792, -y, y * 0))), 1
)
end = pair[80]
s = np.arange(8, 28.01, 0.1)
points = np.vstack([np.column_stack((s * 0 + x, -s, s * 0)) for x in [-0.792, 0.792]])
accepted, g = guard(pair, points, end)
assert np.array_equal(accepted, pair)
clipped, g = guard(pair, np.empty((0, 3)), end)
assert np.array_equal(clipped, pair[:81])
assert g["unknown_from_m"] == 8
clipped, g = guard(pair, points[points[:, 0] < 0], end)
assert len(clipped) == 81
clipped, g = guard(pair, points + [0, 0, -0.10], end)
assert len(clipped) == 81
clipped, g = guard(pair, np.repeat([[0.792, -9, 0], [-0.792, -9, 0]], 100, axis=0), end)
assert len(clipped) == 81
crooked = points.copy()
crooked[crooked[:, 1] < -14, 0] += 0.25
clipped, g = guard(pair, crooked, end)
assert 14 <= g["accepted_end_m"] <= 16
assert np.array_equal(clipped, pair[: len(clipped)])
print(
    "PASS: supported extension, unknown absence, two heads, vertical veto, duplicate-point rejection, contiguous prefix, immutable observed geometry"
)
