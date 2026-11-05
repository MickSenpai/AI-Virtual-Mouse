import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import util  # noqa: E402


class TestGetAngle:
    def test_right_angle(self):
        assert util.get_angle((1, 0), (0, 0), (0, 1)) == pytest.approx(90)

    def test_straight_line_is_180(self):
        assert util.get_angle((-1, 0), (0, 0), (1, 0)) == pytest.approx(180)

    def test_folded_back_is_0(self):
        assert util.get_angle((1, 0), (0, 0), (2, 0)) == pytest.approx(0)

    def test_symmetric_regardless_of_direction(self):
        a, b, c = (1, 2), (0, 0), (3, -1)
        assert util.get_angle(a, b, c) == pytest.approx(util.get_angle(c, b, a))

    def test_stays_in_0_180_range(self):
        # Rays that make the raw atan2 difference exceed 180 degrees.
        angle = util.get_angle((-1, -0.1), (0, 0), (-1, 0.1))
        assert 0 <= angle <= 180
        assert angle == pytest.approx(math.degrees(2 * math.atan(0.1)), abs=0.01)


class TestDistanceBetweenPoints:
    def test_pythagorean_triple(self):
        assert util.distance_between_points((0, 0), (3, 4)) == pytest.approx(5)

    def test_same_point_is_zero(self):
        assert util.distance_between_points((2, 2), (2, 2)) == 0


class TestInterpolate:
    def test_midpoint(self):
        assert util.interpolate(5, [0, 10], [0, 100]) == pytest.approx(50)

    def test_clamps_outside_source_range(self):
        # np.interp holds the end values, which is what keeps the cursor on screen.
        assert util.interpolate(-50, [0, 10], [0, 100]) == 0
        assert util.interpolate(50, [0, 10], [0, 100]) == 100

    def test_returns_plain_float(self):
        assert isinstance(util.interpolate(1, [0, 2], [0, 4]), float)


class TestGetDistance:
    def test_scales_normalized_distance_to_1000(self):
        assert util.get_distance([(0, 0), (0.3, 0.4)]) == pytest.approx(500)

    def test_only_uses_first_two_points(self):
        assert util.get_distance([(0, 0), (0.1, 0), (9, 9)]) == pytest.approx(100)

    @pytest.mark.parametrize("points", [None, [], [(0, 0)]])
    def test_returns_none_with_fewer_than_two_points(self, points):
        assert util.get_distance(points) is None


class TestClamp:
    @pytest.mark.parametrize(
        "value, expected",
        [(-5, 0), (0, 0), (5, 5), (10, 10), (15, 10)],
    )
    def test_clamp(self, value, expected):
        assert util.clamp(value, 0, 10) == expected


class TestSmoothCursor:
    def test_first_call_returns_target_unsmoothed(self):
        assert util.smooth_cursor(None, (100, 200), 0.7) == (100, 200)

    def test_zero_smoothing_jumps_to_target(self):
        assert util.smooth_cursor((0, 0), (100, 50), 0) == (100, 50)

    def test_moves_a_fraction_toward_target(self):
        x, y = util.smooth_cursor((0, 0), (100, 50), 0.7)
        assert x == pytest.approx(30)
        assert y == pytest.approx(15)

    def test_smoothing_is_capped_so_cursor_still_moves(self):
        # smoothing >= 1 would freeze the cursor; it is clamped to 0.99.
        x, _ = util.smooth_cursor((0, 0), (100, 0), 5)
        assert x == pytest.approx(1)

    def test_converges_to_target(self):
        point = (0.0, 0.0)
        for _ in range(100):
            point = util.smooth_cursor(point, (100, 100), 0.7)
        assert point[0] == pytest.approx(100, abs=0.01)
        assert point[1] == pytest.approx(100, abs=0.01)
