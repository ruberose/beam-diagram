"""
test_piecewise.py — 특이함수 식 → 구간별 식 변환 확인

[역할]
    piecewise.py 가 구간을 올바르게 나누고, 각 구간 식을 손계산과 같게 만드는지 본다.
    기대 식은 자유물체도(FBD)로 직접 세운 손계산 식이다.
"""

from __future__ import annotations

import sympy as sp

from core.extremes import find_extremes
from core.model import Beam, MomentLoad, PointLoad, Support, UniformLoad
from core.piecewise import boundaries, singularity_to_poly, split_segments
from core.solver import solve


def same(a: sp.Expr, b: sp.Expr) -> bool:
    """두 식이 수학적으로 같은지 (a − b 를 간단히 해서 0 인지) 확인한다."""
    return sp.simplify(sp.expand(a - b)) == 0


def test_singularity_unfold_rule():
    """⟨x−a⟩ⁿ: 구간이 a 오른쪽이면 (x−a)ⁿ, 왼쪽이면 0, 음의 차수는 항상 0."""
    x = sp.Symbol("x")
    SF = sp.SingularityFunction
    expr = 3 * SF(x, 0, 1) - 2 * SF(x, 4, 2) + 7 * SF(x, 2, -1)
    assert same(singularity_to_poly(expr, x, 1), 3 * x)
    assert same(singularity_to_poly(expr, x, 5), 3 * x - 2 * (x - 4) ** 2)


def test_partial_uniform_segments():
    """
    단순보 L=8, 0~4 에 w=10 kN/m.
    손계산: ΣM_A = 0 → R_B·8 = 40·2 → R_B = 10, R_A = 30
        0 ≤ x < 4 : V = 30 − 10x,  M = 30x − 5x²
        4 ≤ x < 8 : V = 30 − 40 = −10,  M = 30x − 40(x − 2) = 80 − 10x
    V = 0 : 30 − 10x = 0 → x = 3 → M = 90 − 45 = 45
    """
    beam = Beam(8, [Support("pin", 0), Support("roller", 8)], [UniformLoad(10, 0, 4)])
    result = solve(beam)
    x = result.x
    segs = split_segments(result)

    assert [s.x_start for s in segs] == [0, 4]
    assert [s.x_end for s in segs] == [4, 8]
    assert same(segs[0].V, 30 - 10 * x)
    assert same(segs[0].M, 30 * x - 5 * x**2)
    assert same(segs[1].V, sp.Integer(-10))
    assert same(segs[1].M, 80 - 10 * x)

    ext = find_extremes(segs, x)
    assert ext.M_pos_max.x == 3
    assert ext.M_pos_max.value == 45


def test_boundaries_include_all_positions():
    """경계 = 0, L, 지점, 집중하중, 모멘트, 분포하중 시작·끝 (중복 제거, 정렬)."""
    beam = Beam(
        10,
        [Support("pin", 1), Support("roller", 8)],
        [PointLoad(5, 3), MomentLoad(2, 8), UniformLoad(1, 2, 6)],
    )
    assert boundaries(beam) == [0, 1, 2, 3, 6, 8, 10]


def test_point_and_moment_jumps():
    """
    단순보 L=6, x=2 에 P=12, x=4 에 시계 모멘트 6.
    ΣM_A = 0 (반시계 +): R_B·6 − 12·2 − 6 = 0 → R_B = 5, R_A = 7
        0 ≤ x < 2 : V = 7,  M = 7x
        2 ≤ x < 4 : V = −5, M = 7x − 12(x−2) = 24 − 5x
        4 ≤ x < 6 : V = −5, M = 24 − 5x + 6 = 30 − 5x   (시계 모멘트 → M 이 +6 점프)
    """
    beam = Beam(
        6, [Support("pin", 0), Support("roller", 6)], [PointLoad(12, 2), MomentLoad(6, 4)]
    )
    result = solve(beam)
    x = result.x
    R_A, R_B = (r.R for r in result.reactions)
    assert (R_A, R_B) == (7, 5)
    segs = split_segments(result)
    assert same(segs[0].V, sp.Integer(7)) and same(segs[0].M, 7 * x)
    assert same(segs[1].V, sp.Integer(-5)) and same(segs[1].M, 24 - 5 * x)
    assert same(segs[2].V, sp.Integer(-5)) and same(segs[2].M, 30 - 5 * x)

    ext = find_extremes(segs, x)
    # M 후보: x=2 에서 14, x=4⁻ 에서 4, x=4⁺ 에서 10 → 최대 14 at x=2
    assert ext.M_pos_max.value == 14 and ext.M_pos_max.x == 2
    # V 가 x=2 에서 +7 → −5 로 부호 변화 → "jump" 형 V=0 위치
    jumps = [z for z in ext.zero_shear if z.kind == "jump"]
    assert len(jumps) == 1 and jumps[0].x == 2
