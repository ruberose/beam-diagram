"""
test_deflection.py — 처짐 계산 확인 (Phase 5A 완료 기준)

[역할]
    CLAUDE.md Phase 5A 표의 처짐 공식을 확인한다.
    기대값은 재료역학 교과서의 보 처짐 공식표(단순보·캔틸레버 기본 공식)에서 왔다.
    부호 (지시서 D2): 처짐 δ 아래 +, 처짐각 θ 시계 +.

    E = 205000 MPa, I = 1.2×10⁸ mm⁴ → EI = 205000 × 1.2e8 × 1e-9 = 24600 kN·m²
"""

from __future__ import annotations

import math

import pytest
import sympy as sp

from core.extremes import find_deflection_extremes, find_extremes
from core.model import Beam, LinearLoad, MomentLoad, PointLoad, Support, UniformLoad
from core.piecewise import deflection_at, split_segments
from core.solver import solve, stiffness_EI
from core.steps import build_steps

E, I = 205000.0, 1.2e8
EI = 24600.0  # kN·m²


def approx(v: float):
    """상대오차 1e-9 (기대값 0 이면 절대오차 1e-12)."""
    return pytest.approx(v, rel=1e-9, abs=1e-12)


def run(beam: Beam):
    """solver → piecewise → 처짐 최대값까지."""
    r = solve(beam)
    segs = split_segments(r)
    d = find_deflection_extremes(segs, r.x, r.L, sp.Integer(360))
    return r, segs, d


def simple(L, *loads) -> Beam:
    return Beam(L, [Support("pin", 0), Support("roller", L)], list(loads), E=E, I=I)


def cantilever(L, *loads) -> Beam:
    return Beam(L, [Support("fixed", 0)], list(loads), E=E, I=I)


def theta_delta(r, segs, xv) -> tuple[float, float]:
    t, d = deflection_at(segs, r.x, xv)
    return float(t), float(d)


# ─────────────────────────────────────────────────────────────
# 0. 강성 환산 (지시서 D1): EI = E × I × 10⁻⁹
# ─────────────────────────────────────────────────────────────
def test_stiffness_conversion():
    assert stiffness_EI(simple(6)) == 24600
    assert stiffness_EI(Beam(6, [Support("pin", 0), Support("roller", 6)], [])) is None


def test_no_stiffness_means_no_deflection():
    """E, I 를 넣지 않으면 처짐은 계산하지 않는다 (지금까지의 동작 그대로)."""
    r = solve(Beam(6, [Support("pin", 0), Support("roller", 6)], [UniformLoad(10, 0, 6)]))
    assert r.delta is None and r.theta is None and r.EI is None
    assert split_segments(r)[0].delta is None


def test_only_one_of_E_I_rejected():
    with pytest.raises(ValueError):
        solve(Beam(6, [Support("pin", 0), Support("roller", 6)], [], E=205000))


# ─────────────────────────────────────────────────────────────
# 1. 단순보 등분포 : δ_max = 5wL⁴/(384EI) at L/2,  θ_A = +wL³/(24EI) (시계)
# ─────────────────────────────────────────────────────────────
def test_simple_uniform():
    w, L = 12.5, 7.3
    r, segs, d = run(simple(L, UniformLoad(w, 0, L)))
    assert float(d.delta_down.value) == approx(5 * w * L**4 / (384 * EI))
    assert float(d.delta_down.x) == approx(L / 2)
    assert d.delta_up is None
    # 부호 확인 (D2): 왼쪽 지점은 시계방향(+), 오른쪽 지점은 반시계(−) 회전
    assert theta_delta(r, segs, 0)[0] == approx(w * L**3 / (24 * EI))
    assert theta_delta(r, segs, L)[0] == approx(-w * L**3 / (24 * EI))


# ─────────────────────────────────────────────────────────────
# 2. 단순보 중앙 집중하중 : δ_max = PL³/(48EI) at L/2
# ─────────────────────────────────────────────────────────────
def test_simple_center_point():
    P, L = 37.0, 7.3
    r, segs, d = run(simple(L, PointLoad(P, L / 2)))
    assert float(d.delta_down.value) == approx(P * L**3 / (48 * EI))
    assert float(d.delta_down.x) == approx(L / 2)


# ─────────────────────────────────────────────────────────────
# 3. 단순보 임의 위치 집중하중 (a > b) :
#    δ_max = Pb(L² − b²)^{3/2} / (9√3·L·EI) at x = √((L² − b²)/3)
# ─────────────────────────────────────────────────────────────
def test_simple_offset_point():
    P, a, L = 37.0, 4.6, 7.1
    b = L - a
    r, segs, d = run(simple(L, PointLoad(P, a)))
    assert float(d.delta_down.value) == approx(P * b * (L**2 - b**2) ** 1.5 / (9 * math.sqrt(3) * L * EI))
    assert float(d.delta_down.x) == approx(math.sqrt((L**2 - b**2) / 3))


# ─────────────────────────────────────────────────────────────
# 4. 캔틸레버 등분포 : 자유단 δ = wL⁴/(8EI), θ = wL³/(6EI)
# ─────────────────────────────────────────────────────────────
def test_cantilever_uniform():
    w, L = 12.5, 3.7
    r, segs, d = run(cantilever(L, UniformLoad(w, 0, L)))
    t, dl = theta_delta(r, segs, L)
    assert dl == approx(w * L**4 / (8 * EI))
    assert t == approx(w * L**3 / (6 * EI))
    assert float(d.delta_abs.value) == approx(w * L**4 / (8 * EI))
    assert float(d.delta_abs.x) == approx(L)
    assert theta_delta(r, segs, 0) == (0.0, 0.0)  # 고정단


def test_cantilever_right_fixed_uniform():
    """오른쪽 고정: 왼쪽 자유단 δ = wL⁴/8EI (아래), θ = −wL³/6EI (반시계)."""
    w, L = 12.5, 3.7
    r, segs, d = run(Beam(L, [Support("fixed", L)], [UniformLoad(w, 0, L)], E=E, I=I))
    t, dl = theta_delta(r, segs, 0)
    assert dl == approx(w * L**4 / (8 * EI))
    assert t == approx(-w * L**3 / (6 * EI))


# ─────────────────────────────────────────────────────────────
# 5. 캔틸레버 자유단 집중하중 : δ = PL³/(3EI), θ = PL²/(2EI)
# ─────────────────────────────────────────────────────────────
def test_cantilever_tip_point():
    P, L = 37.0, 2.9
    r, segs, d = run(cantilever(L, PointLoad(P, L)))
    t, dl = theta_delta(r, segs, L)
    assert dl == approx(P * L**3 / (3 * EI))
    assert t == approx(P * L**2 / (2 * EI))


# ─────────────────────────────────────────────────────────────
# 6. 내민보 돌출부 끝 집중하중 (경간 a, 돌출 c) : 끝단 δ = Pc²(a + c)/(3EI) (아래)
#    경간 안은 위로 솟는다 → delta_up 이 있어야 한다
# ─────────────────────────────────────────────────────────────
def test_overhang_tip_point():
    P, a, L = 37.0, 6.0, 8.5
    c = L - a
    r, segs, d = run(Beam(L, [Support("pin", 0), Support("roller", a)], [PointLoad(P, L)], E=E, I=I))
    assert theta_delta(r, segs, L)[1] == approx(P * c**2 * (a + c) / (3 * EI))
    assert float(d.delta_down.value) == approx(P * c**2 * (a + c) / (3 * EI))
    assert d.delta_up is not None and float(d.delta_up.value) < 0
    # 경간 안 최대 솟음 (교과서): P·c·a² / (9√3·EI) at x = a/√3
    assert float(d.delta_up.value) == approx(-P * c * a**2 / (9 * math.sqrt(3) * EI))
    assert float(d.delta_up.x) == approx(a / math.sqrt(3))


# ─────────────────────────────────────────────────────────────
# 7. 여러 하중 조합 = 개별 처짐의 중첩
# ─────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "supports",
    [
        [Support("pin", 0), Support("roller", 7.3)],
        [Support("pin", 1.2), Support("roller", 5.8)],
        [Support("fixed", 0)],
    ],
)
def test_superposition(supports):
    L = 7.3
    loads = [
        UniformLoad(12.5, 0, L),
        PointLoad(37.0, 2.3),
        LinearLoad(3.0, 8.0, 1.0, 6.0),
        MomentLoad(15.0, 3.3),
    ]
    rc, sc, _ = run(Beam(L, supports, loads, E=E, I=I))
    singles = [run(Beam(L, supports, [ld], E=E, I=I)) for ld in loads]
    for xv in (0.37, 1.55, 2.9, 4.4, 6.66, 7.3):
        tc, dc = theta_delta(rc, sc, xv)
        assert dc == approx(sum(theta_delta(s[0], s[1], xv)[1] for s in singles))
        assert tc == approx(sum(theta_delta(s[0], s[1], xv)[0] for s in singles))


# ─────────────────────────────────────────────────────────────
# 8. 허용처짐 판정 (D4) + 풀이 과정 검증
# ─────────────────────────────────────────────────────────────
def test_allowable_check():
    """δ_max = 9.97 mm, L/360 = 19.72 mm → OK.  n = 1000 이면 L/1000 = 7.1 mm → NG."""
    beam = simple(7.1, PointLoad(37, 4.6))
    r, segs, d = run(beam)
    assert d.ok is True
    d2 = find_deflection_extremes(segs, r.x, r.L, sp.Integer(1000))
    assert d2.ok is False


@pytest.mark.parametrize(
    "beam",
    [
        simple(7.3, UniformLoad(12.5, 0, 7.3), PointLoad(-5, 2), MomentLoad(15, 3.3)),
        simple(6.4, LinearLoad(0, 9, 0, 6.4)),
        Beam(8.5, [Support("pin", 1.2), Support("roller", 6)],
             [UniformLoad(12.5, 0, 8.5), PointLoad(30, 8.5), LinearLoad(-2, 4, 2, 5)], E=E, I=I),
        cantilever(3, LinearLoad(0, 12, 0, 3), MomentLoad(5, 3)),
        Beam(4, [Support("fixed", 4)], [UniformLoad(10, 0, 2.5), PointLoad(7, 0)], E=E, I=I),
    ],
)
def test_macaulay_steps_match_solver(beam):
    """풀이 과정(맥컬리법)으로 직접 만든 δ(x) 가 SymPy 와 같아야 한다 → 검증 경고 없음."""
    r = solve(beam)
    segs = split_segments(r)
    ext = find_extremes(segs, r.x)
    d = find_deflection_extremes(segs, r.x, r.L, sp.Integer(360))
    out = build_steps(r, segs, ext, d)
    assert out.warnings == []
    keys = [s.key for s in out.steps]
    assert keys.index("deflection") == keys.index("extremes") + 1
