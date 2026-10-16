"""
test_indeterminate.py — 부정정보 확인 (Phase 5B 완료 기준)

[역할]
    CLAUDE.md Phase 5B 표의 케이스를 확인한다.
    기대값은 구조역학 교과서의 부정정보 공식표(고정-핀, 양단 고정, 등경간 연속보 계수)에서 왔다.
    또한 변형일치법 풀이(compat.py → steps.py)가 SymPy 해와 정확히 같은지(경고 없음) 본다.
"""

from __future__ import annotations

import pytest
import sympy as sp

from core.extremes import find_deflection_extremes, find_extremes
from core.model import Beam, LinearLoad, MomentLoad, PointLoad, Support, UniformLoad
from core.piecewise import deflection_at, split_segments, value_at
from core.solver import solve
from core.steps import build_steps

E, I = 205000.0, 1.2e8
EI = 24600.0


def approx(v: float):
    return pytest.approx(v, rel=1e-9, abs=1e-12)


def analyze(beam: Beam):
    r = solve(beam)
    segs = split_segments(r)
    return r, segs, find_extremes(segs, r.x)


def R(r, label):
    """지점 label 의 (R, M) 을 float 로 (M 이 없으면 None)."""
    reac = next(q for q in r.reactions if q.label == label)
    return float(reac.R), (None if reac.M is None else float(reac.M))


def M_at(r, segs, xv, side="right"):
    return float(value_at(segs, r.x, xv, side)[1])


def continuous(spans: int, Ls: float, *loads, **kw) -> Beam:
    """등경간 연속보: 핀(x=0) + 롤러들, 경간 길이 Ls."""
    sup = [Support("pin", 0)] + [Support("roller", Ls * i) for i in range(1, spans + 1)]
    return Beam(Ls * spans, sup, list(loads), **kw)


# ─────────────────────────────────────────────────────────────
# 1. 고정-핀 등분포 : R_고정 = 5wL/8, R_핀 = 3wL/8, M_고정 = −wL²/8,
#    M+max = 9wL²/128 at 핀에서 3L/8
# ─────────────────────────────────────────────────────────────
def test_propped_uniform():
    w, L = 12.5, 7.3
    r, segs, ext = analyze(Beam(L, [Support("fixed", 0), Support("roller", L)], [UniformLoad(w, 0, L)]))
    assert R(r, "A")[0] == approx(5 * w * L / 8)
    assert R(r, "B")[0] == approx(3 * w * L / 8)
    assert R(r, "A")[1] == approx(-w * L**2 / 8)           # 반력 모멘트 (반시계)
    assert M_at(r, segs, 0) == approx(-w * L**2 / 8)        # 고정단 휨모멘트 (부모멘트)
    assert float(ext.M_pos_max.value) == approx(9 * w * L**2 / 128)
    assert float(ext.M_pos_max.x) == approx(L - 3 * L / 8)  # 핀(x=L)에서 3L/8


def test_propped_uniform_fixed_on_right():
    """좌우 반전: 고정단이 오른쪽이어도 같은 크기, 반력 모멘트는 시계(+)."""
    w, L = 12.5, 7.3
    r, segs, ext = analyze(Beam(L, [Support("pin", 0), Support("fixed", L)], [UniformLoad(w, 0, L)]))
    assert R(r, "A")[0] == approx(3 * w * L / 8)
    assert R(r, "B") == (approx(5 * w * L / 8), approx(w * L**2 / 8))
    assert M_at(r, segs, L, "left") == approx(-w * L**2 / 8)


# ─────────────────────────────────────────────────────────────
# 2. 고정-핀 중앙 집중하중 : R_고정 = 11P/16, R_핀 = 5P/16,
#    M_고정 = −3PL/16, M_하중점 = +5PL/32
# ─────────────────────────────────────────────────────────────
def test_propped_center_point():
    P, L = 37.0, 6.4
    r, segs, ext = analyze(Beam(L, [Support("fixed", 0), Support("pin", L)], [PointLoad(P, L / 2)]))
    assert R(r, "A")[0] == approx(11 * P / 16)
    assert R(r, "B")[0] == approx(5 * P / 16)
    assert M_at(r, segs, 0) == approx(-3 * P * L / 16)
    assert M_at(r, segs, L / 2) == approx(5 * P * L / 32)
    assert float(ext.M_pos_max.value) == approx(5 * P * L / 32)


# ─────────────────────────────────────────────────────────────
# 3. 양단 고정 등분포 : M_단부 = −wL²/12, M_중앙 = +wL²/24
# ─────────────────────────────────────────────────────────────
def test_fixed_fixed_uniform():
    w, L = 12.5, 7.3
    r, segs, ext = analyze(Beam(L, [Support("fixed", 0), Support("fixed", L)], [UniformLoad(w, 0, L)]))
    assert M_at(r, segs, 0) == approx(-w * L**2 / 12)
    assert M_at(r, segs, L, "left") == approx(-w * L**2 / 12)
    assert M_at(r, segs, L / 2) == approx(w * L**2 / 24)
    assert R(r, "A")[0] == approx(w * L / 2)
    # 반력 모멘트(시계 +): 왼쪽은 반시계(−), 오른쪽은 시계(+)
    assert R(r, "A")[1] == approx(-w * L**2 / 12)
    assert R(r, "B")[1] == approx(w * L**2 / 12)


# ─────────────────────────────────────────────────────────────
# 4. 양단 고정 중앙 집중하중 : M_단부 = −PL/8, M_중앙 = +PL/8
# ─────────────────────────────────────────────────────────────
def test_fixed_fixed_center_point():
    P, L = 37.0, 6.4
    r, segs, ext = analyze(Beam(L, [Support("fixed", 0), Support("fixed", L)], [PointLoad(P, L / 2)]))
    assert M_at(r, segs, 0) == approx(-P * L / 8)
    assert M_at(r, segs, L / 2) == approx(P * L / 8)
    assert float(ext.M_neg_min.value) == approx(-P * L / 8)


# ─────────────────────────────────────────────────────────────
# 5. 2경간 연속보 (등경간 L) 등분포 : M_중간지점 = −wL²/8, R_중간 = 5wL/4, R_끝 = 3wL/8
# ─────────────────────────────────────────────────────────────
def test_two_span_uniform():
    w, Ls = 12.5, 4.6
    r, segs, ext = analyze(continuous(2, Ls, UniformLoad(w, 0, 2 * Ls)))
    assert R(r, "A")[0] == approx(3 * w * Ls / 8)
    assert R(r, "B")[0] == approx(5 * w * Ls / 4)
    assert R(r, "C")[0] == approx(3 * w * Ls / 8)
    assert M_at(r, segs, Ls) == approx(-w * Ls**2 / 8)
    assert float(ext.M_neg_min.value) == approx(-w * Ls**2 / 8)


# ─────────────────────────────────────────────────────────────
# 6. 3경간 연속보 (등경간 L) 등분포 : M_내부지점 = −wL²/10, R_끝 = 0.4wL, R_내부 = 1.1wL
# ─────────────────────────────────────────────────────────────
def test_three_span_uniform():
    w, Ls = 12.5, 4.6
    r, segs, ext = analyze(continuous(3, Ls, UniformLoad(w, 0, 3 * Ls)))
    assert R(r, "A")[0] == approx(0.4 * w * Ls)
    assert R(r, "B")[0] == approx(1.1 * w * Ls)
    assert R(r, "C")[0] == approx(1.1 * w * Ls)
    assert R(r, "D")[0] == approx(0.4 * w * Ls)
    assert M_at(r, segs, Ls) == approx(-w * Ls**2 / 10)
    assert M_at(r, segs, 2 * Ls) == approx(-w * Ls**2 / 10)


# ─────────────────────────────────────────────────────────────
# 7. 부정정 + 여러 하중 = 개별 결과의 중첩 (선형 탄성이므로 부정정보에서도 성립)
# ─────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "supports",
    [
        [Support("fixed", 0), Support("roller", 7.3)],
        [Support("roller", 1.1), Support("fixed", 7.3)],  # 왼쪽 돌출 + 오른쪽 고정
        [Support("fixed", 0), Support("fixed", 7.3)],
        [Support("pin", 0), Support("roller", 3.1), Support("roller", 7.3)],
        [Support("roller", 0.6), Support("pin", 2.9), Support("roller", 5.2), Support("roller", 7.3)],
    ],
)
def test_superposition(supports):
    L = 7.3
    loads = [
        UniformLoad(12.5, 0, L),
        PointLoad(37.0, 2.3),
        LinearLoad(3.0, 8.0, 1.0, 6.0),
        MomentLoad(15.0, 3.7),
    ]
    rc, sc, _ = analyze(Beam(L, supports, loads))
    singles = [analyze(Beam(L, supports, [ld])) for ld in loads]
    for i, reac in enumerate(rc.reactions):
        assert float(reac.R) == approx(sum(float(s[0].reactions[i].R) for s in singles))
    for xv in (0.37, 1.55, 3.33, 4.4, 6.66):
        assert M_at(rc, sc, xv) == approx(sum(M_at(s[0], s[1], xv) for s in singles))


# ─────────────────────────────────────────────────────────────
# 8. 변형일치법 풀이 = SymPy (경고 없음), 처짐도 함께 동작
# ─────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "beam",
    [
        Beam(7.3, [Support("fixed", 0), Support("roller", 7.3)], [UniformLoad(12.5, 0, 7.3), PointLoad(20, 2)], E=E, I=I),
        Beam(7.3, [Support("roller", 1.1), Support("fixed", 7.3)], [UniformLoad(12.5, 0, 7.3), MomentLoad(-8, 0)], E=E, I=I),
        Beam(6.4, [Support("fixed", 0), Support("fixed", 6.4)], [LinearLoad(0, 9, 0, 6.4), PointLoad(15, 4)], E=E, I=I),
        continuous(2, 4.6, UniformLoad(12.5, 0, 4.6 * 2), PointLoad(30, 7), E=E, I=I),
        Beam(13, [Support("roller", 0), Support("pin", 4), Support("roller", 8), Support("roller", 12)],
             [UniformLoad(10, 0, 13), LinearLoad(0, 6, 4, 8), MomentLoad(12, 2)], E=E, I=I),
        continuous(3, 4.6, UniformLoad(12.5, 0, 4.6 * 3)),  # E, I 없이도 부정정 반력은 풀린다
    ],
)
def test_force_method_steps_match_solver(beam):
    r, segs, ext = analyze(beam)
    d = None
    if r.delta is not None:
        d = find_deflection_extremes(segs, r.x, r.L, sp.Integer(360))
    out = build_steps(r, segs, ext, d)
    assert out.warnings == []
    reac_step = next(s for s in out.steps if s.key == "reactions")
    assert "변형일치법" in reac_step.title
    for q in r.reactions:
        R_h, M_h = out.hand_reactions[q.label]
        assert sp.simplify(R_h - q.R) == 0
        if q.M is not None:
            assert sp.simplify(M_h - q.M) == 0


def test_fixed_fixed_deflection():
    """양단 고정 처짐 (교과서): 등분포 δ_중앙 = wL⁴/(384EI), 중앙 집중 δ = PL³/(192EI)."""
    w, P, L = 12.5, 37.0, 6.4
    r, segs, _ = analyze(Beam(L, [Support("fixed", 0), Support("fixed", L)], [UniformLoad(w, 0, L)], E=E, I=I))
    assert float(deflection_at(segs, r.x, L / 2)[1]) == approx(w * L**4 / (384 * EI))
    r, segs, _ = analyze(Beam(L, [Support("fixed", 0), Support("fixed", L)], [PointLoad(P, L / 2)], E=E, I=I))
    assert float(deflection_at(segs, r.x, L / 2)[1]) == approx(P * L**3 / (192 * EI))


def test_two_span_supports_do_not_deflect():
    """연속보: 모든 지점에서 처짐 0 (적합조건)."""
    r, segs, _ = analyze(continuous(2, 4.6, UniformLoad(12.5, 0, 4.6 * 2), PointLoad(30, 7), E=E, I=I))
    for xv in (0, 4.6, 9.2):
        assert float(deflection_at(segs, r.x, xv)[1]) == approx(0)
