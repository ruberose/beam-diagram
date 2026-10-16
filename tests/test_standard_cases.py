"""
test_standard_cases.py — 교과서 공식과 엔진 결과 비교 (Phase 1 완료 기준)

[역할]
    지시서 9장 Phase 1 표의 케이스를 모두 확인한다.
    기대값은 재료역학/구조역학 교과서의 보 공식표(단순보·캔틸레버 기본 공식)에서 왔다.
    일부러 "깔끔하지 않은" 숫자(w=12.5, L=7.3 등)를 써서 우연히 맞는 경우를 피한다.

[실행]  프로젝트 루트에서  python -m pytest

[pytest 메모]
    - test_ 로 시작하는 함수를 pytest 가 자동으로 찾아 실행한다.
    - assert 조건 : 조건이 거짓이면 테스트 실패
    - pytest.approx(기대값, rel=1e-9) : 상대오차 1e-9 이내면 같다고 본다
"""

from __future__ import annotations

import math

import pytest

from core import sign
from core.extremes import Extremes, find_extremes
from core.model import Beam, LinearLoad, MomentLoad, PointLoad, Support, UniformLoad
from core.piecewise import Segment, split_segments, value_at
from core.solver import SolverResult, solve

REL = 1e-9  # 지시서 완료 기준: 상대오차 1e-9


def approx(v: float):
    """상대오차 1e-9 비교. 기대값이 0 이면 절대오차 1e-12 로 비교한다."""
    return pytest.approx(v, rel=REL, abs=1e-12)


def analyze(beam: Beam) -> tuple[SolverResult, list[Segment], Extremes]:
    """solver → piecewise → extremes 를 한 번에 실행한다 (테스트 편의용)."""
    result = solve(beam)
    segments = split_segments(result)
    return result, segments, find_extremes(segments, result.x)


def VM(result: SolverResult, segments: list[Segment], xv: float, side: str = "right") -> tuple[float, float]:
    """위치 xv 의 (V, M) 을 float 로 돌려준다."""
    V, M = value_at(segments, result.x, xv, side)
    return float(V), float(M)


def simple(L: float, *loads) -> Beam:
    """핀(x=0) + 롤러(x=L) 단순보를 만든다. *loads : 하중을 몇 개든 받는다."""
    return Beam(L, [Support("pin", 0), Support("roller", L)], list(loads))


# ─────────────────────────────────────────────────────────────
# 0. 부호 보정 자체 확인
# ─────────────────────────────────────────────────────────────
def test_sign_factors_are_unit():
    """보정 계수는 모두 +1 또는 −1 이어야 한다 (SymPy 1.14: shear=+1, moment=+1, moment_load=−1)."""
    f = sign.get_factors()
    for v in (f.load, f.shear, f.moment, f.moment_load):
        assert v in (1, -1)


# ─────────────────────────────────────────────────────────────
# 1. 단순보 등분포 w : R = wL/2, M_max = wL²/8 (중앙)
# ─────────────────────────────────────────────────────────────
def test_simple_uniform():
    w, L = 12.5, 7.3
    result, segs, ext = analyze(simple(L, UniformLoad(w, 0, L)))
    R_A, R_B = (float(r.R) for r in result.reactions)
    assert R_A == approx(w * L / 2)
    assert R_B == approx(w * L / 2)
    assert float(ext.M_pos_max.value) == approx(w * L**2 / 8)
    assert float(ext.M_pos_max.x) == approx(L / 2)
    # 전단력 부호 규약 확인: 왼쪽 지점 V = +wL/2, 오른쪽 지점 V = −wL/2
    assert VM(result, segs, 0)[0] == approx(w * L / 2)
    assert VM(result, segs, L, "left")[0] == approx(-w * L / 2)
    assert ext.M_neg_min is None  # 단순보 하향 하중이면 부모멘트 없음


# ─────────────────────────────────────────────────────────────
# 2. 단순보 중앙 집중하중 P : R = P/2, M_max = PL/4
# ─────────────────────────────────────────────────────────────
def test_simple_center_point():
    P, L = 37.0, 7.3
    result, segs, ext = analyze(simple(L, PointLoad(P, L / 2)))
    for r in result.reactions:
        assert float(r.R) == approx(P / 2)
    assert float(ext.M_pos_max.value) == approx(P * L / 4)
    assert float(ext.M_pos_max.x) == approx(L / 2)
    # 하중점 좌우에서 V 가 +P/2 → −P/2 로 점프
    assert VM(result, segs, L / 2, "left")[0] == approx(P / 2)
    assert VM(result, segs, L / 2, "right")[0] == approx(-P / 2)


# ─────────────────────────────────────────────────────────────
# 3. 단순보 임의 위치 집중하중 : R_A = Pb/L, R_B = Pa/L, M_max = Pab/L
# ─────────────────────────────────────────────────────────────
def test_simple_offset_point():
    P, a, L = 37.0, 2.3, 7.1
    b = L - a
    result, segs, ext = analyze(simple(L, PointLoad(P, a)))
    R_A, R_B = (float(r.R) for r in result.reactions)
    assert R_A == approx(P * b / L)
    assert R_B == approx(P * a / L)
    assert float(ext.M_pos_max.value) == approx(P * a * b / L)
    assert float(ext.M_pos_max.x) == approx(a)


# ─────────────────────────────────────────────────────────────
# 4. 단순보 삼각형 분포 (0 → w) :
#    R_A = wL/6, R_B = wL/3, M_max = wL²/(9√3) at x = L/√3
# ─────────────────────────────────────────────────────────────
def test_simple_triangular():
    w, L = 9.0, 6.4
    result, segs, ext = analyze(simple(L, LinearLoad(0, w, 0, L)))
    R_A, R_B = (float(r.R) for r in result.reactions)
    assert R_A == approx(w * L / 6)
    assert R_B == approx(w * L / 3)
    assert float(ext.M_pos_max.value) == approx(w * L**2 / (9 * math.sqrt(3)))
    assert float(ext.M_pos_max.x) == approx(L / math.sqrt(3))
    # V = 0 위치도 같은 곳이어야 한다
    roots = [z for z in ext.zero_shear if z.kind == "root"]
    assert len(roots) == 1
    assert float(roots[0].x) == approx(L / math.sqrt(3))


# ─────────────────────────────────────────────────────────────
# 5. 단순보 끝단 집중 모멘트 M₀ (시계 +) : R = ±M₀/L, 모멘트도 선형
#    오른쪽 끝(B) 시계 M₀ : ΣM_A = 0 → R_B·L − M₀ = 0 → R_B = +M₀/L, R_A = −M₀/L
#                          M(x) = R_A·x = −M₀·x/L  (B 에서 −M₀)
#    왼쪽 끝(A) 시계 M₀  : R_A = −M₀/L, R_B = +M₀/L, M(x) = M₀(1 − x/L)  (A 에서 +M₀)
# ─────────────────────────────────────────────────────────────
def test_simple_end_moment_right():
    M0, L = 24.0, 5.5
    result, segs, ext = analyze(simple(L, MomentLoad(M0, L)))
    R_A, R_B = (float(r.R) for r in result.reactions)
    assert R_A == approx(-M0 / L)
    assert R_B == approx(M0 / L)
    for xv in (0.0, 1.1, 2.75, 4.4):
        assert VM(result, segs, xv)[1] == approx(-M0 * xv / L)
    assert float(ext.M_neg_min.value) == approx(-M0)
    assert float(ext.M_neg_min.x) == approx(L)


def test_simple_end_moment_left():
    M0, L = 24.0, 5.5
    result, segs, ext = analyze(simple(L, MomentLoad(M0, 0)))
    R_A, R_B = (float(r.R) for r in result.reactions)
    assert R_A == approx(-M0 / L)
    assert R_B == approx(M0 / L)
    for xv in (0.0, 1.1, 2.75, 4.4):
        assert VM(result, segs, xv)[1] == approx(M0 * (1 - xv / L))
    assert float(ext.M_pos_max.value) == approx(M0)
    assert float(ext.M_pos_max.x) == approx(0)


# ─────────────────────────────────────────────────────────────
# 6. 캔틸레버 등분포 : 고정단 M = −wL²/2, V = wL
# ─────────────────────────────────────────────────────────────
def test_cantilever_uniform_left_fixed():
    w, L = 12.5, 3.7
    result, segs, ext = analyze(Beam(L, [Support("fixed", 0)], [UniformLoad(w, 0, L)]))
    (r,) = result.reactions  # 반력이 하나뿐인 목록을 바로 풀어서 받는다
    assert float(r.R) == approx(w * L)
    # 반력 모멘트(시계 +): 하향 하중을 받는 왼쪽 고정단은 반시계 → −wL²/2
    assert float(r.M) == approx(-w * L**2 / 2)
    V0, M0 = VM(result, segs, 0)
    assert V0 == approx(w * L)
    assert M0 == approx(-w * L**2 / 2)
    assert float(ext.M_neg_min.value) == approx(-w * L**2 / 2)
    assert ext.M_pos_max is None


def test_cantilever_uniform_right_fixed():
    """오른쪽 고정: 고정단 V = −wL (왼쪽 자유 부분이 아래로 밀림), M = −wL²/2."""
    w, L = 12.5, 3.7
    result, segs, ext = analyze(Beam(L, [Support("fixed", L)], [UniformLoad(w, 0, L)]))
    V_L, M_L = VM(result, segs, L, "left")
    assert V_L == approx(-w * L)
    assert M_L == approx(-w * L**2 / 2)
    assert float(result.reactions[0].R) == approx(w * L)


# ─────────────────────────────────────────────────────────────
# 7. 캔틸레버 자유단 집중하중 : 고정단 M = −PL
# ─────────────────────────────────────────────────────────────
def test_cantilever_tip_point():
    P, L = 37.0, 2.9
    result, segs, ext = analyze(Beam(L, [Support("fixed", 0)], [PointLoad(P, L)]))
    assert float(result.reactions[0].R) == approx(P)
    assert VM(result, segs, 0)[1] == approx(-P * L)
    assert float(ext.M_neg_min.value) == approx(-P * L)
    assert float(ext.M_neg_min.x) == approx(0)


# ─────────────────────────────────────────────────────────────
# 8. 내민보 (돌출부 끝 집중하중) : 지점 위 부모멘트 = −P·(돌출길이)
#    지점 A(0), B(a), 끝 L 에 P. ΣM_A = 0 → R_B = P·L/a, R_A = P − R_B = −P(L−a)/a
# ─────────────────────────────────────────────────────────────
def test_overhang_tip_point():
    P, a, L = 37.0, 6.0, 8.5
    c = L - a  # 돌출 길이
    result, segs, ext = analyze(
        Beam(L, [Support("pin", 0), Support("roller", a)], [PointLoad(P, L)])
    )
    R_A, R_B = (float(r.R) for r in result.reactions)
    assert R_B == approx(P * L / a)
    assert R_A == approx(-P * c / a)
    assert VM(result, segs, a)[1] == approx(-P * c)
    assert float(ext.M_neg_min.value) == approx(-P * c)
    assert float(ext.M_neg_min.x) == approx(a)


# ─────────────────────────────────────────────────────────────
# 9. 여러 하중 조합 = 개별 결과의 중첩
#    선형 탄성 정정보이므로 반력, V(x), M(x) 모두 하중별 결과의 합과 같아야 한다.
# ─────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "supports",
    [
        [Support("pin", 0), Support("roller", 7.3)],      # 단순보
        [Support("pin", 1.2), Support("roller", 5.8)],    # 양쪽 내민보
        [Support("fixed", 0)],                            # 캔틸레버
    ],
)
def test_superposition(supports):
    """@pytest.mark.parametrize : 같은 테스트를 지점 구성 3가지로 반복 실행한다."""
    L = 7.3
    loads = [
        UniformLoad(12.5, 0, L),
        UniformLoad(-4.0, 2.0, 4.5),   # 위로 작용하는 부분 등분포 (음수 입력)
        PointLoad(37.0, 2.3),
        LinearLoad(3.0, 8.0, 1.0, 6.0),  # 사다리꼴
        MomentLoad(15.0, 3.3),
    ]
    combined_r, combined_s, _ = analyze(Beam(L, supports, loads))
    singles = [analyze(Beam(L, supports, [ld])) for ld in loads]

    # 반력 중첩
    for i, r in enumerate(combined_r.reactions):
        assert float(r.R) == approx(sum(float(s[0].reactions[i].R) for s in singles))
        if r.M is not None:
            assert float(r.M) == approx(sum(float(s[0].reactions[i].M) for s in singles))

    # V, M 중첩 (경계가 아닌 여러 점에서 비교)
    for xv in (0.37, 1.55, 2.9, 3.71, 5.03, 6.66, 7.1):
        V_c, M_c = VM(combined_r, combined_s, xv)
        V_sum = sum(VM(s[0], s[1], xv)[0] for s in singles)
        M_sum = sum(VM(s[0], s[1], xv)[1] for s in singles)
        assert V_c == approx(V_sum)
        assert M_c == approx(M_sum)


# ─────────────────────────────────────────────────────────────
# 10. 입력 검사
# ─────────────────────────────────────────────────────────────
def test_indeterminate_rejected():
    """Phase 1 은 정정보만: 지점 3개(부정정)는 거부해야 한다."""
    beam = Beam(6, [Support("pin", 0), Support("roller", 3), Support("roller", 6)], [])
    with pytest.raises(ValueError):  # with ... : 이 블록 안에서 ValueError 가 나야 통과
        solve(beam)


def test_load_out_of_range_rejected():
    beam = simple(6, PointLoad(10, 7))
    with pytest.raises(ValueError):
        solve(beam)
