"""
test_steps.py — 풀이 과정(steps.py) 확인 (Phase 2 완료 기준)

[역할]
    1. 풀이 과정에서 "직접 세운 평형식"으로 구한 반력이 solver(SymPy) 결과와 정확히 같은지
    2. 구간별 자유물체도 식이 SymPy 구간식과 같은지 (같으면 검증 경고가 없어야 함)
    3. 일부러 틀린 값을 넣으면 검증 경고가 나오는지 (검증 장치 자체가 동작하는지)
    4. 단계 구성·LaTeX 형식·회전 방향 표시
"""

from __future__ import annotations

import pytest
import sympy as sp

from core.extremes import find_extremes
from core.model import Beam, LinearLoad, MomentLoad, PointLoad, Support, UniformLoad
from core.piecewise import split_segments
from core.solver import solve
from core.steps import build_steps, moment_direction_text, poly_latex

# 여러 종류의 보: 하중 정리(직사각형+삼각형 분할), 내민보의 음의 팔길이,
# 위 방향 하중(음수 입력), 모멘트 하중, 오른쪽 고정 캔틸레버 등을 골고루 넣었다.
CASES = {
    "단순보 등분포": Beam(6, [Support("pin", 0), Support("roller", 6)], [UniformLoad(10, 0, 6)]),
    "단순보 삼각형": Beam(6.4, [Support("pin", 0), Support("roller", 6.4)], [LinearLoad(0, 9, 0, 6.4)]),
    "단순보 감소 삼각형 부분": Beam(7, [Support("pin", 0), Support("roller", 7)], [LinearLoad(8, 0, 1, 5)]),
    "단순보 사다리꼴+모멘트": Beam(
        7.3,
        [Support("roller", 0), Support("pin", 7.3)],
        [LinearLoad(3, 8, 1, 6), MomentLoad(15, 3.3), PointLoad(-5, 2)],
    ),
    "양쪽 내민보": Beam(
        8.5,
        [Support("pin", 1.2), Support("roller", 6)],
        [UniformLoad(12.5, 0, 8.5), PointLoad(20, 0), PointLoad(30, 8.5), LinearLoad(-2, 4, 2, 5)],
    ),
    "캔틸레버 왼쪽 고정": Beam(3, [Support("fixed", 0)], [LinearLoad(0, 12, 0, 3), MomentLoad(5, 3)]),
    "캔틸레버 오른쪽 고정": Beam(
        4, [Support("fixed", 4)], [UniformLoad(10, 0, 2.5), PointLoad(7, 0), MomentLoad(-3, 1)]
    ),
}


def run(beam: Beam):
    """solver → piecewise → extremes → steps 전체 실행."""
    r = solve(beam)
    segs = split_segments(r)
    ext = find_extremes(segs, r.x)
    return r, segs, ext, build_steps(r, segs, ext)


# ─────────────────────────────────────────────────────────────
# 1·2. 손계산 반력 = SymPy 반력, 구간식 일치 (경고 없음)
# ─────────────────────────────────────────────────────────────
@pytest.mark.parametrize("name", list(CASES))
def test_hand_reactions_match_solver(name):
    r, segs, ext, out = run(CASES[name])
    assert out.warnings == []  # 반력·구간식 검증 경고가 없어야 한다
    for reac in r.reactions:
        R_h, M_h = out.hand_reactions[reac.label]
        assert sp.simplify(R_h - reac.R) == 0  # 정확히 같은 수 (오차 0)
        if reac.M is not None:
            assert sp.simplify(M_h - reac.M) == 0


# ─────────────────────────────────────────────────────────────
# 3. 검증 장치가 실제로 틀린 값을 잡아내는지
# ─────────────────────────────────────────────────────────────
def test_mismatch_is_reported():
    """solver 반력을 일부러 바꿔 놓으면 '다릅니다' 경고가 나와야 한다."""
    r = solve(CASES["단순보 등분포"])
    segs = split_segments(r)
    ext = find_extremes(segs, r.x)
    r.reactions[0].R = r.reactions[0].R + 1  # 일부러 틀리게
    out = build_steps(r, segs, ext)
    assert any("다릅니다" in w for w in out.warnings)


# ─────────────────────────────────────────────────────────────
# 4. 단계 구성과 형식
# ─────────────────────────────────────────────────────────────
def test_step_order_and_hidden_flag():
    """7장 순서: 하중 정리 → 반력 → 구간식 → 최대값, 특이함수 식은 기본 숨김."""
    *_, out = run(CASES["단순보 등분포"])  # *_ : 앞의 값들은 버리고 마지막만 받는다
    assert [s.key for s in out.steps] == ["loads", "reactions", "segments", "extremes", "singularity"]
    assert [s.hidden for s in out.steps] == [False, False, False, False, True]


@pytest.mark.parametrize("name", list(CASES))
def test_latex_braces_balanced(name):
    """KaTeX 가 그릴 수 있게 모든 LaTeX 의 중괄호 { } 짝이 맞아야 한다."""
    *_, out = run(CASES[name])
    for step in out.steps:
        for line in step.lines:
            depth = 0
            for ch in line.latex:
                depth += {"{": 1, "}": -1}.get(ch, 0)
                assert depth >= 0, line.latex
            assert depth == 0, line.latex


def test_uniform_load_summary_text():
    """지시서 7장 예: w = 10 kN/m, 길이 4 m → W = 10 × 4 = 40 kN, x̄ = 2.0 m"""
    beam = Beam(8, [Support("pin", 0), Support("roller", 8)], [UniformLoad(10, 0, 4)])
    *_, out = run(beam)
    loads = " ".join(l.latex for l in out.steps[0].lines)
    assert r"10.00 \times 4.00 = 40.00" in loads
    assert "= 2.00" in loads


def test_reaction_equations_present():
    """반력 단계에 ΣM_A = 0 → R_B, ΣF_y = 0 → R_A 가 순서대로 있어야 한다."""
    *_, out = run(CASES["양쪽 내민보"])
    latex = [l.latex for l in out.steps[1].lines]
    i_mom = next(i for i, s in enumerate(latex) if s.startswith(r"\sum M_A"))
    i_RB = next(i for i, s in enumerate(latex) if s.startswith("R_B ="))
    i_RA = next(i for i, s in enumerate(latex) if s.startswith("R_A ="))
    assert i_mom < i_RB < i_RA
    # 지점 A(1.2 m) 왼쪽 x=0 의 하중은 팔길이가 음수 → 괄호로 표시
    assert "(-1.20)" in latex[i_mom + 1]


def test_cantilever_moment_direction_shown():
    """반력 모멘트는 시계 + 값과 함께 회전 방향(기호·글자)을 표시한다."""
    *_, out = run(CASES["캔틸레버 왼쪽 고정"])
    last = out.steps[1].lines[-1].latex
    assert "M_A = -(41.00) = -41.00" in last
    assert r"\circlearrowleft" in last and "반시계" in last


def test_moment_direction_text():
    assert moment_direction_text(sp.Integer(-41)) == "반시계 ↺ 41.00 kN·m"
    assert moment_direction_text(sp.Integer(5)) == "시계 ↻ 5.00 kN·m"


def test_poly_latex_ascending():
    """손계산 순서처럼 상수항부터: 25 − 10x"""
    x = sp.Symbol("x")
    assert poly_latex(25 - 10 * x, x) == r"25 - 10\,x"
    assert poly_latex(25 * x - 5 * x**2, x) == r"25\,x - 5\,x^{2}"
    assert poly_latex(sp.Integer(0), x) == "0"
