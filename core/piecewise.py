"""
piecewise.py — 특이함수 식을 구간별 일반 다항식으로 바꾼다

[역할]
    solver 가 만든 V(x), M(x) 는 ⟨x−a⟩ⁿ 이 섞인 식 하나다.
    손계산 검토서는 "0 ≤ x < 2 : V(x) = 25 − 10x" 처럼 구간마다 식을 쓰므로,
    하중·지점 위치를 경계로 보를 자르고 구간마다 특이함수를 풀어서 다항식으로 만든다.
    또한 임의 위치의 V, M 값을 구하는 함수(value_at)도 제공한다(그래프·검증용).

[관계]
    - 이 파일을 부르는 곳: extremes.py, demo.py, tests/, (나중에) steps.py, web/bridge.py
    - 이 파일이 부르는 곳: solver.py(SolverResult, exact), model.py(하중 종류), sympy

[특이함수를 구간 안에서 푸는 규칙]
    구간 (x₁, x₂) 안의 아무 점 x_m (중앙점 사용)에서 판단한다.
        ⟨x − a⟩ⁿ, n ≥ 0 : x_m > a 이면 (x − a)ⁿ,  아니면 0
        ⟨x − a⟩ⁿ, n < 0 : 집중력·집중모멘트의 "점" 효과 → 구간 내부에서는 항상 0
    구간 경계를 모든 하중·지점 위치로 잡았기 때문에, 한 구간 안에서는 어떤
    특이함수도 켜지거나 꺼지지 않는다 → 구간 전체에서 같은 다항식이 성립한다.

[Python 문법 메모]
    - expr.replace(조건함수, 바꿀함수)
        : 식 안의 모든 부분식 중 조건함수가 True 인 것을 바꿀함수 결과로 교체한다.
    - sp.expand(식) : 괄호를 전개해서 25 − 10x 처럼 정리한다.
    - sorted(set(목록)) : set 으로 중복을 없애고 정렬한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import sympy as sp
from sympy import SingularityFunction

from core.model import Beam, LinearLoad, MomentLoad, PointLoad, UniformLoad
from core.solver import SolverResult, exact


@dataclass
class Segment:
    """
    보의 한 구간과 그 구간의 V, M 다항식 (실무 규약).

    속성:
        x_start : 구간 시작 [m] (정확한 수)
        x_end   : 구간 끝 [m]
        V       : x_start < x < x_end 에서의 V(x) [kN]  (일반 다항식)
        M       : x_start < x < x_end 에서의 M(x) [kN·m]
        theta   : 처짐각 θ(x) [rad], 시계 + (EI 를 넣지 않았으면 None)
        delta   : 처짐 δ(x) [m], 아래 + (EI 를 넣지 않았으면 None)

    처짐·처짐각은 집중하중 위치에서도 연속이므로 좌우 구분이 필요 없다.
    구간 끝점에서 집중하중이 있으면 V 가 불연속(점프)이므로,
    끝점 값은 "그 구간 쪽에서 다가간 값(한쪽 극한)"으로 해석한다.
    """

    x_start: sp.Expr
    x_end: sp.Expr
    V: sp.Expr
    M: sp.Expr
    theta: sp.Expr | None = None
    delta: sp.Expr | None = None


def boundaries(beam: Beam) -> list[sp.Rational]:
    """
    구간 경계 위치 목록 (0, L, 지점 위치, 하중 위치·시작·끝)을 정렬해 돌려준다.

    구조역학적 의미:
        V, M 식의 모양이 바뀌는 곳은 하중이 시작·끝나거나 집중력이 작용하는 곳뿐이다.
        그 위치들로 보를 자르면 각 구간 안에서는 식이 하나로 정해진다.

    예: 단순보 L=8, 0~4 등분포 → [0, 4, 8]
    """
    L = exact(beam.L)
    points = {sp.Integer(0), L}  # { } : set (중복 없는 모음)
    for s in beam.supports:
        points.add(exact(s.x))
    for load in beam.loads:
        if isinstance(load, (PointLoad, MomentLoad)):
            points.add(exact(load.x))
        elif isinstance(load, (UniformLoad, LinearLoad)):
            points.add(exact(load.x_start))
            points.add(exact(load.x_end))
    return sorted(p for p in points if 0 <= p <= L)


def singularity_to_poly(expr: sp.Expr, x: sp.Symbol, x_mid: sp.Expr) -> sp.Expr:
    """
    특이함수 식을 x_mid 가 속한 구간의 일반 다항식으로 바꾼다.

    입력:
        expr  : ⟨x−a⟩ⁿ 이 섞인 SymPy 식
        x     : 위치 변수
        x_mid : 구간 안쪽의 한 점 (경계가 아닌 점이어야 한다)
    출력:
        특이함수가 없는 전개된 다항식

    예: expr = 30⟨x⟩¹ − 5⟨x⟩² + 5⟨x−4⟩², x_mid = 2
        → ⟨x−4⟩² 는 x_mid < 4 이므로 0 → 30x − 5x²
    """

    def unfold(f: SingularityFunction) -> sp.Expr:
        # SingularityFunction(변수, a, n) 의 args = (변수식, a, n)
        var, a, n = f.args
        if n < 0:
            # 집중력(n=−1)·집중모멘트(n=−2) 는 한 점에서만 값이 있으므로 구간 내부에서 0
            return sp.Integer(0)
        if var.subs(x, x_mid) > a:
            # 구간이 a 보다 오른쪽 → 이 항은 "켜진" 상태: (x − a)ⁿ
            return (var - a) ** n
        # 구간이 a 보다 왼쪽 → 아직 하중이 시작되지 않았으므로 0
        return sp.Integer(0)

    poly = expr.replace(lambda e: isinstance(e, SingularityFunction), unfold)
    return sp.expand(poly)


def split_segments(result: SolverResult) -> list[Segment]:
    """
    solver 결과를 구간별 다항식 목록으로 바꾼다.

    입력:  result — solver.solve() 결과
    출력:  Segment 목록 (왼쪽 구간부터)

    예 (단순보 L=8, 0~4 에 w=10):
        [Segment(0, 4, V=30−10x, M=30x−5x²),
         Segment(4, 8, V=−10,    M=80−10x)]
    """
    x = result.x
    pts = boundaries(result.beam)
    segments: list[Segment] = []
    # zip(pts[:-1], pts[1:]) : 이웃한 두 경계를 (왼쪽, 오른쪽) 쌍으로 묶는다
    #   pts[:-1] = 마지막 하나를 뺀 목록, pts[1:] = 첫 번째를 뺀 목록
    for x1, x2 in zip(pts[:-1], pts[1:]):
        x_mid = (x1 + x2) / 2
        V = singularity_to_poly(result.V, x, x_mid)
        M = singularity_to_poly(result.M, x, x_mid)
        theta = delta = None
        if result.delta is not None:  # E, I 를 넣은 경우에만
            theta = singularity_to_poly(result.theta, x, x_mid)
            delta = singularity_to_poly(result.delta, x, x_mid)
        segments.append(Segment(x_start=x1, x_end=x2, V=V, M=M, theta=theta, delta=delta))
    return segments


def value_at(
    segments: list[Segment],
    x: sp.Symbol,
    xv: float | sp.Expr,
    side: Literal["left", "right"] = "right",
) -> tuple[sp.Expr, sp.Expr]:
    """
    위치 xv 에서의 (V, M) 값을 구한다.

    입력:
        segments : split_segments() 결과
        x        : 위치 변수
        xv       : 값을 구할 위치 [m]
        side     : 경계 위에 있을 때 어느 쪽 극한을 쓸지
                   "right" = 오른쪽에서 다가간 값 (x⁺), "left" = 왼쪽 값 (x⁻)
    출력:
        (V, M) 정확한 수

    구조역학적 의미:
        집중하중 위치에서는 V 가 하중 크기만큼 점프하므로 좌·우 값이 다르다.
        예: 단순보 중앙 집중하중 P → V(L/2⁻) = +P/2, V(L/2⁺) = −P/2

    예: value_at(segs, x, 2)  → (V(2⁺), M(2))
    """
    xv = exact(xv) if not isinstance(xv, sp.Basic) else xv
    # 오른쪽 극한이면 "xv 에서 시작하는 구간", 왼쪽 극한이면 "xv 에서 끝나는 구간"을 쓴다
    chosen = None
    for seg in segments:
        if side == "right" and seg.x_start <= xv < seg.x_end:
            chosen = seg
            break
        if side == "left" and seg.x_start < xv <= seg.x_end:
            chosen = seg
            break
    if chosen is None:
        # 보 양 끝에서 반대쪽 극한을 요청한 경우(예: x=L 의 오른쪽)는 가장 가까운 구간 사용
        chosen = segments[-1] if xv >= segments[-1].x_end else segments[0]
    return chosen.V.subs(x, xv), chosen.M.subs(x, xv)


def deflection_at(segments: list[Segment], x: sp.Symbol, xv: float | sp.Expr) -> tuple[sp.Expr, sp.Expr]:
    """
    위치 xv 의 (처짐각 θ [rad], 처짐 δ [m]) 를 구한다. EI 가 없으면 ValueError.

    θ, δ 는 보 전체에서 연속이므로 경계 위의 점이라도 좌우 어느 구간 식을 써도 같다.

    예: 캔틸레버(왼쪽 고정) 자유단 → deflection_at(segs, x, L) = (wL³/6EI, wL⁴/8EI)
    """
    if segments[0].delta is None:
        raise ValueError("E, I 가 없어 처짐을 계산하지 않았습니다.")
    seg = segments[0]
    xv = exact(xv) if not isinstance(xv, sp.Basic) else xv
    for s in segments:
        if s.x_start <= xv <= s.x_end:
            seg = s
            break
    return seg.theta.subs(x, xv), seg.delta.subs(x, xv)
