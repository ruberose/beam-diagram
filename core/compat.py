"""
compat.py — 부정정보 반력: 변형일치법 (지시서 Phase 5B, 결정 D5)

[역할]
    부정정보의 반력을 손계산 순서(변형일치법)로 구한다. SymPy 해와 독립된 계산이며,
    steps.py 가 이 결과로 풀이 과정을 쓰고 SymPy 반력과 비교 검증한다.

[변형일치법 순서]
    1. 여분력(redundant) X₁, X₂, … 를 고른다 → 그 구속을 없앤 "이완 구조"는 정정보가 된다.
         고정-핀    : X₁ = 핀·롤러 반력          → 이완 구조 = 캔틸레버
         양단 고정  : X₁ = R_B, X₂ = M_B (오른쪽) → 이완 구조 = 왼쪽 고정 캔틸레버
         연속보     : X = 안쪽 지점 반력들         → 이완 구조 = 양 끝 지점의 단순보
    2. 이완 구조에 실제 하중만 걸 때 여분력 위치의 변위 Δᵢ₀ (처짐 또는 처짐각)
    3. 이완 구조에 단위 여분력 Xⱼ = 1 만 걸 때의 변위 fᵢⱼ (유연도 계수)
    4. 적합조건: 실제 구조에서는 그 위치가 움직이지 않으므로
           Δᵢ = Δᵢ₀ + Σⱼ fᵢⱼ·Xⱼ = 0    →  연립방정식으로 X 를 구한다
    5. 여분력을 하중처럼 이완 구조에 걸고 평형식으로 나머지 반력을 구한다.

    변위는 모두 EI 를 곱한 값(EI·Δ)으로 다룬다. EI 가 일정하면 양변의 EI 가 지워지므로
    반력은 E, I 값과 무관하다 (그래서 E, I 를 넣지 않아도 부정정보를 풀 수 있다).

[부호] 여분력: 힘은 위 +, 모멘트는 시계 + (반력 규약과 같다)
       변위: 처짐 δ 아래 +, 처짐각 θ 시계 +

[관계]
    - 이 파일을 부르는 곳: steps.py
    - 이 파일이 부르는 곳: macaulay.py, model.py, solver.py(exact), sympy
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from core.macaulay import (
    SupportPoint,
    boundary_conditions,
    determinate_reactions,
    elastic_curve,
    moment_expr,
)
from core.model import Beam, Load, MomentLoad, PointLoad


@dataclass
class Redundant:
    """
    여분력 하나.

    속성:
        name    : LaTeX 이름 (예: "R_{B}", "M_{B}")
        label   : 지점 이름 (예: "B")
        kind    : "force"(연직 반력, 위 +) 또는 "moment"(반력 모멘트, 시계 +)
        x       : 위치 [m]
    적합조건: force 면 그 위치의 처짐 δ = 0, moment 면 처짐각 θ = 0.
    """

    name: str
    label: str
    kind: str
    x: sp.Expr

    def as_load(self, value: sp.Expr = sp.Integer(1)) -> Load:
        """
        여분력(값 value, 기본 1 = 단위 여분력)을 이완 구조에 거는 '하중'으로 바꾼다.
        힘은 위 + 이고 하중은 아래 + 이므로 부호를 뒤집는다: 하중 P = −value.
        모멘트는 둘 다 시계 + 라서 그대로.
        (값은 정확한 분수 그대로 넣는다 — solver.exact() 가 SymPy 수를 그대로 받는다)
        """
        if self.kind == "force":
            return PointLoad(P=-value, x=self.x)
        return MomentLoad(M=value, x=self.x)

    def displacement_name(self) -> str:
        """적합조건에 쓰는 변위 이름 (LaTeX). 예: δ_B, θ_B"""
        return rf"\delta_{{{self.label}}}" if self.kind == "force" else rf"\theta_{{{self.label}}}"


@dataclass
class CompatResult:
    """
    변형일치법 결과.

    속성:
        primary     : 이완 구조의 지점 (정정)
        primary_text: 이완 구조 설명 (예: "고정단 A 의 캔틸레버")
        redundants  : 여분력 목록 X₁, X₂, …
        d0          : EI·Δᵢ₀ — 실제 하중에 의한 이완 구조 변위 목록
        F           : EI·fᵢⱼ — 유연도 계수 행렬 (F[i][j] = Xⱼ = 1 에 의한 Δᵢ)
        X           : 여분력 값 목록
        reactions   : 전체 지점 반력 {이름: (R, M 또는 None)}
    """

    primary: list[SupportPoint]
    primary_text: str
    redundants: list[Redundant]
    d0: list[sp.Expr]
    F: list[list[sp.Expr]]
    X: list[sp.Expr]
    reactions: dict[str, tuple[sp.Expr, sp.Expr | None]]


def choose_redundants(beam: Beam, supports: list[SupportPoint]) -> tuple[list[SupportPoint], list[Redundant], str]:
    """
    여분력과 이완 구조를 고른다 (위 docstring 의 표).

    반환: (이완 구조 지점 목록, 여분력 목록, 설명 문장)
    """
    kind = beam.kind()
    if kind == "propped":
        fixed = next(s for s in supports if s.kind == "fixed")
        other = next(s for s in supports if s.kind != "fixed")
        return [fixed], [Redundant(f"R_{{{other.label}}}", other.label, "force", other.x)], \
            f"지점 {other.label} 를 없앤 고정단 {fixed.label} 의 캔틸레버"
    if kind == "fixed_fixed":
        A, B = supports  # 왼쪽 A(x=0), 오른쪽 B(x=L)
        return [A], [
            Redundant(f"R_{{{B.label}}}", B.label, "force", B.x),
            Redundant(f"M_{{{B.label}}}", B.label, "moment", B.x),
        ], f"오른쪽 고정단 {B.label} 를 없앤 고정단 {A.label} 의 캔틸레버"
    if kind == "continuous":
        first, last = supports[0], supports[-1]
        inner = supports[1:-1]
        names = ", ".join(s.label for s in inner)
        return [first, last], [Redundant(f"R_{{{s.label}}}", s.label, "force", s.x) for s in inner], \
            f"안쪽 지점 {names} 를 없앤 지점 {first.label}·{last.label} 의 단순보"
    raise ValueError("정정보에는 변형일치법이 필요 없습니다.")


def primary_displacements(
    x: sp.Symbol,
    L: sp.Expr,
    primary: list[SupportPoint],
    loads: list[Load],
    redundants: list[Redundant],
) -> list[sp.Expr]:
    """
    이완 구조에 loads 가 걸렸을 때, 각 여분력 위치의 변위(EI 곱한 값)를 구한다.
    force 여분력 → EI·δ, moment 여분력 → EI·θ.  계산은 맥컬리법 (macaulay.py).
    """
    reac = determinate_reactions(primary, loads)
    M = moment_expr(x, L, primary, reac, loads)
    curve = elastic_curve(M, x, boundary_conditions(primary))
    out = []
    for r in redundants:
        expr = curve.EI_delta if r.kind == "force" else curve.EI_theta
        out.append(sp.simplify(expr.subs(x, r.x)))  # 정확한 분수 그대로 (nsimplify 는 근사하므로 쓰지 않는다)
    return out


def solve_compat(beam: Beam, x: sp.Symbol, L: sp.Expr, supports: list[SupportPoint]) -> CompatResult:
    """
    변형일치법으로 부정정보의 반력을 모두 구한다.

    입력:
        beam     : 부정정보 (validate 통과)
        x, L     : 위치 변수, 경간
        supports : 모든 지점 (이름·종류·위치, 왼쪽부터)
    출력:
        CompatResult

    예 (고정-핀, 등분포 w, 경간 L):
        X₁ = R_B, EI·Δ₁₀ = wL⁴/8, EI·f₁₁ = −L³/3  →  R_B = 3wL/8
    """
    primary, reds, text = choose_redundants(beam, supports)

    # ── 1. 실제 하중에 의한 변위 Δᵢ₀ ──
    d0 = primary_displacements(x, L, primary, beam.loads, reds)

    # ── 2. 단위 여분력에 의한 변위 fᵢⱼ (j 열 = Xⱼ = 1 만 걸었을 때) ──
    cols = [primary_displacements(x, L, primary, [rj.as_load()], reds) for rj in reds]
    F = [[cols[j][i] for j in range(len(reds))] for i in range(len(reds))]

    # ── 3. 적합조건 Δᵢ₀ + Σ fᵢⱼ Xⱼ = 0 → X ──
    Xs = sp.symbols(f"X1:{len(reds) + 1}")  # X1, X2, ... 기호
    eqs = [d0[i] + sum(F[i][j] * Xs[j] for j in range(len(reds))) for i in range(len(reds))]
    sol = sp.solve(eqs, Xs, dict=True)[0]
    X = [sp.simplify(sol[s]) for s in Xs]

    # ── 4. 여분력을 하중으로 걸고 이완 구조의 평형식 → 나머지 반력 ──
    loads_all = list(beam.loads) + [r.as_load(v) for r, v in zip(reds, X)]
    reactions = dict(determinate_reactions(primary, loads_all))
    for r, v in zip(reds, X):
        R, M = reactions.get(r.label, (None, None))
        if r.kind == "force":
            reactions[r.label] = (v, M)
        else:
            reactions[r.label] = (R, v)

    return CompatResult(primary, text, reds, d0, F, X, reactions)
