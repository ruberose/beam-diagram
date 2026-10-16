"""
macaulay.py — 맥컬리법(특이함수) 손계산 도구 모음

[역할]
    SymPy Beam 을 쓰지 않고, 하중 목록에서 직접 특이함수 식을 만들고 적분하는 함수들.
    "SymPy 결과와 독립적으로 손계산을 다시 해서 비교"하기 위한 두 번째 계산 경로다.
        - 정정보 반력 (평형식)
        - 휨모멘트 M(x) 특이함수 식
        - 두 번 적분 → EI·θ(x), EI·δ(x) 와 적분상수 C₁, C₂
    처짐 풀이(steps.py 5단계)와 부정정보 변형일치법(compat.py)이 함께 쓴다.

[관계]
    - 이 파일을 부르는 곳: steps.py, compat.py
    - 이 파일이 부르는 곳: model.py(하중 종류), solver.py(exact, load_terms), sympy

[부호] 실무 규약: 하중·힘 아래 +, 반력 위 +, 모멘트 시계 +, M 하부 인장 +,
       처짐 δ 아래 +, 처짐각 θ = dδ/dx 시계 +  →  기본식 EI·δ'' = −M

[특이함수 적분 규칙]
    ∫⟨x − a⟩ⁿ dx = ⟨x − a⟩ⁿ⁺¹ / (n + 1)   (n ≥ 0)
    x < a 에서 0 이 되도록 정의되므로 보 전체에서 한 식으로 적분된다.
    → 구간이 몇 개든 적분상수는 C₁, C₂ 두 개뿐이다 (맥컬리법의 장점).
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from core.model import LinearLoad, Load, MomentLoad, PointLoad, UniformLoad
from core.solver import exact, load_terms

SF = sp.SingularityFunction


@dataclass
class SupportPoint:
    """
    평형식·경계조건에 쓰는 지점 정보 (이름, 종류, 위치).
    model.Support 와 비슷하지만 이름(label)과 정확한 위치(분수)를 함께 들고 다닌다.
    """

    label: str
    kind: str
    x: sp.Expr


# ─────────────────────────────────────────────────────────────
# 하중 합력 (평형식용)
# ─────────────────────────────────────────────────────────────
def load_resultants(loads: list[Load]) -> tuple[list[tuple[sp.Expr, sp.Expr]], list[tuple[sp.Expr, sp.Expr]]]:
    """
    하중 목록 → (집중 힘 목록 [(F 아래+, x)], 모멘트 목록 [(M 시계+, x)]).

    분포하중은 합력과 작용점으로 바꾼다. 선형 분포는 직사각형(w₁) + 삼각형(w₂ − w₁)
    두 조각으로 나눈다 (w₁ + w₂ = 0 처럼 합력이 0 이어도 모멘트가 남는 경우까지 정확).
    """
    forces: list[tuple[sp.Expr, sp.Expr]] = []
    moments: list[tuple[sp.Expr, sp.Expr]] = []
    for ld in loads:
        if isinstance(ld, PointLoad):
            forces.append((exact(ld.P), exact(ld.x)))
        elif isinstance(ld, MomentLoad):
            moments.append((exact(ld.M), exact(ld.x)))
        elif isinstance(ld, UniformLoad):
            a, b = exact(ld.x_start), exact(ld.x_end)
            forces.append((exact(ld.w) * (b - a), (a + b) / 2))
        elif isinstance(ld, LinearLoad):
            a, b = exact(ld.x_start), exact(ld.x_end)
            w1, w2 = exact(ld.w_start), exact(ld.w_end)
            ell = b - a
            forces.append((w1 * ell, a + ell / 2))                  # 직사각형
            forces.append(((w2 - w1) * ell / 2, a + 2 * ell / 3))   # 삼각형 (0 → w₂−w₁)
    return forces, moments


def determinate_reactions(supports: list[SupportPoint], loads: list[Load]) -> dict[str, tuple[sp.Expr, sp.Expr | None]]:
    """
    정정 구조(캔틸레버 또는 지점 2개 보)의 반력을 평형식으로 구한다.

    입력:
        supports : 고정단 1개 또는 핀·롤러 2개
        loads    : 하중 목록 (단위 여분력도 PointLoad·MomentLoad 로 넣을 수 있다)
    출력:
        {지점 이름: (R 위+, M 시계+ 또는 None)}

    구조역학적 의미:
        캔틸레버(고정단 A): ΣF_y = 0 → R_A = ΣF,  ΣM_A = 0 → M_A = −(ΣF·a + ΣM)
        지점 2개(A, B)     : ΣM_A = 0 → R_B = (ΣF·a + ΣM)/ℓ_AB,  ΣF_y = 0 → R_A = ΣF − R_B
        (a = 지점 A 에서 힘까지 거리, 오른쪽 +)
    """
    forces, moments = load_resultants(loads)
    F_sum = sum((F for F, _ in forces), sp.Integer(0))
    if len(supports) == 1:
        A = supports[0]
        M_sum = sum((F * (xf - A.x) for F, xf in forces), sp.Integer(0)) + sum((M for M, _ in moments), sp.Integer(0))
        return {A.label: (F_sum, -M_sum)}
    A, B = supports
    M_sum = sum((F * (xf - A.x) for F, xf in forces), sp.Integer(0)) + sum((M for M, _ in moments), sp.Integer(0))
    R_B = M_sum / (B.x - A.x)
    return {A.label: (F_sum - R_B, None), B.label: (R_B, None)}


# ─────────────────────────────────────────────────────────────
# 휨모멘트 특이함수 식
# ─────────────────────────────────────────────────────────────
def moment_expr(
    x: sp.Symbol,
    L: sp.Expr,
    supports: list[SupportPoint],
    reactions: dict[str, tuple[sp.Expr, sp.Expr | None]],
    loads: list[Load],
) -> sp.Expr:
    """
    반력과 하중으로 휨모멘트 M(x) 를 특이함수 식 하나로 만든다 (단면 왼쪽 자유물체도).

        반력 R (위)             : + R⟨x − a⟩¹
        반력 모멘트 M_A (시계)  : + M_A⟨x − a⟩⁰
        모멘트 하중 M₀ (시계)   : + M₀⟨x − a⟩⁰
        하중 세기 q⟨x − a⟩ⁿ (아래): − q⟨x − a⟩ⁿ⁺² / (n+2)!
            집중하중(n=−1): −P⟨x − a⟩¹,  등분포(n=0): −w⟨x − a⟩²/2,  선형(n=1): −s⟨x − a⟩³/6
    """
    M = sp.Integer(0)
    for s in supports:
        R, M_r = reactions[s.label]
        M += R * SF(x, s.x, 1)
        if M_r is not None:
            M += M_r * SF(x, s.x, 0)
    for ld in loads:
        for kind, value, start, order in load_terms(ld, L):
            if kind == "moment":
                M += value * SF(x, start, 0)
            else:
                M -= value * SF(x, start, order + 2) / sp.factorial(order + 2)
    return M


def integrate_sf(expr: sp.Expr) -> sp.Expr:
    """
    특이함수 식을 적분한다 (적분상수 없이):  ∫⟨x − a⟩ⁿ dx = ⟨x − a⟩ⁿ⁺¹ / (n + 1)   (n ≥ 0)
    """
    out = sp.Integer(0)
    for term in sp.Add.make_args(sp.expand(expr)):
        if term == 0:
            continue
        c, f = term.as_coeff_Mul()  # 숫자 계수와 나머지(특이함수)로 나눈다
        if not isinstance(f, SF):
            raise ValueError(f"특이함수가 아닌 항은 적분할 수 없습니다: {term}")
        var, a, n = f.args
        if n < 0:
            raise ValueError("음의 차수 특이함수는 M(x) 에 나오지 않아야 합니다.")
        out += c * SF(var, a, n + 1) / (n + 1)
    return out


# ─────────────────────────────────────────────────────────────
# 탄성곡선 (두 번 적분 + 경계조건)
# ─────────────────────────────────────────────────────────────
@dataclass
class ElasticCurve:
    """
    맥컬리법 결과 (모두 EI 를 곱한 값 → EI 와 무관한 숫자).

    속성:
        EI_theta0 : −∫M dx (적분상수 뺀 부분)
        EI_delta0 : −∬M dx (적분상수 뺀 부분)
        C1, C2    : 적분상수
        EI_theta  : EI·θ(x) = EI_theta0 + C1
        EI_delta  : EI·δ(x) = EI_delta0 + C1·x + C2
    """

    EI_theta0: sp.Expr
    EI_delta0: sp.Expr
    C1: sp.Expr
    C2: sp.Expr
    EI_theta: sp.Expr
    EI_delta: sp.Expr


def boundary_conditions(supports: list[SupportPoint]) -> list[tuple[str, sp.Expr]]:
    """
    적분상수 두 개를 정할 경계조건 두 개를 고른다.
        고정단이 있으면: 그 고정단의 θ = 0, δ = 0
        없으면        : 처음 두 지점의 δ = 0
    (부정정보의 나머지 지점 조건은 반력을 구할 때 적합조건으로 이미 쓰였다.)
    """
    fixed = [s for s in supports if s.kind == "fixed"]
    if fixed:
        return [("theta", fixed[0].x), ("delta", fixed[0].x)]
    return [("delta", supports[0].x), ("delta", supports[1].x)]


def elastic_curve(M: sp.Expr, x: sp.Symbol, bcs: list[tuple[str, sp.Expr]]) -> ElasticCurve:
    """
    EI·δ'' = −M 을 두 번 적분하고 경계조건 두 개로 C₁, C₂ 를 정한다.

    입력:
        M   : 휨모멘트 특이함수 식
        bcs : [("theta" 또는 "delta", 위치), ...] 두 개
    출력:
        ElasticCurve

    예: 단순보 등분포(L=6, w=10): C₁ = wL³/24 = 90,  C₂ = 0
    """
    C1, C2 = sp.symbols("C_1 C_2")
    th0 = -integrate_sf(M)
    de0 = integrate_sf(th0)
    eqs = []
    for what, x0 in bcs:
        if what == "theta":
            eqs.append(th0.subs(x, x0) + C1)
        else:
            eqs.append(de0.subs(x, x0) + C1 * x0 + C2)
    sol = sp.solve(eqs, [C1, C2], dict=True)[0]
    c1, c2 = sol[C1], sol[C2]
    return ElasticCurve(th0, de0, c1, c2, th0 + c1, de0 + c1 * x + c2)
