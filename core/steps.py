"""
steps.py — 풀이 과정(손계산 순서)을 LaTeX 문자열 목록으로 만든다

[역할]
    구조검토서 손계산과 같은 순서로 풀이 과정을 만든다.
        1. 하중 정리      : 분포하중 → 합력 W 와 작용점 x̄
        2. 반력 계산      : ΣM_A = 0 → R_B,  ΣF_y = 0 → R_A  (캔틸레버는 ΣF_y, ΣM)
                            부정정보는 변형일치법 (Phase 5B, compat.py)
        3. 구간별 V, M    : 단면 왼쪽 자유물체도(FBD)로 식을 세운다
        4. 최대값         : V = 0 위치 → M 대입, 경계값 후보 비교
        5. 처짐           : 맥컬리법 (E, I 를 넣었을 때만, Phase 5A)
        6. 특이함수 식    : ⟨x − a⟩ⁿ 형태 (화면에서는 기본 숨김)
    단계마다 "일반식 → 숫자 대입 → 결과" 순서로 쓴다.

    중요: 반력과 구간식은 SymPy 결과를 그대로 옮기지 않고, 하중 목록에서
    "직접" 평형식·자유물체도 식을 만들어 계산한다. 그런 다음 SymPy(solver) 결과와
    정확히 같은지 비교하고, 다르면 경고를 남긴다. (서로 독립된 두 계산이 일치 → 검증)

[관계]
    - 이 파일을 부르는 곳: demo.py, tests/test_steps.py, (나중에) web/bridge.py
    - 이 파일이 부르는 곳: model.py, solver.py(SolverResult, exact),
                          piecewise.py(Segment), extremes.py(Extremes),
                          macaulay.py(맥컬리법), compat.py(변형일치법), sympy

[출력 형식]
    Step 목록. Step 하나 = 제목 + StepLine 목록.
    StepLine = (설명 텍스트, LaTeX 수식). 웹에서는 KaTeX 로 수식을 그린다.
    LaTeX 문자열 안의 역슬래시(\\)는 Python 문자열에서 특별한 뜻이 있으므로
    r"..." (raw 문자열) 로 써서 그대로 두었다.

[Python 문법 메모]
    - r"\\frac{a}{b}" : raw 문자열. 역슬래시를 이스케이프 문자로 해석하지 않는다.
    - f"{값:.2f}"     : f-문자열 서식. 값을 소수 2자리로 표시한다.
    - f-문자열 안에서 LaTeX 의 중괄호 { } 를 글자로 쓰려면 {{ }} 처럼 두 번 쓴다.
    - sp.Poly(식, x).all_coeffs() : 다항식 계수를 높은 차수부터 목록으로 돌려준다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import sympy as sp

from core.compat import solve_compat
from core.extremes import DeflectionExtremes, Extremes, ValueAt
from core.macaulay import SupportPoint, boundary_conditions, integrate_sf, moment_expr
from core.model import LinearLoad, MomentLoad, PointLoad, UniformLoad
from core.piecewise import Segment, singularity_to_poly
from core.solver import SolverResult, exact

DIGITS = 2  # 결과값 표시 소수 자리 (지시서 5장: 기본 2자리)


# ─────────────────────────────────────────────────────────────
# 출력 데이터
# ─────────────────────────────────────────────────────────────
@dataclass
class StepLine:
    """
    풀이 한 줄.

    속성:
        latex : KaTeX 로 그릴 수식 (빈 문자열이면 텍스트만)
        text  : 수식 앞에 붙는 한국어 설명 (빈 문자열 가능)
    """

    latex: str
    text: str = ""


@dataclass
class Step:
    """
    풀이 단계 하나 (화면에서 접기/펼치기 단위).

    속성:
        key    : 단계 식별자 "loads" / "reactions" / "segments" / "extremes" / "singularity"
        title  : 제목 (예: "2. 반력 계산")
        lines  : StepLine 목록
        hidden : True 면 화면에서 기본 숨김 (특이함수 식)
    """

    key: str
    title: str
    lines: list[StepLine] = field(default_factory=list)
    hidden: bool = False


@dataclass
class StepsResult:
    """build_steps() 결과: 단계 목록 + 검증 경고 + 손계산 반력."""

    steps: list[Step]
    warnings: list[str]
    hand_reactions: dict[str, tuple[sp.Expr, sp.Expr | None]]


# ─────────────────────────────────────────────────────────────
# 숫자·수식 서식 도우미
# ─────────────────────────────────────────────────────────────
def num(value: sp.Expr | float, digits: int = DIGITS) -> str:
    """
    값을 소수 digits 자리 문자열로. −0.00 은 0.00 으로 바꾼다.
    예: num(sp.Rational(45, 1)) → "45.00"
    """
    v = round(float(value), digits)
    if v == 0:
        v = 0.0
    return f"{v:.{digits}f}"


def coef(value: sp.Expr | float) -> str:
    """
    식 안의 계수 표시용: 소수 4자리까지, 뒤의 0 은 지운다.
    손계산 식처럼 "25 − 10x" 로 보이게 하려는 것. 2/3 같은 값은 0.6667 로 표시.
    예: coef(10) → "10",  coef(sp.Rational(2,3)) → "0.6667"
    """
    v = round(float(value), 4)
    if v == 0:
        v = 0.0
    s = f"{v:.4f}".rstrip("0").rstrip(".")
    return s


def paren(value: sp.Expr | float, digits: int = DIGITS) -> str:
    """음수면 괄호로 감싼다. 곱셈 대입식에서 '× (−1.20)' 처럼 쓰려고."""
    s = num(value, digits)
    return f"({s})" if float(value) < 0 else s


def shift(a: sp.Expr) -> str:
    """(x − a) 를 LaTeX 로. a = 0 이면 그냥 x."""
    if a == 0:
        return "x"
    return f"(x - {num(a)})"


def shift_pow(a: sp.Expr, n: int) -> str:
    """(x − a)ⁿ 를 LaTeX 로. 예: shift_pow(0, 2) → 'x^{2}', shift_pow(2, 2) → '(x - 2.00)^{2}'"""
    return f"{shift(a)}^{{{n}}}"


def join_terms(terms: list[tuple[int, str]]) -> str:
    """
    (부호, 크기 문자열) 목록을 '+ / −' 로 이어 붙인다.
    예: [(+1,"30"), (-1,"10x")] → "30 - 10x",  빈 목록 → "0"
    """
    if not terms:
        return "0"
    out = ""
    for i, (sgn, body) in enumerate(terms):
        if i == 0:
            out += body if sgn > 0 else f"-{body}"
        else:
            out += f" + {body}" if sgn > 0 else f" - {body}"
    return out


def _ascending_coeffs(expr: sp.Expr, x: sp.Symbol) -> list[tuple[int, sp.Expr]]:
    """
    다항식의 (차수, 계수) 목록을 낮은 차수부터 돌려준다. 계수 0 인 항은 뺀다.
    손계산 식은 보통 '25 − 10x' 처럼 상수항부터 쓰므로 이 순서를 쓴다.

    [문법] reversed(목록) : 목록을 거꾸로 순회한다. all_coeffs() 는 높은 차수부터이므로 뒤집는다.
    """
    coeffs = sp.Poly(sp.expand(expr), x).all_coeffs()
    return [(n, c) for n, c in enumerate(reversed(coeffs)) if c != 0]


def poly_latex(expr: sp.Expr, x: sp.Symbol) -> str:
    """
    다항식을 '25 − 10x' 같은 LaTeX 로 (낮은 차수부터).
    예: poly_latex(-5*x**2 + 30*x, x) → "30\\,x - 5\\,x^{2}"
    """
    if sp.expand(expr) == 0:
        return "0"
    terms: list[tuple[int, str]] = []
    for n, c in _ascending_coeffs(expr, x):
        mag = abs(c)
        if n == 0:
            body = coef(mag)
        else:
            xs = "x" if n == 1 else f"x^{{{n}}}"
            body = xs if mag == 1 else f"{coef(mag)}\\,{xs}"
        terms.append((1 if c > 0 else -1, body))
    return join_terms(terms)


def poly_subst_latex(expr: sp.Expr, x: sp.Symbol, xv: sp.Expr) -> str:
    """
    다항식에 x = xv 를 대입한 모습을 LaTeX 로 (계산 전).
    예: 30x − 5x², xv=3 → "30 \\times 3.00 - 5 \\times 3.00^{2}"
    """
    if sp.expand(expr) == 0:
        return "0"
    xs_val = paren(xv)
    terms: list[tuple[int, str]] = []
    for n, c in _ascending_coeffs(expr, x):
        mag = abs(c)
        if n == 0:
            body = coef(mag)
        else:
            xs = xs_val if n == 1 else f"{xs_val}^{{{n}}}"
            body = xs if mag == 1 else f"{coef(mag)} \\times {xs}"
        terms.append((1 if c > 0 else -1, body))
    return join_terms(terms)


KN = r"\ \text{kN}"
KNM = r"\ \text{kN}\cdot\text{m}"
M_UNIT = r"\ \text{m}"


def moment_direction(M: sp.Expr, with_value: bool = True) -> str:
    """
    모멘트 값(시계 +)의 회전 방향을 LaTeX 로.
    with_value=False 면 크기 없이 기호와 방향만.
    예: −41 → "\\circlearrowleft\\ 41.00 kN·m (반시계)",  (5, False) → "\\circlearrowright\\ 시계"
    """
    if float(M) == 0:
        return ""
    if float(M) > 0:
        arrow, word = r"\circlearrowright", "시계"
    else:
        arrow, word = r"\circlearrowleft", "반시계"
    if not with_value:
        return rf"{arrow}\ \text{{{word}}}"
    return rf"{arrow}\ {num(abs(M))}{KNM}\ (\text{{{word}}})"


def moment_direction_text(M: sp.Expr) -> str:
    """moment_direction 의 일반 텍스트판 (콘솔 출력용). 예: −41 → '반시계 ↺ 41.00 kN·m'"""
    if float(M) == 0:
        return "0"
    if float(M) > 0:
        return f"시계 ↻ {num(abs(M))} kN·m"
    return f"반시계 ↺ {num(abs(M))} kN·m"


def x_label(v: ValueAt) -> str:
    """위치 표시: 경계 좌우 구분이 있으면 x⁻ / x⁺ 로."""
    sup = {"left": "^{-}", "right": "^{+}"}.get(v.side, "")
    return f"{num(v.x)}{sup}"


# ─────────────────────────────────────────────────────────────
# 1단계: 하중 정리
# ─────────────────────────────────────────────────────────────
@dataclass
class Resultant:
    """
    반력 계산에 쓰는 집중 힘 하나 (집중하중 또는 분포하중의 합력).

    속성:
        name : LaTeX 이름 (예: "P_{1}", "W_{2}", "W_{3a}")
        F    : 크기 [kN], 아래 +
        x    : 작용 위치 [m]
    """

    name: str
    F: sp.Expr
    x: sp.Expr


@dataclass
class AppliedMoment:
    """집중 모멘트 하중 하나. name: LaTeX 이름, M: 시계 + [kN·m], x: 위치 [m]"""

    name: str
    M: sp.Expr
    x: sp.Expr


@dataclass
class LoadSummary:
    """
    하중 정리 결과.

    속성:
        forces   : Resultant 목록 (반력 계산용)
        moments  : AppliedMoment 목록
        by_load  : 입력 하중 번호(1부터) → 그 하중이 만든 Resultant 목록
                   (구간식에서 "완전히 왼쪽에 있는 분포하중"을 합력으로 쓸 때 필요)
        lines    : 풀이 줄
    """

    forces: list[Resultant]
    moments: list[AppliedMoment]
    by_load: dict[int, list[Resultant]]
    lines: list[StepLine]


def summarize_loads(result: SolverResult) -> LoadSummary:
    """
    하중을 합력과 작용점으로 정리한다.

    구조역학적 의미:
        등분포 w, 길이 ℓ : W = w·ℓ,  x̄ = a + ℓ/2
        삼각형 0→w       : W = ½·w·ℓ, x̄ = a + 2ℓ/3 (큰 쪽에서 ℓ/3)
        사다리꼴 w₁→w₂   : 직사각형(작은 값) + 삼각형(차이) 으로 나눈다

    예: UniformLoad(10, 0, 4) → W = 10 × 4.00 = 40.00 kN, x̄ = 0 + 4.00/2 = 2.00 m
    """
    forces: list[Resultant] = []
    moments: list[AppliedMoment] = []
    by_load: dict[int, list[Resultant]] = {}
    lines: list[StepLine] = []

    for k, load in enumerate(result.beam.loads, start=1):  # start=1 : 번호를 1부터
        if isinstance(load, PointLoad):
            P, xp = exact(load.P), exact(load.x)
            r = Resultant(f"P_{{{k}}}", P, xp)
            forces.append(r)
            by_load[k] = [r]
            lines.append(StepLine(
                rf"P_{{{k}}} = {num(P)}{KN},\quad x = {num(xp)}{M_UNIT}",
                f"하중 {k} (집중하중)",
            ))

        elif isinstance(load, MomentLoad):
            M, xm = exact(load.M), exact(load.x)
            moments.append(AppliedMoment(f"M_{{{k}}}", M, xm))
            lines.append(StepLine(
                rf"M_{{{k}}} = {num(M)}{KNM}\ ({moment_direction(M, with_value=False)}),\quad x = {num(xm)}{M_UNIT}",
                f"하중 {k} (집중 모멘트, 시계 +)",
            ))

        elif isinstance(load, UniformLoad):
            w, a, b = exact(load.w), exact(load.x_start), exact(load.x_end)
            ell = b - a
            W, xbar = w * ell, a + ell / 2
            r = Resultant(f"W_{{{k}}}", W, xbar)
            forces.append(r)
            by_load[k] = [r]
            lines.append(StepLine(
                rf"W_{{{k}}} = w\,\ell = {num(w)} \times {num(ell)} = {num(W)}{KN}",
                f"하중 {k} (등분포 w = {num(w)} kN/m, {num(a)} ~ {num(b)} m)",
            ))
            lines.append(StepLine(
                rf"\bar{{x}}_{{{k}}} = a + \frac{{\ell}}{{2}} = {num(a)} + \frac{{{num(ell)}}}{{2}} = {num(xbar)}{M_UNIT}"
            ))

        elif isinstance(load, LinearLoad):
            pieces = _linear_pieces(k, load)
            forces.extend(p for p, _ in pieces)
            by_load[k] = [p for p, _ in pieces]
            w1, w2 = exact(load.w_start), exact(load.w_end)
            lines.append(StepLine(
                "",
                f"하중 {k} (선형 분포 {num(w1)} → {num(w2)} kN/m, "
                f"{num(load.x_start)} ~ {num(load.x_end)} m)"
                + (" — 직사각형 + 삼각형으로 나눔" if len(pieces) > 1 else ""),
            ))
            for _, piece_lines in pieces:
                lines.extend(piece_lines)

    return LoadSummary(forces, moments, by_load, lines)


def _linear_pieces(k: int, load: LinearLoad) -> list[tuple[Resultant, list[StepLine]]]:
    """
    선형 분포하중을 직사각형/삼각형 합력으로 나눈다.

    - w₁ = w₂        : 직사각형 하나
    - 한쪽이 0        : 삼각형 하나
    - 그 밖(사다리꼴) : 직사각형(값이 작은 쪽 w_min) + 삼각형(차이 Δw, 큰 쪽이 꼭짓점)
    삼각형 작용점: 큰 쪽(꼭짓점 반대편 = 높은 쪽)에서 ℓ/3 → 증가형이면 a + 2ℓ/3, 감소형이면 a + ℓ/3
    """
    w1, w2 = exact(load.w_start), exact(load.w_end)
    a, b = exact(load.x_start), exact(load.x_end)
    ell = b - a
    out: list[tuple[Resultant, list[StepLine]]] = []

    def rect(name: str, w: sp.Expr) -> None:
        W, xbar = w * ell, a + ell / 2
        out.append((Resultant(name, W, xbar), [
            StepLine(rf"{name} = w\,\ell = {num(w)} \times {num(ell)} = {num(W)}{KN}", "직사각형 부분"),
            StepLine(rf"\bar{{x}} = a + \frac{{\ell}}{{2}} = {num(a)} + \frac{{{num(ell)}}}{{2}} = {num(xbar)}{M_UNIT}"),
        ]))

    def tri(name: str, dw: sp.Expr, increasing: bool) -> None:
        W = dw * ell / 2
        if increasing:
            xbar = a + 2 * ell / 3
            xl = rf"\bar{{x}} = a + \frac{{2\ell}}{{3}} = {num(a)} + \frac{{2 \times {num(ell)}}}{{3}} = {num(xbar)}{M_UNIT}"
        else:
            xbar = a + ell / 3
            xl = rf"\bar{{x}} = a + \frac{{\ell}}{{3}} = {num(a)} + \frac{{{num(ell)}}}{{3}} = {num(xbar)}{M_UNIT}"
        out.append((Resultant(name, W, xbar), [
            StepLine(rf"{name} = \frac{{1}}{{2}}\,\Delta w\,\ell = \frac{{1}}{{2}} \times {num(dw)} \times {num(ell)} = {num(W)}{KN}", "삼각형 부분"),
            StepLine(xl),
        ]))

    if w1 == w2:
        rect(f"W_{{{k}}}", w1)
    elif w1 == 0:
        tri(f"W_{{{k}}}", w2, increasing=True)
    elif w2 == 0:
        tri(f"W_{{{k}}}", w1, increasing=False)
    else:
        w_min = min(w1, w2)
        rect(f"W_{{{k}a}}", w_min)
        tri(f"W_{{{k}b}}", max(w1, w2) - w_min, increasing=(w2 > w1))
    return out


# ─────────────────────────────────────────────────────────────
# 2단계: 반력 계산 (하중 목록에서 직접 평형식을 세운다)
# ─────────────────────────────────────────────────────────────
def _moment_terms(summary: LoadSummary, x_ref: sp.Expr) -> tuple[list[tuple[int, str]], sp.Expr]:
    """
    기준점 x_ref 에 대한 하중 모멘트 합 Σ F·a + Σ M (시계 +) 을 만든다.

    아래로 향하는 힘 F 가 기준점 오른쪽 거리 a 에 있으면 시계방향 모멘트 F·a.
    (왼쪽이면 a 가 음수 → 반시계)

    반환: (숫자 대입 항 목록 [(부호, '40.00 × 2.00'), ...], 정확한 합)
    """
    nums: list[tuple[int, str]] = []
    total = sp.Integer(0)
    for f in summary.forces:
        arm = f.x - x_ref  # 팔길이 a_i (오른쪽 +)
        total += f.F * arm
        nums.append((1 if f.F >= 0 else -1, f"{num(abs(f.F))} \\times {paren(arm)}"))
    for m in summary.moments:
        total += m.M
        nums.append((1 if m.M >= 0 else -1, num(abs(m.M))))
    return nums, total


def _reactions_simple(primary: list, summary: LoadSummary) -> tuple[dict, list[StepLine]]:
    """
    지점 2개(A, B) 보의 반력 — 단순보·내민보, 그리고 연속보의 이완 구조.

    ΣM_A = 0 (시계 +):  Σ F_i·a_i + Σ M_j − R_B·ℓ_AB = 0  →  R_B = (Σ F_i·a_i + Σ M_j) / ℓ_AB
    ΣF_y = 0 (위 +)  :  R_A + R_B − Σ F_i = 0           →  R_A = Σ F_i − R_B
    a_i : 지점 A 에서 하중 i 까지 거리 (오른쪽 +)

    primary: 지점 2개 (label, x 속성이 있는 객체). 이름은 지점 이름을 그대로 쓴다
             (연속보 이완 구조면 A 와 D 처럼).
    """
    A, B = primary[0], primary[1]
    a, b = A.label, B.label
    lAB = B.x - A.x
    lines: list[StepLine] = []

    # ── ΣM_A = 0 → R_B ──
    nums, Msum = _moment_terms(summary, A.x)
    R_B = Msum / lAB
    lines.append(StepLine(
        rf"\sum M_{a} = \sum F_i\,a_i + \sum M_j - R_{b}\,\ell_{{{a}{b}}} = 0",
        f"ΣM_{a} = 0 ({a}점 기준, 시계 +; a_i = {a}점에서 하중까지 거리, ℓ_{a}{b} = {num(lAB)} m)",
    ))
    lines.append(StepLine(rf"{join_terms(nums)} - R_{b} \times {num(lAB)} = 0"))
    lines.append(StepLine(
        rf"R_{b} = \frac{{{num(Msum)}}}{{{num(lAB)}}} = {num(R_B)}{KN}"
        + (r"\ (\downarrow)" if R_B < 0 else ""),
    ))

    # ── ΣF_y = 0 → R_A ──
    Fsum = sum((f.F for f in summary.forces), sp.Integer(0))
    R_A = Fsum - R_B
    f_nums = join_terms([(1 if f.F >= 0 else -1, num(abs(f.F))) for f in summary.forces])
    lines.append(StepLine(rf"R_{a} + R_{b} - \sum F_i = 0", "ΣF_y = 0 (위 +)"))
    lines.append(StepLine(rf"R_{a} + {paren(R_B)} - ({f_nums}) = 0"))
    lines.append(StepLine(
        rf"R_{a} = {num(Fsum)} - {paren(R_B)} = {num(R_A)}{KN}"
        + (r"\ (\downarrow)" if R_A < 0 else ""),
    ))
    return {a: (R_A, None), b: (R_B, None)}, lines


def _reactions_cantilever(primary: list, summary: LoadSummary) -> tuple[dict, list[StepLine]]:
    """
    고정단 하나(A)의 반력 — 캔틸레버, 그리고 고정-핀·양단 고정의 이완 구조.

    ΣF_y = 0 (위 +)  :  R_A − Σ F_i = 0                →  R_A = Σ F_i
    ΣM_A = 0 (시계 +):  M_A + Σ F_i·a_i + Σ M_j = 0     →  M_A = −(Σ F_i·a_i + Σ M_j)
    M_A 는 고정단이 보에 가하는 반력 모멘트 (시계 +). 회전 방향도 함께 표시한다.
    """
    A = primary[0]
    a = A.label
    lines: list[StepLine] = []

    Fsum = sum((f.F for f in summary.forces), sp.Integer(0))
    f_nums = join_terms([(1 if f.F >= 0 else -1, num(abs(f.F))) for f in summary.forces])
    lines.append(StepLine(rf"R_{a} - \sum F_i = 0", "ΣF_y = 0 (위 +)"))
    if len(summary.forces) > 1:
        lines.append(StepLine(rf"R_{a} = {f_nums} = {num(Fsum)}{KN}"))
    else:  # 하중이 하나뿐이면 '18.00 = 18.00' 처럼 같은 값이 두 번 나오지 않게
        lines.append(StepLine(rf"R_{a} = {num(Fsum)}{KN}"))

    nums, Msum = _moment_terms(summary, A.x)
    M_A = -Msum
    lines.append(StepLine(
        rf"\sum M_{a} = M_{a} + \sum F_i\,a_i + \sum M_j = 0",
        f"ΣM_{a} = 0 (고정단 기준, 시계 +; a_i = 고정단에서 하중까지 거리, 오른쪽 +)",
    ))
    lines.append(StepLine(rf"M_{a} + ({join_terms(nums)}) = 0"))
    direction = moment_direction(M_A)
    lines.append(StepLine(
        rf"M_{a} = -({num(Msum)}) = {num(M_A)}{KNM}" + (rf"\quad {direction}" if direction else "")
    ))
    return {a: (Fsum, M_A)}, lines


def _matrix(rows: list[list[str]]) -> str:
    """LaTeX 행렬. 예: [["1","2"],["3","4"]] → \\begin{bmatrix}1 & 2 \\\\ 3 & 4\\end{bmatrix}"""
    return r"\begin{bmatrix}" + r" \\ ".join(" & ".join(r) for r in rows) + r"\end{bmatrix}"


def _reactions_indeterminate(result: SolverResult, summary: LoadSummary) -> tuple[dict, list[StepLine]]:
    """
    부정정보 반력 — 변형일치법 (지시서 D5). 계산은 compat.py, 여기서는 풀이 줄을 만든다.

        1. 부정정 차수, 여분력 Xᵢ, 이완 구조
        2. 적합조건 Δᵢ = Δᵢ₀ + Σ fᵢⱼ Xⱼ = 0
        3. EI·Δᵢ₀, EI·fᵢⱼ (이완 구조를 맥컬리법으로 풀어 구한 값)
        4. 연립방정식 → Xᵢ
        5. 여분력을 하중으로 보고 이완 구조의 평형식 → 나머지 반력
    """
    beam = result.beam
    points = support_points(result)
    comp = solve_compat(beam, result.x, result.L, points)
    reds = comp.redundants
    n = len(reds)
    lines: list[StepLine] = []

    # ── 1. 부정정 차수와 여분력 ──
    n_unknown = beam.unknown_count()
    lines.append(StepLine(
        "", f"부정정 차수 = 미지 반력 {n_unknown}개 − 평형식 2개(ΣF_y, ΣM) = {n_unknown - 2}차"
    ))
    red_desc = ", ".join(
        f"X{i + 1} = {r.name.replace('{', '').replace('}', '')}"
        + (" (연직 반력, 위 +)" if r.kind == "force" else " (반력 모멘트, 시계 +)")
        for i, r in enumerate(reds)
    )
    lines.append(StepLine("", f"여분력: {red_desc}"))
    lines.append(StepLine("", f"이완 구조(정정보): {comp.primary_text}"))

    # ── 2. 적합조건 ──
    has_moment = any(r.kind == "moment" for r in reds)
    compat_text = "적합조건: 실제 구조에서 지점은 처지지 않는다 (δ = 0)" + (
        ", 고정단은 회전하지 않는다 (θ = 0)" if has_moment else ""
    )
    for i, r in enumerate(reds):
        terms = " + ".join(rf"f_{{{i + 1}{j + 1}}}\,X_{j + 1}" for j in range(n))
        lines.append(StepLine(
            rf"\Delta_{i + 1} = {r.displacement_name()} = \Delta_{{{i + 1}0}} + {terms} = 0",
            compat_text if i == 0 else "",  # 설명은 첫 줄에만
        ))

    # ── 3. 이완 구조의 변위 (EI 를 곱한 값) ──
    d0_text = r",\quad ".join(rf"EI\,\Delta_{{{i + 1}0}} = {num(v)}" for i, v in enumerate(comp.d0))
    lines.append(StepLine(
        d0_text,
        "실제 하중만 이완 구조에 걸 때의 변위 (맥컬리법, 처짐 아래 +·처짐각 시계 +; EI 는 양변에서 지워짐)",
    ))
    f_text = r",\quad ".join(
        rf"EI\,f_{{{i + 1}{j + 1}}} = {num(comp.F[i][j])}" for i in range(n) for j in range(n)
    )
    lines.append(StepLine(f_text, "단위 여분력 Xⱼ = 1 만 걸 때 Δᵢ 위치의 변위 (유연도 계수 fᵢⱼ)"))

    # ── 4. 여분력 풀기 ──
    if n == 1:
        lines.append(StepLine(
            rf"X_1 = -\frac{{\Delta_{{10}}}}{{f_{{11}}}} = -\frac{{{num(comp.d0[0])}}}{{{paren(comp.F[0][0])}}}"
            rf" = {num(comp.X[0])}"
        ))
    else:
        Fm = _matrix([[num(comp.F[i][j]) for j in range(n)] for i in range(n)])
        Xv = _matrix([[f"X_{i + 1}"] for i in range(n)])
        Dv = _matrix([[num(-v)] for v in comp.d0])
        lines.append(StepLine(rf"{Fm}{Xv} = {Dv}", "연립방정식 (양변에 EI 를 곱한 꼴)"))
    for r, v in zip(reds, comp.X):
        unit = KN if r.kind == "force" else KNM
        extra = ""
        if r.kind == "moment":
            direction = moment_direction(v)
            extra = rf"\quad {direction}" if direction else ""
        lines.append(StepLine(rf"{r.name} = {num(v)}{unit}{extra}"))

    # ── 5. 나머지 반력: 여분력을 하중으로 보고 이완 구조의 평형 ──
    lines.append(StepLine("", "나머지 반력: 구한 여분력을 하중으로 보고 이완 구조의 평형식을 세운다"))
    ext = LoadSummary(list(summary.forces), list(summary.moments), summary.by_load, [])
    for r, v in zip(reds, comp.X):
        if r.kind == "force":
            ext.forces.append(Resultant(r.name, -v, r.x))  # 위 방향 반력 = 아래 하중 −v
        else:
            ext.moments.append(AppliedMoment(r.name, v, r.x))
    if len(comp.primary) == 1:
        _, eq_lines = _reactions_cantilever(comp.primary, ext)
    else:
        _, eq_lines = _reactions_simple(comp.primary, ext)
    lines.extend(eq_lines)
    return comp.reactions, lines


# ─────────────────────────────────────────────────────────────
# 3단계: 구간별 V(x), M(x) — 단면 왼쪽 자유물체도
# ─────────────────────────────────────────────────────────────
@dataclass
class _Term:
    """
    자유물체도 식의 항 하나. V 항과 M 항을 같이 들고 다닌다.
    *_sym : 일반식(기호) LaTeX, *_num : 숫자 대입 LaTeX (부호 포함 (sgn, body)),
    *_expr: 정확한 SymPy 식 (검증용). V 기여가 없으면 V_* 는 None.
    """

    V_sym: tuple[int, str] | None
    V_num: tuple[int, str] | None
    V_expr: sp.Expr
    M_sym: tuple[int, str]
    M_num: tuple[int, str]
    M_expr: sp.Expr


def _signed(sign: int, value: sp.Expr, suffix: str = "") -> tuple[int, str]:
    """부호와 값을 합쳐 (최종 부호, '|값|suffix') 로. 예: _signed(-1, -5, 'x') → (+1, '5.00x')"""
    s = sign * (1 if value >= 0 else -1)
    return s, f"{num(abs(value))}{suffix}"


def _left_terms(result: SolverResult, summary: LoadSummary, hand: dict, x1: sp.Expr) -> list[_Term]:
    """
    구간 (x1, x2) 안의 단면에서, 단면 왼쪽에 있는 힘들로 V, M 항을 만든다.

    부호 (실무 규약, 단면 왼쪽 자유물체도 기준):
        위로 향하는 힘 F (반력)  : V += F,  M += F·(x − x_F)
        아래로 향하는 힘 P       : V −= P,  M −= P·(x − x_P)
        시계방향 모멘트 M₀       : M += M₀ (V 에는 영향 없음)
        구간을 덮고 있는 분포하중: 단면까지 걸린 부분만 적분
            등분포 w (a 부터)    : V −= w(x − a),  M −= w(x − a)²/2
            선형 w₁ + s(x − a)   : V −= w₁(x − a) + s(x − a)²/2
                                   M −= w₁(x − a)²/2 + s(x − a)³/6
    """
    x = result.x
    items: list[tuple[sp.Expr, _Term]] = []  # (위치, 항) — 위치 순으로 정렬해서 보여준다

    # 반력
    for r in result.reactions:
        if r.x > x1:
            continue
        R, M_r = hand[r.label]
        sh = shift(r.x)
        items.append((r.x, _Term(
            (1, f"R_{r.label}"), _signed(1, R), R,
            (1, f"R_{r.label}\\,{sh}"), _signed(1, R, f"\\,{sh}"), R * (x - r.x),
        )))
        if M_r is not None:
            items.append((r.x, _Term(
                None, None, sp.Integer(0),
                (1, f"M_{r.label}"), _signed(1, M_r), M_r,
            )))

    # 하중
    for k, load in enumerate(result.beam.loads, start=1):
        if isinstance(load, PointLoad):
            xp, P = exact(load.x), exact(load.P)
            if xp > x1:
                continue
            sh = shift(xp)
            items.append((xp, _Term(
                (-1, f"P_{{{k}}}"), _signed(-1, P), -P,
                (-1, f"P_{{{k}}}\\,{sh}"), _signed(-1, P, f"\\,{sh}"), -P * (x - xp),
            )))

        elif isinstance(load, MomentLoad):
            xm, M0 = exact(load.x), exact(load.M)
            if xm > x1:
                continue
            items.append((xm, _Term(
                None, None, sp.Integer(0),
                (1, f"M_{{{k}}}"), _signed(1, M0), M0,
            )))

        else:  # 분포하중 (등분포·선형)
            a, b = exact(load.x_start), exact(load.x_end)
            if a > x1:
                continue  # 아직 시작 안 함
            if b <= x1:
                # 단면 왼쪽에 완전히 들어온 분포하중 → 합력으로 처리
                for res in summary.by_load[k]:
                    sh = shift(res.x)
                    items.append((a, _Term(
                        (-1, res.name), _signed(-1, res.F), -res.F,
                        (-1, f"{res.name}\\,{sh}"), _signed(-1, res.F, f"\\,{sh}"), -res.F * (x - res.x),
                    )))
            else:
                # 단면이 분포하중 구간 안에 있음 → 단면까지만 적분
                items.extend((a, t) for t in _partial_terms(k, load, x))

    items.sort(key=lambda it: it[0])
    return [t for _, t in items]


def _partial_terms(k: int, load: UniformLoad | LinearLoad, x: sp.Symbol) -> list[_Term]:
    """단면이 분포하중 구간 안에 있을 때의 항 (단면 왼쪽 부분만)."""
    a = exact(load.x_start)
    sh1, sh2, sh3 = shift(a), shift_pow(a, 2), shift_pow(a, 3)
    d = x - a
    if isinstance(load, UniformLoad):
        w = exact(load.w)
        return [_Term(
            (-1, f"w_{{{k}}}\\,{sh1}"), _signed(-1, w, f"\\,{sh1}"), -w * d,
            (-1, rf"\frac{{w_{{{k}}}\,{sh2}}}{{2}}"), _signed(-1, w, rf" \cdot \frac{{{sh2}}}{{2}}"), -w * d**2 / 2,
        )]
    # 선형: w(x) = w₁ + s·(x − a)
    w1, w2 = exact(load.w_start), exact(load.w_end)
    s = (w2 - w1) / (exact(load.x_end) - a)
    terms: list[_Term] = []
    if w1 != 0:
        terms.append(_Term(
            (-1, f"w_{{{k},1}}\\,{sh1}"), _signed(-1, w1, f"\\,{sh1}"), -w1 * d,
            (-1, rf"\frac{{w_{{{k},1}}\,{sh2}}}{{2}}"), _signed(-1, w1, rf" \cdot \frac{{{sh2}}}{{2}}"), -w1 * d**2 / 2,
        ))
    if s != 0:
        terms.append(_Term(
            (-1, rf"\frac{{s_{{{k}}}\,{sh2}}}{{2}}"), _signed(-1, s, rf" \cdot \frac{{{sh2}}}{{2}}"), -s * d**2 / 2,
            (-1, rf"\frac{{s_{{{k}}}\,{sh3}}}{{6}}"), _signed(-1, s, rf" \cdot \frac{{{sh3}}}{{6}}"), -s * d**3 / 6,
        ))
    return terms


def _segment_lines(
    result: SolverResult, summary: LoadSummary, hand: dict, segments: list[Segment]
) -> tuple[list[StepLine], list[str]]:
    """각 구간의 V(x), M(x) 를 일반식 → 대입 → 결과로 쓰고 SymPy 구간식과 비교한다."""
    x = result.x
    lines: list[StepLine] = []
    warnings: list[str] = []

    # 선형 분포하중 기호 설명 (식에 s_k 가 나오면 무슨 뜻인지 알려준다)
    for k, load in enumerate(result.beam.loads, start=1):
        if isinstance(load, LinearLoad) and load.w_start != load.w_end:
            w1, w2 = exact(load.w_start), exact(load.w_end)
            ell = exact(load.x_end) - exact(load.x_start)
            s = (w2 - w1) / ell
            lines.append(StepLine(
                rf"s_{{{k}}} = \frac{{w_2 - w_1}}{{\ell}} = \frac{{{num(w2)} - {paren(w1)}}}{{{num(ell)}}} = {coef(s)}\ \text{{kN/m}}^2",
                f"하중 {k} 기울기 (w(x) = w₁ + s·(x − a))",
            ))

    for i, seg in enumerate(segments):
        terms = _left_terms(result, summary, hand, seg.x_start)
        V_terms = [t for t in terms if t.V_sym is not None]
        V_hand = sp.expand(sum((t.V_expr for t in terms), sp.Integer(0)))
        M_hand = sp.expand(sum((t.M_expr for t in terms), sp.Integer(0)))

        lines.append(StepLine(
            "", f"구간 {i + 1}: {num(seg.x_start)} ≤ x < {num(seg.x_end)} m  (단면 왼쪽 자유물체도)"
        ))
        lines.append(StepLine(
            rf"V(x) = {join_terms([t.V_sym for t in V_terms])} = {join_terms([t.V_num for t in V_terms])}"
        ))
        lines.append(StepLine(rf"\quad = {poly_latex(V_hand, x)}{KN}"))
        lines.append(StepLine(
            rf"M(x) = {join_terms([t.M_sym for t in terms])} = {join_terms([t.M_num for t in terms])}"
        ))
        lines.append(StepLine(rf"\quad = {poly_latex(M_hand, x)}{KNM}"))

        # 검증: 손으로 세운 식 = SymPy 특이함수 식을 구간별로 푼 식?
        if sp.expand(V_hand - seg.V) != 0 or sp.expand(M_hand - seg.M) != 0:
            warnings.append(
                f"구간 {i + 1} ({num(seg.x_start)}~{num(seg.x_end)} m) 의 풀이 식이 SymPy 결과와 다릅니다."
            )
    return lines, warnings


# ─────────────────────────────────────────────────────────────
# 4단계: 최대값
# ─────────────────────────────────────────────────────────────
def _extreme_lines(result: SolverResult, segments: list[Segment], ext: Extremes) -> list[StepLine]:
    """V = 0 위치 → M 대입, 경계값 후보 비교, 최종 정·부모멘트 최대와 |V| 최대."""
    x = result.x
    lines: list[StepLine] = []

    # ── (1) V = 0 위치 ──
    if not ext.zero_shear:
        lines.append(StepLine("", "V(x) = 0 이 되는 위치가 없습니다 → M 의 극값은 구간 경계에서 생깁니다."))
    for z in ext.zero_shear:
        if z.kind == "root":
            seg = segments[z.segment_index]
            lines.append(StepLine(
                rf"V(x) = {poly_latex(seg.V, x)} = 0",
                f"구간 {z.segment_index + 1} 에서 V(x) = 0 (dM/dx = V = 0 → M 극값)",
            ))
            V_poly = sp.Poly(sp.expand(seg.V), x)
            if V_poly.degree() == 1:
                c1, c0 = V_poly.all_coeffs()  # V = c1·x + c0 = 0 → x = −c0/c1
                if c1 < 0:
                    c1, c0 = -c1, -c0  # 분모를 양수로 만들어 '−30/−10' 대신 '30/10' 으로 보이게
                lines.append(StepLine(rf"x = \frac{{{coef(-c0)}}}{{{coef(c1)}}} = {num(z.x)}{M_UNIT}"))
            elif V_poly.degree() == 2:
                # 근의 공식: a·x² + b·x + c = 0 → x = (−b ± √(b² − 4ac)) / 2a, 구간 안의 해만 채택
                qa, qb, qc = V_poly.all_coeffs()
                lines.append(StepLine(
                    rf"x = \frac{{-b \pm \sqrt{{b^2 - 4ac}}}}{{2a}}"
                    rf" = \frac{{{coef(-qb)} \pm \sqrt{{{coef(qb**2 - 4 * qa * qc)}}}}}{{{coef(2 * qa)}}}"
                    rf" \Rightarrow x = {num(z.x)}{M_UNIT}",
                    f"근의 공식 (a = {coef(qa)}, b = {coef(qb)}, c = {coef(qc)}), 구간 안의 해만 채택",
                ))
            else:
                lines.append(StepLine(rf"x = {num(z.x)}{M_UNIT}", "(구간 내 해)"))
            lines.append(StepLine(
                rf"M({num(z.x)}) = {poly_subst_latex(seg.M, x, z.x)} = {num(z.M)}{KNM}"
            ))
        else:
            lines.append(StepLine(
                rf"x = {num(z.x)}{M_UNIT}:\ M = {num(z.M)}{KNM}",
                "집중하중(또는 지점) 위치에서 V 의 부호가 바뀜 → M 극값",
            ))

    # ── (2) 구간 경계 값 (후보) ──
    parts: list[str] = []
    for i, seg in enumerate(segments):
        xv = seg.x_start
        M_right = seg.M.subs(x, xv)
        if i == 0:
            parts.append(rf"M({num(xv)}) = {num(M_right)}")
        else:
            M_left = segments[i - 1].M.subs(x, xv)
            if M_left == M_right:
                parts.append(rf"M({num(xv)}) = {num(M_right)}")
            else:  # 집중 모멘트 위치: 좌우 값이 다르다
                parts.append(rf"M({num(xv)}^{{-}}) = {num(M_left)},\ M({num(xv)}^{{+}}) = {num(M_right)}")
    last = segments[-1]
    parts.append(rf"M({num(last.x_end)}) = {num(last.M.subs(x, last.x_end))}")
    lines.append(StepLine(r",\quad ".join(parts), "구간 경계(지점·하중 위치·보 끝)의 M 값도 후보로 비교"))

    # ── (3) 결론 ──
    if ext.M_pos_max is not None:
        lines.append(StepLine(
            rf"M_{{max}}^{{+}} = {num(ext.M_pos_max.value)}{KNM}\quad (x = {x_label(ext.M_pos_max)}{M_UNIT})",
            "정모멘트 최대",
        ))
    else:
        lines.append(StepLine("", "정모멘트 최대: 없음 (정모멘트가 생기지 않음)"))
    if ext.M_neg_min is not None:
        lines.append(StepLine(
            rf"M_{{max}}^{{-}} = {num(ext.M_neg_min.value)}{KNM}\quad (x = {x_label(ext.M_neg_min)}{M_UNIT})",
            "부모멘트 최대",
        ))
    else:
        lines.append(StepLine("", "부모멘트 최대: 없음 (부모멘트가 생기지 않음)"))
    lines.append(StepLine(
        rf"|V|_{{max}} = {num(abs(ext.V_abs_max.value))}{KN}\quad (V = {num(ext.V_abs_max.value)},\ x = {x_label(ext.V_abs_max)}{M_UNIT})",
        "|V| 최대",
    ))
    return lines


# ─────────────────────────────────────────────────────────────
# 5단계: 특이함수 식 (기본 숨김)
# ─────────────────────────────────────────────────────────────
def singularity_latex(expr: sp.Expr) -> str:
    """
    특이함수 식을 ⟨x − a⟩ⁿ 표기 LaTeX 로.
    예: 30⟨x⟩¹ − 5⟨x⟩² → "30\\langle x \\rangle^{1} - 5\\langle x \\rangle^{2}"
    """
    expr = sp.expand(expr)
    terms: list[tuple[sp.Expr, sp.Expr, int]] = []  # (계수, a, n)
    for term in sp.Add.make_args(expr):  # 덧셈 항을 하나씩 꺼낸다
        c, rest = term.as_coeff_Mul()  # 숫자 계수와 나머지(특이함수)로 나눈다
        if isinstance(rest, sp.SingularityFunction):
            _, a, n = rest.args
            terms.append((c, a, int(n)))
        elif rest == 1:
            terms.append((c, None, 0))  # 상수항 (보통 생기지 않음)
    terms.sort(key=lambda t: (float(t[1]) if t[1] is not None else -1, t[2]))
    out: list[tuple[int, str]] = []
    for c, a, n in terms:
        if c == 0:
            continue
        if a is None:
            body = coef(abs(c))
        else:
            inner = "x" if a == 0 else f"x - {num(a)}"
            sf = rf"\langle {inner} \rangle^{{{n}}}"
            body = sf if abs(c) == 1 else f"{coef(abs(c))}\\,{sf}"
        out.append((1 if c > 0 else -1, body))
    return join_terms(out)


# ─────────────────────────────────────────────────────────────
# 5단계: 처짐 — 맥컬리법 (Phase 5A, 지시서 D1~D4)
# ─────────────────────────────────────────────────────────────
def sci(value: float) -> str:
    """
    큰 수를 'a × 10ⁿ' LaTeX 로 (단면2차모멘트 표시용).
    예: sci(1.2e8) → "1.2 \\times 10^{8}",  sci(205000) → "205000"
    """
    v = float(value)
    if v == 0 or 1e-3 <= abs(v) < 1e6:
        return coef(v)
    exp = int(f"{v:e}".split("e")[1])  # 지수 부분 (예: 1.2e+08 → 8)
    mant = v / 10**exp
    return rf"{coef(mant)} \times 10^{{{exp}}}"


def mm(value_m: sp.Expr) -> str:
    """처짐 [m] → mm 문자열 (소수 2자리). 예: 0.00686 → '6.86'"""
    return num(float(value_m) * 1000)


def support_points(result: SolverResult) -> list[SupportPoint]:
    """solver 반력 목록 → macaulay.SupportPoint 목록 (이름·종류·위치)."""
    return [SupportPoint(r.label, r.kind, r.x) for r in result.reactions]


def macaulay_moment(result: SolverResult, hand: dict) -> sp.Expr:
    """
    손계산 반력과 하중 목록으로 휨모멘트를 특이함수 식 하나로 직접 만든다.
    (규칙은 macaulay.moment_expr 참고: 반력 +R⟨x−a⟩¹, 집중하중 −P⟨x−a⟩¹, 등분포 −w⟨x−a⟩²/2 …)
    """
    return moment_expr(result.x, result.L, support_points(result), hand, result.beam.loads)


def _deflection_lines(
    result: SolverResult, hand: dict, segments: list[Segment], dext: DeflectionExtremes
) -> tuple[list[StepLine], list[str]]:
    """
    맥컬리법 처짐 풀이.

        EI·δ''(x) = −M(x)          (δ 아래 +, θ = dδ/dx 시계 +)
        EI·θ(x)  = −∫M dx + C₁
        EI·δ(x)  = −∬M dx + C₁x + C₂
        경계조건(지점 δ = 0, 고정단 θ = 0) → C₁, C₂ → θ = 0 위치 → δ_max → 허용처짐 판정

    손으로 만든 δ(x) 를 SymPy δ(x) 와 구간마다 비교해서 다르면 경고를 낸다.
    """
    x = result.x
    beam = result.beam
    EI = result.EI
    lines: list[StepLine] = []
    warnings: list[str] = []
    C1, C2 = sp.symbols("C_1 C_2")

    # ── (1) 강성 환산 (지시서 D1) ──
    lines.append(StepLine(
        rf"EI = E \times I \times 10^{{-9}} = {sci(beam.E)} \times {sci(beam.I)} \times 10^{{-9}} = {num(EI)}\ \text{{kN}}\cdot\text{{m}}^2",
        "휨강성 (E [MPa] × I [mm⁴] → EI [kN·m²]: 1 MPa = 10³ kN/m², 1 mm⁴ = 10⁻¹² m⁴)",
    ))

    # ── (2) 기본식과 M(x) ──
    M_hand = macaulay_moment(result, hand)
    lines.append(StepLine(
        r"EI\,\delta''(x) = -M(x)",
        "기본식 (처짐 δ 아래 +, 처짐각 θ = dδ/dx 시계 +)",
    ))
    lines.append(StepLine(rf"M(x) = {singularity_latex(M_hand)}", "휨모멘트 (특이함수 식, 2단계 반력 사용)"))

    # ── (3) 두 번 적분 ──
    EI_theta_0 = -integrate_sf(M_hand)  # 적분상수를 뺀 부분
    EI_delta_0 = integrate_sf(EI_theta_0)
    lines.append(StepLine(rf"EI\,\theta(x) = {singularity_latex(EI_theta_0)} + C_1", "1회 적분"))
    lines.append(StepLine(rf"EI\,\delta(x) = {singularity_latex(EI_delta_0)} + C_1 x + C_2", "2회 적분"))

    # ── (4) 경계조건 → C₁, C₂ ──
    points = support_points(result)
    bcs = boundary_conditions(points)
    if bcs[0][0] == "theta":
        bc_text = "경계조건: 고정단의 처짐각 = 0, 처짐 = 0"
    else:
        bc_text = "경계조건: 두 지점의 처짐 = 0"
    lines.append(StepLine("", bc_text))
    eqs = []
    for what, x0 in bcs:
        if what == "theta":
            known = EI_theta_0.subs(x, x0)
            eqs.append(known + C1)
            lines.append(StepLine(rf"\theta({num(x0)}) = 0:\quad {num(known)} + C_1 = 0"))
        else:
            known = EI_delta_0.subs(x, x0)
            eqs.append(known + C1 * x0 + C2)
            lines.append(StepLine(rf"\delta({num(x0)}) = 0:\quad {num(known)} + {num(x0)}\,C_1 + C_2 = 0"))
    sol = sp.solve(eqs, [C1, C2], dict=True)[0]  # dict=True : [{C_1: 값, C_2: 값}] 형태로 받는다
    c1, c2 = sol[C1], sol[C2]
    lines.append(StepLine(
        rf"C_1 = {num(c1)}\ \text{{kN}}\cdot\text{{m}}^2,\quad C_2 = {num(c2)}\ \text{{kN}}\cdot\text{{m}}^3"
    ))

    # 부정정보: 경계조건에 쓰지 않은 나머지 구속(지점 δ = 0, 고정단 θ = 0)도 만족하는지 확인.
    # 반력을 적합조건으로 구했으므로 0 이 나와야 한다 (손계산 검산에 해당).
    used = {(w, x0) for w, x0 in bcs}
    EI_delta_full = EI_delta_0 + c1 * x + c2
    EI_theta_full = EI_theta_0 + c1
    checks = []
    for pt in points:
        conds = [("delta", pt.x)] + ([("theta", pt.x)] if pt.kind == "fixed" else [])
        for what, x0 in conds:
            if (what, x0) in used:
                continue
            val = (EI_delta_full if what == "delta" else EI_theta_full).subs(x, x0)
            sym = r"\delta" if what == "delta" else r"\theta"
            checks.append(rf"EI\,{sym}({num(x0)}) = {num(val)}")
            if sp.simplify(val) != 0:
                warnings.append(f"지점 {pt.label} 의 {'처짐' if what == 'delta' else '처짐각'}이 0 이 아닙니다 (적합조건 불일치).")
    if checks:
        lines.append(StepLine(r",\quad ".join(checks), "검산: 나머지 지점 구속 (0 이면 적합조건 만족)"))

    # ── 검증: 손계산 δ(x) = SymPy δ(x) ? (구간별 다항식으로 비교) ──
    delta_hand = (EI_delta_0 + c1 * x + c2) / EI
    for i, seg in enumerate(segments):
        x_mid = (seg.x_start + seg.x_end) / 2
        if sp.expand(singularity_to_poly(delta_hand, x, x_mid) - seg.delta) != 0:
            warnings.append(f"구간 {i + 1} 의 처짐 풀이 식이 SymPy 결과와 다릅니다.")
            break

    # ── (5) θ = 0 위치 → 처짐 극값 ──
    for z in dext.zero_slope:
        seg = next(s for s in segments if s.x_start < z.x < s.x_end)
        lines.append(StepLine(
            rf"EI\,\theta(x) = {poly_latex(seg.theta * EI, x)} = 0 \;\Rightarrow\; x = {num(z.x)}{M_UNIT}",
            "θ(x) = 0 인 위치 (처짐 극값, 이 구간의 식)",
        ))
        lines.append(StepLine(
            rf"\delta({num(z.x)}) = \frac{{EI\,\delta}}{{EI}} = \frac{{{num(z.value * EI)}}}{{{num(EI)}}}"
            rf" = {num(z.value, 5)}\ \text{{m}} = {mm(z.value)}\ \text{{mm}}"
        ))

    d = dext.delta_abs
    at_zero = any(z.x == d.x for z in dext.zero_slope)
    where = "θ = 0 위치" if at_zero else "구간 끝(자유단·하중점 등)"
    lines.append(StepLine(
        rf"|\delta|_{{max}} = {mm(abs(d.value))}\ \text{{mm}}\quad (\delta = {mm(d.value)}\ \text{{mm}},\ x = {num(d.x)}{M_UNIT})",
        f"최대 처짐 — {where} (후보: θ = 0 위치와 구간 끝)",
    ))

    # ── (6) 허용처짐 판정 (지시서 D4) ──
    verdict = r"\le" if dext.ok else r">"
    ok_text = r"\text{OK}" if dext.ok else r"\text{NG}"
    lines.append(StepLine(
        rf"\delta_{{allow}} = \frac{{L}}{{n}} = \frac{{{num(result.L)}}}{{{coef(dext.n)}}} = {mm(dext.limit)}\ \text{{mm}}"
        rf"\quad\Rightarrow\quad {mm(abs(d.value))} {verdict} {mm(dext.limit)}\ \;{ok_text}",
        "허용처짐 판정 (기준 길이 L = 보 전체 길이)",
    ))
    return lines, warnings


# ─────────────────────────────────────────────────────────────
# 메인 함수
# ─────────────────────────────────────────────────────────────
def build_steps(
    result: SolverResult,
    segments: list[Segment],
    ext: Extremes,
    dext: DeflectionExtremes | None = None,
) -> StepsResult:
    """
    풀이 과정 전체를 만든다.

    입력:
        result   : solver.solve() 결과 (반력 검증 기준)
        segments : piecewise.split_segments() 결과 (구간식 검증 기준)
        ext      : extremes.find_extremes() 결과
    출력:
        StepsResult (Step 목록, 검증 경고, 손계산 반력)

    예:
        r = solve(beam); segs = split_segments(r); ext = find_extremes(segs, r.x)
        out = build_steps(r, segs, ext)
        for step in out.steps: print(step.title)
    """
    warnings: list[str] = []
    steps: list[Step] = []

    # ── 1. 하중 정리 ──
    summary = summarize_loads(result)
    steps.append(Step("loads", "1. 하중 정리", summary.lines))

    # ── 2. 반력 계산 (직접 세운 평형식) + SymPy 해와 비교 ──
    kind = result.beam.kind()
    if kind == "cantilever":
        hand, r_lines = _reactions_cantilever(support_points(result), summary)
        r_title = "2. 반력 계산"
    elif kind == "simple":
        hand, r_lines = _reactions_simple(support_points(result), summary)
        r_title = "2. 반력 계산"
    else:  # 부정정보 (Phase 5B)
        hand, r_lines = _reactions_indeterminate(result, summary)
        r_title = "2. 반력 계산 (변형일치법)"
    steps.append(Step("reactions", r_title, r_lines))
    for r in result.reactions:
        R_h, M_h = hand[r.label]
        if sp.simplify(R_h - r.R) != 0 or (r.M is not None and sp.simplify(M_h - r.M) != 0):
            warnings.append(
                f"지점 {r.label}: 풀이 과정 반력과 SymPy 해가 다릅니다 "
                f"(풀이 {num(R_h)}, SymPy {num(r.R)}). 입력을 확인하세요."
            )

    # ── 3. 구간별 V(x), M(x) ──
    s_lines, s_warn = _segment_lines(result, summary, hand, segments)
    warnings.extend(s_warn)
    steps.append(Step("segments", "3. 구간별 전단력 V(x)·휨모멘트 M(x)", s_lines))

    # ── 4. 최대값 ──
    steps.append(Step("extremes", "4. 최대값", _extreme_lines(result, segments, ext)))

    # ── 5. 처짐 (E, I 를 넣었을 때만) ──
    if result.delta is not None and dext is not None:
        d_lines, d_warn = _deflection_lines(result, hand, segments, dext)
        warnings.extend(d_warn)
        steps.append(Step("deflection", "5. 처짐 (맥컬리법)", d_lines))

    # ── 5. 특이함수 식 (기본 숨김) ──
    steps.append(Step(
        "singularity",
        "특이함수 식 (맥컬리 괄호)",
        [
            StepLine("", "⟨x − a⟩ⁿ = 0 (x < a), (x − a)ⁿ (x ≥ a). 음의 차수는 집중력·집중모멘트."),
            StepLine(rf"V(x) = {singularity_latex(result.V)}"),
            StepLine(rf"M(x) = {singularity_latex(result.M)}"),
        ],
        hidden=True,
    ))

    return StepsResult(steps=steps, warnings=warnings, hand_reactions=hand)
