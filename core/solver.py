"""
solver.py — SymPy Beam 으로 반력과 V(x), M(x) 를 구한다

[역할]
    core.model.Beam(실무 입력)을 SymPy Beam 으로 옮겨서 풀고,
    결과(반력, 전단력 식, 휨모멘트 식)를 실무 부호 규약으로 돌려준다.
    V(x), M(x) 는 "특이함수(singularity function)" 식 하나로 보 전체를 표현한다.

[관계]
    - 이 파일을 부르는 곳: piecewise.py(결과 사용), demo.py, tests/, (나중에) web/bridge.py
    - 이 파일이 부르는 곳: model.py(입력 형식), sign.py(부호 변환), sympy

[Phase 5A] E, I 가 입력되면 처짐각 θ(x), 처짐 δ(x) 도 SymPy 로 구한다 (부호는 sign.py).

[특이함수(맥컬리 괄호) 간단 정리]
    ⟨x − a⟩ⁿ =  0          (x < a)
               (x − a)ⁿ    (x ≥ a),  n ≥ 0
    n = −1 : 집중력(디랙 델타),  n = −2 : 집중 모멘트 (구간 내부에서는 항상 0)
    하중을 이렇게 쓰면 적분만으로 V, M 을 보 전체에 대해 한 식으로 얻는다.
        w(x) --적분--> V(x) --적분--> M(x)

[정확한 계산을 위한 처리]
    입력 실수(예: 0.1)를 그대로 쓰면 이진 부동소수 오차가 생긴다.
    그래서 sp.Rational("0.1") = 1/10 처럼 "정확한 분수"로 바꿔서 계산한다.
    화면 표시할 때만 소수로 바꾼다.

[Python 문법 메모]
    - sp.Symbol / sp.symbols("R_A R_B") : 미지수(기호) 만들기
    - b.apply_load(값, 시작, 차수) : SymPy Beam 에 특이함수 항 하나를 더한다
    - dict : {키: 값} 사전 자료형. b.reaction_loads 는 {R_A: 값, R_B: 값} 형태
    - 리스트 컴프리헨션 [식 for 원소 in 목록] : 반복문으로 새 리스트를 한 줄에 만든다
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp
from sympy.physics.continuum_mechanics.beam import Beam as SympyBeam

from core import sign
from core.model import Beam, LinearLoad, Load, MomentLoad, PointLoad, Support, UniformLoad


# ─────────────────────────────────────────────────────────────
# 결과 데이터
# ─────────────────────────────────────────────────────────────
@dataclass
class Reaction:
    """
    지점 하나의 반력 (실무 규약, 정확한 SymPy 수).

    속성:
        label : 지점 이름 ("A", "B" — 왼쪽부터 순서대로)
        kind  : "pin" / "roller" / "fixed"
        x     : 지점 위치 [m]
        R     : 연직 반력 [kN], 위 +
        M     : 반력 모멘트 [kN·m], 시계 + (고정단만, 나머지는 None)
    """

    label: str
    kind: str
    x: sp.Expr
    R: sp.Expr
    M: sp.Expr | None = None


@dataclass
class SolverResult:
    """
    solve() 의 결과.

    속성:
        beam      : 입력 보 (model.Beam)
        L         : 경간 (정확한 수)
        x         : 위치 변수 기호 (SymPy Symbol)
        reactions : Reaction 목록 (왼쪽 지점부터)
        V         : 실무 전단력 식 V(x) [kN] (특이함수 포함)
        M         : 실무 휨모멘트 식 M(x) [kN·m] (특이함수 포함)
        warnings  : 경고 메시지 목록
        EI        : 휨강성 [kN·m²] (E, I 를 넣지 않았으면 None)
        theta     : 실무 처짐각 식 θ(x) [rad], 시계 + (EI 없으면 None)
        delta     : 실무 처짐 식 δ(x) [m], 아래 + (EI 없으면 None)
    """

    beam: Beam
    L: sp.Expr
    x: sp.Symbol
    reactions: list[Reaction]
    V: sp.Expr
    M: sp.Expr
    warnings: list[str]
    EI: sp.Expr | None = None
    theta: sp.Expr | None = None
    delta: sp.Expr | None = None


# ─────────────────────────────────────────────────────────────
# 보조 함수
# ─────────────────────────────────────────────────────────────
def exact(value: float | int | str) -> sp.Rational:
    """
    입력 숫자를 정확한 분수로 바꾼다.

    str() 을 거치는 이유: 0.1 같은 실수는 컴퓨터 안에서 0.1000000000000000055...
    로 저장된다. 문자열 "0.1" 로 바꾼 뒤 Rational 로 만들면 정확히 1/10 이 된다.

    예: exact(0.1) → 1/10,  exact(2.5) → 5/2

    이미 SymPy 정확한 수(예: 부정정 여분력 값 75/4)가 들어오면 그대로 돌려준다.
    """
    if isinstance(value, sp.Basic):
        return value
    return sp.Rational(str(value))


def stiffness_EI(beam: Beam) -> sp.Rational | None:
    """
    E [MPa], I [mm⁴] → 휨강성 EI [kN·m²] (정확한 분수). E, I 가 없으면 None.

    단위 환산 (고정식):
        1 MPa = 1 N/mm² = 10⁶ N/m² = 10³ kN/m²
        1 mm⁴ = (10⁻³ m)⁴ = 10⁻¹² m⁴
        EI [kN·m²] = E·10³ × I·10⁻¹² = E × I × 10⁻⁹

    예: stiffness_EI(Beam(..., E=205000, I=1.2e8)) → 24600
    """
    if not beam.has_stiffness():
        return None
    return exact(beam.E) * exact(beam.I) / sp.Integer(10) ** 9


def support_labels(supports: list[Support]) -> list[tuple[str, Support]]:
    """
    지점을 왼쪽부터 정렬해서 "A", "B", ... 이름을 붙인다.

    [문법] sorted(목록, key=함수) : 함수가 돌려주는 값(여기서는 위치 x) 기준 정렬
           lambda s: s.x        : 이름 없는 짧은 함수. "s 를 받아 s.x 를 돌려준다"
           enumerate(목록)      : (번호, 원소) 쌍을 차례로 꺼낸다
    """
    ordered = sorted(supports, key=lambda s: s.x)
    return [(chr(ord("A") + i), s) for i, s in enumerate(ordered)]


def load_terms(load: Load, L: sp.Expr) -> list[tuple[str, sp.Expr, sp.Expr, int]]:
    """
    하중 하나를 특이함수 항 목록으로 바꾼다 (실무 부호 그대로).

    반환값: [(종류, 값, 시작위치, 차수), ...]
        종류 : "force"(힘 계열) 또는 "moment"(모멘트 하중)
        차수 : −2 집중모멘트, −1 집중하중, 0 등분포, 1 선형분포

    구조역학적 의미 (구간 a~b 분포하중을 특이함수로 쓰는 법):
        등분포 w        : w⟨x−a⟩⁰ − w⟨x−b⟩⁰
                          (a 에서 켜고, b 에서 같은 크기로 꺼서 b 이후를 0 으로 만든다)
        선형 w₁→w₂      : 기울기 s = (w₂−w₁)/(b−a)
                          w₁⟨x−a⟩⁰ + s⟨x−a⟩¹ − w₂⟨x−b⟩⁰ − s⟨x−b⟩¹
        b = L 이면 "끄는 항" 은 보 밖(x ≥ L)에서만 작용하므로 넣지 않는다.

    예: UniformLoad(10, 0, 4), L=8 → [("force",10,0,0), ("force",-10,4,0)]
    """
    if isinstance(load, PointLoad):
        return [("force", exact(load.P), exact(load.x), -1)]

    if isinstance(load, MomentLoad):
        return [("moment", exact(load.M), exact(load.x), -2)]

    if isinstance(load, UniformLoad):
        w, a, b = exact(load.w), exact(load.x_start), exact(load.x_end)
        terms = [("force", w, a, 0)]
        if b < L:
            terms.append(("force", -w, b, 0))
        return terms

    if isinstance(load, LinearLoad):
        w1, w2 = exact(load.w_start), exact(load.w_end)
        a, b = exact(load.x_start), exact(load.x_end)
        slope = (w2 - w1) / (b - a)  # 하중 세기의 기울기 [kN/m²]
        terms = [("force", w1, a, 0), ("force", slope, a, 1)]
        if b < L:
            terms.append(("force", -w2, b, 0))
            terms.append(("force", -slope, b, 1))
        return terms

    raise TypeError(f"알 수 없는 하중 종류: {type(load).__name__}")


# ─────────────────────────────────────────────────────────────
# 메인 함수
# ─────────────────────────────────────────────────────────────
def solve(beam: Beam) -> SolverResult:
    """
    보를 풀어 반력과 V(x), M(x) 를 실무 규약으로 돌려준다.

    입력:  beam — model.Beam (실무 부호 입력)
    출력:  SolverResult (반력, V, M 은 정확한 SymPy 수/식)

    구조역학적 의미:
        미지 반력을 하중처럼 특이함수 항으로 더한 뒤, SymPy 가
        "보 오른쪽 끝 바로 바깥에서 V = 0, M = 0" (= ΣF_y = 0, ΣM = 0) 조건으로
        반력을 푼다.
        - 정정보   : 평형식 두 개로 미지수 두 개가 정해진다.
        - 부정정보 : 미지수가 더 많으므로, 적합조건(지점 처짐 = 0, 고정단 처짐각 = 0)을
                     함께 써서 푼다 (Phase 5B). 상수 EI 면 반력은 EI 값과 무관하므로
                     E, I 를 넣지 않아도 EI 를 기호로 두고 풀 수 있다.

    예:
        r = solve(Beam(6, [Support("pin",0), Support("roller",6)],
                          [UniformLoad(10, 0, 6)]))
        r.reactions[0].R   # → 30  (= wL/2)
    """
    # ── 1단계: 입력 검사 ──
    warnings = beam.validate()
    L = exact(beam.L)

    # ── 2단계: SymPy 보 만들기 ──
    # SymPy Beam(길이, E, I) 는 처짐 계산에 E·I 를 쓴다. 우리는 단위를 맞춘 EI [kN·m²] 를
    # E 자리에, 1 을 I 자리에 넣는다 (곱 EI 만 의미가 있으므로).
    # EI 가 없으면 반력·V·M 에는 영향이 없으므로 기호로 둔다.
    EI = stiffness_EI(beam)
    if EI is not None:
        b = SympyBeam(L, EI, 1)
    else:
        b = SympyBeam(L, sp.Symbol("EI", positive=True), 1)
    x = b.variable

    # ── 3단계: 미지 반력을 특이함수 항으로 추가 ──
    labeled = support_labels(beam.supports)
    unknowns: list[sp.Symbol] = []           # SymPy 에 풀라고 넘길 미지수
    reaction_symbols: dict[str, tuple] = {}  # 지점 이름 → (R 기호, M 기호 또는 None)

    for label, s in labeled:
        xs = exact(s.x)
        R_sym = sp.Symbol(f"R_{label}")
        b.apply_load(R_sym, xs, -1)  # 집중 반력
        unknowns.append(R_sym)
        M_sym = None
        if s.kind == "fixed":
            M_sym = sp.Symbol(f"M_{label}")
            b.apply_load(M_sym, xs, -2)  # 반력 모멘트
            unknowns.append(M_sym)
        reaction_symbols[label] = (R_sym, M_sym)

    # 경계조건(처짐·처짐각): 모든 지점에서 처짐 = 0, 고정단에서는 처짐각도 = 0.
    #   정정보는 평형식만으로 반력이 풀리고 이 조건은 적분상수(처짐식)만 정한다.
    #   부정정보는 이 조건이 바로 "적합조건"이 되어 남는 반력까지 정한다.
    b.bc_deflection = [(exact(s.x), 0) for _, s in labeled]
    b.bc_slope = [(exact(s.x), 0) for _, s in labeled if s.kind == "fixed"]

    # ── 4단계: 실제 하중 추가 (실무 부호 → SymPy 부호는 sign.py 가 담당) ──
    for load in beam.loads:
        for term_kind, value, start, order in load_terms(load, L):
            if term_kind == "moment":
                b.apply_load(sign.to_sympy_moment_load(value), start, order)
            else:
                b.apply_load(sign.to_sympy_load(value), start, order)

    # ── 5단계: 반력 풀기 (ΣF_y = 0, ΣM = 0) ──
    b.solve_for_reaction_loads(*unknowns)  # *목록 : 목록을 풀어서 인자 여러 개로 넘긴다
    solved = b.reaction_loads             # {R_A: 값, R_B: 값, ...}

    # ── 6단계: 결과를 실무 규약으로 변환 ──
    reactions: list[Reaction] = []
    for label, s in labeled:
        R_sym, M_sym = reaction_symbols[label]
        R = sp.simplify(sign.from_sympy_reaction(solved[R_sym]))
        M = None
        if M_sym is not None:
            M = sp.simplify(sign.from_sympy_reaction_moment(solved[M_sym]))
        reactions.append(Reaction(label=label, kind=s.kind, x=exact(s.x), R=R, M=M))

    # .subs(solved) : 식 안의 R_A, R_B 기호를 구한 값으로 바꿔 넣는다
    V = sign.from_sympy_shear(b.shear_force().subs(solved))
    M = sign.from_sympy_moment(b.bending_moment().subs(solved))

    # ── 7단계: 처짐 (EI 가 있을 때만) ──
    # SymPy 는 3단계에서 넣은 경계조건(지점 δ = 0, 고정단 θ = 0)으로 적분상수를 정한다.
    theta = delta = None
    if EI is not None:
        theta = sign.from_sympy_slope(b.slope().subs(solved))
        delta = sign.from_sympy_deflection(b.deflection().subs(solved))

    return SolverResult(
        beam=beam, L=L, x=x, reactions=reactions, V=V, M=M, warnings=warnings,
        EI=EI, theta=theta, delta=delta,
    )
