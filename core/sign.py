"""
sign.py — SymPy 부호 규약 ↔ 실무 부호 규약 변환 (이 파일에서만 한다)

[역할]
    SymPy Beam 은 자기만의 부호 규약을 쓴다. 게다가 SymPy 버전에 따라 규약이
    바뀐 적이 있다. 그래서 규약을 "가정"하지 않고, 프로그램이 처음 실행될 때
    답을 아는 간단한 보를 SymPy로 풀어 보고 부호를 "측정(보정, calibration)"한다.
    다른 파일은 SymPy 부호를 직접 해석하지 않고 반드시 이 파일의 함수를 거친다.

[관계]
    - 이 파일을 부르는 곳: solver.py (하중 입력 변환, 결과 변환)
    - 이 파일이 부르는 곳: sympy (Beam)

[실무 규약 (목표)]
    - 하중 P, w     : 아래 +
    - 모멘트 하중   : 시계방향 +
    - 반력 R        : 위 +
    - 반력 모멘트   : 시계방향 + (모멘트 하중과 같은 규칙)  ※ 사용자 확인 필요 항목
    - 전단력 V      : 단면 왼쪽 부분이 위로 밀리면 +  (단순보 등분포 왼쪽 끝 V = +wL/2)
    - 휨모멘트 M    : 하부 인장 +                     (단순보 등분포 중앙 M = +wL²/8)

[보정 방법 — 손계산 답을 아는 보 두 개를 SymPy로 푼다]
    보정용 보 1: 단순보 L=4, 중앙에 하중 1 (SymPy 값 +1 로 입력)
        실무 답: V(1) = +P/2 = +0.5,  M(2) = +PL/4 = +1
        → SymPy 결과 부호가 같으면 계수 +1, 반대면 −1.
    보정용 보 2: 단순보 L=4, 왼쪽 끝에 모멘트 하중 1 (SymPy 값 +1 로 입력)
        실무 답(시계방향 M₀ 일 때): M(x) = M₀·(1 − x/L) → M(1) = +0.75
        → (실무로 바꾼) M(1)이 +이면 SymPy +1 은 시계방향, −이면 반시계방향.

[왜 하중 계수는 +1 로 고정해도 되는가]
    SymPy 에서 하중과 반력은 같은 "하중 규약"으로 표현된다. 그래서 하중 방향을
    어느 쪽으로 정하든 반력도 같이 따라간다. 실무 아래(+) 하중을 SymPy에 +로 넣기로
    정하면, SymPy 반력 값은 "아래 방향 힘" 기준이 되므로 실무 반력(위 +) = −(SymPy 반력).
    V·M 의 부호만 보정용 보로 확인하면 된다.

[SymPy 1.14.0 에서 측정한 결과] (README 에도 기록)
    shear = +1, moment = +1, moment_load = −1
    즉 V·M 은 실무 규약과 같고, SymPy 모멘트 하중 + 는 반시계방향이다.

[Python 문법 메모]
    - @lru_cache(maxsize=1)
        : 함수 결과를 기억해 두었다가, 두 번째 호출부터는 계산 없이 바로 돌려준다.
          보정은 SymPy 계산이라 느리므로 프로그램 전체에서 한 번만 하려고 쓴다.
    - sympy.Expr : SymPy 수식 객체의 타입 (예: 25 - 10*x)
    - expr.subs(x, 값) : 수식의 x 자리에 값을 대입한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import sympy as sp
from sympy.physics.continuum_mechanics.beam import Beam as SympyBeam


@dataclass(frozen=True)
class SignFactors:
    """
    SymPy 값에 곱해서 실무 값으로 바꾸는 계수(+1 또는 −1) 모음.

    속성:
        load        : 실무 하중(아래 +) → SymPy 하중 값에 곱하는 계수 (항상 +1로 정함)
        shear       : SymPy V → 실무 V 계수
        moment      : SymPy M → 실무 M 계수
        moment_load : 실무 모멘트 하중(시계 +) → SymPy 모멘트 하중 값 계수
    """

    load: int
    shear: int
    moment: int
    moment_load: int


def _sign_of(value: sp.Expr) -> int:
    """SymPy 숫자의 부호를 +1/−1 로 돌려준다. 0이면 보정 실패이므로 예외."""
    v = float(value)
    if v == 0:
        raise RuntimeError("부호 보정 실패: SymPy 결과가 0입니다. SymPy 버전을 확인하세요.")
    return 1 if v > 0 else -1


def _calibration_beam(L: int) -> tuple[SympyBeam, sp.Symbol, sp.Symbol]:
    """보정용 단순보(핀 x=0, 롤러 x=L)를 만들고, 반력 미지수 두 개를 함께 돌려준다."""
    E, I = sp.symbols("E I", positive=True)  # 처짐은 안 쓰므로 기호로 둔다
    R1, R2 = sp.symbols("R1 R2")
    b = SympyBeam(L, E, I)
    # order=-1 : 집중력 (특이함수 ⟨x−a⟩⁻¹)
    b.apply_load(R1, 0, -1)
    b.apply_load(R2, L, -1)
    b.bc_deflection = [(0, 0), (L, 0)]
    return b, R1, R2


@lru_cache(maxsize=1)
def get_factors() -> SignFactors:
    """
    현재 설치된 SymPy 의 부호 규약을 측정해 SignFactors 로 돌려준다.

    처음 호출 때만 SymPy로 보정용 보를 풀고(약 1초), 이후에는 기억한 값을 쓴다.

    예:
        f = get_factors()
        V_practical = f.shear * V_sympy
    """
    L = 4
    load_factor = 1  # 실무 하중(아래 +)을 SymPy 에 그대로 + 로 넣기로 정함 (위 설명 참고)

    # ── 보정 1: 중앙 집중하중 → V, M 부호 ──
    b, R1, R2 = _calibration_beam(L)
    b.apply_load(load_factor * 1, 2, -1)
    b.solve_for_reaction_loads(R1, R2)
    x = b.variable
    V_s = b.shear_force().subs(b.reaction_loads).subs(x, 1)
    M_s = b.bending_moment().subs(b.reaction_loads).subs(x, 2)
    # 크기 검사: 손계산 |V| = P/2 = 0.5, |M| = PL/4 = 1 이 아니면 규약 이전에 뭔가 크게 다르다
    if abs(float(V_s)) != 0.5 or abs(float(M_s)) != 1.0:
        raise RuntimeError(f"부호 보정 실패: 예상과 다른 크기 (V={V_s}, M={M_s}).")
    shear_factor = _sign_of(V_s)    # 실무 V(1) = +0.5
    moment_factor = _sign_of(M_s)   # 실무 M(2) = +1.0

    # ── 보정 2: 왼쪽 끝 모멘트 하중 → 모멘트 하중 방향 ──
    b, R1, R2 = _calibration_beam(L)
    b.apply_load(1, 0, -2)  # order=-2 : 집중 모멘트 (특이함수 ⟨x−a⟩⁻²)
    b.solve_for_reaction_loads(R1, R2)
    x = b.variable
    M_s = b.bending_moment().subs(b.reaction_loads).subs(x, 1)
    if abs(float(M_s)) != 0.75:
        raise RuntimeError(f"부호 보정 실패: 모멘트 하중 결과 크기가 다름 (M={M_s}).")
    # 실무로 바꾼 M(1)이 + 이면 SymPy +1 모멘트 = 시계방향 → 계수 +1
    moment_load_factor = _sign_of(moment_factor * M_s)

    return SignFactors(
        load=load_factor,
        shear=shear_factor,
        moment=moment_factor,
        moment_load=moment_load_factor,
    )


# ─────────────────────────────────────────────────────────────
# 실무 → SymPy (입력 변환)
# ─────────────────────────────────────────────────────────────
def to_sympy_load(value: sp.Expr) -> sp.Expr:
    """실무 하중 값(아래 +, kN 또는 kN/m 등) → SymPy apply_load 에 넣을 값."""
    return get_factors().load * value


def to_sympy_moment_load(M_cw: sp.Expr) -> sp.Expr:
    """실무 모멘트 하중(시계 +, kN·m) → SymPy apply_load(order=-2) 에 넣을 값."""
    return get_factors().moment_load * M_cw


# ─────────────────────────────────────────────────────────────
# SymPy → 실무 (결과 변환)
# ─────────────────────────────────────────────────────────────
def from_sympy_reaction(R_sympy: sp.Expr) -> sp.Expr:
    """
    SymPy 반력 → 실무 반력(위 +, kN).

    SymPy 반력은 하중과 같은 규약(= 실무 "아래 +")으로 나오므로,
    위 + 로 바꾸려면 부호를 뒤집는다: R_up = −R_sympy / load_factor
    """
    return -R_sympy / get_factors().load


def from_sympy_reaction_moment(M_sympy: sp.Expr) -> sp.Expr:
    """
    SymPy 반력 모멘트 → 실무 반력 모멘트(시계 +, kN·m).

    반력 모멘트도 SymPy 안에서는 모멘트 하중과 같은 규약이므로 같은 계수로 되돌린다.
    """
    return M_sympy / get_factors().moment_load


def from_sympy_shear(V_sympy: sp.Expr) -> sp.Expr:
    """SymPy 전단력 식 → 실무 전단력 식 (단면 왼쪽이 위로 밀리면 +)."""
    return get_factors().shear * V_sympy


def from_sympy_moment(M_sympy: sp.Expr) -> sp.Expr:
    """SymPy 휨모멘트 식 → 실무 휨모멘트 식 (하부 인장 +)."""
    return get_factors().moment * M_sympy
