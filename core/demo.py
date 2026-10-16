"""
demo.py — 콘솔에서 엔진을 돌려 보는 예제

[역할]
    웹 화면 없이 엔진(solver → piecewise → extremes)이 제대로 도는지 눈으로 확인한다.
    실행:  프로젝트 루트에서  python -m core.demo

    `python -m core.demo` 는 "core 패키지 안의 demo 모듈을 실행하라"는 뜻이다.
    이렇게 실행해야 `from core.model import ...` 같은 import 가 동작한다.

[관계]
    - 이 파일이 부르는 곳: model, solver, piecewise, extremes, steps
    - 이 파일을 부르는 곳: 없음 (사람이 직접 실행)
"""

from __future__ import annotations

import sympy as sp

from core.extremes import ValueAt, find_extremes
from core.model import Beam, LinearLoad, MomentLoad, PointLoad, Support, UniformLoad
from core.piecewise import split_segments
from core.solver import solve
from core.steps import build_steps, moment_direction_text


def fmt(value: sp.Expr, digits: int = 2) -> str:
    """정확한 SymPy 수를 소수 digits 자리 문자열로 바꾼다. (−0.00 은 0.00 으로)"""
    v = round(float(value), digits)
    if v == 0:
        v = 0.0
    return f"{v:.{digits}f}"


def fmt_poly(expr: sp.Expr) -> str:
    """
    구간 다항식을 사람이 읽기 좋은 문자열로 바꾼다.
    분수 계수는 소수 4자리로 표시한다 (예: 25 - 10*x).
    """
    if not expr.free_symbols:  # x 가 없는 상수 식 (예: V = −10)
        return fmt(expr)
    # evalf(6) : 유효숫자 6자리 소수로 바꾼다. "**"(Python 거듭제곱) 은 "^" 로 보기 좋게
    return str(sp.expand(expr).evalf(6)).replace("**", "^")


def fmt_at(v: ValueAt | None, unit: str) -> str:
    """ValueAt 을 '값 unit (x = 위치 m)' 형태로 표시한다."""
    if v is None:
        return "없음"
    side = {"left": "⁻", "right": "⁺"}.get(v.side, "")
    return f"{fmt(v.value)} {unit}  (x = {fmt(v.x)}{side} m)"


def run(title: str, beam: Beam) -> None:
    """보 하나를 풀고 결과를 출력한다."""
    print("=" * 64)
    print(title)
    print("=" * 64)

    # ── 1. 반력 ──
    result = solve(beam)
    for w in result.warnings:
        print("[경고]", w)
    print("반력 (위 +, 모멘트 시계 +):")
    for r in result.reactions:
        line = f"  R_{r.label} = {fmt(r.R)} kN   (x = {fmt(r.x)} m, {r.kind})"
        if r.M is not None:
            # 반력 모멘트는 시계 + 값과 함께 실제 회전 방향을 보여준다
            line += f",  M_{r.label} = {fmt(r.M)} kN·m ({moment_direction_text(r.M)})"
        print(line)

    # ── 2. 구간별 식 ──
    segments = split_segments(result)
    print("구간별 식:")
    for seg in segments:
        print(f"  {fmt(seg.x_start)} ≤ x < {fmt(seg.x_end)} :")
        print(f"      V(x) = {fmt_poly(seg.V)}")
        print(f"      M(x) = {fmt_poly(seg.M)}")

    # ── 3. 주요값 ──
    ext = find_extremes(segments, result.x)
    print("주요값:")
    print("  정모멘트 최대 :", fmt_at(ext.M_pos_max, "kN·m"))
    print("  부모멘트 최대 :", fmt_at(ext.M_neg_min, "kN·m"))
    print("  |V| 최대      :", fmt_at(ext.V_abs_max, "kN"))
    for z in ext.zero_shear:
        kind = "구간 내 V=0" if z.kind == "root" else "집중하중에서 V 부호 변화"
        print(f"  V = 0 : x = {fmt(z.x)} m → M = {fmt(z.M)} kN·m  ({kind})")

    # ── 4. 풀이 과정 (LaTeX) ──
    #   웹에서는 KaTeX 로 그리지만, 콘솔에서는 LaTeX 문자열을 그대로 보여준다.
    out = build_steps(result, segments, ext)
    print("풀이 과정 (LaTeX):")
    for step in out.steps:
        hidden = "  [기본 숨김]" if step.hidden else ""
        print(f"  ■ {step.title}{hidden}")
        for line in step.lines:
            if line.text:
                print(f"      # {line.text}")
            if line.latex:
                print(f"      {line.latex}")
    for w in out.warnings:
        print("[검증 경고]", w)
    if not out.warnings:
        print("  (검증: 풀이 과정의 반력·구간식이 SymPy 결과와 일치)")
    print()


def main() -> None:
    """예제 보 몇 개를 차례로 풀어 본다."""
    run(
        "예제 1) 단순보 L=6 m, 전체 등분포 w=10 kN/m   (기대: R=30, M_max=45)",
        Beam(6, [Support("pin", 0), Support("roller", 6)], [UniformLoad(10, 0, 6)]),
    )
    run(
        "예제 2) 내민보 L=8 m (지점 0, 6), 0~4 등분포 10 kN/m + 끝단 P=20 kN",
        Beam(
            8,
            [Support("pin", 0), Support("roller", 6)],
            [UniformLoad(10, 0, 4), PointLoad(20, 8)],
        ),
    )
    run(
        "예제 3) 캔틸레버 L=3 m (왼쪽 고정), 삼각형 0→12 kN/m + 자유단 모멘트 5 kN·m",
        Beam(3, [Support("fixed", 0)], [LinearLoad(0, 12, 0, 3), MomentLoad(5, 3)]),
    )


# 이 파일을 직접 실행했을 때만 main() 을 부른다.
# (다른 파일에서 import 할 때는 실행되지 않게 하는 Python 관용구)
if __name__ == "__main__":
    main()
