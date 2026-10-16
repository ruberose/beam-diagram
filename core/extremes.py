"""
extremes.py — 최대·최소값과 그 위치 찾기 (V = 0 위치 포함)

[역할]
    구간별 다항식(piecewise.Segment)에서 설계에 필요한 값을 찾는다.
        - 정모멘트 최대 (M > 0 중 가장 큰 값)
        - 부모멘트 최대 (M < 0 중 가장 작은 값, 즉 절댓값이 가장 큰 음수)
        - |V| 최대
        - V = 0 이 되는 위치와 그 위치의 M

[관계]
    - 이 파일을 부르는 곳: demo.py, tests/, (나중에) steps.py, web/bridge.py
    - 이 파일이 부르는 곳: piecewise.py(Segment), sympy

[손계산 원리]
    dM/dx = V 이므로 M 이 극값을 갖는 곳은
        (1) 구간 안에서 V(x) = 0 인 점 → 그 x 를 M(x) 에 대입
        (2) 구간 경계 (집중하중 위치에서 V 가 + → − 로 "점프"하며 0 을 지나는 곳 포함,
            지점 위, 보 끝, 집중모멘트 위치 등)
    이 후보들을 모두 계산해 비교한다.

[Python 문법 메모]
    - sp.solve(식, x) : 식 = 0 을 만족하는 x 목록을 정확한 값으로 구한다
    - max(목록, key=함수) : 함수 값이 가장 큰 원소를 고른다
    - float(sympy수) : SymPy 정확한 수를 보통 실수로 바꾼다 (비교·표시용)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import sympy as sp

from core.piecewise import Segment

# 0 판단 허용 오차: 정확한 수로 계산하므로 사실상 0 이지만,
# √ 가 들어간 수를 float 로 바꿀 때의 아주 작은 오차를 무시하려고 둔다.
_ZERO_TOL = 1e-12


@dataclass
class ValueAt:
    """
    어떤 값과 그 위치.

    속성:
        value : 값 (정확한 SymPy 수) [kN 또는 kN·m]
        x     : 위치 [m] (정확한 SymPy 수)
        side  : 경계에서의 좌우 구분 — "left"(x⁻), "right"(x⁺), ""(구간 내부/연속)
    """

    value: sp.Expr
    x: sp.Expr
    side: str = ""


@dataclass
class ZeroShear:
    """
    V = 0 이 되는 위치.

    속성:
        x    : 위치 [m]
        M    : 그 위치의 휨모멘트 [kN·m]
        kind : "root" — 구간 안에서 V(x) = 0 을 풀어서 나온 점
               "jump" — 집중하중 위치에서 V 가 부호를 바꾸며 0 을 건너뛰는 점
        segment_index : root 인 경우 몇 번째 구간의 식에서 풀었는지 (풀이 과정 표시용)
    """

    x: sp.Expr
    M: sp.Expr
    kind: str
    segment_index: int | None = None


@dataclass
class Extremes:
    """
    find_extremes() 결과.

    속성:
        M_pos_max  : 정모멘트 최대 (없으면 None)
        M_neg_min  : 부모멘트 최대(가장 작은 음수) (없으면 None)
        V_abs_max  : |V| 최대 (부호 포함 값)
        V_max      : V 최대
        V_min      : V 최소
        zero_shear : V = 0 위치 목록
    """

    M_pos_max: ValueAt | None
    M_neg_min: ValueAt | None
    V_abs_max: ValueAt
    V_max: ValueAt
    V_min: ValueAt
    zero_shear: list[ZeroShear] = field(default_factory=list)


def shear_roots(seg: Segment, x: sp.Symbol) -> list[sp.Expr]:
    """
    구간 안(경계 제외)에서 V(x) = 0 인 x 를 정확한 값으로 구한다.

    입력:  seg — 한 구간, x — 위치 변수
    출력:  x 값 목록 (작은 것부터)

    구조역학적 의미:
        V = dM/dx = 0 → 이 점에서 M 이 극대·극소.
        예: 단순보 등분포 V(x) = wL/2 − w·x = 0 → x = L/2
            단순보 삼각형(0→w) V(x) = wL/6 − w·x²/(2L) = 0 → x = L/√3
    """
    V = sp.expand(seg.V)
    if V == 0:
        # 구간 전체에서 V ≡ 0 (예: 무하중 돌출부) → M 이 일정, 특정 점 없음
        return []
    roots = []
    for r in sp.solve(V, x):
        # 허수 해는 버리고, 구간 내부에 있는 실수 해만 남긴다
        if not r.is_real:
            continue
        rv = float(r)
        if float(seg.x_start) + _ZERO_TOL < rv < float(seg.x_end) - _ZERO_TOL:
            roots.append(sp.nsimplify(sp.simplify(r)))
    return sorted(roots, key=float)


def _collect_candidates(segments: list[Segment], x: sp.Symbol) -> tuple[list[ValueAt], list[ValueAt]]:
    """
    M 후보와 V 후보를 모은다.

    후보:
        - 각 구간의 양 끝 (한쪽 극한: 시작점은 "right", 끝점은 "left")
        - 각 구간 안의 V = 0 위치 (M 후보만)
    반환: (M 후보 목록, V 후보 목록)
    """
    M_cands: list[ValueAt] = []
    V_cands: list[ValueAt] = []
    for seg in segments:
        for xv, side in ((seg.x_start, "right"), (seg.x_end, "left")):
            M_cands.append(ValueAt(seg.M.subs(x, xv), xv, side))
            V_cands.append(ValueAt(seg.V.subs(x, xv), xv, side))
        for r in shear_roots(seg, x):
            M_cands.append(ValueAt(sp.simplify(seg.M.subs(x, r)), r, ""))
        # V 는 구간 내부에서 극값을 가질 수도 있다(선형 분포하중의 dV/dx = −w = 0 인 점).
        # 그 점은 하중 세기 w(x) = 0 인 곳으로, 경계 사이에서 w 가 부호를 바꿀 때만 생긴다.
        dV = sp.diff(seg.V, x)
        if dV != 0:
            for r in sp.solve(dV, x):
                if r.is_real and float(seg.x_start) < float(r) < float(seg.x_end):
                    V_cands.append(ValueAt(seg.V.subs(x, r), r, ""))
    return M_cands, V_cands


def _pick(cands: list[ValueAt], key) -> ValueAt:
    """
    key(후보) 가 가장 큰 후보를 고른다. 값이 같으면(오차 이내) 왼쪽(x 작은) 후보 우선.

    [문법] key 는 함수를 값처럼 넘긴 것. 예: _pick(c, lambda v: float(v.value))
    """
    best = cands[0]
    for c in cands[1:]:
        diff = key(c) - key(best)
        if diff > _ZERO_TOL * max(1.0, abs(key(best))):
            best = c
    return best


def find_extremes(segments: list[Segment], x: sp.Symbol) -> Extremes:
    """
    구간별 식에서 주요값(최대·최소, V=0 위치)을 찾는다.

    입력:
        segments : piecewise.split_segments() 결과
        x        : 위치 변수
    출력:
        Extremes

    예 (단순보 L=6, 등분포 w=10):
        M_pos_max = 45 (= wL²/8) at x = 3,  V_abs_max = 30 (= wL/2) at x = 0
    """
    # ── 1단계: 후보 모으기 ──
    M_cands, V_cands = _collect_candidates(segments, x)

    # ── 2단계: 정모멘트 최대 / 부모멘트 최대 ──
    positives = [c for c in M_cands if float(c.value) > _ZERO_TOL]
    negatives = [c for c in M_cands if float(c.value) < -_ZERO_TOL]
    M_pos_max = _pick(positives, lambda c: float(c.value)) if positives else None
    M_neg_min = _pick(negatives, lambda c: -float(c.value)) if negatives else None

    # ── 3단계: 전단력 최대·최소·절댓값 최대 ──
    V_max = _pick(V_cands, lambda c: float(c.value))
    V_min = _pick(V_cands, lambda c: -float(c.value))
    V_abs_max = _pick(V_cands, lambda c: abs(float(c.value)))

    # ── 4단계: V = 0 위치 목록 ──
    zero_shear: list[ZeroShear] = []
    for i, seg in enumerate(segments):
        for r in shear_roots(seg, x):
            zero_shear.append(ZeroShear(r, sp.simplify(seg.M.subs(x, r)), "root", i))
        # 이 구간 끝(왼쪽 극한)과 다음 구간 시작(오른쪽 극한)의 V 부호가 반대면
        # 집중하중(또는 지점 반력)에서 V 가 0 을 건너뛴 것이다.
        if i + 1 < len(segments):
            nxt = segments[i + 1]
            V_left = float(seg.V.subs(x, seg.x_end))
            V_right = float(nxt.V.subs(x, nxt.x_start))
            if V_left * V_right < 0:
                zero_shear.append(ZeroShear(seg.x_end, seg.M.subs(x, seg.x_end), "jump"))

    return Extremes(
        M_pos_max=M_pos_max,
        M_neg_min=M_neg_min,
        V_abs_max=V_abs_max,
        V_max=V_max,
        V_min=V_min,
        zero_shear=zero_shear,
    )
