"""
bridge.py — 웹(JS) ↔ 계산 엔진(core) 사이의 JSON 변환 (이 파일 하나만 담당)

[역할]
    브라우저의 app.js 는 Python 객체를 직접 다루지 않고 "JSON 문자열" 하나만 주고받는다.
        app.js ──입력 JSON──▶ run() ──core 호출──▶ 출력 JSON ──▶ app.js
    이렇게 해 두면 나중에 서버(FastAPI 등)로 옮겨도 이 파일만 바꾸면 된다.

[관계]
    - 이 파일을 부르는 곳: web/app.js (Pyodide 안에서 `bridge.run(json)` 호출),
                          tests/test_bridge.py
    - 이 파일이 부르는 곳: core.model, core.solver, core.piecewise, core.extremes, core.steps

[입력 JSON 형식]
    {
      "L": 8,
      "supports": [{"kind": "pin", "x": 0}, {"kind": "roller", "x": 6}],
      "loads": [
        {"type": "point",   "P": 20, "x": 8},
        {"type": "udl",     "w": 10, "x_start": 0, "x_end": 4},
        {"type": "linear",  "w_start": 0, "w_end": 12, "x_start": 0, "x_end": 3},
        {"type": "moment",  "M": 5, "x": 3}
      ],
      "E": 205000, "I": 1.2e8,   (선택: 처짐 계산용 E [MPa], I [mm⁴] — 둘 다 넣어야 계산)
      "defl_n": 360,             (선택: 허용처짐 L/n 의 n, 기본 360)
      "n_points": 400            (선택: 그래프 점 개수, 기본 400)
    }

[출력 JSON 형식]
    성공: {"ok": true, "reactions": [...], "diagram": {"x": [...], "V": [...], "M": [...]},
           "segments": [...], "extremes": {...}, "deflection": {...} 또는 null,
           "steps": [...], "warnings": [...]}
          E, I 가 있으면 diagram 에 "theta"[rad], "delta"[m] 배열도 들어간다.
    실패: {"ok": false, "error": "한국어 오류 메시지"}
    숫자는 모두 float(전체 자릿수)로 보내고, 소수 2자리 표시는 JS 에서 한다.

[Python 문법 메모]
    - json.loads(문자열) / json.dumps(객체) : JSON 문자열 ↔ Python dict/list 변환
    - try / except 예외 : 오류가 나도 프로그램을 멈추지 않고 except 블록으로 넘어간다
    - sp.lambdify(x, 식, "math") : SymPy 식을 보통 Python 함수로 바꾼다.
          f = sp.lambdify(x, 30 - 10*x, "math");  f(2.0) → 10.0
      SymPy 식에 .subs() 로 값을 넣는 것보다 수백 배 빨라서 그래프 점 계산에 쓴다.
    - dataclasses.asdict(객체) : dataclass 객체를 dict 로 바꾼다 (JSON 변환 준비)
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

import sympy as sp

from core.extremes import DeflectionExtremes, Extremes, ValueAt, find_deflection_extremes, find_extremes
from core.model import Beam, LinearLoad, Load, MomentLoad, PointLoad, Support, UniformLoad
from core.piecewise import Segment, split_segments
from core.solver import exact, solve
from core.steps import build_steps, moment_direction_text, poly_latex


# ─────────────────────────────────────────────────────────────
# 입력 JSON → core.model
# ─────────────────────────────────────────────────────────────
def _num(d: dict, key: str) -> float:
    """dict 에서 숫자 하나를 꺼낸다. 비었거나 숫자가 아니면 알기 쉬운 오류를 낸다."""
    if key not in d or d[key] is None or d[key] == "":
        raise ValueError(f"'{key}' 값이 비어 있습니다.")
    try:
        return float(d[key])
    except (TypeError, ValueError):
        raise ValueError(f"'{key}' 값이 숫자가 아닙니다: {d[key]}")


def parse_load(d: dict) -> Load:
    """하중 dict 하나 → model 하중 객체. 'type' 으로 종류를 구분한다."""
    t = d.get("type")
    if t == "point":
        return PointLoad(P=_num(d, "P"), x=_num(d, "x"))
    if t == "udl":
        return UniformLoad(w=_num(d, "w"), x_start=_num(d, "x_start"), x_end=_num(d, "x_end"))
    if t == "linear":
        return LinearLoad(
            w_start=_num(d, "w_start"), w_end=_num(d, "w_end"),
            x_start=_num(d, "x_start"), x_end=_num(d, "x_end"),
        )
    if t == "moment":
        return MomentLoad(M=_num(d, "M"), x=_num(d, "x"))
    raise ValueError(f"알 수 없는 하중 종류입니다: {t}")


def _optional_num(d: dict, key: str) -> float | None:
    """선택 입력 숫자: 비어 있으면 None, 있으면 숫자로."""
    if d.get(key) in (None, ""):
        return None
    return _num(d, key)


def parse_beam(data: dict) -> Beam:
    """입력 dict 전체 → model.Beam"""
    supports = [Support(kind=s.get("kind"), x=_num(s, "x")) for s in data.get("supports", [])]
    loads = [parse_load(ld) for ld in data.get("loads", [])]
    n = _optional_num(data, "defl_n")
    return Beam(
        L=_num(data, "L"), supports=supports, loads=loads,
        E=_optional_num(data, "E"), I=_optional_num(data, "I"),
        defl_n=360.0 if n is None else n,
    )


# ─────────────────────────────────────────────────────────────
# core 결과 → 출력 dict
# ─────────────────────────────────────────────────────────────
def _value_at(v: ValueAt | None) -> dict | None:
    """ValueAt → {"value", "x", "side"} (없으면 None → JSON 의 null)"""
    if v is None:
        return None
    return {"value": float(v.value), "x": float(v.x), "side": v.side}


def _extremes(ext: Extremes) -> dict:
    return {
        "M_pos_max": _value_at(ext.M_pos_max),
        "M_neg_min": _value_at(ext.M_neg_min),
        "V_abs_max": _value_at(ext.V_abs_max),
        "V_max": _value_at(ext.V_max),
        "V_min": _value_at(ext.V_min),
        "zero_shear": [
            {"x": float(z.x), "M": float(z.M), "kind": z.kind} for z in ext.zero_shear
        ],
    }


def _deflection(d: DeflectionExtremes | None, EI: sp.Expr | None) -> dict | None:
    """처짐 결과 → dict (E, I 가 없으면 None). 처짐 값 단위는 m (화면에서 mm 로 표시)."""
    if d is None:
        return None
    return {
        "EI": float(EI),
        "n": float(d.n),
        "limit": float(d.limit),
        "ok": d.ok,
        "delta_down": _value_at(d.delta_down),
        "delta_up": _value_at(d.delta_up),
        "delta_abs": _value_at(d.delta_abs),
        "zero_slope": [{"x": float(z.x), "value": float(z.value)} for z in d.zero_slope],
    }


def sample_diagram(
    segments: list[Segment], x: sp.Symbol, ext: Extremes, n_points: int,
    dext: DeflectionExtremes | None = None,
) -> dict:
    """
    그래프용 x, V, M 배열을 만든다.

    구간마다 따로 점을 찍고 이어 붙인다. 그래서 구간 경계에는 같은 x 가 두 번
    (왼쪽 구간 끝값, 오른쪽 구간 시작값) 들어가고, 그래프에서 집중하중 위치의
    V 점프가 수직선으로 그려진다.
    V = 0 위치(M 극값)도 점으로 꼭 넣어서 그래프 꼭짓점이 정확한 값에 오게 한다.
    """
    L = float(segments[-1].x_end)
    special = [float(z.x) for z in ext.zero_shear if z.kind == "root"]
    has_defl = segments[0].delta is not None
    if dext is not None:
        special += [float(z.x) for z in dext.zero_slope]  # 처짐 최대 위치도 점으로
    xs: list[float] = []
    Vs: list[float] = []
    Ms: list[float] = []
    Ts: list[float] = []
    Ds: list[float] = []
    for seg in segments:
        a, b = float(seg.x_start), float(seg.x_end)
        n = max(20, int(n_points * (b - a) / L))  # 구간 길이에 비례, 최소 20점
        pts = [a + (b - a) * i / n for i in range(n + 1)]
        pts += [s for s in special if a < s < b]
        pts.sort()
        fV = sp.lambdify(x, seg.V, "math")
        fM = sp.lambdify(x, seg.M, "math")
        if has_defl:
            fT = sp.lambdify(x, seg.theta, "math")
            fD = sp.lambdify(x, seg.delta, "math")
        for p in pts:
            xs.append(p)
            Vs.append(float(fV(p)))
            Ms.append(float(fM(p)))
            if has_defl:
                Ts.append(float(fT(p)))
                Ds.append(float(fD(p)))
    out = {"x": xs, "V": Vs, "M": Ms}
    if has_defl:
        out["theta"] = Ts
        out["delta"] = Ds
    return out


def analyze(data: dict) -> dict:
    """
    입력 dict → 출력 dict (계산 전체).

    순서: 입력 해석 → solver(반력, V, M) → piecewise(구간식) → extremes(최대값)
          → steps(풀이 과정) → 그래프 배열
    """
    beam = parse_beam(data)
    result = solve(beam)  # 입력 오류면 여기서 ValueError
    segments = split_segments(result)
    ext = find_extremes(segments, result.x)
    dext = None
    if result.delta is not None:  # E, I 를 넣었을 때만 처짐 최대값·판정
        dext = find_deflection_extremes(segments, result.x, result.L, exact(beam.defl_n))
    steps = build_steps(result, segments, ext, dext)
    x = result.x

    reactions = []
    for r in result.reactions:
        reactions.append({
            "label": r.label,
            "kind": r.kind,
            "x": float(r.x),
            "R": float(r.R),
            "M": None if r.M is None else float(r.M),
            # 반력 모멘트 회전 방향 (예: "반시계 ↺ 41.00 kN·m")
            "M_dir": None if r.M is None else moment_direction_text(r.M),
        })

    return {
        "ok": True,
        "L": float(result.L),
        "reactions": reactions,
        "diagram": sample_diagram(segments, x, ext, int(data.get("n_points", 400)), dext),
        "segments": [
            {
                "x_start": float(s.x_start),
                "x_end": float(s.x_end),
                "V_latex": poly_latex(s.V, x),
                "M_latex": poly_latex(s.M, x),
            }
            for s in segments
        ],
        "extremes": _extremes(ext),
        "deflection": _deflection(dext, result.EI),
        "steps": [asdict(s) for s in steps.steps],
        "warnings": result.warnings + steps.warnings,
    }


# ─────────────────────────────────────────────────────────────
# JS 에서 부르는 입구
# ─────────────────────────────────────────────────────────────
def engine_info() -> str:
    """
    엔진 정보 JSON: SymPy 버전과 측정된 부호 보정 계수.
    Pyodide 의 SymPy 는 PC 에 설치된 버전과 다를 수 있어서, 화면에 표시해 확인할 수 있게 한다.
    """
    from core.sign import get_factors  # 이 함수에서만 쓰므로 안에서 import

    f = get_factors()
    return json.dumps({
        "sympy": sp.__version__,
        "factors": {"shear": f.shear, "moment": f.moment, "moment_load": f.moment_load,
                    "deflection": f.deflection, "slope": f.slope},
    })


def run(input_json: str) -> str:
    """
    JSON 문자열을 받아 JSON 문자열을 돌려준다. app.js 가 부르는 유일한 함수.

    입력 오류(ValueError)는 사용자에게 보여줄 메시지로, 예상 못 한 오류는
    "내부 오류" 로 감싸서 돌려준다. 어떤 경우에도 예외를 JS 로 던지지 않는다.

    예:
        run('{"L":6,"supports":[{"kind":"pin","x":0},{"kind":"roller","x":6}],'
            '"loads":[{"type":"udl","w":10,"x_start":0,"x_end":6}]}')
    """
    try:
        data = json.loads(input_json)
        out: dict[str, Any] = analyze(data)
    except ValueError as e:
        out = {"ok": False, "error": str(e)}
    except Exception as e:  # noqa: BLE001 — 화면이 멈추지 않게 모든 오류를 메시지로 바꾼다
        out = {"ok": False, "error": f"내부 오류: {type(e).__name__}: {e}"}
    # ensure_ascii=False : 한글을 \uXXXX 로 바꾸지 않고 그대로 둔다
    return json.dumps(out, ensure_ascii=False)
