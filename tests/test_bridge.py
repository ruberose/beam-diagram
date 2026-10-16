"""
test_bridge.py — web/bridge.py 의 JSON 입출력 확인

[역할]
    브라우저 없이 일반 Python 에서 bridge.run() 을 불러,
    입력 JSON → 출력 JSON 형식이 app.js 가 기대하는 모양인지 확인한다.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# web/ 폴더는 패키지가 아니므로 import 경로에 직접 추가한다.
# (브라우저에서는 app.js 가 bridge.py 를 Pyodide 홈 폴더에 두므로 이 처리가 필요 없다)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "web"))

import bridge  # noqa: E402  (경로 추가 뒤에 import 해야 하므로 맨 위에 둘 수 없다)


def call(data: dict) -> dict:
    return json.loads(bridge.run(json.dumps(data)))


SIMPLE_UDL = {
    "L": 6,
    "supports": [{"kind": "pin", "x": 0}, {"kind": "roller", "x": 6}],
    "loads": [{"type": "udl", "w": 10, "x_start": 0, "x_end": 6}],
}


def test_simple_udl_output_shape():
    out = call(SIMPLE_UDL)
    assert out["ok"] is True
    assert [r["R"] for r in out["reactions"]] == [30.0, 30.0]
    assert out["extremes"]["M_pos_max"] == {"value": 45.0, "x": 3.0, "side": ""}
    assert out["extremes"]["M_neg_min"] is None
    d = out["diagram"]
    assert len(d["x"]) == len(d["V"]) == len(d["M"]) > 100
    assert max(d["M"]) == 45.0  # V=0 위치를 점으로 넣었으므로 꼭짓점이 정확
    assert [s["key"] for s in out["steps"]][-1] == "singularity"
    assert out["warnings"] == []


def test_point_load_jump_has_duplicate_x():
    """집중하중 위치에서 V 점프가 수직선으로 그려지도록 같은 x 가 두 번 들어간다."""
    out = call({
        "L": 6,
        "supports": [{"kind": "pin", "x": 0}, {"kind": "roller", "x": 6}],
        "loads": [{"type": "point", "P": 12, "x": 3}],
    })
    d = out["diagram"]
    idx = [i for i, xv in enumerate(d["x"]) if xv == 3.0]
    assert len(idx) == 2
    assert d["V"][idx[0]] == 6.0 and d["V"][idx[1]] == -6.0


def test_cantilever_moment_direction():
    out = call({
        "L": 3,
        "supports": [{"kind": "fixed", "x": 0}],
        "loads": [{"type": "linear", "w_start": 0, "w_end": 12, "x_start": 0, "x_end": 3},
                  {"type": "moment", "M": 5, "x": 3}],
    })
    (r,) = out["reactions"]
    assert r["M"] == -41.0
    assert r["M_dir"] == "반시계 ↺ 41.00 kN·m"


def test_input_errors_are_messages():
    """잘못된 입력은 예외가 아니라 ok=false + 한국어 메시지로 돌아와야 한다."""
    out = call({"L": 6, "supports": [{"kind": "pin", "x": 0}], "loads": []})
    assert out["ok"] is False and "지원하는 지점 구성" in out["error"]
    out = call({**SIMPLE_UDL, "L": ""})
    assert out["ok"] is False and "비어" in out["error"]
    assert json.loads(bridge.run("이건 JSON 아님"))["ok"] is False


def test_deflection_output():
    """E, I 를 넣으면 deflection 결과와 그래프용 delta 배열이 나온다 (Phase 5A)."""
    out = call({**SIMPLE_UDL, "E": 205000, "I": 1.2e8, "defl_n": 360})
    d = out["deflection"]
    assert d["EI"] == 24600.0
    assert d["delta_down"]["x"] == 3.0
    assert abs(d["delta_down"]["value"] - 5 * 10 * 6**4 / (384 * 24600)) < 1e-15
    assert d["ok"] is True and abs(d["limit"] - 6 / 360) < 1e-15
    assert len(out["diagram"]["delta"]) == len(out["diagram"]["x"])
    assert "deflection" in [s["key"] for s in out["steps"]]


def test_no_stiffness_no_deflection():
    out = call({**SIMPLE_UDL, "E": "", "I": ""})
    assert out["deflection"] is None and "delta" not in out["diagram"]


def test_app_js_loads_every_core_module():
    """
    app.js 의 CORE_FILES 목록에 core/*.py 가 모두 있어야 Pyodide 에서 import 된다.
    (Phase 5B 에서 macaulay.py, compat.py 를 빠뜨려 브라우저에서만 실패한 적이 있다)
    """
    import re

    root = Path(__file__).resolve().parent.parent
    app_js = (root / "web" / "app.js").read_text(encoding="utf-8")
    block = re.search(r"const CORE_FILES = \[(.*?)\];", app_js, re.S).group(1)
    listed = set(re.findall(r'"([\w.]+\.py)"', block))
    on_disk = {p.name for p in (root / "core").glob("*.py")} - {"demo.py"}
    assert on_disk - listed == set(), f"app.js CORE_FILES 에 빠진 파일: {on_disk - listed}"
