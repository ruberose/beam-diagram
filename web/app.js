/*
  app.js — 화면 동작 전체

  [역할]
    1. Pyodide(브라우저용 Python)를 불러오고, core/*.py 와 bridge.py 를
       Pyodide 가상 파일시스템에 복사한 뒤 import 한다. (빌드 도구 없이)
    2. 입력 패널의 지점·하중 줄을 추가/삭제하고, 마지막 입력을 localStorage 에 저장·복원한다.
    3. [계산] → 입력을 JSON 으로 모아 bridge.run(json) 호출 → 결과 JSON 을 받아
       Plotly 그래프(보 그림 + SFD + BMD), 주요 결과 표, KaTeX 풀이 과정을 그린다.

  [Python ↔ JS 데이터 흐름]
      collectInput()  →  JSON.stringify  →  bridge.run(문자열)   [Python, Pyodide 안]
                                                 │
      render(result)  ←  JSON.parse      ←  JSON 문자열 반환  ◀──┘
    문자열 하나만 주고받으므로 Python 객체를 JS 에서 직접 다룰 일이 없다.

  [관계]
    - index.html 이 이 파일을 불러온다 (Plotly, KaTeX, Pyodide 를 먼저 불러온 뒤).
    - 이 파일이 fetch 하는 파일: ../core/*.py, ./bridge.py

  [JS 문법 메모]
    - const / let       : 상수 / 바꿀 수 있는 변수 선언
    - async / await     : 시간이 걸리는 일(파일 받기, Pyodide 로딩)을 기다리는 문법.
                          await 가 붙은 줄은 끝날 때까지 다음 줄로 넘어가지 않지만,
                          그동안 화면은 멈추지 않는다.
    - 화살표 함수 (a) => a * 2 : 짧은 함수 표기
    - `문자열 ${값}`    : 템플릿 문자열 (Python 의 f-문자열과 같다)
    - document.querySelector("#id") : HTML 요소 찾기
*/

"use strict"; // 오타로 생기는 실수(선언 안 한 변수 등)를 오류로 알려 달라는 설정

// ─────────────────────────────────────────────────────────────
// 설정값
// ─────────────────────────────────────────────────────────────
const PYODIDE_URL = "https://cdn.jsdelivr.net/pyodide/v0.27.2/full/";
// core/ 에 .py 파일을 새로 만들면 여기에도 꼭 추가한다 (tests/test_bridge.py 가 빠진 파일을 잡아낸다).
// demo.py 는 콘솔 전용이라 넣지 않는다.
const CORE_FILES = [
  "__init__.py", "model.py", "sign.py", "solver.py",
  "piecewise.py", "extremes.py", "steps.py",
  "macaulay.py", "compat.py",
];
const STORAGE_KEY = "beam-diagram:last-input:v1";

// 하중 종류별 입력칸: [JSON 키, 화면 라벨]
const LOAD_TYPES = {
  point:  { label: "집중하중", fields: [["P", "P"], ["x", "x"]] },
  udl:    { label: "등분포",   fields: [["w", "w"], ["x_start", "시작"], ["x_end", "끝"]] },
  linear: { label: "선형분포", fields: [["w_start", "w₁"], ["w_end", "w₂"], ["x_start", "시작"], ["x_end", "끝"]] },
  moment: { label: "모멘트",   fields: [["M", "M"], ["x", "x"]] },
};

// 처음 열었을 때(저장된 입력이 없을 때) 보여줄 예제: 내민보
const DEFAULT_INPUT = {
  L: 8,
  supports: [{ kind: "pin", x: 0 }, { kind: "roller", x: 6 }],
  loads: [
    { type: "udl", w: 10, x_start: 0, x_end: 4 },
    { type: "point", P: 20, x: 8 },
  ],
  E: 205000,   // MPa (강재)
  I: 1.2e8,    // mm⁴
  defl_n: 360,
};

// 전역 상태
let bridge = null;      // Pyodide 안의 Python bridge 모듈
let lastResult = null;  // 마지막 계산 결과 (BMD 방향 전환 등 다시 그릴 때 사용)
let lastInput = null;

const $ = (sel) => document.querySelector(sel); // 짧게 쓰려는 도우미

// ─────────────────────────────────────────────────────────────
// 숫자 표시
// ─────────────────────────────────────────────────────────────
/** 소수 2자리 문자열. −0.00 은 0.00 으로. */
function fmt(v, digits = 2) {
  const s = Number(v).toFixed(digits);
  return /^-0\.?0*$/.test(s) ? s.slice(1) : s;
}
/** 경계 좌우 표시: "left" → x⁻, "right" → x⁺ */
function sideMark(side) {
  return side === "left" ? "⁻" : side === "right" ? "⁺" : "";
}

// ─────────────────────────────────────────────────────────────
// 1. Pyodide 로딩
// ─────────────────────────────────────────────────────────────
async function startPython() {
  const status = (t) => { $("#loading-text").textContent = t; };
  try {
    status("Python 실행기(Pyodide) 불러오는 중…");
    // loadPyodide 는 index.html 에서 불러온 pyodide.js 가 만들어 둔 전역 함수
    const pyodide = await loadPyodide({ indexURL: PYODIDE_URL });

    status("SymPy 불러오는 중…");
    await pyodide.loadPackage("sympy");

    status("계산 엔진 파일 복사 중…");
    // core/*.py 를 받아서 Pyodide 가상 파일시스템의 /home/pyodide/core/ 에 쓴다.
    // (Pyodide 안의 Python 은 진짜 하드디스크가 아니라 메모리 속 가상 폴더를 본다)
    const home = "/home/pyodide";
    pyodide.FS.mkdirTree(`${home}/core`);
    for (const name of CORE_FILES) {
      const text = await fetchText(new URL(`../core/${name}`, location.href));
      pyodide.FS.writeFile(`${home}/core/${name}`, text);
    }
    pyodide.FS.writeFile(`${home}/bridge.py`, await fetchText(new URL("bridge.py", location.href)));
    pyodide.runPython(`import sys\nif "${home}" not in sys.path: sys.path.insert(0, "${home}")`);

    status("계산 엔진 준비 중 (부호 규약 확인)…");
    bridge = pyodide.pyimport("bridge");
    // 한 번 미리 계산해 둔다: SymPy 부호 보정(sign.py)이 이때 실행되어 이후 계산이 빨라진다.
    bridge.run(JSON.stringify(DEFAULT_INPUT));
    const info = JSON.parse(bridge.engine_info());
    const f = info.factors;
    $("#engine").textContent =
      `SymPy ${info.sympy} · 부호 보정 V×${f.shear} M×${f.moment} 모멘트하중×${f.moment_load} δ×${f.deflection} θ×${f.slope}`;

    $("#loading").hidden = true;
    $("#calc").disabled = false;
    calculate();
  } catch (err) {
    status(`불러오기 실패: ${err}`);
    $("#loading-text").style.color = "var(--danger)";
    console.error(err);
  }
}

/** 파일 내용을 문자열로 받는다. 개발 중 옛 파일이 캐시에서 나오지 않게 no-cache. */
async function fetchText(url) {
  const res = await fetch(url, { cache: "no-cache" });
  if (!res.ok) throw new Error(`${url} 를 받지 못했습니다 (${res.status})`);
  return res.text();
}

// ─────────────────────────────────────────────────────────────
// 2. 입력 패널
// ─────────────────────────────────────────────────────────────
/** 지점 한 줄을 만든다 (index.html 의 <template> 복사). */
function addSupportRow(s = { kind: "pin", x: 0 }) {
  const row = $("#support-row").content.firstElementChild.cloneNode(true);
  row.querySelector(".kind").value = s.kind;
  row.querySelector(".x").value = s.x ?? ""; // ?? : 왼쪽이 null/undefined 면 오른쪽 값
  row.querySelector(".del").addEventListener("click", () => { row.remove(); saveInput(); });
  $("#supports").appendChild(row);
}

/** 하중 한 줄을 만든다. 종류를 바꾸면 입력칸을 다시 만들되, 같은 이름의 값은 유지한다. */
function addLoadRow(load = { type: "point", P: 10, x: 0 }) {
  const row = document.createElement("div");
  row.className = "row load-row";

  const select = document.createElement("select");
  select.className = "kind";
  for (const [key, def] of Object.entries(LOAD_TYPES)) {
    select.add(new Option(def.label, key));
  }
  select.value = load.type;

  const fields = document.createElement("div");
  fields.style.display = "contents"; // 이 div 는 배치에 영향 없이 묶음 역할만

  const del = document.createElement("button");
  del.type = "button";
  del.className = "del";
  del.title = "삭제";
  del.textContent = "✕";
  del.addEventListener("click", () => { row.remove(); saveInput(); });

  // 종류에 맞는 입력칸 만들기
  const buildFields = (values) => {
    fields.replaceChildren(); // 기존 입력칸 비우기
    for (const [key, label] of LOAD_TYPES[select.value].fields) {
      const lab = document.createElement("label");
      lab.className = "mini";
      lab.textContent = label + " ";
      const input = document.createElement("input");
      input.type = "number";
      input.step = "any";
      input.inputMode = "decimal";
      input.dataset.key = key; // data-key="P" : 나중에 값을 모을 때 JSON 키로 쓴다
      input.value = values[key] ?? "";
      lab.appendChild(input);
      fields.appendChild(lab);
    }
  };
  select.addEventListener("change", () => {
    buildFields(readLoadRow(row)); // 지금 값을 읽어서 같은 키는 그대로 옮긴다
    saveInput();
  });

  row.append(select, fields, del);
  buildFields(load);
  $("#loads").appendChild(row);
}

/** 하중 줄 하나의 값을 객체로 읽는다. 숫자로 바꿀 수 없으면 문자열 그대로 둔다(Python 이 오류 메시지를 만든다). */
function readLoadRow(row) {
  const obj = { type: row.querySelector("select.kind").value };
  for (const input of row.querySelectorAll("input[data-key]")) {
    obj[input.dataset.key] = input.value === "" ? "" : Number(input.value);
  }
  return obj;
}

/** 화면 입력 전체를 bridge.py 가 기대하는 JSON 모양의 객체로 모은다. */
function collectInput() {
  const L = $("#span").value;
  return {
    L: L === "" ? "" : Number(L),
    supports: [...$("#supports").children].map((row) => ({
      kind: row.querySelector(".kind").value,
      x: row.querySelector(".x").value === "" ? "" : Number(row.querySelector(".x").value),
    })),
    loads: [...$("#loads").children].map(readLoadRow),
    // 처짐용 (선택): 비어 있으면 "" 로 보내고 bridge.py 가 '입력 없음'으로 처리한다
    E: numOrEmpty($("#E").value),
    I: numOrEmpty($("#I").value),
    defl_n: numOrEmpty($("#defl-n").value),
  };
}

/** 입력칸 문자열 → 숫자, 비었으면 "" */
function numOrEmpty(v) {
  return v === "" ? "" : Number(v);
}

/** 입력 객체를 화면에 채운다. */
function applyInput(data) {
  $("#span").value = data.L ?? "";
  $("#E").value = data.E ?? "";
  $("#I").value = data.I ?? "";
  $("#defl-n").value = data.defl_n ?? "";
  $("#supports").replaceChildren();
  $("#loads").replaceChildren();
  (data.supports || []).forEach(addSupportRow);
  (data.loads || []).forEach(addLoadRow);
}

/** 마지막 입력 저장 — 사생활 모드 등에서 실패해도 계산은 되게 try/catch */
function saveInput() {
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(collectInput())); } catch (e) { /* 무시 */ }
}
function loadSavedInput() {
  try {
    const s = localStorage.getItem(STORAGE_KEY);
    if (s) return JSON.parse(s);
  } catch (e) { /* 무시 */ }
  return null;
}

/**
 * 지점 프리셋 버튼: 현재 경간 L 기준으로 지점 줄을 다시 만든다.
 *   simple      단순보      핀 0 + 롤러 L
 *   cantilever  캔틸레버    고정 0
 *   propped     고정-핀     고정 0 + 롤러 L          (1차 부정정)
 *   fixed_fixed 양단 고정   고정 0 + 고정 L          (2차 부정정)
 *   span2/3     연속보      핀 0 + 롤러를 등간격으로 (2경간: 1차, 3경간: 2차 부정정)
 * 연속보 지점 위치는 소수 4자리로 반올림한다 (예: L=10, 3경간 → 3.3333, 6.6667).
 */
function applyPreset(kind) {
  const L = Number($("#span").value) || 0;
  const round4 = (v) => Math.round(v * 1e4) / 1e4;
  const PRESETS = {
    simple: [["pin", 0], ["roller", L]],
    cantilever: [["fixed", 0]],
    propped: [["fixed", 0], ["roller", L]],
    fixed_fixed: [["fixed", 0], ["fixed", L]],
    span2: [["pin", 0], ["roller", round4(L / 2)], ["roller", L]],
    span3: [["pin", 0], ["roller", round4(L / 3)], ["roller", round4((2 * L) / 3)], ["roller", L]],
  };
  $("#supports").replaceChildren();
  for (const [k, x] of PRESETS[kind]) addSupportRow({ kind: k, x });
  saveInput();
}

// ─────────────────────────────────────────────────────────────
// 3. 계산
// ─────────────────────────────────────────────────────────────
async function calculate() {
  if (!bridge) return;
  const input = collectInput();
  saveInput();
  const btn = $("#calc");
  btn.disabled = true;
  btn.textContent = "계산 중…";
  // 버튼 글자가 화면에 먼저 그려지도록 아주 잠깐 양보한다.
  // (Python 계산은 화면과 같은 스레드에서 돌기 때문에, 계산 중에는 화면이 갱신되지 않는다)
  await new Promise((r) => setTimeout(r, 30));
  try {
    const result = JSON.parse(bridge.run(JSON.stringify(input)));
    if (!result.ok) {
      showError(result.error);
    } else {
      showError(null);
      lastResult = result;
      lastInput = input;
      render();
    }
  } catch (err) {
    showError(`실행 오류: ${err}`);
    console.error(err);
  } finally { // 성공하든 실패하든 버튼은 원래대로
    btn.disabled = false;
    btn.textContent = "계산";
  }
}

function showError(msg) {
  const box = $("#error");
  box.hidden = !msg;
  box.textContent = msg || "";
}

function render() {
  renderPlot(lastResult, lastInput);
  renderWarnings(lastResult.warnings);
  renderSummary(lastResult);
  renderSteps(lastResult.steps);
}

// ─────────────────────────────────────────────────────────────
// 4. 그래프 (Plotly) — 한 그림 안에 3단: 위 보 그림 / 가운데 SFD / 아래 BMD
//    세 단이 x축 하나("x")를 같이 쓰므로 위치가 정확히 정렬된다.
//    E, I 를 넣으면 맨 아래에 처짐도(4단째, y4 축)가 추가된다.
// ─────────────────────────────────────────────────────────────
const COLOR = { beam: "#37474f", V: "#1f6feb", M: "#d84315", D: "#2e7d32", load: "#6a1b9a", support: "#455a64" };

function renderPlot(res, input) {
  const L = res.L;
  const d = res.diagram;
  const bmdDown = $("#bmd-down").checked; // true: 양(+)의 모멘트를 축 아래로 (인장측)

  const traces = [
    {
      x: d.x, y: d.V, yaxis: "y2", name: "V", type: "scatter", mode: "lines",
      line: { color: COLOR.V, width: 2 }, fill: "tozeroy", fillcolor: "rgba(31,111,235,0.12)",
      hovertemplate: "x = %{x:.2f} m<br>V = %{y:.2f} kN<extra></extra>",
    },
    {
      x: d.x, y: d.M, yaxis: "y", name: "M", type: "scatter", mode: "lines",
      line: { color: COLOR.M, width: 2 }, fill: "tozeroy", fillcolor: "rgba(216,67,21,0.12)",
      hovertemplate: "x = %{x:.2f} m<br>M = %{y:.2f} kN·m<extra></extra>",
    },
  ];

  const beam = beamDrawing(res, input);
  const labels = valueLabels(res, bmdDown);
  traces.push(...labels.markers); // ...배열 : 배열 원소를 하나씩 풀어서 넣는다

  // ── 처짐도 (E, I 가 있을 때만) ── 값은 m 로 오므로 mm 로 바꿔 그린다.
  const defl = res.deflection;
  const hasDefl = !!defl && !!d.delta; // !! : 값을 true/false 로 바꾼다
  $("#plot").classList.toggle("with-deflection", hasDefl);
  if (hasDefl) {
    traces.push({
      x: d.x, y: d.delta.map((v) => v * 1000), yaxis: "y4", name: "δ", type: "scatter", mode: "lines",
      line: { color: COLOR.D, width: 2 }, fill: "tozeroy", fillcolor: "rgba(46,125,50,0.10)",
      customdata: d.theta, // 마우스를 올렸을 때 처짐각도 함께 보여준다
      hovertemplate: "x = %{x:.2f} m<br>δ = %{y:.2f} mm<br>θ = %{customdata:.5f} rad<extra></extra>",
    });
    // 최대 처짐(아래)·최대 솟음(위) 라벨
    for (const v of [defl.delta_down, defl.delta_up]) {
      if (!v) continue;
      const ymm = v.value * 1000;
      labels.annotations.push({ xref: "x", yref: "y4", x: v.x, y: ymm,
        text: `δ ${fmt(ymm)} mm<br>x=${fmt(v.x)}`, showarrow: true, arrowhead: 0, arrowwidth: 1,
        arrowcolor: "#777", ax: 0, ay: ymm > 0 ? 26 : -26, // 아래 처짐은 아래쪽에 라벨 (축이 뒤집혀 있음)
        font: { size: 11 }, bgcolor: "rgba(255,255,255,0.85)" });
      traces.push({ x: [v.x], y: [ymm], yaxis: "y4", type: "scatter", mode: "markers",
        hoverinfo: "skip", marker: { size: 6, color: "#222" } });
    }
  }
  // 단 배치 (아래에서 위로): [처짐] → BMD → SFD → 보 그림
  const dom = hasDefl
    ? { y4: [0, 0.22], M: [0.29, 0.52], V: [0.59, 0.79], beam: [0.84, 1] }
    : { M: [0, 0.37], V: [0.44, 0.75], beam: [0.80, 1] };

  const layout = {
    margin: { l: 64, r: 16, t: 12, b: 44 },
    showlegend: false,
    hovermode: "x",
    dragmode: false, // 아이패드에서 그래프를 끌어도 화면 스크롤이 되게 확대·이동 끔
    xaxis: {
      range: [-0.04 * L, 1.04 * L], title: { text: "x (m)" },
      anchor: hasDefl ? "y4" : "y", // x축 눈금은 맨 아래 단 밑에
      zeroline: false, showspikes: true, spikemode: "across", spikethickness: 1,
      spikedash: "dot", spikecolor: "#888",
    },
    // 아래 BMD
    yaxis: {
      domain: dom.M, title: { text: "M (kN·m)" }, zeroline: true, zerolinecolor: "#999",
      autorange: bmdDown ? "reversed" : true,
    },
    // 가운데 SFD
    yaxis2: { domain: dom.V, title: { text: "V (kN)" }, zeroline: true, zerolinecolor: "#999" },
    // 위 보 그림 (숫자 눈금 없음)
    yaxis3: {
      domain: dom.beam, range: [-2.1, 2.1], showticklabels: false, showgrid: false,
      zeroline: false, fixedrange: true,
    },
    // 맨 아래 처짐도: 아래 처짐(+)이 축 아래로 보이도록 뒤집는다 (지시서 D2)
    ...(hasDefl ? { yaxis4: {
      domain: dom.y4, title: { text: "δ (mm)" }, zeroline: true, zerolinecolor: "#999",
      autorange: "reversed",
    } } : {}),
    shapes: beam.shapes,
    annotations: [...beam.annotations, ...labels.annotations],
  };

  Plotly.react($("#plot"), traces, layout, {
    responsive: true,
    displaylogo: false,
    modeBarButtonsToRemove: ["select2d", "lasso2d", "zoom2d", "pan2d", "zoomIn2d", "zoomOut2d", "autoScale2d"],
  });
}

/**
 * 보 형상·지점·하중을 Plotly shape/annotation 으로 그린다 (y3 축, 보는 y=0).
 * 좌표는 데이터 단위(x: m, y: −2.1 ~ 2.1 의 그림용 높이).
 */
function beamDrawing(res, input) {
  const L = res.L;
  const shapes = [];
  const annotations = [];
  const ax = { xref: "x", yref: "y3" };
  const dx = L * 0.022; // 지점 삼각형 반폭
  // 보 양 끝 가까이의 글자는 가운데 정렬하면 그림 밖으로 잘리므로 안쪽으로 붙인다
  const anchor = (x) => (x < 0.1 * L ? "left" : x > 0.9 * L ? "right" : "center");

  // 보 본체
  shapes.push({ ...ax, type: "rect", x0: 0, x1: L, y0: -0.13, y1: 0.13,
    fillcolor: COLOR.beam, line: { width: 0 } });

  // ── 지점 ──
  for (const r of res.reactions) {
    const x = r.x;
    if (r.kind === "fixed") {
      // 고정단: 굵은 수직선 + 바깥쪽 빗금
      const out = x === 0 ? -1 : 1; // 보 바깥 방향
      shapes.push({ ...ax, type: "line", x0: x, x1: x, y0: -0.9, y1: 0.9,
        line: { color: COLOR.support, width: 4 } });
      for (let i = 0; i < 6; i++) {
        const y = -0.85 + i * 0.32;
        shapes.push({ ...ax, type: "line", x0: x, x1: x + out * dx, y0: y, y1: y - 0.25,
          line: { color: COLOR.support, width: 1.5 } });
      }
    } else {
      // 핀·롤러: 삼각형. 롤러는 삼각형과 바닥선 사이에 틈을 둔다.
      const tipY = -0.13, baseY = -0.75;
      shapes.push({ ...ax, type: "path",
        path: `M ${x},${tipY} L ${x - dx},${baseY} L ${x + dx},${baseY} Z`,
        fillcolor: r.kind === "pin" ? COLOR.support : "#fff", line: { color: COLOR.support, width: 1.5 } });
      const groundY = r.kind === "roller" ? -0.95 : baseY;
      shapes.push({ ...ax, type: "line", x0: x - dx * 1.4, x1: x + dx * 1.4, y0: groundY, y1: groundY,
        line: { color: COLOR.support, width: 2 } });
    }
    // 반력 값 라벨
    let text = `R<sub>${r.label}</sub> = ${fmt(r.R)} kN`;
    if (r.M !== null) text += `<br>M<sub>${r.label}</sub> = ${fmt(r.M)} kN·m (${r.M_dir})`;
    annotations.push({ ...ax, x, y: -1.25, text, showarrow: false, yanchor: "top",
      xanchor: anchor(x), align: anchor(x) === "center" ? "center" : anchor(x),
      font: { size: 11, color: COLOR.support } });
  }

  // ── 하중 ──
  // 분포하중 높이는 가장 큰 세기를 기준으로 비율을 맞춘다
  const wMax = Math.max(1e-9, ...input.loads.flatMap((l) =>
    l.type === "udl" ? [Math.abs(l.w)] : l.type === "linear" ? [Math.abs(l.w_start), Math.abs(l.w_end)] : []));
  const h = (w) => 0.25 + 0.95 * Math.abs(w) / wMax; // 그림 높이

  for (const l of input.loads) {
    if (l.type === "point") {
      // 화살표: 꼬리(ax, ay) → 머리(x, y). 아래 방향(+)이면 위에서 내려온다.
      const down = l.P >= 0;
      annotations.push({ ...ax, axref: "x", ayref: "y3",
        x: l.x, y: down ? 0.15 : -0.15, ax: l.x, ay: down ? 1.75 : -1.75,
        showarrow: true, arrowhead: 2, arrowsize: 1.2, arrowwidth: 2, arrowcolor: COLOR.load,
        text: `P = ${fmt(l.P)}`, font: { size: 11, color: COLOR.load },
        xanchor: anchor(l.x), yanchor: down ? "bottom" : "top" });
    } else if (l.type === "udl" || l.type === "linear") {
      const w1 = l.type === "udl" ? l.w : l.w_start;
      const w2 = l.type === "udl" ? l.w : l.w_end;
      // 아래 방향(+)이면 보 위에, 위 방향(−)이면 보 아래에 그린다
      const s1 = w1 >= 0 ? 1 : -1, s2 = w2 >= 0 ? 1 : -1;
      const base = 0.18;
      shapes.push({ ...ax, type: "path",
        path: `M ${l.x_start},${s1 * base} L ${l.x_start},${s1 * (base + h(w1))} ` +
              `L ${l.x_end},${s2 * (base + h(w2))} L ${l.x_end},${s2 * base} Z`,
        fillcolor: "rgba(106,27,154,0.15)", line: { color: COLOR.load, width: 1.5 } });
      const text = l.type === "udl" ? `w = ${fmt(l.w)}` : `w = ${fmt(w1)} → ${fmt(w2)}`;
      const topY = Math.max(base + h(w1), base + h(w2));
      annotations.push({ ...ax, x: (l.x_start + l.x_end) / 2, y: (s1 + s2 >= 0 ? 1 : -1) * (topY + 0.1),
        text, showarrow: false, font: { size: 11, color: COLOR.load },
        yanchor: s1 + s2 >= 0 ? "bottom" : "top" });
    } else if (l.type === "moment") {
      // 회전 화살표 기호: 시계 ↻ / 반시계 ↺.
      // 분포하중 그림과 겹치지 않게 보 바로 위 한 줄에 기호와 크기를 나란히 쓰고,
      // 반투명 흰 배경을 깔아 겹쳐도 읽히게 한다.
      annotations.push({ ...ax, x: l.x, y: 0.45, showarrow: false,
        text: `<b style="font-size:18px">${l.M >= 0 ? "↻" : "↺"}</b> M = ${fmt(Math.abs(l.M))}`,
        xanchor: anchor(l.x), bgcolor: "rgba(255,255,255,0.85)",
        font: { size: 11, color: COLOR.load } });
    }
  }
  return { shapes, annotations };
}

/**
 * SFD·BMD 위의 값 라벨 (최대·최소, V=0 위치).
 * 라벨은 그래프 바깥쪽(값이 그려진 방향)으로 띄워서 선과 겹치지 않게 한다.
 */
function valueLabels(res, bmdDown) {
  const ex = res.extremes;
  const annotations = [];
  const mx = [], my = [], myaxis = [];
  const seen = new Set(); // 같은 위치·축에 라벨이 겹치지 않게

  const add = (axis, x, y, text, reversed) => {
    const key = `${axis}:${x.toFixed(6)}:${y.toFixed(6)}`;
    if (seen.has(key)) return;
    seen.add(key);
    // 화면에서 값이 아래로 그려지면 라벨도 아래로 (ay 는 픽셀, + 가 아래)
    const drawnDown = (y < 0) !== reversed;
    annotations.push({ xref: "x", yref: axis, x, y, text, showarrow: true, arrowhead: 0,
      arrowwidth: 1, arrowcolor: "#777", ax: 0, ay: drawnDown ? 26 : -26,
      font: { size: 11 }, bgcolor: "rgba(255,255,255,0.85)" });
    mx.push(x); my.push(y); myaxis.push(axis);
  };

  // SFD: 최대·최소
  for (const v of [ex.V_max, ex.V_min]) {
    if (v && Math.abs(v.value) > 1e-9) add("y2", v.x, v.value, `${fmt(v.value)}`, false);
  }
  // SFD: 구간 안의 V = 0 위치
  for (const z of ex.zero_shear) {
    if (z.kind === "root") add("y2", z.x, 0, `V=0<br>x=${fmt(z.x)}`, false);
  }
  // BMD: 정·부모멘트 최대, V=0 위치의 M
  if (ex.M_pos_max) add("y", ex.M_pos_max.x, ex.M_pos_max.value,
    `M<sub>max</sub><sup>+</sup> ${fmt(ex.M_pos_max.value)}<br>x=${fmt(ex.M_pos_max.x)}`, bmdDown);
  if (ex.M_neg_min) add("y", ex.M_neg_min.x, ex.M_neg_min.value,
    `M<sub>max</sub><sup>−</sup> ${fmt(ex.M_neg_min.value)}<br>x=${fmt(ex.M_neg_min.x)}`, bmdDown);
  for (const z of ex.zero_shear) {
    add("y", z.x, z.M, `${fmt(z.M)}<br>x=${fmt(z.x)}`, bmdDown);
  }

  // 라벨 위치의 점(마커). Plotly trace 는 y축 하나에만 붙으므로 축(SFD/BMD)마다 하나씩 만든다.
  const markers = ["y2", "y"].map((axis) => ({
    x: mx.filter((_, i) => myaxis[i] === axis),
    y: my.filter((_, i) => myaxis[i] === axis),
    yaxis: axis, type: "scatter", mode: "markers", hoverinfo: "skip",
    marker: { size: 6, color: "#222" },
  }));
  return { annotations, markers };
}

// ─────────────────────────────────────────────────────────────
// 5. 경고·주요 결과·풀이 과정
// ─────────────────────────────────────────────────────────────
function renderWarnings(list) {
  const box = $("#warnings");
  box.hidden = !list.length;
  box.innerHTML = list.length
    ? `<b>확인할 점</b><ul>${list.map((w) => `<li>${escapeHtml(w)}</li>`).join("")}</ul>` : "";
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

/** KaTeX 로 수식 하나를 HTML 문자열로 (오류가 나도 멈추지 않게 throwOnError: false) */
function tex(latex, display = false) {
  return katex.renderToString(latex, { displayMode: display, throwOnError: false, strict: "ignore" });
}

function renderSummary(res) {
  const KIND = { pin: "핀", roller: "롤러", fixed: "고정단" };
  const ex = res.extremes;
  const at = (v) => v ? `${fmt(v.x)}${sideMark(v.side)}` : "—";

  // 반력 표
  let html = `<h3>반력 (위 +, 모멘트 시계 +)</h3><div class="table-wrap"><table>
    <tr><th>지점</th><th>종류</th><th class="num">x (m)</th><th class="num">R (kN)</th><th class="num">M (kN·m)</th></tr>`;
  for (const r of res.reactions) {
    html += `<tr><td>${r.label}</td><td>${KIND[r.kind]}</td><td class="num">${fmt(r.x)}</td>
      <td class="num">${fmt(r.R)}${r.R < 0 ? " ↓" : ""}</td>
      <td class="num">${r.M === null ? "—" : `${fmt(r.M)}<br><span class="muted">${r.M_dir}</span>`}</td></tr>`;
  }
  html += `</table></div>`;

  // 주요값 표
  html += `<h3>주요값</h3><div class="table-wrap"><table>
    <tr><th>항목</th><th class="num">값</th><th class="num">위치 x (m)</th></tr>
    <tr><td>정모멘트 최대</td><td class="num">${ex.M_pos_max ? fmt(ex.M_pos_max.value) + " kN·m" : "없음"}</td><td class="num">${at(ex.M_pos_max)}</td></tr>
    <tr><td>부모멘트 최대</td><td class="num">${ex.M_neg_min ? fmt(ex.M_neg_min.value) + " kN·m" : "없음"}</td><td class="num">${at(ex.M_neg_min)}</td></tr>
    <tr><td>|V| 최대</td><td class="num">${fmt(Math.abs(ex.V_abs_max.value))} kN <span class="muted">(V = ${fmt(ex.V_abs_max.value)})</span></td><td class="num">${at(ex.V_abs_max)}</td></tr>`;
  for (const z of ex.zero_shear) {
    html += `<tr><td>V = 0 ${z.kind === "jump" ? "(하중점에서 부호 변화)" : ""}</td>
      <td class="num">M = ${fmt(z.M)} kN·m</td><td class="num">${fmt(z.x)}</td></tr>`;
  }
  html += `</table></div>`;

  // 처짐 표 (E, I 를 넣었을 때만)
  const df = res.deflection;
  if (df) {
    const mmv = (v) => fmt(v * 1000);
    html += `<h3>처짐 (δ 아래 +, EI = ${fmt(df.EI)} kN·m²)</h3><div class="table-wrap"><table>
      <tr><th>항목</th><th class="num">값</th><th class="num">위치 x (m)</th></tr>
      <tr><td>최대 처짐 (아래)</td><td class="num">${df.delta_down ? mmv(df.delta_down.value) + " mm" : "없음"}</td><td class="num">${at(df.delta_down)}</td></tr>
      <tr><td>최대 솟음 (위)</td><td class="num">${df.delta_up ? mmv(df.delta_up.value) + " mm" : "없음"}</td><td class="num">${at(df.delta_up)}</td></tr>
      <tr><td>허용처짐 L/${fmt(df.n, 0)}</td><td class="num">${mmv(df.limit)} mm</td><td class="num">—</td></tr>
      <tr><td>판정 (|δ|max ≤ L/n)</td><td class="num">${mmv(Math.abs(df.delta_abs.value))} mm
        <span class="${df.ok ? "ok" : "ng"}">${df.ok ? "OK" : "NG"}</span></td><td class="num">${at(df.delta_abs)}</td></tr>
      </table></div>`;
  }

  // 구간별 식 표
  html += `<h3>구간별 식</h3><div class="table-wrap"><table>
    <tr><th>구간 (m)</th><th>V(x) (kN)</th><th>M(x) (kN·m)</th></tr>`;
  for (const s of res.segments) {
    html += `<tr><td class="num">${fmt(s.x_start)} ≤ x &lt; ${fmt(s.x_end)}</td>
      <td>${tex(s.V_latex)}</td><td>${tex(s.M_latex)}</td></tr>`;
  }
  html += `</table></div>`;
  $("#summary").innerHTML = html;
}

/** 풀이 과정: 단계마다 접을 수 있는 <details>. 특이함수 단계는 토글로만 보인다. */
function renderSteps(steps) {
  const box = $("#steps");
  box.replaceChildren();
  const showSing = $("#show-singularity").checked;
  for (const step of steps) {
    const det = document.createElement("details");
    det.open = true;
    det.dataset.key = step.key;
    if (step.hidden) det.hidden = !showSing; // 특이함수 식: 기본 숨김
    let body = "";
    for (const line of step.lines) {
      if (line.text) body += `<p class="note">${escapeHtml(line.text)}</p>`;
      if (line.latex) body += `<div class="eq">${tex(line.latex, true)}</div>`;
    }
    det.innerHTML = `<summary>${escapeHtml(step.title)}</summary><div class="step-body">${body}</div>`;
    box.appendChild(det);
  }
}

// ─────────────────────────────────────────────────────────────
// 6. 시작
// ─────────────────────────────────────────────────────────────
function init() {
  applyInput(loadSavedInput() || DEFAULT_INPUT);

  $("#add-support").addEventListener("click", () => { addSupportRow({ kind: "roller", x: "" }); saveInput(); });
  $("#add-load").addEventListener("click", () => { addLoadRow(); saveInput(); });
  $("#calc").addEventListener("click", calculate);
  document.querySelectorAll("[data-preset]").forEach((b) =>
    b.addEventListener("click", () => applyPreset(b.dataset.preset)));

  // 입력이 바뀔 때마다 저장 (이벤트 위임: 입력 패널 전체에서 한 번에 받는다)
  $(".input-panel").addEventListener("input", saveInput);
  // Enter 키로 계산
  $(".input-panel").addEventListener("keydown", (e) => { if (e.key === "Enter") calculate(); });

  // 표시 옵션: 다시 계산하지 않고 마지막 결과로 다시 그린다
  $("#bmd-down").addEventListener("change", () => { if (lastResult) renderPlot(lastResult, lastInput); });
  $("#show-singularity").addEventListener("change", (e) => {
    document.querySelectorAll('#steps details[data-key="singularity"]').forEach((d) => { d.hidden = !e.target.checked; });
  });

  startPython();
}

init();
