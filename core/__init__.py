"""
core 패키지 — 보 계산 엔진 (순수 Python)

이 폴더의 파일들은 HTML/JS/Pyodide/Plotly를 전혀 모른다.
입력은 core.model 의 데이터 클래스, 출력도 Python 객체이다.
웹(web/bridge.py)이든 서버든 콘솔(core/demo.py)이든 같은 엔진을 그대로 쓴다.

계산 흐름 (손계산 순서와 같다):
    model.Beam  ──solver.solve()──▶  반력, V(x), M(x) (특이함수 식)
                ──piecewise.split_segments()──▶  구간별 다항식
                ──extremes.find_extremes()──▶  최대·최소값과 위치

참고: 폴더 안에 __init__.py 파일이 있으면 Python은 그 폴더를 "패키지"로 인식하고
      `from core.model import Beam` 처럼 불러올 수 있게 된다.
"""
