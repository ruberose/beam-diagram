"""
model.py — 보·지점·하중 데이터 정의

[역할]
    계산에 필요한 입력값(경간, 지점, 하중)을 담는 "그릇"만 정의한다.
    계산은 하지 않는다. 입력이 말이 되는지(위치가 보 안에 있는지 등) 검사만 한다.

[관계]
    - 이 파일을 부르는 곳: solver.py, piecewise.py, demo.py, tests/, (나중에) web/bridge.py
    - 이 파일이 부르는 곳: 없음 (표준 라이브러리만 사용)

[부호 규약 — 실무 규약, 입력 기준]
    - 집중하중 P, 분포하중 w : 아래 방향이 양(+)
    - 집중 모멘트 M          : 시계방향이 양(+)
    - 위치 x                 : 보 왼쪽 끝을 0으로, 오른쪽으로 갈수록 증가 [m]

[단위] 길이 m, 힘 kN, 분포하중 kN/m, 모멘트 kN·m

[Python 문법 메모]
    - `from __future__ import annotations`
        : 타입 힌트를 문자열처럼 늦게 해석하게 해서, 아직 정의되지 않은 클래스 이름도
          힌트에 쓸 수 있게 한다. (동작에는 영향 없음)
    - 타입 힌트 `x: float`
        : "x에는 실수가 들어간다"는 표시. Python이 강제하지는 않지만 읽는 사람과
          편집기가 이해하기 쉬워진다.
    - @dataclass
        : 클래스 위에 붙이면 __init__(생성자) 등을 자동으로 만들어 준다.
          `PointLoad(P=10, x=2)` 처럼 바로 객체를 만들 수 있다.
          frozen=True 를 주면 만든 뒤에 값을 바꿀 수 없다(실수로 입력이 바뀌는 것 방지).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Union

# Literal["pin", "roller", "fixed"] : 이 세 문자열 중 하나만 허용한다는 타입 힌트
SupportKind = Literal["pin", "roller", "fixed"]


# ─────────────────────────────────────────────────────────────
# 지점
# ─────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Support:
    """
    지점 하나.

    매개변수:
        kind : "pin"(핀, 힌지) / "roller"(롤러) / "fixed"(고정단)
        x    : 지점 위치 [m]

    구조역학적 의미:
        - 핀·롤러 : 연직 반력 R 하나를 만든다 (이 프로그램은 연직하중만 다루므로
                    핀의 수평 반력은 항상 0이라 계산하지 않는다)
        - 고정단  : 연직 반력 R 과 반력 모멘트 M 두 개를 만든다

    예:
        Support("pin", 0.0), Support("roller", 6.0)   # 경간 6 m 단순보
        Support("fixed", 0.0)                          # 왼쪽 고정 캔틸레버
    """

    kind: SupportKind
    x: float


# ─────────────────────────────────────────────────────────────
# 하중 4종류
# ─────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class PointLoad:
    """
    집중하중.

    매개변수:
        P : 하중 크기 [kN], 아래 방향 +
        x : 작용 위치 [m]

    예: PointLoad(P=20, x=3)   # x=3 m 에 아래로 20 kN
    """

    P: float
    x: float


@dataclass(frozen=True)
class UniformLoad:
    """
    등분포하중 (전체 또는 부분).

    매개변수:
        w       : 하중 세기 [kN/m], 아래 방향 +
        x_start : 시작 위치 [m]
        x_end   : 끝 위치 [m]  (x_start < x_end)

    구조역학적 의미:
        합력 W = w × (x_end − x_start), 작용점은 구간 중앙.

    예: UniformLoad(w=10, x_start=0, x_end=6)   # 전체 6 m 에 10 kN/m
    """

    w: float
    x_start: float
    x_end: float


@dataclass(frozen=True)
class LinearLoad:
    """
    선형 변화 분포하중 (삼각형·사다리꼴).

    매개변수:
        w_start : 시작점 하중 세기 [kN/m], 아래 방향 +
        w_end   : 끝점 하중 세기 [kN/m], 아래 방향 +
        x_start : 시작 위치 [m]
        x_end   : 끝 위치 [m]

    구조역학적 의미:
        w(x) = w_start + (w_end − w_start)·(x − x_start)/(x_end − x_start)
        0 → w 인 삼각형이면 합력 W = w·ℓ/2, 작용점은 큰 쪽에서 ℓ/3.

    예: LinearLoad(w_start=0, w_end=12, x_start=0, x_end=6)   # 0→12 kN/m 삼각형
    """

    w_start: float
    w_end: float
    x_start: float
    x_end: float


@dataclass(frozen=True)
class MomentLoad:
    """
    집중 모멘트 하중.

    매개변수:
        M : 모멘트 크기 [kN·m], 시계방향 +
        x : 작용 위치 [m]

    예: MomentLoad(M=30, x=6)   # x=6 m 에 시계방향 30 kN·m
    """

    M: float
    x: float


# Union[A, B, ...] : "A 또는 B 또는 ..." 중 하나라는 타입 힌트
Load = Union[PointLoad, UniformLoad, LinearLoad, MomentLoad]


# ─────────────────────────────────────────────────────────────
# 보 전체
# ─────────────────────────────────────────────────────────────
@dataclass
class Beam:
    """
    보 하나 (경간 + 지점 목록 + 하중 목록).

    매개변수:
        L        : 보 전체 길이 [m] (내민보라면 돌출부 포함 전체 길이)
        supports : Support 목록
        loads    : 하중 목록 (PointLoad / UniformLoad / LinearLoad / MomentLoad)
        E        : 탄성계수 [MPa] (선택 — E, I 를 모두 넣어야 처짐을 계산한다)
        I        : 단면2차모멘트 [mm⁴] (선택)
        defl_n   : 허용처짐 L/n 의 n (기본 360)

    처짐 계산은 EI [kN·m²] 를 쓴다. 실무 단위 E [MPa], I [mm⁴] 로 받아서
        EI = E × I × 10⁻⁹   (1 MPa = 10³ kN/m², 1 mm⁴ = 10⁻¹² m⁴)
    로 고정 환산한다 (solver.stiffness_EI).

    Phase 1 지원 범위 (정정보만):
        - 단순보/내민보 : 핀·롤러 2개 (위치는 보 안 어디든)
        - 캔틸레버      : 고정단 1개 (보 왼쪽 끝 또는 오른쪽 끝)

    예:
        beam = Beam(
            L=6.0,
            supports=[Support("pin", 0), Support("roller", 6)],
            loads=[UniformLoad(w=10, x_start=0, x_end=6)],
        )

    [문법] field(default_factory=list)
        목록(list)을 기본값으로 쓸 때는 이렇게 써야 한다. 그냥 `= []` 라고 쓰면
        모든 Beam 객체가 같은 목록 하나를 공유하는 문제가 생겨서 dataclass가 막는다.
    """

    L: float
    supports: list[Support] = field(default_factory=list)
    loads: list[Load] = field(default_factory=list)
    E: float | None = None       # [MPa]  (float | None : 실수 또는 None(값 없음))
    I: float | None = None       # [mm⁴]
    defl_n: float = 360.0        # 허용처짐 L/n

    def has_stiffness(self) -> bool:
        """E 와 I 가 모두 입력되어 처짐을 계산할 수 있으면 True."""
        return self.E is not None and self.I is not None

    def kind(self) -> Literal["simple", "cantilever"]:
        """
        보 종류를 돌려준다: 고정단이 있으면 "cantilever", 아니면 "simple"(내민보 포함).
        validate()를 통과한 보에서만 의미가 있다.
        """
        if any(s.kind == "fixed" for s in self.supports):
            return "cantilever"
        return "simple"

    def validate(self) -> list[str]:
        """
        입력이 Phase 1 범위에서 계산 가능한지 검사한다.

        반환값:
            경고 메시지 목록 (계산은 가능하지만 사용자에게 알릴 것).
        예외:
            계산이 불가능한 입력이면 ValueError 를 일으킨다(raise).
            메시지는 화면에 그대로 보여줄 수 있게 한국어로 쓴다.
        """
        warnings: list[str] = []

        # ── 1. 경간 ──
        if self.L <= 0:
            raise ValueError("경간 L은 0보다 커야 합니다.")

        # ── 2. 지점 구성: 정정보(미지 반력 2개)인지 확인 ──
        #   연직 평형식은 ΣF_y = 0, ΣM = 0 두 개뿐이므로
        #   미지수(반력)가 정확히 2개여야 손계산(정정)으로 풀린다.
        fixed = [s for s in self.supports if s.kind == "fixed"]
        hinged = [s for s in self.supports if s.kind in ("pin", "roller")]

        for s in self.supports:
            if s.kind not in ("pin", "roller", "fixed"):
                raise ValueError(f"알 수 없는 지점 종류입니다: {s.kind}")
            if not (0 <= s.x <= self.L):
                raise ValueError(f"지점 위치 x={s.x:g} m 가 보 범위(0 ~ {self.L:g} m) 밖입니다.")

        if len(fixed) == 1 and len(hinged) == 0:
            # 캔틸레버: 고정단 하나 (반력 R, M → 미지수 2개)
            if fixed[0].x not in (0, self.L):
                raise ValueError("고정단은 보의 왼쪽 끝(x=0) 또는 오른쪽 끝(x=L)에 있어야 합니다.")
        elif len(fixed) == 0 and len(hinged) == 2:
            # 단순보/내민보: 핀·롤러 둘 (반력 R_A, R_B → 미지수 2개)
            if hinged[0].x == hinged[1].x:
                raise ValueError("두 지점의 위치가 같습니다. 서로 다른 위치에 두세요.")
            kinds = sorted(s.kind for s in hinged)
            if kinds != ["pin", "roller"]:
                # 연직하중만 다루므로 계산 결과(연직 반력)는 같다. 수평 안정성만 알린다.
                warnings.append(
                    "지점이 핀+롤러 조합이 아닙니다. 연직하중만 계산하므로 결과는 같지만, "
                    "실제 구조에서는 수평 방향 구속을 확인하세요."
                )
        else:
            raise ValueError(
                "Phase 1에서는 정정보만 지원합니다: 핀·롤러 2개(단순보/내민보) 또는 고정단 1개(캔틸레버)."
            )

        # ── 3. 하중 위치 ──
        if not self.loads:
            warnings.append("하중이 없습니다. 모든 값이 0입니다.")

        # ── 4. 강성 (처짐용, 선택) ──
        if (self.E is None) != (self.I is None):
            # != : 둘 중 하나만 None 이면 True → 하나만 입력한 경우
            raise ValueError("처짐을 계산하려면 E 와 I 를 모두 입력하세요 (처짐이 필요 없으면 둘 다 비우세요).")
        if self.has_stiffness() and (self.E <= 0 or self.I <= 0):
            raise ValueError("E 와 I 는 0보다 커야 합니다.")
        if self.defl_n <= 0:
            raise ValueError("허용처짐 L/n 의 n 은 0보다 커야 합니다.")

        for load in self.loads:
            if isinstance(load, (PointLoad, MomentLoad)):
                # isinstance(객체, 클래스) : 객체가 그 클래스인지 확인하는 내장 함수
                if not (0 <= load.x <= self.L):
                    raise ValueError(f"하중 위치 x={load.x:g} m 가 보 범위(0 ~ {self.L:g} m) 밖입니다.")
            else:
                if not (0 <= load.x_start < load.x_end <= self.L):
                    raise ValueError(
                        f"분포하중 구간 {load.x_start:g} ~ {load.x_end:g} m 가 올바르지 않습니다 "
                        f"(0 ≤ 시작 < 끝 ≤ L={self.L:g} 이어야 함)."
                    )

        return warnings
