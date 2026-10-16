"""
conftest.py — pytest 설정 파일 (프로젝트 루트)

이 파일이 루트에 있으면 pytest 가 루트 폴더를 import 경로에 넣어 주므로,
tests/ 안에서 `from core.model import Beam` 처럼 core 패키지를 불러올 수 있다.
내용은 비어 있어도 된다.
"""
