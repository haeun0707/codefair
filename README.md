# MoaView

MoaView는 1~3개의 이미지에서 차량 후보를 탐지하고, 사용자가 직접 목표 차량을 선택한 뒤 원본 화질과 **보정 참고 영상**을 비교하는 교육용 AI/SW 프로젝트입니다.

AI 결과는 차량을 확정하거나 동일 차량임을 판정하지 않습니다. 보정 참고 영상도 원본에 존재하지 않는 번호판 문자나 세부 정보를 복원하지 않습니다.

## 현재 구현된 기능

- JPG/PNG 이미지 1~3개 입력
- 카메라 이름, 촬영 시각, 이동 방향 입력
- 개인정보 없는 가상 이미지 3장과 Mock 탐지로 전체 흐름 시연
- 로컬 Ultralytics YOLO 모델 lazy-load 및 CPU 차량 탐지
- 여러 차량 후보의 crop 표시와 사용자 직접 선택
- 객체탐지 확신도, 선명도, 평균 밝기, 대비 표시
- 원본 crop과 보수적인 OpenCV 보정 참고 영상 비교
- 선택된 crop 중 가장 선명한 근거와 한계 안내
- UI와 분리된 핵심 로직 및 pytest 단위 테스트

번호판 OCR, 자동 다중 CCTV 재식별, 얼굴 인식, 소유주 조회, 실시간 CCTV 연결은 현재 범위에 포함하지 않습니다.

## 빠른 실행 — Windows

Python 3.11 설치 후 저장소 폴더에서 PowerShell을 엽니다.

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pytest -q
streamlit run app.py
```

설치가 끝난 뒤에는 `run_starter_windows.bat`를 더블클릭해도 됩니다.

### Mock 모드

앱의 기본 설정입니다. 사이드바에서 `Mock`과 `가상 데모 이미지 3장 사용`을 선택하면 실제 이미지, 네트워크, YOLO 모델 없이 전체 UI 흐름을 확인할 수 있습니다.

Mock 모드의 탐지 상자, 차종, 확신도는 모두 화면 시험용 값입니다.

### 로컬 YOLO 모드

1. Ultralytics와 호환되는 경량 `.pt` 모델을 준비합니다.
2. 저장소의 `models` 폴더(기본 경로 `models/yolo11n.pt`)에 놓습니다.
3. 앱에서 `로컬 YOLO`를 선택하고 모델 경로를 입력합니다.
4. JPG/PNG 이미지 1~3개를 올린 뒤 `차량 탐지 시작`을 누릅니다.

앱은 모델을 자동 다운로드하지 않습니다. 테스트에서도 네트워크와 모델 다운로드가 발생하지 않습니다. `models/`와 `*.pt`는 `.gitignore`에 포함되어 있습니다.

## 지표 해석

- 객체탐지 확신도: YOLO가 해당 상자를 특정 객체 종류로 탐지한 내부 확신도입니다. 동일 차량일 확률이 아닙니다.
- 선명도: grayscale Laplacian 분산입니다. 높을수록 경계 변화가 많다는 뜻이지만 절대적인 품질 보증은 아닙니다.
- 평균 밝기: grayscale 픽셀의 평균값(0~255)입니다.
- 대비: grayscale 픽셀의 표준편차입니다.

지표는 보정본이 아닌 원본 crop에서 계산합니다.

## 테스트

```powershell
pytest -q
```

테스트는 fake/mock detector를 사용하고 YOLO를 열거나 다운로드하지 않습니다.

## 자주 생기는 오류

### `python`을 찾을 수 없음

Python 3.11을 설치할 때 `Add Python to PATH`를 선택했는지 확인하고 터미널을 다시 여세요. Windows에서는 `py -3.11` 명령으로 대신 실행할 수도 있습니다.

### `No module named streamlit` 또는 `No module named cv2`

가상환경이 활성화된 상태에서 다음 명령을 다시 실행하세요.

```powershell
python -m pip install -r requirements.txt
```

### YOLO 모델 파일을 찾지 못함

`models/yolo11n.pt`에 호환 모델을 준비하거나 앱에서 정확한 로컬 경로를 입력하세요. UI 확인만 필요하면 Mock 모드를 사용하세요.

### 차량 후보가 나오지 않음

차량이 더 크고 선명하게 보이는 JPG/PNG를 사용하거나 최소 탐지 확신도를 조금 낮추세요. 후보가 여러 대면 시스템이 자동 확정하지 않으므로 사용자가 직접 선택해야 합니다.

### 앱이 느림

CPU 환경에서는 큰 이미지와 YOLO 추론이 느릴 수 있습니다. 시연 전 실제 노트북에서 처리 시간을 확인하고, 필요하면 이미지 크기를 줄인 사본을 사용하세요.

## 데이터 안전

- 실제 영상과 입력 이미지는 `inputs/` 같은 로컬 폴더에만 두고 Git에 올리지 마세요.
- 실제 번호판이나 사람 얼굴 대신 가상 번호판, 모형 자동차, 허가받은 촬영 자료를 사용하세요.
- 업로드 이미지는 앱의 현재 세션에서 처리할 뿐 저장소에 쓰지 않습니다.

