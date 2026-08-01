@echo off
setlocal

if not exist ".venv\Scripts\python.exe" (
  echo [MoaView] 가상환경이 없습니다. 먼저 다음 명령을 실행하세요:
  echo python -m venv .venv
  echo .venv\Scripts\python -m pip install -r requirements.txt
  pause
  exit /b 1
)

echo [MoaView] Streamlit 앱을 시작합니다.
".venv\Scripts\python.exe" -m streamlit run app.py

endlocal

