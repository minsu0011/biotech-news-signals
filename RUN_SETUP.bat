@echo off
chcp 65001 >nul
cd /d "%~dp0"
call conda activate py310
python -m pip install -r requirements.txt
if errorlevel 1 goto :fail

python bio_news_30m_v3.py --self-test
if errorlevel 1 goto :fail

python bio_news_30m_v3.py --build-universe
if errorlevel 1 goto :fail

echo.
echo ================================================================
echo Universe complete.
echo US SEC events can be collected automatically.
echo KR exact KIND events and historical KR 1-minute bars must exist
echo before a combined real SEALED certification can be valid.
echo ================================================================
python bio_news_30m_v3.py --collect-sec
if errorlevel 1 goto :fail

echo.
echo Next, place KIND exact timestamp CSV and KR 1m data, then run:
echo python bio_news_30m_v3.py --kind-csv kind_events.csv
echo python bio_news_30m_v3.py --combine-events
echo python bio_news_30m_v3.py --label
echo python bio_news_30m_v3.py --search
echo python bio_news_30m_v3.py --seal
pause
exit /b 0

:fail
echo FAILED.
pause
exit /b 1
