@echo off
REM Run unit tests and show coverage report
python -m pytest --cov=. --cov-report=term
pause
