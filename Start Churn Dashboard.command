#!/bin/zsh
cd "$(dirname "$0")"
exec .venv/bin/python -m streamlit run dashboard/app.py --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false
