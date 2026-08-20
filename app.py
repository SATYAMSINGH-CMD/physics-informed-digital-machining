"""
Root Application Launcher for Digital Machining AI Digital Twin.
Run with: streamlit run app.py
"""

import sys
import os

# Delegate to streamlit_app/app.py
current_dir = os.path.dirname(os.path.abspath(__file__))
app_path = os.path.join(current_dir, "streamlit_app", "app.py")

with open(app_path, "r", encoding="utf-8") as f:
    code = f.read()

exec(compile(code, app_path, 'exec'))
