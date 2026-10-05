@echo off
rem Local run: more tiles allowed than on the hosted app. Opens in the browser.
set GRAIN_MAX_TILES=40
streamlit run app.py
