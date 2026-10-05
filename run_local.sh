#!/usr/bin/env bash
# Local run: more tiles allowed than on the hosted app. Opens in the browser.
export GRAIN_MAX_TILES=${GRAIN_MAX_TILES:-40}
exec streamlit run app.py
