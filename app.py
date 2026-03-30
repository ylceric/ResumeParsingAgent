"""Resume Parsing Agent — Streamlit entry (HR Copilot MVP)."""

from __future__ import annotations

from pathlib import Path

from utils.bootstrap import ensure_project_on_syspath

ensure_project_on_syspath()

import streamlit as st
from utils.ui import (
    inject_branding_sidebar_and_footer,
    inject_material_header_styles,
    render_material_header,
)

st.set_page_config(
    page_title="Resume Parsing Agent",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_branding_sidebar_and_footer()

def _home_page() -> None:
    inject_material_header_styles()
    render_material_header("home", "首页", "README")
    readme_path = Path(__file__).resolve().parent / "README.md"
    if readme_path.is_file():
        st.markdown(readme_path.read_text(encoding="utf-8", errors="replace"))
    else:
        st.warning("未找到 README.md")


navigation = st.navigation(
    [
        st.Page(_home_page, title="首页", icon=":material/home:", default=True),
        st.Page("pages/1_人才库.py", title="人才库", icon=":material/folder:"),
        st.Page("pages/2_JD匹配.py", title="JD匹配", icon=":material/target:"),
    ]
)
navigation.run()
