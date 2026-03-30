"""Shared Streamlit UI helpers."""

from __future__ import annotations

import streamlit as st

_BRANDING_TEXT = "ylceric 用于北控数科面试任务"


def inject_branding_sidebar_and_footer() -> None:
    """Sidebar + fixed footer on every page (call once per run from app entry)."""
    with st.sidebar:
        st.caption(_BRANDING_TEXT)
    st.markdown(
        f"""
<style>
.bkg-interview-footer {{
  position: fixed;
  bottom: 0;
  left: 0;
  right: 0;
  z-index: 999990;
  text-align: center;
  padding: 6px 12px 10px;
  font-size: 0.8rem;
  opacity: 0.9;
  border-top: 1px solid rgba(128,128,128,0.25);
  background: rgba(255,255,255,0.92);
  backdrop-filter: blur(6px);
}}
@media (prefers-color-scheme: dark) {{
  .bkg-interview-footer {{
    background: rgba(14,17,23,0.92);
    border-top-color: rgba(255,255,255,0.12);
  }}
}}
</style>
<div class="bkg-interview-footer">{_BRANDING_TEXT}</div>
""",
        unsafe_allow_html=True,
    )


def inject_material_header_styles() -> None:
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,400,0,0');
.section-header {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  border-radius: 10px;
  background: rgba(120,120,120,0.08);
  margin: 10px 0 8px 0;
}
.section-header .material-symbols-rounded {
  font-size: 22px;
}
.section-header-title {
  font-weight: 600;
}
.section-header-subtitle {
  font-size: 0.86rem;
  opacity: 0.85;
}
</style>
""",
        unsafe_allow_html=True,
    )


def render_material_header(icon: str, title: str, subtitle: str | None = None) -> None:
    subtitle_html = (
        f"<div class='section-header-subtitle'>{subtitle}</div>" if subtitle else ""
    )
    st.markdown(
        f"""
<div class="section-header">
  <span class="material-symbols-rounded">{icon}</span>
  <div>
    <div class="section-header-title">{title}</div>
    {subtitle_html}
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
