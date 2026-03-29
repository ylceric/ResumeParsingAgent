"""Shared Streamlit UI helpers."""

from __future__ import annotations

import streamlit as st


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
