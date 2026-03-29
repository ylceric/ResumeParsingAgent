"""Candidate ingestion and talent library."""

from __future__ import annotations

import base64
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from utils.bootstrap import ensure_project_on_syspath

ensure_project_on_syspath()

from repositories.candidate_repository import CandidateRepository
from repositories.vector_repository import VectorRepository
from services.candidate_chat import answer_candidate_question
from services.document_parser import SUPPORTED_EXTENSIONS, is_supported_filename
from services.ingestion import IngestionService
from utils.config import (
    SQLITE_PATH,
    AppConfig,
    ensure_data_dirs,
    resolve_stored_resume_path,
)
from utils.ui import inject_material_header_styles, render_material_header


@st.cache_resource
def _config() -> AppConfig:
    return AppConfig.from_env()


@st.cache_resource
def _candidate_repo() -> CandidateRepository:
    ensure_data_dirs()
    return CandidateRepository(SQLITE_PATH)


@st.cache_resource
def _vector_repo() -> VectorRepository | None:
    cfg = _config()
    if not cfg.openai_api_key:
        return None
    try:
        return VectorRepository(
            api_key=cfg.openai_api_key,
            embedding_model=cfg.openai_embedding_model,
        )
    except Exception:
        return None


def _ingestion() -> IngestionService:
    return IngestionService(_config(), _candidate_repo(), _vector_repo())


inject_material_header_styles()


st.title("人才录入 / 人才库")

cfg = _config()
if not cfg.openai_api_key:
    st.warning("未检测到 `OPENAI_API_KEY`，无法使用 LLM 解析与向量入库。请在 `.env` 中配置。")

render_material_header(
    "upload_file",
    "批量上传简历",
    "支持 PDF / DOCX / TXT / PNG / JPG / JPEG",
)
st.caption(
    f"支持：{', '.join(sorted(SUPPORTED_EXTENSIONS))} · 图片优先 OCR，文本过短时可走视觉模型（见 `OPENAI_VISION_FALLBACK`）"
)

uploaded = st.file_uploader(
    "选择文件（可多选）",
    type=[x.lstrip(".") for x in SUPPORTED_EXTENSIONS],
    accept_multiple_files=True,
)

if uploaded and st.button("开始解析并入库", type="primary"):
    service = _ingestion()
    results = []
    progress = st.progress(0.0, text="处理中…")
    for idx, file in enumerate(uploaded):
        if not is_supported_filename(file.name):
            results.append(
                {
                    "file": file.name,
                    "ok": False,
                    "message": "Unsupported extension",
                }
            )
            continue
        suffix = Path(file.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file.getbuffer())
            tmp_path = tmp.name
        try:
            r = service.ingest_file(tmp_path, file.name)
            results.append(
                {
                    "file": r.source_file,
                    "ok": r.success,
                    "candidate_id": r.candidate_id or "",
                    "message": r.message,
                }
            )
        finally:
            Path(tmp_path).unlink(missing_ok=True)
        progress.progress((idx + 1) / len(uploaded), text=f"已完成 {idx + 1}/{len(uploaded)}")
    progress.empty()
    st.dataframe(pd.DataFrame(results), width="stretch")

st.divider()
render_material_header("search", "人才检索", "关键词搜索与自然语言检索")
col_a, col_b = st.columns(2)
with col_a:
    kw = st.text_input(
        "关键词（摘要 / 原文 / 姓名 / 技能 / 手机 / 微信 / 地点 / 求职意向 / 最高学历）",
        "",
    )
with col_b:
    nl = st.text_input("自然语言", "")

repo = _candidate_repo()
vec = _vector_repo()

records = []
if kw.strip():
    records = repo.keyword_search(kw.strip())
elif nl.strip() and vec:
    try:
        docs = vec.similarity_search_with_scores(nl.strip(), k=30)
        ids = []
        seen = set()
        for doc, _score in docs:
            cid = (doc.metadata or {}).get("candidate_id")
            if cid and cid not in seen:
                seen.add(cid)
                ids.append(cid)
        by_id = repo.get_many(ids)
        records = [by_id[i] for i in ids if i in by_id]
    except Exception as exc:  # noqa: BLE001
        st.error(f"向量检索失败：{exc}")
elif nl.strip() and not vec:
    st.info("向量库不可用（缺少 API Key 或初始化失败），请使用关键词搜索。")

if not kw.strip() and not nl.strip():
    records = repo.list_all(limit=200)

st.metric("当前库内候选人数", repo.count())

if records:
    rows = []
    for r in records:
        rows.append(
            {
                "candidate_id": r.candidate_id,
                "name": r.profile.name,
                "phone": r.profile.phone,
                "wechat": r.profile.wechat,
                "location": r.profile.location,
                "job_intent": (r.profile.job_intent[:40] + "…")
                if r.profile.job_intent and len(r.profile.job_intent) > 40
                else r.profile.job_intent,
                "birth_year": r.profile.birth_year,
                "latest_graduation": r.profile.latest_graduation_date,
                "highest_education": r.profile.highest_education,
                "email": r.profile.email,
                "years_exp": r.profile.years_of_experience,
                "skills": ", ".join(r.profile.skills[:12]),
                "summary": (r.profile.summary[:120] + "…")
                if len(r.profile.summary) > 120
                else r.profile.summary,
                "source": r.source_file,
            }
        )
    st.dataframe(pd.DataFrame(rows), width="stretch")

    pick = st.selectbox(
        "查看详情",
        options=[r.candidate_id for r in records],
        format_func=lambda cid: next(
            (f"{r.profile.name or '（未命名）'} — {cid[:8]}…" for r in records if r.candidate_id == cid),
            cid,
        ),
    )
    detail = next((r for r in records if r.candidate_id == pick), None)
    if detail:
        left_col, right_col = st.columns([7, 3])
        with left_col:
            render_material_header(
                "badge",
                "已选中人才",
                f"{detail.profile.name or '候选人'} · {detail.candidate_id[:8]}…",
            )
            c1, c2, c3 = st.columns(3)
            c1.write(f"**邮箱** {detail.profile.email or '—'}")
            c2.write(f"**手机** {detail.profile.phone or '—'}")
            c3.write(f"**微信** {detail.profile.wechat or '—'}")
            c4, c5, c6 = st.columns(3)
            c4.write(f"**地点** {detail.profile.location or '—'}")
            c5.write(f"**求职意向** {detail.profile.job_intent or '—'}")
            c6.write(f"**经验年限** {detail.profile.years_of_experience or '—'}")
            c7, c8, c9 = st.columns(3)
            c7.write(f"**出生年份** {detail.profile.birth_year or '—'}")
            c8.write(f"**最近毕业时间** {detail.profile.latest_graduation_date or '—'}")
            c9.write(f"**最高学历** {detail.profile.highest_education or '—'}")
            st.write("**摘要**")
            st.write(detail.profile.summary or "—")
            st.write("**技能**")
            st.write(", ".join(detail.profile.skills) or "—")
            st.write("**风险标记**")
            st.write(", ".join(detail.profile.risk_flags) or "—")
            with st.expander("原始文件与提取文本", expanded=False):
                st.caption(f"入库文件名：`{detail.source_file}`")
                col_orig, col_txt = st.columns(2)
                with col_orig:
                    st.markdown("**原始文件**")
                    stored = resolve_stored_resume_path(
                        detail.candidate_id, detail.source_file
                    )
                    if stored and stored.is_file():
                        ext = stored.suffix.lower()
                        data = stored.read_bytes()
                        try:
                            if ext == ".pdf":
                                b64 = base64.b64encode(data).decode("ascii")
                                pdf_html = (
                                    "<iframe "
                                    "src='data:application/pdf;base64,"
                                    f"{b64}' "
                                    "width='100%' height='700' "
                                    "style='border:1px solid #ddd;border-radius:6px;'>"
                                    "</iframe>"
                                )
                                components.html(pdf_html, height=720, scrolling=True)
                                st.download_button(
                                    "下载 PDF 查看",
                                    data=data,
                                    file_name=stored.name,
                                    mime="application/pdf",
                                    key=f"dl_pdf_{detail.candidate_id}",
                                )
                            elif ext in (".png", ".jpg", ".jpeg", ".webp", ".gif"):
                                st.image(data)
                            elif ext == ".txt":
                                st.text(
                                    stored.read_text(encoding="utf-8", errors="replace")[
                                        :50000
                                    ]
                                    or "—"
                                )
                            elif ext == ".docx":
                                st.download_button(
                                    "下载 DOCX",
                                    data=data,
                                    file_name=stored.name,
                                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                    key=f"dl_docx_{detail.candidate_id}",
                                )
                                st.caption("DOCX 请下载后用 Word / WPS 打开预览。")
                            else:
                                st.download_button(
                                    "下载原始文件",
                                    data=data,
                                    file_name=stored.name,
                                    key=f"dl_any_{detail.candidate_id}",
                                )
                        except Exception as exc:  # noqa: BLE001
                            st.warning(f"无法预览：{exc}")
                            st.download_button(
                                "下载原始文件",
                                data=data,
                                file_name=stored.name,
                                key=f"dl_fallback_{detail.candidate_id}",
                            )
                    else:
                        st.info(
                            "本地未找到原始文件副本（可能为早期数据或入库时复制失败）。"
                        )
                with col_txt:
                    st.markdown("**提取文本**（解析 / OCR 结果）")
                    st.text((detail.raw_text or "—")[:50000])
                    if len(detail.raw_text or "") > 50000:
                        st.caption("文本过长，此处仅显示前 5 万字符。")
        with right_col:
            render_material_header("chat", "候选人简历问答")
            st.caption(
                f"当前选中：**{detail.profile.name or '（未命名）'}**  \nID: `{detail.candidate_id}`"
            )
            if not cfg.openai_api_key:
                st.info("未配置 `OPENAI_API_KEY`，无法使用对话问答。")
            else:
                if "candidate_chat_histories" not in st.session_state:
                    st.session_state["candidate_chat_histories"] = {}
                chat_map = st.session_state["candidate_chat_histories"]
                if detail.candidate_id not in chat_map:
                    chat_map[detail.candidate_id] = []
                chat_history = chat_map[detail.candidate_id]

                if st.button(
                    "清空当前候选人对话",
                    key=f"clear_chat_{detail.candidate_id}",
                    use_container_width=True,
                ):
                    chat_map[detail.candidate_id] = []
                    st.rerun()

                for msg in chat_history:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])

                user_question = st.chat_input(
                    "问 AI：请基于该候选人简历回答…",
                    key=f"chat_input_{detail.candidate_id}",
                )
                if user_question:
                    chat_history.append({"role": "user", "content": user_question})
                    with st.chat_message("user"):
                        st.markdown(user_question)
                    with st.chat_message("assistant"):
                        with st.spinner("思考中…"):
                            try:
                                answer = answer_candidate_question(
                                    candidate=detail,
                                    question=user_question,
                                    config=cfg,
                                    history=chat_history[:-1],
                                )
                            except Exception as exc:  # noqa: BLE001
                                answer = f"抱歉，问答失败：{type(exc).__name__}: {exc}"
                        st.markdown(answer)
                    chat_history.append({"role": "assistant", "content": answer})
else:
    st.info("暂无候选人，请先上传简历。")
