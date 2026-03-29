"""JD matching with explainable scores."""

from __future__ import annotations

import streamlit as st

from utils.bootstrap import ensure_project_on_syspath

ensure_project_on_syspath()

from repositories.candidate_repository import CandidateRepository
from repositories.vector_repository import VectorRepository
from services.jd_match_chat import answer_jd_match_question
from services.matching import MatchingService
from utils.config import (
    DEFAULT_JD_RETRIEVAL_TOP_K,
    DEFAULT_MATCH_CANDIDATE_TOP_N,
    SQLITE_PATH,
    AppConfig,
    ensure_data_dirs,
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


def _matching() -> MatchingService:
    return MatchingService(_config(), _candidate_repo(), _vector_repo())


inject_material_header_styles()
st.title("JD 匹配")

cfg = _config()
if not cfg.openai_api_key:
    st.warning("未检测到 `OPENAI_API_KEY`，无法进行 JD 结构化抽取与匹配分析。")

render_material_header("work", "岗位 JD 输入", "粘贴完整 JD 后开始召回与匹配")
jd_text = st.text_area("岗位 JD（粘贴完整描述）", height=240, placeholder="在此粘贴职位描述…")

with st.expander("匹配参数", expanded=False):
    st.caption("向量召回条数越大，候选池越宽；参与分析人数越多，调用 LLM 次数与耗时越高。")
    c1, c2 = st.columns(2)
    with c1:
        jd_retrieval_top_k = st.number_input(
            "向量召回 top-k（chunk 条数，去重后得到候选池）",
            min_value=5,
            max_value=100,
            value=DEFAULT_JD_RETRIEVAL_TOP_K,
            step=1,
            help="从 Chroma 取相似 chunk 的数量；同一候选人可能对应多条 chunk。",
        )
    with c2:
        match_candidate_top_n = st.number_input(
            "参与匹配分析的人数",
            min_value=1,
            max_value=40,
            value=DEFAULT_MATCH_CANDIDATE_TOP_N,
            step=1,
            help="对前 N 名候选人逐一调用匹配分析链。",
        )

if st.button("开始匹配", type="primary") and jd_text.strip():
    with st.spinner("正在抽取 JD 要求并分析候选人…"):
        try:
            jd, results = _matching().match(
                jd_text.strip(),
                keyword_fallback_query=jd_text[:500],
                jd_retrieval_top_k=int(jd_retrieval_top_k),
                match_candidate_top_n=int(match_candidate_top_n),
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"{type(exc).__name__}: {exc}")
            st.stop()

    st.session_state["jd_match_context"] = {
        "jd_text": jd_text.strip(),
        "jd_structured": jd.model_dump(),
        "results": [r.model_dump() for r in results],
    }
    st.session_state["jd_match_chat_history"] = []

    render_material_header("fact_check", "结构化 JD 需求（LLM 抽取）")
    st.json(jd.model_dump())

    if not results:
        st.warning("人才库为空或未能加载候选人。")
        st.stop()

    render_material_header("groups", "匹配结果（按总分排序）")
    for r in results:
        a = r.analysis
        with st.container():
            st.markdown(f"#### {r.name or '未命名'} · `{r.candidate_id}`")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("总分", f"{a.total_match_score:.1f}")
            c2.metric("技能", f"{a.skill_match_score:.1f}")
            c3.metric("经验", f"{a.experience_match_score:.1f}")
            c4.metric("项目", f"{a.project_relevance_score:.1f}")
            c5.metric("领域", f"{a.domain_relevance_score:.1f}")
            col_l, col_r = st.columns(2)
            with col_l:
                st.write("**匹配证据**")
                for x in a.matched_evidence:
                    st.write(f"- {x}")
                st.write("**优势**")
                for x in a.strengths:
                    st.write(f"- {x}")
            with col_r:
                st.write("**缺失 / 薄弱证据**")
                for x in a.missing_or_weak_evidence:
                    st.write(f"- {x}")
                st.write("**疑虑**")
                for x in a.concerns:
                    st.write(f"- {x}")
            st.write("**建议面试题**")
            for x in a.interview_questions:
                st.write(f"- {x}")
            st.divider()

if "jd_match_context" in st.session_state:
    ctx = st.session_state["jd_match_context"]
    render_material_header("chat", "JD 匹配结果问答（Chat Agent）")
    st.caption("可追问：排序依据、候选人对比、风险优先级、面试题定制、是否放宽 JD 条件等。")
    if not cfg.openai_api_key:
        st.info("未配置 `OPENAI_API_KEY`，无法使用问答。")
    else:
        if "jd_match_chat_history" not in st.session_state:
            st.session_state["jd_match_chat_history"] = []
        history = st.session_state["jd_match_chat_history"]

        col_a, col_b = st.columns([3, 1])
        with col_a:
            st.caption(
                f"当前上下文：{len(ctx.get('results', []))} 位候选人匹配结果"
            )
        with col_b:
            if st.button("清空本次问答", key="clear_jd_chat", use_container_width=True):
                st.session_state["jd_match_chat_history"] = []
                st.rerun()

        for msg in history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_question = st.chat_input(
            "问 AI：基于本次 JD 匹配结果回答…",
            key="jd_match_chat_input",
        )
        if user_question:
            history.append({"role": "user", "content": user_question})
            with st.chat_message("user"):
                st.markdown(user_question)
            with st.chat_message("assistant"):
                with st.spinner("思考中…"):
                    try:
                        from schemas.jd import JDRequirements
                        from schemas.match import CandidateMatchResult

                        jd_obj = JDRequirements.model_validate(ctx["jd_structured"])
                        result_objs = [
                            CandidateMatchResult.model_validate(x)
                            for x in ctx["results"]
                        ]
                        answer = answer_jd_match_question(
                            jd_text=ctx["jd_text"],
                            jd_structured=jd_obj,
                            results=result_objs,
                            question=user_question,
                            config=cfg,
                            history=history[:-1],
                        )
                    except Exception as exc:  # noqa: BLE001
                        answer = f"抱歉，问答失败：{type(exc).__name__}: {exc}"
                st.markdown(answer)
            history.append({"role": "assistant", "content": answer})

elif not jd_text.strip():
    st.caption("输入 JD 后点击「开始匹配」。流程：抽取结构化 JD → 向量召回 top-k → 对每位候选人单独做可解释匹配。")
