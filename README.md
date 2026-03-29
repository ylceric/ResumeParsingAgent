# Resume Parsing Agent（简历解析智能体）

面向 HR 场景的本地可运行 MVP：  
**批量简历入库** -> **结构化候选人画像** -> **SQLite + Chroma** -> **关键词/语义检索** -> **JD 结构化抽取** -> **召回 top-k** -> **可解释匹配分析** -> **对话式追问（Chat Agent）**。

## 项目开发方式说明

- 前期规划阶段：使用 OpenAI GPT 进行项目框架设计、模块拆分与主链路方案评估。
- 方案确定阶段：基于已确定的架构与边界，使用 Cursor 进行代码编写、重构、联调与页面迭代。

---

## 当前功能（已实现）

### 1) 人才录入 / 人才库

- 批量上传：`PDF`、`DOCX`、`TXT`、`PNG`、`JPG`、`JPEG`
- 文档解析策略：
  - 文本文件优先提取文本
  - 图片简历优先 OCR（`pytesseract`）
  - OCR 文本不足时可走多模态兜底（`OPENAI_VISION_FALLBACK=true`）
- LLM 解析为标准化 `CandidateProfile`（Pydantic 校验）
- 写入 SQLite + Chroma（分块向量，带 `candidate_id` 关联）
- 人才库检索：
  - 关键词检索（SQLite）
  - 自然语言检索（Chroma）
- 详情区支持：
  - 原始文件 + 提取文本并排查看
  - 候选人 Chat Agent（按候选人维度 session 会话）

### 2) JD 匹配

- 输入 JD 后先抽取结构化需求（`JDRequirements`）
- 向量召回候选人，再进行逐人匹配分析
- 输出可解释结果：
  - `total_match_score`
  - `skill_match_score`
  - `experience_match_score`
  - `project_relevance_score`
  - `domain_relevance_score`
  - `matched_evidence`
  - `missing_or_weak_evidence`
  - `strengths`
  - `concerns`
  - `interview_questions`
- 匹配参数放在页面里可调（非 `.env`）：
  - 向量召回 `top-k`
  - 参与分析人数 `top-n`
- JD 匹配页支持结果追问 Chat Agent（基于“本次 JD + 本次排序结果”）

---

## 候选人画像字段（核心）

除基础信息外，已支持以下关键字段落库并展示：

- `phone`（手机号）
- `wechat`（微信号）
- `location`（地理地点）
- `job_intent`（求职意向）
- `birth_year`（出生年份）
- `latest_graduation_date`（最近毕业时间）
- `highest_education`（最高学历）
- `years_of_experience`（经验年限）
- `summary`、`risk_flags`、`skills`
- `raw_text`、`source_file`

---

## 技术栈

- 前端：Streamlit（`app.py` + `pages/`）
- LLM 编排：LangChain（`chains/`）
- 数据校验：Pydantic（`schemas/`）
- 结构化存储：SQLite（`data/app.db`）
- 向量检索：Chroma（`data/chroma/`）
- 文档解析：`pypdf`、`python-docx`、`pytesseract`（可选）

---

## 快速开始

**环境**：Python 3.11+（推荐）

```bash
cd /path/to/ResumeParsingAgent
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# 编辑 .env 填入 OPENAI_API_KEY
streamlit run app.py
```

---

## .env 配置项

`.env.example` 当前主要保留：

- `OPENAI_API_KEY`（必填）
- `OPENAI_MODEL`（可选）
- `OPENAI_EMBEDDING_MODEL`（可选）
- `OPENAI_VISION_FALLBACK`（可选，默认 `true`）

---

## 测试数据

已内置 20 份跨领域测试简历（中文 TXT）：

- 路径：`data/sample_resumes/`
- 覆盖技术、产品、销售、财务、法务、医疗、教育、供应链、土木、建筑、应届生等背景

可直接在人才库页面批量上传做演示。

---

## 目录说明

| 路径 | 职责 |
|------|------|
| `app.py` | Streamlit 入口 + 导航配置 |
| `pages/` | `人才库` / `JD匹配` 页面 |
| `schemas/` | Pydantic 模型：画像/JD/匹配结果 |
| `chains/` | LangChain 链：简历解析/JD抽取/匹配分析/视觉兜底 |
| `services/` | 业务编排：入库、匹配、候选人问答、JD问答 |
| `repositories/` | SQLite 与 Chroma 访问 |
| `prompts/` | 提示词模板 |
| `utils/` | 配置、导航 bootstrap、UI 工具 |
| `data/` | 上传副本、DB、向量库、样本简历 |
| `tests/` | 基础 smoke tests |

---

## 清空本地数据

删除以下目录/文件可重置：

- `data/app.db`
- `data/chroma/`
- `data/uploads/`

请先自行备份。

---
