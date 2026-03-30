# Resume Parsing Agent（简历解析智能体）

一个面向 HR 场景的 AI 招聘助手，用于解决**简历录入、候选人筛选、JD 匹配与面试准备自动化**问题。

该项目重点在于：
> **用 AI 重构 HR 的“简历处理 + 初筛 + 面试准备”工作流，而非简单做文本解析。**

---

# 快速开始

访问 https://resume-agent.ylceric.com/ 即可直接在线体验。（⚠️注意：网页服务未设置用户鉴权，仅用于展示与测试，请勿上传敏感信息。） 

或本地部署： 
```bash
cd /path/to/ResumeParsingAgent
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# 填入 OPENAI_API_KEY
streamlit run app.py
```
之后在浏览器中访问 http://localhost:8501 即可看到项目首页。

在左侧侧边栏中可以切换不同功能页面。

---

# 一、项目核心价值（Why this project）

在实际招聘流程中，HR 常见痛点包括：

- 简历来源复杂（PDF / Word / 图片 / 扫描件）
- 简历格式不统一，难以横向比较
- JD 与候选人匹配耗时，尤其技术岗位
- 初筛评语、优劣势总结、面试问题高度重复
- 很难快速建立“候选人画像”

本项目的目标是：

**将非结构化简历 → 标准化候选人画像 → 可检索人才库 → JD 匹配 → 面试辅助**

---

# 二、系统整体流程（核心设计）
批量上传简历
→ 文本提取 / OCR / 多模态解析
→ LLM 结构化解析（Candidate Profile）
→ 存入 SQLite + Chroma
→ 人才库检索（关键词 / 语义）
→ 输入 JD
→ 向量召回 Top-K 候选人
→ LLM 做可解释匹配分析
→ 输出 strengths / concerns / interview questions
→ 支持对话式追问（Chat Agent）


---

# 三、核心功能（MVP）

## 1）人才录入 / 人才库

支持 HR 将多来源简历统一入库，并生成标准化候选人画像：

- 批量上传：
  - PDF / DOCX / TXT
  - 图片简历（PNG / JPG / JPEG）
- 文档解析策略：
  - 文本优先解析
  - 图片优先 OCR（`pytesseract`）
  - OCR 不足时支持多模态兜底（可选）
- LLM 输出结构化候选人画像（Pydantic 校验）
- 数据写入：
  - SQLite（结构化信息）
  - Chroma（向量检索）
- 人才库能力：
  - 关键词搜索（结构化字段）
  - 自然语言搜索（语义检索）
- 候选人详情页：
  - 原始文件 + 提取文本对比
  - 候选人专属 Chat Agent（支持追问）

---

## 2）JD 匹配（核心亮点）

输入 JD 后，系统会：

### Step 1：结构化 JD
将 JD 转换为结构化需求（技能 / 年限 /领域等）

### Step 2：候选人召回
通过向量检索从人才库召回 Top-K 候选人

### Step 3：可解释匹配分析（LLM）
对每个候选人输出：

- `total_match_score`
- 多维匹配评分：
  - skill / experience / project / domain
- `matched_evidence`
- `missing_or_weak_evidence`
- `strengths`
- `concerns`
- `interview_questions`

**不是黑盒评分，而是“可解释匹配”**

---

## 3）对话式分析（Agent）

系统支持两类对话：

### 候选人维度 Chat
- 针对单个候选人提问
- 基于该候选人简历内容回答

### JD 匹配结果 Chat
- 针对本次 JD + 排序结果进行追问
- 例如：
  - 为什么这个人排第一？
  - 哪些候选人更适合 backend？

---

# 四、核心数据模型（Candidate Profile）

系统核心是“标准化候选人画像”，而非原始简历。

关键字段包括：

- 基本信息：
  - phone / wechat / location
- 教育与背景：
  - highest_education / graduation_date
- 职业信息：
  - years_of_experience
  - job_intent
- 内容摘要：
  - summary
  - skills
  - risk_flags
- 原始数据：
  - raw_text
  - source_file

👉 该结构用于：
- 展示
- 检索
- 匹配分析

---

# 五、RAG 设计（关键工程点）

本项目采用**轻量 RAG 架构**：

## 双存储设计

### 1）SQLite（结构化）
用于：
- 候选人信息管理
- 表格展示
- 精确过滤

### 2）Chroma（向量库）
用于：
- 语义检索
- JD 候选人召回

向量数据包括：
- summary
- 项目描述
- 技能证据片段
- 简历分块文本

👉 每条向量数据带 `candidate_id`，用于回查 SQLite

---

# 六、技术栈

- 前端：Streamlit
- LLM 编排：LangChain
- 数据校验：Pydantic
- 结构化存储：SQLite
- 向量数据库：Chroma
- 文档解析：
  - pypdf
  - python-docx
  - pytesseract（OCR）

---

# 七、工程设计原则

本项目采用：

- ✅ LangChain 做 LLM workflow 封装
- ✅ SQLite + Chroma 本地部署
- ✅ 模块化 service / repository 架构
- ✅ 先打通主链路，再做增强

---

# 八、项目结构

| 路径 | 职责 |
|------|------|
| `app.py` | Streamlit 入口 |
| `pages/` | 人才库 / JD 匹配 页面 |
| `schemas/` | 数据模型（Pydantic） |
| `chains/` | LangChain 封装 |
| `services/` | 业务逻辑 |
| `repositories/` | SQLite + Chroma |
| `prompts/` | Prompt 模板 |
| `utils/` | 工具函数 |
| `data/` | DB / 向量库 / 上传文件 |

---

# 九、测试数据

内置 20 份跨领域中文简历：

技术 / 产品 / 销售 / 财务 / 法务 / 医疗 / 教育 / 应届生 等 

路径：`data/sample_resumes/`

---

# 十、开发方式说明
设计阶段：使用 LLM 辅助系统架构与流程设计 

实现阶段：使用 Cursor 进行代码生成与重构 

优先保证：
- 主链路完整
- 模块边界清晰
- 可快速 demo

---

# 十一、未来优化方向
- 更精细的匹配评分模型（规则 + Multi Agent Debate）
- 更强的多模态简历解析
- 候选人去重 / 相似度检测
- 批量 JD 匹配报告导出

---