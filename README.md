# 医学教育测验系统

一个 AI 驱动的医学教育平台，从临床指南自动生成选择题，并提供基于 ChatGPT 的交互式辅导，支持 RAG（检索增强生成）。

## 功能特性

### 🎯 题目生成
- 从 40 个医学指南 PDF 自动生成题目
- 每个选项都有解释（为什么正确/错误）
- 来源追踪（PDF 名称 + 页码）
- 使用 OpenAI GPT-4o-mini 生成高质量题目

### 💬 ChatGPT 集成
- 对任何主题提出后续问题
- 完整的上下文感知（题目、答案、解释）
- 医学导师角色，采用苏格拉底式教学法
- 来自指南的来源引用

### 🔍 RAG（检索增强生成）
- 两阶段检索：FAISS → ColBERTv2 重排序
- 对所有医学指南进行语义搜索
- 自动来源引用，包含页码
- 服务器不可用时回退到关键词搜索

## 项目结构

```
LLM_Agent_for_Education/
├── medical-quiz.html      # 主 UI（单页应用）
├── questions.json         # 生成的题目（200 道）
├── generate_questions.py  # 题目生成脚本
├── rag_server.py          # RAG 后端（FAISS + ColBERTv2）
├── api-key.js             # 你的 OpenAI API 密钥（不在 Git 中）
├── api-key.example.js     # API 密钥模板
├── README.md              # 本文件
└── Clinical Guidelines/   # 40 个医学 PDF 文件
    ├── Guidelines for AAA repair.pdf
    ├── Management of Acute Appendicitis.pdf
    └── ...（还有 38 个 PDF）
```

## 快速开始

### 步骤 1：配置 API 密钥

1. 复制 `api-key.example.js` 为 `api-key.js`：
   ```bash
   copy api-key.example.js api-key.js
   ```

2. 编辑 `api-key.js`，添加你的 OpenAI API 密钥：
   ```javascript
   const OPENAI_API_KEY = 'sk-你的实际密钥';
   ```

3. 获取 API 密钥：https://platform.openai.com/account/api-keys

### 步骤 2：启动 UI

**方式 A：简单 HTTP 服务器（推荐）**
```bash
# 启动服务器
python -m http.server 8000

# 在浏览器中打开
# http://localhost:8000/medical-quiz.html
```

**方式 B：带 RAG 服务器（增强版）**
```bash
# 终端 1：启动 RAG 服务器（首次运行需要 5-10 分钟构建索引）
python rag_server.py

# 终端 2：启动 HTTP 服务器
python -m http.server 8000

# 打开：http://localhost:8000/medical-quiz.html
```

## 使用指南

### 做测验

1. 阅读题目并选择答案（A、B、C 或 D）
2. 点击 "Submit" 查看反馈
3. 查看每个选项的解释
4. 使用 上一题/下一题 导航

### 向 ChatGPT 提问

1. 提交答案后，在聊天框中输入问题
2. 示例：
   - "为什么 B 是错误的？"
   - "解释一下病理生理学"
   - "鉴别诊断有哪些？"
3. ChatGPT 会返回带引用的回答

### 重新生成题目

从 PDF 生成新题目：

```bash
python generate_questions.py
```

配置（在 `generate_questions.py` 中）：
- `questions_per_doc = 5` - 每个 PDF 的题目数
- `model = "gpt-4o-mini"` - OpenAI 模型
- `chunk_size = 512` - 每块的 token 数

## 技术细节

### 题目生成流程

```
PDF 文件 → PyPDFLoader → 分割成块
    → 随机选择（每个文档 5 个）
    → GPT-4o-mini 生成题目
    → 保存到 questions.json
```

### RAG 流程

```
学生问题 → OpenAI 向量
    → FAISS（前 30 个候选）
    → ColBERTv2 重排序（前 5 个）
    → 添加到 ChatGPT 上下文
```

### 数据格式（questions.json）

```json
{
  "question": "...的一线治疗是什么？",
  "options": {
    "A": "选项文本",
    "B": "选项文本",
    "C": "选项文本",
    "D": "选项文本"
  },
  "correct_answer": "B",
  "explanations": {
    "A": "错误。原因...",
    "B": "正确。原因...",
    "C": "错误。原因...",
    "D": "错误。原因..."
  },
  "source": "Guidelines for AAA repair.pdf",
  "source_page": 12,
  "source_chunk": "来自 PDF 的原始文本..."
}
```

## 依赖项

### Python 依赖

```bash
pip install openai langchain-community pypdf flask flask-cors faiss-cpu ragatouille transformers
```

### 使用的模型

| 组件 | 模型 | 用途 |
|------|------|------|
| 题目生成 | GPT-4o-mini | 生成选择题 |
| 向量 | text-embedding-3-small | 语义搜索 |
| 重排序 | ColBERTv2 | 提高相关性 |

### 预估费用

| 任务 | 大约费用 |
|------|----------|
| 生成 200 道题目 | $0.10 - $0.30 |
| 构建 RAG 索引 | $0.05 - $0.10 |
| ChatGPT 对话 | 每条消息 $0.001 |

## 故障排除

### "Cannot load questions.json"
- 确保你运行了本地服务器
- 不要直接打开 HTML 文件（file:// 协议）
- 运行：`python -m http.server 8000`

### "API key not configured"
- 从模板创建 `api-key.js`
- 或在浏览器控制台中输入 API 密钥：
  ```javascript
  localStorage.setItem('openai_api_key', 'sk-你的密钥');
  ```

### "RAG Server not available"
- 如果 `rag_server.py` 没运行，这是正常的
- 系统会回退到关键词搜索
- 为获得更好的结果，请运行 RAG 服务器

### 终端显示乱码
- 这是 PowerShell 编码问题
- 实际输出文件（questions.json）是正确的
- 使用 Windows Terminal 或 VS Code 终端可以获得更好的显示

## 安全说明

- `api-key.js` 在 `.gitignore` 中 - 你的密钥不会被上传
- 切勿分享你的 API 密钥
- 在 OpenAI 后台设置使用限额
- 密钥存储在浏览器的 localStorage 中供 UI 使用

## 许可证

MIT 许可证 - 仅用于教育目的

## 作者

AI for Education (Wenchao Qin)

