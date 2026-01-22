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
├── parse_qbank_openai.py  # Amboss PDF 解析脚本
├── merge_qbanks.py        # 合并题库脚本
├── rag_server.py          # RAG 后端（FAISS + ColBERTv2）
├── start.py              # 统一启动脚本（推荐）
├── start.bat              # Windows 快捷启动
├── api-key.js             # 你的 OpenAI API 密钥（不在 Git 中）
├── api-key.example.js     # API 密钥模板
├── README.md              # 本文件
├── Clinical Guidelines/   # 40 个医学 PDF 文件
└── Qbanks and Practice Exams/  # 题库 PDF 文件
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

### 步骤 2：安装依赖

```bash
pip install openai langchain-community pypdf flask flask-cors faiss-cpu ragatouille transformers pymupdf
```

### 步骤 3：启动服务器和 UI

#### 方式 A：统一启动脚本（推荐）

**Windows:**
```bash
python start.py
```
或双击 `start.bat`

**Linux/WSL:**
```bash
python3 start.py
```

**启动选项：**

```bash
# 启动所有服务（默认：主UI + RAG服务器）
python start.py

# 只启动主UI和RAG服务器
python start.py --ui --rag

# 只启动评估界面
python start.py --evaluator

# 自定义端口
python start.py --ui --port 8000 --evaluator --evaluator-port 8002

# 重启RAG服务器（停止现有进程后重新启动）
python start.py --restart-rag
```

**启动后：**
- 主UI：http://localhost:8000/medical-quiz.html
- RAG服务器：http://localhost:5000/health
- 评估界面：http://localhost:8001/question_evaluator.html（如果启动）

**注意：**
- Windows 环境可以启动所有服务，但 ColBERTv2 重排序器可能不可用
- WSL/Linux 环境支持完整功能，包括 ColBERTv2 重排序器
- 首次运行需要 5-10 分钟构建 RAG 索引（仅一次）
- 按 `Ctrl+C` 停止所有服务器

#### 方式 B：分别启动（适合调试）

**Windows 环境（仅 FAISS，无 ColBERTv2 重排序）：**

```bash
# 终端 1：启动 RAG 服务器（仅 FAISS 模式）
python rag_server.py
# 服务器启动在：http://localhost:5000

# 终端 2：启动 HTTP 服务器
python -m http.server 8000
# 服务器启动在：http://localhost:8000

# 在浏览器中打开
# http://localhost:8000/medical-quiz.html
```

**WSL/Linux 环境（完整功能，包含 ColBERTv2）：**

```bash
# 终端 1：启动 RAG 服务器（包含 ColBERTv2 重排序）
python3 rag_server.py
# 服务器启动在：http://localhost:5000

# 终端 2：启动 HTTP 服务器
python3 -m http.server 8000
# 服务器启动在：http://localhost:8000

# 在浏览器中打开
# http://localhost:8000/medical-quiz.html
```

#### 服务说明

| 服务 | 端口 | 说明 | 访问地址 | 必需性 |
|------|------|------|----------|--------|
| 主UI | 8000 | 医学测验系统主界面 | http://localhost:8000/medical-quiz.html | ✅ 必需 |
| RAG服务器 | 5000 | 语义搜索API | http://localhost:5000/health | ⚠️ 可选（推荐） |
| 评估界面 | 8001 | 问题质量评估工具 | http://localhost:8001/question_evaluator.html | ⚠️ 可选 |

**RAG 服务器不可用时：**
- UI 仍可正常使用
- ChatGPT 功能仍可用，但会回退到关键词搜索
- 语义搜索功能不可用

**验证服务器状态：**
```bash
# 检查 RAG 服务器
curl http://localhost:5000/health

# 检查 HTTP 服务器
curl http://localhost:8000/medical-quiz.html
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

### 解析题库 PDF

从 Amboss PDF 解析题目：
```bash
python parse_qbank_openai.py --dir "Qbanks and Practice Exams" --output qbank_amboss_openai.json
```

参数说明：
- `--dir`: PDF 文件目录
- `--output`: 输出 JSON 文件路径
- `--model`: OpenAI 模型（默认：gpt-5）
- `--batch-size`: 每批处理的页数（默认：3）
- `--overlap`: 批次重叠页数（默认：2）

### 合并题库

合并多个题库文件：
```bash
python merge_qbanks.py
```

### 重新生成题目

从 PDF 生成新题目：
```bash
python generate_questions.py
```

配置（在 `generate_questions.py` 中）：
- `questions_per_doc = 5` - 每个 PDF 的题目数
- `model = "gpt-4o-mini"` - OpenAI 模型
- `chunk_size = 1024` - 每块的 token 数

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
pip install openai langchain-community pypdf flask flask-cors faiss-cpu ragatouille transformers pymupdf
```

### 使用的模型

| 组件 | 模型 | 用途 |
|------|------|------|
| 题目生成 | GPT-4o-mini | 生成选择题 |
| PDF 解析 | GPT-5 / GPT-4o-2024-11-20 | 解析 Amboss PDF |
| 向量 | text-embedding-3-small | 语义搜索 |
| 重排序 | ColBERTv2 | 提高相关性 |

### 预估费用

| 任务 | 大约费用 |
|------|----------|
| 生成 200 道题目 | $0.10 - $0.30 |
| 解析 Amboss PDF | $0.50 - $2.00 |
| 构建 RAG 索引 | $0.05 - $0.10 |
| ChatGPT 对话 | 每条消息 $0.001 |

## 高级配置

### 安装 ColBERTv2 重排序器（可选）

ColBERTv2 重排序器需要编译 C++ 扩展。有两种方式：

#### 方案 1：安装 Visual Studio Build Tools（Windows）

1. **下载 Visual Studio Build Tools**
   - 直接下载：https://aka.ms/vs/17/release/vs_buildtools.exe
   - 或访问：https://visualstudio.microsoft.com/downloads/
   - 选择 "Build Tools for Visual Studio 2022"

2. **安装时选择组件**
   - 运行安装程序
   - 在 "工作负载" 标签页中，勾选：
     - ✅ **C++ build tools** 工作负载
   - 在右侧 "安装详细信息" 中，确保包含：
     - ✅ MSVC v143 - VS 2022 C++ x64/x86 build tools
     - ✅ Windows 10/11 SDK（最新版本）
     - ✅ C++ CMake tools for Windows
   - 点击 "安装"（需要约 3-5 GB 空间）

3. **安装完成后**
   - 关闭所有终端窗口
   - 重新打开终端
   - 运行：`python start.py`

4. **验证安装**
   ```powershell
   where cl
   ```
   如果显示编译器路径，说明安装成功。

#### 方案 2：使用 WSL（推荐，更稳定）

ColBERTv2 的 C++ 扩展使用了 POSIX 线程库（pthread.h），在 Windows 上可能不兼容。使用 WSL 可以避免这个问题。

**步骤 1：安装 WSL**

在 PowerShell（管理员权限）中运行：
```powershell
wsl --install
```

或安装特定版本：
```powershell
wsl --install -d Ubuntu-22.04
```

**重要**：安装完成后需要**重启电脑**。

**步骤 2：在 WSL 中设置环境**

1. 打开 WSL（Ubuntu）
2. 更新系统并安装依赖：
   ```bash
   sudo apt update
   sudo apt upgrade -y
   sudo apt install -y python3 python3-pip python3-venv build-essential git
   ```

3. 升级 pip：
   ```bash
   python3 -m pip install --upgrade pip setuptools wheel
   ```

4. 安装 PyTorch：
   ```bash
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

5. 安装项目依赖（分批安装）：
   ```bash
   pip3 install faiss-cpu
   pip3 install flask flask-cors
   pip3 install openai
   pip3 install langchain-community
   pip3 install pypdf
   pip3 install pymupdf
   pip3 install ragatouille
   ```

**或者使用虚拟环境（推荐）**：
```bash
# 创建虚拟环境
cd /mnt/d/LLM_Agent_for_Education
python3 -m venv venv_wsl
source venv_wsl/bin/activate

# 安装依赖
pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install faiss-cpu flask flask-cors openai langchain-community pypdf pymupdf ragatouille
```

**步骤 3：在 WSL 中运行服务器**

```bash
cd /mnt/d/LLM_Agent_for_Education
python3 start.py
```

**注意事项**：
- Windows 驱动器在 WSL 中位于 `/mnt/d/`、`/mnt/c/` 等
- WSL 和 Windows 共享 localhost，可以直接访问
- 文件路径需要使用 `/mnt/d/` 前缀访问 Windows 驱动器

**在 WSL 中使用统一启动脚本：**
```bash
cd /mnt/d/LLM_Agent_for_Education
python3 start.py
```

#### 方案 3：使用仅 FAISS 模式（当前状态）

如果不想安装编译器或 WSL，系统已经可以正常工作：
- ✓ FAISS 检索正常工作
- ✓ 系统功能完整
- ✗ 仅缺少 ColBERTv2 重排序优化

**注意**：FAISS 检索已经足够使用，重排序只是优化项。

### 重建 RAG 索引

如果修改了 chunk 大小或其他配置，需要重建索引。

#### 方法 1：通过 API 重建（推荐，无需重启服务器）

在终端中运行：
```bash
curl -X POST http://localhost:5000/rebuild
```

**优点**：
- 无需停止服务器
- 后台重建，可以继续使用系统

**等待时间**：约 5-10 分钟

#### 方法 2：重启服务器重建（完全重建）

1. **停止当前服务器**
   在运行 `python start.py` 的终端中按 `Ctrl+C`

2. **删除旧索引文件**
   ```bash
   rm -f faiss_index.bin all_chunks.json
   ```

3. **重新启动服务器**
   ```bash
   python start.py
   ```

服务器会自动检测到索引文件不存在，然后重建索引。

**重建过程说明**：
- 重新加载所有 PDF 文件（从 `Clinical Guidelines` 目录）
- 使用新的 chunk 大小（1024 tokens）分割文档
- 为每个块生成向量（通过 OpenAI API）
- 构建 FAISS 索引
- 保存索引文件

**费用**：约 $0.05-0.10（OpenAI Embeddings API）

**验证重建成功**：
```bash
curl http://localhost:5000/health
```

应该看到：
- `chunks_count`: 新的块数量
- `reranker`: ColBERTv2（如果已安装）

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

### "重排序器导入失败"
- 如果看到 `[!] 重排序器导入失败`，这是正常的
- 系统会使用仅 FAISS 模式
- 要启用重排序，请参考"安装 ColBERTv2 重排序器"章节

### 终端显示乱码
- 这是 PowerShell 编码问题
- 实际输出文件（questions.json）是正确的
- 使用 Windows Terminal 或 VS Code 终端可以获得更好的显示

### WSL 中 pip 安装失败

如果遇到 `pip` 安装错误：

1. **升级 pip**：
   ```bash
   python3 -m pip install --upgrade pip
   ```

2. **清除缓存**：
   ```bash
   pip cache purge
   ```

3. **使用虚拟环境**（推荐）：
   ```bash
   python3 -m venv venv_wsl
   source venv_wsl/bin/activate
   pip install ...
   ```

4. **分批安装**：
   ```bash
   pip3 install faiss-cpu
   pip3 install flask flask-cors
   pip3 install openai
   pip3 install langchain-community
   pip3 install pypdf
   pip3 install ragatouille
   ```

### 编译 ragatouille 很慢
- 这是正常的，首次编译可能需要 5-10 分钟
- 请耐心等待

## 安全说明

- `api-key.js` 在 `.gitignore` 中 - 你的密钥不会被上传
- 切勿分享你的 API 密钥
- 在 OpenAI 后台设置使用限额
- 密钥存储在浏览器的 localStorage 中供 UI 使用

## 许可证

MIT 许可证 - 仅用于教育目的

## 作者

AI for Education (Wenchao Qin)
