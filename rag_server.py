"""
================================================================================
RAG 服务器 - 医学教育知识库
================================================================================

概述：
    基于 Flask 的 REST API 服务器，为医学测验系统提供检索增强生成（RAG）能力。
    支持对医学指南 PDF 进行语义搜索，为学生问题提供相关上下文。

架构：
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                        两阶段 RAG 管道                                    │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │   学生问题                                                                │
    │        │                                                                 │
    │        ▼                                                                 │
    │   ┌─────────────────────┐                                               │
    │   │  OpenAI Embeddings  │  将问题转换为向量                               │
    │   │  (text-embedding-3) │                                               │
    │   └─────────────────────┘                                               │
    │        │                                                                 │
    │        ▼                                                                 │
    │   ┌─────────────────────┐                                               │
    │   │  第一阶段：FAISS    │  快速近似最近邻搜索                             │
    │   │  返回 30 个候选     │  使用余弦相似度                                 │
    │   └─────────────────────┘                                               │
    │        │                                                                 │
    │        ▼                                                                 │
    │   ┌─────────────────────┐                                               │
    │   │  第二阶段：ColBERTv2│  token 级别重排序                              │
    │   │  返回前 5 个结果    │  使用 Late Interaction 机制                    │
    │   └─────────────────────┘                                               │
    │        │                                                                 │
    │        ▼                                                                 │
    │   最终结果（包含源 PDF + 页码）                                           │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

功能：
    - 加载并索引 40 个医学指南 PDF
    - 将文档分割成约 512 token 的块
    - 使用 OpenAI text-embedding-3-small 生成句子嵌入（sentence embeddings）
    - 构建 FAISS 索引用于快速向量相似度搜索
    - 使用 ColBERTv2 重排序提高相关性
    - REST API 支持搜索、健康检查和索引重建

检索流程：
    1. 知识库嵌入阶段：
       - 所有文档块通过 OpenAI Embeddings API 转换为向量
       - 每个块生成一个 1536 维的稠密向量表示
       - 向量捕捉语义含义，相似的文本会有相似的向量
    
    2. 向量存储：
       - 所有块向量存储在 FAISS 向量数据库中
       - 向量被归一化（L2 归一化）以支持余弦相似度
    
    3. 查询检索：
       - 用户查询通过同一模型转换为查询向量
       - 查询向量也被归一化
       - 通过相似度搜索从向量数据库返回最接近的文档
    
    4. 技术挑战：
       - 给定查询向量，快速找到最近邻向量
       - 需要选择：距离度量和搜索算法

距离度量选择：
    我们选择余弦相似度（Cosine Similarity），原因：
    - 计算两个向量之间相对角度的余弦值
    - 比较向量的方向，不考虑向量大小
    - 与我们的嵌入模型配合良好
    - 使用归一化向量时，余弦相似度 = 内积（点积）
    
    其他距离度量：
    - 点积（Dot Product）：考虑向量大小，但增加向量长度会使其更相似
    - 欧氏距离（Euclidean Distance）：向量两端点之间的距离
    - 归一化后，这些距离度量在数学上等价

搜索算法选择：
    我们选择 Facebook 的 FAISS（Facebook AI Similarity Search）：
    - 对于大多数用例性能足够好
    - 广为人知，得到广泛应用
    - 支持精确搜索（IndexFlatIP）和近似搜索（IVF, HNSW）
    - 我们使用 IndexFlatIP（内积）在归一化向量上实现余弦相似度

依赖：
    pip install flask flask-cors openai faiss-cpu langchain-community pypdf ragatouille
    （仅使用 ColBERTv2，无其他后备重排序器）

使用方法：
    1. 在 api-key.js 中设置 API 密钥
    2. 运行：python rag_server.py
    3. 服务器启动在 http://localhost:5000
    4. API 端点：
       - POST /search     - 搜索相关文本块
       - GET  /health     - 健康检查
       - POST /rebuild    - 重建索引

作者：AI for Education (Wenchao Qin)
================================================================================
"""

# =============================================================================
# 导入模块
# =============================================================================

from collections import defaultdict
import json
import os
from pathlib import Path
import re
from threading import Lock
import time
import types
from typing import Dict, List, Optional

from flask import Flask, jsonify, request, send_from_directory, redirect
from flask_cors import CORS
import langchain
import numpy as np
from openai import OpenAI

# -----------------------------------------------------------------------------
# LangChain - 文档加载和处理
# -----------------------------------------------------------------------------
# LangChain 提供加载 PDF 和分割文档的工具
try:
    from langchain_community.document_loaders import PyPDFLoader
except ImportError:
    print("正在安装 langchain-community...")
    import subprocess
    subprocess.check_call(["pip", "install", "langchain-community", "pypdf"])
    from langchain_community.document_loaders import PyPDFLoader

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

# -----------------------------------------------------------------------------
# FAISS - Facebook AI 相似度搜索
# -----------------------------------------------------------------------------
# FAISS 是一个用于高效向量相似度搜索的库
# 我们使用 IndexFlatIP（内积）在归一化向量上实现余弦相似度
try:
    import faiss
except ImportError:
    print("正在安装 faiss-cpu...")
    import subprocess
    subprocess.check_call(["pip", "install", "faiss-cpu"])
    import faiss

# -----------------------------------------------------------------------------
# RAGatouille - ColBERTv2 重排序器
# -----------------------------------------------------------------------------
# ColBERTv2 使用 Late Interaction 机制进行更精确的重排序
# 它在 token 级别比较查询和文档的向量

# 修复 langchain.retrievers 兼容性问题
# 必须在导入 ragatouille 之前创建兼容层


# 检查并创建 langchain.retrievers 兼容层
if not hasattr(langchain, 'retrievers'):
    langchain.retrievers = types.ModuleType('retrievers')
    # 创建 document_compressors 子模块
    langchain.retrievers.document_compressors = types.ModuleType('document_compressors')
    # 创建 base 子模块
    langchain.retrievers.document_compressors.base = types.ModuleType('base')
    # 创建一个虚拟的 BaseDocumentCompressor 类
    class BaseDocumentCompressor:
        pass
    langchain.retrievers.document_compressors.base.BaseDocumentCompressor = BaseDocumentCompressor
    # 确保模块已注册到 sys.modules
    import sys
    sys.modules['langchain.retrievers'] = langchain.retrievers
    sys.modules['langchain.retrievers.document_compressors'] = langchain.retrievers.document_compressors
    sys.modules['langchain.retrievers.document_compressors.base'] = langchain.retrievers.document_compressors.base

# 尝试导入重排序器
HAS_RERANKER = False
try:
    from ragatouille import RAGPretrainedModel
    HAS_RERANKER = True
except (ImportError, ModuleNotFoundError) as e:
    HAS_RERANKER = False
    print(f"[!] 重排序器导入失败: {e}")


# =============================================================================
# 配置参数
# =============================================================================

# 包含医学指南 PDF 的目录
PDF_DIRECTORY = "Clinical Guidelines"

# OpenAI 向量模型
# text-embedding-3-small：质量和成本的良好平衡
# text-embedding-3-large：更高质量但更贵
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536  # text-embedding-3-small 的向量维度

# 文档分块大小
# 对于医学指南，建议使用更大的 chunk 以保持上下文完整性
# 1024 token：平衡检索精度和上下文完整性（推荐）
# 1536 token：更多上下文，适合复杂医学概念
# 较小的块 = 更精确的检索，较大的块 = 更多上下文
CHUNK_SIZE = 1024  # 从 512 增加到 1024，提供更完整的医学上下文
CHUNK_OVERLAP = 100  # 块之间的重叠，保持上下文连贯性（从 50 增加到 100）

# 索引文件保存路径
INDEX_FILE = "data/indexes/faiss_index.bin"    # FAISS 向量索引
CHUNKS_FILE = "data/indexes/all_chunks.json"   # 所有块的元数据（文本、来源、页码）

# -----------------------------------------------------------------------------
# RAG 检索配置
# -----------------------------------------------------------------------------
# 两阶段检索在保持速度的同时提高准确性
NUM_RETRIEVED_DOCS = 30  # 第一阶段：FAISS 检索 30 个候选（快速）
NUM_DOCS_FINAL = 5       # 第二阶段：重排序器选出前 5 个（准确）

# =============================================================================
# Flask 应用设置
# =============================================================================

app = Flask(__name__)

# 启用 CORS 允许来自前端（medical-quiz.html）的请求
# 这是必要的，因为前端运行在不同的端口（8000）
# 使用简单配置，允许所有来源、方法和头部
# expose_headers 让前端能读取 X-Cost-Limit-Exceeded（超限即时提示）
CORS(app, supports_credentials=True, expose_headers=["X-Cost-Limit-Exceeded"])


@app.errorhandler(500)
def handle_500(e):
    """确保 500 错误始终返回 JSON，便于前端显示具体错误信息。"""
    import traceback
    traceback.print_exc()
    err = str(e) if e else "Internal server error"
    return jsonify({"success": False, "error": err}), 500

# =============================================================================
# 全局状态
# =============================================================================
# 这些在启动时加载一次，在请求之间共享

faiss_index = None      # FAISS 向量索引，用于相似度搜索
all_chunks = []         # 所有文档块及其元数据的列表
openai_client = None    # OpenAI API 客户端，用于生成向量
reranker = None         # ColBERTv2 重排序模型


# =============================================================================
# 辅助函数
# =============================================================================

def get_api_key() -> str:
    """
    从环境变量或 api-key.js 文件获取 OpenAI API 密钥。
    
    优先级：
        1. 环境变量：OPENAI_API_KEY
        2. 本地文件：api-key.js（用于开发）
    
    返回：
        str：API 密钥，如果未找到则返回空字符串
    
    api-key.js 格式示例：
        const OPENAI_API_KEY = 'sk-your-api-key-here';
    """
    # 首先尝试环境变量
    api_key = os.environ.get('OPENAI_API_KEY')
    
    # 如果环境变量中没有，尝试 api-key.js 文件
    if not api_key:
        config_path = Path(__file__).parent / 'api-key.js'
        if config_path.exists():
            try:
                content = config_path.read_text(encoding='utf-8')
                # 使用正则表达式提取 API 密钥
                # 匹配：OPENAI_API_KEY = 'key' 或 OPENAI_API_KEY: 'key'
                match = re.search(r"OPENAI_API_KEY\s*[=:]\s*['\"]([^'\"]+)['\"]", content)
                if match:
                    api_key = match.group(1)
                    print("[OK] 已从 api-key.js 加载 API 密钥")
            except Exception as e:
                print(f"[!] 读取 api-key.js 出错：{e}")
    
    return api_key


# =============================================================================
# Per-user cost limit: 三分钟五毛钱 ($0.5 per 3 min per user, rolling window)
# =============================================================================

COST_LIMIT_WINDOW_SECONDS = 180   # 3 分钟
COST_LIMIT = 0.5                  # 五毛钱 / $0.5 per user per 3 min
_user_costs: Dict[str, List[tuple]] = defaultdict(list)  # 用户ID -> [(时间戳, 费用), ...]，用于限额检查
_user_tokens: Dict[str, List[tuple]] = defaultdict(list)  # 用户ID -> [(时间戳, pt, ct), ...]，用于限额
_user_totals: Dict[str, Dict] = defaultdict(lambda: {"cost": 0.0, "prompt_tokens": 0, "completion_tokens": 0, "request_count": 0})  # 累计总量（不清理）
_cost_lock = Lock()  # 线程锁，保证并发安全

# OpenAI 定价（美元/百万token），近似值
_PRICE = {
    "gpt-4o-mini": (0.15, 0.60),       # (输入, 输出)
    "gpt-4o": (2.50, 10.0),
    "gpt-4o-2024": (2.50, 10.0),
    "text-embedding-3-small": (0.02, 0.02),
}


def _get_user_id() -> str:
    """
    从请求中获取用户标识，用于按用户统计费用、执行每小时限额。
    优先级：X-User-Id（登录用户名）> X-Session-Id（会话）> IP
    Cost limit is per logged-in user; same username across devices/tabs shares the limit.
    """
    if request:
        uid = request.headers.get("X-User-Id")
        if uid and uid.strip():
            return ("user:" + uid.strip())[:64]
        sid = request.headers.get("X-Session-Id")
        if sid and sid.strip():
            return ("sess:" + sid.strip())[:64]
        addr = request.remote_addr or "unknown"
        return f"ip:{addr}"
    return "unknown"


def _estimate_chat_cost(prompt_tokens: int, completion_tokens: int, model: str) -> float:
    """根据 token 数量估算 chat 费用（美元）。"""
    prices = _PRICE.get(model, _PRICE["gpt-4o-mini"])
    inp, out = prices
    return (prompt_tokens * inp + completion_tokens * out) / 1_000_000


def _estimate_embedding_cost(total_tokens: int) -> float:
    """估算 embedding 费用（美元）。"""
    p = _PRICE["text-embedding-3-small"][0]
    return total_tokens * p / 1_000_000


def cost_add(user_id: str, cost: float) -> None:
    """记录用户当前时刻产生的费用。"""
    with _cost_lock:
        _user_costs[user_id].append((time.time(), cost))
        _user_totals[user_id]["cost"] += cost
        _user_totals[user_id]["request_count"] += 1
        cutoff = time.time() - COST_LIMIT_WINDOW_SECONDS
        _user_costs[user_id] = [(t, c) for t, c in _user_costs[user_id] if t > cutoff]


def token_add(user_id: str, prompt_tokens: int, completion_tokens: int) -> None:
    """记录用户 token 用量（用于统计展示）。"""
    if prompt_tokens <= 0 and completion_tokens <= 0:
        return
    with _cost_lock:
        _user_tokens[user_id].append((time.time(), prompt_tokens, completion_tokens))
        _user_totals[user_id]["prompt_tokens"] += prompt_tokens
        _user_totals[user_id]["completion_tokens"] += completion_tokens
        cutoff = time.time() - COST_LIMIT_WINDOW_SECONDS
        _user_tokens[user_id] = [(t, a, b) for t, a, b in _user_tokens[user_id] if t > cutoff]


def usage_get_all() -> List[Dict]:
    """Return usage for all users (totals + past-hour cost for limit display)."""
    cutoff = time.time() - COST_LIMIT_WINDOW_SECONDS
    with _cost_lock:
        users = set(_user_costs.keys()) | set(_user_tokens.keys()) | set(_user_totals.keys())
        result = []
        for uid in users:
            t = _user_totals[uid]
            cost_hourly = sum(c for t2, c in _user_costs[uid] if t2 > cutoff)
            result.append({
                "user_id": uid[:16] + "..." if len(uid) > 16 else uid,
                "cost_total": round(t["cost"], 6),
                "cost_last_hour": round(cost_hourly, 6),
                "prompt_tokens": t["prompt_tokens"],
                "completion_tokens": t["completion_tokens"],
                "total_tokens": t["prompt_tokens"] + t["completion_tokens"],
                "request_count": t["request_count"],
            })
        return sorted(result, key=lambda x: -x["cost_total"])


def cost_get_hourly(user_id: str) -> float:
    """Return user's total cost (USD) in the past hour (rolling window)."""
    cutoff = time.time() - COST_LIMIT_WINDOW_SECONDS
    with _cost_lock:
        return sum(c for t, c in _user_costs[user_id] if t > cutoff)


def _cost_limit_exceeded_after_add() -> bool:
    """本次请求记账后是否超出限额。用于在 200 响应中附加 X-Cost-Limit-Exceeded 头。"""
    uid = _get_user_id()
    return cost_get_hourly(uid) >= COST_LIMIT


def cost_check_limit() -> Optional[tuple]:
    """
    Check if user exceeds cost limit ($0.5 per 3 min per user).
    Returns None if under limit; otherwise (response dict, status_code).
    """
    uid = _get_user_id()
    if cost_get_hourly(uid) >= COST_LIMIT:
        return (
            {
                "error": "cost_limit_exceeded",
                "message": "Cost limit exceeded. Please try again later.",
                "limit_exceeded": True,
            },
            429,
        )
    return None


def cost_get_retry_after_seconds(user_id: str) -> int:
    """
    Return seconds until user can retry after exceeding the hourly cost limit.
    Based on rolling 1-hour window.
    """
    now = time.time()
    cutoff = now - COST_LIMIT_WINDOW_SECONDS
    with _cost_lock:
        costs = [(t, c) for t, c in _user_costs[user_id] if t > cutoff]
    if not costs:
        return 0
    if sum(c for _, c in costs) < COST_LIMIT:
        return 0
    costs_sorted = sorted(costs, key=lambda x: x[0])
    for t, _ in costs_sorted:
        T = t + COST_LIMIT_WINDOW_SECONDS
        if T < now:
            continue
        remaining = sum(c for t2, c in costs_sorted if t2 > T - COST_LIMIT_WINDOW_SECONDS)
        if remaining < COST_LIMIT:
            return max(0, int(T - now))
    return COST_LIMIT_WINDOW_SECONDS


def load_pdfs(directory: str) -> List[Dict]:
    """
    从目录加载所有 PDF 文件并分割成块。
    
    此函数：
        1. 查找目录中所有 PDF 文件
        2. 使用 PyPDFLoader 加载每个 PDF
        3. 使用 RecursiveCharacterTextSplitter 分割成块
        4. 返回包含元数据的块（源文件、页码）
    
    参数：
        directory：包含 PDF 文件的目录路径
    
    返回：
        字典列表，每个包含：
            - text：块内容
            - source：PDF 文件名
            - page：页码（1 开始，方便人类阅读）
    
    示例：
        [
            {"text": "AAA 修复...", "source": "Guidelines for AAA repair.pdf", "page": 5},
            {"text": "适应症...", "source": "Guidelines for AAA repair.pdf", "page": 5},
            ...
        ]
    """
    print(f"\n[1] 正在从 {directory} 加载 PDF...")
    
    dir_path = Path(directory)
    if not dir_path.exists():
        print(f"[错误] 目录不存在：{directory}")
        return []
    
    pdf_files = list(dir_path.glob("*.pdf"))
    print(f"找到 {len(pdf_files)} 个 PDF 文件")
    
    # 配置文本分割器
    # RecursiveCharacterTextSplitter 尝试在有意义的边界分割
    # （段落、句子）然后再回退到字符级分割
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE * 4,      # 将 token 转换为大约的字符数
        chunk_overlap=CHUNK_OVERLAP * 4,
        add_start_index=True,           # 跟踪在原始文档中的位置
        strip_whitespace=True,          # 移除首尾空白
        separators=[                    # 按这些边界分割（按顺序）
            "\n\n",  # 段落分隔
            "\n",    # 换行
            ". ",    # 句子结尾
            " ",     # 单词分隔
            ""       # 字符级回退
        ]
    )
    
    all_chunks = []
    
    # 处理每个 PDF 文件
    for pdf_file in pdf_files:
        try:
            # 使用 LangChain 的 PyPDFLoader 加载 PDF
            # 这会提取每一页的文本
            loader = PyPDFLoader(str(pdf_file))
            docs = loader.load()
            
            # 将文档分割成块
            chunks = text_splitter.split_documents(docs)
            
            # 转换为我们的格式并添加元数据
            for chunk in chunks:
                # PyPDFLoader 使用 0 开始的页码，转换为 1 开始
                page_num = chunk.metadata.get("page", 0) + 1
                all_chunks.append({
                    "text": chunk.page_content,
                    "source": pdf_file.name,
                    "page": page_num
                })
            
            print(f"  [OK] {pdf_file.name}：{len(chunks)} 个块")
            
        except Exception as e:
            print(f"  [X] {pdf_file.name}：{e}")
    
    print(f"\n总计：从 {len(pdf_files)} 个 PDF 中提取了 {len(all_chunks)} 个块")
    return all_chunks


def get_embeddings(texts: List[str], client: OpenAI) -> np.ndarray:
    """
    使用 OpenAI API 为文本列表生成句子嵌入（sentence embeddings）。
    
    句子嵌入是文本的稠密向量表示，能够捕捉语义含义。
    相似的文本会有相似的向量（高余弦相似度）。
    
    什么是句子嵌入？
        - 将文本转换为固定维度的向量（本系统使用 1536 维）
        - 捕捉语义信息：意思相近的文本在向量空间中距离更近
        - 支持语义搜索：通过向量相似度找到相关文档，而不仅仅是关键词匹配
    
    参数：
        texts：要生成向量的文本字符串列表（通常是文档块）
        client：OpenAI API 客户端
    
    返回：
        numpy 数组，形状为 (len(texts), EMBEDDING_DIMENSION)
        每一行是对应文本的向量表示
        - 维度：1536（text-embedding-3-small）
        - 数据类型：float32
        - 注意：返回的向量尚未归一化，需要在构建索引时归一化
    
    说明：
        - 每批处理 100 个文本，避免 API 限制（OpenAI 最多允许 2048 个输入）
        - 使用 text-embedding-3-small 以获得成本效益
        - 费用：约 $0.02/百万 token
        - 所有块使用同一模型嵌入，确保查询和文档在同一向量空间中
    """
    print(f"\n[2] 正在为 {len(texts)} 个块生成向量...")
    
    embeddings = []
    batch_size = 100  # OpenAI 每次请求最多允许 2048 个输入
    
    # 分批处理
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(texts) + batch_size - 1) // batch_size
        print(f"  正在处理批次 {batch_num}/{total_batches}...")
        
        # 调用 OpenAI Embeddings API
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch
        )
        
        # 从响应中提取向量
        batch_embeddings = [item.embedding for item in response.data]
        embeddings.extend(batch_embeddings)
    
    # 转换为 numpy 数组供 FAISS 使用
    return np.array(embeddings, dtype='float32')


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    构建用于快速相似度搜索的 FAISS 索引。
    
    我们使用 IndexFlatIP（内积）在归一化向量上实现余弦相似度。
    
    ========================================================================
    距离度量：余弦相似度（Cosine Similarity）
    ========================================================================
    
    为什么选择余弦相似度？
        1. 与我们的嵌入模型配合良好
        2. 比较向量的方向，不考虑向量大小
        3. 适合文本相似度任务：我们关心语义方向，而不是向量长度
    
    余弦相似度公式：
        cos(A, B) = (A · B) / (||A|| * ||B||)
    
    归一化的重要性：
        当向量归一化后（L2 归一化，||A|| = ||B|| = 1）：
            cos(A, B) = A · B（内积）
        
        这使我们能够：
            - 使用 FAISS IndexFlatIP（内积索引）实现余弦相似度
            - 提高计算效率（内积比余弦相似度计算更快）
            - 确保所有向量在同一尺度上比较
    
    归一化方法：
        使用 faiss.normalize_L2() 进行 L2 归一化：
            normalized_vector = vector / ||vector||
        其中 ||vector|| = sqrt(sum(vector²))
    
    ========================================================================
    搜索算法：FAISS IndexFlatIP
    ========================================================================
    
    IndexFlatIP（内积，精确搜索）：
        - 精确搜索：返回真正的最近邻，无近似误差
        - 适合中小规模数据集（< 100万向量）
        - 时间复杂度：O(n * d)，其中 n=向量数，d=维度
    
    对于更大数据集，可以考虑：
        - IndexIVFFlat：倒排文件索引，近似搜索，更快
        - IndexHNSW：分层导航小世界图，近似搜索，高召回率
    
    参数：
        embeddings：numpy 数组，形状为 (n_vectors, dimension)
                   注意：输入向量会被归一化（原地修改）
    
    返回：
        faiss.IndexFlatIP：准备好进行相似度搜索的索引
        - 所有向量已归一化
        - 使用内积计算相似度（等价于余弦相似度）
    
    复杂度：
        - 构建：O(n * d) - 存储和归一化 n 个 d 维向量
        - 搜索：O(n * d) - 精确搜索，需要比较所有向量
        - 内存：O(n * d * 4 bytes) - 每个 float32 占 4 字节
    """
    print(f"\n[3] 正在构建 FAISS 索引...")
    
    # ========================================================================
    # 关键步骤：归一化向量
    # ========================================================================
    # 使用 L2 归一化将所有向量缩放到单位范数（||v|| = 1）
    # 
    # 为什么必须归一化？
    #   1. 实现余弦相似度：归一化后，内积 = 余弦相似度
    #   2. 公平比较：所有向量在同一尺度上，不受向量长度影响
    #   3. 与查询向量一致：查询向量也会归一化，确保比较正确
    #
    # 归一化公式：
    #   normalized_vector = vector / ||vector||
    #   其中 ||vector|| = sqrt(sum(vector²)) 是 L2 范数
    #
    # 注意：faiss.normalize_L2() 会原地修改向量（in-place）
    # ========================================================================
    faiss.normalize_L2(embeddings)
    
    # ========================================================================
    # 创建 FAISS 索引
    # ========================================================================
    # IndexFlatIP：内积索引（Inner Product）
    # - 在归一化向量上，内积 = 余弦相似度
    # - 精确搜索：返回真正的最近邻，无近似误差
    # - 适合中小规模数据集（< 100万向量）
    # ========================================================================
    index = faiss.IndexFlatIP(embeddings.shape[1])
    
    # 将归一化后的向量添加到索引
    # 注意：向量必须已归一化，否则相似度计算不正确
    index.add(embeddings)
    
    print(f"  [OK] 索引已构建，包含 {index.ntotal} 个向量")
    return index


def search_similar(
    query: str, 
    client: OpenAI, 
    index: faiss.IndexFlatIP, 
    chunks: List[Dict], 
    num_retrieved: int = 30, 
    num_final: int = 5
) -> List[Dict]:
    """
    两阶段检索：FAISS 向量搜索 → ColBERTv2 重排序。
    
    ========================================================================
    检索流程
    ========================================================================
    
    步骤 1：查询向量化
        - 用户查询通过同一嵌入模型（text-embedding-3-small）转换为向量
        - 查询向量被归一化（L2 归一化）
        - 确保查询和文档在同一向量空间中比较
    
    步骤 2：FAISS 向量搜索（第一阶段）
        - 在向量数据库中快速找到最接近的文档
        - 使用余弦相似度（归一化向量的内积）
        - 返回前 num_retrieved 个候选（默认 30）
        - 时间复杂度：O(n * d)，其中 n=向量数，d=维度
    
    步骤 3：ColBERTv2 重排序（第二阶段，可选）
        - 对 FAISS 返回的候选进行精确重排序
        - 使用 token 级别比较（Late Interaction）
        - 返回前 num_final 个结果（默认 5）
        - 提高相关性，但计算成本更高
    
    ========================================================================
    为什么需要两阶段？
    ========================================================================
    
    第一阶段（FAISS）：
        优点：
            - 快速：可以在毫秒内搜索数百万向量
            - 可扩展：适合大规模向量数据库
            - 语义搜索：捕捉语义相似性
        
        局限：
            - 单向量比较：整个文档块用一个向量表示
            - 可能遗漏细微匹配：关键词匹配可能不够精确
    
    第二阶段（ColBERTv2）：
        优点：
            - Token 级别比较：更细粒度的相似度计算
            - 更好的相关性：理解查询-文档的精确匹配
            - Late Interaction：查询和文档的 token 向量交互
        
        局限：
            - 计算成本高：需要对每个候选进行 token 级别计算
            - 需要重排序模型：额外的模型加载和推理时间
    
    两阶段策略的优势：
        - 结合速度和准确性
        - FAISS 快速筛选候选，ColBERTv2 精确排序
        - 适合生产环境：在速度和准确性之间取得平衡
    
    ========================================================================
    参数
    ========================================================================
    
    query：str
        用户的搜索查询文本
    
    client：OpenAI
        OpenAI API 客户端，用于生成查询向量
    
    index：faiss.IndexFlatIP
        包含所有文档向量的 FAISS 索引（已归一化）
    
    chunks：List[Dict]
        所有文档块及其元数据的列表
        - text：块内容
        - source：PDF 文件名
        - page：页码
    
    num_retrieved：int (默认 30)
        从 FAISS 获取的候选数量
        - 更多候选 = 更高召回率，但重排序更慢
        - 建议值：20-50
    
    num_final：int (默认 5)
        重排序后的最终结果数量
        - 返回给用户的结果数
        - 建议值：3-10
    
    ========================================================================
    返回
    ========================================================================
    
    List[Dict]：按相关性排序的结果列表，每个包含：
        - text：str - 块内容
        - source：str - PDF 文件名
        - page：int - 页码
        - score：float - 相关性分数（越高越相关）
            - 如果使用重排序：ColBERTv2 分数
            - 否则：FAISS 余弦相似度分数
    """
    global reranker
    
    print(f"=> 通过 FAISS 检索 {num_retrieved} 个文档...")
    
    # -------------------------------------------------------------------------
    # 步骤 1：生成查询向量
    # -------------------------------------------------------------------------
    # 使用与文档相同的嵌入模型生成查询向量
    # 这确保查询和文档在同一向量空间中，可以正确比较
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[query]
    )
    query_embedding = np.array([response.data[0].embedding], dtype='float32')
    # 从 embeddings API 获取 token 用量（用于费用统计）
    embedding_total_tokens = 0
    if hasattr(response, 'usage') and response.usage:
        embedding_total_tokens = getattr(response.usage, 'total_tokens', 0) or 0
    
    # 归一化查询向量以实现余弦相似度
    # 重要：查询向量必须与文档向量使用相同的归一化方法
    # 归一化后，内积 = 余弦相似度
    # 如果不归一化，相似度计算将不正确
    faiss.normalize_L2(query_embedding)
    
    # -------------------------------------------------------------------------
    # 步骤 2：FAISS 向量搜索（第一阶段）
    # -------------------------------------------------------------------------
    # 在向量数据库中搜索最相似的文档
    #
    # 搜索原理：
    #   1. 计算查询向量与所有文档向量的内积（已归一化，所以是余弦相似度）
    #   2. 返回相似度最高的 num_retrieved 个文档
    #
    # search() 返回：
    #   - distances：相似度分数数组，形状 (1, num_retrieved)
    #     * 对于 IndexFlatIP：内积值（已归一化，所以是余弦相似度）
    #     * 值范围：[-1, 1]，1 表示完全相同，-1 表示完全相反
    #     * 越高越相似
    #   - indices：文档索引数组，形状 (1, num_retrieved)
    #     * 在 chunks 数组中的位置
    #     * 按相似度从高到低排序
    #
    # 时间复杂度：O(n * d)，其中 n=向量数，d=维度
    # ========================================================================
    distances, indices = index.search(query_embedding, num_retrieved)
    
    # 从 FAISS 构建结果
    faiss_results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(chunks):
            faiss_results.append({
                "text": chunks[idx]["text"],
                "source": chunks[idx]["source"],
                "page": chunks[idx].get("page", 0),
                "faiss_score": float(distances[0][i])
            })
    
    print(f"  [OK] FAISS 返回了 {len(faiss_results)} 个候选")
    
    # -------------------------------------------------------------------------
    # 步骤 3：ColBERTv2 重排序（第二阶段）
    # -------------------------------------------------------------------------
    if reranker and len(faiss_results) > num_final:
        print(f"=> 使用 ColBERTv2 重排序...")
        try:
            # 提取文本用于重排序
            texts_to_rerank = [r["text"] for r in faiss_results]
            
            # 使用 ColBERTv2 重排序
            # 这会比较查询和每个候选之间的 token 向量
            reranked = reranker.rerank(query, texts_to_rerank, k=num_final)
            
            # 将重排序结果映射回原始元数据
            final_results = []
            for item in reranked:
                content = item["content"]
                # 找到原始块信息
                for r in faiss_results:
                    if r["text"] == content:
                        final_results.append({
                            "text": content,
                            "source": r["source"],
                            "page": r["page"],
                            "score": float(item["score"]),  # ColBERTv2 分数
                            "faiss_score": r["faiss_score"]  # 保留用于调试
                        })
                        break
            
            print(f"  [OK] 重排序后返回前 {len(final_results)} 个文档")
            return final_results, embedding_total_tokens
            
        except Exception as e:
            print(f"  [!] 重排序失败：{e}，使用 FAISS 结果")
    
    # -------------------------------------------------------------------------
    # 回退：返回 FAISS 的前 N 个结果（不重排序）
    # -------------------------------------------------------------------------
    results = []
    for r in faiss_results[:num_final]:
        results.append({
            "text": r["text"],
            "source": r["source"],
            "page": r["page"],
            "score": r["faiss_score"]
        })
    
    return results, embedding_total_tokens


# =============================================================================
# 初始化
# =============================================================================

def initialize_rag():
    """
    在服务器启动时初始化 RAG 系统。
    
    此函数：
        1. 加载 OpenAI API 密钥
        2. 加载 ColBERTv2 重排序模型
        3. 加载现有索引或构建新索引
    
    索引构建过程：
        1. 从 Clinical Guidelines 目录加载 PDF
        2. 分割成块（每块约 512 token）
        3. 通过 OpenAI API 生成向量
        4. 构建 FAISS 索引
        5. 保存索引和块到磁盘
    
    返回：
        bool：初始化成功返回 True，否则返回 False
    
    说明：
        - 首次运行需要 5-10 分钟（生成向量）
        - 后续运行会立即加载保存的索引
        - 删除 faiss_index.bin 和 all_chunks.json 可以重建索引
    """
    global faiss_index, all_chunks, openai_client, reranker
    
    print("=" * 60)
    print("RAG 服务器 - 初始化中...")
    print("=" * 60)
    print(f"  检索：FAISS -> {NUM_RETRIEVED_DOCS} 个候选")
    print(f"  重排序：ColBERTv2 -> {NUM_DOCS_FINAL} 个最终结果")
    print("=" * 60)
    
    # -------------------------------------------------------------------------
    # 步骤 0：加载 API 密钥并创建 OpenAI 客户端
    # -------------------------------------------------------------------------
    api_key = get_api_key()
    if not api_key:
        print("[错误] 未找到 OpenAI API 密钥！")
        print("  请设置 OPENAI_API_KEY 环境变量或创建 api-key.js")
        return False
    
    openai_client = OpenAI(api_key=api_key)
    
    # -------------------------------------------------------------------------
    # 步骤 1：加载重排序器（仅 ColBERTv2）
    # -------------------------------------------------------------------------
    if HAS_RERANKER:
        print("\n[0] 正在加载 ColBERTv2 重排序器...")
        try:
            import os
            os.environ['COLBERT_LOAD_TORCH_EXTENSION_VERBOSE'] = 'False'
            os.environ['COLBERT_USE_CPP'] = 'False'
            reranker = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
            print("  [OK] ColBERTv2 重排序器已加载")
        except Exception as e:
            error_msg = str(e)
            print(f"  [!] 加载 ColBERTv2 失败：{e}")
            if 'pthread.h' in error_msg or 'No such file or directory' in error_msg or 'cl' in error_msg or 'compiler' in error_msg.lower():
                print("  [!] 原因：ColBERTv2 的 C++ 扩展在 Windows 上不兼容，请在 WSL/Linux 下运行以启用重排序")
            reranker = None

    if reranker is None:
        print("\n[!] 无重排序器，将仅使用 FAISS 检索")
    
    # -------------------------------------------------------------------------
    # 步骤 2：加载或构建 FAISS 索引
    # -------------------------------------------------------------------------
    index_path = Path(INDEX_FILE)
    chunks_path = Path(CHUNKS_FILE)
    
    if index_path.exists() and chunks_path.exists():
        # 加载现有索引（快速）
        print("\n[OK] 正在加载现有索引...")
        faiss_index = faiss.read_index(str(index_path))
        with open(chunks_path, 'r', encoding='utf-8') as f:
            all_chunks = json.load(f)
        print(f"  已加载 {faiss_index.ntotal} 个向量，{len(all_chunks)} 个块")
    else:
        # 构建新索引（慢，仅首次运行）
        print("\n[!] 正在构建新索引（可能需要几分钟）...")
        
        # 加载并分块 PDF
        all_chunks = load_pdfs(PDF_DIRECTORY)
        if not all_chunks:
            return False
        
        # 生成向量
        texts = [chunk["text"] for chunk in all_chunks]
        embeddings = get_embeddings(texts, openai_client)
        
        # 构建 FAISS 索引
        faiss_index = build_faiss_index(embeddings)
        
        # 保存到磁盘供后续运行使用
        print("\n[4] 正在保存索引...")
        faiss.write_index(faiss_index, str(index_path))
        with open(chunks_path, 'w', encoding='utf-8') as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=2)
        print(f"  [OK] 已保存到 {INDEX_FILE} 和 {CHUNKS_FILE}")
    
    # -------------------------------------------------------------------------
    # 强制要求重排序（REQUIRE_RERANKER=1 时，未加载则拒绝启动）
    # -------------------------------------------------------------------------
    if os.environ.get('REQUIRE_RERANKER', '').strip().lower() in ('1', 'true', 'yes'):
        if reranker is None:
            print("\n" + "=" * 60)
            print("[错误] 已设置 REQUIRE_RERANKER，但 ColBERTv2 重排序器未加载")
            print("  - 请在 WSL 或 Linux 环境下运行以启用 ColBERTv2")
            print("  - 或取消设置 REQUIRE_RERANKER 后仅使用 FAISS 检索")
            print("=" * 60)
            return False

    # -------------------------------------------------------------------------
    # 完成
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("[完成] RAG 服务器就绪！")
    print(f"  - FAISS 索引：{faiss_index.ntotal} 个向量")
    print(f"  - 重排序器：{'ColBERTv2' if reranker else '无（仅 FAISS）'}")
    print("=" * 60)
    return True


# =============================================================================
# API 端点
# =============================================================================

@app.route('/search', methods=['POST'])
def search():
    """搜索入口：先检查用户是否超出每小时成本限额。"""
    blocked = cost_check_limit()
    if blocked:
        return jsonify(blocked[0]), blocked[1]
    return _search_impl()


def _search_impl():
    """
    搜索端点 - 为查询查找相关块。
    
    两阶段检索：
        1. FAISS 检索前 30 个候选（快速、近似）
        2. ColBERTv2 重排序返回前 5 个（慢、准确）
    
    请求：
        POST /search
        Content-Type: application/json
        {
            "query": "AAA 修复的适应症是什么？",
            "num_retrieved": 30,  // 可选，默认 30
            "num_final": 5        // 可选，默认 5
        }
    
    响应：
        {
            "query": "AAA 修复的适应症是什么？",
            "num_retrieved": 30,
            "num_final": 5,
            "reranker": "ColBERTv2",
            "results": [
                {
                    "text": "AAA 修复的适应症包括...",
                    "source": "Guidelines for AAA repair.pdf",
                    "page": 12,
                    "score": 0.892
                },
                ...
            ]
        }
    
    错误响应：
        {"error": "错误信息"}, 状态码 400 或 500
    """
    try:
        data = request.json
        query = data.get('query', '')
        num_retrieved = data.get('num_retrieved', NUM_RETRIEVED_DOCS)
        num_final = data.get('num_final', NUM_DOCS_FINAL)
        
        if not query:
            return jsonify({"error": "未提供查询"}), 400
        
        # 执行两阶段检索
        results, embedding_tokens = search_similar(
            query, 
            openai_client, 
            faiss_index, 
            all_chunks, 
            num_retrieved=num_retrieved,
            num_final=num_final
        )
        # 按实际 token 数记录 embedding 费用
        uid = _get_user_id()
        if embedding_tokens > 0:
            cost_add(uid, _estimate_embedding_cost(embedding_tokens))
            token_add(uid, embedding_tokens, 0)  # embedding 只有输入 token
        else:
            cost_add(uid, 0.000002)  # 兜底：API 未返回 usage 时用估算
        resp = jsonify({
            "query": query,
            "num_retrieved": num_retrieved,
            "num_final": num_final,
            "reranker": "ColBERTv2" if reranker else "None",
            "results": results
        })
        if _cost_limit_exceeded_after_add():
            resp.headers["X-Cost-Limit-Exceeded"] = "true"
        return resp
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/chat_completion', methods=['POST'])
def chat_completion():
    """
    OpenAI chat proxy: forward request, track cost, enforce $0.5/3min per-user limit.
    超限时返回 429。
    """
    blocked = cost_check_limit()
    if blocked:
        return jsonify(blocked[0]), blocked[1]

    try:
        data = request.json or {}
        model = data.get("model", "gpt-4o-mini")
        messages = data.get("messages", [])
        max_tokens = data.get("max_tokens", 1000)
        temperature = data.get("temperature", 0.7)
        response_format = data.get("response_format")

        if not messages:
            return jsonify({"error": "messages required"}), 400

        kwargs = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if response_format:
            kwargs["response_format"] = response_format

        resp = openai_client.chat.completions.create(**kwargs)
        usage = getattr(resp, "usage", None)
        if usage:
            pt = getattr(usage, "prompt_tokens", 0) or 0
            ct = getattr(usage, "completion_tokens", 0) or 0
            cost = _estimate_chat_cost(pt, ct, model)
            uid = _get_user_id()
            cost_add(uid, cost)
            token_add(uid, pt, ct)

        choice = resp.choices[0] if resp.choices else None
        if not choice:
            return jsonify({"error": "no response"}), 500

        content = choice.message.content
        resp = jsonify({
            "choices": [{"message": {"content": content, "role": "assistant"}}],
            "usage": {
                "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                "completion_tokens": getattr(usage, "completion_tokens", 0),
            } if usage else None,
        })
        if _cost_limit_exceeded_after_add():
            resp.headers["X-Cost-Limit-Exceeded"] = "true"
        return resp
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """
    健康检查端点 - 验证服务器正在运行且配置正确。
    
    请求：
        GET /health
    
    响应：
        {
            "status": "ok",
            "index_size": 1500,
            "chunks_count": 1500,
            "reranker": "ColBERTv2",
            "config": {
                "num_retrieved": 30,
                "num_final": 5
            }
        }
    
    使用场景：
        - medical-quiz.html 检查 RAG 服务器是否可用
        - 监控系统进行健康检查
    """
    return jsonify({
        "status": "ok",
        "index_size": faiss_index.ntotal if faiss_index else 0,
        "chunks_count": len(all_chunks),
        "reranker": "ColBERTv2" if reranker else "None",
        "config": {
            "num_retrieved": NUM_RETRIEVED_DOCS,
            "num_final": NUM_DOCS_FINAL
        }
    })


@app.route('/usage_stats', methods=['GET'])
def usage_stats():
    """
    返回各用户在过去 1 小时内的 token 用量与费用统计（用于管理界面）。
    Returns per-user token usage and cost in the last hour.
    """
    return jsonify({"users": usage_get_all(), "limit_per_hour": COST_LIMIT})


@app.route('/cost_limit_status', methods=['GET'])
def cost_limit_status():
    """
    返回当前用户的限额状态，用于前端显示倒计时。
    Returns: { "limited": bool, "retry_after_seconds": int }
    """
    uid = _get_user_id()
    limited = cost_get_hourly(uid) >= COST_LIMIT
    retry = cost_get_retry_after_seconds(uid) if limited else 0
    return jsonify({"limited": limited, "retry_after_seconds": retry})


# -----------------------------------------------------------------------------
# Per-user log files: button clicks, tutor conversations, knowledge profile, usage, events (login/logout)
# -----------------------------------------------------------------------------
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USER_LOGS_DIR = os.path.join(_BASE_DIR, "user_logs")
_user_log_lock = Lock()


def _safe_username(user: str) -> str:
    """Safe filename from user id (strip user: prefix and path chars)."""
    s = (user or "unknown").strip()
    if s.startswith("user:"):
        s = s[5:]
    s = re.sub(r'[<>:"/\\|?*]', "_", s)[:64]
    return s or "unknown"


def _get_user_log_path(user: str) -> str:
    safe = _safe_username(user)
    os.makedirs(USER_LOGS_DIR, exist_ok=True)
    return os.path.join(USER_LOGS_DIR, f"{safe}.json")


def _read_user_log(user: str) -> dict:
    path = _get_user_log_path(user)
    with _user_log_lock:
        if not os.path.exists(path):
            return {"user": _safe_username(user), "button_clicks": [], "tutor_conversations": [], "knowledge_profile": None, "display_snapshot": None, "usage": {"cost_total": 0.0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "request_count": 0}, "events": []}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            data = {}
    if not isinstance(data, dict):
        data = {}
    for key in ["button_clicks", "tutor_conversations", "knowledge_profile", "display_snapshot", "usage", "events"]:
        if key not in data:
            if key == "usage":
                data["usage"] = {"cost_total": 0.0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "request_count": 0}
            elif key == "knowledge_profile":
                data["knowledge_profile"] = None
            elif key == "display_snapshot":
                data["display_snapshot"] = None
            elif key == "events":
                data["events"] = []
            else:
                data[key] = []
    return data


def _write_user_log(user: str, data: dict) -> None:
    path = _get_user_log_path(user)
    with _user_log_lock:
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except IOError:
            pass


def _append_button_click(user: str, timestamp: str, button: str, question_id: Optional[int] = None) -> None:
    data = _read_user_log(user)
    data["button_clicks"].append({"timestamp": timestamp, "button": button, "question_id": question_id})
    _write_user_log(user, data)


def _append_tutor_conversation(user: str, question_id: Optional[int], question_text: str, options: str, messages: List[dict]) -> None:
    """Append or update one tutor conversation. messages = [{"user": "...", "assistant": "..."}, ...]"""
    data = _read_user_log(user)
    convos = data["tutor_conversations"]
    found = None
    for i, c in enumerate(convos):
        if c.get("question_id") == question_id:
            found = i
            break
    entry = {"question_id": question_id, "question_text": question_text or "", "options": options or "", "messages": list(messages)}
    if found is not None:
        convos[found] = entry
    else:
        convos.append(entry)
    _write_user_log(user, data)


def _update_user_log_knowledge_profile(user: str, profile: dict, display_snapshot: Optional[dict] = None) -> None:
    data = _read_user_log(user)
    data["knowledge_profile"] = profile
    if display_snapshot is not None:
        data["display_snapshot"] = display_snapshot
    _write_user_log(user, data)


def _append_user_log_event(user: str, event: str, timestamp: str, profile: Optional[dict] = None) -> None:
    """Append login/logout event to this user's log file (no shared file)."""
    data = _read_user_log(user)
    if "events" not in data or not isinstance(data["events"], list):
        data["events"] = []
    entry = {"event": event, "timestamp": timestamp}
    if profile is not None:
        profile_copy = dict(profile)
        profile_copy["_scope"] = "cumulative"
        profile_copy["_scope_description"] = "questionsAnswered and knowledgeMap.* are totals across all sessions."
        entry["profile"] = profile_copy
    data["events"].append(entry)
    _write_user_log(user, data)


@app.route('/user_log/event', methods=['POST'])
def user_log_event():
    """Log login/logout to this user's own file. Body: { user, event, timestamp, profile? }."""
    try:
        body = request.get_json() or {}
        user = (body.get("user") or request.headers.get("X-User-Id") or "unknown").strip()
        event = body.get("event") or "login"
        ts = body.get("timestamp") or time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
        profile = body.get("profile")
        _append_user_log_event(user, event, ts, profile)
        path = _get_user_log_path(user)
        print(f"[user_log] {event} for user '{user}' -> {path}")
        return jsonify({"ok": True})
    except Exception as e:
        print(f"[user_log] ERROR: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500


def _sync_user_log_usage(user: str) -> None:
    """Write in-memory _user_totals for this user into their log file."""
    uid = ("user:" + user.strip())[:64] if user and not user.startswith("user:") else (user or "unknown")[:64]
    with _cost_lock:
        t = _user_totals.get(uid)
        if not t:
            return
        usage = {
            "cost_total": round(t["cost"], 6),
            "prompt_tokens": t["prompt_tokens"],
            "completion_tokens": t["completion_tokens"],
            "total_tokens": t["prompt_tokens"] + t["completion_tokens"],
            "request_count": t["request_count"],
        }
    data = _read_user_log(user)
    data["usage"] = usage
    _write_user_log(user, data)


@app.route('/user_log/button', methods=['POST'])
def user_log_button():
    """Log a button click. Body: { user, timestamp, button, question_id? }."""
    try:
        body = request.get_json() or {}
        user = (body.get("user") or request.headers.get("X-User-Id") or "unknown").strip()
        ts = body.get("timestamp") or time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
        button = body.get("button") or "unknown"
        question_id = body.get("question_id")
        if question_id is not None and not isinstance(question_id, int):
            try:
                question_id = int(question_id)
            except (TypeError, ValueError):
                question_id = None
        _append_button_click(user, ts, button, question_id)
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route('/user_log/conversation', methods=['POST'])
def user_log_conversation():
    """Log tutor conversation for a question. Body: { user, question_id, question_text?, options?, messages: [{user, assistant}, ...] }."""
    try:
        body = request.get_json() or {}
        user = (body.get("user") or request.headers.get("X-User-Id") or "unknown").strip()
        question_id = body.get("question_id")
        if question_id is not None and not isinstance(question_id, int):
            try:
                question_id = int(question_id)
            except (TypeError, ValueError):
                question_id = None
        question_text = body.get("question_text") or ""
        options = body.get("options") or ""
        messages = body.get("messages") or []
        if not isinstance(messages, list):
            messages = []
        _append_tutor_conversation(user, question_id, question_text, options, messages)
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route('/user_log/knowledge_profile', methods=['POST'])
def user_log_knowledge_profile():
    """Save latest knowledge profile snapshot. Body: { user, profile, display_snapshot? }.
    display_snapshot: { totalSubtopicsInBank, domainTotals: { "Domain A": 81, ... } } so the file contains full UI denominators."""
    try:
        body = request.get_json() or {}
        user = (body.get("user") or request.headers.get("X-User-Id") or "unknown").strip()
        profile = body.get("profile")
        display_snapshot = body.get("display_snapshot")
        _update_user_log_knowledge_profile(user, profile or {}, display_snapshot)
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route('/user_log/usage', methods=['GET'])
def user_log_usage():
    """Return usage (tokens, cost) for the given user. Persist in-memory totals to user log file."""
    user = (request.args.get("user") or request.headers.get("X-User-Id") or "").strip()
    if not user:
        return jsonify({"error": "user required"}), 400
    uid = ("user:" + user)[:64]
    with _cost_lock:
        t = _user_totals.get(uid, {"cost": 0.0, "prompt_tokens": 0, "completion_tokens": 0, "request_count": 0})
        usage = {
            "cost_total": round(t["cost"], 6),
            "prompt_tokens": t["prompt_tokens"],
            "completion_tokens": t["completion_tokens"],
            "total_tokens": t["prompt_tokens"] + t["completion_tokens"],
            "request_count": t["request_count"],
        }
    _sync_user_log_usage(user)
    data = _read_user_log(user)
    data["usage"] = usage
    _write_user_log(user, data)
    return jsonify(usage)


@app.route('/user_log', methods=['GET'])
def user_log_get():
    """Return full log for user (button_clicks, tutor_conversations, knowledge_profile, usage)."""
    user = (request.args.get("user") or request.headers.get("X-User-Id") or "").strip()
    if not user:
        return jsonify({"error": "user required"}), 400
    _sync_user_log_usage(user)
    data = _read_user_log(user)
    return jsonify(data)


@app.route('/activity_log.html', methods=['GET'])
def serve_activity_log_page():
    """Serve the activity log UI (can show per-user log via GET /user_log?user=)."""
    _dir = os.path.dirname(os.path.abspath(__file__))
    return send_from_directory(_dir, "activity_log.html")


@app.route('/user_log/users', methods=['GET'])
def user_log_list_users():
    """Return list of usernames that have a log file (for activity_log.html dropdown)."""
    if not os.path.isdir(USER_LOGS_DIR):
        return jsonify({"users": []})
    users = []
    for name in os.listdir(USER_LOGS_DIR):
        if name.endswith(".json"):
            users.append(name[:-5])
    return jsonify({"users": sorted(users)})


@app.route('/rebuild', methods=['POST'])
def rebuild():
    """
    重建端点 - 从 PDF 重新创建索引。
    
    使用场景：
        - PDF 文件被添加/修改/删除
        - 索引似乎损坏
        - 更改块大小或向量模型
    
    请求：
        POST /rebuild
    
    响应：
        {
            "status": "rebuilt",
            "chunks_count": 1500,
            "index_size": 1500
        }
    
    警告：
        - 这很慢（5-10 分钟）且花费金钱（API 调用）
        - 重建期间服务器将无响应
    """
    global faiss_index, all_chunks
    
    try:
        # 删除现有文件
        if Path(INDEX_FILE).exists():
            Path(INDEX_FILE).unlink()
        if Path(CHUNKS_FILE).exists():
            Path(CHUNKS_FILE).unlink()
        
        # 重建所有内容
        all_chunks = load_pdfs(PDF_DIRECTORY)
        texts = [chunk["text"] for chunk in all_chunks]
        embeddings = get_embeddings(texts, openai_client)
        faiss_index = build_faiss_index(embeddings)
        
        # 保存新索引
        faiss.write_index(faiss_index, INDEX_FILE)
        with open(CHUNKS_FILE, 'w', encoding='utf-8') as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=2)
        
        return jsonify({
            "status": "rebuilt",
            "chunks_count": len(all_chunks),
            "index_size": faiss_index.ntotal
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =============================================================================
# 提示生成端点 (Proactive Question Generation)
# =============================================================================

# 导入提示生成器
hint_generator = None

def initialize_hint_generator():
    """
    初始化苏格拉底式提示生成器。
    使用 rag_server 的 get_api_key（与 RAG/embedding 相同），确保密钥一致。
    """
    global hint_generator
    try:
        from proactive_question_generator import ProactiveQuestionGenerator
        api_key = get_api_key()
        if not api_key:
            print("[!] 提示生成器：未找到 API 密钥")
            hint_generator = None
            return False
        hint_generator = ProactiveQuestionGenerator(api_key=api_key)
        print("[OK] 提示生成器已加载")
        return True
    except Exception as e:
        print(f"[!] 提示生成器加载失败: {e}")
        import traceback
        traceback.print_exc()
        hint_generator = None
        return False


@app.route('/generate_hints', methods=['POST'])
def generate_hints():
    """
    生成苏格拉底式提示问题端点。
    
    当学生回答错误或请求提示时，生成引导性子问题帮助学生
    自己发现正确答案，而不是直接告诉他们。
    
    请求：
        POST /generate_hints
        Content-Type: application/json
        {
            "question": "问题文本",
            "choices": {"A": "选项A", "B": "选项B", "C": "选项C", "D": "选项D"},
            "student_answer": "B",
            "correct_answer": "A",
            "explanations": {"A": "解释A", ...},  // 可选
            "conversation_history": [...],  // 可选，多轮提示时使用
            "source_context": "相关上下文"  // 可选，RAG检索的上下文
        }
    
    响应（基于 arXiv:2305.14999 交互式苏格拉底提问）：
        {
            "success": true,
            "sub_questions": [
                {
                    "id": 1,
                    "question": "学生需要回答的子问题",
                    "hint": "如果学生困难可以提供的提示",
                    "expected_concepts": ["期望的关键概念"]
                },
                {
                    "id": 2,
                    "question": "下一个子问题",
                    "hint": "提示",
                    "expected_concepts": ["概念"]
                }
            ],
            "connection_to_main": "解释回答这些问题如何帮助解决主问题",
            "encouragement": "鼓励信息",
            "round_number": 1,
            "is_final_round": false,
            "formatted_display": "格式化的显示文本"
        }
    
    错误响应：
        {"success": false, "error": "错误信息"}, 状态码 400 或 500
    """
    blocked = cost_check_limit()
    if blocked:
        return jsonify(blocked[0]), blocked[1]
    global hint_generator
    
    try:
        # 检查生成器是否可用
        if hint_generator is None:
            # 尝试初始化
            if not initialize_hint_generator():
                return jsonify({
                    "success": False,
                    "error": "提示生成器未初始化，请检查 API 密钥配置"
                }), 500
        
        data = request.json
        
        # 验证必需字段
        required_fields = ['question', 'choices', 'student_answer', 'correct_answer']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    "success": False,
                    "error": f"缺少必需字段: {field}"
                }), 400
        
        # 导入请求类
        from proactive_question_generator import HintRequest
        
        # 构建请求
        hint_request = HintRequest(
            question=data['question'],
            choices=data['choices'],
            student_answer=data['student_answer'],
            correct_answer=data['correct_answer'],
            explanations=data.get('explanations'),
            conversation_history=data.get('conversation_history'),
            source_context=data.get('source_context')
        )
        
        # 生成推理步骤链 (基于 MedTutor-R1 方法)
        usage_out = {}
        response = hint_generator.generate_sub_questions(hint_request, usage_out=usage_out)
        
        # 格式化显示
        formatted = hint_generator.format_hints_for_display(response)
        
        # 转换为可序列化格式
        response_data = hint_generator.to_dict(response)
        
        # 按实际 token 记录费用
        uid = _get_user_id()
        pt = usage_out.get("prompt_tokens", 0) or 0
        ct = usage_out.get("completion_tokens", 0) or 0
        model = usage_out.get("model", "gpt-4o")
        if pt or ct:
            cost_add(uid, _estimate_chat_cost(pt, ct, model))
            token_add(uid, pt, ct)
        else:
            cost_add(uid, 0.02)  # 兜底
        resp = jsonify({
            "success": True,
            "decomposition": response_data["decomposition"],
            "active_step_id": response_data["active_step_id"],
            "total_steps": response_data["total_steps"],
            "is_complete": response_data["is_complete"],
            "formatted_display": formatted
        })
        if _cost_limit_exceeded_after_add():
            resp.headers["X-Cost-Limit-Exceeded"] = "true"
        return resp
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/evaluate_answer', methods=['POST'])
def evaluate_answer():
    """
    评估学生对推理步骤的回答，并根据需要生成更简单的子步骤。
    
    基于 MedTutor-R1 方法：如果学生不理解，将步骤分解成更简单的子步骤。
    
    请求：
        POST /evaluate_answer
        Content-Type: application/json
        {
            "question": "原始问题",
            "choices": {...},
            "student_answer": "学生对原问题的答案",
            "correct_answer": "正确答案",
            "step_id": "1",
            "key_question": "推理步骤问题",
            "step_summary": "步骤目的",
            "expected_understanding": "期望的理解",
            "student_response": "学生的回答"
        }
    
    响应：
        {
            "success": true,
            "understood": true/false,
            "feedback": "反馈信息",
            "sub_steps": [...],  // 如果不理解，更简单的子步骤
            "missing_concept": "缺失的概念"
        }
    """
    blocked = cost_check_limit()
    if blocked:
        return jsonify(blocked[0]), blocked[1]
    global hint_generator
    
    try:
        if hint_generator is None:
            if not initialize_hint_generator():
                return jsonify({
                    "success": False,
                    "error": "提示生成器未初始化"
                }), 500
        
        data = request.json
        
        from proactive_question_generator import HintRequest, ReasoningStep
        
        # 构建请求上下文
        hint_request = HintRequest(
            question=data['question'],
            choices=data['choices'],
            student_answer=data['student_answer'],
            correct_answer=data['correct_answer']
        )
        
        # 构建推理步骤
        step = ReasoningStep(
            step_id=data['step_id'],
            key_question=data['key_question'],
            step_summary=data.get('step_summary', ''),
            expected_understanding=data.get('expected_understanding', '')
        )
        
        # 评估学生回答
        usage_out = {}
        result = hint_generator.evaluate_response(
            request=hint_request,
            step=step,
            student_response=data['student_response'],
            usage_out=usage_out
        )
        
        # 按实际 token 记录费用
        uid = _get_user_id()
        pt = usage_out.get("prompt_tokens", 0) or 0
        ct = usage_out.get("completion_tokens", 0) or 0
        model = usage_out.get("model", "gpt-4o")
        if pt or ct:
            cost_add(uid, _estimate_chat_cost(pt, ct, model))
            token_add(uid, pt, ct)
        else:
            cost_add(uid, 0.02)
        
        # 转换子步骤为可序列化格式
        sub_steps = []
        for s in result.sub_steps:
            sub_steps.append({
                "step_id": s.step_id,
                "key_question": s.key_question,
                "step_summary": s.step_summary,
                "expected_understanding": s.expected_understanding
            })
        
        cost_add(_get_user_id(), 0.02)
        resp = jsonify({
            "success": True,
            "step_id": result.step_id,
            "understood": result.understood,
            "feedback": result.feedback,
            "sub_steps": sub_steps,
            "missing_concept": result.missing_concept
        })
        if _cost_limit_exceeded_after_add():
            resp.headers["X-Cost-Limit-Exceeded"] = "true"
        return resp
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/evaluate_student_thinking', methods=['POST'])
def evaluate_student_thinking():
    """
    Evaluate student's thinking process after wrong answer and decide action (decompose or clarify).
    评估学生答错后的思考过程，并决定采取的行动（分解或澄清）。
    
    This endpoint implements the tau-bench educational agent policy:
    此端点实现tau-bench教育助手策略：
    
    1. Receives student's explanation of their reasoning after answering incorrectly
       接收学生答错后对其推理过程的解释
    
    2. Uses LLM to analyze understanding level and decide action:
       使用LLM分析理解程度并决定行动：
       - If student has fundamental gaps → DECOMPOSE (break into simpler questions)
         如果学生有基础缺口 → 分解（拆分为更简单的子问题）
       - If student is close but has minor errors → CLARIFY (provide direct explanation)
         如果学生接近但有小错误 → 澄清（提供直接解释）
    
    3. Returns structured guidance based on decision
       根据决定返回结构化指导
    
    Request Body / 请求体:
        POST /evaluate_student_thinking
        Content-Type: application/json
        {
            "question": "原始医学问题",
            "choices": {"A": "选项A文本", "B": "选项B文本", ...},
            "student_answer": "B",  // 学生选择的错误答案
            "correct_answer": "A",   // 正确答案（仅用于LLM参考，不会透露给学生）
            "student_thinking": "学生解释他们的思考过程，例如：'我认为B是正确的，因为...' 或 '我不知道'"
        }
    
    Response / 响应:
        {
            "success": true,
            "action_type": "decompose" | "clarify",  // 决定的行动类型
            "understanding_level": "none" | "partial" | "close",  // 理解程度
            "reasoning": "为什么选择这个行动（LLM的推理过程）",
            "feedback": "鼓励性反馈消息",
            "missing_concept": "学生需要理解的概念（如果不理解）",
            "clarification": "澄清文本（如果action_type是'clarify'）",
            "sub_questions": ["子问题1", "子问题2", ...]  // 如果action_type是'decompose'，提供1-3个子问题
        }
    
    Error Response / 错误响应:
        {
            "success": false,
            "error": "错误信息"
        }
    
    Example Usage / 使用示例:
        Frontend sends:
        前端发送:
        {
            "question": "What is the treatment for appendicitis?",
            "choices": {"A": "Surgery", "B": "Antibiotics only", ...},
            "student_answer": "B",
            "correct_answer": "A",
            "student_thinking": "I thought antibiotics would be enough"
        }
        
        Backend returns:
        后端返回:
        {
            "success": true,
            "action_type": "clarify",
            "understanding_level": "close",
            "reasoning": "Student understands treatment concept but confused about urgency",
            "feedback": "Good thinking about antibiotics, but let's consider the urgency...",
            "clarification": "Appendicitis requires immediate surgery because...",
            "sub_questions": []
        }
    """
    blocked = cost_check_limit()
    if blocked:
        return jsonify(blocked[0]), blocked[1]
    global hint_generator
    
    try:
        # Check if hint generator is initialized
        # 检查提示生成器是否已初始化
        if hint_generator is None:
            if not initialize_hint_generator():
                return jsonify({
                    "success": False,
                    "error": "Hint generator not initialized"
                }), 500
        
        # Parse request data
        # 解析请求数据
        data = request.json
        if not data:
            return jsonify({"success": False, "error": "Request body is empty or invalid JSON"}), 400
        
        # Validate required fields
        # 验证必需字段
        required_fields = ['question', 'choices', 'student_answer', 'correct_answer', 'student_thinking']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    "success": False,
                    "error": f"Missing required field: {field}"
                }), 400
        
        from proactive_question_generator import HintRequest
        
        # Build HintRequest object with question context
        # 构建包含问题上下文的HintRequest对象
        hint_request = HintRequest(
            question=data['question'],  # Original medical question
            choices=data['choices'],  # Answer options dictionary
            student_answer=data['student_answer'],  # Student's wrong answer
            correct_answer=data['correct_answer'],  # Correct answer (for LLM reference only)
            source_context=data.get('source_context')  # Optional: RAG retrieved context
        )
        
        # Call the evaluation method to analyze student thinking
        # This uses LLM to decide between DECOMPOSE and CLARIFY actions
        # 调用评估方法分析学生思考
        # 这使用LLM在DECOMPOSE和CLARIFY行动之间做决定
        usage_out = {}
        result = hint_generator.evaluate_student_thinking(
            request=hint_request,
            student_thinking=data['student_thinking'],
            usage_out=usage_out
        )
        
        # 按实际 token 记录费用
        uid = _get_user_id()
        pt = usage_out.get("prompt_tokens", 0) or 0
        ct = usage_out.get("completion_tokens", 0) or 0
        model = usage_out.get("model", "gpt-4o")
        if pt or ct:
            cost_add(uid, _estimate_chat_cost(pt, ct, model))
            token_add(uid, pt, ct)
        else:
            cost_add(uid, 0.02)
        
        # Note: evaluate_student_thinking already generates sub_questions for "decompose" action
        # using MedTutor-R1 style decomposition (see proactive_question_generator.py line 1079-1086)
        # 注意：evaluate_student_thinking已经为"decompose"操作生成sub_questions
        # 使用MedTutor-R1风格的分解（参见proactive_question_generator.py第1079-1086行）
        
        # Return structured response with all guidance information
        # 返回包含所有指导信息的结构化响应
        cost_add(_get_user_id(), 0.02)
        resp = jsonify({
            "success": True,
            "action_type": result.get("action_type"),  # "decompose" or "clarify" or None (if understood)
            "understanding_level": result.get("understanding_level", "partial"),  # "none" | "partial" | "close" | "understood"
            "understood": result.get("understood", False),  # FLOW TERMINATION: True if student understands
            "flow_terminated": result.get("flow_terminated", False),  # FLOW TERMINATION: True if guidance should end
            "reasoning": result.get("reasoning", ""),  # LLM's explanation of why this action was chosen
            "feedback": result.get("feedback", ""),  # Encouraging feedback message
            "missing_concept": result.get("missing_concept", ""),  # What concept student needs to understand
            "clarification": result.get("clarification", ""),  # Clarification text (if action_type is "clarify")
            "sub_questions": result.get("sub_questions", []),  # List of simpler questions (if action_type is "decompose")
            "summary": result.get("summary", ""),  # Summary if student understands
            "decomposition_failed": result.get("decomposition_failed", False),  # True if generate_sub_questions threw
            "decomposition_error": result.get("decomposition_error", "")  # Error message when decomposition failed
        })
        if _cost_limit_exceeded_after_add():
            resp.headers["X-Cost-Limit-Exceeded"] = "true"
        return resp

    except Exception as e:
        import traceback
        tb_str = traceback.format_exc()
        traceback.print_exc()
        # 写入日志文件便于排查
        try:
            (Path(__file__).parent / "tutor_error.log").write_text(
                f"=== {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n{tb_str}\n", encoding="utf-8"
            )
        except Exception:
            pass
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/evaluate_guidance_response', methods=['POST'])
def evaluate_guidance_response():
    """
    Evaluate student's response during iterative guidance loop.
    评估学生在迭代指导循环中的回答。
    
    This endpoint continues the guidance process:
    此端点继续指导过程：
    1. Evaluates student's response to sub-questions or clarifications
       评估学生对子问题或澄清的回答
    2. Determines if student understands
       确定学生是否理解
    3. If not understood: Decides next action (decompose/clarify)
       如果不理解：决定下一步行动（分解/澄清）
    4. If understood: Confirms understanding and provides summary
       如果理解：确认理解并提供总结
    
    Request Body / 请求体:
        {
            "original_question": "原始问题",
            "choices": {"A": "...", "B": "...", ...},
            "student_answer": "B",
            "correct_answer": "A",
            "current_action": "decompose" | "clarify",
            "current_sub_questions": ["question1", ...],
            "current_clarification": "clarification text",
            "student_response": "学生的回答",
            "conversation_history": [...],
            "understanding_level": "none" | "partial" | "close",
            "round_number": 1
        }
    
    Response / 响应:
        {
            "success": true,
            "understood": true/false,
            "understanding_level": "none" | "partial" | "close" | "understood",
            "feedback": "反馈消息",
            "next_action_type": "decompose" | "clarify" | null,
            "next_sub_questions": [...],  // If next_action_type is "decompose"
            "next_clarification": "...",  // If next_action_type is "clarify"
            "summary": "总结（如果理解了）"
        }
    """
    blocked = cost_check_limit()
    if blocked:
        return jsonify(blocked[0]), blocked[1]
    global hint_generator
    
    try:
        if hint_generator is None:
            if not initialize_hint_generator():
                return jsonify({
                    "success": False,
                    "error": "Hint generator not initialized"
                }), 500
        
        data = request.json
        
        from proactive_question_generator import HintRequest
        
        # Build request context
        # 构建请求上下文
        hint_request = HintRequest(
            question=data['original_question'],
            choices=data['choices'],
            student_answer=data['student_answer'],
            correct_answer=data['correct_answer']
        )
        
        # Call evaluation method
        # 调用评估方法
        usage_out = {}
        result = hint_generator.evaluate_guidance_response(
            request=hint_request,
            current_action=data.get('current_action'),
            current_sub_questions=data.get('current_sub_questions', []),
            current_clarification=data.get('current_clarification', ''),
            student_response=data['student_response'],
            conversation_history=data.get('conversation_history', []),
            current_understanding_level=data.get('understanding_level', 'partial'),
            round_number=data.get('round_number', 1),
            cannot_decompose_further=data.get('cannot_decompose_further', False),
            usage_out=usage_out
        )
        
        # 按实际 token 记录费用
        pt = usage_out.get("prompt_tokens", 0) or 0
        ct = usage_out.get("completion_tokens", 0) or 0
        model = usage_out.get("model", "gpt-4o")
        if pt or ct:
            cost_add(_get_user_id(), _estimate_chat_cost(pt, ct, model))
        else:
            cost_add(_get_user_id(), 0.02)
        resp = jsonify({
            "success": True,
            "understood": result.get("understood", False),
            "understanding_level": result.get("understanding_level", "partial"),
            "feedback": result.get("feedback", ""),
            "next_action_type": result.get("next_action_type"),
            "next_sub_questions": result.get("next_sub_questions", []),
            "next_clarification": result.get("next_clarification", ""),
            "summary": result.get("summary", "")
        })
        if _cost_limit_exceeded_after_add():
            resp.headers["X-Cost-Limit-Exceeded"] = "true"
        return resp
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/check_hint_trigger', methods=['POST'])
def check_hint_trigger():
    """
    检查是否应该触发提示生成。
    
    触发条件：
        1. 学生回答错误
        2. 学生请求提示（输入包含 "hint", "help", "提示" 等关键词）
    
    请求：
        POST /check_hint_trigger
        Content-Type: application/json
        {
            "student_answer": "B",
            "correct_answer": "A",
            "user_message": "Can you give me a hint?"  // 可选
        }
    
    响应：
        {
            "should_trigger": true,
            "reason": "wrong_answer" | "student_request" | "no_trigger"
        }
    """
    try:
        data = request.json
        
        student_answer = data.get('student_answer', '')
        correct_answer = data.get('correct_answer', '')
        user_message = data.get('user_message')
        
        # 检查触发条件
        should_trigger = False
        reason = "no_trigger"
        
        # 条件1：答案错误
        if student_answer and correct_answer and student_answer != correct_answer:
            should_trigger = True
            reason = "wrong_answer"
        
        # 条件2：用户请求提示
        if user_message:
            hint_keywords = [
                "hint", "help", "clue", "guide", "stuck",
                "don't understand", "confused", "explain",
                "提示", "帮助", "不懂", "不明白", "给我提示"
            ]
            message_lower = user_message.lower()
            if any(keyword in message_lower for keyword in hint_keywords):
                should_trigger = True
                reason = "student_request"
        
        return jsonify({
            "should_trigger": should_trigger,
            "reason": reason
        })
        
    except Exception as e:
        return jsonify({
            "should_trigger": False,
            "reason": "error",
            "error": str(e)
        }), 500


# =============================================================================
# 单端口模式：同时提供静态文件（UI）和 API
# =============================================================================

@app.route('/', methods=['GET'])
def index_redirect():
    """Redirect root to main quiz UI."""
    return redirect('/medical-quiz.html', code=302)


@app.route('/<path:path>', methods=['GET'])
def serve_static(path):
    """
    Serve static files (HTML, JS, JSON, etc.) from project root.
    Only GET; path must not contain '..'. Used when running in single-port mode.
    """
    if '..' in path or path.startswith('/'):
        return jsonify({"error": "Not found"}), 404
    base = _BASE_DIR
    full = os.path.normpath(os.path.join(base, path))
    if not os.path.realpath(full).startswith(os.path.realpath(base)):
        return jsonify({"error": "Not found"}), 404
    if not os.path.isfile(full):
        return jsonify({"error": "Not found"}), 404
    return send_from_directory(base, path)


# =============================================================================
# 主入口
# =============================================================================

def _load_user_totals_from_logs() -> None:
    """On startup, load persisted usage from user_logs/*.json into _user_totals."""
    if not os.path.isdir(USER_LOGS_DIR):
        return
    with _cost_lock:
        for name in os.listdir(USER_LOGS_DIR):
            if not name.endswith(".json"):
                continue
            path = os.path.join(USER_LOGS_DIR, name)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, IOError):
                continue
            usage = data.get("usage") if isinstance(data, dict) else None
            if not usage or not isinstance(usage, dict):
                continue
            username = name[:-5]  # strip .json
            uid = ("user:" + username)[:64]
            _user_totals[uid]["cost"] = float(usage.get("cost_total") or 0)
            _user_totals[uid]["prompt_tokens"] = int(usage.get("prompt_tokens") or 0)
            _user_totals[uid]["completion_tokens"] = int(usage.get("completion_tokens") or 0)
            _user_totals[uid]["request_count"] = int(usage.get("request_count") or 0)


if __name__ == '__main__':
    """
    启动 RAG 服务器。
    
    初始化：
        1. 加载 API 密钥
        2. 加载/构建 FAISS 索引
        3. 加载 ColBERTv2 重排序器
    
    服务器：
        - 主机：0.0.0.0（可从任何 IP 访问）
        - 端口：5000
        - 调试：False（为了生产稳定性）
    
    使用方法：
        python rag_server.py
    
    然后访问：
        - http://localhost:5000/health - 检查状态
        - http://localhost:5000/search - 搜索 API
    """
    _load_user_totals_from_logs()
    if initialize_rag():
        # 初始化提示生成器
        initialize_hint_generator()
        
        port = int(os.environ.get('RAG_PORT', 5000))
        print(f"\n服务器启动在 http://localhost:{port}")
        print("\nAPI 端点：")
        print("  POST /search          - 搜索相关块")
        print("       Body: {\"query\": \"你的问题\"}")
        print("  POST /generate_hints  - 生成苏格拉底式提示")
        print("       Body: {question, choices, student_answer, correct_answer}")
        print("  POST /check_hint_trigger - 检查是否触发提示")
        print("  GET  /health          - 健康检查")
        print("  POST /user_log/event   - 登录/登出记入该用户自己的日志")
        print("  POST /user_log/button - 记录按钮点击（user, timestamp, button, question_id?）")
        print("  POST /user_log/conversation - 记录 Tutor 对话（question_id, question_text, options, messages）")
        print("  POST /user_log/knowledge_profile - 记录知识画像")
        print("  GET  /user_log/usage?user= - 当前用户 token/费用")
        print("  GET  /user_log?user= - 当前用户完整日志")
        print("  POST /rebuild         - 从 PDF 重建索引")
        print("\n按 Ctrl+C 停止服务器\n")
        
        # 启动 Flask 服务器（支持 RAG_PORT 环境变量，用于 WSL 避让端口冲突）
        app.run(host='0.0.0.0', port=port, debug=False)
    else:
        print("\n[错误] RAG 系统初始化失败")
        print("请检查上面的错误信息")

