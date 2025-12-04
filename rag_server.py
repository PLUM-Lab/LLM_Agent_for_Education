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

import os
import json
import re
from pathlib import Path
from typing import List, Dict, Optional

# Flask Web 框架，用于 REST API
from flask import Flask, request, jsonify
from flask_cors import CORS  # 启用跨域资源共享

# 数值计算
import numpy as np

# OpenAI API 客户端，用于生成向量
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
try:
    from ragatouille import RAGPretrainedModel
    HAS_RERANKER = True
except ImportError:
    print("[!] RAGatouille 未安装，正在安装...")
    import subprocess
    subprocess.check_call(["pip", "install", "ragatouille"])
    try:
        from ragatouille import RAGPretrainedModel
        HAS_RERANKER = True
    except:
        HAS_RERANKER = False
        print("[!] 重排序器不可用，将仅使用 FAISS")



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
# 512 token 是上下文和精度之间的良好平衡
# 较小的块 = 更精确的检索，较大的块 = 更多上下文
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50  # 块之间的重叠，保持上下文连贯性

# 索引文件保存路径
INDEX_FILE = "faiss_index.bin"    # FAISS 向量索引
CHUNKS_FILE = "all_chunks.json"   # 所有块的元数据（文本、来源、页码）

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
CORS(app)

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
                content = config_path.read_text()
                # 使用正则表达式提取 API 密钥
                # 匹配：OPENAI_API_KEY = 'key' 或 OPENAI_API_KEY: 'key'
                match = re.search(r"OPENAI_API_KEY\s*[=:]\s*['\"]([^'\"]+)['\"]", content)
                if match:
                    api_key = match.group(1)
                    print("[OK] 已从 api-key.js 加载 API 密钥")
            except Exception as e:
                print(f"[!] 读取 api-key.js 出错：{e}")
    
    return api_key


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
            return final_results
            
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
    
    return results


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
    # 步骤 1：加载 ColBERTv2 重排序器
    # -------------------------------------------------------------------------
    if HAS_RERANKER:
        print("\n[0] 正在加载 ColBERTv2 重排序器...")
        try:
            # 加载预训练的 ColBERTv2 模型
            # 首次运行可能会下载模型（约 500MB）
            reranker = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
            print("  [OK] ColBERTv2 重排序器已加载")
        except Exception as e:
            print(f"  [!] 加载重排序器失败：{e}")
            reranker = None
    else:
        print("\n[!] 重排序器不可用")
        reranker = None
    
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
        results = search_similar(
            query, 
            openai_client, 
            faiss_index, 
            all_chunks, 
            num_retrieved=num_retrieved,
            num_final=num_final
        )
        
        return jsonify({
            "query": query,
            "num_retrieved": num_retrieved,
            "num_final": num_final,
            "reranker": "ColBERTv2" if reranker else "None",
            "results": results
        })
    
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
# 主入口
# =============================================================================

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
    if initialize_rag():
        print("\n服务器启动在 http://localhost:5000")
        print("\nAPI 端点：")
        print("  POST /search     - 搜索相关块")
        print("       Body: {\"query\": \"你的问题\"}")
        print("  GET  /health     - 健康检查")
        print("  POST /rebuild    - 从 PDF 重建索引")
        print("\n按 Ctrl+C 停止服务器\n")
        
        # 启动 Flask 服务器
        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        print("\n[错误] RAG 系统初始化失败")
        print("请检查上面的错误信息")

