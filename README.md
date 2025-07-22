# Cursor-like Code Indexing Client

一个参考Cursor实现原理的代码库索引客户端，实现了默克尔树、代码分块、嵌入向量生成等核心功能。

## 功能特性

- 🌳 **默克尔树构建**: 高效的文件变更检测和增量更新
- 🔍 **智能代码分块**: 基于AST的语义感知代码分割
- 🧠 **嵌入向量生成**: 使用OpenAI API生成代码语义向量
- 🔄 **增量同步**: 只处理变更的文件，节省带宽和时间
- 🔐 **路径混淆**: 保护敏感文件路径信息
- 📊 **语义搜索**: 基于向量相似度的代码搜索
- 🚀 **FastAPI服务**: 提供REST API接口，支持Web服务调用

## 安装

```bash
pip install -r requirements.txt
```

## 配置

创建 `.env` 文件并设置OpenAI API密钥：

```
OPENAI_API_KEY=your_openai_api_key_here
```

## 使用方法

### FastAPI Web服务 (推荐)

1. 启动服务器：
```bash
python start_server.py
```

2. 访问API文档：
- 浏览器打开 http://127.0.0.1:8080 查看API概览
- 访问 http://127.0.0.1:8080/docs 查看详细API文档

3. 使用API客户端：
```python
# 运行示例客户端
python api_client_example.py
```

#### API接口说明

- `POST /index` - 索引项目代码库
- `POST /search` - 搜索代码
- `POST /context` - 获取代码上下文
- `GET /status` - 查看所有项目状态
- `GET /status/{project_path}` - 查看特定项目状态
- `POST /update/{project_path}` - 更新项目索引
- `GET /stats` - 获取统计信息
- `DELETE /delete/{project_path}` - 删除项目索引
- `POST /cleanup` - 清理存储空间
- `GET /config` - 获取配置
- `POST /config` - 设置配置

#### API使用示例

```python
import requests

# 索引项目
response = requests.post("http://127.0.0.1:8080/index", json={
    "project_path": "/path/to/your/project",
    "force": False,
    "max_tokens": 1000
})

# 搜索代码
response = requests.post("http://127.0.0.1:8080/search", json={
    "query": "function to handle user authentication",
    "top_k": 10
})

# 获取代码上下文
response = requests.post("http://127.0.0.1:8080/context", json={
    "query": "how to handle errors",
    "max_chunks": 5
})
```

### 命令行接口

```bash
# 索引代码库
python main.py index /path/to/your/project

# 搜索代码
python main.py search "function to handle user authentication"

# 检查变更并更新索引
python main.py update /path/to/your/project
```

### 编程接口

```python
from src.client import CodeIndexClient

# 创建客户端
client = CodeIndexClient()

# 索引项目
await client.index_project("/path/to/project")

# 搜索代码
results = await client.search("authentication function")

# 获取相关代码块
context = await client.get_context_for_query("how to handle errors")
```

## 项目结构

```
├── src/
│   ├── merkle_tree.py      # 默克尔树实现
│   ├── code_chunker.py     # 代码分块器
│   ├── embeddings.py       # 嵌入向量生成
│   ├── client.py          # 主客户端类
│   ├── storage.py         # 本地存储管理
│   └── utils.py           # 工具函数
├── tests/                 # 测试文件
├── main.py               # 命令行入口
├── app.py                # FastAPI应用
├── start_server.py       # 服务器启动脚本
├── api_client_example.py # API客户端示例
└── requirements.txt      # 依赖列表
```

## 技术原理

本项目参考了Cursor的核心技术实现：

1. **默克尔树**: 用于高效检测文件变更，实现增量更新
2. **AST分块**: 基于抽象语法树的智能代码分割
3. **嵌入向量**: 使用OpenAI嵌入模型生成代码语义表示
4. **向量搜索**: 通过余弦相似度进行语义代码搜索
5. **REST API**: 基于FastAPI提供高性能Web服务接口

## 注意事项

- 需要有效的OpenAI API密钥
- 大型项目首次索引可能需要较长时间
- 建议在.gitignore中添加索引缓存目录
- FastAPI服务默认运行在 http://127.0.0.1:8080