import numpy as np
from typing import List, Dict, Optional, Tuple, Any
import asyncio
import aiofiles
from asyncio_throttle import Throttler
import json
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
import time
from sklearn.metrics.pairwise import cosine_similarity
import dashscope
from dashscope import TextEmbedding

from .code_chunker import CodeChunk


@dataclass
class EmbeddingResult:
    """嵌入向量结果"""
    chunk_hash: str
    embedding: List[float]
    model: str
    created_at: float
    metadata: Dict[str, Any]


class EmbeddingGenerator:
    """嵌入向量生成器"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "text-embedding-v2"):
        self.api_key = api_key or os.getenv('DASHSCOPE_API_KEY')
        if not self.api_key:
            raise ValueError("DashScope API key is required. Set DASHSCOPE_API_KEY environment variable.")
        
        dashscope.api_key = self.api_key
        self.model = model
        # 增加并发限制，允许更多并行请求
        self.semaphore = asyncio.Semaphore(100)  # 最多10个并发请求
        
        # 缓存
        self.embedding_cache: Dict[str, EmbeddingResult] = {}
        self.cache_file: Optional[Path] = None
    
    def _get_model_dimension(self) -> int:
        """根据模型获取嵌入向量维度"""
        model_dimensions = {
            'text-embedding-v1': 1536,
            'text-embedding-v2': 1536,
            'text-embedding-v3': 1024,
            'text-embedding-v4': 1024,
        }
        return model_dimensions.get(self.model, 1024)  # 默认1024维
    
    def set_cache_file(self, cache_file: Path):
        """设置缓存文件路径"""
        self.cache_file = cache_file
        self.load_cache()
    
    def load_cache(self):
        """从文件加载缓存"""
        if not self.cache_file or not self.cache_file.exists():
            return
        
        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            for chunk_hash, data in cache_data.items():
                self.embedding_cache[chunk_hash] = EmbeddingResult(**data)
                
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"Warning: Could not load embedding cache: {e}")
    
    def save_cache(self):
        """保存缓存到文件"""
        if not self.cache_file:
            return
        
        try:
            # 确保目录存在
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            
            cache_data = {}
            for chunk_hash, result in self.embedding_cache.items():
                cache_data[chunk_hash] = asdict(result)
            
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"Warning: Could not save embedding cache: {e}")
    
    async def generate_embedding(self, text: str, chunk_hash: Optional[str] = None) -> List[float]:
        """生成单个文本的嵌入向量"""
        # 检查缓存
        if chunk_hash and chunk_hash in self.embedding_cache:
            return self.embedding_cache[chunk_hash].embedding
        
        # 使用信号量控制并发数量，而不是严格的限流
        async with self.semaphore:
            try:
                # 添加小的延迟以避免过于频繁的请求
                await asyncio.sleep(0.1)
                
                response = await asyncio.to_thread(
                    TextEmbedding.call,
                    model=self.model,
                    input=text
                )
                
                if response.status_code != 200:
                    raise Exception(f"API request failed: {response.message}")
                
                # 处理不同的响应格式
                if hasattr(response.output, 'embeddings'):
                    # 对象格式
                    embedding = response.output.embeddings[0]
                elif isinstance(response.output, dict) and 'embeddings' in response.output:
                    # 字典格式
                    embeddings_data = response.output['embeddings']
                    
                    if isinstance(embeddings_data[0], dict) and 'embedding' in embeddings_data[0]:
                        embedding = embeddings_data[0]['embedding']
                    else:
                        # 可能直接就是嵌入向量列表
                        embedding = embeddings_data[0]
                else:
                    # 尝试其他可能的格式
                    print(f"Unexpected response format: {type(response.output)}")
                    print(f"Response output: {response.output}")
                    raise Exception(f"Unexpected response format from API")
                
                # 缓存结果
                if chunk_hash:
                    result = EmbeddingResult(
                        chunk_hash=chunk_hash,
                        embedding=embedding,
                        model=self.model,
                        created_at=time.time(),
                        metadata={'text_length': len(text)}
                    )
                    self.embedding_cache[chunk_hash] = result
                
                return embedding
                
            except Exception as e:
                print(f"Error generating embedding: {e}")
                # 返回零向量作为fallback，使用正确的模型维度
                return [0.0] * self._get_model_dimension()

    async def generate_embeddings_batch(self, chunks: List[CodeChunk], batch_size: int = 5) -> List[EmbeddingResult]:
        """批量生成嵌入向量 - 改进的并行处理"""
        results = []
        
        # 过滤已缓存的块
        chunks_to_process = []
        for chunk in chunks:
            if chunk.hash_value in self.embedding_cache:
                results.append(self.embedding_cache[chunk.hash_value])
            else:
                chunks_to_process.append(chunk)
        
        if not chunks_to_process:
            return results
        
        print(f"Generating embeddings for {len(chunks_to_process)} chunks with improved parallel processing...")
        
        # 分批处理，但每批内部真正并行
        for i in range(0, len(chunks_to_process), batch_size):
            batch = chunks_to_process[i:i + batch_size]
            
            # 准备批次任务 - 每个任务都是独立的协程
            batch_tasks = []
            chunk_task_mapping = []
            
            for chunk in batch:
                # 准备输入文本
                input_text = self._prepare_chunk_text(chunk)
                # 创建独立的协程任务
                task = asyncio.create_task(self.generate_embedding(input_text, chunk.hash_value))
                batch_tasks.append(task)
                chunk_task_mapping.append(chunk)
            
            print(f"Processing batch {i//batch_size + 1}: {len(batch_tasks)} parallel tasks")
            
            # 真正并行执行所有任务
            try:
                # 使用asyncio.gather实现真正的并行执行
                embeddings_batch = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                # 处理结果
                batch_results = []
                for j, (chunk, embedding) in enumerate(zip(chunk_task_mapping, embeddings_batch)):
                    if isinstance(embedding, Exception):
                        print(f"Error processing chunk {chunk.hash_value}: {embedding}")
                        continue
                    
                    try:
                        result = EmbeddingResult(
                            chunk_hash=chunk.hash_value,
                            embedding=embedding,
                            model=self.model,
                            created_at=time.time(),
                            metadata={
                                'file_path': chunk.file_path,
                                'chunk_type': chunk.chunk_type,
                                'language': chunk.language,
                                'start_line': chunk.start_line,
                                'end_line': chunk.end_line
                            }
                        )
                        batch_results.append(result)
                        
                        # 添加到缓存
                        self.embedding_cache[chunk.hash_value] = result
                        
                    except Exception as e:
                        print(f"Error creating EmbeddingResult for chunk {chunk.hash_value}: {e}")
                        continue
                
                results.extend(batch_results)
                
                # 显示进度
                processed = min(i + batch_size, len(chunks_to_process))
                print(f"Progress: {processed}/{len(chunks_to_process)} chunks processed ({len(batch_results)} successful)")
                
                # 定期保存缓存
                if len(batch_results) > 0:
                    self.save_cache()
                    
                # 批次间添加短暂延迟，避免过于频繁的请求
                if i + batch_size < len(chunks_to_process):
                    await asyncio.sleep(0.5)
                    
            except Exception as e:
                print(f"Error in batch processing: {e}")
                # 如果批处理失败，回退到逐个处理
                print("Falling back to sequential processing for this batch...")
                for chunk in batch:
                    try:
                        input_text = self._prepare_chunk_text(chunk)
                        embedding = await self.generate_embedding(input_text, chunk.hash_value)
                        
                        result = EmbeddingResult(
                            chunk_hash=chunk.hash_value,
                            embedding=embedding,
                            model=self.model,
                            created_at=time.time(),
                            metadata={
                                'file_path': chunk.file_path,
                                'chunk_type': chunk.chunk_type,
                                'language': chunk.language,
                                'start_line': chunk.start_line,
                                'end_line': chunk.end_line
                            }
                        )
                        results.append(result)
                        self.embedding_cache[chunk.hash_value] = result
                        
                    except Exception as chunk_error:
                        print(f"Error processing chunk {chunk.hash_value}: {chunk_error}")
                        continue
                
                # 保存缓存
                self.save_cache()
        
        return results
    
    def _prepare_chunk_text(self, chunk: CodeChunk) -> str:
        """准备用于嵌入的文本 - 增强版本"""
        try:
            # 基础上下文信息
            context_info = f"File: {chunk.file_path}\n"
            context_info += f"Language: {chunk.language}\n"
            context_info += f"Type: {chunk.chunk_type}\n"
            context_info += f"Lines: {chunk.start_line}-{chunk.end_line}\n"
            
            # 添加语义信息
            if hasattr(chunk, 'metadata') and chunk.metadata:
                # 函数/类名称
                if 'name' in chunk.metadata:
                    context_info += f"Name: {chunk.metadata['name']}\n"
                
                # 函数签名
                if 'signature' in chunk.metadata:
                    context_info += f"Signature: {chunk.metadata['signature']}\n"
                
                # 装饰器或修饰符
                if 'decorators' in chunk.metadata:
                    context_info += f"Decorators: {', '.join(chunk.metadata['decorators'])}\n"
                
                if 'modifiers' in chunk.metadata:
                    context_info += f"Modifiers: {', '.join(chunk.metadata['modifiers'])}\n"
            
            # 添加代码语义描述
            try:
                semantic_description = self._generate_semantic_description(chunk)
                if semantic_description:
                    context_info += f"Description: {semantic_description}\n"
            except Exception as e:
                print(f"Warning: Failed to generate semantic description: {e}")
            
            # 添加关键词提取
            try:
                keywords = self._extract_code_keywords(chunk.content)
                if keywords:
                    context_info += f"Keywords: {', '.join(keywords)}\n"
            except Exception as e:
                print(f"Warning: Failed to extract keywords: {e}")
            
            context_info += "\n"
            return context_info + chunk.content
            
        except Exception as e:
            print(f"Error in _prepare_chunk_text: {e}")
            # 回退到简单格式
            return f"File: {chunk.file_path}\nLanguage: {chunk.language}\nType: {chunk.chunk_type}\n\n{chunk.content}"
    
    def _generate_semantic_description(self, chunk: CodeChunk) -> str:
        """生成代码块的语义描述"""
        descriptions = []
        
        # 基于chunk类型生成描述
        if chunk.chunk_type == 'function_declaration':
            descriptions.append("函数定义")
        elif chunk.chunk_type == 'class_declaration':
            descriptions.append("类定义")
        elif chunk.chunk_type == 'method_definition':
            descriptions.append("方法定义")
        elif chunk.chunk_type == 'variable_declaration':
            descriptions.append("变量声明")
        elif chunk.chunk_type == 'for_statement':
            descriptions.append("循环语句")
        elif chunk.chunk_type == 'if_statement':
            descriptions.append("条件语句")
        elif chunk.chunk_type == 'try_statement':
            descriptions.append("异常处理")
        
        # 基于内容分析添加描述
        content_lower = chunk.content.lower()
        
        # 时间相关
        if any(keyword in content_lower for keyword in ['time', 'date', 'schedule', 'period', 'cycle', '时间', '周期', '训练']):
            descriptions.append("时间处理")
        
        # 数据处理
        if any(keyword in content_lower for keyword in ['data', 'process', 'transform', '数据', '处理']):
            descriptions.append("数据处理")
        
        # 配置相关
        if any(keyword in content_lower for keyword in ['config', 'setting', 'option', '配置', '设置']):
            descriptions.append("配置管理")
        
        # 训练相关
        if any(keyword in content_lower for keyword in ['train', 'model', 'epoch', 'batch', '训练', '模型']):
            descriptions.append("机器学习训练")
        
        return " ".join(descriptions)
    
    def _extract_code_keywords(self, content: str) -> List[str]:
        """从代码中提取关键词"""
        keywords = set()
        
        # 提取标识符（变量名、函数名等）
        identifiers = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', content)
        
        # 过滤常见关键字和短标识符
        common_keywords = {'if', 'else', 'for', 'while', 'def', 'class', 'import', 'from', 'return', 'try', 'except'}
        
        for identifier in identifiers:
            if len(identifier) > 2 and identifier.lower() not in common_keywords:
                # 分割驼峰命名
                camel_parts = re.findall(r'[A-Z][a-z]*|[a-z]+', identifier)
                if len(camel_parts) > 1:
                    keywords.update(part.lower() for part in camel_parts)
                else:
                    keywords.add(identifier.lower())
        
        # 提取字符串字面量中的关键词
        string_literals = re.findall(r'["\']([^"\']+)["\']', content)
        for literal in string_literals:
            if len(literal) > 3 and len(literal) < 50:  # 合理长度的字符串
                keywords.add(literal.lower())
        
        # 提取注释中的关键词
        comments = re.findall(r'#\s*(.+)', content)  # Python注释
        comments.extend(re.findall(r'//\s*(.+)', content))  # JavaScript注释
        
        for comment in comments:
            # 简单的中英文词汇提取
            words = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+', comment)
            for word in words:
                if len(word) > 2:
                    keywords.add(word.lower())
        
        return list(keywords)[:20]  # 限制关键词数量
    
    def search_similar_chunks(self, query: str, embeddings: List[EmbeddingResult], top_k: int = 10) -> List[Tuple[EmbeddingResult, float]]:
        """搜索相似的代码块"""
        if not embeddings:
            return []
        
        # 生成查询的嵌入向量
        query_embedding = asyncio.run(self.generate_embedding(query))
        query_vector = np.array(query_embedding).reshape(1, -1)
        
        # 计算相似度
        embedding_matrix = np.array([result.embedding for result in embeddings])
        similarities = cosine_similarity(query_vector, embedding_matrix)[0]
        
        # 排序并返回top_k结果
        indexed_similarities = list(enumerate(similarities))
        indexed_similarities.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for i, similarity in indexed_similarities[:top_k]:
            results.append((embeddings[i], similarity))
        
        return results
    
    def get_embedding_stats(self) -> Dict[str, Any]:
        """获取嵌入向量统计信息"""
        if not self.embedding_cache:
            return {}
        
        embeddings = list(self.embedding_cache.values())
        
        # 统计模型使用情况
        models = {}
        languages = {}
        chunk_types = {}
        
        for result in embeddings:
            models[result.model] = models.get(result.model, 0) + 1
            
            if 'language' in result.metadata:
                lang = result.metadata['language']
                languages[lang] = languages.get(lang, 0) + 1
            
            if 'chunk_type' in result.metadata:
                chunk_type = result.metadata['chunk_type']
                chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
        
        return {
            'total_embeddings': len(embeddings),
            'models': models,
            'languages': languages,
            'chunk_types': chunk_types,
            'cache_size_mb': len(json.dumps([asdict(e) for e in embeddings])) / (1024 * 1024)
        }
    
    def cleanup_old_embeddings(self, max_age_days: int = 30):
        """清理过期的嵌入向量"""
        current_time = time.time()
        max_age_seconds = max_age_days * 24 * 3600
        
        old_hashes = []
        for chunk_hash, result in self.embedding_cache.items():
            if current_time - result.created_at > max_age_seconds:
                old_hashes.append(chunk_hash)
        
        for chunk_hash in old_hashes:
            del self.embedding_cache[chunk_hash]
        
        if old_hashes:
            print(f"Cleaned up {len(old_hashes)} old embeddings")
            self.save_cache()


class VectorDatabase:
    """简单的向量数据库实现"""
    
    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.embeddings: List[EmbeddingResult] = []
        self.index_file = storage_path / "vector_index.json"
        
        # 确保存储目录存在
        storage_path.mkdir(parents=True, exist_ok=True)
        
        # 加载现有数据
        self.load_index()
    
    def add_embeddings(self, embeddings: List[EmbeddingResult]):
        """添加嵌入向量到数据库"""
        # 去重：移除已存在的嵌入向量
        existing_hashes = {e.chunk_hash for e in self.embeddings}
        new_embeddings = [e for e in embeddings if e.chunk_hash not in existing_hashes]
        
        self.embeddings.extend(new_embeddings)
        self.save_index()
        
        print(f"Added {len(new_embeddings)} new embeddings to database")
    
    def remove_embeddings(self, chunk_hashes: List[str]):
        """从数据库中移除嵌入向量"""
        hash_set = set(chunk_hashes)
        self.embeddings = [e for e in self.embeddings if e.chunk_hash not in hash_set]
        self.save_index()
        
        print(f"Removed {len(chunk_hashes)} embeddings from database")
    
    def search(self, query_embedding: List[float], top_k: int = 10, filter_metadata: Optional[Dict[str, Any]] = None) -> List[Tuple[EmbeddingResult, float]]:
        """搜索相似的嵌入向量"""
        if not self.embeddings:
            return []
        
        # 应用过滤器
        filtered_embeddings = self.embeddings
        if filter_metadata:
            filtered_embeddings = []
            for embedding in self.embeddings:
                match = True
                for key, value in filter_metadata.items():
                    if key not in embedding.metadata or embedding.metadata[key] != value:
                        match = False
                        break
                if match:
                    filtered_embeddings.append(embedding)
        
        if not filtered_embeddings:
            return []
        
        # 计算相似度
        query_vector = np.array(query_embedding).reshape(1, -1)
        embedding_matrix = np.array([e.embedding for e in filtered_embeddings])
        similarities = cosine_similarity(query_vector, embedding_matrix)[0]
        
        # 排序并返回结果
        indexed_similarities = list(enumerate(similarities))
        indexed_similarities.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for i, similarity in indexed_similarities[:top_k]:
            results.append((filtered_embeddings[i], similarity))
        
        return results
    
    def save_index(self):
        """保存索引到文件"""
        try:
            # 确保父目录存在
            self.index_file.parent.mkdir(parents=True, exist_ok=True)
            
            data = [asdict(e) for e in self.embeddings]
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving vector index: {e}")
            print(f"Index file path: {self.index_file}")
            print(f"Storage path: {self.storage_path}")
            # 尝试创建目录并重试
            try:
                self.storage_path.mkdir(parents=True, exist_ok=True)
                self.index_file.parent.mkdir(parents=True, exist_ok=True)
                data = [asdict(e) for e in self.embeddings]
                with open(self.index_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                print("Successfully saved index after creating directories")
            except Exception as retry_e:
                print(f"Failed to save index even after creating directories: {retry_e}")
    
    def load_index(self):
        """从文件加载索引"""
        if not self.index_file.exists():
            return
        
        try:
            with open(self.index_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.embeddings = [EmbeddingResult(**item) for item in data]
            print(f"Loaded {len(self.embeddings)} embeddings from index")
            
        except Exception as e:
            print(f"Error loading vector index: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取数据库统计信息"""
        if not self.embeddings:
            return {'total_embeddings': 0}
        
        languages = {}
        chunk_types = {}
        files = set()
        
        for embedding in self.embeddings:
            if 'language' in embedding.metadata:
                lang = embedding.metadata['language']
                languages[lang] = languages.get(lang, 0) + 1
            
            if 'chunk_type' in embedding.metadata:
                chunk_type = embedding.metadata['chunk_type']
                chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
            
            if 'file_path' in embedding.metadata:
                files.add(embedding.metadata['file_path'])
        
        return {
            'total_embeddings': len(self.embeddings),
            'total_files': len(files),
            'languages': languages,
            'chunk_types': chunk_types,
            'index_size_mb': self.index_file.stat().st_size / (1024 * 1024) if self.index_file.exists() else 0
        }