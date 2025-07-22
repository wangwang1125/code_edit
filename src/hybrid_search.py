"""
混合搜索引擎 - 结合向量搜索、关键词匹配和重排序
参考Claude Code的多层搜索策略
"""

import re
import math
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import Counter
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .embeddings import EmbeddingResult, EmbeddingGenerator
from .query_processor import QueryProcessor, QueryAnalysis


@dataclass
class SearchResult:
    """搜索结果"""
    embedding_result: EmbeddingResult
    vector_score: float
    keyword_score: float
    semantic_score: float
    final_score: float
    match_reasons: List[str]


class HybridSearchEngine:
    """混合搜索引擎"""
    
    def __init__(self, embedding_generator: EmbeddingGenerator):
        self.embedding_generator = embedding_generator
        self.query_processor = QueryProcessor()
        
        # 搜索权重配置
        self.weights = {
            'vector_similarity': 0.4,
            'keyword_match': 0.3,
            'semantic_match': 0.2,
            'type_match': 0.1
        }
    
    async def search(self, query: str, embeddings: List[EmbeddingResult], 
                    top_k: int = 10, filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """执行混合搜索"""
        
        # 1. 查询分析
        query_analysis = self.query_processor.analyze_query(query)
        
        # 2. 应用过滤器
        filtered_embeddings = self._apply_filters(embeddings, filters)
        
        if not filtered_embeddings:
            return []
        
        # 3. 向量相似度搜索
        vector_results = await self._vector_search(query_analysis, filtered_embeddings)
        
        # 4. 关键词匹配
        keyword_results = self._keyword_search(query_analysis, filtered_embeddings)
        
        # 5. 语义匹配
        semantic_results = self._semantic_search(query_analysis, filtered_embeddings)
        
        # 6. 结果融合和重排序
        final_results = self._merge_and_rerank(
            query_analysis, vector_results, keyword_results, semantic_results, filtered_embeddings
        )
        
        return final_results[:top_k]
    
    def _apply_filters(self, embeddings: List[EmbeddingResult], 
                      filters: Optional[Dict[str, Any]]) -> List[EmbeddingResult]:
        """应用搜索过滤器"""
        if not filters:
            return embeddings
        
        filtered = []
        for embedding in embeddings:
            match = True
            
            for key, value in filters.items():
                if key not in embedding.metadata:
                    match = False
                    break
                
                if isinstance(value, list):
                    if embedding.metadata[key] not in value:
                        match = False
                        break
                else:
                    if embedding.metadata[key] != value:
                        match = False
                        break
            
            if match:
                filtered.append(embedding)
        
        return filtered
    
    async def _vector_search(self, query_analysis: QueryAnalysis, 
                           embeddings: List[EmbeddingResult]) -> Dict[str, float]:
        """向量相似度搜索"""
        # 生成查询向量
        query_embedding = await self.embedding_generator.generate_embedding(
            query_analysis.expanded_query
        )
        query_vector = np.array(query_embedding).reshape(1, -1)
        
        # 计算相似度
        embedding_matrix = np.array([e.embedding for e in embeddings])
        similarities = cosine_similarity(query_vector, embedding_matrix)[0]
        
        # 返回结果字典
        results = {}
        for i, embedding in enumerate(embeddings):
            results[embedding.chunk_hash] = similarities[i]
        
        return results
    
    def _keyword_search(self, query_analysis: QueryAnalysis, 
                       embeddings: List[EmbeddingResult]) -> Dict[str, float]:
        """关键词匹配搜索"""
        results = {}
        
        # 准备查询关键词
        query_keywords = set()
        query_keywords.update(query_analysis.keywords)
        query_keywords.update(query_analysis.programming_terms)
        query_keywords.update(query_analysis.natural_language_terms)
        
        for embedding in embeddings:
            score = self._calculate_keyword_score(query_keywords, embedding)
            results[embedding.chunk_hash] = score
        
        return results
    
    def _calculate_keyword_score(self, query_keywords: set, 
                               embedding: EmbeddingResult) -> float:
        """计算关键词匹配分数"""
        if not query_keywords:
            return 0.0
        
        # 从元数据中提取文本内容
        text_content = ""
        
        # 文件路径
        if 'file_path' in embedding.metadata:
            text_content += embedding.metadata['file_path'] + " "
        
        # 函数/类名称
        if 'name' in embedding.metadata:
            text_content += embedding.metadata['name'] + " "
        
        # 函数签名
        if 'signature' in embedding.metadata:
            text_content += embedding.metadata['signature'] + " "
        
        # 关键词
        if 'keywords' in embedding.metadata:
            text_content += " ".join(embedding.metadata['keywords']) + " "
        
        # 描述
        if 'description' in embedding.metadata:
            text_content += embedding.metadata['description'] + " "
        
        text_content = text_content.lower()
        
        # 计算匹配分数
        matches = 0
        total_keywords = len(query_keywords)
        
        for keyword in query_keywords:
            if keyword.lower() in text_content:
                matches += 1
                
                # 精确匹配给更高分数
                if f" {keyword.lower()} " in f" {text_content} ":
                    matches += 0.5
        
        return matches / total_keywords if total_keywords > 0 else 0.0
    
    def _semantic_search(self, query_analysis: QueryAnalysis, 
                        embeddings: List[EmbeddingResult]) -> Dict[str, float]:
        """语义匹配搜索"""
        results = {}
        
        for embedding in embeddings:
            score = self._calculate_semantic_score(query_analysis, embedding)
            results[embedding.chunk_hash] = score
        
        return results
    
    def _calculate_semantic_score(self, query_analysis: QueryAnalysis, 
                                embedding: EmbeddingResult) -> float:
        """计算语义匹配分数"""
        score = 0.0
        
        # 查询类型匹配
        chunk_type = embedding.metadata.get('chunk_type', '')
        
        if query_analysis.query_type.value == 'function' and 'function' in chunk_type:
            score += 0.3
        elif query_analysis.query_type.value == 'class' and 'class' in chunk_type:
            score += 0.3
        elif query_analysis.query_type.value == 'concept':
            # 概念搜索，检查描述匹配
            description = embedding.metadata.get('description', '')
            if any(term in description for term in query_analysis.natural_language_terms):
                score += 0.4
        
        # 编程语言匹配
        if 'language' in embedding.metadata:
            # 如果查询中包含特定语言关键词
            lang = embedding.metadata['language'].lower()
            query_text = query_analysis.original_query.lower()
            if lang in query_text:
                score += 0.2
        
        # 意图匹配
        for intent in query_analysis.intent_keywords:
            if intent == 'modify' and 'function' in chunk_type:
                score += 0.2
            elif intent == 'find' and any(kw in chunk_type for kw in ['function', 'class', 'method']):
                score += 0.1
        
        return min(score, 1.0)
    
    def _merge_and_rerank(self, query_analysis: QueryAnalysis,
                         vector_results: Dict[str, float],
                         keyword_results: Dict[str, float],
                         semantic_results: Dict[str, float],
                         embeddings: List[EmbeddingResult]) -> List[SearchResult]:
        """合并结果并重新排序"""
        
        # 创建chunk_hash到embedding的映射
        embedding_map = {emb.chunk_hash: emb for emb in embeddings}
        
        # 收集所有chunk_hash
        all_hashes = set()
        all_hashes.update(vector_results.keys())
        all_hashes.update(keyword_results.keys())
        all_hashes.update(semantic_results.keys())
        
        # 计算最终分数
        final_results = []
        
        for chunk_hash in all_hashes:
            if chunk_hash not in embedding_map:
                continue
                
            embedding = embedding_map[chunk_hash]
            
            vector_score = vector_results.get(chunk_hash, 0.0)
            keyword_score = keyword_results.get(chunk_hash, 0.0)
            semantic_score = semantic_results.get(chunk_hash, 0.0)
            
            # 加权计算最终分数
            final_score = (
                vector_score * self.weights['vector_similarity'] +
                keyword_score * self.weights['keyword_match'] +
                semantic_score * self.weights['semantic_match']
            )
            
            # 生成匹配原因
            match_reasons = self._generate_match_reasons(
                query_analysis, embedding, vector_score, keyword_score, semantic_score
            )
            
            result = SearchResult(
                embedding_result=embedding,
                vector_score=vector_score,
                keyword_score=keyword_score,
                semantic_score=semantic_score,
                final_score=final_score,
                match_reasons=match_reasons
            )
            
            final_results.append(result)
        
        # 按最终分数排序
        final_results.sort(key=lambda x: x.final_score, reverse=True)
        
        return final_results
    
    def _generate_match_reasons(self, query_analysis: QueryAnalysis, 
                              embedding: EmbeddingResult,
                              vector_score: float, keyword_score: float, 
                              semantic_score: float) -> List[str]:
        """生成匹配原因说明"""
        reasons = []
        
        if vector_score > 0.7:
            reasons.append("高向量相似度匹配")
        elif vector_score > 0.5:
            reasons.append("中等向量相似度匹配")
        
        if keyword_score > 0.5:
            reasons.append("关键词匹配")
        
        if semantic_score > 0.3:
            reasons.append("语义类型匹配")
        
        # 具体匹配内容
        chunk_type = embedding.metadata.get('chunk_type', '')
        if 'function' in chunk_type and query_analysis.query_type.value == 'function':
            reasons.append("函数类型匹配")
        
        if 'name' in embedding.metadata:
            name = embedding.metadata['name']
            for keyword in query_analysis.keywords:
                if keyword.lower() in name.lower():
                    reasons.append(f"名称包含关键词: {keyword}")
        
        return reasons
    
    def explain_search_results(self, query: str, results: List[SearchResult]) -> Dict[str, Any]:
        """解释搜索结果"""
        query_analysis = self.query_processor.analyze_query(query)
        
        explanation = {
            'query_analysis': {
                'original_query': query_analysis.original_query,
                'query_type': query_analysis.query_type.value,
                'keywords': query_analysis.keywords,
                'programming_terms': query_analysis.programming_terms,
                'confidence': query_analysis.confidence
            },
            'search_strategy': {
                'weights': self.weights,
                'total_results': len(results)
            },
            'top_results': []
        }
        
        for i, result in enumerate(results[:5]):
            result_info = {
                'rank': i + 1,
                'final_score': result.final_score,
                'component_scores': {
                    'vector': result.vector_score,
                    'keyword': result.keyword_score,
                    'semantic': result.semantic_score
                },
                'match_reasons': result.match_reasons,
                'metadata': result.embedding_result.metadata
            }
            explanation['top_results'].append(result_info)
        
        return explanation