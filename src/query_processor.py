"""
查询处理器 - 智能查询分析和预处理
参考Claude Code的查询理解机制
"""

import re
import jieba
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class QueryType(Enum):
    """查询类型"""
    FUNCTION_SEARCH = "function"      # 查找函数
    CLASS_SEARCH = "class"           # 查找类
    VARIABLE_SEARCH = "variable"     # 查找变量
    CONCEPT_SEARCH = "concept"       # 概念搜索
    IMPLEMENTATION_SEARCH = "implementation"  # 实现搜索
    BUG_SEARCH = "bug"              # 错误搜索
    FEATURE_SEARCH = "feature"       # 功能搜索
    GENERAL_SEARCH = "general"       # 通用搜索


@dataclass
class QueryAnalysis:
    """查询分析结果"""
    original_query: str
    query_type: QueryType
    keywords: List[str]
    programming_terms: List[str]
    natural_language_terms: List[str]
    intent_keywords: List[str]
    expanded_query: str
    confidence: float


class QueryProcessor:
    """智能查询处理器"""
    
    def __init__(self):
        # 编程相关关键词
        self.programming_keywords = {
            'function', 'method', 'class', 'variable', 'constant', 'interface',
            'module', 'package', 'import', 'export', 'return', 'parameter',
            'argument', 'loop', 'condition', 'if', 'else', 'for', 'while',
            'try', 'catch', 'exception', 'error', 'debug', 'test', 'api',
            'database', 'query', 'insert', 'update', 'delete', 'select',
            'async', 'await', 'promise', 'callback', 'event', 'handler',
            '函数', '方法', '类', '变量', '常量', '接口', '模块', '包',
            '导入', '导出', '返回', '参数', '循环', '条件', '异常', '错误',
            '调试', '测试', '接口', '数据库', '查询', '插入', '更新', '删除',
            '异步', '回调', '事件', '处理器'
        }
        
        # 意图关键词
        self.intent_keywords = {
            'find': ['找', '查找', '搜索', 'find', 'search', 'locate'],
            'create': ['创建', '新建', '添加', 'create', 'add', 'new'],
            'modify': ['修改', '更改', '编辑', 'modify', 'change', 'edit', 'update'],
            'delete': ['删除', '移除', 'delete', 'remove'],
            'fix': ['修复', '解决', '修正', 'fix', 'solve', 'repair'],
            'implement': ['实现', '开发', 'implement', 'develop'],
            'optimize': ['优化', '改进', 'optimize', 'improve'],
            'debug': ['调试', '排错', 'debug', 'troubleshoot']
        }
        
        # 时间相关词汇
        self.time_keywords = {
            '秒', '分钟', '小时', '天', '周', '月', '年',
            'second', 'minute', 'hour', 'day', 'week', 'month', 'year',
            '训练', '周期', '时间', '期间', '间隔', 'training', 'cycle', 'period', 'interval'
        }
        
        # 初始化jieba分词
        jieba.initialize()
    
    def analyze_query(self, query: str) -> QueryAnalysis:
        """分析查询意图和内容"""
        # 基础清理
        cleaned_query = self._clean_query(query)
        
        # 分词
        tokens = self._tokenize(cleaned_query)
        
        # 识别查询类型
        query_type = self._identify_query_type(tokens, cleaned_query)
        
        # 提取关键词
        keywords = self._extract_keywords(tokens)
        programming_terms = self._extract_programming_terms(tokens)
        natural_language_terms = self._extract_natural_language_terms(tokens)
        intent_keywords = self._extract_intent_keywords(tokens)
        
        # 扩展查询
        expanded_query = self._expand_query(cleaned_query, keywords, programming_terms)
        
        # 计算置信度
        confidence = self._calculate_confidence(query_type, keywords, programming_terms)
        
        return QueryAnalysis(
            original_query=query,
            query_type=query_type,
            keywords=keywords,
            programming_terms=programming_terms,
            natural_language_terms=natural_language_terms,
            intent_keywords=intent_keywords,
            expanded_query=expanded_query,
            confidence=confidence
        )
    
    def _clean_query(self, query: str) -> str:
        """清理查询文本"""
        # 移除多余空格
        query = re.sub(r'\s+', ' ', query.strip())
        
        # 标准化标点符号
        query = re.sub(r'[，。！？；：]', ' ', query)
        
        return query
    
    def _tokenize(self, query: str) -> List[str]:
        """分词处理"""
        # 中文分词
        chinese_tokens = list(jieba.cut(query))
        
        # 英文分词（简单空格分割）
        english_tokens = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', query)
        
        # 数字提取
        numbers = re.findall(r'\d+', query)
        
        # 合并所有token
        all_tokens = chinese_tokens + english_tokens + numbers
        
        # 过滤空token
        return [token.strip() for token in all_tokens if token.strip()]
    
    def _identify_query_type(self, tokens: List[str], query: str) -> QueryType:
        """识别查询类型"""
        query_lower = query.lower()
        
        # 函数搜索
        if any(keyword in query_lower for keyword in ['function', 'method', '函数', '方法']):
            return QueryType.FUNCTION_SEARCH
        
        # 类搜索
        if any(keyword in query_lower for keyword in ['class', 'object', '类', '对象']):
            return QueryType.CLASS_SEARCH
        
        # 变量搜索
        if any(keyword in query_lower for keyword in ['variable', 'var', '变量']):
            return QueryType.VARIABLE_SEARCH
        
        # 实现搜索
        if any(keyword in query_lower for keyword in ['implement', 'how to', '如何', '怎么', '实现']):
            return QueryType.IMPLEMENTATION_SEARCH
        
        # 错误搜索
        if any(keyword in query_lower for keyword in ['error', 'bug', 'fix', '错误', '修复', '问题']):
            return QueryType.BUG_SEARCH
        
        # 功能搜索
        if any(keyword in query_lower for keyword in ['feature', 'functionality', '功能', '特性']):
            return QueryType.FEATURE_SEARCH
        
        # 概念搜索（包含时间、训练等概念）
        if any(keyword in tokens for keyword in self.time_keywords):
            return QueryType.CONCEPT_SEARCH
        
        return QueryType.GENERAL_SEARCH
    
    def _extract_keywords(self, tokens: List[str]) -> List[str]:
        """提取关键词"""
        keywords = []
        
        for token in tokens:
            # 过滤停用词
            if len(token) > 1 and token not in ['的', '了', '是', '在', '有', '和', '与']:
                keywords.append(token)
        
        return keywords
    
    def _extract_programming_terms(self, tokens: List[str]) -> List[str]:
        """提取编程相关术语"""
        programming_terms = []
        
        for token in tokens:
            if token.lower() in self.programming_keywords:
                programming_terms.append(token)
        
        return programming_terms
    
    def _extract_natural_language_terms(self, tokens: List[str]) -> List[str]:
        """提取自然语言术语"""
        natural_terms = []
        
        for token in tokens:
            if token.lower() not in self.programming_keywords and len(token) > 1:
                # 中文词汇或英文单词
                if re.match(r'[\u4e00-\u9fff]+', token) or re.match(r'[a-zA-Z]+', token):
                    natural_terms.append(token)
        
        return natural_terms
    
    def _extract_intent_keywords(self, tokens: List[str]) -> List[str]:
        """提取意图关键词"""
        intent_keywords = []
        
        for intent, keywords in self.intent_keywords.items():
            for token in tokens:
                if token.lower() in keywords:
                    intent_keywords.append(intent)
                    break
        
        return intent_keywords
    
    def _expand_query(self, query: str, keywords: List[str], programming_terms: List[str]) -> str:
        """扩展查询"""
        expanded_parts = [query]
        
        # 添加同义词
        synonyms = self._get_synonyms(keywords + programming_terms)
        if synonyms:
            expanded_parts.extend(synonyms)
        
        # 添加相关编程概念
        related_concepts = self._get_related_concepts(programming_terms)
        if related_concepts:
            expanded_parts.extend(related_concepts)
        
        return ' '.join(expanded_parts)
    
    def _get_synonyms(self, terms: List[str]) -> List[str]:
        """获取同义词"""
        synonym_map = {
            '训练': ['training', 'train', '学习', 'learning'],
            '周期': ['cycle', 'period', '时间', 'time', '间隔', 'interval'],
            '月': ['month', '30天', '30 days'],
            '修改': ['modify', 'change', 'update', 'edit'],
            '函数': ['function', 'method', 'func'],
            '类': ['class', 'object'],
            '变量': ['variable', 'var'],
        }
        
        synonyms = []
        for term in terms:
            if term in synonym_map:
                synonyms.extend(synonym_map[term])
        
        return synonyms
    
    def _get_related_concepts(self, programming_terms: List[str]) -> List[str]:
        """获取相关编程概念"""
        concept_map = {
            'training': ['model', 'epoch', 'batch', 'learning_rate'],
            'cycle': ['loop', 'iteration', 'repeat'],
            'time': ['duration', 'timeout', 'delay'],
        }
        
        related = []
        for term in programming_terms:
            if term.lower() in concept_map:
                related.extend(concept_map[term.lower()])
        
        return related
    
    def _calculate_confidence(self, query_type: QueryType, keywords: List[str], 
                            programming_terms: List[str]) -> float:
        """计算查询分析的置信度"""
        confidence = 0.5  # 基础置信度
        
        # 有编程术语增加置信度
        if programming_terms:
            confidence += 0.2
        
        # 关键词数量影响置信度
        if len(keywords) >= 3:
            confidence += 0.2
        elif len(keywords) >= 2:
            confidence += 0.1
        
        # 查询类型明确性
        if query_type != QueryType.GENERAL_SEARCH:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def generate_search_variants(self, analysis: QueryAnalysis) -> List[str]:
        """生成搜索变体"""
        variants = [analysis.original_query]
        
        # 基于关键词的变体
        if analysis.keywords:
            variants.append(' '.join(analysis.keywords))
        
        # 基于编程术语的变体
        if analysis.programming_terms:
            variants.append(' '.join(analysis.programming_terms))
        
        # 基于自然语言术语的变体
        if analysis.natural_language_terms:
            variants.append(' '.join(analysis.natural_language_terms))
        
        # 扩展查询
        if analysis.expanded_query != analysis.original_query:
            variants.append(analysis.expanded_query)
        
        return list(set(variants))  # 去重