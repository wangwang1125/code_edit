import os
import hashlib
import shutil
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
import time
import asyncio
from cryptography.fernet import Fernet

from .merkle_tree import MerkleTree
from .code_chunker import CodeChunker, CodeChunk
from .embeddings import EmbeddingGenerator, VectorDatabase, EmbeddingResult
from .storage import StorageManager, ConfigManager, ProjectMetadata
from .smart_semantic_editor import SmartSemanticEditor as SemanticCodeEditor, SemanticEdit, SemanticEditType


class CodeIndexClient:
    """代码索引客户端 - 主要接口类"""
    
    def __init__(self, storage_path: Optional[Path] = None, config_path: Optional[Path] = None):
        # 设置默认路径
        if storage_path is None:
            storage_path = Path(".") / ".cursor_like_client" / "storage"
        if config_path is None:
            # 优先使用项目根目录的config.json
            project_config = Path(".") / "config.json"
            if project_config.exists():
                config_path = project_config
            else:
                config_path = Path(".") / ".cursor_like_client" / "config.json"
        
        # 初始化组件
        self.storage = StorageManager(storage_path)
        self.config = ConfigManager(config_path)
        
        # 初始化核心组件
        self.merkle_tree = MerkleTree()
        self.chunker = CodeChunker()
        
        # 延迟初始化（需要API密钥）
        self.embedding_generator: Optional[EmbeddingGenerator] = None
        self.vector_db: Optional[VectorDatabase] = None
        self.code_editor: Optional[SemanticCodeEditor] = None
        
        # 当前项目状态
        self.current_project_path: Optional[str] = None
        self.current_metadata: Optional[ProjectMetadata] = None
    
    def _ensure_embedding_generator(self):
        """确保嵌入生成器已初始化"""
        if self.embedding_generator is None:
            api_key = self.config.get('dashscope_api_key') or os.getenv('DASHSCOPE_API_KEY')
            if not api_key:
                raise ValueError("Dashscope API key is required. Set it in config or DASHSCOPE_API_KEY environment variable.")
            
            model = self.config.get('embedding_model', 'text-embedding-v2')
            self.embedding_generator = EmbeddingGenerator(api_key, model)
    
    def _ensure_code_editor(self):
        """确保代码编辑器已初始化"""
        if self.code_editor is None:
            api_key = self.config.get('dashscope_api_key') or os.getenv('DASHSCOPE_API_KEY')
            if not api_key:
                raise ValueError("Dashscope API key is required for code editing. Set it in config or DASHSCOPE_API_KEY environment variable.")
            
            model = self.config.get('ai_model', 'qwen-plus')
            backup_dir = self.storage.base_path / "backups"
            self.code_editor = SemanticCodeEditor(api_key, model, backup_dir)
    
    def _ensure_vector_db(self, project_path: str):
        """确保向量数据库已初始化"""
        if self.vector_db is None or self.current_project_path != project_path:
            vector_db_path = self.storage.get_vector_db_path(project_path)
            
            print(f"vector_db_path: {vector_db_path}")
            self.vector_db = VectorDatabase(vector_db_path)
            self.current_project_path = project_path
    
    def _generate_obfuscation_key(self, project_path: str) -> bytes:
        """生成项目特定的混淆密钥"""
        # 使用项目路径和一些固定字符串生成密钥
        key_material = f"cursor_like_client_{project_path}".encode()
        key_hash = hashlib.sha256(key_material).digest()
        return Fernet.generate_key()  # 实际应用中可以基于key_hash生成
    
    async def index_project(self, project_path: str, force_rebuild: bool = False) -> Dict[str, Any]:
        """索引项目"""
        project_path = str(Path(project_path).resolve())
        print(f"Starting to index project: {project_path}")
        
        # 检查项目是否存在
        if not Path(project_path).exists():
            raise ValueError(f"Project path does not exist: {project_path}")
        
        # 加载现有元数据
        existing_metadata = self.storage.load_project_metadata(project_path)
        
        # 构建默克尔树
        print("Building Merkle tree...")
        file_extensions = self.config.get('supported_file_types')
        new_tree = MerkleTree()
        
        # 设置混淆密钥
        obfuscation_key = self._generate_obfuscation_key(project_path)
        new_tree.set_obfuscation_key(obfuscation_key)
        
        new_tree.build_tree(Path(project_path), file_extensions)
        
        # 检查是否需要重新索引
        need_full_reindex = force_rebuild
        changed_files = []
        
        if existing_metadata and not force_rebuild:
            # 加载旧的默克尔树
            old_tree_path = self.storage.get_merkle_tree_path(project_path)
            old_tree = MerkleTree()
            
            if old_tree.load_from_file(old_tree_path):
                changed_files = new_tree.get_changed_files(old_tree)
                if not changed_files:
                    print("No changes detected. Index is up to date.")
                    return {
                        'status': 'up_to_date',
                        'project_path': project_path,
                        'total_files': existing_metadata.total_files,
                        'total_chunks': existing_metadata.total_chunks
                    }
                print(f"Detected {len(changed_files)} changed files")
            else:
                need_full_reindex = True
        else:
            need_full_reindex = True
        
        # 代码分块
        print("Chunking code...")
        max_tokens = self.config.get('max_tokens_per_chunk', 1000)
        
        if need_full_reindex:
            # 全量分块
            chunks = self.chunker.chunk_project(project_path, file_extensions, max_tokens)
        else:
            # 增量分块：只处理变更的文件
            chunks = []
            for file_path in changed_files:
                full_file_path = Path(project_path) / file_path
                if full_file_path.exists():
                    file_chunks = self.chunker.chunk_file(str(full_file_path), max_tokens)
                    # 更新相对路径
                    for chunk in file_chunks:
                        chunk.file_path = file_path
                    chunks.extend(file_chunks)
        
        print(f"Generated {len(chunks)} code chunks")
        
        # 生成嵌入向量
        self._ensure_embedding_generator()
        self._ensure_vector_db(project_path)
        # 设置缓存路径
        cache_path = self.storage.get_embeddings_cache_path(project_path)
        self.embedding_generator.set_cache_file(cache_path)
        
        print("Generating embeddings...")
        batch_size = self.config.get('batch_size', 10)
        embeddings = await self.embedding_generator.generate_embeddings_batch(chunks, batch_size)
        
        # 更新向量数据库
        if need_full_reindex:
            # 清空旧数据
            self.vector_db.embeddings.clear()
        else:
            # 移除变更文件的旧嵌入向量
            old_chunks = self.storage.load_chunks(project_path)
            old_hashes_to_remove = []
            
            for old_chunk in old_chunks:
                if old_chunk.get('file_path') in changed_files:
                    old_hashes_to_remove.append(old_chunk.get('hash_value'))
            
            if old_hashes_to_remove:
                self.vector_db.remove_embeddings(old_hashes_to_remove)
        
        # 添加新的嵌入向量
        self.vector_db.add_embeddings(embeddings)
        
        # 保存数据
        print("Saving index data...")
        
        # 保存默克尔树
        merkle_tree_path = self.storage.get_merkle_tree_path(project_path)
        new_tree.save_to_file(merkle_tree_path)
        
        # 保存代码块
        if need_full_reindex:
            all_chunks = chunks
        else:
            # 合并新旧代码块
            old_chunks_data = self.storage.load_chunks(project_path)
            old_chunks = [CodeChunk(**data) for data in old_chunks_data if data.get('file_path') not in changed_files]
            all_chunks = old_chunks + chunks
        
        self.storage.save_chunks(project_path, all_chunks)
        
        # 保存项目元数据
        metadata = ProjectMetadata(
            project_path=project_path,
            project_name=Path(project_path).name,
            last_indexed=time.time(),
            total_files=len(new_tree.file_hashes),
            total_chunks=len(all_chunks),
            merkle_root_hash=new_tree.get_root_hash(),
            obfuscation_key=obfuscation_key.decode() if obfuscation_key else None,
            settings={
                'file_extensions': file_extensions,
                'max_tokens_per_chunk': max_tokens,
                'embedding_model': self.config.get('embedding_model')
            }
        )
        
        self.storage.save_project_metadata(project_path, metadata)
        self.current_metadata = metadata
        
        print("Indexing completed successfully!")
        
        return {
            'status': 'completed',
            'project_path': project_path,
            'total_files': metadata.total_files,
            'total_chunks': metadata.total_chunks,
            'changed_files': len(changed_files) if changed_files else metadata.total_files,
            'embeddings_generated': len(embeddings)
        }
    
    async def search(self, query: str, project_path: Optional[str] = None, top_k: int = 10, 
                    filter_language: Optional[str] = None, filter_file_type: Optional[str] = None,
                    use_hybrid_search: bool = True) -> List[Dict[str, Any]]:
        """搜索代码 - 增强版本"""
        if project_path is None:
            project_path = self.current_project_path
        
        if not project_path:
            raise ValueError("No project specified. Please provide project_path or index a project first.")
        
        self._ensure_embedding_generator()
        self._ensure_vector_db(project_path)
        
        # 准备过滤器
        filters = {}
        if filter_language:
            filters['language'] = filter_language
        if filter_file_type:
            filters['chunk_type'] = filter_file_type
        
        if use_hybrid_search:
            try:
                # 使用混合搜索引擎
                from .hybrid_search import HybridSearchEngine
                hybrid_engine = HybridSearchEngine(self.embedding_generator)
                search_results = await hybrid_engine.search(
                    query=query,
                    embeddings=self.vector_db.embeddings,
                    top_k=top_k,
                    filters=filters
                )
                # 转换为原有格式
                results = []
                for search_result in search_results:
                    embedding = search_result.embedding_result
                    result = {
                        'similarity': float(search_result.final_score),
                        'file_path': embedding.metadata.get('file_path', ''),
                        'chunk_type': embedding.metadata.get('chunk_type', ''),
                        'language': embedding.metadata.get('language', ''),
                        'start_line': embedding.metadata.get('start_line', 0),
                        'end_line': embedding.metadata.get('end_line', 0),
                        'chunk_hash': embedding.chunk_hash,
                        'match_reasons': search_result.match_reasons,
                        'component_scores': {
                            'vector_score': search_result.vector_score,
                            'keyword_score': search_result.keyword_score,
                            'semantic_score': search_result.semantic_score
                        }
                    }
                    results.append(result)
                
                return results
            except Exception as e:
                print(f"混合搜索失败，回退到传统搜索: {e}")
                # 回退到传统搜索
                return await self._legacy_search(query, project_path, top_k, filters)
        else:
            # 使用原有的向量搜索
            return await self._legacy_search(query, project_path, top_k, filters)
    
    async def _legacy_search(self, query: str, project_path: str, top_k: int, 
                           filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """原有的向量搜索方法"""
        # 生成查询的嵌入向量
        query_embedding = await self.embedding_generator.generate_embedding(query)
        
        # 搜索相似的嵌入向量
        results = self.vector_db.search(query_embedding, top_k, filters)
        
        # 格式化结果
        formatted_results = []
        for embedding_result, similarity in results:
            result = {
                'similarity': float(similarity),
                'file_path': embedding_result.metadata.get('file_path', ''),
                'chunk_type': embedding_result.metadata.get('chunk_type', ''),
                'language': embedding_result.metadata.get('language', ''),
                'start_line': embedding_result.metadata.get('start_line', 0),
                'end_line': embedding_result.metadata.get('end_line', 0),
                'chunk_hash': embedding_result.chunk_hash
            }
            formatted_results.append(result)
        
        return formatted_results
    
    async def get_context_for_query(self, query: str, project_path: Optional[str] = None, 
                                   max_context_chunks: int = 5) -> Dict[str, Any]:
        """获取查询相关的代码上下文"""
        search_results = await self.search(query, project_path, max_context_chunks)
        
        if not search_results:
            return {'context': '', 'sources': []}
        
        # 读取实际的代码内容
        context_parts = []
        sources = []
        
        if project_path is None:
            project_path = self.current_project_path
        
        for result in search_results:
            file_path = Path(project_path) / result['file_path']
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                start_line = max(0, result['start_line'] - 1)
                end_line = min(len(lines), result['end_line'])
                
                code_content = ''.join(lines[start_line:end_line])
                
                context_part = f"# File: {result['file_path']} (Lines {result['start_line']}-{result['end_line']})\n"
                context_part += f"# Language: {result['language']}, Type: {result['chunk_type']}\n"
                context_part += f"# Similarity: {result['similarity']:.3f}\n\n"
                context_part += code_content + "\n\n"
                
                context_parts.append(context_part)
                sources.append(result)
                
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
                continue
        
        return {
            'context': ''.join(context_parts),
            'sources': sources,
            'total_chunks': len(sources)
        }
    
    def get_context_from_search_results(self, search_results: List[Dict[str, Any]], 
                                      project_path: Optional[str] = None) -> Dict[str, Any]:
        """基于搜索结果获取代码上下文"""
        if not search_results:
            return {'context': '', 'sources': []}
        
        # 读取实际的代码内容
        context_parts = []
        sources = []
        
        if project_path is None:
            project_path = self.current_project_path
        
        for result in search_results:
            file_path = Path(project_path) / result['file_path']
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                start_line = max(0, result['start_line'] - 1)
                end_line = min(len(lines), result['end_line'])
                
                code_content = ''.join(lines[start_line:end_line])
                
                context_part = f"# File: {result['file_path']} (Lines {result['start_line']}-{result['end_line']})\n"
                context_part += f"# Language: {result['language']}, Type: {result['chunk_type']}\n"
                context_part += f"# Similarity: {result['similarity']:.3f}\n\n"
                context_part += code_content + "\n\n"
                
                context_parts.append(context_part)
                sources.append(result)
                
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
                continue
        
        return {
            'context': ''.join(context_parts),
            'sources': sources,
            'total_chunks': len(sources)
        }
    
    def get_project_status(self, project_path: str) -> Dict[str, Any]:
        """获取项目状态"""
        metadata = self.storage.load_project_metadata(project_path)
        
        if not metadata:
            return {'status': 'not_indexed', 'project_path': project_path}
        
        # 检查是否需要更新
        merkle_tree_path = self.storage.get_merkle_tree_path(project_path)
        current_tree = MerkleTree()
        current_tree.build_tree(Path(project_path), metadata.settings.get('file_extensions'))
        
        needs_update = False
        if current_tree.get_root_hash() != metadata.merkle_root_hash:
            needs_update = True
        
        return {
            'status': 'indexed',
            'project_path': project_path,
            'project_name': metadata.project_name,
            'last_indexed': metadata.last_indexed,
            'total_files': metadata.total_files,
            'total_chunks': metadata.total_chunks,
            'needs_update': needs_update,
            'settings': metadata.settings
        }
    
    def list_projects(self) -> List[Dict[str, Any]]:
        """列出所有已索引的项目"""
        return self.storage.list_indexed_projects()
    
    def delete_project_index(self, project_path: str) -> bool:
        """删除项目索引"""
        return self.storage.delete_project_data(project_path)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取客户端统计信息"""
        storage_stats = self.storage.get_storage_stats()
        
        stats = {
            'storage': storage_stats,
            'config': {
                'embedding_model': self.config.get('embedding_model'),
                'max_tokens_per_chunk': self.config.get('max_tokens_per_chunk'),
                'supported_extensions': self.config.get('supported_file_types')
            }
        }
        
        # 如果有当前项目，添加项目统计
        if self.current_project_path and self.vector_db:
            stats['current_project'] = {
                'path': self.current_project_path,
                'vector_db_stats': self.vector_db.get_stats()
            }
        
        return stats
    
    async def update_project_index(self, project_path: str) -> Dict[str, Any]:
        """更新项目索引（增量更新）"""
        return await self.index_project(project_path, force_rebuild=False)
    
    def cleanup_storage(self, max_age_days: int = 30):
        """清理存储空间"""
        # 清理临时文件
        self.storage.cleanup_temp_files(24)  # 24小时
        
        # 清理过期的嵌入向量缓存
        if self.embedding_generator:
            self.embedding_generator.cleanup_old_embeddings(max_age_days)
    
    # ==================== 代码编辑功能 ====================
    
    async def search_and_analyze_edit(self, 
                                    query: str, 
                                    project_path: Optional[str] = None, 
                                    top_k: int = 10,
                                    filter_language: Optional[str] = None, 
                                    filter_file_type: Optional[str] = None,
                                    use_hybrid_search: bool = True) -> Dict[str, Any]:
        """搜索代码并直接分析语义编辑请求 - 合并操作避免重复"""
        if project_path is None:
            project_path = self.current_project_path
        
        if not project_path:
            raise ValueError("No project specified. Please provide project_path or index a project first.")
        
        # 1. 执行搜索
        print(f"搜索代码: {query}")
        search_results = await self.search(
            query=query, 
            project_path=project_path, 
            top_k=top_k,
            filter_language=filter_language,
            filter_file_type=filter_file_type,
            use_hybrid_search=use_hybrid_search
        )
        
        # 2. 如果有搜索结果，直接进行语义编辑分析
        analysis_result = {
            'search_results': search_results,
            'search_count': len(search_results),
            'edits': [],
            'analysis_success': False,
            'error': None
        }
        
        if search_results:
            try:
                print(f"基于 {len(search_results)} 个搜索结果分析语义编辑...")
                edits = await self.analyze_code_modification(
                    request=query,
                    search_results=search_results,
                    project_path=project_path
                )
                
                analysis_result.update({
                    'edits': [edit.to_dict() for edit in edits] if edits else [],
                    'analysis_success': len(edits) > 0,
                    'edit_count': len(edits)
                })
                
                if edits:
                    # 计算平均置信度
                    avg_confidence = sum(edit.confidence for edit in edits) / len(edits)
                    analysis_result['average_confidence'] = avg_confidence
                    
                    print(f"生成了 {len(edits)} 个语义编辑操作，平均置信度: {avg_confidence:.2f}")
                    for i, edit in enumerate(edits, 1):
                        print(f"  编辑 {i}: {edit.description}")
                        print(f"    类型: {edit.edit_type.value}")
                        print(f"    置信度: {edit.confidence:.2f}")
                        if edit.location:
                            print(f"    位置: {edit.location.symbol_name} ({edit.location.start_line}-{edit.location.end_line}行)")
                else:
                    print("未生成有效的语义编辑操作")
                    analysis_result['error'] = "无法生成有效的语义编辑操作"
                    
            except Exception as e:
                print(f"语义编辑分析失败: {e}")
                analysis_result.update({
                    'analysis_success': False,
                    'error': str(e)
                })
        else:
            print("没有找到相关的代码，无法进行语义编辑分析")
            analysis_result['error'] = "没有找到相关的代码"
        
        return analysis_result
    
    async def search_and_edit(self, 
                            query: str, 
                            project_path: Optional[str] = None, 
                            top_k: int = 10,
                            filter_language: Optional[str] = None, 
                            filter_file_type: Optional[str] = None,
                            use_hybrid_search: bool = True,
                            auto_apply: bool = False,
                            confidence_threshold: float = 0.7,
                            generate_patch: bool = False) -> Dict[str, Any]:
        """搜索代码并生成编辑建议 - 支持差异补丁模式"""
        # 1. 先进行搜索和分析
        analysis_result = await self.search_and_analyze_edit(
            query=query,
            project_path=project_path,
            top_k=top_k,
            filter_language=filter_language,
            filter_file_type=filter_file_type,
            use_hybrid_search=use_hybrid_search
        )
        
        # 2. 构建基础结果
        edit_result = {
            'search_results': analysis_result['search_results'],
            'search_count': analysis_result['search_count'],
            'edits': analysis_result['edits'],
            'analysis_success': analysis_result['analysis_success'],
            'edit_applied': False,
            'applied_edits': [],
            'error': analysis_result.get('error'),
            'warnings': [],
            'patches': []  # 新增：差异补丁列表
        }
        
        if analysis_result['analysis_success'] and analysis_result['edits']:
            avg_confidence = analysis_result.get('average_confidence', 0)
            
            # 3. 生成差异补丁（如果启用）
            if generate_patch:
                try:
                    patches = await self.generate_diff_patches(
                        query=query,
                        search_results=analysis_result['search_results'],
                        project_path=project_path
                    )
                    edit_result['patches'] = patches
                    print(f"生成了 {len(patches)} 个差异补丁")
                except Exception as e:
                    edit_result['warnings'].append(f"生成差异补丁失败: {str(e)}")
                    print(f"生成差异补丁失败: {e}")
            
            # 4. 自动应用编辑（如果启用且满足条件）
            if auto_apply and avg_confidence >= confidence_threshold and not generate_patch:
                try:
                    print(f"置信度 {avg_confidence:.2f} >= {confidence_threshold}，自动应用编辑...")
                    
                    # 执行编辑
                    edits, apply_result = await self.edit_code_with_plan(
                        request=query,
                        search_results=analysis_result['search_results'],
                        project_path=project_path,
                        auto_apply=True
                    )
                    
                    if apply_result and apply_result['success']:
                        edit_result.update({
                            'edit_applied': True,
                            'applied_edits': apply_result['applied_edits'],
                            'backup_path': apply_result.get('backup_path'),
                            'diff': apply_result.get('diff')
                        })
                        print(f"成功应用了 {len(apply_result['applied_edits'])} 个编辑操作")
                    else:
                        edit_result['error'] = apply_result.get('errors', ['编辑应用失败'])[0] if apply_result else '编辑应用失败'
                        print(f"编辑应用失败: {edit_result['error']}")
                        
                except Exception as e:
                    edit_result['error'] = f"编辑执行失败: {str(e)}"
                    print(f"编辑执行失败: {e}")
            else:
                if generate_patch:
                    edit_result['warnings'].append("已生成差异补丁，等待用户确认应用")
                elif not auto_apply:
                    edit_result['warnings'].append("未启用自动应用，需要手动确认编辑")
                else:
                    edit_result['warnings'].append(f"置信度 {avg_confidence:.2f} < {confidence_threshold}，跳过自动应用")
                print(f"跳过自动应用编辑 (auto_apply={auto_apply}, confidence={avg_confidence:.2f}, generate_patch={generate_patch})")
        
        return edit_result
    
    async def analyze_code_modification(self, 
                                      request: str, 
                                      search_results: List[Dict[str, Any]], 
                                      project_path: Optional[str] = None) -> List[SemanticEdit]:
        """分析代码修改请求，生成语义编辑操作"""
        self._ensure_code_editor()
        
        if project_path is None:
            project_path = self.current_project_path
        
        if not project_path:
            raise ValueError("必须指定项目路径")
        
        # 获取代码上下文
        import json
        context_data = self.get_context_from_search_results(search_results, project_path)
        context = context_data['context']
        
        # 确定主要文件信息
        if not search_results:
            raise ValueError("需要提供搜索结果以确定修改目标")
        
        # 使用第一个搜索结果作为主要目标文件
        main_result = search_results[0]
        file_path = str(Path(project_path) / main_result['file_path'])
        
        # 调用语义编辑器的edit_code方法（不自动应用）
        edits, success = await self.code_editor.edit_code(
            request=request, 
            file_path=file_path,
            cursor_position=None,  # 暂时不使用光标位置
            selection=context[:2000] if context else None,  # 将上下文作为选中内容传递，限制长度
            auto_apply=False  # 只分析，不应用
        )
        
        return edits if success else []
    
    async def edit_code_with_plan(self, 
                                request: str, 
                                search_results: List[Dict[str, Any]], 
                                project_path: Optional[str] = None,
                                auto_apply: bool = False) -> Tuple[List[SemanticEdit], Optional[Dict[str, Any]]]:
        """基于搜索结果编辑代码"""
        self._ensure_code_editor()
        
        if project_path is None:
            project_path = self.current_project_path
        
        if not project_path:
            raise ValueError("必须指定项目路径")
        
        # 获取代码上下文
        context_data = self.get_context_from_search_results(search_results, project_path)
        context = context_data['context']
        
        # 确定主要文件信息
        if not search_results:
            raise ValueError("需要提供搜索结果以确定修改目标")
        
        # 使用第一个搜索结果作为主要目标文件
        main_result = search_results[0]
        file_path = str(Path(project_path) / main_result['file_path'])
        
        # 调用语义编辑器
        # edit_code返回 Tuple[List[SemanticEdit], bool]
        edits, success = await self.code_editor.edit_code(
            request=request, 
            file_path=file_path,
            cursor_position=None,  # 暂时不使用光标位置
            selection=context[:2000] if context else None,  # 将上下文作为选中内容传递，限制长度
            auto_apply=auto_apply
        )
        
        # 构建结果字典
        result = None
        if auto_apply:
            result = {
                'success': success,
                'applied_edits': edits if success else [],
                'errors': [] if success else ['编辑应用失败'],
                'warnings': [],
                'backup_path': None,  # 备份路径由编辑器内部管理
                'diff': None
            }
            
            # 如果成功应用了修改，更新项目索引
            if success and edits:
                try:
                    # 异步更新索引（不阻塞）
                    asyncio.create_task(self._update_modified_files_index([file_path], project_path))
                except Exception as e:
                    print(f"更新索引时出错: {e}")
        
        return edits, result
    
    async def generate_diff_patches(self, 
                                  query: str, 
                                  search_results: List[Dict], 
                                  project_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """生成差异补丁而不直接修改文件"""
        patches = []
        
        try:
            # 1. 分析代码修改需求
            edits = await self.analyze_code_modification(
                request=query,
                search_results=search_results,
                project_path=project_path
            )
            
            if not edits:
                return patches
            
            # 2. 为每个编辑操作生成差异补丁
            for edit in edits:
                try:
                    # 读取原始文件内容
                    file_path = edit.location.file_path
                    if not os.path.exists(file_path):
                        continue
                    
                    with open(file_path, 'r', encoding='utf-8') as f:
                        original_content = f.read()
                    
                    # 模拟应用编辑操作，生成修改后的内容
                    modified_content = await self._simulate_edit_application(
                        original_content, edit
                    )
                    
                    # 生成差异补丁
                    import difflib
                    diff_lines = list(difflib.unified_diff(
                        original_content.splitlines(keepends=True),
                        modified_content.splitlines(keepends=True),
                        fromfile=f"a/{os.path.basename(file_path)}",
                        tofile=f"b/{os.path.basename(file_path)}",
                        lineterm=''
                    ))
                    
                    if diff_lines:
                        patch_info = {
                            'file_path': file_path,
                            'edit_type': edit.edit_type.value,
                            'description': edit.description,
                            'confidence': edit.confidence,
                            'diff': ''.join(diff_lines),
                            'original_content': original_content,
                            'modified_content': modified_content,
                            'line_range': {
                                'start': edit.location.start_line,
                                'end': edit.location.end_line
                            }
                        }
                        patches.append(patch_info)
                        
                except Exception as e:
                    print(f"生成文件 {edit.location.file_path} 的差异补丁失败: {e}")
                    continue
            
            return patches
            
        except Exception as e:
            print(f"生成差异补丁失败: {e}")
            return patches

    async def _simulate_edit_application(self, original_content: str, edit) -> str:
        """模拟应用编辑操作，返回修改后的内容"""
        from .smart_semantic_editor import SmartSemanticEditor
        
        # 创建临时编辑器实例
        editor = SmartSemanticEditor()
        
        # 将内容写入临时文件
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as temp_file:
            temp_file.write(original_content)
            temp_path = temp_file.name
        
        try:
            # 应用编辑操作 - 注意：apply_edit不是异步方法，返回bool
            success = editor.apply_edit(temp_path, edit)
            
            # 读取修改后的内容
            if success:
                with open(temp_path, 'r', encoding='utf-8') as f:
                    modified_content = f.read()
                return modified_content
            else:
                # 如果智能编辑失败，尝试简单的文本替换
                return self._apply_simple_edit(original_content, edit)
                
        finally:
            # 清理临时文件
            try:
                os.unlink(temp_path)
            except:
                pass

    def _apply_simple_edit(self, content: str, edit) -> str:
        """简单的编辑应用，用作备选方案"""
        lines = content.splitlines()
        
        try:
            if edit.edit_type.value == 'replace':
                # 替换指定行范围
                start_idx = max(0, edit.location.start_line - 1)
                end_idx = min(len(lines), edit.location.end_line)
                
                new_lines = lines[:start_idx] + edit.new_content.splitlines() + lines[end_idx:]
                return '\n'.join(new_lines)
                
            elif edit.edit_type.value == 'insert':
                # 在指定位置插入
                insert_idx = max(0, edit.location.start_line - 1)
                new_lines = lines[:insert_idx] + edit.new_content.splitlines() + lines[insert_idx:]
                return '\n'.join(new_lines)
                
            elif edit.edit_type.value == 'delete':
                # 删除指定行范围
                start_idx = max(0, edit.location.start_line - 1)
                end_idx = min(len(lines), edit.location.end_line)
                
                new_lines = lines[:start_idx] + lines[end_idx:]
                return '\n'.join(new_lines)
                
        except Exception as e:
            print(f"简单编辑应用失败: {e}")
            
        return content  # 如果失败，返回原始内容

    async def apply_diff_patch(self, patch_info: Dict[str, Any], create_backup: bool = True) -> Dict[str, Any]:
        """应用差异补丁到文件"""
        result = {
            'success': False,
            'file_path': patch_info['file_path'],
            'backup_path': None,
            'error': None
        }
        
        try:
            file_path = patch_info['file_path']
            
            # 1. 创建备份
            if create_backup:
                backup_path = f"{file_path}.backup_{int(time.time())}"
                shutil.copy2(file_path, backup_path)
                result['backup_path'] = backup_path
            
            # 2. 应用修改
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(patch_info['modified_content'])
            
            result['success'] = True
            print(f"成功应用差异补丁到文件: {file_path}")
            
        except Exception as e:
            result['error'] = str(e)
            print(f"应用差异补丁失败: {e}")
            
        return result

    async def apply_multiple_patches(self, patches: List[Dict[str, Any]], create_backup: bool = True) -> Dict[str, Any]:
        """批量应用多个差异补丁"""
        results = {
            'success': True,
            'applied_patches': [],
            'failed_patches': [],
            'backup_paths': []
        }
        
        for patch in patches:
            result = await self.apply_diff_patch(patch, create_backup)
            
            if result['success']:
                results['applied_patches'].append(patch['file_path'])
                if result['backup_path']:
                    results['backup_paths'].append(result['backup_path'])
            else:
                results['failed_patches'].append({
                    'file_path': patch['file_path'],
                    'error': result['error']
                })
                results['success'] = False
        
        return results

    def apply_semantic_edits(self, edits: List[SemanticEdit]) -> Dict[str, Any]:
        """应用语义编辑操作"""
        if not self.code_editor:
            raise ValueError("代码编辑器未初始化")
        
        if not edits:
            return {
                'success': False,
                'applied_edits': [],
                'errors': ['没有提供编辑操作'],
                'warnings': [],
                'backup_path': None,
                'diff': None
            }
        
        # 获取文件路径（假设所有编辑都在同一个文件中）
        file_path = edits[0].location.file_path
        
        # 应用编辑
        success = self.code_editor.apply_edits(edits, file_path)
        
        # 构建结果
        result = {
            'success': success,
            'applied_edits': edits if success else [],
            'errors': [] if success else ['编辑应用失败'],
            'warnings': [],
            'backup_path': None,  # 备份路径由编辑器内部管理
            'diff': None
        }
        
        # 如果成功应用了修改，更新项目索引
        if success and edits:
            try:
                project_path = self.current_project_path
                if project_path:
                    # 获取修改的文件列表
                    modified_files = list(set(edit.location.file_path for edit in edits))
                    asyncio.create_task(self._update_modified_files_index(modified_files, project_path))
            except Exception as e:
                print(f"更新索引时出错: {e}")
        
        return result
    
    def get_edit_history(self) -> List[Dict[str, Any]]:
        """获取编辑历史"""
        if not self.code_editor:
            return []
        return self.code_editor.get_edit_history()
    
    def rollback_last_edit(self) -> bool:
        """回滚最后一次编辑"""
        if not self.code_editor:
            return False
        
        success = self.code_editor.rollback_last_edit()
        
        # 如果回滚成功，更新项目索引
        if success and self.current_project_path:
            try:
                asyncio.create_task(self.update_project_index(self.current_project_path))
            except Exception as e:
                print(f"更新索引时出错: {e}")
        
        return success
    
    async def _update_modified_files_index(self, modified_files: List[str], project_path: str):
        """更新修改文件的索引"""
        try:
            if modified_files:
                print(f"更新 {len(modified_files)} 个修改文件的索引...")
                # 这里可以实现增量更新逻辑
                # 暂时使用全量更新
                await self.update_project_index(project_path)
        except Exception as e:
            print(f"更新索引失败: {e}")