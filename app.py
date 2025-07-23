#!/usr/bin/env python3
"""
Cursor-like Code Indexing FastAPI Server
基于FastAPI的代码索引服务
"""

import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, Body
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.client import CodeIndexClient
from src.utils import validate_project_path, get_project_info, format_file_size, format_duration

# 创建FastAPI应用
app = FastAPI(
    title="Code Index API",
    description="Cursor-like 代码索引服务",
    version="1.0.0"
)

# 全局客户端实例
client = None

# Pydantic模型定义
class ProjectIndexRequest(BaseModel):
    project_path: str = Field(..., description="项目路径")
    force: bool = Field(False, description="是否强制重建索引")
    max_tokens: int = Field(1000, description="每个代码块的最大token数")

class SearchRequest(BaseModel):
    query: str = Field(..., description="搜索查询")
    project_path: Optional[str] = Field(None, description="项目路径（可选）")
    top_k: int = Field(10, description="返回结果数量")
    language: Optional[str] = Field(None, description="过滤编程语言")

class ContextRequest(BaseModel):
    query: Optional[str] = Field(None, description="查询内容（当不提供search_results时必需）")
    project_path: Optional[str] = Field(None, description="项目路径（可选）")
    max_chunks: int = Field(5, description="最大上下文块数")
    search_results: Optional[List[Dict]] = Field(None, description="搜索结果（可选，如果提供则基于此获取上下文）")

class ConfigRequest(BaseModel):
    key: str = Field(..., description="配置键")
    value: str = Field(..., description="配置值")

class SearchAndAnalyzeRequest(BaseModel):
    query: str = Field(..., description="搜索和编辑查询")
    project_path: Optional[str] = Field(None, description="项目路径（可选）")
    top_k: int = Field(10, description="返回结果数量")
    filter_language: Optional[str] = Field(None, description="过滤编程语言")
    filter_file_type: Optional[str] = Field(None, description="过滤文件类型")
    use_hybrid_search: bool = Field(True, description="是否使用混合搜索")

class SearchAndEditRequest(BaseModel):
    query: str = Field(..., description="搜索和编辑查询")
    project_path: Optional[str] = Field(None, description="项目路径（可选）")
    top_k: int = Field(10, description="返回结果数量")
    filter_language: Optional[str] = Field(None, description="过滤编程语言")
    filter_file_type: Optional[str] = Field(None, description="过滤文件类型")
    use_hybrid_search: bool = Field(True, description="是否使用混合搜索")
    auto_apply: bool = Field(False, description="是否自动应用编辑")
    confidence_threshold: float = Field(0.7, description="自动应用的置信度阈值")
    generate_patch: bool = Field(False, description="是否生成差异补丁而不直接修改文件")

class CodeEditRequest(BaseModel):
    request: str = Field(..., description="代码修改请求描述")
    search_results: List[Dict] = Field(..., description="搜索结果，用于确定修改目标")
    project_path: Optional[str] = Field(None, description="项目路径（可选）")
    auto_apply: bool = Field(False, description="是否自动应用修改")

class SemanticEditResponse(BaseModel):
    edit_type: str
    location: Dict[str, Any]
    old_code: Optional[str]
    new_code: str
    description: str
    confidence: float

class EditAnalysisResponse(BaseModel):
    edits: List[SemanticEditResponse]
    total_edits: int

class EditResultResponse(BaseModel):
    success: bool
    applied_edits: List[SemanticEditResponse]
    errors: List[str]
    warnings: List[str]
    backup_path: Optional[str]
    diff: Optional[str]

class IndexResponse(BaseModel):
    status: str
    message: str
    project_name: str
    total_files: int
    total_chunks: int
    changed_files: int
    duration: float

class SearchResult(BaseModel):
    file_path: str
    similarity: float
    language: str
    chunk_type: str
    start_line: int
    end_line: int
    content: Optional[str] = None

class ProjectStatus(BaseModel):
    project_name: str
    project_path: str
    status: str
    last_indexed: Optional[float]
    total_files: int
    total_chunks: int
    needs_update: bool

@app.on_event("startup")
async def startup_event():
    """应用启动时初始化客户端"""
    global client
    try:
        config_path = Path("./config.json")
        client = CodeIndexClient(None,config_path)
        print("Code Index Client initialized successfully")
    except Exception as e:
        print(f"Failed to initialize client: {e}")
        raise

@app.get("/", response_class=HTMLResponse)
async def root():
    """返回API文档页面"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Code Index API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 800px; margin: 0 auto; }
            .endpoint { background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }
            .method { color: #fff; padding: 3px 8px; border-radius: 3px; font-weight: bold; }
            .get { background: #61affe; }
            .post { background: #49cc90; }
            .delete { background: #f93e3e; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Code Index API</h1>
            <p>Cursor-like 代码索引服务 API 接口</p>
            
            <div class="endpoint">
                <span class="method post">POST</span> <strong>/index</strong>
                <p>索引项目代码库</p>
            </div>
            
            <div class="endpoint">
                <span class="method post">POST</span> <strong>/search</strong>
                <p>搜索代码</p>
            </div>
            
            <div class="endpoint">
                <span class="method post">POST</span> <strong>/search_and_analyze_edit</strong>
                <p>搜索代码并分析语义编辑 (新功能)</p>
            </div>
            
            <div class="endpoint">
                <span class="method post">POST</span> <strong>/search_and_edit</strong>
                <p>搜索代码并直接执行编辑 (新功能)</p>
            </div>
            
            <div class="endpoint">
                <span class="method post">POST</span> <strong>/context</strong>
                <p>获取查询相关的代码上下文</p>
            </div>
            
            <div class="endpoint">
                <span class="method post">POST</span> <strong>/edit/analyze</strong>
                <p>分析代码修改请求，生成编辑计划</p>
            </div>
            
            <div class="endpoint">
                <span class="method post">POST</span> <strong>/edit/apply</strong>
                <p>基于搜索结果编辑代码</p>
            </div>
            
            <div class="endpoint">
                <span class="method get">GET</span> <strong>/edit/history</strong>
                <p>获取编辑历史</p>
            </div>
            
            <div class="endpoint">
                <span class="method post">POST</span> <strong>/edit/rollback</strong>
                <p>回滚最后一次编辑</p>
            </div>
            
            <div class="endpoint">
                <span class="method get">GET</span> <strong>/status</strong>
                <p>查看所有项目状态</p>
            </div>
            
            <div class="endpoint">
                <span class="method get">GET</span> <strong>/status/{project_path:path}</strong>
                <p>查看特定项目状态</p>
            </div>
            
            <div class="endpoint">
                <span class="method post">POST</span> <strong>/update/{project_path:path}</strong>
                <p>更新项目索引</p>
            </div>
            
            <div class="endpoint">
                <span class="method get">GET</span> <strong>/stats</strong>
                <p>显示客户端统计信息</p>
            </div>
            
            <div class="endpoint">
                <span class="method delete">DELETE</span> <strong>/delete/{project_path:path}</strong>
                <p>删除项目索引</p>
            </div>
            
            <div class="endpoint">
                <span class="method post">POST</span> <strong>/cleanup</strong>
                <p>清理存储空间</p>
            </div>
            
            <div class="endpoint">
                <span class="method get">GET</span> <strong>/config</strong>
                <p>获取所有配置</p>
            </div>
            
            <div class="endpoint">
                <span class="method post">POST</span> <strong>/config</strong>
                <p>设置配置</p>
            </div>
            
            <p><a href="/docs">查看详细API文档</a></p>
        </div>
    </body>
    </html>
    """

@app.post("/index", response_model=IndexResponse)
async def index_project(request: ProjectIndexRequest, background_tasks: BackgroundTasks):
    """索引项目代码库"""
    if not client:
        raise HTTPException(status_code=500, detail="客户端未初始化")
    
    project_path = str(Path(request.project_path).resolve())
    
    # 验证项目路径
    if not validate_project_path(project_path):
        raise HTTPException(status_code=400, detail=f"无效的项目路径: {project_path}")
    
    # 获取项目信息
    project_info = get_project_info(project_path)
    
    # 设置配置
    if request.max_tokens != 1000:
        client.config.set('max_tokens_per_chunk', request.max_tokens)
    
    start_time = time.time()
    
    try:
        result = await client.index_project(project_path, force_rebuild=request.force)
        duration = time.time() - start_time
        
        return IndexResponse(
            status=result['status'],
            message=f"索引{result['status']}",
            project_name=project_info['name'],
            total_files=result['total_files'],
            total_chunks=result['total_chunks'],
            changed_files=result.get('changed_files', 0),
            duration=duration
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"索引失败: {str(e)}")

@app.post("/search")
async def search_code(request: SearchRequest):
    """搜索代码"""
    if not client:
        raise HTTPException(status_code=500, detail="客户端未初始化")
    
    project_path = None
    if request.project_path:
        project_path = str(Path(request.project_path).resolve())
    
    try:
        # 调用搜索方法，默认使用混合搜索
        results = await client.search(
            query=request.query,
            project_path=project_path,
            top_k=request.top_k,
            filter_language=request.language,
            use_hybrid_search=True  # 启用混合搜索
        )
        
        search_results = []
        for result in results:
            search_results.append(SearchResult(
                file_path=result['file_path'],
                similarity=result['similarity'],
                language=result['language'],
                chunk_type=result['chunk_type'],
                start_line=result['start_line'],
                end_line=result['end_line']
            ))
        
        return {
            "query": request.query,
            "total_results": len(search_results),
            "results": search_results
        }
    except Exception as e:
        import traceback
        print(f"搜索错误详情: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"搜索失败: {str(e)}")

@app.post("/search_and_analyze_edit")
async def search_and_analyze_edit(request: SearchAndAnalyzeRequest):
    """搜索代码并直接分析语义编辑请求 - 合并操作避免重复"""
    if not client:
        raise HTTPException(status_code=500, detail="客户端未初始化")
    
    project_path = None
    if request.project_path:
        project_path = str(Path(request.project_path).resolve())
    
    try:
        result = await client.search_and_analyze_edit(
            query=request.query,
            project_path=project_path,
            top_k=request.top_k,
            filter_language=request.filter_language,
            filter_file_type=request.filter_file_type,
            use_hybrid_search=request.use_hybrid_search
        )
        
        return result
        
    except Exception as e:
        import traceback
        print(f"搜索和分析错误详情: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"搜索和分析失败: {str(e)}")

@app.post("/search_and_edit")
async def search_and_edit(request: SearchAndEditRequest):
    """搜索代码并直接执行语义编辑 - 一站式操作"""
    if not client:
        raise HTTPException(status_code=500, detail="客户端未初始化")
    
    project_path = None
    if request.project_path:
        project_path = str(Path(request.project_path).resolve())
    
    try:
        result = await client.search_and_edit(
            query=request.query,
            project_path=project_path,
            top_k=request.top_k,
            filter_language=request.filter_language,
            filter_file_type=request.filter_file_type,
            use_hybrid_search=request.use_hybrid_search,
            auto_apply=request.auto_apply,
            confidence_threshold=request.confidence_threshold,
            generate_patch=request.generate_patch
        )
        
        return result
        
    except Exception as e:
        import traceback
        print(f"搜索和编辑错误详情: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"搜索和编辑失败: {str(e)}")

@app.post("/context")
async def get_context(request: ContextRequest):
    """获取查询相关的代码上下文"""
    if not client:
        raise HTTPException(status_code=500, detail="客户端未初始化")
    
    # 验证输入参数
    if not request.query and not request.search_results:
        raise HTTPException(status_code=400, detail="必须提供 query 或 search_results 中的一个")
    
    project_path = None
    if request.project_path:
        project_path = str(Path(request.project_path).resolve())
    
    try:
        # 模式1：基于搜索结果获取上下文（推荐）
        if request.search_results:
            # 限制搜索结果数量
            limited_results = request.search_results[:request.max_chunks]
            result = client.get_context_from_search_results(
                search_results=limited_results,
                project_path=project_path
            )
            
            return {
                "query": request.query or "基于搜索结果",
                "mode": "from_search_results",
                "total_chunks": result['total_chunks'],
                "context": result['context'],
                "sources": result['sources']
            }
        
        # 模式2：基于查询重新搜索
        else:
            result = await client.get_context_for_query(
                query=request.query,
                project_path=project_path,
                max_context_chunks=request.max_chunks
            )
            
            if not result or not result['context']:
                return {
                    "query": request.query,
                    "mode": "from_query",
                    "total_chunks": 0,
                    "context": "",
                    "sources": []
                }
            
            return {
                "query": request.query,
                "mode": "from_query",
                "total_chunks": result['total_chunks'],
                "context": result['context'],
                "sources": result['sources']
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取上下文失败: {str(e)}")

@app.get("/status")
async def get_all_projects_status():
    """查看所有项目状态"""
    if not client:
        raise HTTPException(status_code=500, detail="客户端未初始化")
    
    try:
        projects = client.list_projects()
        return {
            "total_projects": len(projects),
            "projects": projects
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取状态失败: {str(e)}")

@app.get("/status/{project_path:path}")
async def get_project_status(project_path: str):
    """查看特定项目状态"""
    if not client:
        raise HTTPException(status_code=500, detail="客户端未初始化")
    
    project_path = str(Path(project_path).resolve())
    
    try:
        project_status = client.get_project_status(project_path)
        
        if project_status['status'] == 'not_indexed':
            raise HTTPException(status_code=404, detail=f"项目未索引: {project_path}")
        
        return ProjectStatus(
            project_name=project_status['project_name'],
            project_path=project_status['project_path'],
            status=project_status['status'],
            last_indexed=project_status.get('last_indexed'),
            total_files=project_status.get('total_files', 0),
            total_chunks=project_status.get('total_chunks', 0),
            needs_update=project_status.get('needs_update', False)
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取项目状态失败: {str(e)}")

@app.post("/update/{project_path:path}")
async def update_project(project_path: str):
    """更新项目索引"""
    if not client:
        raise HTTPException(status_code=500, detail="客户端未初始化")
    
    project_path = str(Path(project_path).resolve())
    
    try:
        result = await client.update_project_index(project_path)
        return {
            "message": "更新完成",
            "project_path": project_path,
            "changed_files": result.get('changed_files', 0)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"更新失败: {str(e)}")

@app.get("/stats")
async def get_stats():
    """显示客户端统计信息"""
    if not client:
        raise HTTPException(status_code=500, detail="客户端未初始化")
    
    try:
        stats = client.get_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取统计信息失败: {str(e)}")

@app.delete("/delete/{project_path:path}")
async def delete_project(project_path: str, confirm: bool = Query(False, description="确认删除")):
    """删除项目索引"""
    if not client:
        raise HTTPException(status_code=500, detail="客户端未初始化")
    
    if not confirm:
        raise HTTPException(status_code=400, detail="请设置confirm=true确认删除操作")
    
    project_path = str(Path(project_path).resolve())
    
    try:
        success = client.delete_project_index(project_path)
        
        if success:
            return {"message": f"已删除项目索引: {project_path}"}
        else:
            raise HTTPException(status_code=404, detail=f"删除失败或项目不存在: {project_path}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除失败: {str(e)}")

@app.post("/cleanup")
async def cleanup_storage():
    """清理存储空间"""
    if not client:
        raise HTTPException(status_code=500, detail="客户端未初始化")
    
    try:
        client.cleanup_storage()
        return {"message": "清理完成"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"清理失败: {str(e)}")

@app.get("/config")
async def get_config():
    """获取所有配置"""
    if not client:
        raise HTTPException(status_code=500, detail="客户端未初始化")
    
    try:
        config_data = client.config.config.copy()
        
        # 隐藏敏感信息
        if 'openai_api_key' in config_data and config_data['openai_api_key']:
            key = config_data['openai_api_key']
            config_data['openai_api_key'] = '*' * 8 + key[-4:] if len(key) > 4 else '*' * len(key)
        
        return {"config": config_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取配置失败: {str(e)}")

@app.post("/config")
async def set_config(request: ConfigRequest):
    """设置配置"""
    if not client:
        raise HTTPException(status_code=500, detail="客户端未初始化")
    
    try:
        client.config.set(request.key, request.value)
        return {"message": f"已设置 {request.key} = {request.value}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"设置配置失败: {str(e)}")

@app.get("/config/{key}")
async def get_config_value(key: str):
    """获取特定配置值"""
    if not client:
        raise HTTPException(status_code=500, detail="客户端未初始化")
    
    try:
        value = client.config.get(key)
        
        # 隐藏敏感信息
        if key == 'openai_api_key' and value:
            value = '*' * 8 + value[-4:] if len(value) > 4 else '*' * len(value)
        
        return {key: value}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取配置失败: {str(e)}")

# ==================== 代码编辑API端点 ====================

@app.post("/edit/analyze")
async def analyze_code_modification(request: CodeEditRequest):
    """分析代码修改请求，生成语义编辑操作"""
    if not client:
        raise HTTPException(status_code=500, detail="客户端未初始化")
    
    project_path = None
    if request.project_path:
        project_path = str(Path(request.project_path).resolve())
    
    try:
        edits = await client.analyze_code_modification(
            request=request.request,
            search_results=request.search_results,
            project_path=project_path
        )
        print(f"分析代码修改请求: {edits}")
        
        # 转换语义编辑操作为响应格式
        edit_responses = []
        for edit in edits:
            edit_responses.append(SemanticEditResponse(
                edit_type=edit.edit_type.value,
                location={
                    'file_path': edit.location.file_path,
                    'symbol_name': edit.location.symbol_name,
                    'symbol_type': getattr(edit.location, 'symbol_type', None),
                    'start_line': edit.location.start_line,
                    'end_line': edit.location.end_line
                },
                old_code=edit.old_code,
                new_code=edit.new_code,
                description=edit.description,
                confidence=edit.confidence
            ))
        
        return EditAnalysisResponse(
            edits=edit_responses,
            total_edits=len(edit_responses)
        )
        
    except Exception as e:
        import traceback
        print(f"代码分析错误详情: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"代码分析失败: {str(e)}")

@app.post("/edit/apply")
async def apply_code_edit(request: CodeEditRequest):
    """基于搜索结果编辑代码"""
    if not client:
        raise HTTPException(status_code=500, detail="客户端未初始化")
    
    project_path = None
    if request.project_path:
        project_path = str(Path(request.project_path).resolve())
    
    try:
        edits, result = await client.edit_code_with_plan(
            request=request.request,
            search_results=request.search_results,
            project_path=project_path,
            auto_apply=request.auto_apply
        )
        
        # 转换语义编辑操作为响应格式
        edit_responses = []
        for edit in edits:
            edit_responses.append(SemanticEditResponse(
                edit_type=edit.edit_type.value,
                location={
                    'file_path': edit.location.file_path,
                    'symbol_name': edit.location.symbol_name,
                    'symbol_type': getattr(edit.location, 'symbol_type', None),
                    'start_line': edit.location.start_line,
                    'end_line': edit.location.end_line
                },
                old_code=edit.old_code,
                new_code=edit.new_code,
                description=edit.description,
                confidence=edit.confidence
            ))
        
        # 如果有编辑结果，转换为字典
        result_data = None
        if result:
            applied_edit_responses = []
            for edit in result.get('applied_edits', []):
                applied_edit_responses.append(SemanticEditResponse(
                    edit_type=edit.edit_type.value,
                    location={
                        'file_path': edit.location.file_path,
                        'symbol_name': edit.location.symbol_name,
                        'symbol_type': getattr(edit.location, 'symbol_type', None),
                        'start_line': edit.location.start_line,
                        'end_line': edit.location.end_line
                    },
                    old_code=edit.old_code,
                    new_code=edit.new_code,
                    description=edit.description,
                    confidence=edit.confidence
                ))
            
            result_data = EditResultResponse(
                success=result.get('success', False),
                applied_edits=applied_edit_responses,
                errors=result.get('errors', []),
                warnings=result.get('warnings', []),
                backup_path=result.get('backup_path'),
                diff=result.get('diff')
            )
        
        return {
            'edits': edit_responses,
            'result': result_data,
            'applied': result is not None
        }
        
    except Exception as e:
        import traceback
        print(f"代码编辑错误详情: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"代码编辑失败: {str(e)}")

@app.get("/edit/history")
async def get_edit_history():
    """获取编辑历史"""
    if not client:
        raise HTTPException(status_code=500, detail="客户端未初始化")
    
    try:
        history = client.get_edit_history()
        
        # 转换历史记录为字典格式
        history_data = []
        for record in history:
            # 处理编辑记录中的编辑列表
            applied_edit_responses = []
            edits = record.get('edits', [])
            
            for edit in edits:
                # 检查edit是否是SemanticEdit对象还是字典
                if hasattr(edit, 'edit_type'):
                    # 是SemanticEdit对象
                    applied_edit_responses.append(SemanticEditResponse(
                        edit_type=edit.edit_type.value,
                        location={
                            'file_path': edit.location.file_path,
                            'symbol_name': edit.location.symbol_name,
                            'symbol_type': getattr(edit.location, 'symbol_type', None),
                            'start_line': edit.location.start_line,
                            'end_line': edit.location.end_line
                        },
                        old_code=edit.old_code,
                        new_code=edit.new_code,
                        description=edit.description,
                        confidence=edit.confidence
                    ))
                else:
                    # 是字典格式，直接使用
                    applied_edit_responses.append(SemanticEditResponse(
                        edit_type=edit.get('edit_type', 'unknown'),
                        location=edit.get('location', {}),
                        old_code=edit.get('old_code', ''),
                        new_code=edit.get('new_code', ''),
                        description=edit.get('description', ''),
                        confidence=edit.get('confidence', 0.0)
                    ))
            
            # 构建历史记录响应
            history_data.append({
                'timestamp': record.get('timestamp'),
                'file_path': record.get('file_path'),
                'backup_path': record.get('backup_path'),
                'applied_edits': applied_edit_responses,
                'success': True,  # 历史记录中的都是成功应用的
                'errors': [],
                'warnings': []
            })
        
        return {
            'success': True,
            'total_edits': len(history_data),
            'history': history_data
        }
        
    except Exception as e:
        import traceback
        print(f"获取编辑历史错误详情: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"获取编辑历史失败: {str(e)}")

@app.post("/edit/rollback")
async def rollback_last_edit():
    """回滚最后一次编辑"""
    if not client:
        raise HTTPException(status_code=500, detail="客户端未初始化")
    
    try:
        success = client.rollback_last_edit()
        
        if success:
            return {"message": "成功回滚最后一次编辑", "success": True}
        else:
            return {"message": "没有可回滚的编辑或回滚失败", "success": False}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"回滚失败: {str(e)}")

class PatchInfo(BaseModel):
    file_path: str = Field(..., description="文件路径")
    diff: str = Field(..., description="差异补丁内容")
    edit_type: str = Field(..., description="编辑类型")
    confidence: float = Field(..., description="置信度")
    description: str = Field(..., description="编辑描述")

class ApplyPatchRequest(BaseModel):
    patch_info: PatchInfo = Field(..., description="补丁信息")
    create_backup: bool = Field(True, description="是否创建备份")

class ApplyMultiplePatchesRequest(BaseModel):
    patches: List[PatchInfo] = Field(..., description="补丁列表")
    create_backup: bool = Field(True, description="是否创建备份")

@app.post("/edit/apply_patch")
async def apply_diff_patch(request: ApplyPatchRequest):
    """应用单个差异补丁"""
    if not client:
        raise HTTPException(status_code=500, detail="客户端未初始化")
    
    try:
        # 构造补丁信息字典
        patch_info = {
            'file_path': request.patch_info.file_path,
            'diff': request.patch_info.diff,
            'edit_type': request.patch_info.edit_type,
            'confidence': request.patch_info.confidence,
            'description': request.patch_info.description
        }
        
        result = client.apply_diff_patch(patch_info, request.create_backup)
        return result
        
    except Exception as e:
        import traceback
        print(f"应用补丁错误详情: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"应用补丁失败: {str(e)}")

@app.post("/edit/apply_patches")
async def apply_multiple_patches(request: ApplyMultiplePatchesRequest):
    """批量应用多个差异补丁"""
    if not client:
        raise HTTPException(status_code=500, detail="客户端未初始化")
    
    try:
        # 构造补丁信息列表
        patches = []
        for patch in request.patches:
            patches.append({
                'file_path': patch.file_path,
                'diff': patch.diff,
                'edit_type': patch.edit_type,
                'confidence': patch.confidence,
                'description': patch.description
            })
        
        result = client.apply_multiple_patches(patches, request.create_backup)
        return result
        
    except Exception as e:
        import traceback
        print(f"批量应用补丁错误详情: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"批量应用补丁失败: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8080,
        reload=True,
        log_level="info"
    )