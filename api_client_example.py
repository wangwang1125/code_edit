#!/usr/bin/env python3
"""
FastAPI 客户端示例
演示如何使用代码索引API

正确的使用流程：
1. 索引项目 - 建立代码索引
2. 搜索代码 - 根据查询词找到相关代码片段
3. 获取上下文 - 基于搜索结果获取具体的代码内容

注意：
- get_context_from_search_results() 方法展示了如何基于搜索结果获取上下文（推荐）
- get_context() 方法会重新搜索，这在某些场景下可能不是最优的
"""

import requests
import json
from pathlib import Path

class CodeIndexAPIClient:
    def __init__(self, base_url="http://127.0.0.1:8080"):
        self.base_url = base_url
    
    def index_project(self, project_path, force=False, max_tokens=1000):
        """索引项目"""
        url = f"{self.base_url}/index"
        data = {
            "project_path": project_path,
            "force": force,
            "max_tokens": max_tokens
        }
        response = requests.post(url, json=data)
        return response.json()
    
    def search_code(self, query, project_path=None, top_k=10, language=None):
        """搜索代码"""
        url = f"{self.base_url}/search"
        data = {
            "query": query,
            "top_k": top_k
        }
        if project_path:
            data["project_path"] = project_path
        if language:
            data["language"] = language
        
        response = requests.post(url, json=data)
        return response.json()
    
    def get_context_from_search_results(self, search_results, project_path, max_results=3):
        """基于搜索结果获取代码上下文"""
        if not search_results.get('results'):
            return {'contexts': [], 'total_results': 0}
        
        contexts = []
        for i, result in enumerate(search_results['results'][:max_results]):
            try:
                file_path = Path(project_path) / result['file_path']
                if file_path.exists():
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    
                    start_line = max(0, result['start_line'] - 1)
                    end_line = min(len(lines), result['end_line'])
                    code_content = ''.join(lines[start_line:end_line])
                    
                    context = {
                        'file_path': result['file_path'],
                        'start_line': result['start_line'],
                        'end_line': result['end_line'],
                        'similarity': result['similarity'],
                        'language': result['language'],
                        'chunk_type': result['chunk_type'],
                        'code_content': code_content.rstrip()
                    }
                    contexts.append(context)
                else:
                    contexts.append({
                        'file_path': result['file_path'],
                        'error': '文件不存在或无法访问'
                    })
            except Exception as e:
                contexts.append({
                    'file_path': result['file_path'],
                    'error': f'读取文件时出错: {e}'
                })
        
        return {
            'contexts': contexts,
            'total_results': len(search_results['results']),
            'displayed_results': len(contexts)
        }

    def get_context(self, query=None, project_path=None, max_chunks=5, search_results=None):
        """获取代码上下文"""
        url = f"{self.base_url}/context"
        data = {
            "max_chunks": max_chunks
        }
        
        # 添加可选参数
        if query:
            data["query"] = query
        if project_path:
            data["project_path"] = project_path
        if search_results:
            data["search_results"] = search_results
        
        response = requests.post(url, json=data)
        return response.json()
    
    def get_all_projects_status(self):
        """获取所有项目状态"""
        url = f"{self.base_url}/status"
        response = requests.get(url)
        return response.json()
    
    def get_project_status(self, project_path):
        """获取特定项目状态"""
        url = f"{self.base_url}/status/{project_path}"
        response = requests.get(url)
        return response.json()
    
    def update_project(self, project_path):
        """更新项目索引"""
        url = f"{self.base_url}/update/{project_path}"
        response = requests.post(url)
        return response.json()
    
    def get_stats(self):
        """获取统计信息"""
        url = f"{self.base_url}/stats"
        response = requests.get(url)
        return response.json()
    
    def delete_project(self, project_path, confirm=True):
        """删除项目索引"""
        url = f"{self.base_url}/delete/{project_path}"
        params = {"confirm": confirm}
        response = requests.delete(url, params=params)
        return response.json()
    
    def cleanup_storage(self):
        """清理存储空间"""
        url = f"{self.base_url}/cleanup"
        response = requests.post(url)
        return response.json()
    
    def get_config(self):
        """获取配置"""
        url = f"{self.base_url}/config"
        response = requests.get(url)
        return response.json()
    
    def set_config(self, key, value):
        """设置配置"""
        url = f"{self.base_url}/config"
        data = {"key": key, "value": value}
        response = requests.post(url, json=data)
        return response.json()
    
    # ==================== 代码编辑功能 ====================
    
    def analyze_code_modification(self, request, search_results=None, project_path=None):
        """分析代码修改请求，生成编辑计划"""
        url = f"{self.base_url}/edit/analyze"
        data = {
            "request": request,
            "auto_apply": False  # 分析时不自动应用
        }
        if search_results:
            data["search_results"] = search_results
        if project_path:
            data["project_path"] = project_path
        
        response = requests.post(url, json=data)
        return response.json()
    
    def apply_code_edit(self, request, search_results=None, project_path=None, auto_apply=True):
        """应用代码编辑"""
        url = f"{self.base_url}/edit/apply"
        data = {
            "request": request,
            "auto_apply": auto_apply
        }
        if search_results:
            data["search_results"] = search_results
        if project_path:
            data["project_path"] = project_path
        
        response = requests.post(url, json=data)
        return response.json()
    
    def get_edit_history(self):
        """获取编辑历史"""
        url = f"{self.base_url}/edit/history"
        response = requests.get(url)
        return response.json()
    
    def rollback_last_edit(self):
        """回滚最后一次编辑"""
        url = f"{self.base_url}/edit/rollback"
        response = requests.post(url)
        return response.json()

def main():
    """示例用法"""
    client = CodeIndexAPIClient()
    
    # 示例项目路径
    project_path = str(Path(__file__).parent / "test_html_project")
    
    try:
        print("=== 代码索引API客户端示例 ===\n")
        
        # 1. 索引项目
        print("1. 索引项目...")
        result = client.index_project(project_path)
        print(f"索引结果: {json.dumps(result, indent=2, ensure_ascii=False)}\n")
        
        # 2. 搜索代码
        print("2. 搜索代码...")
        search_result = client.search_code("去掉使用泡沫轴相关的运动", project_path)
        print(f"搜索结果: {json.dumps(search_result, indent=2, ensure_ascii=False)}\n")
        
        # 3. 获取项目状态
        print("4. 获取项目状态...")
        status_result = client.get_project_status(project_path)
        print(f"项目状态: {json.dumps(status_result, indent=2, ensure_ascii=False)}\n")
        
        # 4. 语义代码编辑功能示例
        print("4. 语义代码编辑功能示例...")
        
        # 4.1 分析代码修改请求
        print("4.1 分析语义编辑请求...")
        modification_request = "去掉使用泡沫轴相关的运动"
        
        if search_result.get('results') and len(search_result['results']) > 0:
            # 分析修改请求
            analysis_result = client.analyze_code_modification(
                request=modification_request,
                search_results=search_result['results'],
                project_path=project_path
            )
            print(f"语义分析结果:{analysis_result}")
            # 检查是否有编辑操作（而不是检查success字段）
            if analysis_result.get('edits') and len(analysis_result.get('edits', [])) > 0:
                edits = analysis_result.get('edits', [])
                print(f"  - 生成了 {len(edits)} 个语义编辑操作")
                for i, edit in enumerate(edits, 1):  
                    print(f"  - 编辑 {i}: {edit.get('description', '无描述')}")
                    print(f"    类型: {edit.get('edit_type', '未知')}")
                    print(f"    置信度: {edit.get('confidence', 0):.2f}")
                    if edit.get('location'):
                        loc = edit['location']
                        print(f"    位置: {loc.get('symbol_name', '未知符号')} ({loc.get('start_line', 0)}-{loc.get('end_line', 0)}行)")
            else:
                print(f"  - 分析失败: {analysis_result.get('error', '未知错误')}")
            print()
            
            # 4.2 应用语义编辑（如果有有效的编辑操作）
            if analysis_result.get('edits') and len(analysis_result.get('edits', [])) > 0:
                # 计算平均置信度作为安全分数
                edits = analysis_result.get('edits', [])
                avg_confidence = sum(edit.get('confidence', 0) for edit in edits) / len(edits) if edits else 0
                
                if avg_confidence > 0.7:
                    print("4.2 应用语义编辑...")
                    edit_result = client.apply_code_edit(
                        request=modification_request,
                        search_results=search_result['results'],
                        project_path=project_path,
                        auto_apply=True
                    )
                    
                    print(f"语义编辑结果:{edit_result}")
                    if edit_result.get('edits') and len(edit_result.get('edits', [])) > 0:
                        applied_edits = edit_result.get('applied_edits', [])
                        print(f"  - 成功应用了 {len(applied_edits)} 个编辑操作")
                        if edit_result.get('backup_path'):
                            print(f"  - 备份文件: {edit_result.get('backup_path')}")
                        if edit_result.get('warnings'):
                            print(f"  - 警告: {edit_result.get('warnings')}")
                    else:
                        print(f"  - 应用失败: {edit_result.get('error', '未知错误')}")
                        if edit_result.get('errors'):
                            for error in edit_result.get('errors', []):
                                print(f"    错误: {error}")
                    print()
                    
                    # 5.3 获取编辑历史
                    print("5.3 获取语义编辑历史...")
                    history_result = client.get_edit_history()
                    print(f"编辑历史:")
                    if history_result.get('success'):
                        history = history_result.get('history', [])
                        print(f"  - 总共 {len(history)} 次编辑记录")
                        for i, record in enumerate(history[-3:], 1):  # 显示最近3次
                            print(f"  - 记录 {i}:")
                            print(f"    时间: {record.get('timestamp', '未知')}")
                            print(f"    文件: {record.get('file_path', '未知')}")
                            applied_edits = record.get('applied_edits', [])
                            print(f"    编辑数: {len(applied_edits)}")
                    else:
                        print(f"  - 获取历史失败: {history_result.get('error', '未知错误')}")
                    print()
                    
                    # 5.4 回滚最后一次编辑（可选）
                    selected_flag = input("是否需要回滚最后一次语义编辑? [y/N]: ")
                    if selected_flag.lower() == 'y':
                        print("5.4 回滚最后一次语义编辑...")
                        rollback_result = client.rollback_last_edit()
                        print(f"回滚结果:")
                        if rollback_result.get('success'):
                            print(f"  - 回滚成功")
                            if rollback_result.get('restored_file'):
                                print(f"  - 恢复文件: {rollback_result.get('restored_file')}")
                        else:
                            print(f"  - 回滚失败: {rollback_result.get('error', '未知错误')}")
                        print()
                    else:
                        print("用户选择不回滚，跳过回滚操作\n")
                else:
                    print(f"平均置信度过低 ({avg_confidence:.2f} < 0.7)，跳过自动应用编辑\n")
            else:
                print("没有生成有效的语义编辑操作，跳过应用步骤\n")
        else:
            print("没有搜索结果，跳过语义编辑示例\n")
        

        
    except requests.exceptions.ConnectionError:
        print("错误: 无法连接到API服务器。请确保服务器正在运行。")
        print("运行命令: python start_server.py")
    except Exception as e:
        print(f"错误: {e}")

if __name__ == "__main__":
    main()