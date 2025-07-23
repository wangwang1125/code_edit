#!/usr/bin/env python3
"""
代码索引API客户端示例
演示如何使用代码索引API进行搜索、分析和编辑操作
"""

import json
import os
import requests
import time
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any

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
    
    def search_and_analyze_edit(self, query, project_path=None, top_k=10, 
                               filter_language=None, filter_file_type=None, 
                               use_hybrid_search=True):
        """搜索代码并直接分析语义编辑请求 - 合并操作避免重复"""
        url = f"{self.base_url}/search_and_analyze_edit"
        data = {
            "query": query,
            "top_k": top_k,
            "use_hybrid_search": use_hybrid_search
        }
        if project_path:
            data["project_path"] = project_path
        if filter_language:
            data["filter_language"] = filter_language
        if filter_file_type:
            data["filter_file_type"] = filter_file_type
        
        response = requests.post(url, json=data)
        return response.json()
    
    def search_and_edit(self, query, project_path=None, top_k=10, 
                       filter_language=None, filter_file_type=None, 
                       use_hybrid_search=True, auto_apply=False, 
                       confidence_threshold=0.7, generate_patch=False):
        """搜索代码并直接执行语义编辑 - 一站式操作"""
        url = f"{self.base_url}/search_and_edit"
        data = {
            "query": query,
            "top_k": top_k,
            "use_hybrid_search": use_hybrid_search,
            "auto_apply": auto_apply,
            "confidence_threshold": confidence_threshold,
            "generate_patch": generate_patch
        }
        if project_path:
            data["project_path"] = project_path
        if filter_language:
            data["filter_language"] = filter_language
        if filter_file_type:
            data["filter_file_type"] = filter_file_type
        
        response = requests.post(url, json=data)
        return response.json()
    
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

    def apply_multiple_patches(self, patches: List[Dict[str, Any]], create_backup: bool = True):
        """批量应用多个差异补丁 - HTTP API方式"""
        url = f"{self.base_url}/edit/apply_patches"
        data = {
            "patches": patches,
            "create_backup": create_backup
        }
        response = requests.post(url, json=data)
        return response.json()

    # ==================== 本地补丁应用功能 ====================
    
    def apply_patch_locally(self, patch_info: Dict[str, Any], create_backup: bool = True) -> Dict[str, Any]:
        """本地应用单个差异补丁"""
        result = {
            'success': False,
            'file_path': patch_info['file_path'],
            'backup_path': None,
            'error': None
        }
        
        try:
            file_path = patch_info['file_path']
            
            # 检查文件是否存在
            if not os.path.exists(file_path):
                result['error'] = f"文件不存在: {file_path}"
                return result
            
            # 1. 创建备份
            if create_backup:
                backup_path = f"{file_path}.backup_{int(time.time())}"
                shutil.copy2(file_path, backup_path)
                result['backup_path'] = backup_path
                print(f"📁 创建备份文件: {backup_path}")
            
            # 2. 读取原始文件内容
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            print(f"📖 读取原始文件: {file_path}")
            print(f"原始文件大小: {len(original_content)} 字符")
            
            # 3. 应用差异补丁
            modified_content = self._apply_diff_to_content(original_content, patch_info['diff'])
            
            # 4. 检查内容是否真的发生了变化
            if modified_content == original_content:
                print("⚠️  警告: 应用补丁后内容没有变化")
                # 尝试使用智能差异应用作为备选方案
                print("🔄 尝试使用智能差异应用...")
                modified_content = self._apply_smart_diff(original_content, patch_info['diff'])
                
                if modified_content == original_content:
                    result['error'] = "差异补丁应用后内容没有变化，可能是补丁格式问题"
                    return result
            
            # 5. 写入修改后的内容
            with open(file_path, 'w', encoding='utf-8', newline='') as f:
                f.write(modified_content)
            
            print(f"💾 写入修改后的文件: {file_path}")
            print(f"修改后文件大小: {len(modified_content)} 字符")
            
            # 6. 验证文件是否真的被修改了
            with open(file_path, 'r', encoding='utf-8') as f:
                verification_content = f.read()
            
            if verification_content != original_content:
                result['success'] = True
                print(f"✅ 成功应用差异补丁到文件: {file_path}")
                print(f"内容变化: {len(original_content)} -> {len(verification_content)} 字符")
            else:
                result['error'] = "文件写入后验证失败，内容没有实际改变"
                print(f"❌ 文件写入验证失败: {file_path}")
            
        except Exception as e:
            result['error'] = str(e)
            print(f"❌ 应用差异补丁失败: {e}")
            
        return result
    
    def _apply_diff_to_content(self, original_content: str, diff_text: str) -> str:
        """将差异补丁应用到内容上"""
        try:
            # 首先尝试使用Python的difflib来应用补丁
            if self._is_unified_diff(diff_text):
                return self._apply_unified_diff(original_content, diff_text)
            else:
                # 对于非标准格式，尝试智能解析
                return self._apply_smart_diff(original_content, diff_text)
            
        except Exception as e:
            print(f"差异补丁应用失败: {e}")
            print(f"差异内容: {diff_text[:200]}...")
            return original_content
    
    def _is_unified_diff(self, diff_text: str) -> bool:
        """检查是否为标准的unified diff格式"""
        return '@@' in diff_text and (diff_text.startswith('---') or '@@' in diff_text.split('\n')[0:3])
    
    def _apply_unified_diff(self, original_content: str, diff_text: str) -> str:
        """应用标准的unified diff格式，支持多个hunk的智能处理"""
        print("应用标准unified diff格式")
        import re
        
        # 确保原始内容使用统一的行结束符
        original_content = original_content.replace('\r\n', '\n').replace('\r', '\n')
        lines = original_content.splitlines(keepends=True)
        diff_lines = diff_text.splitlines()
        
        print(f"原始文件行数: {len(lines)}")
        print(f"差异补丁行数: {len(diff_lines)}")
        
        # 🔧 优化：解析所有hunk，然后从下往上应用
        hunks = []
        i = 0
        while i < len(diff_lines):
            line = diff_lines[i]
            
            if line.startswith('@@'):
                # 解析hunk头部 @@-start,count +start,count@@
                match = re.match(r'@@\s*-(\d+)(?:,(\d+))?\s*\+(\d+)(?:,(\d+))?\s*@@', line)
                if match:
                    old_start = int(match.group(1)) - 1  # 转换为0索引
                    old_count = int(match.group(2)) if match.group(2) else 1
                    new_start = int(match.group(3)) - 1  # 转换为0索引
                    new_count = int(match.group(4)) if match.group(4) else 1
                    
                    # 收集hunk内容
                    hunk_lines = []
                    i += 1
                    while i < len(diff_lines) and not diff_lines[i].startswith('@@'):
                        hunk_lines.append(diff_lines[i])
                        i += 1
                    
                    hunks.append({
                        'old_start': old_start,
                        'old_count': old_count,
                        'new_start': new_start,
                        'new_count': new_count,
                        'lines': hunk_lines
                    })
                    continue
            
            i += 1
        
        # 🔧 关键优化：按old_start从大到小排序，从下往上应用
        hunks.sort(key=lambda h: h['old_start'], reverse=True)
        
        print(f"解析到 {len(hunks)} 个hunk，将从下往上应用:")
        for idx, hunk in enumerate(hunks):
            print(f"  Hunk {idx+1}: 行{hunk['old_start']+1}-{hunk['old_start']+hunk['old_count']}")
        
        result_lines = lines[:]  # 复制原始行
        
        # 从下往上应用每个hunk
        for hunk_idx, hunk in enumerate(hunks):
            print(f"\n应用 Hunk {hunk_idx+1}/{len(hunks)}: 行{hunk['old_start']+1}-{hunk['old_start']+hunk['old_count']}")
            
            old_start = hunk['old_start']
            old_count = hunk['old_count']
            hunk_lines = hunk['lines']
            
            # 构建新的行内容
            new_lines = []
            hunk_line_idx = 0
            original_lines_processed = 0
            
            while hunk_line_idx < len(hunk_lines):
                hunk_line = hunk_lines[hunk_line_idx]
                
                if hunk_line.startswith(' '):
                    # 上下文行，保持不变
                    new_lines.append(hunk_line[1:])
                    if not hunk_line[1:].endswith('\n') and len(result_lines) > 0 and result_lines[0].endswith('\n'):
                        new_lines[-1] += '\n'
                    original_lines_processed += 1
                elif hunk_line.startswith('-'):
                    # 删除行，跳过（不添加到new_lines）
                    print(f"  删除行 {old_start + original_lines_processed + 1}: {hunk_line[1:].strip()}")
                    original_lines_processed += 1
                elif hunk_line.startswith('+'):
                    # 添加行
                    new_line = hunk_line[1:]
                    if not new_line.endswith('\n') and len(result_lines) > 0 and result_lines[0].endswith('\n'):
                        new_line += '\n'
                    print(f"  添加行: {new_line.strip()}")
                    new_lines.append(new_line)
                
                hunk_line_idx += 1
            
            # 替换原始文件中的对应行
            end_idx = old_start + old_count
            if end_idx > len(result_lines):
                end_idx = len(result_lines)
            
            print(f"  替换行范围: {old_start+1}-{end_idx} -> {len(new_lines)} 行")
            result_lines[old_start:end_idx] = new_lines
        
        result_content = ''.join(result_lines)
        print(f"\n✅ 差异应用完成:")
        print(f"  原始内容: {len(original_content)} 字符, {len(lines)} 行")
        print(f"  应用后: {len(result_content)} 字符, {len(result_lines)} 行")
        
        return result_content
    
    def _apply_smart_diff(self, original_content: str, diff_text: str) -> str:
        """智能应用非标准格式的差异补丁"""
        print("正在应用智能差异补丁...")
        try:
            # 尝试解析包含 +/- 的简单差异格式
            diff_lines = diff_text.splitlines()
            original_lines = original_content.splitlines()
            
            # 查找删除和添加的行
            lines_to_remove = []
            lines_to_add = []
            
            for line in diff_lines:
                if line.startswith('-'):
                    # 删除行（去掉前缀-）
                    clean_line = line[1:].strip()
                    if clean_line:
                        lines_to_remove.append(clean_line)
                elif line.startswith('+'):
                    # 添加行（去掉前缀+）
                    clean_line = line[1:]
                    lines_to_add.append(clean_line)
            
            # 如果有明确的删除和添加操作
            if lines_to_remove or lines_to_add:
                result_lines = []
                
                for original_line in original_lines:
                    # 检查是否需要删除这一行
                    should_remove = False
                    for remove_line in lines_to_remove:
                        if remove_line in original_line:
                            should_remove = True
                            break
                    
                    if not should_remove:
                        result_lines.append(original_line)
                    else:
                        # 如果删除了一行，检查是否有对应的添加行
                        for add_line in lines_to_add:
                            if add_line not in result_lines:
                                result_lines.append(add_line)
                                lines_to_add.remove(add_line)
                                break
                
                # 添加剩余的新行
                result_lines.extend(lines_to_add)
                
                return '\n'.join(result_lines) + '\n' if original_content.endswith('\n') else '\n'.join(result_lines)
            
            # 如果无法解析，尝试直接文本替换
            return self._apply_text_replacement(original_content, diff_text)
            
        except Exception as e:
            print(f"智能差异应用失败: {e}")
            return original_content
    
    def _apply_text_replacement(self, original_content: str, diff_text: str) -> str:
        """尝试通过文本替换应用差异"""
        try:
            # 查找可能的替换模式
            lines = diff_text.splitlines()
            
            for i, line in enumerate(lines):
                if line.startswith('-') and i + 1 < len(lines) and lines[i + 1].startswith('+'):
                    # 找到替换模式：-old_text +new_text
                    old_text = line[1:].strip()
                    new_text = lines[i + 1][1:].strip()
                    
                    if old_text in original_content:
                        original_content = original_content.replace(old_text, new_text)
                        print(f"🔄 执行文本替换: '{old_text[:50]}...' -> '{new_text[:50]}...'")
            
            return original_content
            
        except Exception as e:
            print(f"文本替换失败: {e}")
            return original_content
    
    def interactive_apply_patches(self, patches: List[Dict[str, Any]], create_backup: bool = True) -> Dict[str, Any]:
        """交互式应用多个差异补丁，类似git的交互式模式"""
        results = {
            'success': True,
            'applied_patches': [],
            'skipped_patches': [],
            'failed_patches': [],
            'backup_paths': []
        }
        
        print(f"\n🔍 发现 {len(patches)} 个差异补丁，开始交互式应用...")
        print("📝 操作说明:")
        print("  y - 应用此补丁")
        print("  n - 跳过此补丁") 
        print("  q - 退出，不再处理后续补丁")
        print("  a - 应用此补丁及所有后续补丁")
        print("  d - 显示详细差异内容")
        print("  s - 显示补丁统计信息")
        print("=" * 60)
        
        # 🔧 优化：按文件分组并按行号从大到小排序，避免行号偏移问题
        patches_by_file = {}
        for patch in patches:
            file_path = patch['file_path']
            if file_path not in patches_by_file:
                patches_by_file[file_path] = []
            patches_by_file[file_path].append(patch)
        
        # 对每个文件的补丁按行号从大到小排序
        for file_path in patches_by_file:
            patches_by_file[file_path].sort(
                key=lambda p: p.get('line_range', {}).get('start', 0), 
                reverse=True  # 从下往上应用
            )
        
        print(f"📊 补丁分组统计:")
        for file_path, file_patches in patches_by_file.items():
            print(f"  📄 {file_path}: {len(file_patches)} 个补丁")
        print("=" * 60)
        
        apply_all = False
        patch_index = 0
        
        # 按文件处理补丁
        for file_path, file_patches in patches_by_file.items():
            print(f"\n📁 处理文件: {file_path}")
            print(f"🔢 该文件共有 {len(file_patches)} 个补丁，将从下往上应用以避免行号偏移")
            
            for i, patch in enumerate(file_patches):
                patch_index += 1
                if apply_all:
                    # 自动应用所有剩余补丁
                    result = self.apply_patch_locally(patch, create_backup)
                    self._process_patch_result(result, results)
                    continue
                
                # 显示补丁信息
                print(f"\n📄 补丁 {patch_index}/{len(patches)}: {patch['file_path']}")
                print(f"🔧 编辑类型: {patch.get('edit_type', 'unknown')}")
                print(f"📊 置信度: {patch.get('confidence', 0):.2f}")
                
                if 'line_range' in patch:
                    line_range = patch['line_range']
                    print(f"📍 行数范围: {line_range.get('start', '?')}-{line_range.get('end', '?')}")
                    print(f"🔄 处理顺序: 文件内第 {i+1}/{len(file_patches)} 个补丁 (从下往上)")
                
                # 显示简短的差异预览
                diff_lines = patch['diff'].splitlines()
                preview_lines = [line for line in diff_lines[:10] if line.strip()]
                if preview_lines:
                    print("🔍 差异预览:")
                    for line in preview_lines[:5]:
                        if line.startswith('+'):
                            print(f"  \033[32m{line}\033[0m")  # 绿色
                        elif line.startswith('-'):
                            print(f"  \033[31m{line}\033[0m")  # 红色
                        else:
                            print(f"  {line}")
                    if len(diff_lines) > 10:
                        print("  ... (更多内容，输入 'd' 查看完整差异)")
                
                # 获取用户选择
                while True:
                    choice = input(f"\n应用此补丁? [y/n/q/a/d/s]: ").strip().lower()
                    
                    if choice == 'y':
                        result = self.apply_patch_locally(patch, create_backup)
                        self._process_patch_result(result, results)
                        break
                    elif choice == 'n':
                        results['skipped_patches'].append(patch['file_path'])
                        print(f"⏭️  跳过补丁: {patch['file_path']}")
                        break
                    elif choice == 'q':
                        print("🛑 用户选择退出")
                        return results
                    elif choice == 'a':
                        print("🚀 应用此补丁及所有后续补丁")
                        apply_all = True
                        result = self.apply_patch_locally(patch, create_backup)
                        self._process_patch_result(result, results)
                        break
                    elif choice == 'd':
                        print("\n📋 完整差异内容:")
                        print("-" * 50)
                        for line in patch['diff'].splitlines():
                            if line.startswith('+'):
                                print(f"\033[32m{line}\033[0m")  # 绿色
                            elif line.startswith('-'):
                                print(f"\033[31m{line}\033[0m")  # 红色
                            else:
                                print(line)
                        print("-" * 50)
                    elif choice == 's':
                        self._show_patch_stats(patch)
                    else:
                        print("❌ 无效选择，请输入 y/n/q/a/d/s")
        
        # 显示最终统计
        self._show_final_stats(results)
        return results
    
    def _process_patch_result(self, result: Dict[str, Any], results: Dict[str, Any]):
        """处理单个补丁应用结果"""
        if result['success']:
            results['applied_patches'].append(result['file_path'])
            if result['backup_path']:
                results['backup_paths'].append(result['backup_path'])
        else:
            results['failed_patches'].append({
                'file_path': result['file_path'],
                'error': result['error']
            })
            results['success'] = False
    
    def _show_patch_stats(self, patch: Dict[str, Any]):
        """显示补丁统计信息"""
        print(f"\n📊 补丁统计信息:")
        print(f"  文件路径: {patch['file_path']}")
        print(f"  编辑类型: {patch.get('edit_type', 'unknown')}")
        print(f"  置信度: {patch.get('confidence', 0):.2f}")
        
        # 分析差异内容
        diff_lines = patch['diff'].splitlines()
        add_count = len([l for l in diff_lines if l.startswith('+')])
        del_count = len([l for l in diff_lines if l.startswith('-')])
        context_count = len([l for l in diff_lines if l.startswith(' ')])
        
        print(f"  添加行数: {add_count}")
        print(f"  删除行数: {del_count}")
        print(f"  上下文行数: {context_count}")
        
        if 'line_range' in patch:
            line_range = patch['line_range']
            print(f"  影响行范围: {line_range.get('start', '?')}-{line_range.get('end', '?')}")
    
    def _show_final_stats(self, results: Dict[str, Any]):
        """显示最终统计信息"""
        print(f"\n📈 应用结果统计:")
        print(f"  ✅ 成功应用: {len(results['applied_patches'])} 个")
        print(f"  ⏭️  跳过: {len(results['skipped_patches'])} 个")
        print(f"  ❌ 失败: {len(results['failed_patches'])} 个")
        
        if results['backup_paths']:
            print(f"  📁 备份文件: {len(results['backup_paths'])} 个")
            for backup in results['backup_paths']:
                print(f"    - {backup}")
        
        if results['failed_patches']:
            print(f"  ❌ 失败详情:")
            for failed in results['failed_patches']:
                print(f"    - {failed['file_path']}: {failed['error']}")


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
        
        # 2. 搜索代码并分析语义编辑 - 合并操作
        print("2. 搜索代码并分析语义编辑...")
        modification_request = "去掉使用泡沫轴相关的运动"
        
        # 使用差异补丁模式
        search_and_analysis_result = client.search_and_edit(
            query=modification_request,
            project_path=project_path,
            auto_apply=False,  # 不自动应用
            generate_patch=True  # 生成差异补丁
        )
        
        print(f"搜索和分析结果:")
        print(f"  - 搜索到 {search_and_analysis_result.get('search_count', 0)} 个结果")
        print(f"  - 分析成功: {search_and_analysis_result.get('analysis_success', False)}")
        print(f"  - 差异补丁数量: {len(search_and_analysis_result.get('patches', []))}")
        
        # 应用差异补丁
        if search_and_analysis_result.get('patches'):
            print(f"\n=== 发现 {len(search_and_analysis_result['patches'])} 个差异补丁 ===")
            
            # 使用本地交互式补丁应用
            apply_result = client.interactive_apply_patches(
                search_and_analysis_result['patches'], 
                create_backup=True
            )
            
            print(f"\n🎉 补丁应用完成! 成功: {len(apply_result.get('applied_patches', []))} 个")
        else:
            print("没有生成差异补丁")
        

        # 3. 获取项目状态
        print("\n3. 获取项目状态...")
        status_result = client.get_project_status(project_path)
        print(f"项目状态: {json.dumps(status_result, indent=2, ensure_ascii=False)}")
        
    except requests.exceptions.ConnectionError:
        print("错误: 无法连接到API服务器。请确保服务器正在运行。")
        print("运行命令: python start_server.py")
    except Exception as e:
        print(f"错误: {e}")

if __name__ == "__main__":
    main()