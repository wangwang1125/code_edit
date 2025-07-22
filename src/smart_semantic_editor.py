"""
智能语义代码编辑器 - 集成版本
结合了原有的语义分析功能和新的智能编辑策略
参考 Cursor 和 Claude 的做法，减少对精确行号的依赖
"""

import json
import re
import time
import difflib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio


class SemanticEditType(Enum):
    """语义编辑类型"""
    REPLACE = "replace"
    INSERT = "insert"
    DELETE = "delete"
    MODIFY = "modify"


@dataclass
class CodeLocation:
    """代码位置信息"""
    file_path: Optional[str] = None
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    start_column: Optional[int] = None
    end_column: Optional[int] = None
    symbol_name: Optional[str] = None
    symbol_type: Optional[str] = None  # 符号类型，如 'function', 'class', 'variable' 等
    context: Optional[str] = None


@dataclass
class SemanticEdit:
    """语义编辑操作"""
    description: str
    edit_type: SemanticEditType
    location: CodeLocation
    old_code: Optional[str] = None
    new_code: Optional[str] = None
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'description': self.description,
            'edit_type': self.edit_type.value,
            'location': asdict(self.location),
            'old_code': self.old_code,
            'new_code': self.new_code,
            'confidence': self.confidence
        }


class SmartSemanticEditor:
    """智能语义代码编辑器"""
    
    def __init__(self, api_key: str = None, model: str = "qwen-plus", backup_dir: Optional[Path] = None):
        self.api_key = api_key
        self.model = model
        self.backup_dir = backup_dir or Path(".") / ".cursor_like_client" / "backups"
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.edit_history: List[Dict[str, Any]] = []
    
    def apply_edit(self, file_path: str, edit: SemanticEdit) -> bool:
        """应用单个编辑操作"""
        try:
            # 创建备份
            backup_path = self._create_backup(file_path)
            
            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 应用智能编辑
            new_content, success = self._apply_smart_edit(content, edit)
            
            if success and new_content != content:
                # 写回文件
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                
                print(f"成功应用编辑: {edit.description}")
                
                # 记录编辑历史
                self.edit_history.append({
                    'timestamp': time.time(),
                    'file_path': file_path,
                    'edit': edit.to_dict(),
                    'backup_path': str(backup_path)
                })
                
                return True
            else:
                print(f"编辑未产生变化: {edit.description}")
                return False
                
        except Exception as e:
            print(f"应用编辑失败: {e}")
            # 尝试恢复备份
            if 'backup_path' in locals():
                self._restore_from_backup(file_path, backup_path)
            return False
    
    def apply_edits(self, file_path: str, edits: List[SemanticEdit]) -> bool:
        """应用多个编辑操作"""
        try:
            # 创建备份
            backup_path = self._create_backup(file_path)
            
            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            success_count = 0
            
            # 按优先级排序编辑操作
            sorted_edits = self._sort_edits_by_priority(edits)
            
            # 应用每个编辑
            for edit in sorted_edits:
                try:
                    new_content, success = self._apply_smart_edit(content, edit)
                    
                    if success and new_content != content:
                        print(f"成功应用编辑: {edit.description}")
                        content = new_content
                        success_count += 1
                    else:
                        print(f"编辑未产生变化: {edit.description}")
                        
                except Exception as e:
                    print(f"应用编辑失败: {edit.description}, 错误: {e}")
                    continue
            
            # 如果有成功的编辑，写回文件
            if success_count > 0 and content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                print(f"成功应用 {success_count}/{len(edits)} 个编辑操作")
                
                # 记录编辑历史
                self.edit_history.append({
                    'timestamp': time.time(),
                    'file_path': file_path,
                    'edits': [edit.to_dict() for edit in sorted_edits],
                    'backup_path': str(backup_path),
                    'success_count': success_count
                })
                
                return True
            else:
                print("没有成功的编辑操作")
                return False
                
        except Exception as e:
            print(f"应用编辑失败: {e}")
            # 尝试恢复备份
            if 'backup_path' in locals():
                self._restore_from_backup(file_path, backup_path)
            return False
    
    def _apply_smart_edit(self, content: str, edit: SemanticEdit) -> Tuple[str, bool]:
        """应用智能编辑 - 参考Cursor和Claude的做法"""
        try:
            # 策略1: 基于符号的编辑
            if edit.location.symbol_name and self._is_valid_identifier(edit.location.symbol_name):
                result = self._apply_symbol_based_edit(content, edit)
                if result[1]:
                    return result
            
            # 策略2: 基于内容匹配的编辑
            if edit.old_code and edit.old_code.strip():
                result = self._apply_content_based_edit(content, edit)
                if result[1]:
                    return result
            
            # 策略3: 基于行号的编辑（如果有的话）
            if edit.location.start_line and edit.location.end_line:
                result = self._apply_line_based_edit(content, edit)
                if result[1]:
                    return result
            
            # 策略4: 模糊匹配编辑
            result = self._apply_fuzzy_edit(content, edit)
            if result[1]:
                return result
            
            # 策略5: 智能插入（如果是插入操作）
            if edit.edit_type == SemanticEditType.INSERT:
                return self._apply_smart_insert(content, edit)
            
            return content, False
            
        except Exception as e:
            print(f"智能编辑失败: {e}")
            return content, False
    
    def _apply_symbol_based_edit(self, content: str, edit: SemanticEdit) -> Tuple[str, bool]:
        """基于符号的编辑"""
        import re
        
        symbol_name = edit.location.symbol_name
        lines = content.split('\n')
        
        # 查找符号定义
        for i, line in enumerate(lines):
            # 查找函数定义
            if re.search(rf'def\s+{re.escape(symbol_name)}\s*\(', line):
                end_line = self._find_function_end(lines, i)
                return self._replace_line_range(lines, i, end_line, edit)
            
            # 查找类定义
            if re.search(rf'class\s+{re.escape(symbol_name)}\s*[\(:]', line):
                end_line = self._find_class_end(lines, i)
                return self._replace_line_range(lines, i, end_line, edit)
            
            # 查找变量赋值
            if re.search(rf'^(\s*){re.escape(symbol_name)}\s*=', line):
                return self._replace_line_range(lines, i, i, edit)
        
        return content, False
    
    def _apply_content_based_edit(self, content: str, edit: SemanticEdit) -> Tuple[str, bool]:
        """基于内容匹配的编辑"""
        old_code = edit.old_code.strip()
        
        # 精确匹配
        if old_code in content:
            if edit.edit_type == SemanticEditType.DELETE:
                new_content = content.replace(old_code, "", 1)
            else:
                new_content = content.replace(old_code, edit.new_code or "", 1)
            return new_content, True
        
        # 忽略缩进的匹配
        lines = content.split('\n')
        old_lines = old_code.split('\n')
        
        for i in range(len(lines) - len(old_lines) + 1):
            match = True
            for j, old_line in enumerate(old_lines):
                if lines[i + j].strip() != old_line.strip():
                    match = False
                    break
            
            if match:
                # 找到匹配，进行替换
                if edit.edit_type == SemanticEditType.DELETE:
                    new_lines = lines[:i] + lines[i + len(old_lines):]
                else:
                    new_code_lines = (edit.new_code or "").split('\n')
                    # 保持原有的缩进
                    indent = self._get_line_indent(lines[i])
                    indented_new_lines = [indent + line if line.strip() else line 
                                        for line in new_code_lines]
                    new_lines = lines[:i] + indented_new_lines + lines[i + len(old_lines):]
                
                return '\n'.join(new_lines), True
        
        return content, False
    
    def _apply_line_based_edit(self, content: str, edit: SemanticEdit) -> Tuple[str, bool]:
        """基于行号的编辑"""
        lines = content.split('\n')
        start_line = edit.location.start_line - 1  # 转换为0索引
        end_line = edit.location.end_line - 1
        
        # 验证行号范围
        if start_line < 0 or end_line >= len(lines) or start_line > end_line:
            return content, False
        
        return self._replace_line_range(lines, start_line, end_line, edit)
    
    def _apply_fuzzy_edit(self, content: str, edit: SemanticEdit) -> Tuple[str, bool]:
        """基于模糊匹配的编辑"""
        # 尝试使用描述中的关键词进行匹配
        description = edit.description.lower()
        lines = content.split('\n')
        
        # 提取可能的目标
        targets = []
        if edit.location.symbol_name:
            targets.append(edit.location.symbol_name)
        if edit.old_code:
            targets.extend(edit.old_code.split())
        
        # 查找包含目标的行
        matching_lines = []
        for target in targets:
            for i, line in enumerate(lines):
                if target in line and i not in matching_lines:
                    matching_lines.append(i)
        
        if matching_lines:
            # 选择第一个匹配的行
            line_idx = matching_lines[0]
            
            if edit.edit_type == SemanticEditType.DELETE:
                new_lines = lines[:line_idx] + lines[line_idx + 1:]
            elif edit.edit_type == SemanticEditType.INSERT:
                indent = self._get_line_indent(lines[line_idx])
                new_code_lines = (edit.new_code or "").split('\n')
                indented_new_lines = [indent + line if line.strip() else line 
                                    for line in new_code_lines]
                new_lines = lines[:line_idx + 1] + indented_new_lines + lines[line_idx + 1:]
            else:  # replace
                indent = self._get_line_indent(lines[line_idx])
                new_code_lines = (edit.new_code or "").split('\n')
                indented_new_lines = [indent + line if line.strip() else line 
                                    for line in new_code_lines]
                new_lines = lines[:line_idx] + indented_new_lines + lines[line_idx + 1:]
            
            return '\n'.join(new_lines), True
        
        return content, False
    
    def _apply_smart_insert(self, content: str, edit: SemanticEdit) -> Tuple[str, bool]:
        """智能插入"""
        lines = content.split('\n')
        
        # 如果有符号名称，尝试在相关位置插入
        if edit.location.symbol_name:
            symbol_name = edit.location.symbol_name
            for i, line in enumerate(lines):
                if symbol_name in line:
                    indent = self._get_line_indent(line)
                    new_code_lines = (edit.new_code or "").split('\n')
                    indented_new_lines = [indent + code_line if code_line.strip() else code_line 
                                        for code_line in new_code_lines]
                    new_lines = lines[:i + 1] + [""] + indented_new_lines + lines[i + 1:]
                    return '\n'.join(new_lines), True
        
        # 默认在文件末尾插入
        lines.append("")
        lines.extend((edit.new_code or "").split('\n'))
        return '\n'.join(lines), True
    
    def _sort_edits_by_priority(self, edits: List[SemanticEdit]) -> List[SemanticEdit]:
        """按优先级排序编辑操作"""
        def get_priority(edit: SemanticEdit) -> int:
            # 删除操作优先级最高
            if edit.edit_type == SemanticEditType.DELETE:
                return 1
            # 替换操作次之
            elif edit.edit_type == SemanticEditType.REPLACE:
                return 2
            # 修改操作
            elif edit.edit_type == SemanticEditType.MODIFY:
                return 3
            # 插入操作优先级最低
            else:
                return 4
        
        return sorted(edits, key=get_priority)
    
    def _find_function_end(self, lines: List[str], start_line: int) -> int:
        """查找函数定义的结束行"""
        indent_level = len(lines[start_line]) - len(lines[start_line].lstrip())
        
        for i in range(start_line + 1, len(lines)):
            line = lines[i]
            if line.strip() == "":
                continue
            
            current_indent = len(line) - len(line.lstrip())
            if current_indent <= indent_level and line.strip():
                return i - 1
        
        return len(lines) - 1
    
    def _find_class_end(self, lines: List[str], start_line: int) -> int:
        """查找类定义的结束行"""
        return self._find_function_end(lines, start_line)
    
    def _replace_line_range(self, lines: List[str], start_line: int, end_line: int, 
                          edit: SemanticEdit) -> Tuple[str, bool]:
        """替换指定行范围"""
        if edit.edit_type == SemanticEditType.DELETE:
            new_lines = lines[:start_line] + lines[end_line + 1:]
        else:
            # 保持原有缩进
            indent = self._get_line_indent(lines[start_line]) if start_line < len(lines) else ""
            new_code_lines = (edit.new_code or "").split('\n')
            indented_new_lines = [indent + line if line.strip() else line 
                                for line in new_code_lines]
            new_lines = lines[:start_line] + indented_new_lines + lines[end_line + 1:]
        
        return '\n'.join(new_lines), True
    
    def _get_line_indent(self, line: str) -> str:
        """获取行的缩进"""
        return line[:len(line) - len(line.lstrip())]
    
    def _is_valid_identifier(self, name: str) -> bool:
        """检查是否是有效的标识符"""
        return re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', name) is not None
    
    def _create_backup(self, file_path: str) -> Path:
        """创建文件备份"""
        import shutil
        
        file_path = Path(file_path)
        timestamp = int(time.time())
        backup_name = f"{file_path.name}.{timestamp}.backup"
        backup_path = self.backup_dir / backup_name
        
        shutil.copy2(file_path, backup_path)
        return backup_path
    
    def _restore_from_backup(self, file_path: str, backup_path: Path) -> bool:
        """从备份恢复文件"""
        try:
            import shutil
            shutil.copy2(backup_path, file_path)
            print(f"已从备份恢复文件: {backup_path}")
            return True
        except Exception as e:
            print(f"恢复备份失败: {e}")
            return False
    
    async def edit_code(self, request: str, file_path: str, cursor_position: Optional[int] = None, 
                       selection: Optional[str] = None, auto_apply: bool = False) -> Tuple[List[SemanticEdit], bool]:
        """
        编辑代码 - 兼容性方法
        
        Args:
            request: 编辑请求描述
            file_path: 文件路径
            cursor_position: 光标位置（暂未使用）
            selection: 选中的代码内容
            auto_apply: 是否自动应用编辑
            
        Returns:
            Tuple[List[SemanticEdit], bool]: (编辑操作列表, 是否成功)
        """
        try:
            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 基于请求和选中内容生成编辑操作
            edits = await self._generate_edits_from_request(request, file_path, content, selection)
            
            if auto_apply and edits:
                # 自动应用编辑
                success = self.apply_edits(file_path, edits)
                return edits, success
            else:
                # 只返回编辑操作，不应用
                return edits, True
                
        except Exception as e:
            print(f"编辑代码失败: {e}")
            return [], False
            
    async def _call_ai_api(self, prompt: str) -> str:
        """调用AI API - 参考项目中的实现"""
        import requests
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            },
            "parameters": {
                "temperature": 0.1,
                "max_tokens": 2000
            }
        }
        
        base_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
        
        # 使用同步请求（在实际应用中可以改为异步）
        response = requests.post(base_url, headers=headers, json=data)
        
        if response.status_code != 200:
            raise Exception(f"API调用失败: {response.status_code} - {response.text}")
        
        result = response.json()
        
        # 处理Dashscope API的响应格式
        if result.get("output", {}).get("text"):
            return result["output"]["text"]
        elif result.get("output", {}).get("choices"):
            # 兼容OpenAI格式
            return result["output"]["choices"][0]["message"]["content"]
        else:
            raise Exception(f"API返回格式错误: {result}")
    
    def _build_edit_analysis_prompt(self, request: str, file_path: str, content: str, selection: Optional[str] = None) -> str:
        """构建编辑分析提示词"""
        lines = content.split('\n')
        total_lines = len(lines)
        
        # 添加行号前缀，方便AI理解
        numbered_lines = []
        for i, line in enumerate(lines, 1):
            numbered_lines.append(f"{i:3d}: {line.rstrip()}")
        
        numbered_content = '\n'.join(numbered_lines)
        
        # 构建提示词
        prompt = f"""
你是一个专业的代码编辑助手。请分析以下代码修改请求，并生成详细的编辑操作。

## 修改请求
{request}

## 文件信息
文件路径: {file_path}
总行数: {total_lines}

## 完整文件内容（带行号）
{numbered_content}
"""
        
        # 如果有选中内容，添加到提示词中
        if selection and selection.strip():
            prompt += f"""

## 选中的代码上下文
{selection}
"""
        
        prompt += f"""

## 重要约束
1. 行号必须在1到{total_lines}之间
2. 对同一行只能有一个操作，不能重复修改
3. 仔细检查要修改的内容是否真的存在于文件中
4. 如果要删除包含特定内容的行，请先确认该内容确实存在
5. 优先使用符号名称而不是行号来定位代码

## 要求
1. 分析修改意图和范围
2. 确定需要修改的具体代码位置
3. 生成安全的编辑操作
4. 评估修改的影响和风险

请以JSON格式返回编辑计划，包含以下字段：
{{
    "description": "编辑计划描述",
    "operations": [
        {{
            "edit_type": "replace|insert|delete",
            "symbol_name": "符号名称（如函数名、类名等，可选）",
            "symbol_type": "符号类型（function|class|variable等，可选）",
            "start_line": 行号,
            "end_line": 行号,
            "old_code": "原始代码",
            "new_code": "新代码",
            "description": "操作描述",
            "confidence": 0.0-1.0
        }}
    ],
    "estimated_impact": "影响评估",
    "safety_score": 0.0-1.0
}}
"""
        
        return prompt
    
    def _parse_ai_edit_response(self, response: str) -> Dict[str, Any]:
        """解析AI编辑响应"""
        import re
        import json
        
        try:
            # 尝试提取JSON部分
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                
                # 尝试直接解析
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError as e:
                    print(f"JSON解析失败，尝试修复: {str(e)}")
                    
                    # 尝试修复常见的JSON格式问题
                    fixed_json = self._fix_json_format(json_str)
                    try:
                        return json.loads(fixed_json)
                    except json.JSONDecodeError:
                        print(f"JSON修复失败，原始响应: {response[:500]}...")
                        # 返回一个默认的编辑计划
                        return self._create_fallback_edit_plan()
            else:
                raise ValueError("响应中未找到有效的JSON")
        except Exception as e:
            print(f"解析AI响应时出错: {str(e)}")
            return self._create_fallback_edit_plan()
    
    def _fix_json_format(self, json_str: str) -> str:
        """尝试修复JSON格式问题"""
        # 移除可能的多余字符
        json_str = json_str.strip()
        
        # 修复常见的逗号问题
        # 在 } 或 ] 前面添加缺失的逗号
        json_str = re.sub(r'(["\d\]\}])\s*\n\s*(["\{\[])', r'\1,\n\2', json_str)
        
        # 移除多余的逗号（在 } 或 ] 前面）
        json_str = re.sub(r',(\s*[\}\]])', r'\1', json_str)
        
        # 修复引号问题
        json_str = re.sub(r'([{,]\s*)(\w+)(\s*:)', r'\1"\2"\3', json_str)
        
        return json_str
    
    def _create_fallback_edit_plan(self) -> Dict[str, Any]:
        """创建备用编辑计划"""
        return {
            "description": "AI解析失败，使用备用计划",
            "operations": [],
            "estimated_impact": "无影响",
            "safety_score": 1.0
        }
    
    def _convert_to_semantic_edits(self, analysis_result: Dict[str, Any], file_path: str) -> List[SemanticEdit]:
        """将AI分析结果转换为SemanticEdit对象"""
        edits = []
        
        for op_data in analysis_result.get("operations", []):
            try:
                # 创建位置信息
                location = CodeLocation(
                    file_path=file_path,
                    symbol_name=op_data.get('symbol_name'),
                    symbol_type=op_data.get('symbol_type'),
                    start_line=op_data.get('start_line'),
                    end_line=op_data.get('end_line')
                )
                
                # 创建编辑操作
                edit = SemanticEdit(
                    description=op_data.get('description', ''),
                    edit_type=SemanticEditType(op_data.get('edit_type', 'replace')),
                    location=location,
                    old_code=op_data.get('old_code', ''),
                    new_code=op_data.get('new_code', ''),
                    confidence=op_data.get('confidence', 0.8)
                )
                
                edits.append(edit)
                
            except Exception as e:
                print(f"转换编辑操作失败: {e}")
                continue
        
        return edits
    
    async def _generate_edits_from_request(self, request: str, file_path: str, 
                                         content: str, selection: Optional[str] = None) -> List[SemanticEdit]:
        """
        基于请求生成编辑操作
        集成AI分析，参考项目中的代码编辑流程
        """
        edits = []
        
        try:
            # 构建AI分析提示词
            prompt = self._build_edit_analysis_prompt(request, file_path, content, selection)
            
            # 调用AI API分析
            response = await self._call_ai_api(prompt)
            
            # 解析AI响应
            analysis_result = self._parse_ai_edit_response(response)
            
            # 转换为SemanticEdit对象
            edits = self._convert_to_semantic_edits(analysis_result, file_path)
            
            print(f"AI生成了 {len(edits)} 个编辑操作")
            
        except Exception as e:
            print(f"AI分析失败: {e}")
        
        return edits
    
    def get_edit_history(self) -> List[Dict[str, Any]]:
        """获取编辑历史"""
        return self.edit_history
    
    def rollback_last_edit(self) -> bool:
        """回滚最后一次编辑"""
        if not self.edit_history:
            print("没有可回滚的编辑历史")
            return False
        
        try:
            last_edit = self.edit_history[-1]
            backup_path = Path(last_edit['backup_path'])
            file_path = last_edit['file_path']
            
            success = self._restore_from_backup(file_path, backup_path)
            if success:
                self.edit_history.pop()  # 移除已回滚的编辑
                print(f"成功回滚编辑: {file_path}")
            
            return success
            
        except Exception as e:
            print(f"回滚编辑失败: {e}")
            return False


# 为了兼容性，保留原有的类名
SemanticCodeEditor = SmartSemanticEditor


# 使用示例
def demo():
    """演示智能语义编辑器的使用"""
    editor = SmartSemanticEditor()
    
    # 创建测试文件
    test_file = "test_demo.py"
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write('''def hello():
    print("Hello, World!")

class MyClass:
    def __init__(self):
        self.value = 42
    
    def get_value(self):
        return self.value
''')
    
    # 编辑1: 替换函数
    edit1 = SemanticEdit(
        description="修改hello函数",
        edit_type=SemanticEditType.REPLACE,
        location=CodeLocation(symbol_name="hello"),
        old_code='def hello():\n    print("Hello, World!")',
        new_code='def hello(name):\n    print(f"Hello, {name}!")'
    )
    
    success1 = editor.apply_edit(test_file, edit1)
    print(f"编辑1结果: {'成功' if success1 else '失败'}")
    
    # 编辑2: 修改方法
    edit2 = SemanticEdit(
        description="修改get_value方法",
        edit_type=SemanticEditType.REPLACE,
        location=CodeLocation(symbol_name="get_value"),
        old_code='    def get_value(self):\n        return self.value',
        new_code='    def get_value(self):\n        return self.value * 2'
    )
    
    success2 = editor.apply_edit(test_file, edit2)
    print(f"编辑2结果: {'成功' if success2 else '失败'}")
    
    # 显示最终结果
    with open(test_file, 'r', encoding='utf-8') as f:
        print("\n最终文件内容:")
        print(f.read())


if __name__ == "__main__":
    demo()