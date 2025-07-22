#!/usr/bin/env python3
"""
AI Code Editor - 基于AI的智能代码编辑引擎
支持自然语言描述的代码修改需求
"""

import os
import re
import json
import time
import hashlib
import difflib
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum

import requests


class EditType(Enum):
    """编辑类型枚举"""
    REPLACE = "replace"  # 替换代码
    INSERT = "insert"   # 插入代码
    DELETE = "delete"   # 删除代码
    REFACTOR = "refactor"  # 重构代码
    ADD_FUNCTION = "add_function"  # 添加函数
    ADD_CLASS = "add_class"  # 添加类
    MODIFY_FUNCTION = "modify_function"  # 修改函数
    MODIFY_CLASS = "modify_class"  # 修改类


@dataclass
class CodeLocation:
    """代码位置信息"""
    file_path: str
    start_line: int
    end_line: int
    start_column: Optional[int] = None
    end_column: Optional[int] = None


@dataclass
class EditOperation:
    """编辑操作"""
    edit_type: EditType
    location: CodeLocation
    old_code: str
    new_code: str
    description: str
    confidence: float = 0.0


@dataclass
class EditPlan:
    """编辑计划"""
    operations: List[EditOperation]
    description: str
    estimated_impact: str
    safety_score: float
    requires_confirmation: bool = True


@dataclass
class EditResult:
    """编辑结果"""
    success: bool
    operations_applied: List[EditOperation]
    errors: List[str]
    warnings: List[str]
    backup_path: Optional[str] = None
    diff: Optional[str] = None


class AICodeAnalyzer:
    """AI代码分析器"""
    
    def __init__(self, api_key: str, model: str = "qwen-plus"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
    
    async def analyze_modification_request(self, 
                                         request: str, 
                                         context: str, 
                                         file_info: Dict[str, Any]) -> EditPlan:
        """分析修改请求，生成编辑计划"""
        
        prompt = self._build_analysis_prompt(request, context, file_info)
        
        try:
            response = await self._call_ai_api(prompt)
            plan_data = self._parse_ai_response(response)
            return self._create_edit_plan(plan_data, file_info)
        except Exception as e:
            raise Exception(f"AI分析失败: {str(e)}")
    
    def _build_analysis_prompt(self, request: str, context: str, file_info: Dict[str, Any]) -> str:
        """构建AI分析提示词"""
        # 获取完整文件内容
        file_content = self._get_file_content(file_info.get('file_path', ''))
        
        return f"""
你是一个专业的代码编辑助手。请分析以下代码修改请求，并生成详细的编辑计划。

## 修改请求
{request}

## 代码上下文
{context}

## 文件信息
文件路径: {file_info.get('file_path', 'unknown')}
编程语言: {file_info.get('language', 'unknown')}
总行数: {file_info.get('total_lines', 'unknown')}

## 完整文件内容
{file_content}

## 重要约束
1. 行号必须在1到{file_info.get('total_lines', 1)}之间
2. 对同一行只能有一个操作，不能重复修改
3. 仔细检查要修改的内容是否真的存在于文件中
4. 如果要删除包含特定内容的行，请先确认该内容确实存在

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
            "start_line": 行号,
            "end_line": 行号,
            "old_code": "原始代码",
            "new_code": "新代码",
            "description": "操作描述",
            "confidence": 0.0-1.0
        }}
    ],
    "estimated_impact": "影响评估",
    "safety_score": 0.0-1.0,
    "requires_confirmation": true/false
}}
"""
    
    def _get_file_content(self, file_path: str) -> str:
        """获取文件内容"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # 添加行号前缀，方便AI理解
            numbered_lines = []
            for i, line in enumerate(lines, 1):
                numbered_lines.append(f"{i:3d}: {line.rstrip()}")
            
            return '\n'.join(numbered_lines)
        except Exception as e:
            return f"无法读取文件: {str(e)}"
    
    async def _call_ai_api(self, prompt: str) -> str:
        """调用AI API"""
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
        
        # 使用同步请求（在实际应用中可以改为异步）
        response = requests.post(self.base_url, headers=headers, json=data)
        
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
    
    def _parse_ai_response(self, response: str) -> Dict[str, Any]:
        """解析AI响应"""
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
                        return self._create_fallback_plan()
            else:
                raise ValueError("响应中未找到有效的JSON")
        except Exception as e:
            print(f"解析AI响应时出错: {str(e)}")
            return self._create_fallback_plan()
    
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
    
    def _create_fallback_plan(self) -> Dict[str, Any]:
        """创建备用编辑计划"""
        return {
            "description": "AI解析失败，使用备用计划",
            "operations": [],
            "estimated_impact": "无影响",
            "safety_score": 1.0,
            "requires_confirmation": True
        }
    
    def _create_edit_plan(self, plan_data: Dict[str, Any], file_info: Dict[str, Any]) -> EditPlan:
        """创建编辑计划对象"""
        operations = []
        processed_lines = set()  # 记录已处理的行号，防止重复操作
        
        for op_data in plan_data.get("operations", []):
            start_line = op_data.get('start_line', 1)
            end_line = op_data.get('end_line', 1)
            
            # 检查行号范围是否有效
            total_lines = file_info.get('total_lines', 1)
            if start_line < 1 or end_line > total_lines or start_line > end_line:
                print(f"跳过无效行号范围的操作: {start_line}-{end_line} (文件总行数: {total_lines})")
                continue
            
            # 检查是否与已处理的行号重复
            line_range = set(range(start_line, end_line + 1))
            if line_range & processed_lines:
                print(f"跳过重复操作: 行 {start_line}-{end_line} 已被处理")
                continue
            
            # 记录已处理的行号
            processed_lines.update(line_range)
            
            location = CodeLocation(
                file_path=file_info.get('file_path', ''),
                start_line=start_line,
                end_line=end_line
            )
            
            operation = EditOperation(
                edit_type=EditType(op_data.get('edit_type', 'replace')),
                location=location,
                old_code=op_data.get('old_code', ''),
                new_code=op_data.get('new_code', ''),
                description=op_data.get('description', ''),
                confidence=op_data.get('confidence', 0.5)
            )
            operations.append(operation)
        
        return EditPlan(
            operations=operations,
            description=plan_data.get('description', ''),
            estimated_impact=plan_data.get('estimated_impact', ''),
            safety_score=plan_data.get('safety_score', 0.5),
            requires_confirmation=plan_data.get('requires_confirmation', True)
        )


class CodeEditor:
    """代码编辑器"""
    
    def __init__(self, backup_dir: Optional[Path] = None):
        self.backup_dir = backup_dir or Path(".") / ".cursor_like_client" / "backups"
        self.backup_dir.mkdir(parents=True, exist_ok=True)
    
    def apply_edit_plan(self, plan: EditPlan) -> EditResult:
        """应用编辑计划"""
        applied_operations = []
        errors = []
        warnings = []
        backup_paths = []
        
        try:
            # 按文件分组操作
            file_operations = self._group_operations_by_file(plan.operations)
            
            for file_path, operations in file_operations.items():
                try:
                    # 创建备份
                    backup_path = self._create_backup(file_path)
                    backup_paths.append(backup_path)
                    
                    # 应用文件级别的编辑
                    file_result = self._apply_file_operations(file_path, operations)
                    applied_operations.extend(file_result.operations_applied)
                    errors.extend(file_result.errors)
                    warnings.extend(file_result.warnings)
                    
                except Exception as e:
                    errors.append(f"处理文件 {file_path} 时出错: {str(e)}")
            
            # 生成差异报告
            diff = self._generate_diff_report(applied_operations)
            
            return EditResult(
                success=len(errors) == 0,
                operations_applied=applied_operations,
                errors=errors,
                warnings=warnings,
                backup_path=str(backup_paths[0]) if backup_paths else None,
                diff=diff
            )
            
        except Exception as e:
            return EditResult(
                success=False,
                operations_applied=applied_operations,
                errors=[f"编辑失败: {str(e)}"],
                warnings=warnings
            )
    
    def _group_operations_by_file(self, operations: List[EditOperation]) -> Dict[str, List[EditOperation]]:
        """按文件分组操作"""
        file_ops = {}
        for op in operations:
            file_path = op.location.file_path
            if file_path not in file_ops:
                file_ops[file_path] = []
            file_ops[file_path].append(op)
        
        # 按行号排序（从后往前，避免行号偏移）
        for file_path in file_ops:
            file_ops[file_path].sort(key=lambda x: x.location.start_line, reverse=True)
        
        return file_ops
    
    def _create_backup(self, file_path: str) -> str:
        """创建文件备份"""
        source_path = Path(file_path)
        if not source_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        # 生成备份文件名
        timestamp = int(time.time())
        backup_name = f"{source_path.name}.{timestamp}.backup"
        backup_path = self.backup_dir / backup_name
        
        # 复制文件
        import shutil
        shutil.copy2(source_path, backup_path)
        
        return str(backup_path)
    
    def _apply_file_operations(self, file_path: str, operations: List[EditOperation]) -> EditResult:
        """应用文件级别的编辑操作"""
        applied_ops = []
        errors = []
        warnings = []
        
        try:
            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # 应用每个操作
            for operation in operations:
                try:
                    lines = self._apply_single_operation(lines, operation)
                    applied_ops.append(operation)
                except Exception as e:
                    errors.append(f"应用操作失败 {operation.description}: {str(e)}")
            
            # 写回文件
            if applied_ops:  # 只有成功应用了操作才写回
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.writelines(lines)
            
        except Exception as e:
            errors.append(f"处理文件 {file_path} 失败: {str(e)}")
        
        return EditResult(
            success=len(errors) == 0,
            operations_applied=applied_ops,
            errors=errors,
            warnings=warnings
        )
    
    def _apply_single_operation(self, lines: List[str], operation: EditOperation) -> List[str]:
        """应用单个编辑操作"""
        start_line = operation.location.start_line - 1  # 转换为0索引
        end_line = operation.location.end_line - 1
        
        # 验证行号范围
        if start_line < 0 or end_line >= len(lines) or start_line > end_line:
            raise ValueError(f"无效的行号范围: {start_line+1}-{end_line+1}")
        
        if operation.edit_type == EditType.REPLACE:
            # 替换代码
            new_lines = operation.new_code.split('\n')
            if not operation.new_code.endswith('\n') and len(new_lines) > 0:
                new_lines = [line + '\n' for line in new_lines[:-1]] + [new_lines[-1]]
            else:
                new_lines = [line + '\n' for line in new_lines if line or new_lines.index(line) < len(new_lines) - 1]
            
            return lines[:start_line] + new_lines + lines[end_line + 1:]
        
        elif operation.edit_type == EditType.INSERT:
            # 插入代码
            new_lines = operation.new_code.split('\n')
            new_lines = [line + '\n' for line in new_lines if line]
            return lines[:start_line] + new_lines + lines[start_line:]
        
        elif operation.edit_type == EditType.DELETE:
            # 删除代码
            return lines[:start_line] + lines[end_line + 1:]
        
        else:
            # 其他类型暂时按替换处理
            return self._apply_single_operation(
                lines, 
                EditOperation(
                    EditType.REPLACE, 
                    operation.location, 
                    operation.old_code, 
                    operation.new_code, 
                    operation.description
                )
            )
    
    def _generate_diff_report(self, operations: List[EditOperation]) -> str:
        """生成差异报告"""
        if not operations:
            return "无修改"
        
        diff_lines = []
        diff_lines.append("=== 代码修改报告 ===\n")
        
        for i, op in enumerate(operations, 1):
            diff_lines.append(f"\n{i}. {op.description}")
            diff_lines.append(f"   文件: {op.location.file_path}")
            diff_lines.append(f"   位置: 第{op.location.start_line}-{op.location.end_line}行")
            diff_lines.append(f"   类型: {op.edit_type.value}")
            
            if op.old_code and op.new_code:
                # 生成详细的diff
                old_lines = op.old_code.split('\n')
                new_lines = op.new_code.split('\n')
                
                diff = list(difflib.unified_diff(
                    old_lines, new_lines,
                    fromfile='原始代码',
                    tofile='修改后代码',
                    lineterm=''
                ))
                
                if diff:
                    diff_lines.append("   差异:")
                    for line in diff[2:]:  # 跳过文件头
                        diff_lines.append(f"   {line}")
        
        return '\n'.join(diff_lines)
    
    def rollback_from_backup(self, backup_path: str, target_path: str) -> bool:
        """从备份恢复文件"""
        try:
            import shutil
            shutil.copy2(backup_path, target_path)
            return True
        except Exception as e:
            print(f"恢复失败: {str(e)}")
            return False


class AICodeEditor:
    """AI代码编辑器主类"""
    
    def __init__(self, api_key: str, model: str = "qwen-plus", backup_dir: Optional[Path] = None):
        self.analyzer = AICodeAnalyzer(api_key, model)
        self.editor = CodeEditor(backup_dir)
        self.edit_history: List[EditResult] = []
    
    async def edit_code(self, 
                       request: str, 
                       context: str, 
                       file_info: Dict[str, Any],
                       auto_apply: bool = False) -> Tuple[EditPlan, Optional[EditResult]]:
        """编辑代码的主入口"""
        
        # 1. 分析修改请求
        plan = await self.analyzer.analyze_modification_request(request, context, file_info)
        
        # 2. 安全检查
        if plan.safety_score < 0.3:
            raise Exception(f"修改风险过高 (安全分数: {plan.safety_score})")
        
        # 3. 应用编辑（如果自动应用或不需要确认）
        result = None
        if auto_apply or not plan.requires_confirmation:
            result = self.editor.apply_edit_plan(plan)
            self.edit_history.append(result)
        
        return plan, result
    
    def apply_plan(self, plan: EditPlan) -> EditResult:
        """手动应用编辑计划"""
        result = self.editor.apply_edit_plan(plan)
        self.edit_history.append(result)
        return result
    
    def get_edit_history(self) -> List[EditResult]:
        """获取编辑历史"""
        return self.edit_history.copy()
    
    def rollback_last_edit(self) -> bool:
        """回滚最后一次编辑"""
        if not self.edit_history:
            return False
        
        last_result = self.edit_history[-1]
        if not last_result.backup_path:
            return False
        
        # 回滚所有修改的文件
        success = True
        for op in last_result.operations_applied:
            file_path = op.location.file_path
            if not self.editor.rollback_from_backup(last_result.backup_path, file_path):
                success = False
        
        return success


# 使用示例
if __name__ == "__main__":
    import asyncio
    
    async def example_usage():
        # 初始化编辑器
        api_key = os.getenv('DASHSCOPE_API_KEY')
        if not api_key:
            print("请设置 DASHSCOPE_API_KEY 环境变量")
            return
        
        editor = AICodeEditor(api_key)
        
        # 示例：修改请求
        request = "将这个函数的返回值从字符串改为字典格式"
        context = """
def get_user_info(user_id):
    # 获取用户信息
    name = "张三"
    age = 25
    return f"用户: {name}, 年龄: {age}"
"""
        
        file_info = {
            'file_path': 'example.py',
            'language': 'python',
            'total_lines': 10
        }
        
        try:
            # 生成编辑计划
            plan, result = await editor.edit_code(request, context, file_info, auto_apply=False)
            
            print("=== 编辑计划 ===")
            print(f"描述: {plan.description}")
            print(f"安全分数: {plan.safety_score}")
            print(f"需要确认: {plan.requires_confirmation}")
            
            for i, op in enumerate(plan.operations, 1):
                print(f"\n操作 {i}:")
                print(f"  类型: {op.edit_type.value}")
                print(f"  位置: 第{op.location.start_line}-{op.location.end_line}行")
                print(f"  描述: {op.description}")
                print(f"  置信度: {op.confidence}")
            
        except Exception as e:
            print(f"编辑失败: {str(e)}")
    
    # 运行示例
    # asyncio.run(example_usage())