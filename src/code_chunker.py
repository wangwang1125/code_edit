try:
    import tree_sitter
    from tree_sitter import Language, Parser
except ImportError:
    print("Warning: tree-sitter not installed. Install with: pip install tree-sitter")
    tree_sitter = None
    Language = None
    Parser = None

import os
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass
import hashlib
import re


@dataclass
class CodeChunk:
    """代码块数据结构"""
    content: str
    file_path: str
    start_line: int
    end_line: int
    chunk_type: str  # 'function', 'class', 'method', 'block', etc.
    language: str
    hash_value: str
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        if not self.hash_value:
            self.hash_value = hashlib.sha256(self.content.encode()).hexdigest()


class CodeChunker:
    """基于AST的智能代码分块器"""
    
    def __init__(self):
        self.parsers: Dict[str, Parser] = {}
        self.languages: Dict[str, Language] = {}
        self._setup_languages()
    
    def _setup_languages(self):
        """设置支持的编程语言解析器"""
        if not tree_sitter or not Parser or not Language:
            print("Warning: Tree-sitter not available. Falling back to text-based chunking.")
            return
            
        print("Info: Setting up AST parsers for semantic chunking...")
        
        # 语言映射配置
        language_configs = {
            'python': ('tree_sitter_python', ['.py']),
            'javascript': ('tree_sitter_javascript', ['.js', '.jsx']),
            'typescript': ('tree_sitter_typescript', ['.ts']),
            'tsx': ('tree_sitter_typescript', ['.tsx']),
            'java': ('tree_sitter_java', ['.java']),
            'cpp': ('tree_sitter_cpp', ['.cpp', '.cc', '.cxx', '.c++', '.hpp', '.h']),
            'c': ('tree_sitter_c', ['.c']),
            'go': ('tree_sitter_go', ['.go']),
            'html': ('tree_sitter_html', ['.html', '.htm']),
            'css': ('tree_sitter_css', ['.css', '.scss', '.sass', '.less']),
        }
        
        successful_languages = []
        
        for lang_name, (module_name, extensions) in language_configs.items():
            try:
                # 尝试导入语言模块
                language = None
                if module_name == 'tree_sitter_python':
                    try:
                        import tree_sitter_python as ts_python
                        language = Language(ts_python.language())
                    except ImportError:
                        print(f"Warning: tree-sitter-python not installed. Install with: pip install tree-sitter-python")
                        continue
                    except Exception as e:
                        print(f"Warning: Failed to load Python parser: {e}")
                        continue
                        
                elif module_name == 'tree_sitter_javascript':
                    try:
                        import tree_sitter_javascript as ts_javascript
                        language = Language(ts_javascript.language())
                    except ImportError:
                        print(f"Warning: tree-sitter-javascript not installed. Install with: pip install tree-sitter-javascript")
                        continue
                    except Exception as e:
                        print(f"Warning: Failed to load JavaScript parser: {e}")
                        continue
                        
                elif module_name == 'tree_sitter_typescript':
                    try:
                        import tree_sitter_typescript as ts_typescript
                        if lang_name == 'tsx':
                            language = Language(ts_typescript.language_tsx())
                        else:
                            language = Language(ts_typescript.language_typescript())
                    except ImportError:
                        print(f"Warning: tree-sitter-typescript not installed. Install with: pip install tree-sitter-typescript")
                        continue
                    except Exception as e:
                        print(f"Warning: Failed to load TypeScript parser: {e}")
                        continue
                        
                elif module_name == 'tree_sitter_java':
                    try:
                        import tree_sitter_java as ts_java
                        language = Language(ts_java.language())
                    except ImportError:
                        print(f"Warning: tree-sitter-java not installed. Install with: pip install tree-sitter-java")
                        continue
                    except Exception as e:
                        print(f"Warning: Failed to load Java parser: {e}")
                        continue
                        
                elif module_name == 'tree_sitter_cpp':
                    try:
                        import tree_sitter_cpp as ts_cpp
                        language = Language(ts_cpp.language())
                    except ImportError:
                        print(f"Warning: tree-sitter-cpp not installed. Install with: pip install tree-sitter-cpp")
                        continue
                    except Exception as e:
                        print(f"Warning: Failed to load C++ parser: {e}")
                        continue
                        
                elif module_name == 'tree_sitter_c':
                    try:
                        import tree_sitter_c as ts_c
                        language = Language(ts_c.language())
                    except ImportError:
                        print(f"Warning: tree-sitter-c not installed. Install with: pip install tree-sitter-c")
                        continue
                    except Exception as e:
                        print(f"Warning: Failed to load C parser: {e}")
                        continue
                        
                elif module_name == 'tree_sitter_go':
                    try:
                        import tree_sitter_go as ts_go
                        language = Language(ts_go.language())
                    except ImportError:
                        print(f"Warning: tree-sitter-go not installed. Install with: pip install tree-sitter-go")
                        continue
                    except Exception as e:
                        print(f"Warning: Failed to load Go parser: {e}")
                        continue
                        
                elif module_name == 'tree_sitter_html':
                    try:
                        import tree_sitter_html as ts_html
                        language = Language(ts_html.language())
                    except ImportError:
                        print(f"Warning: tree-sitter-html not installed. Install with: pip install tree-sitter-html")
                        continue
                    except Exception as e:
                        print(f"Warning: Failed to load HTML parser: {e}")
                        continue
                        
                elif module_name == 'tree_sitter_css':
                    try:
                        import tree_sitter_css as ts_css
                        language = Language(ts_css.language())
                    except ImportError:
                        print(f"Warning: tree-sitter-css not installed. Install with: pip install tree-sitter-css")
                        continue
                    except Exception as e:
                        print(f"Warning: Failed to load CSS parser: {e}")
                        continue
                
                if language:
                    parser = Parser()
                    # 使用正确的API设置语言
                    try:
                        parser.language = language
                    except AttributeError:
                        # 如果新API不可用，尝试旧API
                        try:
                            parser.set_language(language)
                        except AttributeError:
                            print(f"Warning: Cannot set language for {lang_name} parser")
                            continue
                    
                    self.languages[lang_name] = language
                    self.parsers[lang_name] = parser
                    
                    # 为每个文件扩展名映射到语言
                    for ext in extensions:
                        self.parsers[ext] = parser
                    
                    successful_languages.append(lang_name)
                    
            except Exception as e:
                print(f"Warning: Could not load {lang_name} parser: {e}")
        
        if successful_languages:
            print(f"Successfully loaded parsers for: {', '.join(successful_languages)}")
        else:
            print("Warning: No AST parsers loaded. Falling back to text-based chunking.")
    
    def _detect_language(self, file_path: str) -> Optional[str]:
        """根据文件扩展名检测编程语言"""
        ext = Path(file_path).suffix.lower()
        
        # 直接映射
        ext_to_lang = {
            '.py': 'python',
            '.js': 'javascript',
            '.jsx': 'javascript',
            '.ts': 'typescript',
            '.tsx': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.cc': 'cpp',
            '.cxx': 'cpp',
            '.c++': 'cpp',
            '.hpp': 'cpp',
            '.h': 'cpp',
            '.c': 'c',
            '.go': 'go',
            '.html': 'html',
            '.htm': 'html',
            '.css': 'css',
            '.scss': 'css',
            '.sass': 'css',
            '.less': 'css',
            '.json': 'json',
            '.xml': 'xml',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.md': 'markdown',
            '.txt': 'text',
            '.sql': 'sql',
            '.sh': 'bash',
            '.bat': 'batch',
            '.ps1': 'powershell',
        }
        
        return ext_to_lang.get(ext)
    
    def _get_node_text(self, node, source_code: bytes) -> str:
        """获取AST节点对应的源代码文本"""
        return source_code[node.start_byte:node.end_byte].decode('utf-8', errors='ignore')
    
    def _extract_node_name(self, node, source_code: bytes, language: str) -> Optional[str]:
        """提取节点名称（函数名、类名等）"""
        try:
            if language == 'python':
                if node.type in ['function_definition', 'async_function_definition']:
                    # 查找函数名
                    for child in node.children:
                        if child.type == 'identifier':
                            return self._get_node_text(child, source_code)
                elif node.type == 'class_definition':
                    # 查找类名
                    for child in node.children:
                        if child.type == 'identifier':
                            return self._get_node_text(child, source_code)
            
            elif language in ['javascript', 'typescript']:
                if node.type in ['function_declaration', 'method_definition']:
                    # 查找函数名
                    for child in node.children:
                        if child.type in ['identifier', 'property_identifier']:
                            return self._get_node_text(child, source_code)
                elif node.type == 'class_declaration':
                    # 查找类名
                    for child in node.children:
                        if child.type == 'type_identifier':
                            return self._get_node_text(child, source_code)
            
            elif language == 'java':
                if node.type == 'method_declaration':
                    # 查找方法名
                    for child in node.children:
                        if child.type == 'identifier':
                            return self._get_node_text(child, source_code)
                elif node.type in ['class_declaration', 'interface_declaration']:
                    # 查找类名或接口名
                    for child in node.children:
                        if child.type == 'identifier':
                            return self._get_node_text(child, source_code)
            
            elif language in ['cpp', 'c']:
                if node.type == 'function_definition':
                    # 查找函数名
                    for child in node.children:
                        if child.type == 'function_declarator':
                            for subchild in child.children:
                                if subchild.type == 'identifier':
                                    return self._get_node_text(subchild, source_code)
                elif node.type in ['class_specifier', 'struct_specifier']:
                    # 查找类名或结构体名
                    for child in node.children:
                        if child.type == 'type_identifier':
                            return self._get_node_text(child, source_code)
            
            elif language == 'go':
                if node.type == 'function_declaration':
                    # 查找函数名
                    for child in node.children:
                        if child.type == 'identifier':
                            return self._get_node_text(child, source_code)
                elif node.type == 'type_declaration':
                    # 查找类型名
                    for child in node.children:
                        if child.type == 'type_spec':
                            for subchild in child.children:
                                if subchild.type == 'type_identifier':
                                    return self._get_node_text(subchild, source_code)
        
        except Exception:
            pass
        
        return None
    
    def _extract_function_signature(self, node, source_code: bytes, language: str) -> Optional[str]:
        """提取函数签名"""
        try:
            if language == 'python':
                if node.type in ['function_definition', 'async_function_definition']:
                    # 提取函数定义行
                    lines = self._get_node_text(node, source_code).split('\n')
                    for line in lines:
                        if 'def ' in line and ':' in line:
                            return line.strip()
            
            elif language in ['javascript', 'typescript']:
                if node.type in ['function_declaration', 'method_definition']:
                    # 提取函数声明行
                    lines = self._get_node_text(node, source_code).split('\n')
                    if lines:
                        return lines[0].strip()
            
            elif language == 'java':
                if node.type == 'method_declaration':
                    # 提取方法声明
                    lines = self._get_node_text(node, source_code).split('\n')
                    signature_lines = []
                    for line in lines:
                        signature_lines.append(line.strip())
                        if '{' in line:
                            break
                    return ' '.join(signature_lines).replace('{', '').strip()
            
            # 对于其他语言，返回第一行
            lines = self._get_node_text(node, source_code).split('\n')
            if lines:
                return lines[0].strip()
        
        except Exception:
            pass
        
        return None
    
    
    def _extract_chunks_from_ast(self, tree, source_code: bytes, file_path: str, language: str, max_tokens: int = 1000) -> List[CodeChunk]:
        """从AST中提取代码块"""
        chunks = []
        source_lines = source_code.decode('utf-8', errors='ignore').split('\n')
        
        # 定义每种语言的重要节点类型
        important_nodes = {
            'python': {
                'primary': ['function_definition', 'async_function_definition', 'class_definition'],
                'secondary': ['if_statement', 'for_statement', 'while_statement', 'try_statement', 'with_statement']
            },
            'javascript': {
                'primary': ['function_declaration', 'function_expression', 'arrow_function', 'class_declaration', 'method_definition'],
                'secondary': ['if_statement', 'for_statement', 'while_statement', 'try_statement']
            },
            'typescript': {
                'primary': ['function_declaration', 'function_expression', 'arrow_function', 'class_declaration', 'method_definition', 'interface_declaration'],
                'secondary': ['if_statement', 'for_statement', 'while_statement', 'try_statement']
            },
            'java': {
                'primary': ['method_declaration', 'class_declaration', 'interface_declaration', 'constructor_declaration'],
                'secondary': ['if_statement', 'for_statement', 'while_statement', 'try_statement']
            },
            'cpp': {
                'primary': ['function_definition', 'class_specifier', 'struct_specifier', 'namespace_definition'],
                'secondary': ['if_statement', 'for_statement', 'while_statement', 'try_statement']
            },
            'c': {
                'primary': ['function_definition', 'struct_specifier'],
                'secondary': ['if_statement', 'for_statement', 'while_statement']
            },
            'go': {
                'primary': ['function_declaration', 'method_declaration', 'type_declaration'],
                'secondary': ['if_statement', 'for_statement', 'range_clause']
            },
            'html': {
                'primary': ['element', 'script_element', 'style_element'],
                'secondary': ['attribute', 'text']
            },
            'css': {
                'primary': ['rule_set', 'at_rule', 'media_statement'],
                'secondary': ['declaration', 'selector']
            }
        }
        
        lang_nodes = important_nodes.get(language, {'primary': [], 'secondary': []})
        
        def traverse_node(node, parent_context: Dict[str, Any] = None):
            """递归遍历AST节点"""
            if parent_context is None:
                parent_context = {}
            
            node_type = node.type
            
            # 检查是否是主要节点（函数、类等）
            if node_type in lang_nodes['primary']:
                content = self._get_node_text(node, source_code)
                estimated_tokens = len(content) // 4
                
                # 提取节点名称和签名
                node_name = self._extract_node_name(node, source_code, language)
                signature = self._extract_function_signature(node, source_code, language)
                
                # 构建增强的元数据
                metadata = {
                    'node_type': node_type,
                    'estimated_tokens': estimated_tokens,
                    'name': node_name,
                    'signature': signature,
                    'parent_context': parent_context.copy()
                }
                
                # 添加语言特定的元数据
                if language == 'python':
                    metadata['is_async'] = node_type == 'async_function_definition'
                    metadata['decorators'] = self._extract_decorators(node, source_code)
                elif language in ['javascript', 'typescript']:
                    metadata['is_arrow_function'] = node_type == 'arrow_function'
                    metadata['is_method'] = node_type == 'method_definition'
                elif language == 'java':
                    metadata['modifiers'] = self._extract_modifiers(node, source_code, language)
                
                if estimated_tokens <= max_tokens:
                    # 如果块大小合适，直接添加
                    chunk = CodeChunk(
                        content=content,
                        file_path=file_path,
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        chunk_type=node_type,
                        language=language,
                        hash_value="",
                        metadata=metadata
                    )
                    chunks.append(chunk)
                else:
                    # 如果块太大，尝试分割
                    sub_chunks = self._split_large_node(node, source_code, file_path, language, max_tokens, metadata)
                    chunks.extend(sub_chunks)
                
                # 更新父上下文
                new_context = parent_context.copy()
                if node_type == 'class_definition' or 'class' in node_type:
                    new_context['class_name'] = node_name
                elif 'function' in node_type or 'method' in node_type:
                    new_context['function_name'] = node_name
                
                # 继续遍历子节点，但不重复处理已经作为独立块的内容
                for child in node.children:
                    if child.type not in lang_nodes['primary']:
                        traverse_node(child, new_context)
            
            # 检查是否是次要节点（控制结构等）
            elif node_type in lang_nodes['secondary']:
                content = self._get_node_text(node, source_code)
                estimated_tokens = len(content) // 4
                
                if estimated_tokens > max_tokens // 2:  # 只有足够大的控制结构才单独成块
                    metadata = {
                        'node_type': node_type,
                        'estimated_tokens': estimated_tokens,
                        'parent_context': parent_context.copy()
                    }
                    
                    chunk = CodeChunk(
                        content=content,
                        file_path=file_path,
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        chunk_type=node_type,
                        language=language,
                        hash_value="",
                        metadata=metadata
                    )
                    chunks.append(chunk)
                else:
                    # 继续遍历子节点
                    for child in node.children:
                        traverse_node(child, parent_context)
            else:
                # 对于其他节点，继续遍历子节点
                for child in node.children:
                    traverse_node(child, parent_context)
        
        # 开始遍历
        traverse_node(tree.root_node)
        
        # 如果没有找到重要节点，使用智能文本分块
        if not chunks:
            chunks = self._intelligent_text_chunking(source_code.decode('utf-8'), file_path, language, max_tokens)
        
        # 后处理：合并小块，分割大块
        chunks = self._post_process_chunks(chunks, max_tokens)
        
        return chunks
    
    def _split_large_node(self, node, source_code: bytes, file_path: str, language: str, max_tokens: int, parent_metadata: Dict[str, Any]) -> List[CodeChunk]:
        """分割过大的节点"""
        chunks = []
        content = self._get_node_text(node, source_code)
        
        # 尝试按子节点分割
        for child in node.children:
            child_content = self._get_node_text(child, source_code)
            child_tokens = len(child_content) // 4
            
            if child_tokens > 50:  # 只处理有意义的子节点
                metadata = parent_metadata.copy()
                metadata.update({
                    'node_type': child.type,
                    'estimated_tokens': child_tokens,
                    'is_sub_chunk': True
                })
                
                chunk = CodeChunk(
                    content=child_content,
                    file_path=file_path,
                    start_line=child.start_point[0] + 1,
                    end_line=child.end_point[0] + 1,
                    chunk_type=child.type,
                    language=language,
                    hash_value="",
                    metadata=metadata
                )
                chunks.append(chunk)
        
        # 如果子节点分割不够，回退到行分割
        if not chunks:
            chunks = self._fallback_line_chunking(content, file_path, language, max_tokens)
        
        return chunks
    
    def _extract_decorators(self, node, source_code: bytes) -> List[str]:
        """提取Python装饰器"""
        decorators = []
        try:
            for child in node.children:
                if child.type == 'decorator':
                    decorator_text = self._get_node_text(child, source_code).strip()
                    decorators.append(decorator_text)
        except Exception:
            pass
        return decorators
    
    def _extract_modifiers(self, node, source_code: bytes, language: str) -> List[str]:
        """提取访问修饰符等"""
        modifiers = []
        try:
            if language == 'java':
                for child in node.children:
                    if child.type == 'modifiers':
                        modifier_text = self._get_node_text(child, source_code).strip()
                        modifiers.extend(modifier_text.split())
        except Exception:
            pass
        return modifiers
    
    def _intelligent_text_chunking(self, content: str, file_path: str, language: str, max_tokens: int) -> List[CodeChunk]:
        """智能文本分块（当AST解析失败时的回退方案）"""
        lines = content.split('\n')
        chunks = []
        current_chunk_lines = []
        current_tokens = 0
        start_line = 1  # 使用1索引，与编辑器行号一致
        
        # 语言特定的分块提示
        chunk_indicators = {
            'python': [r'def\s+\w+', r'class\s+\w+', r'if\s+__name__', r'@\w+'],
            'javascript': [r'function\s+\w+', r'class\s+\w+', r'const\s+\w+\s*=', r'export'],
            'java': [r'public\s+class', r'private\s+\w+', r'public\s+\w+', r'@\w+'],
            'cpp': [r'\w+::\w+', r'class\s+\w+', r'struct\s+\w+', r'namespace'],
            'go': [r'func\s+\w+', r'type\s+\w+', r'package\s+\w+'],
            'html': [r'<\w+[^>]*>', r'<script[^>]*>', r'<style[^>]*>', r'<!DOCTYPE', r'<!--'],
            'css': [r'[\w\-\.#]+\s*{', r'@media', r'@import', r'@keyframes', r'/\*']
        }
        
        indicators = chunk_indicators.get(language, [])
        
        for i, line in enumerate(lines):
            line_tokens = len(line) // 4
            current_line_number = i + 1  # 当前行号（1索引）
            
            # 检查是否是新的逻辑块开始
            is_new_block = False
            for pattern in indicators:
                if re.search(pattern, line):
                    is_new_block = True
                    break
            
            if (current_tokens + line_tokens > max_tokens and current_chunk_lines) or \
               (is_new_block and current_chunk_lines and current_tokens > max_tokens // 3):
                # 创建当前块
                chunk_content = '\n'.join(current_chunk_lines)
                end_line = start_line + len(current_chunk_lines) - 1
                
                # 验证行号范围
                if start_line <= end_line <= len(lines):
                    chunk = CodeChunk(
                        content=chunk_content,
                        file_path=file_path,
                        start_line=start_line,
                        end_line=end_line,
                        chunk_type='text_block',
                        language=language,
                        hash_value="",
                        metadata={
                            'estimated_tokens': current_tokens,
                            'chunking_method': 'intelligent_text',
                            'line_count': len(current_chunk_lines),
                            'validated_range': True
                        }
                    )
                    chunks.append(chunk)
                
                # 重置
                current_chunk_lines = [line]
                current_tokens = line_tokens
                start_line = current_line_number
            else:
                current_chunk_lines.append(line)
                current_tokens += line_tokens
        
        # 添加最后一个块
        if current_chunk_lines:
            chunk_content = '\n'.join(current_chunk_lines)
            end_line = start_line + len(current_chunk_lines) - 1
            
            # 验证行号范围
            if start_line <= end_line <= len(lines):
                chunk = CodeChunk(
                    content=chunk_content,
                    file_path=file_path,
                    start_line=start_line,
                    end_line=end_line,
                    chunk_type='text_block',
                    language=language,
                    hash_value="",
                    metadata={
                        'estimated_tokens': current_tokens,
                        'chunking_method': 'intelligent_text',
                        'line_count': len(current_chunk_lines),
                        'validated_range': True
                    }
                )
                chunks.append(chunk)
        
        return chunks
    
    def _post_process_chunks(self, chunks: List[CodeChunk], max_tokens: int) -> List[CodeChunk]:
        """后处理代码块：合并小块，确保质量"""
        if not chunks:
            return chunks
        
        processed_chunks = []
        i = 0
        
        while i < len(chunks):
            current_chunk = chunks[i]
            current_tokens = current_chunk.metadata.get('estimated_tokens', 0)
            
            # 如果当前块太小，尝试与下一个块合并
            if current_tokens < max_tokens // 4 and i + 1 < len(chunks):
                next_chunk = chunks[i + 1]
                next_tokens = next_chunk.metadata.get('estimated_tokens', 0)
                
                # 检查是否可以合并（同一文件，相邻行，总大小不超过限制）
                if (current_chunk.file_path == next_chunk.file_path and
                    current_chunk.end_line + 1 >= next_chunk.start_line and
                    current_tokens + next_tokens <= max_tokens):
                    
                    # 合并块
                    merged_content = current_chunk.content + '\n' + next_chunk.content
                    
                    # 计算合并后的行号范围
                    merged_start_line = current_chunk.start_line
                    merged_end_line = next_chunk.end_line
                    
                    # 验证行号范围的合理性
                    expected_line_count = merged_end_line - merged_start_line + 1
                    actual_line_count = len(merged_content.split('\n'))
                    
                    # 如果行数不匹配，重新计算
                    if abs(expected_line_count - actual_line_count) > 1:
                        merged_end_line = merged_start_line + actual_line_count - 1
                    
                    merged_metadata = current_chunk.metadata.copy()
                    merged_metadata['estimated_tokens'] = current_tokens + next_tokens
                    merged_metadata['merged_from'] = [current_chunk.chunk_type, next_chunk.chunk_type]
                    merged_metadata['line_count_verified'] = True
                    merged_metadata['merged_line_range'] = f"{merged_start_line}-{merged_end_line}"
                    
                    merged_chunk = CodeChunk(
                        content=merged_content,
                        file_path=current_chunk.file_path,
                        start_line=merged_start_line,
                        end_line=merged_end_line,
                        chunk_type='merged_block',
                        language=current_chunk.language,
                        hash_value="",
                        metadata=merged_metadata
                    )
                    
                    processed_chunks.append(merged_chunk)
                    i += 2  # 跳过下一个块
                    continue
            
            # 验证当前块的行号范围
            if current_chunk.content:
                actual_line_count = len(current_chunk.content.split('\n'))
                expected_line_count = current_chunk.end_line - current_chunk.start_line + 1
                
                # 如果行数不匹配，更新元数据标记
                if abs(expected_line_count - actual_line_count) > 1:
                    updated_metadata = current_chunk.metadata.copy()
                    updated_metadata['line_count_mismatch'] = True
                    updated_metadata['expected_lines'] = expected_line_count
                    updated_metadata['actual_lines'] = actual_line_count
                    
                    # 创建修正后的块
                    corrected_chunk = CodeChunk(
                        content=current_chunk.content,
                        file_path=current_chunk.file_path,
                        start_line=current_chunk.start_line,
                        end_line=current_chunk.start_line + actual_line_count - 1,
                        chunk_type=current_chunk.chunk_type,
                        language=current_chunk.language,
                        hash_value=current_chunk.hash_value,
                        metadata=updated_metadata
                    )
                    processed_chunks.append(corrected_chunk)
                else:
                    processed_chunks.append(current_chunk)
            else:
                processed_chunks.append(current_chunk)
            i += 1
         
        return processed_chunks
        
    def _detect_language(self, file_path: str) -> Optional[str]:
        """根据文件扩展名检测编程语言"""
        ext = Path(file_path).suffix.lower()
        
        # 直接映射
        ext_to_lang = {
            '.py': 'python',
            '.js': 'javascript',
            '.jsx': 'javascript',
            '.ts': 'typescript',
            '.tsx': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.cc': 'cpp',
            '.cxx': 'cpp',
            '.c++': 'cpp',
            '.hpp': 'cpp',
            '.h': 'cpp',
            '.c': 'c',
            '.go': 'go',
            '.html': 'html',
            '.htm': 'html',
            '.css': 'css',
            '.scss': 'css',
            '.sass': 'css',
            '.less': 'css',
            '.json': 'json',
            '.xml': 'xml',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.md': 'markdown',
            '.txt': 'text',
            '.sql': 'sql',
            '.sh': 'bash',
            '.bat': 'batch',
            '.ps1': 'powershell',
        }
        
        return ext_to_lang.get(ext)
    
    def _get_node_text(self, node, source_code: bytes) -> str:
        """获取AST节点对应的源代码文本"""
        return source_code[node.start_byte:node.end_byte].decode('utf-8', errors='ignore')
    
    def _extract_node_name(self, node, source_code: bytes, language: str) -> str:
        """提取节点名称（函数名、类名等）"""
        try:
            if language == 'python':
                if node.type in ['function_definition', 'async_function_definition']:
                    for child in node.children:
                        if child.type == 'identifier':
                            return self._get_node_text(child, source_code)
                elif node.type == 'class_definition':
                    for child in node.children:
                        if child.type == 'identifier':
                            return self._get_node_text(child, source_code)
            
            elif language in ['javascript', 'typescript']:
                if node.type in ['function_declaration', 'method_definition']:
                    for child in node.children:
                        if child.type == 'identifier':
                            return self._get_node_text(child, source_code)
                elif node.type == 'class_declaration':
                    for child in node.children:
                        if child.type == 'identifier':
                            return self._get_node_text(child, source_code)
                elif node.type == 'function_expression':
                    # 匿名函数
                    return '<anonymous>'
                elif node.type == 'arrow_function':
                    return '<arrow_function>'
            
            elif language == 'java':
                if node.type in ['method_declaration', 'constructor_declaration']:
                    for child in node.children:
                        if child.type == 'identifier':
                            return self._get_node_text(child, source_code)
                elif node.type in ['class_declaration', 'interface_declaration']:
                    for child in node.children:
                        if child.type == 'identifier':
                            return self._get_node_text(child, source_code)
            
            elif language in ['cpp', 'c']:
                if node.type == 'function_definition':
                    # C/C++函数定义较复杂，尝试找到函数名
                    for child in node.children:
                        if child.type == 'function_declarator':
                            for grandchild in child.children:
                                if grandchild.type == 'identifier':
                                    return self._get_node_text(grandchild, source_code)
                elif node.type in ['class_specifier', 'struct_specifier']:
                    for child in node.children:
                        if child.type == 'type_identifier':
                            return self._get_node_text(child, source_code)
            
            elif language == 'go':
                if node.type in ['function_declaration', 'method_declaration']:
                    for child in node.children:
                        if child.type == 'identifier':
                            return self._get_node_text(child, source_code)
                elif node.type == 'type_declaration':
                    for child in node.children:
                        if child.type == 'type_spec':
                            for grandchild in child.children:
                                if grandchild.type == 'type_identifier':
                                    return self._get_node_text(grandchild, source_code)
            
            elif language == 'html':
                if node.type == 'element':
                    # 提取HTML元素标签名
                    for child in node.children:
                        if child.type == 'start_tag':
                            for grandchild in child.children:
                                if grandchild.type == 'tag_name':
                                    return self._get_node_text(grandchild, source_code)
                elif node.type in ['script_element', 'style_element']:
                    return node.type.replace('_element', '')
            
            elif language == 'css':
                if node.type == 'rule_set':
                    # 提取CSS选择器
                    for child in node.children:
                        if child.type == 'selectors':
                            selector_text = self._get_node_text(child, source_code).strip()
                            return selector_text[:50] + '...' if len(selector_text) > 50 else selector_text
                elif node.type in ['at_rule', 'media_statement']:
                    content = self._get_node_text(node, source_code)
                    first_line = content.split('\n')[0].strip()
                    return first_line[:50] + '...' if len(first_line) > 50 else first_line
        
        except Exception:
            pass
        
        return '<unknown>'
    
    def _extract_function_signature(self, node, source_code: bytes, language: str) -> str:
        """提取函数签名"""
        try:
            if language == 'python':
                if node.type in ['function_definition', 'async_function_definition']:
                    # 提取函数定义行
                    for child in node.children:
                        if child.type == 'identifier':
                            # 找到函数名后，继续找参数列表
                            func_name = self._get_node_text(child, source_code)
                            for sibling in node.children:
                                if sibling.type == 'parameters':
                                    params = self._get_node_text(sibling, source_code)
                                    prefix = 'async def ' if node.type == 'async_function_definition' else 'def '
                                    return f"{prefix}{func_name}{params}"
            
            elif language in ['javascript', 'typescript']:
                if node.type == 'function_declaration':
                    # 提取整个函数声明的第一行
                    content = self._get_node_text(node, source_code)
                    first_line = content.split('\n')[0].strip()
                    if '{' in first_line:
                        return first_line[:first_line.index('{')].strip()
                    return first_line
                elif node.type == 'method_definition':
                    content = self._get_node_text(node, source_code)
                    first_line = content.split('\n')[0].strip()
                    if '{' in first_line:
                        return first_line[:first_line.index('{')].strip()
                    return first_line
            
            elif language == 'java':
                if node.type in ['method_declaration', 'constructor_declaration']:
                    content = self._get_node_text(node, source_code)
                    first_line = content.split('\n')[0].strip()
                    if '{' in first_line:
                        return first_line[:first_line.index('{')].strip()
                    return first_line
            
            elif language in ['cpp', 'c']:
                if node.type == 'function_definition':
                    content = self._get_node_text(node, source_code)
                    lines = content.split('\n')
                    # C/C++函数可能跨多行，找到包含参数的部分
                    signature_lines = []
                    for line in lines:
                        signature_lines.append(line.strip())
                        if '{' in line:
                            break
                    signature = ' '.join(signature_lines)
                    if '{' in signature:
                        signature = signature[:signature.index('{')].strip()
                    return signature
            
            elif language == 'go':
                if node.type in ['function_declaration', 'method_declaration']:
                    content = self._get_node_text(node, source_code)
                    first_line = content.split('\n')[0].strip()
                    if '{' in first_line:
                        return first_line[:first_line.index('{')].strip()
                    return first_line
        
        except Exception:
            pass
        
        # 回退：返回节点的第一行
        try:
            content = self._get_node_text(node, source_code)
            first_line = content.split('\n')[0].strip()
            return first_line[:100] + '...' if len(first_line) > 100 else first_line
        except Exception:
            return '<signature_unknown>'
    
    def _fallback_line_chunking(self, content: str, file_path: str, language: str, max_tokens: int) -> List[CodeChunk]:
        """回退方案：按行分块"""
        lines = content.split('\n')
        chunks = []
        current_chunk_lines = []
        current_tokens = 0
        start_line = 1
        
        for i, line in enumerate(lines):
            line_tokens = len(line) // 4  # 粗略估计
            
            if current_tokens + line_tokens > max_tokens and current_chunk_lines:
                # 创建当前块
                chunk_content = '\n'.join(current_chunk_lines)
                chunk = CodeChunk(
                    content=chunk_content,
                    file_path=file_path,
                    start_line=start_line,
                    end_line=start_line + len(current_chunk_lines) - 1,
                    chunk_type='block',
                    language=language,
                    hash_value="",
                    metadata={'estimated_tokens': current_tokens}
                )
                chunks.append(chunk)
                
                # 重置
                current_chunk_lines = [line]
                current_tokens = line_tokens
                start_line = i + 1
            else:
                current_chunk_lines.append(line)
                current_tokens += line_tokens
        
        # 添加最后一个块
        if current_chunk_lines:
            chunk_content = '\n'.join(current_chunk_lines)
            chunk = CodeChunk(
                content=chunk_content,
                file_path=file_path,
                start_line=start_line,
                end_line=start_line + len(current_chunk_lines) - 1,
                chunk_type='block',
                language=language,
                hash_value="",
                metadata={'estimated_tokens': current_tokens}
            )
            chunks.append(chunk)
        
        return chunks
    
    def chunk_file(self, file_path: str, max_tokens: int = 1000) -> List[CodeChunk]:
        """对单个文件进行分块"""
        try:
            with open(file_path, 'rb') as f:
                source_code = f.read()
        except (IOError, OSError) as e:
            print(f"Error reading file {file_path}: {e}")
            return []
        
        language = self._detect_language(file_path)
        if not language or language not in self.parsers:
            # 如果不支持该语言，使用简单的文本分块
            return self._fallback_line_chunking(
                source_code.decode('utf-8', errors='ignore'),
                file_path,
                language or 'text',
                max_tokens
            )
        
        try:
            parser = self.parsers[language]
            tree = parser.parse(source_code)
            
            return self._extract_chunks_from_ast(tree, source_code, file_path, language, max_tokens)
        
        except Exception as e:
            print(f"Error parsing {file_path} with {language} parser: {e}")
            # 回退到简单分块
            return self._fallback_line_chunking(
                source_code.decode('utf-8', errors='ignore'),
                file_path,
                language,
                max_tokens
            )
    
    def chunk_project(self, project_path: str, file_extensions: Optional[List[str]] = None, max_tokens: int = 1000) -> List[CodeChunk]:
        """对整个项目进行分块"""
        if file_extensions is None:
            file_extensions = [
                '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.h', '.go',
                '.html', '.htm', '.css', '.scss', '.sass', '.less', '.json', '.xml',
                '.yaml', '.yml', '.md', '.txt', '.sql', '.sh', '.bat', '.ps1'
            ]
        
        project_path = Path(project_path)
        all_chunks = []
        
        # 收集所有符合条件的文件
        files = []
        for ext in file_extensions:
            files.extend(project_path.rglob(f'*{ext}'))
        
        # 过滤文件
        ignore_patterns = {'.git', '__pycache__', 'node_modules', '.vscode', '.idea', 'dist', 'build'}
        filtered_files = []
        
        for file_path in files:
            if any(ignore_dir in file_path.parts for ignore_dir in ignore_patterns):
                continue
            if file_path.is_file():
                filtered_files.append(file_path)
        
        # 对每个文件进行分块
        for file_path in filtered_files:
            try:
                relative_path = str(file_path.relative_to(project_path))
                chunks = self.chunk_file(str(file_path), max_tokens)
                
                # 更新文件路径为相对路径
                for chunk in chunks:
                    chunk.file_path = relative_path
                
                all_chunks.extend(chunks)
                
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
                continue
        
        return all_chunks
    
    def get_chunk_summary(self, chunks: List[CodeChunk]) -> Dict[str, Any]:
        """获取分块统计信息"""
        if not chunks:
            return {}
        
        languages = {}
        chunk_types = {}
        total_tokens = 0
        
        for chunk in chunks:
            # 统计语言
            languages[chunk.language] = languages.get(chunk.language, 0) + 1
            
            # 统计块类型
            chunk_types[chunk.chunk_type] = chunk_types.get(chunk.chunk_type, 0) + 1
            
            # 统计token数
            total_tokens += chunk.metadata.get('estimated_tokens', 0)
        
        return {
            'total_chunks': len(chunks),
            'languages': languages,
            'chunk_types': chunk_types,
            'total_estimated_tokens': total_tokens,
            'average_tokens_per_chunk': total_tokens / len(chunks) if chunks else 0
        }