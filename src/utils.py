import re
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import hashlib
import mimetypes


def is_text_file(file_path: Path) -> bool:
    """判断文件是否为文本文件"""
    try:
        # 检查MIME类型
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if mime_type and mime_type.startswith('text/'):
            return True
        
        # 检查文件扩展名
        text_extensions = {
            '.txt', '.md', '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.h',
            '.go', '.rs', '.php', '.rb', '.swift', '.kt', '.scala', '.clj', '.hs', '.ml',
            '.css', '.scss', '.sass', '.less', '.html', '.htm', '.xml', '.json', '.yaml',
            '.yml', '.toml', '.ini', '.cfg', '.conf', '.sh', '.bat', '.ps1', '.sql',
            '.dockerfile', '.gitignore', '.gitattributes', '.editorconfig'
        }
        
        if file_path.suffix.lower() in text_extensions:
            return True
        
        # 尝试读取文件开头判断是否为文本
        with open(file_path, 'rb') as f:
            chunk = f.read(1024)
            if b'\0' in chunk:  # 包含空字节，可能是二进制文件
                return False
            
            # 尝试解码为UTF-8
            try:
                chunk.decode('utf-8')
                return True
            except UnicodeDecodeError:
                return False
                
    except Exception:
        return False


def should_ignore_file(file_path: Path, ignore_patterns: List[str]) -> bool:
    """判断文件是否应该被忽略"""
    file_str = str(file_path)
    
    for pattern in ignore_patterns:
        if pattern in file_str:
            return True
    
    # 检查隐藏文件（以.开头的文件和目录）
    for part in file_path.parts:
        if part.startswith('.') and part not in {'.', '..'}:
            # 允许一些常见的配置文件
            allowed_hidden = {'.gitignore', '.gitattributes', '.editorconfig', '.env'}
            if part not in allowed_hidden:
                return True
    
    return False


def calculate_file_hash(file_path: Path) -> str:
    """计算文件的SHA256哈希值"""
    try:
        with open(file_path, 'rb') as f:
            content = f.read()
            return hashlib.sha256(content).hexdigest()
    except Exception:
        return hashlib.sha256(b'').hexdigest()


def estimate_tokens(text: str) -> int:
    """估算文本的token数量（粗略估计）"""
    # 简单的token估算：平均4个字符 = 1个token
    return len(text) // 4


def clean_code_content(content: str) -> str:
    """清理代码内容，移除过多的空行和注释"""
    lines = content.split('\n')
    cleaned_lines = []
    
    prev_empty = False
    for line in lines:
        stripped = line.strip()
        
        # 跳过连续的空行
        if not stripped:
            if not prev_empty:
                cleaned_lines.append(line)
            prev_empty = True
        else:
            cleaned_lines.append(line)
            prev_empty = False
    
    return '\n'.join(cleaned_lines)


def extract_function_signature(code: str, language: str) -> Optional[str]:
    """提取函数签名"""
    patterns = {
        'python': [
            r'def\s+(\w+)\s*\([^)]*\)',
            r'async\s+def\s+(\w+)\s*\([^)]*\)',
            r'class\s+(\w+)\s*\([^)]*\):'
        ],
        'javascript': [
            r'function\s+(\w+)\s*\([^)]*\)',
            r'(\w+)\s*:\s*function\s*\([^)]*\)',
            r'(\w+)\s*=>\s*',
            r'class\s+(\w+)\s*{'
        ],
        'typescript': [
            r'function\s+(\w+)\s*\([^)]*\)',
            r'(\w+)\s*:\s*\([^)]*\)\s*=>\s*',
            r'class\s+(\w+)\s*{'
        ],
        'java': [
            r'(public|private|protected)?\s*(static)?\s*\w+\s+(\w+)\s*\([^)]*\)',
            r'class\s+(\w+)\s*{'
        ],
        'cpp': [
            r'\w+\s+(\w+)\s*\([^)]*\)',
            r'class\s+(\w+)\s*{'
        ]
    }
    
    if language not in patterns:
        return None
    
    for pattern in patterns[language]:
        match = re.search(pattern, code, re.MULTILINE)
        if match:
            return match.group(0)
    
    return None


def format_file_size(size_bytes: int) -> str:
    """格式化文件大小"""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"


def format_duration(seconds: float) -> str:
    """格式化时间间隔"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def validate_project_path(project_path: str) -> bool:
    """验证项目路径是否有效"""
    path = Path(project_path)
    
    if not path.exists():
        return False
    
    if not path.is_dir():
        return False
    
    # 检查是否包含代码文件
    code_extensions = {'.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.h', '.go'}
    
    for ext in code_extensions:
        if list(path.rglob(f'*{ext}')):
            return True
    
    return False


def get_project_info(project_path: str) -> Dict[str, Any]:
    """获取项目基本信息"""
    path = Path(project_path)
    
    if not path.exists():
        return {'error': 'Project path does not exist'}
    
    info = {
        'name': path.name,
        'path': str(path.resolve()),
        'size_bytes': 0,
        'file_count': 0,
        'code_file_count': 0,
        'languages': set(),
        'has_git': False,
        'has_package_json': False,
        'has_requirements_txt': False,
        'has_dockerfile': False
    }
    
    code_extensions = {
        '.py': 'Python',
        '.js': 'JavaScript',
        '.ts': 'TypeScript',
        '.jsx': 'JavaScript',
        '.tsx': 'TypeScript',
        '.java': 'Java',
        '.cpp': 'C++',
        '.c': 'C',
        '.h': 'C/C++',
        '.go': 'Go',
        '.rs': 'Rust',
        '.php': 'PHP',
        '.rb': 'Ruby',
        '.swift': 'Swift'
    }
    
    try:
        for file_path in path.rglob('*'):
            if file_path.is_file():
                info['file_count'] += 1
                
                try:
                    info['size_bytes'] += file_path.stat().st_size
                except OSError:
                    continue
                
                # 检查代码文件
                ext = file_path.suffix.lower()
                if ext in code_extensions:
                    info['code_file_count'] += 1
                    info['languages'].add(code_extensions[ext])
                
                # 检查特殊文件
                file_name = file_path.name.lower()
                if file_name == 'package.json':
                    info['has_package_json'] = True
                elif file_name == 'requirements.txt':
                    info['has_requirements_txt'] = True
                elif file_name in ['dockerfile', 'dockerfile.dev']:
                    info['has_dockerfile'] = True
            
            elif file_path.is_dir() and file_path.name == '.git':
                info['has_git'] = True
    
    except Exception as e:
        info['error'] = str(e)
    
    # 转换集合为列表
    info['languages'] = list(info['languages'])
    
    return info


def create_progress_bar(current: int, total: int, width: int = 50) -> str:
    """创建进度条"""
    if total == 0:
        return "[" + "=" * width + "] 100%"
    
    progress = current / total
    filled_width = int(width * progress)
    
    bar = "=" * filled_width + "-" * (width - filled_width)
    percentage = int(progress * 100)
    
    return f"[{bar}] {percentage}%"


def safe_read_file(file_path: Path, max_size_mb: int = 10) -> Optional[str]:
    """安全地读取文件内容"""
    try:
        # 检查文件大小
        file_size = file_path.stat().st_size
        if file_size > max_size_mb * 1024 * 1024:
            return None
        
        # 检查是否为文本文件
        if not is_text_file(file_path):
            return None
        
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    
    except Exception:
        return None


def normalize_path(path: str) -> str:
    """标准化路径"""
    return str(Path(path).resolve())


def get_relative_path(file_path: str, base_path: str) -> str:
    """获取相对路径"""
    try:
        return str(Path(file_path).relative_to(Path(base_path)))
    except ValueError:
        return file_path


def find_git_root(path: Path) -> Optional[Path]:
    """查找Git仓库根目录"""
    current = path.resolve()
    
    while current != current.parent:
        if (current / '.git').exists():
            return current
        current = current.parent
    
    return None


def get_git_info(project_path: str) -> Dict[str, Any]:
    """获取Git仓库信息"""
    try:
        import git
        
        git_root = find_git_root(Path(project_path))
        if not git_root:
            return {'has_git': False}
        
        repo = git.Repo(git_root)
        
        info = {
            'has_git': True,
            'root_path': str(git_root),
            'current_branch': repo.active_branch.name,
            'is_dirty': repo.is_dirty(),
            'untracked_files': len(repo.untracked_files),
            'total_commits': len(list(repo.iter_commits())),
        }
        
        # 获取最近的提交信息
        try:
            latest_commit = repo.head.commit
            info['latest_commit'] = {
                'sha': latest_commit.hexsha[:8],
                'message': latest_commit.message.strip(),
                'author': str(latest_commit.author),
                'date': latest_commit.committed_datetime.isoformat()
            }
        except Exception:
            pass
        
        return info
    
    except ImportError:
        return {'has_git': False, 'error': 'GitPython not installed'}
    except Exception as e:
        return {'has_git': False, 'error': str(e)}