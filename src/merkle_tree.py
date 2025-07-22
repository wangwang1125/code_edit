import hashlib
import json
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import os


class MerkleNode:
    """默克尔树节点"""
    
    def __init__(self, hash_value: str, is_leaf: bool = False, file_path: Optional[str] = None):
        self.hash_value = hash_value
        self.is_leaf = is_leaf
        self.file_path = file_path
        self.children: List['MerkleNode'] = []
        self.parent: Optional['MerkleNode'] = None
    
    def add_child(self, child: 'MerkleNode'):
        """添加子节点"""
        child.parent = self
        self.children.append(child)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'hash_value': self.hash_value,
            'is_leaf': self.is_leaf,
            'file_path': self.file_path,
            'children': [child.to_dict() for child in self.children]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MerkleNode':
        """从字典创建节点"""
        node = cls(data['hash_value'], data['is_leaf'], data.get('file_path'))
        for child_data in data.get('children', []):
            child = cls.from_dict(child_data)
            node.add_child(child)
        return node


class MerkleTree:
    """默克尔树实现，用于高效检测文件变更"""
    
    def __init__(self):
        self.root: Optional[MerkleNode] = None
        self.file_hashes: Dict[str, str] = {}
        self.obfuscation_key: Optional[bytes] = None
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """计算文件的SHA256哈希值"""
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
                return hashlib.sha256(content).hexdigest()
        except (IOError, OSError):
            # 文件无法读取时返回空哈希
            return hashlib.sha256(b'').hexdigest()
    
    def _combine_hashes(self, left_hash: str, right_hash: str) -> str:
        """组合两个哈希值生成新的哈希"""
        combined = left_hash + right_hash
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def _obfuscate_path(self, file_path: str) -> str:
        """混淆文件路径以保护隐私"""
        if not self.obfuscation_key:
            return file_path
        
        from cryptography.fernet import Fernet
        fernet = Fernet(self.obfuscation_key)
        
        # 分割路径并分别加密
        parts = file_path.replace('\\', '/').split('/')
        obfuscated_parts = []
        
        for part in parts:
            if part and part not in ['.', '..']:
                encrypted = fernet.encrypt(part.encode()).decode()
                obfuscated_parts.append(encrypted)
            else:
                obfuscated_parts.append(part)
        
        return '/'.join(obfuscated_parts)
    
    def set_obfuscation_key(self, key: bytes):
        """设置路径混淆密钥"""
        self.obfuscation_key = key
    
    def build_tree(self, project_path: Path, file_extensions: Optional[List[str]] = None) -> MerkleNode:
        """构建默克尔树"""
        if file_extensions is None:
            file_extensions = ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.go', '.rs']
        
        # 收集所有符合条件的文件
        files = []
        for ext in file_extensions:
            files.extend(project_path.rglob(f'*{ext}'))
        
        # 过滤掉不需要的文件和目录
        filtered_files = []
        ignore_patterns = {'.git', '__pycache__', 'node_modules', '.vscode', '.idea', 'dist', 'build'}
        
        for file_path in files:
            # 检查是否在忽略的目录中
            if any(ignore_dir in file_path.parts for ignore_dir in ignore_patterns):
                continue
            
            if file_path.is_file():
                filtered_files.append(file_path)
        
        # 按路径排序确保一致性
        filtered_files.sort(key=lambda x: str(x))
        
        # 计算文件哈希并创建叶节点
        leaf_nodes = []
        for file_path in filtered_files:
            file_hash = self._calculate_file_hash(file_path)
            relative_path = str(file_path.relative_to(project_path))
            self.file_hashes[relative_path] = file_hash
            
            leaf_node = MerkleNode(
                hash_value=file_hash,
                is_leaf=True,
                file_path=relative_path
            )
            leaf_nodes.append(leaf_node)
        
        # 构建树结构
        if not leaf_nodes:
            # 空项目的情况
            self.root = MerkleNode(hashlib.sha256(b'empty').hexdigest())
            return self.root
        
        # 自底向上构建树
        current_level = leaf_nodes
        
        while len(current_level) > 1:
            next_level = []
            
            # 成对组合节点
            for i in range(0, len(current_level), 2):
                left_node = current_level[i]
                
                if i + 1 < len(current_level):
                    right_node = current_level[i + 1]
                    combined_hash = self._combine_hashes(left_node.hash_value, right_node.hash_value)
                else:
                    # 奇数个节点时，复制最后一个节点
                    right_node = left_node
                    combined_hash = self._combine_hashes(left_node.hash_value, left_node.hash_value)
                
                parent_node = MerkleNode(combined_hash)
                parent_node.add_child(left_node)
                if right_node != left_node:
                    parent_node.add_child(right_node)
                
                next_level.append(parent_node)
            
            current_level = next_level
        
        self.root = current_level[0]
        return self.root
    
    def get_changed_files(self, other_tree: 'MerkleTree') -> List[str]:
        """比较两个默克尔树，返回变更的文件列表"""
        changed_files = []
        
        # 比较文件哈希
        all_files = set(self.file_hashes.keys()) | set(other_tree.file_hashes.keys())
        
        for file_path in all_files:
            current_hash = self.file_hashes.get(file_path)
            other_hash = other_tree.file_hashes.get(file_path)
            
            if current_hash != other_hash:
                changed_files.append(file_path)
        
        return changed_files
    
    def save_to_file(self, file_path: Path):
        """保存默克尔树到文件"""
        data = {
            'root': self.root.to_dict() if self.root else None,
            'file_hashes': self.file_hashes
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def load_from_file(self, file_path: Path) -> bool:
        """从文件加载默克尔树"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if data['root']:
                self.root = MerkleNode.from_dict(data['root'])
            self.file_hashes = data.get('file_hashes', {})
            return True
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            return False
    
    def get_root_hash(self) -> Optional[str]:
        """获取根哈希值"""
        return self.root.hash_value if self.root else None
    
    def verify_integrity(self) -> bool:
        """验证树的完整性"""
        if not self.root:
            return True
        
        def verify_node(node: MerkleNode) -> bool:
            if node.is_leaf:
                return True
            
            if len(node.children) == 0:
                return True
            elif len(node.children) == 1:
                expected_hash = self._combine_hashes(
                    node.children[0].hash_value,
                    node.children[0].hash_value
                )
            else:
                expected_hash = self._combine_hashes(
                    node.children[0].hash_value,
                    node.children[1].hash_value
                )
            
            if node.hash_value != expected_hash:
                return False
            
            return all(verify_node(child) for child in node.children)
        
        return verify_node(self.root)