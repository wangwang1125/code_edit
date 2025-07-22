import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any
import time
from dataclasses import dataclass, asdict
import hashlib


@dataclass
class ProjectMetadata:
    """项目元数据"""
    project_path: str
    project_name: str
    last_indexed: float
    total_files: int
    total_chunks: int
    merkle_root_hash: Optional[str]
    obfuscation_key: Optional[str]
    settings: Dict[str, Any]


class StorageManager:
    """本地存储管理器"""
    
    def __init__(self, base_storage_path: Path):
        self.base_path = base_storage_path
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # 存储结构
        self.projects_dir = self.base_path / "projects"
        self.cache_dir = self.base_path / "cache"
        self.temp_dir = self.base_path / "temp"
        
        # 创建目录
        for dir_path in [self.projects_dir, self.cache_dir, self.temp_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def get_project_storage_path(self, project_path: str) -> Path:
        """获取项目的存储路径"""
        # 使用项目路径的哈希作为存储目录名
        project_hash = hashlib.sha256(project_path.encode()).hexdigest()[:16]
        project_name = Path(project_path).name
        storage_name = f"{project_name}_{project_hash}"
        return self.projects_dir / storage_name
    
    def save_project_metadata(self, project_path: str, metadata: ProjectMetadata):
        """保存项目元数据"""
        storage_path = self.get_project_storage_path(project_path)
        storage_path.mkdir(parents=True, exist_ok=True)
        
        metadata_file = storage_path / "metadata.json"
        
        try:
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(metadata), f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving project metadata: {e}")
    
    def load_project_metadata(self, project_path: str) -> Optional[ProjectMetadata]:
        """加载项目元数据"""
        storage_path = self.get_project_storage_path(project_path)
        metadata_file = storage_path / "metadata.json"
        
        if not metadata_file.exists():
            return None
        
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return ProjectMetadata(**data)
        except Exception as e:
            print(f"Error loading project metadata: {e}")
            return None
    
    def get_merkle_tree_path(self, project_path: str) -> Path:
        """获取默克尔树存储路径"""
        storage_path = self.get_project_storage_path(project_path)
        return storage_path / "merkle_tree.json"
    
    def get_chunks_path(self, project_path: str) -> Path:
        """获取代码块存储路径"""
        storage_path = self.get_project_storage_path(project_path)
        return storage_path / "chunks.json"
    
    def get_embeddings_cache_path(self, project_path: str) -> Path:
        """获取嵌入向量缓存路径"""
        storage_path = self.get_project_storage_path(project_path)
        return storage_path / "embeddings_cache.json"
    
    def get_vector_db_path(self, project_path: str) -> Path:
        """获取向量数据库存储路径"""
        storage_path = self.get_project_storage_path(project_path)
        vector_db_path = storage_path / "vector_db"
        vector_db_path.mkdir(parents=True, exist_ok=True)
        return vector_db_path
    
    def save_chunks(self, project_path: str, chunks: List[Any]):
        """保存代码块数据"""
        chunks_file = self.get_chunks_path(project_path)
        
        try:
            # 将CodeChunk对象转换为字典
            chunks_data = []
            for chunk in chunks:
                if hasattr(chunk, '__dict__'):
                    chunks_data.append(chunk.__dict__)
                else:
                    chunks_data.append(chunk)
            
            with open(chunks_file, 'w', encoding='utf-8') as f:
                json.dump(chunks_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"Error saving chunks: {e}")
    
    def load_chunks(self, project_path: str) -> List[Dict[str, Any]]:
        """加载代码块数据"""
        chunks_file = self.get_chunks_path(project_path)
        
        if not chunks_file.exists():
            return []
        
        try:
            with open(chunks_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading chunks: {e}")
            return []
    
    def list_indexed_projects(self) -> List[Dict[str, Any]]:
        """列出所有已索引的项目"""
        projects = []
        
        for project_dir in self.projects_dir.iterdir():
            if not project_dir.is_dir():
                continue
            
            metadata_file = project_dir / "metadata.json"
            if not metadata_file.exists():
                continue
            
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                # 添加存储信息
                metadata['storage_path'] = str(project_dir)
                metadata['storage_size_mb'] = self._get_directory_size(project_dir) / (1024 * 1024)
                
                projects.append(metadata)
                
            except Exception as e:
                print(f"Error reading project metadata from {project_dir}: {e}")
                continue
        
        # 按最后索引时间排序
        projects.sort(key=lambda x: x.get('last_indexed', 0), reverse=True)
        return projects
    
    def delete_project_data(self, project_path: str) -> bool:
        """删除项目的所有存储数据"""
        storage_path = self.get_project_storage_path(project_path)
        
        if not storage_path.exists():
            return False
        
        try:
            shutil.rmtree(storage_path)
            print(f"Deleted project data for: {project_path}")
            return True
        except Exception as e:
            print(f"Error deleting project data: {e}")
            return False
    
    def cleanup_temp_files(self, max_age_hours: int = 24):
        """清理临时文件"""
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        cleaned_count = 0
        
        for temp_file in self.temp_dir.iterdir():
            try:
                if current_time - temp_file.stat().st_mtime > max_age_seconds:
                    if temp_file.is_file():
                        temp_file.unlink()
                    elif temp_file.is_dir():
                        shutil.rmtree(temp_file)
                    cleaned_count += 1
            except Exception as e:
                print(f"Error cleaning temp file {temp_file}: {e}")
        
        if cleaned_count > 0:
            print(f"Cleaned up {cleaned_count} temporary files")
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """获取存储统计信息"""
        stats = {
            'total_projects': 0,
            'total_storage_mb': 0,
            'projects_dir_mb': 0,
            'cache_dir_mb': 0,
            'temp_dir_mb': 0
        }
        
        try:
            # 统计项目数量
            stats['total_projects'] = len([d for d in self.projects_dir.iterdir() if d.is_dir()])
            
            # 统计存储大小
            stats['projects_dir_mb'] = self._get_directory_size(self.projects_dir) / (1024 * 1024)
            stats['cache_dir_mb'] = self._get_directory_size(self.cache_dir) / (1024 * 1024)
            stats['temp_dir_mb'] = self._get_directory_size(self.temp_dir) / (1024 * 1024)
            stats['total_storage_mb'] = stats['projects_dir_mb'] + stats['cache_dir_mb'] + stats['temp_dir_mb']
            
        except Exception as e:
            print(f"Error calculating storage stats: {e}")
        
        return stats
    
    def _get_directory_size(self, directory: Path) -> int:
        """计算目录大小（字节）"""
        total_size = 0
        
        try:
            for item in directory.rglob('*'):
                if item.is_file():
                    total_size += item.stat().st_size
        except Exception:
            pass
        
        return total_size
    
    def export_project_data(self, project_path: str, export_path: Path) -> bool:
        """导出项目数据"""
        storage_path = self.get_project_storage_path(project_path)
        
        if not storage_path.exists():
            print(f"No data found for project: {project_path}")
            return False
        
        try:
            # 创建导出目录
            export_path.mkdir(parents=True, exist_ok=True)
            
            # 复制项目数据
            export_project_path = export_path / storage_path.name
            shutil.copytree(storage_path, export_project_path, dirs_exist_ok=True)
            
            print(f"Exported project data to: {export_project_path}")
            return True
            
        except Exception as e:
            print(f"Error exporting project data: {e}")
            return False
    
    def import_project_data(self, import_path: Path, project_path: str) -> bool:
        """导入项目数据"""
        if not import_path.exists():
            print(f"Import path does not exist: {import_path}")
            return False
        
        try:
            storage_path = self.get_project_storage_path(project_path)
            
            # 如果目标已存在，先备份
            if storage_path.exists():
                backup_path = storage_path.with_suffix('.backup')
                if backup_path.exists():
                    shutil.rmtree(backup_path)
                shutil.move(storage_path, backup_path)
            
            # 复制导入的数据
            shutil.copytree(import_path, storage_path)
            
            print(f"Imported project data from: {import_path}")
            return True
            
        except Exception as e:
            print(f"Error importing project data: {e}")
            return False


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.default_config = {
            'openai_api_key': '',
            'embedding_model': 'text-embedding-3-small',
            'max_tokens_per_chunk': 1000,
            'file_extensions': ['.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.h', '.go'],
            'ignore_patterns': ['.git', '__pycache__', 'node_modules', '.vscode', '.idea', 'dist', 'build'],
            'auto_update_interval_minutes': 10,
            'max_embedding_cache_age_days': 30,
            'batch_size': 10,
            'rate_limit_per_minute': 100
        }
        
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """加载配置"""
        if not self.config_path.exists():
            self.save_config(self.default_config)
            return self.default_config.copy()
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # 合并默认配置（添加新的配置项）
            merged_config = self.default_config.copy()
            merged_config.update(config)
            
            return merged_config
            
        except Exception as e:
            print(f"Error loading config: {e}")
            return self.default_config.copy()
    
    def save_config(self, config: Optional[Dict[str, Any]] = None):
        """保存配置"""
        if config is None:
            config = self.config
        
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any):
        """设置配置值"""
        self.config[key] = value
        self.save_config()
    
    def update(self, updates: Dict[str, Any]):
        """批量更新配置"""
        self.config.update(updates)
        self.save_config()
    
    def reset_to_default(self):
        """重置为默认配置"""
        self.config = self.default_config.copy()
        self.save_config()