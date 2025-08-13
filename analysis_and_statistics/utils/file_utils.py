"""
File Utilities
==============

Common file operations for analysis and statistics module.
"""

import os
import json
import shutil
import tarfile
import zipfile
from typing import List, Dict, Any, Optional
from datetime import datetime


class FileUtils:
    """Utility functions for file operations"""
    
    @staticmethod
    def ensure_directory(directory: str) -> str:
        """Ensure directory exists, create if it doesn't
        
        Args:
            directory: Directory path to create
            
        Returns:
            Absolute path to directory
        """
        abs_path = os.path.abspath(directory)
        os.makedirs(abs_path, exist_ok=True)
        return abs_path
    
    @staticmethod
    def safe_remove(file_path: str) -> bool:
        """Safely remove a file
        
        Args:
            file_path: Path to file to remove
            
        Returns:
            True if successfully removed, False otherwise
        """
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                return True
            return False
        except Exception as e:
            print(f"⚠️  Error removing {file_path}: {e}")
            return False
    
    @staticmethod
    def get_file_size_mb(file_path: str) -> float:
        """Get file size in megabytes
        
        Args:
            file_path: Path to file
            
        Returns:
            File size in MB
        """
        try:
            if os.path.exists(file_path):
                return os.path.getsize(file_path) / (1024 * 1024)
            return 0.0
        except Exception:
            return 0.0
    
    @staticmethod
    def get_directory_size_mb(directory: str) -> float:
        """Get total size of directory in megabytes
        
        Args:
            directory: Directory path
            
        Returns:
            Total size in MB
        """
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(directory):
                for filename in filenames:
                    file_path = os.path.join(dirpath, filename)
                    if os.path.exists(file_path):
                        total_size += os.path.getsize(file_path)
            return total_size / (1024 * 1024)
        except Exception:
            return 0.0
    
    @staticmethod
    def compress_file(file_path: str, compression_type: str = "gzip") -> str:
        """Compress a file
        
        Args:
            file_path: Path to file to compress
            compression_type: Type of compression ("gzip" or "zip")
            
        Returns:
            Path to compressed file
        """
        try:
            if compression_type == "gzip":
                compressed_path = file_path + ".gz"
                with open(file_path, 'rb') as f_in:
                    with tarfile.open(compressed_path, 'w:gz') as tar:
                        tarinfo = tarfile.TarInfo(os.path.basename(file_path))
                        tarinfo.size = os.path.getsize(file_path)
                        tar.addfile(tarinfo, f_in)
                        
            elif compression_type == "zip":
                compressed_path = file_path + ".zip"
                with zipfile.ZipFile(compressed_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    zipf.write(file_path, os.path.basename(file_path))
            else:
                raise ValueError(f"Unsupported compression type: {compression_type}")
            
            return compressed_path
            
        except Exception as e:
            print(f"⚠️  Error compressing {file_path}: {e}")
            return file_path
    
    @staticmethod
    def decompress_file(compressed_path: str) -> str:
        """Decompress a file
        
        Args:
            compressed_path: Path to compressed file
            
        Returns:
            Path to decompressed file
        """
        try:
            if compressed_path.endswith('.gz'):
                decompressed_path = compressed_path[:-3]  # Remove .gz extension
                with tarfile.open(compressed_path, 'r:gz') as tar:
                    tar.extractall(path=os.path.dirname(decompressed_path))
                    
            elif compressed_path.endswith('.zip'):
                decompressed_path = compressed_path[:-4]  # Remove .zip extension
                with zipfile.ZipFile(compressed_path, 'r') as zipf:
                    zipf.extractall(path=os.path.dirname(decompressed_path))
            else:
                return compressed_path
            
            return decompressed_path
            
        except Exception as e:
            print(f"⚠️  Error decompressing {compressed_path}: {e}")
            return compressed_path
    
    @staticmethod
    def backup_file(file_path: str, backup_dir: str = None) -> str:
        """Create a backup copy of a file
        
        Args:
            file_path: Path to file to backup
            backup_dir: Directory to store backup (default: same as original)
            
        Returns:
            Path to backup file
        """
        try:
            if not os.path.exists(file_path):
                return ""
            
            if backup_dir is None:
                backup_dir = os.path.dirname(file_path)
            else:
                os.makedirs(backup_dir, exist_ok=True)
            
            # Create backup filename with timestamp
            filename = os.path.basename(file_path)
            name, ext = os.path.splitext(filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"{name}_backup_{timestamp}{ext}"
            backup_path = os.path.join(backup_dir, backup_filename)
            
            # Copy file
            shutil.copy2(file_path, backup_path)
            return backup_path
            
        except Exception as e:
            print(f"⚠️  Error backing up {file_path}: {e}")
            return ""
    
    @staticmethod
    def save_json(data: Dict[str, Any], file_path: str, indent: int = 2) -> bool:
        """Save data to JSON file
        
        Args:
            data: Data to save
            file_path: Path to JSON file
            indent: JSON indentation
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=indent, default=str)
            return True
            
        except Exception as e:
            print(f"⚠️  Error saving JSON to {file_path}: {e}")
            return False
    
    @staticmethod
    def load_json(file_path: str) -> Dict[str, Any]:
        """Load data from JSON file
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Loaded data or empty dict if error
        """
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    return json.load(f)
            return {}
            
        except Exception as e:
            print(f"⚠️  Error loading JSON from {file_path}: {e}")
            return {}
    
    @staticmethod
    def find_files_by_pattern(directory: str, pattern: str) -> List[str]:
        """Find files matching a pattern
        
        Args:
            directory: Directory to search
            pattern: File pattern (supports wildcards)
            
        Returns:
            List of matching file paths
        """
        import glob
        
        try:
            search_pattern = os.path.join(directory, pattern)
            return glob.glob(search_pattern)
        except Exception as e:
            print(f"⚠️  Error searching for pattern {pattern}: {e}")
            return []
    
    @staticmethod
    def cleanup_old_files(directory: str, pattern: str, keep_n: int = 10) -> int:
        """Clean up old files matching pattern
        
        Args:
            directory: Directory to clean
            pattern: File pattern to match
            keep_n: Number of newest files to keep
            
        Returns:
            Number of files removed
        """
        try:
            files = FileUtils.find_files_by_pattern(directory, pattern)
            if len(files) <= keep_n:
                return 0
            
            # Sort by modification time (newest first)
            files_with_time = [(f, os.path.getmtime(f)) for f in files]
            files_with_time.sort(key=lambda x: x[1], reverse=True)
            
            # Remove old files
            removed_count = 0
            for file_path, _ in files_with_time[keep_n:]:
                if FileUtils.safe_remove(file_path):
                    removed_count += 1
            
            return removed_count
            
        except Exception as e:
            print(f"⚠️  Error cleaning up files: {e}")
            return 0
    
    @staticmethod
    def get_file_info(file_path: str) -> Dict[str, Any]:
        """Get comprehensive file information
        
        Args:
            file_path: Path to file
            
        Returns:
            Dictionary with file information
        """
        try:
            if not os.path.exists(file_path):
                return {}
            
            stat = os.stat(file_path)
            
            return {
                'path': os.path.abspath(file_path),
                'name': os.path.basename(file_path),
                'size_bytes': stat.st_size,
                'size_mb': stat.st_size / (1024 * 1024),
                'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'is_file': os.path.isfile(file_path),
                'is_dir': os.path.isdir(file_path),
                'extension': os.path.splitext(file_path)[1]
            }
            
        except Exception as e:
            print(f"⚠️  Error getting file info for {file_path}: {e}")
            return {}
