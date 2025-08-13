"""
System Monitor
==============

System resource monitoring for training processes.
Tracks CPU, RAM, GPU usage and performance metrics.
"""

import time
import torch
import psutil
from typing import Dict, Tuple, Optional, List, Any
from datetime import datetime


class SystemMonitor:
    """Monitor system resources during training"""
    
    def __init__(self):
        self.start_time = time.time()
        self.gpu_available = torch.cuda.is_available()
        self.psutil_available = self._check_psutil()
        
        if self.gpu_available:
            self.device_count = torch.cuda.device_count()
            self.gpu_names = [torch.cuda.get_device_name(i) for i in range(self.device_count)]
        else:
            self.device_count = 0
            self.gpu_names = []
        
        print(f"🖥️  System Monitor initialized:")
        print(f"   - GPU available: {self.gpu_available} ({self.device_count} devices)")
        print(f"   - psutil available: {self.psutil_available}")
        if self.gpu_names:
            for i, name in enumerate(self.gpu_names):
                print(f"   - GPU {i}: {name}")
    
    def _check_psutil(self) -> bool:
        """Check if psutil is available for system monitoring"""
        try:
            import psutil
            return True
        except ImportError:
            return False
    
    def get_gpu_memory_info(self, device_id: int = 0) -> Dict[str, float]:
        """Get GPU memory information for specified device
        
        Args:
            device_id: GPU device ID
            
        Returns:
            Dictionary with memory info in GB
        """
        if not self.gpu_available:
            return {'allocated': 0.0, 'cached': 0.0, 'total': 0.0, 'free': 0.0}
        
        try:
            # Set device
            torch.cuda.set_device(device_id)
            
            # Get memory info
            allocated = torch.cuda.memory_allocated(device_id) / (1024**3)  # GB
            cached = torch.cuda.memory_reserved(device_id) / (1024**3)  # GB
            total = torch.cuda.get_device_properties(device_id).total_memory / (1024**3)  # GB
            free = total - allocated
            
            return {
                'allocated': allocated,
                'cached': cached, 
                'total': total,
                'free': free
            }
            
        except Exception as e:
            print(f"⚠️  Error getting GPU memory info: {e}")
            return {'allocated': 0.0, 'cached': 0.0, 'total': 0.0, 'free': 0.0}
    
    def get_gpu_utilization(self, device_id: int = 0) -> float:
        """Get GPU utilization percentage
        
        Args:
            device_id: GPU device ID
            
        Returns:
            GPU utilization percentage (0-100)
        """
        if not self.gpu_available:
            return 0.0
        
        try:
            # Note: PyTorch doesn't provide direct GPU utilization
            # This is a placeholder - in practice you might use nvidia-ml-py
            return 0.0  # Would need nvidia-ml-py for actual utilization
        except Exception:
            return 0.0
    
    def get_cpu_info(self) -> Dict[str, float]:
        """Get CPU usage information
        
        Returns:
            Dictionary with CPU usage info
        """
        if not self.psutil_available:
            return {'usage_percent': 0.0, 'freq_current': 0.0, 'freq_max': 0.0}
        
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_freq = psutil.cpu_freq()
            
            return {
                'usage_percent': cpu_percent,
                'freq_current': cpu_freq.current if cpu_freq else 0.0,
                'freq_max': cpu_freq.max if cpu_freq else 0.0
            }
            
        except Exception as e:
            print(f"⚠️  Error getting CPU info: {e}")
            return {'usage_percent': 0.0, 'freq_current': 0.0, 'freq_max': 0.0}
    
    def get_memory_info(self) -> Dict[str, float]:
        """Get RAM usage information
        
        Returns:
            Dictionary with memory usage info in GB
        """
        if not self.psutil_available:
            return {'total': 0.0, 'available': 0.0, 'used': 0.0, 'percent': 0.0}
        
        try:
            memory = psutil.virtual_memory()
            
            return {
                'total': memory.total / (1024**3),  # GB
                'available': memory.available / (1024**3),  # GB
                'used': memory.used / (1024**3),  # GB
                'percent': memory.percent
            }
            
        except Exception as e:
            print(f"⚠️  Error getting memory info: {e}")
            return {'total': 0.0, 'available': 0.0, 'used': 0.0, 'percent': 0.0}
    
    def get_disk_info(self, path: str = "/") -> Dict[str, float]:
        """Get disk usage information for given path
        
        Args:
            path: Path to check disk usage for
            
        Returns:
            Dictionary with disk usage info in GB
        """
        if not self.psutil_available:
            return {'total': 0.0, 'used': 0.0, 'free': 0.0, 'percent': 0.0}
        
        try:
            disk = psutil.disk_usage(path)
            
            return {
                'total': disk.total / (1024**3),  # GB
                'used': disk.used / (1024**3),  # GB
                'free': disk.free / (1024**3),  # GB
                'percent': (disk.used / disk.total) * 100
            }
            
        except Exception as e:
            print(f"⚠️  Error getting disk info: {e}")
            return {'total': 0.0, 'used': 0.0, 'free': 0.0, 'percent': 0.0}
    
    def get_process_info(self) -> Dict[str, float]:
        """Get current process resource usage
        
        Returns:
            Dictionary with process resource info
        """
        if not self.psutil_available:
            return {'cpu_percent': 0.0, 'memory_mb': 0.0, 'memory_percent': 0.0}
        
        try:
            process = psutil.Process()
            
            return {
                'cpu_percent': process.cpu_percent(),
                'memory_mb': process.memory_info().rss / (1024**2),  # MB
                'memory_percent': process.memory_percent()
            }
            
        except Exception as e:
            print(f"⚠️  Error getting process info: {e}")
            return {'cpu_percent': 0.0, 'memory_mb': 0.0, 'memory_percent': 0.0}
    
    def get_comprehensive_status(self, device_id: int = 0) -> Dict[str, Any]:
        """Get comprehensive system status
        
        Args:
            device_id: GPU device ID to monitor
            
        Returns:
            Dictionary with all system metrics
        """
        current_time = time.time()
        
        status = {
            'timestamp': datetime.now().isoformat(),
            'uptime_seconds': current_time - self.start_time,
            'gpu': {
                'available': self.gpu_available,
                'device_count': self.device_count,
                'memory': self.get_gpu_memory_info(device_id),
                'utilization': self.get_gpu_utilization(device_id)
            },
            'cpu': self.get_cpu_info(),
            'memory': self.get_memory_info(),
            'process': self.get_process_info()
        }
        
        return status
    
    def log_system_status(self, device_id: int = 0):
        """Print current system status to console
        
        Args:
            device_id: GPU device ID to monitor
        """
        status = self.get_comprehensive_status(device_id)
        
        print("🖥️  System Status:")
        print(f"   ⏱️  Uptime: {status['uptime_seconds']/3600:.1f} hours")
        
        if status['gpu']['available']:
            gpu_mem = status['gpu']['memory']
            print(f"   🎮 GPU {device_id}: {gpu_mem['allocated']:.1f}GB / {gpu_mem['total']:.1f}GB")
        
        cpu_info = status['cpu']
        print(f"   💻 CPU: {cpu_info['usage_percent']:.1f}%")
        
        mem_info = status['memory']
        print(f"   🧠 RAM: {mem_info['used']:.1f}GB / {mem_info['total']:.1f}GB ({mem_info['percent']:.1f}%)")
        
        proc_info = status['process']
        print(f"   📊 Process: CPU {proc_info['cpu_percent']:.1f}%, RAM {proc_info['memory_mb']:.0f}MB")
    
    def check_resource_warnings(self, device_id: int = 0) -> List[str]:
        """Check for resource usage warnings
        
        Args:
            device_id: GPU device ID to check
            
        Returns:
            List of warning messages
        """
        warnings = []
        status = self.get_comprehensive_status(device_id)
        
        # GPU memory warnings
        if status['gpu']['available']:
            gpu_mem = status['gpu']['memory']
            gpu_usage_percent = (gpu_mem['allocated'] / gpu_mem['total']) * 100
            
            if gpu_usage_percent > 90:
                warnings.append(f"⚠️  GPU memory critical: {gpu_usage_percent:.1f}% used")
            elif gpu_usage_percent > 80:
                warnings.append(f"⚠️  GPU memory high: {gpu_usage_percent:.1f}% used")
        
        # RAM warnings
        mem_percent = status['memory']['percent']
        if mem_percent > 90:
            warnings.append(f"⚠️  RAM critical: {mem_percent:.1f}% used")
        elif mem_percent > 80:
            warnings.append(f"⚠️  RAM high: {mem_percent:.1f}% used")
        
        # CPU warnings
        cpu_percent = status['cpu']['usage_percent']
        if cpu_percent > 95:
            warnings.append(f"⚠️  CPU critical: {cpu_percent:.1f}% used")
        
        return warnings
    
    def suggest_optimizations(self, device_id: int = 0) -> List[str]:
        """Suggest optimizations based on current resource usage
        
        Args:
            device_id: GPU device ID to analyze
            
        Returns:
            List of optimization suggestions
        """
        suggestions = []
        status = self.get_comprehensive_status(device_id)
        
        # GPU optimizations
        if status['gpu']['available']:
            gpu_mem = status['gpu']['memory']
            gpu_usage_percent = (gpu_mem['allocated'] / gpu_mem['total']) * 100
            
            if gpu_usage_percent > 85:
                suggestions.append("💡 Consider reducing batch size to free GPU memory")
                suggestions.append("💡 Enable mixed precision training (fp16)")
                suggestions.append("💡 Use gradient checkpointing to trade compute for memory")
        
        # RAM optimizations
        mem_percent = status['memory']['percent']
        if mem_percent > 85:
            suggestions.append("💡 Reduce number of data loader workers")
            suggestions.append("💡 Disable memory pinning if enabled")
            suggestions.append("💡 Consider using data streaming instead of loading all data")
        
        # CPU optimizations
        cpu_percent = status['cpu']['usage_percent']
        if cpu_percent > 90:
            suggestions.append("💡 Reduce data preprocessing complexity")
            suggestions.append("💡 Use fewer CPU workers for data loading")
        
        return suggestions
