"""
Advanced Resource Monitoring for GAN Training
============================================

Comprehensive system resource monitoring including GPU memory, utilization,
temperature, and performance bottleneck detection.
"""

import psutil
import time
import threading
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import deque
import json
import os

try:
    import pynvml
    NVIDIA_ML_AVAILABLE = True
except ImportError:
    NVIDIA_ML_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class SystemSnapshot:
    """Single point-in-time system resource snapshot"""
    timestamp: float
    
    # CPU metrics
    cpu_percent: float
    cpu_count: int
    cpu_freq_current: float
    cpu_freq_max: float
    
    # Memory metrics
    ram_total_gb: float
    ram_used_gb: float
    ram_percent: float
    ram_available_gb: float
    
    # GPU metrics (if available)
    gpu_count: int
    gpu_memory_used_mb: List[float]
    gpu_memory_total_mb: List[float]
    gpu_utilization_percent: List[float]
    gpu_temperature_c: List[float]
    gpu_power_draw_w: List[float]
    
    # PyTorch GPU memory (if available)
    torch_gpu_allocated_mb: List[float]
    torch_gpu_cached_mb: List[float]
    
    # System I/O
    disk_usage_percent: float
    network_bytes_sent: int
    network_bytes_recv: int


@dataclass
class PerformanceAlert:
    """Performance alert/warning"""
    timestamp: float
    level: str  # "warning", "critical"
    category: str  # "memory", "gpu", "cpu", "thermal"
    message: str
    value: float
    threshold: float


class AdvancedResourceMonitor:
    """Advanced system resource monitoring for GAN training"""
    
    def __init__(self, monitoring_interval: float = 5.0, history_size: int = 720):
        """
        Initialize resource monitor
        
        Args:
            monitoring_interval: Seconds between measurements
            history_size: Number of snapshots to keep in memory (default: 1 hour at 5s intervals)
        """
        self.monitoring_interval = monitoring_interval
        self.history_size = history_size
        
        # Initialize hardware monitoring
        self._init_gpu_monitoring()
        
        # Storage
        self.snapshots: deque = deque(maxlen=history_size)
        self.alerts: deque = deque(maxlen=100)
        
        # Monitoring state
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # Performance thresholds
        self.thresholds = {
            'cpu_percent': 90.0,
            'ram_percent': 85.0,
            'gpu_memory_percent': 90.0,
            'gpu_utilization': 95.0,
            'gpu_temperature': 85.0,
            'disk_usage_percent': 90.0
        }
        
        # Baseline measurements
        self.baseline_snapshot: Optional[SystemSnapshot] = None
        self.training_start_time: Optional[float] = None
    
    def _init_gpu_monitoring(self):
        """Initialize GPU monitoring capabilities"""
        self.nvidia_ml_initialized = False
        self.gpu_count = 0
        
        if NVIDIA_ML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.gpu_count = pynvml.nvmlDeviceGetCount()
                self.nvidia_ml_initialized = True
                print(f"✅ NVIDIA-ML initialized: {self.gpu_count} GPU(s) detected")
            except Exception as e:
                print(f"⚠️ NVIDIA-ML initialization failed: {e}")
        else:
            print("⚠️ pynvml not available - GPU monitoring limited")
    
    def take_snapshot(self) -> SystemSnapshot:
        """Take a single system resource snapshot"""
        timestamp = time.time()
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()
        cpu_freq_current = cpu_freq.current if cpu_freq else 0.0
        cpu_freq_max = cpu_freq.max if cpu_freq else 0.0
        
        # Memory metrics
        memory = psutil.virtual_memory()
        ram_total_gb = memory.total / (1024**3)
        ram_used_gb = memory.used / (1024**3)
        ram_percent = memory.percent
        ram_available_gb = memory.available / (1024**3)
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        disk_usage_percent = disk.percent
        
        # Network metrics
        network = psutil.net_io_counters()
        network_bytes_sent = network.bytes_sent
        network_bytes_recv = network.bytes_recv
        
        # GPU metrics
        gpu_memory_used_mb = []
        gpu_memory_total_mb = []
        gpu_utilization_percent = []
        gpu_temperature_c = []
        gpu_power_draw_w = []
        torch_gpu_allocated_mb = []
        torch_gpu_cached_mb = []
        
        if self.nvidia_ml_initialized:
            for i in range(self.gpu_count):
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    
                    # Memory info
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    gpu_memory_used_mb.append(mem_info.used / (1024**2))
                    gpu_memory_total_mb.append(mem_info.total / (1024**2))
                    
                    # Utilization
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_utilization_percent.append(util.gpu)
                    
                    # Temperature
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    gpu_temperature_c.append(temp)
                    
                    # Power (if available)
                    try:
                        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
                        gpu_power_draw_w.append(power)
                    except:
                        gpu_power_draw_w.append(0.0)
                        
                except Exception as e:
                    # Fill with zeros on error
                    gpu_memory_used_mb.append(0.0)
                    gpu_memory_total_mb.append(0.0)
                    gpu_utilization_percent.append(0.0)
                    gpu_temperature_c.append(0.0)
                    gpu_power_draw_w.append(0.0)
        
        # PyTorch GPU memory (if available)
        if TORCH_AVAILABLE and torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                try:
                    allocated = torch.cuda.memory_allocated(i) / (1024**2)
                    cached = torch.cuda.memory_reserved(i) / (1024**2)
                    torch_gpu_allocated_mb.append(allocated)
                    torch_gpu_cached_mb.append(cached)
                except:
                    torch_gpu_allocated_mb.append(0.0)
                    torch_gpu_cached_mb.append(0.0)
        
        # Ensure lists have same length as gpu_count
        while len(torch_gpu_allocated_mb) < self.gpu_count:
            torch_gpu_allocated_mb.append(0.0)
            torch_gpu_cached_mb.append(0.0)
        
        snapshot = SystemSnapshot(
            timestamp=timestamp,
            cpu_percent=cpu_percent,
            cpu_count=cpu_count,
            cpu_freq_current=cpu_freq_current,
            cpu_freq_max=cpu_freq_max,
            ram_total_gb=ram_total_gb,
            ram_used_gb=ram_used_gb,
            ram_percent=ram_percent,
            ram_available_gb=ram_available_gb,
            gpu_count=self.gpu_count,
            gpu_memory_used_mb=gpu_memory_used_mb,
            gpu_memory_total_mb=gpu_memory_total_mb,
            gpu_utilization_percent=gpu_utilization_percent,
            gpu_temperature_c=gpu_temperature_c,
            gpu_power_draw_w=gpu_power_draw_w,
            torch_gpu_allocated_mb=torch_gpu_allocated_mb,
            torch_gpu_cached_mb=torch_gpu_cached_mb,
            disk_usage_percent=disk_usage_percent,
            network_bytes_sent=network_bytes_sent,
            network_bytes_recv=network_bytes_recv
        )
        
        return snapshot
    
    def start_monitoring(self):
        """Start continuous monitoring in background thread"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.training_start_time = time.time()
        
        # Take baseline snapshot
        self.baseline_snapshot = self.take_snapshot()
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        print(f"🔍 Resource monitoring started (interval: {self.monitoring_interval}s)")
    
    def stop_monitoring(self):
        """Stop continuous monitoring"""
        self.is_monitoring = False
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)
        
        print("🔍 Resource monitoring stopped")
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.is_monitoring:
            try:
                snapshot = self.take_snapshot()
                self.snapshots.append(snapshot)
                
                # Check for alerts
                self._check_alerts(snapshot)
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                print(f"⚠️ Monitoring error: {e}")
                time.sleep(self.monitoring_interval)
    
    def _check_alerts(self, snapshot: SystemSnapshot):
        """Check for performance alerts"""
        timestamp = snapshot.timestamp
        
        # CPU alert
        if snapshot.cpu_percent > self.thresholds['cpu_percent']:
            alert = PerformanceAlert(
                timestamp=timestamp,
                level="warning",
                category="cpu",
                message=f"High CPU usage: {snapshot.cpu_percent:.1f}%",
                value=snapshot.cpu_percent,
                threshold=self.thresholds['cpu_percent']
            )
            self.alerts.append(alert)
        
        # RAM alert
        if snapshot.ram_percent > self.thresholds['ram_percent']:
            alert = PerformanceAlert(
                timestamp=timestamp,
                level="critical" if snapshot.ram_percent > 95 else "warning",
                category="memory",
                message=f"High RAM usage: {snapshot.ram_percent:.1f}% ({snapshot.ram_used_gb:.1f}GB)",
                value=snapshot.ram_percent,
                threshold=self.thresholds['ram_percent']
            )
            self.alerts.append(alert)
        
        # GPU alerts
        for i, (used_mb, total_mb, utilization, temperature) in enumerate(zip(
            snapshot.gpu_memory_used_mb,
            snapshot.gpu_memory_total_mb,
            snapshot.gpu_utilization_percent,
            snapshot.gpu_temperature_c
        )):
            if total_mb > 0:  # GPU is available
                gpu_memory_percent = (used_mb / total_mb) * 100
                
                # GPU memory alert
                if gpu_memory_percent > self.thresholds['gpu_memory_percent']:
                    alert = PerformanceAlert(
                        timestamp=timestamp,
                        level="critical" if gpu_memory_percent > 98 else "warning",
                        category="gpu_memory",
                        message=f"GPU {i} high memory usage: {gpu_memory_percent:.1f}% ({used_mb:.0f}MB)",
                        value=gpu_memory_percent,
                        threshold=self.thresholds['gpu_memory_percent']
                    )
                    self.alerts.append(alert)
                
                # GPU temperature alert
                if temperature > self.thresholds['gpu_temperature']:
                    alert = PerformanceAlert(
                        timestamp=timestamp,
                        level="critical" if temperature > 90 else "warning",
                        category="thermal",
                        message=f"GPU {i} high temperature: {temperature}°C",
                        value=temperature,
                        threshold=self.thresholds['gpu_temperature']
                    )
                    self.alerts.append(alert)
    
    def get_current_stats(self) -> Dict:
        """Get current system statistics"""
        if not self.snapshots:
            return {"error": "No snapshots available"}
        
        latest = self.snapshots[-1]
        
        stats = {
            "timestamp": latest.timestamp,
            "cpu": {
                "usage_percent": latest.cpu_percent,
                "frequency_mhz": latest.cpu_freq_current,
                "cores": latest.cpu_count
            },
            "memory": {
                "total_gb": latest.ram_total_gb,
                "used_gb": latest.ram_used_gb,
                "usage_percent": latest.ram_percent,
                "available_gb": latest.ram_available_gb
            },
            "disk": {
                "usage_percent": latest.disk_usage_percent
            }
        }
        
        # Add GPU stats if available
        if latest.gpu_count > 0:
            stats["gpu"] = {
                "count": latest.gpu_count,
                "devices": []
            }
            
            for i in range(latest.gpu_count):
                gpu_stats = {
                    "id": i,
                    "memory": {
                        "used_mb": latest.gpu_memory_used_mb[i] if i < len(latest.gpu_memory_used_mb) else 0,
                        "total_mb": latest.gpu_memory_total_mb[i] if i < len(latest.gpu_memory_total_mb) else 0,
                        "usage_percent": (latest.gpu_memory_used_mb[i] / latest.gpu_memory_total_mb[i] * 100) 
                                       if i < len(latest.gpu_memory_used_mb) and i < len(latest.gpu_memory_total_mb) 
                                       and latest.gpu_memory_total_mb[i] > 0 else 0
                    },
                    "utilization_percent": latest.gpu_utilization_percent[i] if i < len(latest.gpu_utilization_percent) else 0,
                    "temperature_c": latest.gpu_temperature_c[i] if i < len(latest.gpu_temperature_c) else 0,
                    "power_draw_w": latest.gpu_power_draw_w[i] if i < len(latest.gpu_power_draw_w) else 0
                }
                
                # Add PyTorch memory if available
                if i < len(latest.torch_gpu_allocated_mb):
                    gpu_stats["torch_memory"] = {
                        "allocated_mb": latest.torch_gpu_allocated_mb[i],
                        "cached_mb": latest.torch_gpu_cached_mb[i]
                    }
                
                stats["gpu"]["devices"].append(gpu_stats)
        
        return stats
    
    def get_training_summary(self) -> Dict:
        """Get training session resource summary"""
        if not self.snapshots or not self.baseline_snapshot:
            return {"error": "Insufficient data for summary"}
        
        # Calculate statistics over training period
        cpu_usage = [s.cpu_percent for s in self.snapshots]
        ram_usage = [s.ram_percent for s in self.snapshots]
        
        summary = {
            "training_duration_minutes": (time.time() - self.training_start_time) / 60 if self.training_start_time else 0,
            "total_snapshots": len(self.snapshots),
            "cpu": {
                "mean_usage_percent": sum(cpu_usage) / len(cpu_usage),
                "max_usage_percent": max(cpu_usage),
                "baseline_usage_percent": self.baseline_snapshot.cpu_percent
            },
            "memory": {
                "mean_usage_percent": sum(ram_usage) / len(ram_usage),
                "max_usage_percent": max(ram_usage),
                "peak_usage_gb": max(s.ram_used_gb for s in self.snapshots),
                "baseline_usage_gb": self.baseline_snapshot.ram_used_gb
            },
            "alerts": {
                "total_count": len(self.alerts),
                "warning_count": sum(1 for a in self.alerts if a.level == "warning"),
                "critical_count": sum(1 for a in self.alerts if a.level == "critical")
            }
        }
        
        # GPU summary
        if self.snapshots[-1].gpu_count > 0:
            gpu_summary = {"devices": []}
            
            for i in range(self.snapshots[-1].gpu_count):
                gpu_memory_usage = []
                gpu_utilization = []
                gpu_temperature = []
                
                for snapshot in self.snapshots:
                    if i < len(snapshot.gpu_memory_used_mb) and i < len(snapshot.gpu_memory_total_mb):
                        if snapshot.gpu_memory_total_mb[i] > 0:
                            usage_percent = (snapshot.gpu_memory_used_mb[i] / snapshot.gpu_memory_total_mb[i]) * 100
                            gpu_memory_usage.append(usage_percent)
                    
                    if i < len(snapshot.gpu_utilization_percent):
                        gpu_utilization.append(snapshot.gpu_utilization_percent[i])
                    
                    if i < len(snapshot.gpu_temperature_c):
                        gpu_temperature.append(snapshot.gpu_temperature_c[i])
                
                device_summary = {
                    "id": i,
                    "memory_usage": {
                        "mean_percent": sum(gpu_memory_usage) / len(gpu_memory_usage) if gpu_memory_usage else 0,
                        "max_percent": max(gpu_memory_usage) if gpu_memory_usage else 0
                    },
                    "utilization": {
                        "mean_percent": sum(gpu_utilization) / len(gpu_utilization) if gpu_utilization else 0,
                        "max_percent": max(gpu_utilization) if gpu_utilization else 0
                    },
                    "temperature": {
                        "mean_c": sum(gpu_temperature) / len(gpu_temperature) if gpu_temperature else 0,
                        "max_c": max(gpu_temperature) if gpu_temperature else 0
                    }
                }
                
                gpu_summary["devices"].append(device_summary)
            
            summary["gpu"] = gpu_summary
        
        return summary
    
    def save_monitoring_data(self, output_path: str):
        """Save monitoring data to file"""
        try:
            data = {
                "metadata": {
                    "monitoring_interval": self.monitoring_interval,
                    "training_start_time": self.training_start_time,
                    "total_snapshots": len(self.snapshots),
                    "gpu_count": self.gpu_count
                },
                "summary": self.get_training_summary(),
                "snapshots": [asdict(snapshot) for snapshot in self.snapshots],
                "alerts": [asdict(alert) for alert in self.alerts]
            }
            
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"📊 Monitoring data saved to {output_path}")
            
        except Exception as e:
            print(f"❌ Failed to save monitoring data: {e}")
    
    def get_bottleneck_analysis(self) -> Dict:
        """Analyze potential performance bottlenecks"""
        if not self.snapshots:
            return {"error": "No data available"}
        
        # Recent data analysis (last 20 snapshots)
        recent_snapshots = list(self.snapshots)[-20:]
        
        bottlenecks = []
        
        # CPU bottleneck
        avg_cpu = sum(s.cpu_percent for s in recent_snapshots) / len(recent_snapshots)
        if avg_cpu > 80:
            bottlenecks.append({
                "type": "cpu",
                "severity": "high" if avg_cpu > 90 else "medium",
                "description": f"High CPU usage ({avg_cpu:.1f}% average)",
                "recommendation": "Consider reducing batch size or using more efficient data loading"
            })
        
        # Memory bottleneck
        avg_ram = sum(s.ram_percent for s in recent_snapshots) / len(recent_snapshots)
        if avg_ram > 80:
            bottlenecks.append({
                "type": "memory",
                "severity": "critical" if avg_ram > 95 else "high",
                "description": f"High RAM usage ({avg_ram:.1f}% average)",
                "recommendation": "Reduce batch size, enable gradient checkpointing, or add more RAM"
            })
        
        # GPU bottlenecks
        if self.gpu_count > 0:
            for i in range(self.gpu_count):
                gpu_memory_usage = []
                gpu_utilization = []
                
                for snapshot in recent_snapshots:
                    if (i < len(snapshot.gpu_memory_used_mb) and 
                        i < len(snapshot.gpu_memory_total_mb) and
                        snapshot.gpu_memory_total_mb[i] > 0):
                        usage_percent = (snapshot.gpu_memory_used_mb[i] / snapshot.gpu_memory_total_mb[i]) * 100
                        gpu_memory_usage.append(usage_percent)
                    
                    if i < len(snapshot.gpu_utilization_percent):
                        gpu_utilization.append(snapshot.gpu_utilization_percent[i])
                
                if gpu_memory_usage:
                    avg_gpu_memory = sum(gpu_memory_usage) / len(gpu_memory_usage)
                    if avg_gpu_memory > 85:
                        bottlenecks.append({
                            "type": f"gpu_{i}_memory",
                            "severity": "critical" if avg_gpu_memory > 95 else "high",
                            "description": f"GPU {i} high memory usage ({avg_gpu_memory:.1f}% average)",
                            "recommendation": "Reduce batch size, use gradient accumulation, or enable mixed precision"
                        })
                
                if gpu_utilization:
                    avg_gpu_util = sum(gpu_utilization) / len(gpu_utilization)
                    if avg_gpu_util < 70:
                        bottlenecks.append({
                            "type": f"gpu_{i}_underutilization",
                            "severity": "low",
                            "description": f"GPU {i} underutilized ({avg_gpu_util:.1f}% average)",
                            "recommendation": "Increase batch size, optimize data loading, or check for CPU bottlenecks"
                        })
        
        return {
            "analysis_window": len(recent_snapshots),
            "bottlenecks_detected": len(bottlenecks),
            "bottlenecks": bottlenecks
        }


# Global monitor instance
_global_monitor: Optional[AdvancedResourceMonitor] = None


def get_resource_monitor(start_monitoring: bool = True) -> AdvancedResourceMonitor:
    """Get global resource monitor instance"""
    global _global_monitor
    
    if _global_monitor is None:
        _global_monitor = AdvancedResourceMonitor()
        
        if start_monitoring:
            _global_monitor.start_monitoring()
    
    return _global_monitor


def cleanup_resource_monitor():
    """Cleanup global resource monitor"""
    global _global_monitor
    
    if _global_monitor is not None:
        _global_monitor.stop_monitoring()
        _global_monitor = None
