"""
Performance Monitor for Efficient MedSAM2 Web Application
========================================================
Monitors system performance, memory usage, and inference metrics.
"""

import time
import psutil
import tracemalloc
import threading
import queue
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import numpy as np

class PerformanceMonitor:
    """Monitors performance metrics during inference"""
    
    def __init__(self, monitoring_interval: float = 0.1):
        """Initialize performance monitor"""
        self.monitoring_interval = monitoring_interval
        self.is_monitoring = False
        self.metrics_queue = queue.Queue()
        self.monitoring_thread = None
        
        # Performance history
        self.performance_history = []
        self.max_history_length = 1000
        
        # Current session metrics
        self.session_metrics = {
            'inference_count': 0,
            'total_inference_time': 0.0,
            'total_memory_peak': 0.0,
            'session_start': datetime.now()
        }
    
    def start_monitoring(self):
        """Start system performance monitoring"""
        if not self.is_monitoring:
            self.is_monitoring = True
            tracemalloc.start()
            self.start_time = time.time()
            
            # Start background monitoring thread
            self.monitoring_thread = threading.Thread(target=self._monitor_system_metrics)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
    
    def stop_monitoring(self) -> Dict:
        """Stop monitoring and return collected metrics"""
        if self.is_monitoring:
            self.is_monitoring = False
            
            # Get memory metrics
            try:
                current_memory, peak_memory = tracemalloc.get_traced_memory()
                tracemalloc.stop()
            except Exception:
                current_memory, peak_memory = 0, 0
            
            # Calculate total time
            total_time = time.time() - self.start_time
            
            # Collect system metrics from queue
            system_metrics = []
            while not self.metrics_queue.empty():
                try:
                    system_metrics.append(self.metrics_queue.get_nowait())
                except queue.Empty:
                    break
            
            # Calculate average system metrics
            avg_cpu_percent = np.mean([m['cpu_percent'] for m in system_metrics]) if system_metrics else 0
            avg_memory_percent = np.mean([m['memory_percent'] for m in system_metrics]) if system_metrics else 0
            
            metrics = {
                'total_time_seconds': total_time,
                'peak_memory_mb': peak_memory / (1024 * 1024),
                'current_memory_mb': current_memory / (1024 * 1024),
                'avg_cpu_percent': avg_cpu_percent,
                'avg_memory_percent': avg_memory_percent,
                'timestamp': datetime.now()
            }
            
            # Update session metrics
            self.session_metrics['inference_count'] += 1
            self.session_metrics['total_inference_time'] += total_time
            self.session_metrics['total_memory_peak'] = max(
                self.session_metrics['total_memory_peak'],
                metrics['peak_memory_mb']
            )
            
            # Add to history
            self._add_to_history(metrics)
            
            return metrics
        
        return {}
    
    def _monitor_system_metrics(self):
        """Background thread to monitor system metrics"""
        while self.is_monitoring:
            try:
                # Get CPU and memory usage
                cpu_percent = psutil.cpu_percent()
                memory = psutil.virtual_memory()
                
                system_metric = {
                    'timestamp': time.time(),
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'memory_available_mb': memory.available / (1024 * 1024),
                    'memory_used_mb': memory.used / (1024 * 1024)
                }
                
                self.metrics_queue.put(system_metric)
                
            except Exception:
                pass  # Ignore monitoring errors
            
            time.sleep(self.monitoring_interval)
    
    def _add_to_history(self, metrics: Dict):
        """Add metrics to performance history"""
        self.performance_history.append(metrics)
        
        # Limit history size
        if len(self.performance_history) > self.max_history_length:
            self.performance_history = self.performance_history[-self.max_history_length:]
    
    def get_session_statistics(self) -> Dict:
        """Get current session statistics"""
        session_duration = datetime.now() - self.session_metrics['session_start']
        
        avg_inference_time = (
            self.session_metrics['total_inference_time'] / 
            max(1, self.session_metrics['inference_count'])
        )
        
        return {
            'session_duration_minutes': session_duration.total_seconds() / 60,
            'total_inferences': self.session_metrics['inference_count'],
            'average_inference_time_seconds': avg_inference_time,
            'peak_memory_usage_mb': self.session_metrics['total_memory_peak'],
            'inferences_per_minute': (
                self.session_metrics['inference_count'] / 
                max(1, session_duration.total_seconds() / 60)
            )
        }
    
    def get_performance_trends(self, hours: int = 24) -> Dict:
        """Get performance trends over specified time period"""
        if not self.performance_history:
            return {}
        
        # Filter recent history
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_metrics = [
            m for m in self.performance_history 
            if m['timestamp'] > cutoff_time
        ]
        
        if not recent_metrics:
            return {}
        
        # Calculate trends
        inference_times = [m['total_time_seconds'] for m in recent_metrics]
        memory_usage = [m['peak_memory_mb'] for m in recent_metrics]
        cpu_usage = [m['avg_cpu_percent'] for m in recent_metrics]
        
        return {
            'period_hours': hours,
            'total_inferences': len(recent_metrics),
            'inference_time': {
                'mean': np.mean(inference_times),
                'std': np.std(inference_times),
                'min': np.min(inference_times),
                'max': np.max(inference_times),
                'percentile_95': np.percentile(inference_times, 95)
            },
            'memory_usage': {
                'mean': np.mean(memory_usage),
                'std': np.std(memory_usage),
                'min': np.min(memory_usage),
                'max': np.max(memory_usage),
                'percentile_95': np.percentile(memory_usage, 95)
            },
            'cpu_usage': {
                'mean': np.mean(cpu_usage),
                'std': np.std(cpu_usage),
                'min': np.min(cpu_usage),
                'max': np.max(cpu_usage)
            }
        }
    
    def get_system_info(self) -> Dict:
        """Get current system information"""
        try:
            # CPU information
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory information
            memory = psutil.virtual_memory()
            
            # Disk information
            disk = psutil.disk_usage('/')
            
            return {
                'cpu': {
                    'count': cpu_count,
                    'frequency_mhz': cpu_freq.current if cpu_freq else 0,
                    'usage_percent': cpu_percent
                },
                'memory': {
                    'total_gb': memory.total / (1024**3),
                    'available_gb': memory.available / (1024**3),
                    'used_gb': memory.used / (1024**3),
                    'usage_percent': memory.percent
                },
                'disk': {
                    'total_gb': disk.total / (1024**3),
                    'free_gb': disk.free / (1024**3),
                    'used_gb': disk.used / (1024**3),
                    'usage_percent': (disk.used / disk.total) * 100
                }
            }
        except Exception as e:
            return {'error': str(e)}
    
    def estimate_inference_capacity(self) -> Dict:
        """Estimate system capacity for inference operations"""
        system_info = self.get_system_info()
        
        if 'error' in system_info:
            return {'error': system_info['error']}
        
        # Estimate based on available memory and CPU
        available_memory_gb = system_info['memory']['available_gb']
        cpu_cores = system_info['cpu']['count']
        
        # Rough estimates based on model requirements
        estimated_memory_per_inference_mb = 50  # Conservative estimate
        max_concurrent_inferences = int(available_memory_gb * 1024 / estimated_memory_per_inference_mb)
        
        # CPU-based estimate (assuming each inference needs some CPU time)
        cpu_based_capacity = cpu_cores * 2  # Conservative multiplier
        
        # Take the minimum as the bottleneck
        estimated_capacity = min(max_concurrent_inferences, cpu_based_capacity)
        
        return {
            'estimated_max_concurrent_inferences': max(1, estimated_capacity),
            'memory_bottleneck': max_concurrent_inferences < cpu_based_capacity,
            'cpu_bottleneck': cpu_based_capacity < max_concurrent_inferences,
            'available_memory_gb': available_memory_gb,
            'cpu_cores': cpu_cores,
            'estimated_memory_per_inference_mb': estimated_memory_per_inference_mb
        }
    
    def create_performance_report(self) -> Dict:
        """Create comprehensive performance report"""
        return {
            'session_stats': self.get_session_statistics(),
            'recent_trends': self.get_performance_trends(hours=1),
            'system_info': self.get_system_info(),
            'capacity_estimate': self.estimate_inference_capacity(),
            'report_timestamp': datetime.now(),
            'total_history_entries': len(self.performance_history)
        }
    
    def export_performance_data(self, filename: str = None) -> str:
        """Export performance data to JSON format"""
        import json
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_report_{timestamp}.json"
        
        report = self.create_performance_report()
        
        # Convert datetime objects to strings for JSON serialization
        def json_serializer(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        
        try:
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=json_serializer)
            return filename
        except Exception as e:
            return f"Export failed: {str(e)}"
    
    def reset_session_metrics(self):
        """Reset session metrics"""
        self.session_metrics = {
            'inference_count': 0,
            'total_inference_time': 0.0,
            'total_memory_peak': 0.0,
            'session_start': datetime.now()
        }
    
    def clear_history(self):
        """Clear performance history"""
        self.performance_history.clear()