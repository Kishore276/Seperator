"""
Deployment Optimization for Plant Classification System
Includes model quantization, caching, and performance monitoring
"""

import os
import json
import time
import psutil
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import redis
import hashlib
from functools import wraps
from typing import Dict, Any, Optional
import logging
from datetime import datetime, timedelta
import pickle
import threading
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelOptimizer:
    """Optimize models for production deployment"""
    
    def __init__(self):
        self.optimized_models = {}
    
    def quantize_model(self, model_path: str, output_path: str, 
                      quantization_type: str = 'dynamic') -> str:
        """
        Quantize model for faster inference
        
        Args:
            model_path: Path to original model
            output_path: Path to save quantized model
            quantization_type: Type of quantization ('dynamic', 'int8', 'float16')
        """
        logger.info(f"Quantizing model: {model_path}")
        
        try:
            # Load original model
            model = keras.models.load_model(model_path)
            
            if quantization_type == 'dynamic':
                # Dynamic range quantization
                converter = tf.lite.TFLiteConverter.from_keras_model(model)
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                quantized_model = converter.convert()
                
            elif quantization_type == 'int8':
                # Integer quantization (requires representative dataset)
                converter = tf.lite.TFLiteConverter.from_keras_model(model)
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_types = [tf.int8]
                quantized_model = converter.convert()
                
            elif quantization_type == 'float16':
                # Float16 quantization
                converter = tf.lite.TFLiteConverter.from_keras_model(model)
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_types = [tf.float16]
                quantized_model = converter.convert()
            
            # Save quantized model
            with open(output_path, 'wb') as f:
                f.write(quantized_model)
            
            # Compare model sizes
            original_size = os.path.getsize(model_path)
            quantized_size = len(quantized_model)
            compression_ratio = original_size / quantized_size
            
            logger.info(f"Model quantized successfully!")
            logger.info(f"Original size: {original_size / 1024 / 1024:.2f} MB")
            logger.info(f"Quantized size: {quantized_size / 1024 / 1024:.2f} MB")
            logger.info(f"Compression ratio: {compression_ratio:.2f}x")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Model quantization failed: {e}")
            return model_path
    
    def optimize_for_inference(self, model_path: str) -> keras.Model:
        """Optimize model for inference"""
        model = keras.models.load_model(model_path)
        
        # Compile with optimized settings for inference
        model.compile(
            optimizer='adam',  # Optimizer doesn't matter for inference
            loss='categorical_crossentropy',
            metrics=['accuracy'],
            run_eagerly=False  # Ensure graph mode
        )
        
        return model

class CacheManager:
    """Manage caching for improved performance"""
    
    def __init__(self, cache_type: str = 'memory', redis_host: str = 'localhost', 
                 redis_port: int = 6379, max_memory_cache: int = 1000):
        """
        Initialize cache manager
        
        Args:
            cache_type: 'memory' or 'redis'
            redis_host: Redis server host
            redis_port: Redis server port
            max_memory_cache: Maximum items in memory cache
        """
        self.cache_type = cache_type
        self.max_memory_cache = max_memory_cache
        
        if cache_type == 'memory':
            self.memory_cache = {}
            self.cache_access_times = {}
        elif cache_type == 'redis':
            try:
                self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=False)
                self.redis_client.ping()
                logger.info("Connected to Redis cache")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}. Falling back to memory cache.")
                self.cache_type = 'memory'
                self.memory_cache = {}
                self.cache_access_times = {}
    
    def _generate_cache_key(self, image_data: bytes, model_name: str) -> str:
        """Generate cache key from image data and model name"""
        image_hash = hashlib.md5(image_data).hexdigest()
        return f"{model_name}:{image_hash}"
    
    def get_prediction(self, image_data: bytes, model_name: str) -> Optional[Dict]:
        """Get cached prediction"""
        cache_key = self._generate_cache_key(image_data, model_name)
        
        try:
            if self.cache_type == 'memory':
                if cache_key in self.memory_cache:
                    self.cache_access_times[cache_key] = time.time()
                    return self.memory_cache[cache_key]
            
            elif self.cache_type == 'redis':
                cached_result = self.redis_client.get(cache_key)
                if cached_result:
                    return pickle.loads(cached_result)
            
        except Exception as e:
            logger.error(f"Cache retrieval error: {e}")
        
        return None
    
    def cache_prediction(self, image_data: bytes, model_name: str, 
                        prediction: Dict, ttl: int = 3600):
        """Cache prediction result"""
        cache_key = self._generate_cache_key(image_data, model_name)
        
        try:
            if self.cache_type == 'memory':
                # Implement LRU eviction if cache is full
                if len(self.memory_cache) >= self.max_memory_cache:
                    self._evict_lru_item()
                
                self.memory_cache[cache_key] = prediction
                self.cache_access_times[cache_key] = time.time()
            
            elif self.cache_type == 'redis':
                serialized_prediction = pickle.dumps(prediction)
                self.redis_client.setex(cache_key, ttl, serialized_prediction)
            
        except Exception as e:
            logger.error(f"Cache storage error: {e}")
    
    def _evict_lru_item(self):
        """Evict least recently used item from memory cache"""
        if not self.cache_access_times:
            return
        
        lru_key = min(self.cache_access_times.keys(), 
                     key=lambda k: self.cache_access_times[k])
        
        del self.memory_cache[lru_key]
        del self.cache_access_times[lru_key]
    
    def clear_cache(self):
        """Clear all cached items"""
        if self.cache_type == 'memory':
            self.memory_cache.clear()
            self.cache_access_times.clear()
        elif self.cache_type == 'redis':
            self.redis_client.flushall()
        
        logger.info("Cache cleared")

class PerformanceMonitor:
    """Monitor system performance and model inference times"""
    
    def __init__(self, log_file: str = 'performance.log'):
        self.log_file = log_file
        self.metrics = defaultdict(list)
        self.start_time = time.time()
        
        # Start monitoring thread
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_system)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def _monitor_system(self):
        """Monitor system resources"""
        while self.monitoring:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                
                # Memory usage
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                
                # GPU usage (if available)
                gpu_percent = 0
                try:
                    import GPUtil
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu_percent = gpus[0].load * 100
                except ImportError:
                    pass
                
                # Log metrics
                timestamp = datetime.now().isoformat()
                self.metrics['cpu'].append((timestamp, cpu_percent))
                self.metrics['memory'].append((timestamp, memory_percent))
                self.metrics['gpu'].append((timestamp, gpu_percent))
                
                # Keep only last 1000 measurements
                for key in self.metrics:
                    if len(self.metrics[key]) > 1000:
                        self.metrics[key] = self.metrics[key][-1000:]
                
                time.sleep(5)  # Monitor every 5 seconds
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(10)
    
    def log_inference_time(self, model_name: str, inference_time: float, 
                          image_size: tuple, batch_size: int = 1):
        """Log model inference time"""
        timestamp = datetime.now().isoformat()
        
        log_entry = {
            'timestamp': timestamp,
            'model_name': model_name,
            'inference_time': inference_time,
            'image_size': image_size,
            'batch_size': batch_size,
            'throughput': batch_size / inference_time
        }
        
        # Add to metrics
        self.metrics['inference_times'].append(log_entry)
        
        # Write to log file
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        stats = {}
        
        # CPU stats
        if self.metrics['cpu']:
            cpu_values = [x[1] for x in self.metrics['cpu'][-100:]]  # Last 100 measurements
            stats['cpu'] = {
                'avg': np.mean(cpu_values),
                'max': np.max(cpu_values),
                'min': np.min(cpu_values)
            }
        
        # Memory stats
        if self.metrics['memory']:
            memory_values = [x[1] for x in self.metrics['memory'][-100:]]
            stats['memory'] = {
                'avg': np.mean(memory_values),
                'max': np.max(memory_values),
                'min': np.min(memory_values)
            }
        
        # Inference time stats
        if self.metrics['inference_times']:
            recent_inferences = self.metrics['inference_times'][-100:]
            inference_times = [x['inference_time'] for x in recent_inferences]
            throughputs = [x['throughput'] for x in recent_inferences]
            
            stats['inference'] = {
                'avg_time': np.mean(inference_times),
                'max_time': np.max(inference_times),
                'min_time': np.min(inference_times),
                'avg_throughput': np.mean(throughputs)
            }
        
        stats['uptime'] = time.time() - self.start_time
        
        return stats
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring = False
        if self.monitor_thread.is_alive():
            self.monitor_thread.join()

def cached_prediction(cache_manager: CacheManager, model_name: str, ttl: int = 3600):
    """Decorator for caching predictions"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract image data for cache key
            image_data = None
            if args and hasattr(args[0], 'tobytes'):
                image_data = args[0].tobytes()
            elif 'image' in kwargs and hasattr(kwargs['image'], 'tobytes'):
                image_data = kwargs['image'].tobytes()
            
            if image_data:
                # Try to get from cache
                cached_result = cache_manager.get_prediction(image_data, model_name)
                if cached_result:
                    logger.debug(f"Cache hit for {model_name}")
                    return cached_result
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Cache result
            if image_data and result:
                cache_manager.cache_prediction(image_data, model_name, result, ttl)
                logger.debug(f"Cached result for {model_name}")
            
            return result
        return wrapper
    return decorator

def timed_inference(performance_monitor: PerformanceMonitor, model_name: str):
    """Decorator for timing model inference"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            
            inference_time = end_time - start_time
            
            # Extract image size if available
            image_size = (224, 224)  # Default
            if args and hasattr(args[0], 'shape'):
                image_size = args[0].shape[:2]
            
            performance_monitor.log_inference_time(
                model_name, inference_time, image_size
            )
            
            return result
        return wrapper
    return decorator

class ProductionOptimizer:
    """Main class for production optimization"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize production optimizer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Initialize components
        self.model_optimizer = ModelOptimizer()
        self.cache_manager = CacheManager(
            cache_type=config.get('cache_type', 'memory'),
            redis_host=config.get('redis_host', 'localhost'),
            redis_port=config.get('redis_port', 6379)
        )
        self.performance_monitor = PerformanceMonitor(
            log_file=config.get('performance_log', 'performance.log')
        )
        
        logger.info("Production optimizer initialized")
    
    def optimize_models(self, model_paths: Dict[str, str]):
        """Optimize all models for production"""
        optimized_paths = {}
        
        for model_name, model_path in model_paths.items():
            if os.path.exists(model_path):
                # Quantize model
                quantized_path = f"optimized_{model_name}.tflite"
                optimized_path = self.model_optimizer.quantize_model(
                    model_path, quantized_path, 
                    self.config.get('quantization_type', 'dynamic')
                )
                optimized_paths[model_name] = optimized_path
            else:
                logger.warning(f"Model not found: {model_path}")
        
        return optimized_paths
    
    def get_system_status(self) -> Dict:
        """Get comprehensive system status"""
        return {
            'performance_stats': self.performance_monitor.get_performance_stats(),
            'cache_type': self.cache_manager.cache_type,
            'optimization_config': self.config,
            'timestamp': datetime.now().isoformat()
        }
    
    def cleanup(self):
        """Cleanup resources"""
        self.performance_monitor.stop_monitoring()
        self.cache_manager.clear_cache()
        logger.info("Production optimizer cleaned up")

# Example usage and configuration
def create_production_config() -> Dict[str, Any]:
    """Create production configuration"""
    return {
        'cache_type': 'memory',  # or 'redis'
        'redis_host': 'localhost',
        'redis_port': 6379,
        'quantization_type': 'dynamic',  # 'dynamic', 'int8', 'float16'
        'performance_log': 'performance.log',
        'max_memory_cache': 1000
    }

def optimize_for_production(model_paths: Dict[str, str]) -> ProductionOptimizer:
    """Setup production optimization"""
    config = create_production_config()
    optimizer = ProductionOptimizer(config)
    
    # Optimize models
    optimized_paths = optimizer.optimize_models(model_paths)
    logger.info(f"Optimized models: {optimized_paths}")
    
    return optimizer
