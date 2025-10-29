# import subprocess
# import asyncio
# import json
# import requests
# import logging
# import os
# import signal
# from typing import Dict, Any, Optional
# from datetime import datetime, timedelta
# import tempfile
# import threading
# from concurrent.futures import ThreadPoolExecutor
# from urllib.parse import urlparse, urlunparse

# from .models import TrafficPattern, TrafficPatternType

# logger = logging.getLogger(__name__)

# class LocustTrafficGenerator:
#     def __init__(self):
#         self.running_processes = {}
#         self.locust_web_port_base = 8089
#         self.locust_stats = {}
#         self.executor = ThreadPoolExecutor(max_workers=10)

#     def check_status(self) -> bool:
#         """Check if Locust is available"""
#         try:
#             result = subprocess.run(
#                 ['locust', '--version'],
#                 capture_output=True,
#                 text=True,
#                 timeout=10
#             )
#             return result.returncode == 0
#         except Exception:
#             return False

#     def _normalize_target_url(self, target_url: str) -> str:
#         """Ensure target URL is properly formatted without any path"""
#         try:
#             # Add http:// if no scheme provided
#             if not target_url.startswith(('http://', 'https://')):
#                 target_url = f"http://{target_url}"

#             parsed = urlparse(target_url)

#             # Keep only scheme + netloc (host[:port]), strip any path/query/fragment
#             normalized = urlunparse((
#                 parsed.scheme,
#                 parsed.netloc,
#                 '', '', '', ''
#             ))

#             # Remove trailing slash if present
#             normalized = normalized.rstrip('/')

#             logger.info(f"Normalized URL from '{target_url}' to '{normalized}'")
#             return normalized
#         except Exception as e:
#             logger.warning(f"Failed to normalize URL '{target_url}': {e}")
#             return target_url.rstrip('/')

#     async def _validate_target_url(self, target_url: str) -> bool:
#         """Validate that target URL is accessible"""
#         try:
#             # Prefer HEAD, then fallback to GET
#             response = requests.head(target_url, timeout=5, allow_redirects=True)
#             if response.status_code < 500:
#                 logger.info(f"Target URL validation successful: {response.status_code} for {target_url}")
#                 return True

#             response = requests.get(target_url, timeout=10, allow_redirects=True)
#             logger.info(f"Target URL validation: {response.status_code} for {target_url}")
#             return response.status_code < 500
#         except Exception as e:
#             logger.warning(f"Target URL validation failed for {target_url}: {e}")
#             return False

#     async def start_traffic(self, task_id: str, pattern: TrafficPattern) -> Dict[str, Any]:
#         """Start traffic generation based on pattern"""
#         try:
#             logger.info(f"Starting traffic generation for task: {task_id}")
#             logger.info(f"Pattern: {pattern.pattern_type}, Target: {pattern.target_url}")

#             # Normalize the target URL to prevent path issues like /get/get
#             normalized_url = self._normalize_target_url(pattern.target_url)
#             pattern.target_url = normalized_url
#             logger.info(f"Using base host (no path): {pattern.target_url}")

#             # Optional: validate reachability (non-blocking for success)
#             await self._validate_target_url(normalized_url)

#             # Generate locustfile for this pattern
#             locustfile_path = await self._generate_locustfile(task_id, pattern)
#             logger.info(f"Generated locustfile: {locustfile_path}")

#             # Start Locust process
#             web_port = self._get_available_port()
#             logger.info(f"Using port: {web_port}")

#             process = await self._start_locust_process_sync(
#                 task_id, locustfile_path, pattern, web_port
#             )

#             self.running_processes[task_id] = {
#                 'process': process,
#                 'pattern': pattern,
#                 'locustfile_path': locustfile_path,
#                 'web_port': web_port,
#                 'start_time': datetime.utcnow()
#             }

#             # Give Locust a moment to boot
#             logger.info("Waiting for Locust to initialize...")
#             await asyncio.sleep(5)

#             # Check if process is still running
#             if process.poll() is not None:
#                 stdout, stderr = process.communicate()
#                 logger.error(f"Locust process failed to start. Return code: {process.returncode}")
#                 logger.error(f"Stdout: {stdout}")
#                 logger.error(f"Stderr: {stderr}")
#                 raise Exception(f"Locust process failed with return code: {process.returncode}")

#             # Verify Locust web interface is responding
#             await self._wait_for_locust_ready(web_port)

#             # Start the test via API
#             logger.info("Starting test via API...")
#             await self._start_test_via_api(task_id, pattern, web_port)

#             # Schedule safety auto-stop after duration_minutes
#             asyncio.create_task(self._auto_stop_after(task_id, web_port, getattr(pattern, 'duration_minutes', 10)))

#             return {
#                 'task_id': task_id,
#                 'status': 'started',
#                 'web_port': web_port,
#                 'locustfile': locustfile_path,
#                 'target_url': normalized_url
#             }

#         except Exception as e:
#             logger.error(f"Failed to start traffic for {task_id}: {str(e)}", exc_info=True)

#             # Cleanup on failure
#             if task_id in self.running_processes:
#                 try:
#                     process_info = self.running_processes[task_id]
#                     if 'process' in process_info and process_info['process'].poll() is None:
#                         process_info['process'].terminate()
#                     if 'locustfile_path' in process_info and os.path.exists(process_info['locustfile_path']):
#                         os.unlink(process_info['locustfile_path'])
#                     del self.running_processes[task_id]
#                 except Exception as cleanup_error:
#                     logger.warning(f"Error during cleanup: {cleanup_error}")

#             raise

#     async def _auto_stop_after(self, task_id: str, web_port: int, duration_minutes: Optional[int]):
#         """Safety auto-stop to ensure test ends after duration"""
#         try:
#             seconds = max(1, int(duration_minutes or 10) * 60)
#         except Exception:
#             seconds = 600
#         await asyncio.sleep(seconds)
#         # If still running, stop and collect final stats
#         if task_id in self.running_processes:
#             try:
#                 await self.stop_traffic(task_id)
#                 logger.info(f"Auto-stopped task {task_id} after {seconds} seconds")
#             except Exception as e:
#                 logger.warning(f"Auto-stop failed for {task_id}: {e}")

#     async def _wait_for_locust_ready(self, web_port: int, max_attempts: int = 10):
#         """Wait for Locust web interface to be ready"""
#         for attempt in range(max_attempts):
#             try:
#                 response = requests.get(f"http://localhost:{web_port}/", timeout=5)
#                 if response.status_code == 200:
#                     logger.info("Locust web interface is ready")
#                     return
#             except Exception as e:
#                 logger.debug(f"Locust not ready yet, attempt {attempt + 1}: {e}")

#             await asyncio.sleep(2)

#         raise Exception("Locust web interface failed to become ready")

#     async def stop_traffic(self, task_id: str) -> Dict[str, Any]:
#         """Stop traffic generation"""
#         if task_id not in self.running_processes:
#             raise ValueError(f"Task {task_id} not found")

#         try:
#             process_info = self.running_processes[task_id]
#             process = process_info['process']
#             web_port = process_info['web_port']

#             # Stop test via API first
#             try:
#                 await self._stop_test_via_api(web_port)
#                 await asyncio.sleep(2)
#             except Exception as e:
#                 logger.warning(f"Failed to stop via API: {e}")

#             # Terminate process
#             def terminate_process():
#                 try:
#                     if process.poll() is None:  # Process is still running
#                         process.terminate()
#                         process.wait(timeout=10)
#                 except subprocess.TimeoutExpired:
#                     process.kill()
#                     process.wait()
#                 except Exception as e:
#                     logger.warning(f"Error terminating process: {e}")

#             # Run termination in thread pool
#             loop = asyncio.get_event_loop()
#             await loop.run_in_executor(self.executor, terminate_process)

#             # Clean up
#             locustfile_path = process_info['locustfile_path']
#             if os.path.exists(locustfile_path):
#                 os.unlink(locustfile_path)

#             # Get final stats
#             final_stats = await self._get_final_stats(web_port)

#             del self.running_processes[task_id]

#             return {
#                 'task_id': task_id,
#                 'status': 'stopped',
#                 'final_stats': final_stats
#             }

#         except Exception as e:
#             logger.error(f"Failed to stop traffic for {task_id}: {str(e)}")
#             raise

#     def get_current_stats(self) -> Dict[str, Any]:
#         """Get current statistics for all running tasks"""
#         stats = {}
#         for task_id, process_info in self.running_processes.items():
#             try:
#                 web_port = process_info['web_port']
#                 response = requests.get(
#                     f"http://localhost:{web_port}/stats/requests",
#                     timeout=5
#                 )
#                 if response.status_code == 200:
#                     stats[task_id] = response.json()
#             except Exception as e:
#                 logger.warning(f"Failed to get stats for {task_id}: {e}")
#                 stats[task_id] = {'error': str(e)}

#         return stats

#     async def _generate_locustfile(self, task_id: str, pattern: TrafficPattern) -> str:
#         """Generate Locust file based on traffic pattern with retries/timeouts"""
#         if pattern.pattern_type == TrafficPatternType.BURST:
#             locust_content = self._generate_burst_locustfile(pattern)
#         elif pattern.pattern_type == TrafficPatternType.STEADY:
#             locust_content = self._generate_steady_locustfile(pattern)
#         elif pattern.pattern_type == TrafficPatternType.WAVE:
#             locust_content = self._generate_wave_locustfile(pattern)
#         elif pattern.pattern_type == TrafficPatternType.STEP:
#             locust_content = self._generate_step_locustfile(pattern)
#         else:
#             raise ValueError(f"Unsupported pattern type: {pattern.pattern_type}")

#         # Write to temporary file
#         temp_file = tempfile.NamedTemporaryFile(
#             mode='w', suffix=f'_{task_id}.py', delete=False
#         )
#         temp_file.write(locust_content)
#         temp_file.close()

#         return temp_file.name

#     async def _start_locust_process_sync(self, task_id: str, locustfile_path: str,
#                                          pattern: TrafficPattern, web_port: int):
#         """Start Locust process using synchronous subprocess"""
#         cmd = [
#             'locust',
#             '-f', locustfile_path,
#             '--web-port', str(web_port),
#             '--host', pattern.target_url,  # base host only (no path)
#             '--html', f'locust_report_{task_id}.html',
#             '--loglevel', 'INFO'
#         ]

#         logger.info(f"Starting Locust with command: {' '.join(cmd)}")

#         def start_process():
#             return subprocess.Popen(
#                 cmd,
#                 stdout=subprocess.PIPE,
#                 stderr=subprocess.PIPE,
#                 text=True,
#                 creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
#             )

#         # Run in thread pool to avoid blocking
#         loop = asyncio.get_event_loop()
#         process = await loop.run_in_executor(self.executor, start_process)

#         return process

#     def _generate_burst_locustfile(self, pattern: TrafficPattern) -> str:
#         """Burst traffic; clean base host + relative paths; retries + timeouts"""
#         base_host = pattern.target_url
#         connect_timeout = getattr(pattern, 'connection_timeout', 10)
#         read_timeout = getattr(pattern, 'request_timeout', 30)

#         return f'''
# from locust import HttpUser, task, between
# import logging
# from requests.adapters import HTTPAdapter
# from urllib3.util.retry import Retry

# logger = logging.getLogger(__name__)

# class BurstUser(HttpUser):
#     wait_time = between(0.1, 0.5)
#     host = "{base_host}"

#     def on_start(self):
#         retry = Retry(
#             total=3,
#             backoff_factor=0.5,
#             status_forcelist=[429, 500, 502, 503, 504],
#             allowed_methods=["HEAD", "GET", "OPTIONS"]
#         )
#         adapter = HTTPAdapter(max_retries=retry)
#         self.client.mount("http://", adapter)
#         self.client.mount("https://", adapter)
#         logger.info(f"BurstUser started with host={{self.host}}")

#     @task(6)
#     def root(self):
#         try:
#             with self.client.get("/", timeout=({connect_timeout}, {read_timeout}), catch_response=True, name="GET /") as r:
#                 if r.status_code >= 500:
#                     r.failure(f"Server error {{r.status_code}}")
#         except Exception as e:
#             logger.error(f"GET / failed: {{e}}")

#     @task(2)
#     def health(self):
#         try:
#             with self.client.get("/health", timeout=({connect_timeout}, {read_timeout}), catch_response=True, name="GET /health") as r:
#                 if r.status_code == 404:
#                     r.success()
#                 elif r.status_code >= 500:
#                     r.failure(f"Server error {{r.status_code}}")
#         except Exception as e:
#             logger.debug(f"GET /health failed: {{e}}")

#     @task(2)
#     def status(self):
#         try:
#             with self.client.get("/status", timeout=({connect_timeout}, {read_timeout}), catch_response=True, name="GET /status") as r:
#                 if r.status_code == 404:
#                     r.success()
#                 elif r.status_code >= 500:
#                     r.failure(f"Server error {{r.status_code}}")
#         except Exception as e:
#             logger.debug(f"GET /status failed: {{e}}")
# '''

#     def _generate_steady_locustfile(self, pattern: TrafficPattern) -> str:
#         """Steady traffic; clean base host + relative paths; retries + timeouts"""
#         logger.info("hello how")
#         base_host = pattern.target_url
#         connect_timeout = getattr(pattern, 'connection_timeout', 10)
#         read_timeout = getattr(pattern, 'request_timeout', 30)

#         return f'''
# from locust import HttpUser, task, between
# import logging
# from requests.adapters import HTTPAdapter
# from urllib3.util.retry import Retry

# logger = logging.getLogger(__name__)

# class SteadyUser(HttpUser):
#     wait_time = between(1, 3)
#     host = "{base_host}"

#     def on_start(self):
#         retry = Retry(
#             total=3,
#             backoff_factor=0.5,
#             status_forcelist=[429, 500, 502, 503, 504],
#             allowed_methods=["HEAD", "GET", "OPTIONS"]
#         )
#         adapter = HTTPAdapter(max_retries=retry)
#         self.client.mount("http://", adapter)
#         self.client.mount("https://", adapter)
#         logger.info(f"SteadyUser started with host={{self.host}}")
#         try:
#             r = self.client.get("/", timeout=({connect_timeout}, {read_timeout}), name="startup")
#             logger.info(f"Startup probe: {{r.status_code}}")
#         except Exception as e:
#             logger.warning(f"Startup probe failed: {{e}}")

#     @task(10)
#     def root(self):
#         try:
#             with self.client.get("/", timeout=({connect_timeout}, {read_timeout}), catch_response=True, name="GET /") as r:
#                 if r.status_code >= 500:
#                     r.failure(f"Server error {{r.status_code}}")
#         except Exception as e:
#             logger.error(f"GET / failed: {{e}}")

#     @task(3)
#     def health(self):
#         try:
#             with self.client.get("/health", timeout=({connect_timeout}, {read_timeout}), catch_response=True, name="GET /health") as r:
#                 if r.status_code == 404:
#                     r.success()
#                 elif r.status_code >= 500:
#                     r.failure(f"Server error {{r.status_code}}")
#         except Exception as e:
#             logger.debug(f"GET /health failed: {{e}}")

#     @task(1)
#     def status(self):
#         try:
#             with self.client.get("/status", timeout=({connect_timeout}, {read_timeout}), catch_response=True, name="GET /status") as r:
#                 if r.status_code == 404:
#                     r.success()
#                 elif r.status_code >= 500:
#                     r.failure(f"Server error {{r.status_code}}")
#         except Exception as e:
#             logger.debug(f"GET /status failed: {{e}}")
# '''

#     def _generate_wave_locustfile(self, pattern: TrafficPattern) -> str:
#         """Wave traffic; clean base host + relative paths; retries + timeouts"""
#         base_host = pattern.target_url
#         connect_timeout = getattr(pattern, 'connection_timeout', 10)
#         read_timeout = getattr(pattern, 'request_timeout', 30)
#         cycle_minutes = getattr(pattern, 'cycle_duration_minutes', 5) or 5

#         return f'''
# from locust import HttpUser, task, between
# import math
# import time
# import logging
# from requests.adapters import HTTPAdapter
# from urllib3.util.retry import Retry

# logger = logging.getLogger(__name__)

# class WaveUser(HttpUser):
#     wait_time = between(0.5, 2)
#     host = "{base_host}"

#     def on_start(self):
#         self.start_time = time.time()
#         retry = Retry(
#             total=3,
#             backoff_factor=0.5,
#             status_forcelist=[429, 500, 502, 503, 504],
#             allowed_methods=["HEAD", "GET", "OPTIONS"]
#         )
#         adapter = HTTPAdapter(max_retries=retry)
#         self.client.mount("http://", adapter)
#         self.client.mount("https://", adapter)
#         logger.info(f"WaveUser started with host={{self.host}}")

#     @task(5)
#     def adaptive(self):
#         try:
#             elapsed = time.time() - self.start_time
#             cycle_pos = (elapsed % ({cycle_minutes} * 60)) / ({cycle_minutes} * 60)
#             intensity = 0.5 + 0.5 * math.sin(2 * math.pi * cycle_pos)

#             with self.client.get("/", timeout=({connect_timeout}, {read_timeout}), catch_response=True, name="GET /") as r:
#                 if r.status_code >= 500:
#                     r.failure(f"Server error {{r.status_code}}")

#             if intensity > 0.7:
#                 with self.client.get("/health", timeout=({connect_timeout}, {read_timeout}), catch_response=True, name="GET /health") as r:
#                     if r.status_code == 404:
#                         r.success()
#         except Exception as e:
#             logger.error(f"Wave request failed: {{e}}")

#     @task(2)
#     def baseline(self):
#         try:
#             with self.client.get("/", timeout=({connect_timeout}, {read_timeout}), catch_response=True, name="GET / (baseline)") as r:
#                 if r.status_code >= 500:
#                     r.failure(f"Server error {{r.status_code}}")
#         except Exception as e:
#             logger.error(f"Baseline failed: {{e}}")
# '''

#     def _generate_step_locustfile(self, pattern: TrafficPattern) -> str:
#         """Step traffic; clean base host + relative paths; retries + timeouts"""
#         base_host = pattern.target_url
#         connect_timeout = getattr(pattern, 'connection_timeout', 10)
#         read_timeout = getattr(pattern, 'request_timeout', 30)

#         return f'''
# from locust import HttpUser, task, between
# import time
# import logging
# from requests.adapters import HTTPAdapter
# from urllib3.util.retry import Retry

# logger = logging.getLogger(__name__)

# class StepUser(HttpUser):
#     wait_time = between(1, 2)
#     host = "{base_host}"

#     def on_start(self):
#         self.start_time = time.time()
#         retry = Retry(
#             total=3,
#             backoff_factor=0.5,
#             status_forcelist=[429, 500, 502, 503, 504],
#             allowed_methods=["HEAD", "GET", "OPTIONS"]
#         )
#         adapter = HTTPAdapter(max_retries=retry)
#         self.client.mount("http://", adapter)
#         self.client.mount("https://", adapter)
#         logger.info(f"StepUser started with host={{self.host}}")

#     @task(6)
#     def main(self):
#         try:
#             with self.client.get("/", timeout=({connect_timeout}, {read_timeout}), catch_response=True, name="GET /") as r:
#                 if r.status_code >= 500:
#                     r.failure(f"Server error {{r.status_code}}")
#         except Exception as e:
#             logger.error(f"Main step failed: {{e}}")

#     @task(2)
#     def secondary(self):
#         try:
#             with self.client.get("/health", timeout=({connect_timeout}, {read_timeout}), catch_response=True, name="GET /health") as r:
#                 if r.status_code == 404:
#                     r.success()
#         except Exception as e:
#             logger.debug(f"Secondary step failed: {{e}}")

#     @task(1)
#     def intensive(self):
#         try:
#             elapsed = time.time() - self.start_time
#             step_level = int(elapsed // 30) % 4

#             with self.client.get("/", timeout=({connect_timeout}, {read_timeout}), catch_response=True, name="GET / (intensive)") as r:
#                 if r.status_code >= 500:
#                     r.failure(f"Server error {{r.status_code}}")

#             if step_level >= 2:
#                 with self.client.get("/status", timeout=({connect_timeout}, {read_timeout}), catch_response=True, name="GET /status") as r:
#                     if r.status_code == 404:
#                         r.success()
#         except Exception as e:
#             logger.error(f"Intensive step failed: {{e}}")
# '''

#     async def _start_test_via_api(self, task_id: str, pattern: TrafficPattern, web_port: int):
#         """Start test via Locust web API"""
#         url = f"http://localhost:{web_port}/swarm"
#         logger.info(pattern)
#         # Handle missing attributes safely
#         user_count = getattr(pattern, 'initial_users', None) or \
#                      getattr(pattern, 'users', None) or \
#                      getattr(pattern, 'peak_users', None) or \
#                      getattr(pattern, 'max_users', None) or 10

#         # If ramp_time_seconds is provided, compute spawn_rate dynamically
#         ramp_time_seconds = getattr(pattern, 'ramp_time_seconds', None)

#         if ramp_time_seconds:
#             spawn_rate = user_count / ramp_time_seconds
#         else:
#             spawn_rate = getattr(pattern, 'spawn_rate',0.0)

#         duration_minutes = getattr(pattern, 'duration_minutes', 10)
#         run_time = f"{int(duration_minutes)}m"

#         data = {
#             'user_count': user_count,
#             'spawn_rate': spawn_rate,
#             'run_time': run_time
#         }

#         logger.info(f"Starting test with {user_count} users, spawn_rate: {spawn_rate:.2f}, run_time: {run_time}")


#         max_retries = 5
#         for attempt in range(max_retries):
#             try:
#                 await asyncio.sleep(3)  # Give Locust more time to start
#                 response = requests.post(url, data=data, timeout=15)
#                 if response.status_code == 200:
#                     logger.info(f"Successfully swarm started test for {task_id}")
#                     return
#                 else:
#                     logger.warning(f"API start failed, attempt {attempt + 1}: {response.status_code} - {response.text}")
#             except Exception as e:
#                 logger.warning(f"API start attempt {attempt + 1} failed: {e}")

#             await asyncio.sleep(3)

#         raise Exception(f"Failed to start test via API after {max_retries} attempts")

#     async def _stop_test_via_api(self, web_port: int):
#         """Stop test via Locust web API"""
#         url = f"http://localhost:{web_port}/stop"
#         response = requests.get(url, timeout=10)
#         response.raise_for_status()
#         logger.info("Successfully stopped test via API")

#     async def _get_final_stats(self, web_port: int) -> Dict[str, Any]:
#         """Get final statistics from Locust"""
#         try:
#             response = requests.get(
#                 f"http://localhost:{web_port}/stats/requests",
#                 timeout=10
#             )
#             if response.status_code == 200:
#                 return response.json()
#         except Exception as e:
#             logger.warning(f"Failed to get final stats: {e}")

#         return {}

#     def _get_available_port(self) -> int:
#         """Get available port for Locust web interface"""
#         import socket

#         for port in range(self.locust_web_port_base, self.locust_web_port_base + 100):
#             if port not in [info['web_port'] for info in self.running_processes.values()]:
#                 sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#                 try:
#                     sock.bind(('localhost', port))
#                     sock.close()
#                     return port
#                 except OSError:
#                     continue

#         raise Exception("No available ports for Locust web interface")

import subprocess
import asyncio
import json
import requests
import logging
import os
import tempfile
from typing import Dict, Any, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse, urlunparse


# Your existing models
from .models import TrafficPattern, TrafficPatternType  # pattern/status as you defined

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class LocustTrafficGenerator:
    def __init__(self, use_fast_http: bool = False):
        self.running_processes: Dict[str, Dict[str, Any]] = {}
        self.locust_web_port_base = 8089
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.use_fast_http = use_fast_http

    def check_status(self) -> bool:
        try:
            result = subprocess.run(
                ['locust', '--version'],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except Exception:
            return False

    def _normalize_target_url(self, target_url: str) -> str:
        try:
            if not target_url.startswith(('http://', 'https://')):
                target_url = f"http://{target_url}"
            parsed = urlparse(target_url)
            normalized = urlunparse((parsed.scheme, parsed.netloc, '', '', '', ''))
            normalized = normalized.rstrip('/')
            logger.info(f"Normalized URL from '{target_url}' to '{normalized}'")
            return normalized
        except Exception as e:
            logger.warning(f"Failed to normalize URL '{target_url}': {e}")
            return target_url.rstrip('/')

    async def _validate_target_url(self, target_url: str) -> bool:
        try:
            response = requests.head(target_url, timeout=5, allow_redirects=True)
            if response.status_code < 500:
                logger.info(f"Target URL validation successful: {response.status_code} for {target_url}")
                return True
            response = requests.get(target_url, timeout=10, allow_redirects=True)
            logger.info(f"Target URL validation: {response.status_code} for {target_url}")
            return response.status_code < 500
        except Exception as e:
            logger.warning(f"Target URL validation failed for {target_url}: {e}")
            return False

    async def start_traffic(self, task_id: str, pattern: TrafficPattern) -> Dict[str, Any]:
        try:
            logger.info(f"Starting traffic generation for task: {task_id}")
            logger.info(f"Pattern: {pattern.pattern_type}, Target: {pattern.target_url}")

            normalized_url = self._normalize_target_url(pattern.target_url)
            pattern.target_url = normalized_url
            await self._validate_target_url(normalized_url)

            locustfile_path = await self._generate_locustfile(task_id, pattern)
            web_port = self._get_available_port()
            process = await self._start_locust_process_sync(task_id, locustfile_path, pattern, web_port)

            self.running_processes[task_id] = {
                'process': process,
                'pattern': pattern,
                'locustfile_path': locustfile_path,
                'web_port': web_port,
                'start_time': datetime.utcnow()
            }

            await self._wait_for_locust_ready(web_port, max_attempts=20)
            asyncio.create_task(self._apply_dynamic_user_schedule(task_id, web_port, pattern))
            asyncio.create_task(self._auto_stop_after(task_id, web_port, getattr(pattern, 'duration_minutes', 10)))

            return {
                'task_id': task_id,
                'status': 'started',
                'web_port': web_port,
                'locustfile': locustfile_path,
                'target_url': normalized_url
            }

        except Exception as e:
            logger.error(f"Failed to start traffic for {task_id}: {str(e)}", exc_info=True)
            process_info = self.running_processes.get(task_id)
            if process_info:
                try:
                    proc = process_info.get('process')
                    if proc and proc.poll() is None:
                        proc.terminate()
                    path = process_info.get('locustfile_path')
                    if path and os.path.exists(path):
                        os.unlink(path)
                    self.running_processes.pop(task_id, None)
                except Exception as cleanup_error:
                    logger.warning(f"Error during cleanup: {cleanup_error}")
            raise

    async def _apply_dynamic_user_schedule(self, task_id: str, web_port: int, pattern: TrafficPattern):
        pt = getattr(pattern, "pattern_type", "").lower()

        if pt == "burst":
            user_count = int(getattr(pattern, "peak_users", 0) or getattr(pattern, "users", 10) or 10)
            spawn_rate = float(getattr(pattern, "spawn_rate", 1.0) or 1.0)
            duration = int(getattr(pattern, "duration_minutes", 5) or 5)
            try:
                requests.post(
                    f"http://localhost:{web_port}/swarm",
                    data={"user_count": user_count, "spawn_rate": spawn_rate, "run_time": f"{duration}m"},
                    timeout=10,
                )
                logger.info(f"[{task_id}] Burst: started {user_count} users for {duration}m")
            except Exception as e:
                logger.warning(f"[{task_id}] Burst schedule error: {e}")

        elif pt == "steady":
            user_count = int(getattr(pattern, "users", 10) or 10)
            spawn_rate = float(getattr(pattern, "spawn_rate", 1.0) or 1.0)
            duration = int(getattr(pattern, "duration_minutes", 10) or 10)
            try:
                requests.post(
                    f"http://localhost:{web_port}/swarm",
                    data={"user_count": user_count, "spawn_rate": spawn_rate, "run_time": f"{duration}m"},
                    timeout=10,
                )
                logger.info(f"[{task_id}] Steady: started {user_count} users for {duration}m")
            except Exception as e:
                logger.warning(f"[{task_id}] Steady schedule error: {e}")

        elif pt == "step":
            initial = int(getattr(pattern, "initial_users", 10) or 10)
            target = int(getattr(pattern, "users", 40) or initial)
            step = int(getattr(pattern, "step_size", 10) or 10)
            step_min = int(getattr(pattern, "step_duration_minutes", 2) or 2)
            spawn_rate = float(getattr(pattern, "spawn_rate", 1.0) or 1.0)
            duration = int(getattr(pattern, "duration_minutes", 12) or 12)
            current = initial
            hold_time = duration * 60
            start_time = datetime.utcnow()
            while current <= target and (datetime.utcnow() - start_time).total_seconds() < hold_time:
                try:
                    requests.post(
                        f"http://localhost:{web_port}/swarm",
                        data={"user_count": current, "spawn_rate": spawn_rate, "run_time": f"{step_min}m"},
                        timeout=10,
                    )
                    logger.info(f"[{task_id}] Step: updated to {current} users")
                except Exception as e:
                    logger.warning(f"[{task_id}] Step schedule error: {e}")
                    break
                current += step
                await asyncio.sleep(step_min * 60)

        elif pt == "wave":
            min_users = int(getattr(pattern, "min_users", 10) or 10)
            max_users = int(getattr(pattern, "max_users", 60) or 60)
            cycle = int(getattr(pattern, "cycle_duration_minutes", 5) or 5)
            duration = int(getattr(pattern, "duration_minutes", 10) or 10)
            spawn_rate = float(getattr(pattern, "spawn_rate", 1.0) or 1.0)
            import math

            points = 12
            cycles = max(1, duration // cycle)
            interval_sec = (cycle * 60) // points
            start_time = datetime.utcnow()

            for n in range(cycles * points):
                phase = 2 * math.pi * (n % points) / points
                avg = (min_users + max_users) / 2
                amp = (max_users - min_users) / 2
                user_count = int(round(avg + amp * math.sin(phase)))
                try:
                    requests.post(
                        f"http://localhost:{web_port}/swarm",
                        data={"user_count": user_count, "spawn_rate": spawn_rate, "run_time": f"{interval_sec // 60}m"},
                        timeout=10,
                    )
                    logger.info(f"[{task_id}] Wave: set users to {user_count}")
                except Exception as e:
                    logger.warning(f"[{task_id}] Wave schedule error: {e}")
                await asyncio.sleep(interval_sec)
                if (datetime.utcnow() - start_time).total_seconds() > duration * 60:
                    break

    async def _auto_stop_after(self, task_id: str, web_port: int, duration_minutes: Optional[int]):
        try:
            seconds = max(1, int(duration_minutes or 10) * 60)
        except Exception:
            seconds = 600
        await asyncio.sleep(seconds)
        if task_id in self.running_processes:
            try:
                await self.stop_traffic(task_id)
                logger.info(f"Auto-stopped task {task_id} after {seconds} seconds")
            except Exception as e:
                logger.warning(f"Auto-stop failed for {task_id}: {e}")

    async def _wait_for_locust_ready(self, web_port: int, max_attempts: int = 10):
        for attempt in range(max_attempts):
            try:
                response = requests.get(f"http://localhost:{web_port}/", timeout=5)
                if response.status_code == 200:
                    logger.info("Locust web interface is ready")
                    return
            except Exception as e:
                logger.debug(f"Locust not ready yet, attempt {attempt + 1}: {e}")
            await asyncio.sleep(1.5)
        raise Exception("Locust web interface failed to become ready")

    async def stop_traffic(self, task_id: str) -> Dict[str, Any]:
        if task_id not in self.running_processes:
            raise ValueError(f"Task {task_id} not found")
        try:
            process_info = self.running_processes[task_id]
            process = process_info['process']
            web_port = process_info['web_port']

            try:
                await self._stop_test_via_api(web_port)
                await asyncio.sleep(1.5)
            except Exception as e:
                logger.warning(f"Failed to stop via API: {e}")

            def terminate_process():
                try:
                    if process.poll() is None:
                        process.terminate()
                        process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                except Exception as e:
                    logger.warning(f"Error terminating process: {e}")

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self.executor, terminate_process)

            locustfile_path = process_info['locustfile_path']
            if os.path.exists(locustfile_path):
                os.unlink(locustfile_path)

            final_stats = await self._get_final_stats(web_port)

            del self.running_processes[task_id]
            return {'task_id': task_id, 'status': 'stopped', 'final_stats': final_stats}

        except Exception as e:
            logger.error(f"Failed to stop traffic for {task_id}: {str(e)}")
            raise

    def get_current_stats(self) -> Dict[str, Any]:
        stats = {}
        for task_id, process_info in self.running_processes.items():
            web_port = process_info['web_port']
            url = f"http://localhost:{web_port}/stats/requests"
            last_err = None

            for retry in range(3):
                try:
                    response = requests.get(url, timeout=15)
                    if response.status_code == 200:
                        stats[task_id] = response.json()
                        break
                    last_err = f"HTTP {response.status_code}"
                except Exception as e:
                    last_err = str(e)
                    if retry < 2:
                        import time
                        time.sleep(0.5)
            else:
                logger.warning(f"Failed to get stats for {task_id}: {last_err}")
                stats[task_id] = {'error': last_err}

        return stats

    async def _generate_locustfile(self, task_id: str, pattern: TrafficPattern) -> str:
        # Generate only user classes, NO LoadTestShape
        if pattern.pattern_type == TrafficPatternType.BURST:
            content = self._generate_burst_locustfile(pattern)
        elif pattern.pattern_type == TrafficPatternType.STEADY:
            content = self._generate_steady_locustfile(pattern)
        elif pattern.pattern_type == TrafficPatternType.WAVE:
            content = self._generate_wave_locustfile(pattern)
        elif pattern.pattern_type == TrafficPatternType.STEP:
            content = self._generate_step_locustfile(pattern)
        else:
            raise ValueError(f"Unsupported pattern type: {pattern.pattern_type}")

        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix=f'_{task_id}.py', delete=False)
        temp_file.write(content)
        temp_file.close()
        return temp_file.name

    async def _start_locust_process_sync(self, task_id: str, locustfile_path: str, pattern: TrafficPattern, web_port: int):
        cmd = [
            'locust',
            '-f', locustfile_path,
            '--web-port', str(web_port),
            '--host', pattern.target_url,
            '--html', f'locust_report_{task_id}.html',
            '--loglevel', 'INFO'
        ]
        logger.info(f"Starting Locust with command: {' '.join(cmd)}")

        def start_process():
            return subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            )

        loop = asyncio.get_event_loop()
        process = await loop.run_in_executor(self.executor, start_process)
        return process

    def _locust_user_header(self) -> str:
        return "from locust import FastHttpUser as HttpUser, task, between" if self.use_fast_http \
            else "from locust import HttpUser, task, between"

    def _common_http_setup_block(self, connect_timeout: int, read_timeout: int) -> str:
        return f"""
    def on_start(self):
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        retry = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.client.mount("http://", adapter)
        self.client.mount("https://", adapter)
        try:
            r = self.client.get("/", timeout=({connect_timeout}, {read_timeout}), name="startup")
            logger.info(f"Startup probe: {{r.status_code}}")
        except Exception as e:
            logger.warning(f"Startup probe failed: {{e}}")
"""

    # User-class-only locustfile generators (EXAMPLES, adapt tasks as needed)
    def _generate_burst_locustfile(self, pattern: TrafficPattern) -> str:
        base_host = pattern.target_url
        connect_timeout = getattr(pattern, 'connection_timeout', 10)
        read_timeout = getattr(pattern, 'request_timeout', 30)
        return f"""
{self._locust_user_header()}
import logging
logger = logging.getLogger(__name__)

class BurstUser(HttpUser):
    wait_time = between(0.05, 0.2)
    host = "{base_host}"
{self._common_http_setup_block(connect_timeout, read_timeout)}
    @task(6)
    def root(self):
        try:
            with self.client.get("/", timeout=({connect_timeout}, {read_timeout}), catch_response=True, name="GET /") as r:
                if r.status_code >= 500:
                    r.failure(f"Server error {{r.status_code}}")
        except Exception as e:
            logger.error(f"GET / failed: {{e}}")

    @task(2)
    def health(self):
        try:
            with self.client.get("/health", timeout=({connect_timeout}, {read_timeout}), catch_response=True, name="GET /health") as r:
                if r.status_code == 404:
                    r.success()
                elif r.status_code >= 500:
                    r.failure(f"Server error {{r.status_code}}")
        except Exception as e:
            logger.debug(f"GET /health failed: {{e}}")

    @task(2)
    def status(self):
        try:
            with self.client.get("/status", timeout=({connect_timeout}, {read_timeout}), catch_response=True, name="GET /status") as r:
                if r.status_code == 404:
                    r.success()
                elif r.status_code >= 500:
                    r.failure(f"Server error {{r.status_code}}")
        except Exception as e:
            logger.debug(f"GET /status failed: {{e}}")
"""

    def _generate_steady_locustfile(self, pattern: TrafficPattern) -> str:
        base_host = pattern.target_url
        connect_timeout = getattr(pattern, 'connection_timeout', 10)
        read_timeout = getattr(pattern, 'request_timeout', 30)
        return f"""
{self._locust_user_header()}
import logging
logger = logging.getLogger(__name__)

class SteadyUser(HttpUser):
    wait_time = between(1, 3)
    host = "{base_host}"
{self._common_http_setup_block(connect_timeout, read_timeout)}
    @task(10)
    def root(self):
        try:
            with self.client.get("/", timeout=({connect_timeout}, {read_timeout}), catch_response=True, name="GET /") as r:
                if r.status_code >= 500:
                    r.failure(f"Server error {{r.status_code}}")
        except Exception as e:
            logger.error(f"GET / failed: {{e}}")

    @task(3)
    def health(self):
        try:
            with self.client.get("/health", timeout=({connect_timeout}, {read_timeout}), catch_response=True, name="GET /health") as r:
                if r.status_code == 404:
                    r.success()
                elif r.status_code >= 500:
                    r.failure(f"Server error {{r.status_code}}")
        except Exception as e:
            logger.debug(f"GET /health failed: {{e}}")

    @task(1)
    def status(self):
        try:
            with self.client.get("/status", timeout=({connect_timeout}, {read_timeout}), catch_response=True, name="GET /status") as r:
                if r.status_code == 404:
                    r.success()
                elif r.status_code >= 500:
                    r.failure(f"Server error {{r.status_code}}")
        except Exception as e:
            logger.debug(f"GET /status failed: {{e}}")
"""

    def _generate_wave_locustfile(self, pattern: TrafficPattern) -> str:
        base_host = pattern.target_url
        connect_timeout = getattr(pattern, 'connection_timeout', 10)
        read_timeout = getattr(pattern, 'request_timeout', 30)
        cycle_minutes = getattr(pattern, 'cycle_duration_minutes', 5) or 5
        return f"""
{self._locust_user_header()}
import math, time, logging
logger = logging.getLogger(__name__)

class WaveUser(HttpUser):
    wait_time = between(0.5, 2)
    host = "{base_host}"
    def on_start(self):
        self.start_time = time.time()
{self._common_http_setup_block(connect_timeout, read_timeout)}
    @task(5)
    def adaptive(self):
        try:
            elapsed = time.time() - self.start_time
            cycle_pos = (elapsed % ({cycle_minutes} * 60)) / ({cycle_minutes} * 60)
            intensity = 0.5 + 0.5 * math.sin(2 * math.pi * cycle_pos)

            with self.client.get("/", timeout=({connect_timeout}, {read_timeout}), catch_response=True, name="GET /") as r:
                if r.status_code >= 500:
                    r.failure(f"Server error {{r.status_code}}")

            if intensity > 0.7:
                with self.client.get("/health", timeout=({connect_timeout}, {read_timeout}), catch_response=True, name="GET /health") as r:
                    if r.status_code == 404:
                        r.success()
        except Exception as e:
            logger.error(f"Wave request failed: {{e}}")

    @task(2)
    def baseline(self):
        try:
            with self.client.get("/", timeout=({connect_timeout}, {read_timeout}), catch_response=True, name="GET / (baseline)") as r:
                if r.status_code >= 500:
                    r.failure(f"Server error {{r.status_code}}")
        except Exception as e:
            logger.error(f"Baseline failed: {{e}}")
"""

    def _generate_step_locustfile(self, pattern: TrafficPattern) -> str:
        base_host = pattern.target_url
        connect_timeout = getattr(pattern, 'connection_timeout', 10)
        read_timeout = getattr(pattern, 'request_timeout', 30)
        return f"""
{self._locust_user_header()}
import time, logging
logger = logging.getLogger(__name__)

class StepUser(HttpUser):
    wait_time = between(1, 2)
    host = "{base_host}"
    def on_start(self):
        self.start_time = time.time()
{self._common_http_setup_block(connect_timeout, read_timeout)}
    @task(6)
    def main(self):
        try:
            with self.client.get("/", timeout=({connect_timeout}, {read_timeout}), catch_response=True, name="GET /") as r:
                if r.status_code >= 500:
                    r.failure(f"Server error {{r.status_code}}")
        except Exception as e:
            logger.error(f"Main step failed: {{e}}")

    @task(2)
    def secondary(self):
        try:
            with self.client.get("/health", timeout=({connect_timeout}, {read_timeout}), catch_response=True, name="GET /health") as r:
                if r.status_code == 404:
                    r.success()
        except Exception as e:
            logger.debug(f"Secondary step failed: {{e}}")

    @task(1)
    def intensive(self):
        try:
            elapsed = time.time() - self.start_time
            step_level = int(elapsed // 30) % 4
            with self.client.get("/", timeout=({connect_timeout}, {read_timeout}), catch_response=True, name="GET / (intensive)") as r:
                if r.status_code >= 500:
                    r.failure(f"Server error {{r.status_code}}")
            if step_level >= 2:
                with self.client.get("/status", timeout=({connect_timeout}, {read_timeout}), catch_response=True, name="GET /status") as r:
                    if r.status_code == 404:
                        r.success()
        except Exception as e:
            logger.error(f"Intensive step failed: {{e}}")
"""
    
   

    async def _start_test_via_api(self, task_id: str, pattern: TrafficPattern, web_port: int):
        """Start test via Locust web API with corrected user_count selection"""
        url = f"http://localhost:{web_port}/swarm"

        # Get pattern_type safely
        pt = getattr(pattern, 'pattern_type', None)
        
        # Normalize to string for comparison
        if hasattr(pt, 'value'):
            pt_str = str(pt.value).lower()
        elif hasattr(pt, 'name'):
            pt_str = str(pt.name).lower()
        else:
            pt_str = str(pt).lower() if pt else ""

        # Pattern-specific user_count selection
        user_count = None
        if pt_str == "burst":
            user_count = getattr(pattern, 'peak_users', None)
        elif pt_str == "wave":
            user_count = getattr(pattern, 'max_users', None)
        elif pt_str == "step":
            user_count = getattr(pattern, 'initial_users', None)

        # Generic fallback to 'users' field
        if user_count is None:
            user_count = getattr(pattern, 'users', None)

        # Final safety fallback
        user_count = int(user_count) if user_count is not None else 10

        # Compute spawn_rate: prefer ramp_time_seconds when present
        ramp_time_seconds = getattr(pattern, 'ramp_time_seconds', None)
        if ramp_time_seconds and ramp_time_seconds > 0:
            spawn_rate = max(0.1, float(user_count) / float(ramp_time_seconds))
        else:
            spawn_rate = float(getattr(pattern, 'spawn_rate', 1.0) or 1.0)

        # Duration -> run_time string
        duration_minutes = int(getattr(pattern, 'duration_minutes', 10) or 10)
        run_time = f"{duration_minutes}m"

        data = {
            'user_count': int(user_count),
            'spawn_rate': float(spawn_rate),
            'run_time': run_time
        }
        logger.info(f"Starting test with {user_count} users, spawn_rate: {spawn_rate:.2f}, run_time: {run_time}")

        # Retry with longer timeout
        max_retries = 6
        for attempt in range(max_retries):
            try:
                await asyncio.sleep(2)
                response = requests.post(url, data=data, timeout=15)
                if response.status_code == 200:
                    logger.info(f"Successfully started test for {task_id}")
                    return
                else:
                    logger.warning(f"API start failed, attempt {attempt + 1}: {response.status_code} - {response.text}")
            except Exception as e:
                logger.warning(f"API start attempt {attempt + 1} failed: {e}")
            await asyncio.sleep(2)

        raise Exception(f"Failed to start test via API after {max_retries} attempts")



    async def _stop_test_via_api(self, web_port: int):
        url = f"http://localhost:{web_port}/stop"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        logger.info("Successfully stopped test via API")

    async def _get_final_stats(self, web_port: int) -> Dict[str, Any]:
        try:
            response = requests.get(f"http://localhost:{web_port}/stats/requests", timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.warning(f"Failed to get final stats: {e}")
        return {}

    def _get_available_port(self) -> int:
        import socket
        in_use = [info['web_port'] for info in self.running_processes.values()]
        for port in range(self.locust_web_port_base, self.locust_web_port_base + 200):
            if port in in_use:
                continue
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                sock.bind(('localhost', port))
                sock.close()
                return port
            except OSError:
                continue
        raise Exception("No available ports for Locust web interface")
