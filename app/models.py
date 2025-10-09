from pydantic import BaseModel, validator
from typing import Optional, Dict, Any, List
from enum import Enum
from datetime import datetime

class TrafficPatternType(str, Enum):
    BURST = "burst"
    STEADY = "steady"
    WAVE = "wave"
    STEP = "step"

class TrafficPattern(BaseModel):
    pattern_type: TrafficPatternType
    # target_url: str = "http://php-apache.default.svc.cluster.local"
    target_url: str = "http://localhost:8080"
    duration_minutes: int = 10
    
    # Pattern-specific parameters
    users: Optional[int] = 10
    spawn_rate: Optional[float] = 1.0
    
    # Burst pattern
    peak_users: Optional[int] = None
    ramp_time_seconds: Optional[int] = None
    
    # Wave pattern
    min_users: Optional[int] = None
    max_users: Optional[int] = None
    cycle_duration_minutes: Optional[int] = 5  # Add default value
    
    # Step pattern
    initial_users: Optional[int] = None
    step_size: Optional[int] = None
    step_duration_minutes: Optional[int] = None
    
    # HTTP settings
    request_timeout: int = 30
    connection_timeout: int = 10


class TrafficStatus(BaseModel):
    task_id: str
    status: str  # starting, running, completed, failed, stopped
    pattern: TrafficPattern
    start_time: datetime
    end_time: Optional[datetime] = None
    current_users: Optional[int] = None
    total_requests: Optional[int] = None
    failed_requests: Optional[int] = None
    average_response_time: Optional[float] = None
    error: Optional[str] = None

class DeploymentStatus(BaseModel):
    name: str
    namespace: str = "default"
    ready_replicas: int
    desired_replicas: int
    available_replicas: int
    status: str
    created_at: datetime
    labels: Dict[str, str]
    
class PodMetrics(BaseModel):
    pod_name: str
    namespace: str
    cpu_usage: Optional[str] = None
    memory_usage: Optional[str] = None
    cpu_requests: Optional[str] = None
    memory_requests: Optional[str] = None
    cpu_limits: Optional[str] = None
    memory_limits: Optional[str] = None
    status: str
