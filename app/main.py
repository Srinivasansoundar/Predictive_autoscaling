import asyncio
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import asyncio
import logging
from datetime import datetime, timedelta
import uvicorn
import os

from .models import TrafficPattern, TrafficStatus, DeploymentStatus
from .traffic_generator import LocustTrafficGenerator
from .kubernetes_client import KubernetesManager
from .state_adapter import build_state_vector
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Workload Management - Step 1: Traffic Generator",
    description="FastAPI service to deploy sample app and generate traffic patterns",
    version="1.0.0"
)

# Global instances
k8s_manager = KubernetesManager()
traffic_generator = LocustTrafficGenerator()

# Background task storage
background_tasks_status = {}

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting Workload Management Step 1 service...")
    await k8s_manager.initialize()
    logger.info("Service initialized successfully")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "Workload Management Step 1",
        "status": "running",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    k8s_status = await k8s_manager.check_connection()
    locust_status = traffic_generator.check_status()
    
    return {
        "status": "healthy" if k8s_status and locust_status else "unhealthy",
        "kubernetes": "connected" if k8s_status else "disconnected",
        "locust": "available" if locust_status else "unavailable",
        "timestamp": datetime.utcnow().isoformat()
    }

# helper to pick an Aggregated row (Locust returns per-endpoint and an Aggregated line)
def _extract_aggregated(locust_stats_for_task: dict) -> dict:
    # Locust /stats/requests returns a dict with "stats" list and often an "Aggregated" row
    # Fallback if only per-endpoint rows exist: reduce into a single aggregate
    stats_list = (locust_stats_for_task or {}).get("stats", [])
    agg = next((s for s in stats_list if s.get("name") in ("Aggregated", "")), None)
    if not agg and stats_list:
        # crude aggregation fallback
        total_req = sum(s.get("num_requests", 0) for s in stats_list)
        total_fail = sum(s.get("num_failures", 0) for s in stats_list)
        rps = sum(float(s.get("current_rps", 0.0)) for s in stats_list)
        # pick a robust latency proxy if p95 not present in API; controller can refine later
        p90 = max(float(s.get("ninetieth_response_time", 0.0)) for s in stats_list)
        agg = {
            "name": "Aggregated",
            "num_requests": total_req,
            "num_failures": total_fail,
            "current_rps": rps,
            "ninetieth_response_time": p90,
        }
    return agg or {}

@app.get("/state/vector")
async def get_state_vector():
    # 1) Workload: pull latest Locust stats (choose the latest task if multiple)
    locust_all = traffic_generator.get_current_stats()  # {task_id: {...}}
    agg = {}
    if locust_all:
        # choose the last key by start time if you track it, else arbitrary pick
        any_task_id = next(iter(locust_all.keys()))
        agg = _extract_aggregated(locust_all.get(any_task_id, {}))

    # 2) Infra: deployment/pod metrics
    deploy_status = await k8s_manager.get_deployment_status()  # includes ready_replicas, labels
    pods = await k8s_manager.get_pod_metrics()  # list of PodMetrics dicts

    # 3) HPA desired + node count (see KubernetesManager additions below)
    try:
        hpa_desired = await k8s_manager.get_hpa_desired(name="php-apache", namespace="default")
    except Exception:
        hpa_desired = deploy_status.desired_replicas if hasattr(deploy_status, "desired_replicas") else 0
    try:
        node_count = await k8s_manager.get_node_count()
    except Exception:
        node_count = 0

    # 4) Scaling context (store these in memory when actions are executed; default to zeros)
    last_action_delta = 0
    steps_since_action = 0

    # 5) Optional cost hints (stub or wire later)
    cost_per_min_usd = 0.0
    spot_ratio = 0.0

    # 6) Build the state vector
    state = build_state_vector(
        locust_agg=agg,
        pod_metrics=[pm.dict() if hasattr(pm, "dict") else pm for pm in pods],
        deploy_status=deploy_status.dict() if hasattr(deploy_status, "dict") else deploy_status,
        hpa_desired=hpa_desired,
        node_count=node_count,
        last_action_delta=last_action_delta,
        steps_since_action=steps_since_action,
        cost_per_min_usd=cost_per_min_usd,
        spot_ratio=spot_ratio
    )
    return state
@app.post("/deploy/php-apache")
async def deploy_php_apache():
    """Deploy the php-apache sample application"""
    try:
        deployment_result = await k8s_manager.deploy_php_apache()
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "message": "php-apache application deployed successfully",
                "deployment": deployment_result,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    except Exception as e:
        logger.error(f"Failed to deploy php-apache: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Deployment failed: {str(e)}")

@app.get("/deployment/status")
async def get_deployment_status():
    """Get current deployment status"""
    try:
        status = await k8s_manager.get_deployment_status()
        return status
    except Exception as e:
        logger.error(f"Failed to get deployment status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")

@app.post("/traffic/start")
async def start_traffic_generation(
    background_tasks: BackgroundTasks,
    pattern: TrafficPattern
):
    """Start traffic generation with specified pattern"""
    try:
        task_id = f"traffic_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        # Start traffic generation in background
        background_tasks.add_task(
            run_traffic_generation, 
            task_id, 
            pattern
        )
        
        background_tasks_status[task_id] = {
            "status": "starting",
            "pattern": pattern.dict(),
            "start_time": datetime.utcnow().isoformat(),
            "task_id": task_id
        }
        
        return {
            "status": "success",
            "message": "Traffic generation started",
            "task_id": task_id,
            "pattern": pattern.dict()
        }
    except Exception as e:
        logger.error(f"Failed to start traffic generation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Traffic generation failed: {str(e)}")

@app.get("/traffic/status/{task_id}")
async def get_traffic_status(task_id: str):
    """Get traffic generation status"""
    if task_id not in background_tasks_status:
        raise HTTPException(status_code=404, detail="Task not found")
    
    status = background_tasks_status[task_id]
    locust_stats = traffic_generator.get_current_stats()
    
    return {
        "task_status": status,
        "locust_stats": locust_stats,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/traffic/stop/{task_id}")
async def stop_traffic_generation(task_id: str):
    """Stop traffic generation"""
    try:
        if task_id not in background_tasks_status:
            raise HTTPException(status_code=404, detail="Task not found")
        
        result = await traffic_generator.stop_traffic(task_id)
        background_tasks_status[task_id]["status"] = "stopped"
        background_tasks_status[task_id]["end_time"] = datetime.utcnow().isoformat()
        
        return {
            "status": "success",
            "message": "Traffic generation stopped",
            "task_id": task_id,
            "result": result
        }
    except Exception as e:
        logger.error(f"Failed to stop traffic generation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Stop failed: {str(e)}")

@app.get("/traffic/patterns")
async def get_available_patterns():
    """Get available traffic patterns"""
    return {
        "patterns": [
            {
                "name": "burst",
                "description": "Sudden traffic spike",
                "parameters": ["peak_users", "duration", "ramp_time"]
            },
            {
                "name": "steady",
                "description": "Consistent load",
                "parameters": ["users", "duration"]
            },
            {
                "name": "wave",
                "description": "Oscillating load pattern",
                "parameters": ["min_users", "max_users", "cycle_duration", "total_duration"]
            },
            {
                "name": "step",
                "description": "Gradual step increases",
                "parameters": ["initial_users", "step_size", "step_duration", "max_users"]
            }
        ]
    }

@app.get("/metrics/current")
async def get_current_metrics():
    """Get current system metrics"""
    try:
        k8s_metrics = await k8s_manager.get_pod_metrics()
        traffic_metrics = traffic_generator.get_current_stats()
        
        return {
            "kubernetes_metrics": k8s_metrics,
            "traffic_metrics": traffic_metrics,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Metrics collection failed: {str(e)}")

async def run_traffic_generation(task_id: str, pattern: TrafficPattern):
    """Background task to run traffic generation"""
    try:
        background_tasks_status[task_id]["status"] = "running"
        
        result = await traffic_generator.start_traffic(task_id, pattern)
        
        background_tasks_status[task_id]["status"] = "completed"
        background_tasks_status[task_id]["result"] = result
        background_tasks_status[task_id]["end_time"] = datetime.utcnow().isoformat()
        
    except Exception as e:
        logger.error(f"Traffic generation task {task_id} failed: {str(e)}")
        background_tasks_status[task_id]["status"] = "failed"
        background_tasks_status[task_id]["error"] = str(e)
        background_tasks_status[task_id]["end_time"] = datetime.utcnow().isoformat()

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True
    )
