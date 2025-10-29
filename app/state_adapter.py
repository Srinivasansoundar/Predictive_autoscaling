# app/state_adapter.py
import math
from datetime import datetime, timezone

def parse_cpu_cores(val: str) -> float:
    if not val:
        return 0.0

    v = val.strip().lower()
    
    if v.endswith("m"):
        return float(v[:-1]) / 1000.0
    elif v.endswith("n"):
        return float(v[:-1]) / 1_000_000_000.0
    else:
        return float(v)

def parse_mem_mebibytes(val: str) -> float:
    if not val:
        return 0.0
    v = val.strip().lower()
    mult = 1.0
    if v.endswith("ki"): mult, v = 1/1024, v[:-2]
    elif v.endswith("mi"): mult, v = 1.0, v[:-2]
    elif v.endswith("gi"): mult, v = 1024.0, v[:-2]
    elif v.endswith("ti"): mult, v = 1024.0*1024.0, v[:-2]
    elif v.endswith("k"): mult, v = 1/1024, v[:-1]
    elif v.endswith("m"): mult, v = 1.0, v[:-1]
    elif v.endswith("g"): mult, v = 1024.0, v[:-1]
    return float(v) * mult


def build_state_vector(
    locust_agg: dict,
    pod_metrics: list,
    deploy_status: dict,
    hpa_desired: int,
    node_count: int,
    last_action_delta: int,
    steps_since_action: int,
    cost_per_min_usd: float = 0.0,
    spot_ratio: float = 0.0,
    previous_metrics=None,
) -> dict:
    # Workload
    num_req = max(1, int(locust_agg.get("num_requests", 0)))
    num_fail = int(locust_agg.get("num_failures", 0))
    error_rate_pct = 100.0 * (num_fail / num_req)
    rps = float(locust_agg.get("current_rps", locust_agg.get("total_rps", 0.0)))
    p95 = float(locust_agg.get("ninety_fifth_response_time",
             locust_agg.get("ninetieth_response_time", 0.0)))
    queue_length = float(locust_agg.get("queue_length", 0.0))

    # Infra
    total_cpu_usage = total_cpu_req = total_cpu_lim = 0.0
    total_mem_usage = total_mem_req = total_mem_lim = 0.0

    for pm in pod_metrics:
        cpu_u = parse_cpu_cores(pm.get("cpu_usage")) if pm.get("cpu_usage") not in [None, "N/A"] else 0.0
        mem_u = parse_mem_mebibytes(pm.get("memory_usage")) if pm.get("memory_usage") not in [None, "N/A"] else 0.0
        cpu_r = parse_cpu_cores(pm.get("cpu_requests"))
        mem_r = parse_mem_mebibytes(pm.get("memory_requests"))
        cpu_l = parse_cpu_cores(pm.get("cpu_limits"))
        mem_l = parse_mem_mebibytes(pm.get("memory_limits"))
        total_cpu_usage += cpu_u; total_mem_usage += mem_u
        total_cpu_req += cpu_r;   total_mem_req += mem_r
        total_cpu_lim += cpu_l;   total_mem_lim += mem_l

    cpu_den = total_cpu_req if total_cpu_req > 0 else (total_cpu_lim if total_cpu_lim > 0 else 1e-6)
    mem_den = total_mem_req if total_mem_req > 0 else (total_mem_lim if total_mem_lim > 0 else 1e-6)
    cpu_util_pct = 100.0 * (total_cpu_usage / cpu_den)
    mem_util_pct = 100.0 * (total_mem_usage / mem_den)

    # Representative per-pod request/limit
    if pod_metrics:
        pod0 = pod_metrics[0]
        pod_cpu_req = parse_cpu_cores(pod0.get("cpu_requests"))
        pod_cpu_lim = parse_cpu_cores(pod0.get("cpu_limits"))
    else:
        pod_cpu_req = pod_cpu_lim = 0.0

    # Time features
    now = datetime.now(timezone.utc)
    minute_of_day = now.hour * 60 + now.minute + now.second/60.0
    theta = (minute_of_day / (24*60)) * 2 * math.pi
    minute_sin, minute_cos = math.sin(theta), math.cos(theta)
    day_of_week = now.weekday()

    # Compute slopes if previous_metrics provided
    if previous_metrics:
        cpu_slope = cpu_util_pct - previous_metrics.get('cpu_util_pct', cpu_util_pct)
        rps_slope = rps - previous_metrics.get('rps', rps)
        latency_slope = p95 - previous_metrics.get('p95_latency_ms', p95)
    else:
        cpu_slope = 0.0
        rps_slope = 0.0
        latency_slope = 0.0

    return {
        "workload": {
            "rps": rps,
            "p95_latency_ms": p95,
            "error_rate_pct": error_rate_pct,
            "queue_length": queue_length
        },
        "infra": {
            "pods_ready": int(deploy_status.get("ready_replicas", 0)),
            "hpa_desired_replicas": int(hpa_desired),
            "cpu_utilization_pct": cpu_util_pct,
            "mem_utilization_pct": mem_util_pct,
            "node_count": int(node_count),
            "pod_cpu_request_cores": pod_cpu_req,
            "pod_cpu_limit_cores": pod_cpu_lim
        },
        "cost": {
            "cost_per_min_usd": float(cost_per_min_usd),
            "spot_ratio": float(spot_ratio)
        },
        "time": {
            "minute_of_day_sin": minute_sin,
            "minute_of_day_cos": minute_cos,
            "day_of_week": int(day_of_week)
        },
        "scaling": {
            "last_action_delta": int(last_action_delta),
            "steps_since_action": int(steps_since_action)
        },
        "trend": {
            "cpu_slope": cpu_slope,
            "rps_slope": rps_slope,
            "latency_slope": latency_slope
        }
    }

