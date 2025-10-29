import asyncio
import logging
from typing import Dict, Any
from datetime import datetime
from kubernetes import client

logger = logging.getLogger(__name__)


class ScalingController:
    """Controller to apply PPO scaling actions to Kubernetes"""
    
    def __init__(self, k8s_manager, ppo_agent):
        self.k8s_manager = k8s_manager
        self.ppo_agent = ppo_agent
        self.last_action_delta = 0
        self.steps_since_action = 0
        self.control_interval = 15  # seconds
        self.is_running = False
        self.prev_metrics = None  # Store previous metrics here
        
    async def apply_scaling_action(self, new_replicas: int, namespace: str = "default", 
                                   deployment_name: str = "php-apache"):
        """Apply scaling action to Kubernetes Deployment"""
        try:
            apps_v1 = client.AppsV1Api()
            
            # Patch the deployment with new replica count
            body = {
                "spec": {
                    "replicas": new_replicas
                }
            }
            
            apps_v1.patch_namespaced_deployment_scale(
                name=deployment_name,
                namespace=namespace,
                body=body
            )
            
            logger.info(f"Successfully scaled {deployment_name} to {new_replicas} replicas")
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply scaling action: {e}")
            return False
    
    async def control_loop(self, state_provider_func):
        """Main PPO control loop"""
        logger.info(f"Starting PPO control loop (interval: {self.control_interval}s)")
        self.is_running = True
        
        while self.is_running:
            try:
                # Get current state
                state_dict = await state_provider_func()
                
                # Extract current metrics required for slopes
                current_metrics = {
                    'cpu_util_pct': state_dict['infra'].get('cpu_utilization_pct', 0),
                    'rps': state_dict['workload'].get('rps', 0),
                    'p95_latency_ms': state_dict['workload'].get('p95_latency_ms', 0)
                }
                
                # Calculate slopes if prev_metrics exist
                # Compute slopes if previous_metrics exists
                if hasattr(self, 'prev_metrics') and self.prev_metrics is not None:
                    cpu_slope = current_metrics['cpu_util_pct'] - self.prev_metrics['cpu_util_pct']
                    rps_slope = current_metrics['rps'] - self.prev_metrics['rps']
                    latency_slope = current_metrics['p95_latency_ms'] - self.prev_metrics['p95_latency_ms']
                else:
                    cpu_slope = rps_slope = latency_slope = 0.0
                
                # Save current metrics for next iteration
                self.prev_metrics = current_metrics
                
                # Add trend info to state
                state_dict['trend'] = {
                    'cpu_slope': cpu_slope,
                    'rps_slope': rps_slope,
                    'latency_slope': latency_slope
                }
                
                # Update scaling context
                state_dict['scaling']['last_action_delta'] = self.last_action_delta
                state_dict['scaling']['steps_since_action'] = self.steps_since_action
                
                # Predict action via PPO
                new_replicas, action_delta = self.ppo_agent.predict_action(state_dict)
                
                current_replicas = state_dict['infra']['pods_ready']
                if new_replicas != current_replicas:
                    success = await self.apply_scaling_action(new_replicas)
                    if success:
                        self.last_action_delta = action_delta
                        self.steps_since_action = 0
                    else:
                        self.steps_since_action += 1
                else:
                    self.steps_since_action += 1
                
                await asyncio.sleep(self.control_interval)
            
            except Exception as e:
                logger.error(f"Error in control loop: {e}")
                await asyncio.sleep(self.control_interval)


    def stop(self):
        """Stop the control loop"""
        self.is_running = False
        logger.info("PPO control loop stopped")