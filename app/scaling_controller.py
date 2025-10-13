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
                
                # Update scaling context
                state_dict['scaling']['last_action_delta'] = self.last_action_delta
                state_dict['scaling']['steps_since_action'] = self.steps_since_action
                
                # Get PPO action
                new_replicas, action_delta = self.ppo_agent.predict_action(state_dict)
                
                # Apply action if different from current
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
                
                # Wait for next control interval
                await asyncio.sleep(self.control_interval)
                
            except Exception as e:
                logger.error(f"Error in control loop: {e}")
                await asyncio.sleep(self.control_interval)
    
    def stop(self):
        """Stop the control loop"""
        self.is_running = False
        logger.info("PPO control loop stopped")
