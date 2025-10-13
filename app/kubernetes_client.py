import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import yaml
import os

try:
    from kubernetes import client, config
    from kubernetes.client.rest import ApiException
    KUBERNETES_AVAILABLE = True
except ImportError:
    KUBERNETES_AVAILABLE = False

from .models import DeploymentStatus, PodMetrics

logger = logging.getLogger(__name__)

class KubernetesManager:
    def __init__(self):
        self.v1 = None
        self.apps_v1 = None
        self.metrics_v1beta1 = None
        self.initialized = False
        
    async def initialize(self):
        """Initialize Kubernetes client"""
        if not KUBERNETES_AVAILABLE:
            logger.warning("Kubernetes library not available")
            return False
            
        try:
            # Try in-cluster config first, then local config
            try:
                config.load_incluster_config()
                logger.info("Using in-cluster Kubernetes config")
            except:
                config.load_kube_config()
                logger.info("Using local Kubernetes config")
            
            self.v1 = client.CoreV1Api()
            self.apps_v1 = client.AppsV1Api()
            
            # Test connection
            await self.check_connection()
            self.initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Kubernetes client: {e}")
            return False
    
    async def get_hpa_desired(self, name: str, namespace: str = "default") -> int:
        autoscaling_v2 = client.AutoscalingV2Api()
        hpa = autoscaling_v2.read_namespaced_horizontal_pod_autoscaler(name=name, namespace=namespace)
        # desiredReplicas is under status in HPA resources
        return int(getattr(hpa.status, "desired_replicas", 0))

    async def get_node_count(self) -> int:
        nodes = self.v1.list_node()
        return len(nodes.items or [])
    
    async def check_connection(self) -> bool:
        """Check Kubernetes connection"""
        if not self.initialized:
            return False
            
        try:
            # Simple API call to test connection
            self.v1.list_namespace(limit=1)
            return True
        except Exception as e:
            logger.error(f"Kubernetes connection check failed: {e}")
            return False
    
    async def deploy_php_apache(self) -> Dict[str, Any]:
        """Deploy php-apache sample application"""
        if not self.initialized:
            raise Exception("Kubernetes client not initialized")
        
        try:
            # Create deployment
            deployment = self._create_php_apache_deployment()
            self.apps_v1.create_namespaced_deployment(
                namespace="default",
                body=deployment
            )
            
            # Create service
            service = self._create_php_apache_service()
            self.v1.create_namespaced_service(
                namespace="default",
                body=service
            )
            
            # Create HPA
            hpa = self._create_php_apache_hpa()
            autoscaling_v1 = client.AutoscalingV1Api()
            autoscaling_v1.create_namespaced_horizontal_pod_autoscaler(
                namespace="default",
                body=hpa
            )
            
            return {
                "deployment": "php-apache",
                "service": "php-apache",
                "hpa": "php-apache-hpa",
                "namespace": "default",
                "status": "created"
            }
            
        except ApiException as e:
            if e.status == 409:  # Already exists
                logger.info("php-apache already exists, updating...")
                return await self._update_php_apache()
            else:
                raise Exception(f"Kubernetes API error: {e}")
        except Exception as e:
            logger.error(f"Failed to deploy php-apache: {e}")
            raise
    
    async def get_deployment_status(self) -> DeploymentStatus:
        """Get deployment status"""
        if not self.initialized:
            raise Exception("Kubernetes client not initialized")
        
        try:
            deployment = self.apps_v1.read_namespaced_deployment(
                name="php-apache",
                namespace="default"
            )
            
            return DeploymentStatus(
                name=deployment.metadata.name,
                namespace=deployment.metadata.namespace,
                ready_replicas=deployment.status.ready_replicas or 0,
                desired_replicas=deployment.spec.replicas,
                available_replicas=deployment.status.available_replicas or 0,
                status=self._get_deployment_status_string(deployment.status),
                created_at=deployment.metadata.creation_timestamp,
                labels=deployment.metadata.labels or {}
            )
            
        except ApiException as e:
            if e.status == 404:
                raise Exception("php-apache deployment not found")
            else:
                raise Exception(f"Failed to get deployment status: {e}")
    
    async def get_pod_metrics(self) -> List[PodMetrics]:
        """Get pod metrics"""
        if not self.initialized:
            raise Exception("Kubernetes client not initialized")
        
        try:
            # Get pods
            pods = self.v1.list_namespaced_pod(
                namespace="default",
                label_selector="app=php-apache"
            )
            
            metrics = []
            for pod in pods.items:
                pod_metrics = PodMetrics(
                    pod_name=pod.metadata.name,
                    namespace=pod.metadata.namespace,
                    status=pod.status.phase,
                    cpu_requests=self._get_resource_value(pod, "requests", "cpu"),
                    memory_requests=self._get_resource_value(pod, "requests", "memory"),
                    cpu_limits=self._get_resource_value(pod, "limits", "cpu"),
                    memory_limits=self._get_resource_value(pod, "limits", "memory")
                )
                
                # Try to get actual usage metrics if metrics server is available
                try:
                    if self.metrics_v1beta1 is None:
                        self.metrics_v1beta1 = client.CustomObjectsApi()
                    
                    pod_metrics_data = self.metrics_v1beta1.get_namespaced_custom_object(
                        group="metrics.k8s.io",
                        version="v1beta1",
                        namespace="default",
                        plural="pods",
                        name=pod.metadata.name
                    )
                    
                    if 'containers' in pod_metrics_data and pod_metrics_data['containers']:
                        container = pod_metrics_data['containers'][0]
                        pod_metrics.cpu_usage = container['usage'].get('cpu', 'N/A')
                        pod_metrics.memory_usage = container['usage'].get('memory', 'N/A')
                        
                except Exception as e:
                    logger.debug(f"Metrics not available for pod {pod.metadata.name}: {e}")
                
                metrics.append(pod_metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get pod metrics: {e}")
            raise
    
    def _create_php_apache_deployment(self):
        """Create php-apache deployment specification"""
        return client.V1Deployment(
            api_version="apps/v1",
            kind="Deployment",
            metadata=client.V1ObjectMeta(
                name="php-apache",
                labels={"app": "php-apache"}
            ),
            spec=client.V1DeploymentSpec(
                replicas=1,
                selector=client.V1LabelSelector(
                    match_labels={"app": "php-apache"}
                ),
                template=client.V1PodTemplateSpec(
                    metadata=client.V1ObjectMeta(
                        labels={"app": "php-apache"}
                    ),
                    spec=client.V1PodSpec(
                        containers=[
                            client.V1Container(
                                name="php-apache",
                                image="k8s.gcr.io/hpa-example",
                                ports=[client.V1ContainerPort(container_port=80)],
                                resources=client.V1ResourceRequirements(
                                    requests={"cpu": "200m", "memory": "128Mi"},
                                    limits={"cpu": "500m", "memory": "256Mi"}
                                )
                            )
                        ]
                    )
                )
            )
        )
    
    def _create_php_apache_service(self):
        """Create php-apache service specification"""
        return client.V1Service(
            api_version="v1",
            kind="Service",
            metadata=client.V1ObjectMeta(
                name="php-apache",
                labels={"app": "php-apache"}
            ),
            spec=client.V1ServiceSpec(
                selector={"app": "php-apache"},
                ports=[client.V1ServicePort(
                    port=80,
                    target_port=80,
                    protocol="TCP"
                )],
                type="ClusterIP"
            )
        )
    
    def _create_php_apache_hpa(self):
        """Create HPA for php-apache"""
        return client.V1HorizontalPodAutoscaler(
            api_version="autoscaling/v1",
            kind="HorizontalPodAutoscaler",
            metadata=client.V1ObjectMeta(
                name="php-apache-hpa"
            ),
            spec=client.V1HorizontalPodAutoscalerSpec(
                scale_target_ref=client.V1CrossVersionObjectReference(
                    api_version="apps/v1",
                    kind="Deployment",
                    name="php-apache"
                ),
                min_replicas=1,
                max_replicas=10,
                target_cpu_utilization_percentage=50
            )
        )
    
    async def _update_php_apache(self):
        """Update existing php-apache deployment"""
        deployment = self._create_php_apache_deployment()
        self.apps_v1.patch_namespaced_deployment(
            name="php-apache",
            namespace="default",
            body=deployment
        )
        
        return {
            "deployment": "php-apache",
            "namespace": "default",
            "status": "updated"
        }
    
    def _get_deployment_status_string(self, status) -> str:
        """Get human-readable deployment status"""
        if status.ready_replicas == status.replicas:
            return "Ready"
        elif status.available_replicas and status.available_replicas > 0:
            return "Partially Ready"
        else:
            return "Not Ready"
    
    def _get_resource_value(self, pod, resource_type: str, resource_name: str) -> Optional[str]:
        """Extract resource value from pod spec"""
        try:
            for container in pod.spec.containers:
                if container.resources:
                    resources = getattr(container.resources, resource_type, {})
                    if resources and resource_name in resources:
                        return resources[resource_name]
        except Exception:
            pass
        return None
    async def get_hpa_desired(self, name: str, namespace: str = "default") -> int:
        """Get HPA desired replicas"""
        try:
            autoscaling_v2 = client.AutoscalingV2Api()
            hpa = autoscaling_v2.read_namespaced_horizontal_pod_autoscaler(
                name=name, 
                namespace=namespace
            )
            return int(getattr(hpa.status, "desired_replicas", 1))
        except Exception as e:
            logger.warning(f"Could not get HPA desired replicas: {e}")
            return 1

    async def get_node_count(self) -> int:
        """Get current node count"""
        try:
            nodes = self.v1.list_node()
            return len(nodes.items or [])
        except Exception as e:
            logger.warning(f"Could not get node count: {e}")
            return 1
