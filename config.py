import os

# Kubernetes configuration
KUBERNETES_NAMESPACE = os.getenv("K8S_NAMESPACE", "default")
KUBERNETES_CONFIG_PATH = os.getenv("KUBECONFIG", None)

# Application settings
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))

# Locust settings
LOCUST_PORT_BASE = int(os.getenv("LOCUST_PORT_BASE", 8089))
LOCUST_TIMEOUT = int(os.getenv("LOCUST_TIMEOUT", 30))

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
