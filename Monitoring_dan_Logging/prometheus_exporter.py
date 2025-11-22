import os
import time
import logging
import psutil
import requests
from flask import Flask, request, jsonify, Response
from prometheus_client import (
    Counter, Histogram, Gauge,
    generate_latest, CONTENT_TYPE_LATEST
)

# ================== LOGGING ==================
if not os.path.exists("logs"):
    os.makedirs("logs")

logging.basicConfig(
    filename="logs/monitoring.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)

app = Flask(__name__)

# ================== METRIK API ==================
TOTAL_REQUESTS = Counter('api_requests_total', 'Total incoming API requests')
REQ_LATENCY = Histogram('api_request_latency_seconds', 'API response latency')
REQ_THROUGHPUT = Counter('api_requests_rate', 'Request throughput')
FAIL_COUNT = Counter('api_failure_total', 'Failed API requests')
OK_COUNT = Counter('api_success_total', 'Successful API requests')
PAYLOAD_SIZE = Histogram('api_payload_size_bytes', 'Request payload size')
RESULT_SIZE = Histogram('api_result_size_bytes', 'Response payload size')

# ================== METRIK SISTEM ==================
CPU_LOAD = Gauge('sys_cpu_load', 'CPU load percentage')
MEM_USAGE = Gauge('sys_memory_usage', 'RAM usage percentage')
STORAGE_USAGE = Gauge('sys_storage_usage', 'Disk usage percentage')
NETWORK_OUT = Gauge('sys_network_out', 'Bytes sent')
NETWORK_IN = Gauge('sys_network_in', 'Bytes received')

# ================== SAFE DISK CHECK WINDOWS ==================
def safe_get_disk_usage():
    """Mengambil disk usage aman untuk Windows."""
    try:
        return psutil.disk_usage("C:\\").percent
    except Exception:
        for part in psutil.disk_partitions():
            try:
                return psutil.disk_usage(part.mountpoint).percent
            except:
                continue
    return 0.0


# ================== METRICS ENDPOINT ==================
@app.route('/metrics', methods=['GET'])
def metrics():
    try:
        CPU_LOAD.set(psutil.cpu_percent(interval=0.5))
        MEM_USAGE.set(psutil.virtual_memory().percent)
        STORAGE_USAGE.set(safe_get_disk_usage())

        net = psutil.net_io_counters()
        NETWORK_OUT.set(net.bytes_sent)
        NETWORK_IN.set(net.bytes_recv)

        logger.info("Metrics scraped successfully")
    except Exception as e:
        logger.error(f"Error updating metrics: {e}")

    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


# ================== PREDICT ENDPOINT ==================
@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    TOTAL_REQUESTS.inc()

    data = request.get_json()
    if data:
        PAYLOAD_SIZE.observe(len(str(data).encode('utf-8')))

    api_url = "http://127.0.0.1:5005/invocations"

    try:
        res = requests.post(api_url, json=data, timeout=10)
        latency = time.time() - start_time

        REQ_LATENCY.observe(latency)
        OK_COUNT.inc()
        RESULT_SIZE.observe(len(res.content))

        logger.info(f"Prediction success | Latency={latency:.3f}s")
        return jsonify(res.json())

    except Exception as e:
        FAIL_COUNT.inc()
        logger.error(f"Prediction FAILED: {e}")
        return jsonify({"error": str(e)}), 500


# ================== RUN PRODUCTION SERVER ==================
def start_server():
    from waitress import serve
    print("\n🚀 PRODUCTION SERVER RUNNING")
    print("→ http://127.0.0.1:9100/metrics")
    print("→ http://127.0.0.1:9100/predict\n")

    serve(app, host='127.0.0.1', port=9100)


if __name__ == '__main__':
    start_server()

