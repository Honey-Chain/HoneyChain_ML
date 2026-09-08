#!/usr/bin/env python3
"""
HoneyChain - Hive Health Independent Machine Learning Microservice
Exposes HTTP REST endpoints on 0.0.0.0:${PORT:-5001} for high-performance ML inference.
Loads all 26 model artifacts once into memory at startup.
"""

import sys
import os
import json
import argparse
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler

# Ensure models directory is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
if MODELS_DIR not in sys.path:
    sys.path.insert(0, MODELS_DIR)

try:
    from predict import get_model, predict as run_predict
except ImportError as e:
    print(f"[FATAL] Failed to import predict module from {MODELS_DIR}: {e}", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [HoneyChain-ML] %(levelname)s: %(message)s'
)
logger = logging.getLogger("HoneyChain-ML")

# Global singleton model loaded once
_HIVE_MODEL = None

def init_model():
    global _HIVE_MODEL
    if _HIVE_MODEL is None:
        logger.info(f"Loading Hive Health Model artifacts from: {MODELS_DIR}")
        _HIVE_MODEL = get_model()
        logger.info(f"Model successfully loaded. Available tiers: {list(_HIVE_MODEL.tiers.keys())}")
    return _HIVE_MODEL

DASHBOARD_FILE = os.path.join(BASE_DIR, 'dashboard.html')
_DASHBOARD_HTML = None

def get_dashboard_html():
    global _DASHBOARD_HTML
    if _DASHBOARD_HTML is None or os.environ.get('ENV') == 'development':
        if os.path.exists(DASHBOARD_FILE):
            with open(DASHBOARD_FILE, 'r', encoding='utf-8') as f:
                _DASHBOARD_HTML = f.read()
        else:
            _DASHBOARD_HTML = "<!DOCTYPE html><html><body><h1>HoneyChain ML Diagnostic Studio</h1><p>dashboard.html not found</p></body></html>"
    return _DASHBOARD_HTML

def get_configured_api_key():
    key = os.environ.get('ML_API_KEY', '').strip()
    return key if key else None

class MLRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Silence default access log output; keep custom logging
        pass

    def _send_json(self, status_code, data):
        response_bytes = json.dumps(data).encode('utf-8')
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(response_bytes)))
        # Restrictive backend-to-backend CORS
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, X-ML-API-Key')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.end_headers()
        self.wfile.write(response_bytes)

    def _send_html(self, status_code, html_content):
        response_bytes = html_content.encode('utf-8')
        self.send_response(status_code)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(response_bytes)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(response_bytes)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, X-ML-API-Key')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.end_headers()

    def _is_authorized(self):
        expected_key = get_configured_api_key()
        if not expected_key:
            # No API key configured -> open access (standard for local development)
            return True
        provided_key = self.headers.get('X-ML-API-Key', '').strip()
        return provided_key == expected_key

    def do_GET(self):
        clean_path = self.path.split('?')[0].rstrip('/')
        if clean_path == '/health':
            try:
                model = init_model()
                tiers = list(model.tiers.keys()) if model else []
                self._send_json(200, {
                    "status": "ok",
                    "service": "honeychain-hive-health-ml",
                    "version": "1.0.0",
                    "modelLoaded": model is not None,
                    "tiers": tiers
                })
            except Exception as e:
                logger.error(f"Health check model initialization failure: {e}", exc_info=True)
                self._send_json(503, {
                    "status": "error",
                    "service": "honeychain-hive-health-ml",
                    "version": "1.0.0",
                    "modelLoaded": False,
                    "error": str(e)
                })
        elif clean_path in ('', '/dashboard', '/frontend', '/ui'):
            accept = self.headers.get('Accept', '')
            if clean_path == '' and 'application/json' in accept and 'text/html' not in accept:
                try:
                    model = init_model()
                    tiers = list(model.tiers.keys()) if model else []
                    self._send_json(200, {
                        "status": "ok",
                        "service": "honeychain-hive-health-ml",
                        "version": "1.0.0",
                        "modelLoaded": model is not None,
                        "tiers": tiers
                    })
                except Exception as e:
                    self._send_json(503, {"status": "error", "error": str(e)})
            else:
                html = get_dashboard_html()
                self._send_html(200, html)
        elif clean_path == '/api/sample-input':
            sample_path = os.path.join(MODELS_DIR, 'sample_input.json')
            if os.path.exists(sample_path):
                with open(sample_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self._send_json(200, data)
            else:
                self._send_json(404, {"error": "sample_input.json not found"})
        else:
            self._send_json(404, {"error": "Not Found", "path": self.path})

    def do_POST(self):
        if self.path != '/predict':
            return self._send_json(404, {"error": "Not Found", "path": self.path})

        # Validate API Key if configured
        if not self._is_authorized():
            logger.warning("Rejected unauthorized POST /predict request (invalid or missing X-ML-API-Key)")
            return self._send_json(401, {
                "error": "Unauthorized",
                "message": "Invalid or missing X-ML-API-Key header"
            })

        content_len = int(self.headers.get('Content-Length', 0))
        if content_len == 0:
            return self._send_json(400, {
                "status": "NO_DATA",
                "message": "Empty request body"
            })

        try:
            raw_body = self.rfile.read(content_len).decode('utf-8')
            payload = json.loads(raw_body)
        except Exception as e:
            return self._send_json(400, {
                "status": "NO_DATA",
                "message": f"Malformed JSON: {str(e)}"
            })

        hive_id = payload.get('hiveId', 'UNKNOWN')
        readings = payload.get('readings', [])

        if not isinstance(readings, list) or len(readings) == 0:
            return self._send_json(200, {
                "hiveId": hive_id,
                "status": "NO_DATA",
                "message": "No sensor readings provided for inference.",
                "tier": None,
                "healthScore": None,
                "stressRisk": None,
                "stressProbability": None,
                "abnormalityRisk": None
            })

        try:
            model = init_model()
            result = model.predict(readings=readings, hive_id=hive_id)
            return self._send_json(200, result)
        except Exception as err:
            logger.error(f"Inference error for hive {hive_id}: {err}", exc_info=True)
            return self._send_json(500, {
                "hiveId": hive_id,
                "status": "INFERENCE_ERROR",
                "message": "Model inference failed internally.",
                "tier": None,
                "healthScore": None,
                "stressRisk": None
            })

def run_server(host='0.0.0.0', port=5001):
    init_model()
    server = HTTPServer((host, port), MLRequestHandler)
    api_key = get_configured_api_key()
    logger.info(f"HoneyChain Hive Health Inference API active on http://{host}:{port}")
    logger.info(f"Endpoints: GET /health | POST /predict")
    logger.info(f"Security: API Key Protection {'ENABLED' if api_key else 'DISABLED (Open Mode)'}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down inference server...")
        server.server_close()

def main():
    parser = argparse.ArgumentParser(description="HoneyChain Standalone ML Inference Microservice")
    parser.add_argument('--host', default=os.environ.get('HOST', '0.0.0.0'), help="Host binding (default: 0.0.0.0)")
    parser.add_argument('--port', type=int, default=int(os.environ.get('PORT', os.environ.get('ML_PORT', '5001'))), help="Port binding (default: 5001)")
    parser.add_argument('--predict-json', help="Run single prediction on JSON string and exit")
    parser.add_argument('--health-check', action='store_true', help="Check model loading and exit")

    args = parser.parse_args()

    if args.health_check:
        model = init_model()
        print(json.dumps({"status": "ok", "tiers": list(model.tiers.keys())}))
        sys.exit(0)

    if args.predict_json:
        init_model()
        data = json.loads(args.predict_json)
        res = run_predict(data.get('readings', []), hive_id=data.get('hiveId', 'UNKNOWN'))
        print(json.dumps(res, indent=2))
        sys.exit(0)

    run_server(host=args.host, port=args.port)

if __name__ == '__main__':
    main()
