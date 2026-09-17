#!/usr/bin/env python3
"""
Unit and Integration Tests for HoneyChain Standalone ML Microservice
"""

import os
import sys
import json
import time
import unittest
import threading
import urllib.request
import urllib.error

# Add parent directory to sys.path
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(TEST_DIR)
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, os.path.join(REPO_DIR, 'models'))

from service import init_model, run_server, HTTPServer, MLRequestHandler

TEST_PORT = 5099
BASE_URL = f"http://127.0.0.1:{TEST_PORT}"

class TestHoneyChainMLService(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Set up a test server thread
        cls.server = HTTPServer(('127.0.0.1', TEST_PORT), MLRequestHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever)
        cls.thread.daemon = True
        cls.thread.start()
        time.sleep(0.5)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def test_01_model_loading(self):
        model = init_model()
        self.assertIsNotNone(model)
        expected_tiers = ['T1', 'T6', 'T12', 'T24', 'T36', 'T48']
        for tier in expected_tiers:
            self.assertIn(tier, model.tiers)

    def test_02_health_endpoint(self):
        req = urllib.request.Request(f"{BASE_URL}/health")
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode('utf-8'))
            self.assertEqual(data["status"], "ok")
            self.assertEqual(data["service"], "honeychain-hive-health-ml")
            self.assertTrue(data["modelLoaded"])
            self.assertEqual(len(data["tiers"]), 6)

    def test_03_predict_empty_readings(self):
        payload = json.dumps({"hiveId": "TEST-HIVE-01", "readings": []}).encode('utf-8')
        req = urllib.request.Request(
            f"{BASE_URL}/predict",
            data=payload,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode('utf-8'))
            self.assertEqual(data["status"], "NO_DATA")
            self.assertIsNone(data["healthScore"])

    def test_04_predict_valid_inference(self):
        sample_file = os.path.join(REPO_DIR, 'models', 'sample_input.json')
        self.assertTrue(os.path.exists(sample_file), "sample_input.json must exist")
        with open(sample_file, 'r') as f:
            sample_data = json.load(f)

        payload = json.dumps(sample_data).encode('utf-8')
        req = urllib.request.Request(
            f"{BASE_URL}/predict",
            data=payload,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode('utf-8'))
            self.assertEqual(data["status"], "OK")
            self.assertIn(data["tier"], ['T1', 'T6', 'T12', 'T24', 'T36', 'T48'])
            self.assertIsInstance(data["healthScore"], (int, float))
            self.assertIn(data["stressRisk"], ['LOW', 'MEDIUM', 'HIGH'])
            self.assertIn("recommendation", data)

    def test_05_api_key_protection(self):
        # Temporarily enable API key
        os.environ['ML_API_KEY'] = 'test-secret-key-123'
        try:
            payload = json.dumps({"hiveId": "TEST-HIVE-01", "readings": []}).encode('utf-8')

            # 1. Request without key -> 401
            req_no_key = urllib.request.Request(
                f"{BASE_URL}/predict",
                data=payload,
                headers={"Content-Type": "application/json"}
            )
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                urllib.request.urlopen(req_no_key)
            self.assertEqual(ctx.exception.code, 401)

            # 2. Request with invalid key -> 401
            req_wrong_key = urllib.request.Request(
                f"{BASE_URL}/predict",
                data=payload,
                headers={"Content-Type": "application/json", "X-ML-API-Key": "wrong-key"}
            )
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                urllib.request.urlopen(req_wrong_key)
            self.assertEqual(ctx.exception.code, 401)

            # 3. Request with valid key -> 200
            req_valid_key = urllib.request.Request(
                f"{BASE_URL}/predict",
                data=payload,
                headers={"Content-Type": "application/json", "X-ML-API-Key": "test-secret-key-123"}
            )
            with urllib.request.urlopen(req_valid_key) as resp:
                self.assertEqual(resp.status, 200)

            # 4. Health endpoint remains accessible without key
            req_health = urllib.request.Request(f"{BASE_URL}/health")
            with urllib.request.urlopen(req_health) as resp:
                self.assertEqual(resp.status, 200)
        finally:
            os.environ.pop('ML_API_KEY', None)

    def test_06_dashboard_endpoints(self):
        for path in ['/', '/dashboard', '/frontend', '/ui']:
            req = urllib.request.Request(f"{BASE_URL}{path}")
            with urllib.request.urlopen(req) as resp:
                self.assertEqual(resp.status, 200)
                self.assertIn("text/html", resp.headers.get("Content-Type", ""))
                html = resp.read().decode('utf-8')
                self.assertIn("HoneyChain ML Diagnostic Studio", html)

        # Ensure Accept: application/json on / returns JSON
        req_json = urllib.request.Request(
            f"{BASE_URL}/",
            headers={"Accept": "application/json"}
        )
        with urllib.request.urlopen(req_json) as resp:
            self.assertEqual(resp.status, 200)
            self.assertIn("application/json", resp.headers.get("Content-Type", ""))
            data = json.loads(resp.read().decode('utf-8'))
            self.assertEqual(data["status"], "ok")

    def test_07_static_assets(self):
        for path, expected_ctype in [
            ('/favicon.ico', 'image/x-icon'),
            ('/logoml.png', 'image/png'),
            ('/favicon.png', 'image/png'),
            ('/favicon-32x32.png', 'image/png'),
            ('/dashboard/logoml.png', 'image/png')
        ]:
            req = urllib.request.Request(f"{BASE_URL}{path}")
            with urllib.request.urlopen(req) as resp:
                self.assertEqual(resp.status, 200)
                self.assertIn(expected_ctype, resp.headers.get("Content-Type", ""))
                content = resp.read()
                self.assertGreater(len(content), 0)

if __name__ == '__main__':
    unittest.main()
