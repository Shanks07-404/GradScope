"""
Quick test of FastAPI endpoints and WebSocket handshake
"""

import unittest
from fastapi.testclient import TestClient
from server.main import app


class TestServerAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_endpoint(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")

    def test_websocket_initial_snapshot(self):
        with self.client.websocket_connect("/ws/train") as ws:
            data = ws.receive_json()
            self.assertEqual(data["type"], "snapshot")
            self.assertEqual(data["step"], 0)
            self.assertIn("graph", data)
            self.assertIn("manifold", data)
            self.assertGreater(len(data["manifold"]), 0)

            # Test step command
            ws.send_json({"action": "step"})
            step_data = ws.receive_json()
            self.assertEqual(step_data["step"], 1)


if __name__ == '__main__':
    unittest.main()
