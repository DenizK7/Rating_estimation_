import unittest
from fastapi.testclient import TestClient
from api import app 

class APITestCase(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    def test_predict(self):
        response = self.client.post("/predict", json={
            "review_text": "This is a great product. I really enjoyed using it!",
            "summary": "Great product",
            "helpful_ratio": 0.8
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn("predicted_rating", response.json())

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
