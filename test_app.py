"""Comprehensive tests for the integrated KidneyVision Flask web application."""
import base64
import unittest
from io import BytesIO
from PIL import Image
import pymupdf

from app import create_app
from segmentation_pdf import build_segmentation_pdf


def sample_png_data_url(color=(255, 0, 0)):
    img = Image.new("RGB", (16, 16), color)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


class IntegratedAppChecks(unittest.TestCase):
    def setUp(self):
        png_url = sample_png_data_url()

        def sample_classifier(stream):
            with Image.open(stream) as img:
                img.verify()
            return {
                "label": "stone",
                "probabilities": {
                    "normal": 0.01, "stone": 0.97, "tumor": 0.01, "cyst": 0.01,
                },
                "segmentation": {
                    "overlay": png_url,
                    "mask": png_url,
                    "regions": [{"label": "stone", "pixels": 100, "threshold": 0.55}],
                },
            }

        self.app = create_app({
            "TESTING": True,
            "SECRET_KEY": "test-secret",
            "STATELESS_SESSIONS": False,
            "WEB_ORIGINS": {"http://localhost:5500"},
            "TRUSTED_HOSTS": ["localhost", "127.0.0.1", "jarvis.tail6ebedb.ts.net"],
        }, classifier=sample_classifier)
        self.client = self.app.test_client()

    def test_root_serves_frontend(self):
        page = self.client.get("/")
        self.assertEqual(page.status_code, 200)
        content = page.get_data(as_text=True)
        self.assertIn("KidneyVision", content)
        self.assertIn("CT scan analysis", content)
        self.assertIn("Kidney Assistant", content)
        self.assertEqual(self.client.get("/health").get_json(), {"ok": True})
        self.assertTrue(self.client.get("/api/chat/session").get_json()["ok"])

    def test_static_assets_served(self):
        css = self.client.get("/css/style.css")
        self.assertEqual(css.status_code, 200)

        js = self.client.get("/js/script.js")
        self.assertEqual(js.status_code, 200)

        hero = self.client.get("/kidney-hero.png")
        self.assertEqual(hero.status_code, 200)

    def test_trusted_hosts(self):
        for host in ("localhost", "127.0.0.1", "jarvis.tail6ebedb.ts.net"):
            with self.subTest(host=host):
                response = self.client.get("/health", headers={"Host": host})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.get_json(), {"ok": True})

        for host in ("attacker.example", "attacker-owned.vercel.app"):
            with self.subTest(host=host):
                self.assertEqual(
                    self.client.get("/health", headers={"Host": host}).status_code,
                    400,
                )

    def test_local_dev_cors(self):
        good = self.client.get("/api/chat/session", headers={"Origin": "http://localhost:5500"})
        self.assertEqual(good.headers.get("Access-Control-Allow-Origin"), "http://localhost:5500")
        self.assertEqual(good.headers.get("Access-Control-Allow-Credentials"), "true")
        bad = self.client.get("/api/chat/session", headers={"Origin": "https://untrusted.example"})
        self.assertIsNone(bad.headers.get("Access-Control-Allow-Origin"))

    def test_prediction_direct_and_api(self):
        session = self.client.get("/api/chat/session").get_json()
        image = BytesIO()
        Image.new("L", (64, 64), 120).save(image, format="PNG")

        # Test /api/predict
        result = self.client.post(
            "/api/predict",
            data={"image": (BytesIO(image.getvalue()), "ct.png")},
            headers={"X-CSRF-Token": session["csrf_token"]},
        ).get_json()
        self.assertEqual(result["label"], "stone")
        self.assertEqual(result["score_type"], "uncalibrated_softmax")
        self.assertIsInstance(result["scan_id"], str)
        self.assertIn("pdf", result["segmentation"])

        # Test direct /predict
        session2 = self.client.get("/chat/session").get_json()
        result2 = self.client.post(
            "/predict",
            data={"image": (BytesIO(image.getvalue()), "ct.png")},
            headers={"X-CSRF-Token": session2["csrf_token"]},
        ).get_json()
        self.assertEqual(result2["label"], "stone")
        self.assertEqual(result2["score_type"], "uncalibrated_softmax")
        self.assertIsInstance(result2["scan_id"], str)

    def test_segmentation_pdf_generation(self):
        png_url = sample_png_data_url()
        pred = {"label": "stone", "confidence": 0.95}
        seg = {"overlay": png_url, "mask": png_url}
        pdf_bytes = build_segmentation_pdf(pred, seg)

        self.assertTrue(pdf_bytes.startswith(b"%PDF-"))
        with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
            self.assertEqual(len(doc), 1)
            text = doc[0].get_text()
            self.assertIn("Stone", text)
            self.assertIn("95.00%", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
