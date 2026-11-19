"""Real-checkpoint smoke test; exercises model.pth, tiling, segmentation, and state."""
import base64
from io import BytesIO
import numpy as np
from PIL import Image


def main():
    pixels = np.random.default_rng(42).integers(0, 256, (256, 320), dtype=np.uint8)
    source = BytesIO()
    Image.fromarray(pixels).save(source, format="PNG")
    image_bytes = source.getvalue()

    # Try live server first; fall back to in-memory test client
    use_live = False
    try:
        import requests
        session = requests.Session()
        res = session.get("http://127.0.0.1:5000/api/chat/session", timeout=1.5)
        if res.status_code == 200:
            use_live = True
    except Exception:
        use_live = False

    if use_live:
        token = res.json()["csrf_token"]
        post_res = session.post(
            "http://127.0.0.1:5000/api/predict",
            headers={"X-CSRF-Token": token},
            files={"image": ("synthetic-smoke.png", image_bytes, "image/png")},
            timeout=180,
        )
        post_res.raise_for_status()
        result = post_res.json()
        restored = session.get("http://127.0.0.1:5000/api/chat/session", timeout=15).json()
    else:
        from app import create_app
        app = create_app({"TESTING": True, "SECRET_KEY": "smoke-test-key"})
        client = app.test_client()
        bootstrap = client.get("/api/chat/session").get_json()
        token = bootstrap["csrf_token"]
        response = client.post(
            "/api/predict",
            headers={"X-CSRF-Token": token},
            data={"image": (BytesIO(image_bytes), "synthetic-smoke.png")},
        )
        result = response.get_json()
        restored = client.get("/api/chat/session").get_json()

    assert result["ok"] is True
    assert set(result["probabilities"]) == {"normal", "stone", "tumor", "cyst"}
    assert abs(sum(result["probabilities"].values()) - 1) < 0.0001
    assert result["label"] == max(result["probabilities"], key=result["probabilities"].get)

    seg = result["segmentation"]
    assert (seg["width"], seg["height"]) == (320, 256)
    for key in ("overlay", "mask"):
        with Image.open(BytesIO(base64.b64decode(seg[key].split(",", 1)[1]))) as image:
            assert image.size == (320, 256)
    assert [r["label"] for r in seg["regions"]] == ["stone", "tumor", "cyst"]
    assert [r["threshold"] for r in seg["regions"]] == [0.55, 0.7, 0.15]
    assert all(0 <= r["pixels"] <= pixels.size for r in seg["regions"])

    assert restored["prediction"]["label"] == result["label"]
    assert "segmentation" not in restored["prediction"]
    mode_str = "live server" if use_live else "in-memory client"
    print(f"PASS ({mode_str}): real checkpoint classification, segmentation PNGs, dimensions, thresholds and chat state.")


if __name__ == "__main__":
    main()
