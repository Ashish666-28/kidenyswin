"""In-memory adapter for the supplied independent-Swin inference pipeline."""
from pathlib import Path
from functools import lru_cache
from threading import Lock
from io import BytesIO
import base64
import sys
import numpy as np
import torch
from PIL import Image, ImageOps

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = BASE_DIR / "models/weights/model.pth"
# The supplied notebook modules use absolute sibling imports.
sys.path.insert(0, str(BASE_DIR / "inference_v2"))
from kv_independent_lesions import load_composite, mask_rule
from kv_data import raster_normalize
from kv_infer import predict_slice, components_2d
from kv_joint_infer import predict_classification

MASK_CLASSES = ("stone", "tumor", "cyst")
COLORS = ((255, 80, 80), (255, 190, 50), (70, 170, 255))

def png_url(array):
    buffer = BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")

class KidneyPredictor:
    def __init__(self, weights_path=DEFAULT_WEIGHTS):
        self.model, self.metadata = load_composite(Path(weights_path), device="cpu")
        self.classes = tuple(self.metadata["classes"])
        self._lock = Lock()

    def predict(self, image_source):
        # Same raster preparation as read_2d; uploads stay in memory.
        with Image.open(image_source) as source:
            image = np.asarray(ImageOps.exif_transpose(source).convert("L"))
        x = raster_normalize(image)
        tile = self.metadata["tile_size"]
        with self._lock, torch.inference_mode():
            probabilities = predict_slice(self.model, x, tile=tile, overlap=.25,
                                          device="cpu", infer_batch=2, tta=False)
            masks = np.stack([
                mask_rule(probabilities[i], self.metadata["segmentation_thresholds"][i],
                          int(self.metadata.get("mask_postprocess", {}).get(name, {}).get("min_pixels", 0)))
                for i, name in enumerate(MASK_CLASSES)
            ])
            scores = predict_classification(self.model, x, tile=tile, device="cpu")
        label = max(scores, key=scores.get)
        rgb = np.repeat((np.clip(x, 0, 1) * 255).astype(np.uint8)[..., None], 3, axis=2)
        overlay = rgb.astype(np.float32)
        mask_rgb = np.zeros_like(rgb)
        regions = []
        for i, name in enumerate(MASK_CLASSES):
            selected = masks[i].astype(bool)
            overlay[selected] = .55 * overlay[selected] + .45 * np.asarray(COLORS[i])
            mask_rgb[selected] = COLORS[i]
            regions.append({"label": name, "pixels": int(selected.sum()),
                            "threshold": float(self.metadata["segmentation_thresholds"][i]),
                            "color": "#%02x%02x%02x" % COLORS[i]})
        lesions = components_2d(masks, spacing=None, provenance="No verified spacing; pixels only")
        warnings = []
        if (masks.sum(0) > 1).any():
            warnings.append("Independent masks overlap; colors are blended in the overlay.")
        if label == "normal" and masks.any():
            warnings.append("Classification is Normal, but segmentation identifies candidate regions.")
        if label != "normal" and not masks.any():
            warnings.append("No segmentation region passed the saved thresholds.")
        return {"label": label, "class_index": self.classes.index(label),
                "confidence": float(scores[label]), "probabilities": scores,
                "score_type": "uncalibrated_softmax",
                "segmentation": {"overlay": png_url(overlay.astype(np.uint8)),
                                 "mask": png_url(mask_rgb), "regions": regions,
                                 "width": int(x.shape[1]), "height": int(x.shape[0]),
                                 "lesions": lesions, "warnings": warnings,
                                 "note": "Predicted regions. Physical size in mm requires verified pixel spacing."}}

@lru_cache(maxsize=1)
def get_predictor():
    return KidneyPredictor()

def predict_image(image_source):
    return get_predictor().predict(image_source)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("image", nargs="?")
    args = parser.parse_args()
    predictor = get_predictor()
    print("Checkpoint strictly loaded; classes:", predictor.classes)
    if args.image:
        result = predictor.predict(args.image)
        print("Classification:", result["label"], result["probabilities"])
        print("Segmentation:", result["segmentation"]["regions"])
