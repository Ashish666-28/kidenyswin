KidneyVision inference source bundle

Extracted unchanged from the bundled sources in KidneyVision_Tumor_Cyst_Context_Recovery_v3.ipynb. CLI wrapper and dependency list added.
Expected checkpoint format: kidneyvision_independent_swin_v2 (used by independent v2 and recovery v3).
The older tumor_cyst_native_v6 checkpoint has NOT been verified compatible. Do not bypass a format or strict-load error.
model.pth is NOT included. No checkpoint was available for runtime verification. All Python sources passed syntax parsing only.

1. Extract this ZIP and put your matching model.pth in this folder.
2. In a terminal in this folder: python -m pip install -r requirements.txt
3. Run: python run_prediction.py --model model.pth --image scan.png --output results

Entry point: kv_independent_lesions.predict_image
Loader: kv_independent_lesions.load_composite
Outputs: classification probabilities, highlighted_ct.png, segmentation_labels.png, independent_masks.npz, lesions.csv, prediction.json.
The original notebook architecture/preprocessing/threshold handling is preserved. Millimetres require verified spacing.
If a format mismatch occurs, obtain the Python files from the exact model export folder rather than changing the format check.
