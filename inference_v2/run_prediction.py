import argparse
import json
from kv_independent_lesions import predict_image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="prediction_output")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()
    result = predict_image(args.model, args.image, args.output, device=args.device)
    print(json.dumps(result, indent=2))
