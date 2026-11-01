"""Measure a warmed service using InsightFace's bundled public test image.

Never enrolls a face, reads user captures or writes to PostgreSQL. A still test
image must be rejected by the existing liveness policy; this is not camera QA.
"""

import argparse
import base64
import hashlib
import json
from time import perf_counter
from importlib.util import find_spec
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import cv2

from app.core.config import settings


def sample_payload(flow=False):
    package = find_spec('insightface')
    if package is None or package.origin is None:
        raise RuntimeError('InsightFace public sample is not installed')
    # Read its bundled sample without importing another full model runtime in
    # this HTTP client; that would compete with the server for scarce memory.
    image = cv2.imread(str(Path(package.origin).parent / 'data/images/t1.jpg'))
    # Fixed single-face fixture in the bundled t1 group photo, not production cropping.
    if image is None or image.shape[:2] != (886, 1280):
        raise RuntimeError("The bundled t1 image changed; review the benchmark fixture")
    image = image[180:450, 410:650]
    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        raise RuntimeError("Could not encode the public test image")
    raw = encoded.tobytes()
    frame = "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii")
    if flow:
        # Different byte streams ensure each of the two capture groups is new;
        # only the final recheck reuses those exact frames, as the real client does.
        frames = []
        for quality in range(70, 84):
            encoded_ok, encoded_frame = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if not encoded_ok:
                raise RuntimeError('Could not encode a flow frame')
            frames.append('data:image/jpeg;base64,' + base64.b64encode(encoded_frame.tobytes()).decode('ascii'))
    else:
        frames = [frame] * 7
    return {"image_frames": frames, "liveness_challenge": "MOVE_UP"}, hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8090")
    parser.add_argument("--label", default="current")
    parser.add_argument("--repeat", type=int, choices=range(1, 6), default=1)
    parser.add_argument("--flow", action="store_true", help="Measure two fresh 7-frame groups and their 14-frame recheck")
    args = parser.parse_args()
    payload, checksum = sample_payload(args.flow)
    print(json.dumps({"label": args.label, "sample": "insightface/data/images/t1", "crop_sha256": checksum,
                      "frames": len(payload["image_frames"])}), flush=True)
    for index in range(args.repeat):
        groups = [('challenge-1', payload['image_frames'][:7]),
                  ('challenge-2', payload['image_frames'][7:]),
                  ('final-recheck', payload['image_frames'])] if args.flow else [('liveness', payload['image_frames'])]
        for phase, frames in groups:
            measure(args, index, phase, {**payload, 'image_frames': frames})


def measure(args, index, phase, payload):
    request = Request(args.url.rstrip("/") + "/api/v1/facial-embeddings/liveness-check",
                      data=json.dumps(payload).encode(),
                      headers={"Content-Type": "application/json", "X-API-Key": settings.api_key})
    started = perf_counter()
    try:
        with urlopen(request, timeout=180) as response:
            status, result = response.status, json.load(response)
    except HTTPError as error:
        status, result = error.code, json.load(error)
    print(json.dumps({"label": args.label, "request": index+1, "phase": phase, "frames": len(payload['image_frames']), "seconds": round(perf_counter()-started, 4),
                      "http_status": status, "result": result}), flush=True)
    if status != 200 or result.get("live") is not False:
        raise RuntimeError("Expected HTTP 200 with a rejected still capture; this is not a valid liveness benchmark")


if __name__ == "__main__":
    main()
