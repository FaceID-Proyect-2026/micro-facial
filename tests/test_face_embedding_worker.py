import base64

import pytest

from app.workers.face_embedding_worker import decode_image_base64

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")


def jpeg_data_uri() -> str:
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def test_decodes_data_uri_image():
    image = decode_image_base64(jpeg_data_uri())

    assert image.shape == (20, 20, 3)


def test_rejects_invalid_base64_image():
    with pytest.raises(ValueError, match="base64"):
        decode_image_base64("not-valid-base64")
