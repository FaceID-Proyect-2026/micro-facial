"""Synthetic landmark signals shared by unit and HTTP integration tests only."""

from app.workers.face_embedding_worker import GeneratedEmbedding


SIGNALS = {
    "MOVE_UP": ("mesh_nose_y", [.50, .50, .48, .46, .44, .42, .42]),
    "MOVE_DOWN": ("mesh_nose_y", [.50, .50, .52, .54, .56, .58, .58]),
}
BOXES = {
    "MOVE_CLOSER": [-6, -6, 106, 106],
    "MOVE_AWAY": [6, 6, 94, 94],
}
CODES = tuple(SIGNALS) + tuple(BOXES)
MESH_CODES = tuple(SIGNALS)


def frames_for(code=None):
    frames = []
    for index in range(7):
        metrics = {
            "mesh_available": 1.0, "mesh_eye_ear": .30,
            "mesh_mouth_open_ratio": .02, "mesh_mouth_width_ratio": .40,
            "mesh_mouth_white_ratio": .01, "mesh_nose_x": 0., "mesh_nose_y": .50,
        }
        if code in SIGNALS:
            key, values = SIGNALS[code]
            metrics[key] = values[index]
        box = [0., 0., 100., 100.]
        if code in BOXES:
            progress = max(0., (index - 1) / 5)
            box = [start + (end - start) * progress for start, end in zip(box, BOXES[code])]
        frames.append(GeneratedEmbedding([.1] * 128, "signal-fixture", 1,
                      [[30., 30.], [70., 30.]], box, metrics))
    return frames


