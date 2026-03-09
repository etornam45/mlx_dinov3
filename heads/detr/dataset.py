import numpy as np
from PIL import Image
from pycocotools.coco import COCO
import mlx.data as dx


def letterbox(image, target_size):
    orig_w, orig_h = image.size
    scale = target_size / max(orig_w, orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    image = image.resize((new_w, new_h), Image.BILINEAR)
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2
    canvas = Image.new("RGB", (target_size, target_size), (114, 114, 114))
    canvas.paste(image, (pad_x, pad_y))
    return canvas, scale, pad_x, pad_y


def transform_boxes(boxes, scale, pad_x, pad_y, target_size):
    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    boxes = np.array(boxes, dtype=np.float32)
    boxes[:, 0] = boxes[:, 0] * scale + pad_x
    boxes[:, 1] = boxes[:, 1] * scale + pad_y
    boxes[:, 2] = boxes[:, 2] * scale
    boxes[:, 3] = boxes[:, 3] * scale
    boxes[:, 0] = (boxes[:, 0] + boxes[:, 2] / 2) / target_size
    boxes[:, 1] = (boxes[:, 1] + boxes[:, 3] / 2) / target_size
    boxes[:, 2] /= target_size
    boxes[:, 3] /= target_size
    return np.clip(boxes, 0.0, 1.0)


def load_coco(img_dir, ann_file, img_size=640):
    """
    Returns a list of lightweight dicts — NO images loaded yet.
    Each dict stores only the file path (as bytes) + pre-parsed annotations.
    Images are loaded lazily when the sample is accessed.
    """
    coco = COCO(ann_file)
    samples = []

    for img_id in coco.getImgIds():
        img_info = coco.loadImgs(img_id)[0]
        path = f"{img_dir}/{img_info['file_name']}"

        # ── annotations only (fast, no I/O) ──────────────────────────────
        ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns = coco.loadAnns(ann_ids)
        orig_w, orig_h = img_info["width"], img_info["height"]

        scale = img_size / max(orig_w, orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        pad_x = (img_size - new_w) // 2
        pad_y = (img_size - new_h) // 2

        boxes = transform_boxes(
            [a["bbox"] for a in anns], scale, pad_x, pad_y, img_size
        )
        labels = np.array([a["category_id"] for a in anns], dtype=np.int32)

        samples.append(
            {
                "image": path.encode("ascii"),  # NOTE: this is actually the path
                "boxes": boxes,
                "labels": labels,
                "num_objects": np.int32(len(labels)),
            }
        )

    return samples


def make_stream(img_dir, ann_file, img_size=640, batch_size=16, shuffle=False):
    samples = load_coco(img_dir, ann_file, img_size)  # instant — no image I/O

    def load_and_letterbox(path_bytes):
        """Called per-sample only when the batch is prefetched."""
        path = path_bytes.tobytes().decode("ascii")
        image = Image.open(path).convert("RGB")
        image, _, _, _ = letterbox(image, img_size)
        return np.array(image, dtype=np.float32) / 255.0

    buffer = dx.buffer_from_vector(samples)
    if shuffle:
        buffer = buffer.shuffle()

    return (
        buffer.to_stream()
        .key_transform("image", load_and_letterbox)  # lazy I/O here
        .batch(batch_size)
        .prefetch(prefetch_size=8, num_threads=8)  # parallel image loading
    ), len(samples)


if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import mlx.core as mx

    img_size = 224

    t0 = time.time()
    stream, _ = make_stream(
        "coco/images/val2017",
        "coco/annotations/instances_val2017.json",
        img_size=img_size,
        batch_size=16,
        shuffle=False,
    )
    print(f"Stream ready in {time.time() - t0:.2f}s")

    stream.reset()
    batch = next(stream)
    images = mx.array(batch["image"])  # (B, H, W, 3)
    boxes = np.array(batch["boxes"])  # (B, N, 4)
    n_objs = np.array(batch["num_objects"])  # (B,)

    img = np.array(images[0])

    print(img.shape)
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    for cx, cy, w, h in boxes[0, : n_objs[0]]:
        x = (cx - w / 2) * img_size
        y = (cy - h / 2) * img_size
        ax.add_patch(
            patches.Rectangle(
                (x, y),
                w * img_size,
                h * img_size,
                linewidth=1,
                edgecolor="pink",
                facecolor="none",
            )
        )
    plt.show()
