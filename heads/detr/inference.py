from heads.detr.dataset import letterbox
import numpy as np
from PIL import Image
import mlx.core as mx
from heads.detr.transformer import build_detr
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dinov3.models import vit_small

COCO_CLASSES = [
    "N/A",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "N/A",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "N/A",
    "backpack",
    "umbrella",
    "N/A",
    "N/A",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "N/A",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "N/A",
    "dining table",
    "N/A",
    "N/A",
    "toilet",
    "N/A",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "N/A",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def run_inference(image_path="test/test5.jpeg", threshold=0.62):
    dinov3_small = vit_small(
        patch_size=16,
        n_storage_tokens=4,
        layerscale_init=1e-5,
        mask_k_bias=True,
    )

    dinov3_small.load_weights("dinov3/checkpoints/model/vit-small.safetensors")
    dinov3_small.eval()

    detr_decoder = build_detr(
        d_model=384,
        num_layers=3,
        n_classes=92,
        n_points=4,
    )
    detr_decoder.load_weights("dinov3/checkpoints/model/detr_decoder.safetensors")
    detr_decoder.eval()

    # 2. Preprocess Image
    img_pil = Image.open(image_path).convert("RGB")
    img_pil, _, _, _ = letterbox(img_pil, 224)
    img_arr = np.array(img_pil, dtype=np.float32) / 255.0
    image = mx.array(img_arr)[None]  # NHWC: (1, 224, 224, 3)

    features = dinov3_small(image, masks=None, is_training=True)
    patch_tokens = features["x_norm_patchtokens"]

    # DETR Decoder
    output = detr_decoder(patch_tokens)

    # 4. Post-process
    # Logits shape: (1, 50, 92), Boxes shape: (1, 50, 4)
    logits = output["logits"][0]
    boxes = output["boxes"][0]

    # Convert logits to probabilities and filter by threshold
    probs = mx.softmax(logits, axis=-1)
    # The last class (index 91) is 'no object'
    scores = mx.max(probs[:, :-1], axis=-1)
    labels = mx.argmax(probs[:, :-1], axis=-1)

    keep = (scores > threshold).tolist()
    keep_indices = [i for i, val in enumerate(keep) if val]

    if len(keep_indices) > 0:
        indices = mx.array(keep_indices)
        scores = np.array(scores[indices].tolist())
        labels = np.array(labels[indices].tolist())
        boxes = np.array(boxes[indices].tolist())
    else:
        scores = np.array([])
        labels = np.array([])
        boxes = np.array([]).reshape(0, 4)

    # 5. Visualization
    img_size = 224
    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.imshow(img_arr)

    for score, label, (cx, cy, w, h) in zip(scores, labels, boxes):
        # Convert [cx, cy, w, h] normalized to [x, y, w, h] pixel
        x = (cx - w / 2) * img_size
        y = (cy - h / 2) * img_size
        pw = w * img_size
        ph = h * img_size

        # Draw Box
        rect = patches.Rectangle(
            (x, y), pw, ph, linewidth=2, edgecolor="red", facecolor="none", alpha=0.9
        )
        ax.add_patch(rect)

        # Add Label
        class_name = (
            COCO_CLASSES[label] if label < len(COCO_CLASSES) else f"Class {label}"
        )
        ax.text(
            x,
            y,
            f"{class_name}: {score:.2f}",
            bbox=dict(facecolor="red", alpha=0.5),
            fontsize=8,
            color="white",
        )

    plt.axis("off")
    plt.title(f"DETR Predictions (threshold={threshold})")
    plt.savefig("detr_output.png")
    print("Results saved to detr_output.png")


if __name__ == "__main__":
    run_inference()
