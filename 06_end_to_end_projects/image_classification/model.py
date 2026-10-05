"""Small CNN, checkpointing and an inference/serving stub."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from data import CLASSES


class SmallCNN(nn.Module):
    """conv-bn-relu-pool x2 -> global average pool -> linear. Works for any input size >= 8."""

    def __init__(self, num_classes: int = len(CLASSES), width: int = 8):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(width, width * 2, 3, padding=1), nn.BatchNorm2d(width * 2), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.head = nn.Linear(width * 2, num_classes)

    def forward(self, x):
        h = self.features(x)
        h = h.mean(dim=(2, 3))            # global average pooling -> (B, C)
        return self.head(h)


def save_checkpoint(model, path, **meta):
    """Save weights plus everything needed to rebuild the model."""
    torch.save({"state_dict": model.state_dict(), "classes": CLASSES,
                "width": model.head.in_features // 2, "meta": meta}, path)


def load_checkpoint(path):
    ck = torch.load(path, map_location="cpu", weights_only=True)  # safe load: tensors/primitives only
    model = SmallCNN(len(ck["classes"]), ck["width"])
    model.load_state_dict(ck["state_dict"])
    model.eval()                          # inference mode: BatchNorm uses running stats
    return model, ck


class Predictor:
    """Serving stub: takes a nested list / tensor image, returns a JSON-friendly dict."""

    def __init__(self, checkpoint_path):
        self.model, ck = load_checkpoint(checkpoint_path)
        self.classes = ck["classes"]

    @torch.inference_mode()
    def predict(self, image, top_k: int = 2):
        x = torch.as_tensor(image, dtype=torch.float32)
        if x.dim() == 2:
            x = x[None, None]             # (H,W) -> (1,1,H,W)
        elif x.dim() == 3:
            x = x[None]
        probs = F.softmax(self.model(x), dim=-1)[0]
        p, idx = probs.topk(min(top_k, len(self.classes)))
        return {"label": self.classes[idx[0].item()],
                "top_k": [{"label": self.classes[i.item()], "prob": round(v.item(), 4)} for v, i in zip(p, idx)]}

    def handle_request(self, payload: dict) -> dict:
        """What a web framework route would call: validate -> predict -> respond."""
        if "image" not in payload:
            return {"error": "missing 'image'"}
        try:
            return self.predict(payload["image"])
        except Exception as e:           # bad shape etc. must not crash the server
            return {"error": str(e)}
