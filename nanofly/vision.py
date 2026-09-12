"""
Visual input: the text is drawn as a picture and fed to the photoreceptors, as in DOOMFLY and stonkfly.

How the retina is built here:
  1. Take the photoreceptors from the graph (group `photoreceptor`) and split them into a left and
     right eye by x.
  2. For each eye, flatten the 3D coordinates onto their principal axes: that gives the hexagonal
     lattice of ommatidia the picture is sampled on.

The resolution comes from biology, not from us: a fly has about 800 ommatidia per eye, so the
"screen" ends up around 28x28. One letter is readable, a whole line is not.

Modes:
  glyph   one character per step, filling the eye
  banner  the whole line as a ticker, the shift comes from the phase
  code    the sentence embedding is scattered over the retina as blobs: a barcode, not reading
"""
import numpy as np

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
]


def eye_layout(graph, min_receptors=64):
    """2D photoreceptor coordinates in [-1, 1] plus an eye label (0 left, 1 right)."""
    idx = graph["grp_photoreceptor"].astype(np.int64) if "grp_photoreceptor" in graph else np.zeros(0, np.int64)
    if len(idx) < min_receptors:
        return None
    pos = graph["pos"][idx].astype(np.float64)
    side = (pos[:, 0] > np.median(pos[:, 0])).astype(np.int64)
    xy = np.zeros((len(idx), 2))
    for s in (0, 1):
        m = side == s
        if m.sum() < 3:
            continue
        p = pos[m] - pos[m].mean(0)
        vals, vecs = np.linalg.eigh(np.cov(p.T))
        order = np.argsort(vals)[::-1][:2]
        q = p @ vecs[:, order]
        span = np.percentile(np.abs(q), 99, axis=0)
        xy[m] = np.clip(q / np.maximum(span, 1e-9), -1, 1)
    return {"idx": idx, "xy": xy.astype(np.float32), "side": side.astype(np.int8)}


def _font(size):
    from PIL import ImageFont
    for path in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def render_text(text, width, height, offset=0.0, margin=0.12):
    """White text on black. `offset`, in fractions of the line width, scrolls the ticker."""
    from PIL import Image, ImageDraw
    img = Image.new("L", (width, height), 0)
    if not text:
        return np.zeros((height, width), np.float32)
    draw = ImageDraw.Draw(img)
    size = max(6, int(height * (1 - 2 * margin)))
    font = _font(size)
    box = draw.textbbox((0, 0), text, font=font)
    tw, th = box[2] - box[0], box[3] - box[1]
    if len(text) <= 2 and tw < width:
        x = (width - tw) / 2 - box[0]
    else:
        x = width - offset * (tw + width) - box[0]
    draw.text((x, (height - th) / 2 - box[1]), text, fill=255, font=font)
    return np.asarray(img, dtype=np.float32) / 255.0


def sample_retina(image, xy, zoom=1.0):
    """Bilinear sampling at the retina points. image [H,W] in [0,1], xy [n,2] in [-1,1]."""
    h, w = image.shape
    u = (xy[:, 0] / zoom * 0.5 + 0.5) * (w - 1)
    v = (-xy[:, 1] / zoom * 0.5 + 0.5) * (h - 1)
    u = np.clip(u, 0, w - 1.001)
    v = np.clip(v, 0, h - 1.001)
    x0, y0 = np.floor(u).astype(int), np.floor(v).astype(int)
    fx, fy = u - x0, v - y0
    out = (image[y0, x0] * (1 - fx) * (1 - fy) + image[y0, x0 + 1] * fx * (1 - fy)
           + image[y0 + 1, x0] * (1 - fx) * fy + image[y0 + 1, x0 + 1] * fx * fy)
    return out.astype(np.float32)


def retina_resolution(xy, side):
    """Rough size of the "screen": receptors per eye and the square they stand in for."""
    per_eye = [int((side == s).sum()) for s in (0, 1)]
    n = max(per_eye) or 1
    return {"per_eye": per_eye, "equivalent_square": int(round(np.sqrt(n)))}


class TextRetina:
    """Text -> picture -> current on the photoreceptors."""

    def __init__(self, layout, mode="glyph", width=96, height=32, zoom=1.0, scale=1.0):
        self.layout = layout
        self.mode = mode
        self.width, self.height, self.zoom, self.scale = width, height, zoom, scale

    def frame(self, text, step=0, total=1):
        if self.mode == "glyph":
            ch = text[step % len(text)] if text else " "
            img = render_text(ch, self.height, self.height)
            img = np.pad(img, ((0, 0), ((self.width - self.height) // 2,) * 2))[:, :self.width]
        elif self.mode == "banner":
            img = render_text(text, self.width, self.height, offset=step / max(total, 1))
        else:
            raise ValueError(f"mode {self.mode} is not drawn here")
        return img

    def current(self, text, step=0, total=1):
        img = self.frame(text, step, total)
        return self.scale * sample_retina(img, self.layout["xy"], self.zoom), img


def embedding_pattern(vec, xy, blobs=64, sigma=0.22, seed=0):
    """Mode `code`: the embedding is scattered over the retina as blobs. A barcode, not reading."""
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-1, 1, size=(blobs, 2))
    w = np.asarray(vec, dtype=np.float32)
    proj = rng.normal(size=(blobs, len(w))) / np.sqrt(len(w))
    amp = np.tanh(proj @ w)
    d2 = ((xy[:, None, :] - centers[None]) ** 2).sum(-1)
    return (np.exp(-d2 / (2 * sigma ** 2)) * amp).sum(1).astype(np.float32)
