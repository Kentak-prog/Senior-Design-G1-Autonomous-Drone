"""A small convolutional neural network written in plain NumPy.

Why NumPy? It trains in minutes on a laptop CPU, needs no GPU or PyTorch install,
and runs on the drone's companion computer with nothing but numpy. The layer math
is the same as torch.nn.Conv2d, so porting to PyTorch later is straightforward.

Layout is NHWC: (batch, height, width, channels).

Architecture (input 80 x 128 RGB, output 20 x 32 x 4 heatmaps):
    conv 3x3 stride 2   3 -> 16     80x128 -> 40x64
    conv 3x3 stride 2  16 -> 32     40x64  -> 20x32
    conv 3x3           32 -> 32
    conv 3x3 dil 2     32 -> 48     (dilation widens the view without shrinking the map)
    conv 3x3 dil 4     48 -> 48
    conv 1x1           48 -> 4      one heatmap per corner: TL, TR, BR, BL
Every conv except the last is followed by ReLU. ~60k parameters.
"""
import numpy as np

ARCH = [  # (in, out, kernel, stride, dilation, relu)
    (3, 16, 3, 2, 1, True),
    (16, 32, 3, 2, 1, True),
    (32, 32, 3, 1, 1, True),
    (32, 48, 3, 1, 2, True),
    (48, 48, 3, 1, 4, True),
    (48, 4, 1, 1, 1, False),
]


class Conv:
    def __init__(self, cin, cout, k, s, d, relu, rng):
        self.cin, self.cout, self.k, self.s, self.d, self.relu = cin, cout, k, s, d, relu
        self.p = d * (k - 1) // 2
        fan_in = cin * k * k
        self.W = (rng.standard_normal((k * k * cin, cout)) * np.sqrt(2 / fan_in)).astype(np.float32)
        self.b = np.zeros(cout, np.float32)

    def _shape(self, H, W):
        Ho = (H + 2 * self.p - self.d * (self.k - 1) - 1) // self.s + 1
        Wo = (W + 2 * self.p - self.d * (self.k - 1) - 1) // self.s + 1
        return Ho, Wo

    def forward(self, x, train=False):
        N, H, W, C = x.shape
        Ho, Wo = self._shape(H, W)
        k, s, d, p = self.k, self.s, self.d, self.p
        xp = np.pad(x, ((0, 0), (p, p), (p, p), (0, 0))) if p else x
        if k == 1:
            cols = xp[:, ::s, ::s, :].reshape(-1, C)
        else:
            cols = np.stack([xp[:, i * d:i * d + s * (Ho - 1) + 1:s, j * d:j * d + s * (Wo - 1) + 1:s, :]
                             for i in range(k) for j in range(k)], axis=3).reshape(-1, k * k * C)
        out = (cols @ self.W + self.b).reshape(N, Ho, Wo, self.cout)
        if self.relu:
            out = np.maximum(out, 0)
        if train:
            self.cache = (cols, xp.shape, out, (N, H, W, C, Ho, Wo))
        return out

    def backward(self, g):
        cols, xp_shape, out, (N, H, W, C, Ho, Wo) = self.cache
        if self.relu:
            g = g * (out > 0)
        g2 = g.reshape(-1, self.cout)
        self.dW = cols.T @ g2
        self.db = g2.sum(0)
        dcols = (g2 @ self.W.T)
        k, s, d, p = self.k, self.s, self.d, self.p
        dxp = np.zeros(xp_shape, np.float32)
        if k == 1:
            dxp[:, ::s, ::s, :] = dcols.reshape(N, Ho, Wo, C)
        else:
            dcols = dcols.reshape(N, Ho, Wo, k * k, C)
            for idx in range(k * k):
                i, j = divmod(idx, k)
                dxp[:, i * d:i * d + s * (Ho - 1) + 1:s, j * d:j * d + s * (Wo - 1) + 1:s, :] += dcols[:, :, :, idx, :]
        self.cache = None
        return dxp[:, p:p + H, p:p + W, :] if p else dxp


class GateNet:
    MEAN = np.array([40, 20, 40], np.float32)   # rough BGR mean of gate frames
    STD = 60.0

    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        self.layers = [Conv(*a, rng) for a in ARCH]

    def preprocess(self, bgr_uint8_batch):
        return (bgr_uint8_batch.astype(np.float32) - self.MEAN) / self.STD

    def forward(self, x, train=False):
        for L in self.layers:
            x = L.forward(x, train)
        return x  # logits, (N, 20, 32, 4)

    def backward(self, g):
        for L in reversed(self.layers):
            g = L.backward(g)

    def params(self):
        for L in self.layers:
            yield L, "W"
            yield L, "b"

    def n_params(self):
        return sum(getattr(L, n).size for L, n in self.params())

    def save(self, path):
        np.savez(path, **{f"{i}_{n}": getattr(L, n) for i, L in enumerate(self.layers) for n in ("W", "b")})

    @classmethod
    def load(cls, path):
        net = cls()
        z = np.load(path)
        for i, L in enumerate(net.layers):
            L.W, L.b = z[f"{i}_W"], z[f"{i}_b"]
        return net


class Adam:
    def __init__(self, net, lr=2e-3, b1=0.9, b2=0.999, wd=1e-5):
        self.net, self.lr, self.b1, self.b2, self.wd, self.t = net, lr, b1, b2, wd, 0
        self.state = {}

    def step(self):
        self.t += 1
        for L, n in self.net.params():
            key = (id(L), n)
            g = getattr(L, "d" + n) + (self.wd * getattr(L, n) if n == "W" else 0)
            m, v = self.state.get(key, (np.zeros_like(g), np.zeros_like(g)))
            m = self.b1 * m + (1 - self.b1) * g
            v = self.b2 * v + (1 - self.b2) * g * g
            self.state[key] = (m, v)
            mh = m / (1 - self.b1 ** self.t)
            vh = v / (1 - self.b2 ** self.t)
            setattr(L, n, (getattr(L, n) - self.lr * mh / (np.sqrt(vh) + 1e-8)).astype(np.float32))


# ------------------------------------------------------------------ heatmap loss

def spatial_softmax(logits):
    """logits (N, h, w, 4) -> probabilities normalised over h*w for each corner."""
    N, h, w, K = logits.shape
    z = logits.reshape(N, h * w, K)
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(1, keepdims=True)).reshape(N, h, w, K)


def heatmap_loss(logits, target, mask):
    """Cross-entropy between predicted spatial distribution and a Gaussian target.
    target (N,h,w,4) sums to 1 per corner; mask (N,4) is 0 for corners outside the image."""
    p = spatial_softmax(logits)
    m = mask[:, None, None, :]
    denom = max(mask.sum(), 1.0)
    loss = -(m * target * np.log(p + 1e-9)).sum() / denom
    grad = (p - target) * m / denom
    return loss, grad.astype(np.float32), p
