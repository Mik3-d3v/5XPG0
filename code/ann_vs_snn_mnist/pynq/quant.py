"""quant.py — int8 quantisation, bit-exact integer models and model.bin packing
for the PicoRV32 ANN/SNN firmware. numpy only; shared by export_model.py
(MNIST) and ../../ann_vs_snn_ecg/pynq/export_ecg.py (ECG).

Weights (both models): per layer one scale s_w = max|W| / 127,
W_q = round(W / s_w) as int8, int32 biases.

SNN (snnTorch Leaky, reset_delay=True, reset by subtraction):
  b_q = round(b / s), thr_q = round(thr / s), beta as Q16, where s is the
  scale of the layer's input current: s = s_w for spike inputs, and
  s = s_w * s_in for a first layer fed with int8 inputs (direct encoding);
  mem_t = ((beta_q16 * mem_{t-1}) >> 16) + I_t - spk_{t-1} * thr_q
  spk_t = mem_t > thr_q
  Spikes are binary, so the scales of different layers never mix.
  Direct encoding: the first layer's current I = W_q x_q + b_q is the same
  every timestep (the float model computes fc1(x) once, too).

ANN (Linear-ReLU-...-Linear):
  inputs x_q = x / s_in: uint8 (MNIST pixels, s_in = 1/255) or int8;
  acc = W_q x_q + b_q            int32, scale s_acc = s_w * s_in
  hidden: h_q = min(255, (relu(acc) * m_q + 2^23) >> 24)   uint8,
          m_q = round(s_acc / s_h * 2^24), s_h = max hidden activation / 255
          calibrated on training data (never the test set)
  output: logits = acc (int32), prediction = first argmax.

model.bin layout (little endian, mirrored in firmware/snn_layout.h):
  u32 magic, n_layers, n_in, timesteps (ANN: 0)
    magic: "SNN2" spike-bitmap inputs   "SNN3" int8 inputs (direct encoding)
           "ANN1" uint8 inputs          "ANN2" int8 inputs
  per layer: u32 n_in, n_out, p0, p1, w_off, b_off, w_stride
             SNN: p0 = beta_q16, p1 = thr_q;  ANN: p0 = m_q (0 = output), p1 = 0
             (offsets in bytes from the start of model.bin)
  weights int8 [n_in][w_stride] (input-major, each row zero-padded to a
             multiple of 16 bytes so the fast firmware can load words),
  biases int32 [n_out]
"""

import struct

import numpy as np

SNN_MAGIC = 0x324E4E53        # "SNN2"  spike inputs
SNN_DIRECT_MAGIC = 0x334E4E53 # "SNN3"  int8 inputs, constant first-layer current
ANN_MAGIC = 0x314E4E41        # "ANN1"  uint8 inputs
ANN_S8_MAGIC = 0x324E4E41     # "ANN2"  int8 inputs
HDR_WORDS = 4
LAYER_WORDS = 7
MODEL_MAX_BYTES = 0x3F000 - 0x13000   # firmware/snn_layout.h
REQ_SHIFT = 24                # ANN requantization: (acc * m_q) >> 24


# ----------------------------------------------------------------- inputs
def quantize_inputs(x, s_in):
    """float [N, F] -> int8 [N, F], symmetric, clipped to +-127."""
    return np.clip(np.rint(np.asarray(x, np.float64) / s_in), -127, 127
                   ).astype(np.int8)


def pad4(a):
    """[N, F] -> [N, 4*ceil(F/4)] zero-padded (the firmware copies words)."""
    return np.pad(a, ((0, 0), (0, -a.shape[1] % 4)))


# ----------------------------------------------------------------- SNN
def quantize_snn(sd, in_scale=None):
    """fc<i>/lif<i> pairs of a state_dict -> list of integer layers.
    in_scale: None for spike inputs, else the int8 input scale s_in of a
    direct-encoded first layer."""
    layers = []
    i = 1
    while f"fc{i}.weight" in sd:
        w = sd[f"fc{i}.weight"].double().numpy()          # [out, in]
        b = sd[f"fc{i}.bias"].double().numpy()
        lif = f"lif{i}."
        assert int(sd[lif + "reset_mechanism_val"]) == 0, "only reset-by-subtraction"
        assert float(sd[lif + "graded_spikes_factor"]) == 1.0
        beta = float(sd[lif + "beta"].clamp(0, 1))
        thr = float(sd[lif + "threshold"])
        w_scale = np.abs(w).max() / 127.0
        scale = w_scale * (in_scale if (i == 1 and in_scale) else 1.0)
        assert thr > 0, "firmware assumes a positive threshold"
        l = {
            "w": np.clip(np.rint(w.T / w_scale), -127, 127).astype(np.int8),
            "b": np.rint(b / scale).astype(np.int32),
            "p0": int(round(beta * 65536)),
            "p1": int(round(thr / scale)),
            "scale": scale, "beta": beta, "threshold": thr,
        }
        if scale != w_scale:                # direct-encoded first layer
            l["w_scale"], l["in_scale"] = w_scale, in_scale
        layers.append(l)
        i += 1
    if not layers:
        raise ValueError("no fc1/lif1 layers in checkpoint")
    return layers


def int_forward_snn(layers, spikes, direct_x=None, return_hidden=False,
                    record=None):
    """Bit-exact model of the firmware -> output spike counts [B, n_out].

    spikes  : [T, B, n_in] 0/1 input spikes, or
    direct_x: (T, int8 [B, n_in]) for a direct-encoded first layer (then
              spikes is ignored).
    return_hidden: also return the spike count of all non-input layers
              except the last, per sample (the firmware's hidden events).
    record:   optional list; per timestep a list of the spike arrays [B, n]
              of every layer is appended (for visualisation)."""
    ws = [l["w"].astype(np.float64) for l in layers]   # exact for these sums
    if direct_x is not None:
        T, xq = direct_x
        B = xq.shape[0]
        cur0 = np.rint(xq.astype(np.float64) @ ws[0]).astype(np.int64) + layers[0]["b"]
    else:
        T, B = spikes.shape[0], spikes.shape[1]
    mem = [np.zeros((B, w.shape[1]), np.int64) for w in ws]
    spk = [np.zeros_like(m) for m in mem]
    counts = np.zeros((B, ws[-1].shape[1]), np.int64)
    hidden = np.zeros(B, np.int64)
    for t in range(T):
        x = None if direct_x is not None else spikes[t].astype(np.float64)
        for k, l in enumerate(layers):
            if k == 0 and direct_x is not None:
                cur = cur0
            else:
                cur = np.rint(x @ ws[k]).astype(np.int64) + l["b"]
            m = ((l["p0"] * mem[k]) >> 16) + cur - spk[k] * l["p1"]
            spk[k] = (m > l["p1"]).astype(np.int64)
            mem[k] = m
            x = spk[k].astype(np.float64)
            if k < len(layers) - 1:
                hidden += spk[k].sum(1)
        counts += spk[-1]
        if record is not None:
            record.append([x.copy() for x in spk])
    return (counts, hidden) if return_hidden else counts


def pack_spikes(s):
    """[T, B, n_in] 0/1 -> [B, T, words] uint32 bitmaps (bit j%32 of word j/32)."""
    s = np.transpose(np.asarray(s), (1, 0, 2)).astype(np.uint8)
    words = (s.shape[2] + 31) // 32
    s = np.pad(s, ((0, 0), (0, 0), (0, 32 * words - s.shape[2])))
    return np.packbits(s, axis=-1, bitorder="little")   # bytes == LE words


# ----------------------------------------------------------------- RSNN
# Recurrent SNN over an event stream (ECG streaming model, magic "RSN1"):
#   layer 0  recurrent LIF, input u_t = [s_in_t (n_ev) | s_h_{t-1} (H)]
#            (input events followed by the layer's own spikes of the
#            previous step), weights [n_ev + H][H], no bias, one scale;
#   layer 1  readout of the hidden spike counts:  A1 = W1^T counts
#            (bins > 1: counts per time window, bin of step t = t*bins // T,
#            stored as [bins][H]; bins = p1 of layer 2)
#   layer 2  readout of the int8 RR features:     A2 = W2^T rr_q
#   logits = A1 * m1 + A2 * m2 + b    (int32; m1 = p0 of layer 1,
#            m2 = p0 of layer 2, b = biases of layer 1, scale s1 / 2^8)
RSNN_MAGIC = 0x314E5352       # "RSN1"
RSNN_M1 = 256


def quantize_rsnn(sd, s_rr, bins=1):
    """RSNN state_dict (train_ecg.RSNN) -> 3 integer layers; s_rr: input scale
    of the RR features (the shared int8 feature scale)."""
    w_in = sd["fc_in.weight"].double().numpy()          # [H, n_ev]
    w_rec = sd["fc_rec.weight"].double().numpy()        # [H, H]
    w0 = np.concatenate([w_in, w_rec], axis=1)          # [H, n_ev + H]
    assert int(sd["lif.reset_mechanism_val"]) == 0
    beta = float(sd["lif.beta"].clamp(0, 1))
    thr = float(sd["lif.threshold"])
    s0 = np.abs(w0).max() / 127.0
    H = w0.shape[0]
    head_w = sd["head.weight"].double().numpy()         # [C, H + 3]
    head_b = sd["head.bias"].double().numpy()
    wc, wr = head_w[:, :bins * H], head_w[:, bins * H:]
    s1 = np.abs(wc).max() / 127.0
    s2 = np.abs(wr).max() / 127.0
    s_l = s1 / RSNN_M1                                  # logit scale
    m2 = int(round(s2 * s_rr / s_l))
    q = lambda w, s: np.clip(np.rint(w.T / s), -127, 127).astype(np.int8)
    C = wc.shape[0]
    return [
        {"w": q(w0, s0), "b": np.zeros(H, np.int32),
         "p0": int(round(beta * 65536)), "p1": int(round(thr / s0)),
         "scale": s0, "beta": beta, "threshold": thr},
        {"w": q(wc, s1), "b": np.rint(head_b / s_l).astype(np.int32),
         "p0": RSNN_M1, "p1": 0, "scale": s1, "logit_scale": s_l},
        {"w": q(wr, s2), "b": np.zeros(C, np.int32),
         "p0": m2, "p1": bins, "scale": s2, "in_scale": s_rr},
    ]


def int_forward_rsnn(layers, spikes, rr_q, record=None):
    """Bit-exact model of the firmware. spikes: [T, B, n_ev] 0/1 input events,
    rr_q: [B, 3] int8 -> (int logits [B, C], hidden spikes per sample [B]).
    record: optional list, per step the hidden spikes [B, H] are appended."""
    l0, l1, l2 = layers
    w0 = l0["w"].astype(np.float64)                     # [n_ev + H, H]
    T, B, n_ev = spikes.shape
    H = w0.shape[1]
    bins = max(l2["p1"], 1)
    mem = np.zeros((B, H), np.int64)
    spk = np.zeros((B, H), np.int64)
    counts = np.zeros((B, bins, H), np.int64)
    for t in range(T):
        u = np.concatenate([spikes[t], spk], axis=1).astype(np.float64)
        cur = np.rint(u @ w0).astype(np.int64)
        m = ((l0["p0"] * mem) >> 16) + cur - (mem > l0["p1"]) * l0["p1"]
        spk = (m > l0["p1"]).astype(np.int64)
        mem = m
        counts[:, t * bins // T] += spk
        if record is not None:
            record.append(spk.copy())
    a1 = counts.reshape(B, bins * H) @ l1["w"].astype(np.int64)
    a2 = rr_q.astype(np.int64) @ l2["w"].astype(np.int64)
    logits = a1 * l1["p0"] + a2 * l2["p0"] + l1["b"]
    assert np.abs(logits).max() < 2 ** 31, "readout overflows int32"
    return logits, counts.sum((1, 2))


# ----------------------------------------------------------------- GRU
# Integer GRU over an int8 sample stream (ECG streaming model, magic "GRU1").
# PyTorch gate order r, z, n:
#   r = sig(Wi_r x + bi_r + Wh_r h + bh_r)      z = sig(... z ...)
#   n = tanh(Wi_n x + bi_n + r * (Wh_n h + bh_n))
#   h = (1 - z) * n + z * h
# Integers: h_q int8 with scale 1/128 (h15 = h_q << 8 is Q15),
#   ah = Wh_q h_q + bh_q        (int32, scale s_wh / 128)  -> mac_layer()
#   ax = Wi_q x_q + bi_q        (int32, scale s_wi * s_x)
#   pre = (ah * MH + ax * MX) >> 16          Q8 (MH, MX: Q16 multipliers)
#   pre_n = (ax_n * MX + ((r * ah_n) >> 15) * MH) >> 16
#   r, z = SIG[idx(pre)], n = TANH[idx(pre_n)]    (Q15, 512-entry LUTs)
#   idx(p) = clamp(p >> 3, -256, 255) + 256       (1/32 steps over +-8)
#   h15' = n + ((z * (h15 - n)) >> 15);  h_q' = clamp((h15' + 128) >> 8, +-127)
# Readout as RSNN on the hidden states at the window ends (bins x H int8):
#   logits = (W1^T hcat) * m1 + (W2^T rr_q) * m2 + b
# model.bin layers: 0 Wh [H][3H] (bias bh_q, p0 = MH, p1 = byte offset of
# the LUTs), 1 Wi [1][3H] (bias bi_q, p0 = MX, p1 = bins), 2 readout of the
# hidden states (p0 = m1, biases b), 3 readout of rr (p0 = m2).
GRU_MAGIC = 0x31555247        # "GRU1"
GRU_PF = 8                    # pre-activations in Q8
GRU_LUT_N = 512


def gru_luts():
    """int16 Q15 sigmoid and tanh over (i - 256) / 32, i = 0 .. 511."""
    v = (np.arange(GRU_LUT_N) - GRU_LUT_N // 2) / 32.0
    sig = np.rint(32767 / (1 + np.exp(-v))).astype(np.int16)
    tnh = np.rint(32767 * np.tanh(v)).astype(np.int16)
    return sig, tnh


def quantize_gru(sd, s_x, bins):
    wih = sd["gru.weight_ih_l0"].double().numpy()       # [3H, 1]
    whh = sd["gru.weight_hh_l0"].double().numpy()       # [3H, H]
    bih = sd["gru.bias_ih_l0"].double().numpy()
    bhh = sd["gru.bias_hh_l0"].double().numpy()
    H = whh.shape[1]
    s_wh = np.abs(whh).max() / 127.0
    s_wi = np.abs(wih).max() / 127.0
    S_h, S_x = s_wh / 128.0, s_wi * s_x
    MH = int(round(S_h * 2 ** GRU_PF * 2 ** 16))
    MX = int(round(S_x * 2 ** GRU_PF * 2 ** 16))
    head_w = sd["head.weight"].double().numpy()         # [C, bins*H + 3]
    head_b = sd["head.bias"].double().numpy()
    wc, wr = head_w[:, :bins * H], head_w[:, bins * H:]
    s1 = np.abs(wc).max() / 127.0
    s2 = np.abs(wr).max() / 127.0
    s_l = s1 / 128.0 / RSNN_M1        # logit scale (hidden states: 1/128)
    m2 = int(round(s2 * s_x / s_l))
    q = lambda w, s: np.clip(np.rint(w.T / s), -127, 127).astype(np.int8)
    C = wc.shape[0]
    return [
        {"w": q(whh, s_wh), "b": np.rint(bhh / S_h).astype(np.int32),
         "p0": MH, "p1": 0, "w_scale": s_wh, "acc_scale": S_h},
        {"w": q(wih, s_wi), "b": np.rint(bih / S_x).astype(np.int32),
         "p0": MX, "p1": bins, "w_scale": s_wi, "acc_scale": S_x},
        {"w": q(wc, s1), "b": np.rint(head_b / s_l).astype(np.int32),
         "p0": RSNN_M1, "p1": 0, "scale": s1, "logit_scale": s_l},
        {"w": q(wr, s2), "b": np.zeros(C, np.int32),
         "p0": m2, "p1": 0, "scale": s2, "in_scale": s_x},
    ]


def gru_window_ends(T, bins):
    return [max(t for t in range(T) if t * bins // T == k) for k in range(bins)]


def int_forward_gru(layers, xq, rr_q, record=None):
    """Bit-exact model of the firmware. xq: [B, T] int8 samples, rr_q: [B, 3]
    int8 -> (int logits [B, C], nonzero hidden values per sample [B]).
    record: optional list, per step the int8 hidden state [B, H] is appended."""
    l0, l1, l2, l3 = layers
    sig, tnh = (t.astype(np.int64) for t in gru_luts())
    wh = l0["w"].astype(np.int64)                       # [H, 3H]
    wi = l1["w"].astype(np.int64)[0]                    # [3H]
    H = wh.shape[0]
    MH, MX, bins = l0["p0"], l1["p0"], l1["p1"]
    B, T = xq.shape
    h = np.zeros((B, H), np.int64)                      # h_q, int8 range
    ends = gru_window_ends(T, bins)
    hcat, nz = [], np.zeros(B, np.int64)
    idx = lambda p: np.clip(p >> 3, -GRU_LUT_N // 2, GRU_LUT_N // 2 - 1) + GRU_LUT_N // 2
    for t in range(T):
        ah = h @ wh + l0["b"]                           # [B, 3H]
        ax = xq[:, t:t + 1].astype(np.int64) * wi + l1["b"]
        pr = (ah[:, :H] * MH + ax[:, :H] * MX) >> 16
        pz = (ah[:, H:2 * H] * MH + ax[:, H:2 * H] * MX) >> 16
        r, z = sig[idx(pr)], sig[idx(pz)]
        pn = (ax[:, 2 * H:] * MX + ((r * ah[:, 2 * H:]) >> 15) * MH) >> 16
        n = tnh[idx(pn)]
        h15 = h << 8
        h15n = n + ((z * (h15 - n)) >> 15)
        h = np.clip((h15n + 128) >> 8, -127, 127)
        nz += (h != 0).sum(1)
        if record is not None:
            record.append(h.copy())
        if t in ends:
            hcat.append(h)
    hcat = np.concatenate(hcat, axis=1)                 # [B, bins * H]
    a1 = hcat @ l2["w"].astype(np.int64)
    a2 = rr_q.astype(np.int64) @ l3["w"].astype(np.int64)
    logits = a1 * l2["p0"] + a2 * l3["p0"] + l2["b"]
    assert np.abs(logits).max() < 2 ** 31, "readout overflows int32"
    return logits, nz


def pack_gru(layers):
    """model.bin of the GRU: the common layout + the two LUTs after the last
    layer (offset in layer 0's p1)."""
    blob = pack_model(layers, GRU_MAGIC, 0)
    blob += b"\0" * (-len(blob) % 4)
    lut_off = len(blob)
    sig, tnh = gru_luts()
    blob += sig.astype("<i2").tobytes() + tnh.astype("<i2").tobytes()
    # patch p1 of layer 0 (header word HDR_WORDS + 3) with the LUT offset
    hdr = bytearray(blob)
    struct.pack_into("<I", hdr, 4 * (HDR_WORDS + 3), lut_off)
    if len(hdr) > MODEL_MAX_BYTES:
        raise ValueError("GRU model does not fit the BRAM layout")
    return bytes(hdr)


# ----------------------------------------------------------------- ANN
def quantize_ann(sd, calib, s_in=1.0 / 255.0):
    """Linear layers of the ANN state_dict -> integer layers.
    calib: float [N, n_in] TRAINING inputs for the activation scales;
    s_in: scale of the integer inputs (MNIST uint8 pixels: 1/255)."""
    names = sorted({k.rsplit(".", 1)[0] for k in sd if k.endswith(".weight")},
                   key=lambda k: int(k.split(".")[-1]))
    layers = []
    x = calib
    for i, k in enumerate(names):
        w = sd[k + ".weight"].double().numpy()            # [out, in]
        b = sd[k + ".bias"].double().numpy()
        s_w = np.abs(w).max() / 127.0
        s_acc = s_w * s_in
        l = {"w": np.clip(np.rint(w.T / s_w), -127, 127).astype(np.int8),
             "b": np.rint(b / s_acc).astype(np.int32),
             "p0": 0, "p1": 0, "w_scale": s_w, "in_scale": s_in, "acc_scale": s_acc}
        x = x @ w.T + b
        if i < len(names) - 1:              # hidden: ReLU + requantize to uint8
            x = np.maximum(x, 0)
            s_h = x.max() / 255.0
            l["p0"] = int(round(s_acc / s_h * 2 ** REQ_SHIFT))
            s_in = s_acc * 2 ** REQ_SHIFT / l["p0"]        # exact scale of h_q
            l["act_scale"] = s_in
        layers.append(l)
    return layers


def int_forward_ann(layers, X, return_hidden=False):
    """Bit-exact model of the firmware. X: [B, n_in] uint8 or int8 -> int logits.
    return_hidden: also return the number of nonzero hidden activations."""
    x = X.astype(np.float64)
    nz = np.zeros(len(X), np.int64)
    for l in layers:
        acc = np.rint(x @ l["w"].astype(np.float64)).astype(np.int64) + l["b"]
        if not l["p0"]:
            return (acc, nz) if return_hidden else acc
        h = np.maximum(acc, 0)
        x = np.minimum((h * l["p0"] + (1 << (REQ_SHIFT - 1))) >> REQ_SHIFT,
                       255).astype(np.float64)
        nz += (x != 0).sum(1)
    raise ValueError("ANN without output layer")


# ----------------------------------------------------------------- common
def pack_model(layers, magic, timesteps):
    body = b""
    descs = []
    off = 4 * (HDR_WORDS + LAYER_WORDS * len(layers))
    for l in layers:
        n_in, n_out = l["w"].shape
        stride = -(-n_out // 16) * 16
        w_off = off + len(body)
        body += np.pad(l["w"], ((0, 0), (0, stride - n_out))).tobytes()
        body += b"\0" * (-len(body) % 4)
        b_off = off + len(body)
        body += l["b"].astype("<i4").tobytes()
        descs += [n_in, n_out, l["p0"], l["p1"], w_off, b_off, stride]
    hdr = [magic, len(layers), layers[0]["w"].shape[0], timesteps] + descs
    blob = struct.pack(f"<{len(hdr)}I", *hdr) + body
    if len(blob) > MODEL_MAX_BYTES:
        raise ValueError(f"model is {len(blob)} B, BRAM room is {MODEL_MAX_BYTES} B")
    return blob


def layer_meta(layers):
    """JSON-friendly per-layer parameters (no weight arrays)."""
    return [{k: v for k, v in l.items() if k not in ("w", "b")} |
            {"n_in": int(l["w"].shape[0]), "n_out": int(l["w"].shape[1])}
            for l in layers]
