"""Web demo:  streamlit run app.py"""
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from forensics.detector import ForensicDetector

ROOT = Path(__file__).resolve().parent
SAMPLES = ROOT / "samples"

st.set_page_config(page_title="Hybrid Image Forensics", page_icon="🔍", layout="wide")

st.markdown("""
<style>
.verdict {padding: 18px 22px; border-radius: 12px; color: white; margin-bottom: 8px}
.verdict h2 {margin: 0; color: white}
.verdict p {margin: 4px 0 0 0; opacity: .9}
.v-Manipulated {background: linear-gradient(90deg,#b71c1c,#e53935)}
.v-Authentic {background: linear-gradient(90deg,#1b5e20,#43a047)}
.v-Uncertain {background: linear-gradient(90deg,#e65100,#fb8c00)}
</style>""", unsafe_allow_html=True)


@st.cache_resource
def get_detector():
    return ForensicDetector()


det = get_detector()

# ---------------------------------------------------------------- sidebar ----
st.sidebar.title("⚙️ Settings")
low, high = st.sidebar.slider("Uncertain band  (P(manipulated))", 0.0, 1.0, (0.35, 0.65), 0.01,
                              help="Below the band → Authentic, above → Manipulated, inside → Uncertain")
det.low, det.high = low, high
det.mask_threshold = st.sidebar.slider("Localization threshold", 0.1, 0.9, 0.5, 0.05)
st.sidebar.markdown(f"**Engine:** {'Hybrid CNN (trained)' if det.mode == 'cnn' else 'Heuristic (no checkpoint)'}")
if det.meta:
    v = det.meta.get("val", {})
    st.sidebar.caption(f"Model: {det.meta.get('streams')} · epoch {det.meta.get('epoch')} · "
                       f"val AUC {v.get('auc', 0):.3f} · val pixel-F1 {v.get('pixel_f1', 0):.3f}")

sample_files = sorted([p for p in SAMPLES.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
                       and "_mask" not in p.stem]) if SAMPLES.exists() else []

# ---------------------------------------------------------------- header -----
st.title("🔍 Hybrid Multi-Domain Image Forensics")
st.caption("AI-assisted manipulation detection & localization · spatial + frequency domain features fused by a dual-stream CNN")

c1, c2 = st.columns([2, 1])
with c1:
    up = st.file_uploader("Upload an image (JPEG / PNG / BMP / WEBP)", type=["jpg", "jpeg", "png", "bmp", "webp", "tif", "tiff"])
with c2:
    options = ["—"] + [p.name for p in sample_files]
    pre = st.query_params.get("sample", "—")  # e.g. http://localhost:8501/?sample=splice.png
    choice = st.selectbox("…or try a sample", options, index=options.index(pre) if pre in options else 0)

src, name = None, None
if up is not None:
    src, name = up, up.name
elif choice != "—":
    src, name = SAMPLES / choice, choice

if src is None:
    st.info("Upload an image or pick a sample to start the forensic analysis.")
    st.subheader("How it works")
    st.markdown("""
| Step | Stage | What happens |
|---|---|---|
| 1 | **Input image** | Any common image format is accepted |
| 2 | **Preprocessing** | Size limiting, colour conversion, normalisation |
| 3 | **Multi-domain feature extraction** | *Spatial:* Error Level Analysis, SRM noise residuals, local noise level, edges · *Frequency:* FFT high-pass energy & spectrum, 8×8 DCT AC energy, JPEG blockiness, wavelet (DWT-HH) noise |
| 4 | **Feature fusion + AI model** | RGB stream and forensic stream (+ fixed SRM & learnable Bayar filters) are fused at every scale with squeeze-and-excitation attention |
| 5 | **Manipulation detection** | Authentic / Manipulated / Uncertain from P(manipulated) |
| 6 | **Localization map** | U-Net decoder predicts a pixel-level map of altered regions |
""")
    st.stop()

with st.spinner("Running multi-domain forensic analysis…"):
    res = det.analyze(src)

# ---------------------------------------------------------------- verdict ----
st.markdown(f"""<div class="verdict v-{res.label}"><h2>{res.label}</h2>
<p>P(manipulated) = <b>{res.probability:.1%}</b> · flagged area = <b>{100 * res.manipulated_area:.1f}%</b> ·
analysed in {res.seconds:.2f}s · engine: {res.mode}</p></div>""", unsafe_allow_html=True)
st.progress(min(max(res.probability, 0.0), 1.0))

a, b, c = st.columns(3)
a.image(res.image, caption=f"Input · {name}", width="stretch")
b.image(res.overlay, caption="Localization heatmap", width="stretch")
c.image(res.mask, caption="Predicted manipulation mask", width="stretch", clamp=True)

gt_path = SAMPLES / f"{Path(name).stem}_mask.png"
if up is None and gt_path.exists():
    with st.expander("Ground-truth mask for this sample"):
        st.image(str(gt_path), width=320)

t1, t2, t3, t4 = st.tabs(["🧩 Spatial domain", "📈 Frequency domain", "🧠 Evidence", "⬇️ Export"])
spatial = {k: v for k, v in res.visuals.items() if k.startswith("Spatial")}
freq = {k: v for k, v in res.visuals.items() if k.startswith("Frequency")}
with t1:
    cols = st.columns(len(spatial))
    for col, (k, v) in zip(cols, spatial.items()):
        col.image(v, caption=k.split(": ")[1], width="stretch")
    st.caption("ELA re-compresses the image and shows where the error level differs; SRM residuals and the "
               "local noise level expose regions whose sensor noise does not match the rest of the photo.")
with t2:
    cols = st.columns(3)
    for k_i, (k, v) in enumerate(freq.items()):
        cols[k_i % 3].image(v, caption=k.split(": ")[1], width="stretch")
    st.caption("The FFT spectrum reveals periodic resampling / generative up-sampling peaks; DCT and blockiness maps "
               "show inconsistent JPEG compression history; the wavelet HH band gives a robust local noise estimate.")
with t3:
    df = pd.DataFrame(res.evidence)
    st.markdown("How strongly each forensic feature separates the flagged region from the rest of the image "
                "(difference in robust z-units, higher = stronger evidence).")
    st.bar_chart(df.set_index("feature")["contrast"])
    st.dataframe(df[["feature", "domain", "description", "contrast"]], hide_index=True, width="stretch")
with t4:
    js = json.dumps({"file": name, **res.summary()}, indent=2)
    st.download_button("Download JSON report", js, file_name=f"{Path(name).stem}_forensics.json", mime="application/json")
    buf = io.BytesIO()
    Image.fromarray(res.overlay).save(buf, format="PNG")
    st.download_button("Download heatmap overlay (PNG)", buf.getvalue(), file_name=f"{Path(name).stem}_overlay.png")
    buf2 = io.BytesIO()
    Image.fromarray(res.mask).save(buf2, format="PNG")
    st.download_button("Download mask (PNG)", buf2.getvalue(), file_name=f"{Path(name).stem}_mask.png")
    st.code(js, language="json")
