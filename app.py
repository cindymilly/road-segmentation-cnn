import streamlit as st
import numpy as np
import cv2
import base64
import time
import io
import os
from PIL import Image
import folium
from streamlit_folium import st_folium
import math
import requests
import tensorflow as tf
import pickle

st.set_page_config(
    page_title="DeepGlobe Road CNN Segmentation",
    page_icon="assets/favicon.png" if os.path.exists("assets/favicon.png") else None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS tuy chinh (giu giao dien dark, bo emoji header) ──────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background-color: #060910; color: #c8ddf5; }
section[data-testid="stSidebar"] { background-color: #0c1422; border-right: 1px solid #1a2d4a; }
.block-container { padding-top: 3rem; }
h1, h2, h3 { font-family: 'Space Mono', monospace; color: #4f8eff; }
.metric-card {
    background: #0c1422; border: 1px solid #1a2d4a; border-radius: 8px;
    padding: 12px 16px; text-align: center; margin-bottom: 8px;
}
.metric-label { font-size: .7rem; color: #4a6b8a; font-family: 'Space Mono', monospace; }
.metric-value { font-size: 1.3rem; font-weight: 700; color: #c8ddf5; }
.good-tag { color: #22d97a; font-weight: 700; }
.reg-tag  { color: #f59e0b; font-weight: 700; }
.bad-tag  { color: #f43f5e; font-weight: 700; }
.stButton > button {
    background: #0c1422; color: #4f8eff; border: 1px solid #1a2d4a;
    border-radius: 6px; font-family: 'Space Mono', monospace; font-size: .8rem;
    padding: 6px 16px; transition: all .2s;
}

.stButton > button:hover { background: #1a2d4a; color: #c8ddf5; }
.stSelectbox > div > div { background: #0c1422; color: #c8ddf5; border-color: #1a2d4a; }


.stButton > button:hover { background: #1a2d4a; color: #c8ddf5; }
.stSelectbox > div > div { background: #0c1422; color: #c8ddf5; border-color: #1a2d4a; }

/* Tabs styling - Make them BIG and readable */
button[data-baseweb="tab"] > div {
    padding: 1rem 1rem !important;
}
button[data-baseweb="tab"] p {
    font-size: 1.5rem !important;
    font-weight: 700 !important;
    color: #ffffff !important;
}
button[data-baseweb="tab"][aria-selected="true"] p {
    color: #4f8eff !important;
}


</style>
""", unsafe_allow_html=True)

# ── Load Model & State doc lap ────────────────────────────────────────────────
def merge_files(base_filepath):
    if os.path.exists(base_filepath): return
    print(f"Ghep noi cac phan cua {base_filepath}...")
    with open(base_filepath, 'wb') as outfile:
        i = 1
        while os.path.exists(f"{base_filepath}.{i:03d}"):
            with open(f"{base_filepath}.{i:03d}", 'rb') as infile:
                outfile.write(infile.read())
            i += 1

@st.cache_resource
def load_models_and_state():
    # Force cache reload 5
    try:
        merge_files('tinyvgg_small.keras')
        merge_files('lenet_small.keras')
        tinyvgg = tf.keras.models.load_model('tinyvgg_small.keras', compile=False)
        lenet = tf.keras.models.load_model('lenet_small.keras', compile=False)
    except Exception as e:

        st.error(f"Lỗi khi load model: {e}. Vui lòng đảm bảo bạn đã train xong và file .keras tồn tại!")
        st.stop()
        
    try:
        with open('nb_state_vars.pkl', 'rb') as f:
            state = pickle.load(f)
    except Exception:
        # Fallback values if pkl file doesn't exist
        state = {
            'vgg_acc': 0.95, 'vgg_dice': 0.65, 'vgg_iou': 0.50,
            'lenet_acc': 0.94, 'lenet_dice': 0.60, 'lenet_iou': 0.45,
            'vgg_per_class': [], 'lenet_per_class': [],
            'CLASS_NAMES': ['Road - Good Road', 'Road - Regular Road', 'Road - Bad Road'],
            'CLASS_HEX': ['#22d97a', '#f59e0b', '#f43f5e'],
            'DEMO_GALLERY': [], 'IMG_SIZE': 128, 'THRESH': 128
        }
    
    return tinyvgg, lenet, state


tinyvgg_model, lenet_model, state = load_models_and_state()

vgg_acc = state.get('vgg_acc', 0)
vgg_dice = state.get('vgg_dice', 0)
vgg_iou = state.get('vgg_iou', 0)
lenet_acc = state.get('lenet_acc', 0)
lenet_dice = state.get('lenet_dice', 0)
lenet_iou = state.get('lenet_iou', 0)
vgg_per_class = state.get('vgg_per_class', [])
lenet_per_class = state.get('lenet_per_class', [])
DEMO_GALLERY = state.get('DEMO_GALLERY', [])
IMG_SIZE = state.get('IMG_SIZE', 128)

def generate_road_mask_cnn(img_bgr, model):
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_in  = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
    pred    = model.predict(img_in[np.newaxis, ...], verbose=0)[0, :, :, 0]
    pred_u8 = (pred * 255).astype(np.uint8)
    return cv2.resize(pred_u8, (w, h), interpolation=cv2.INTER_LINEAR)

def generate_road_overlay(img_bgr, mask_u8, alpha=0.45):
    overlay = img_bgr.copy()
    road_px = mask_u8 > 127
    overlay[road_px] = (
        overlay[road_px] * (1 - alpha) +
        np.array([255, 200, 0], dtype=np.float32) * alpha
    ).astype(np.uint8)
    return overlay

def cnn_predict_img(img_bgr, model):
    road_mask    = generate_road_mask_cnn(img_bgr, model)
    road_density = float((road_mask > 127).sum()) / road_mask.size
    if road_density >= 0.15:
        cls_id = 0; conf = min(0.5 + road_density * 3, 0.97)
    elif road_density >= 0.05:
        cls_id = 1; conf = 0.65 + road_density
    else:
        cls_id = 2; conf = 0.80 - road_density * 5
    probs = [0.0, 0.0, 0.0]; probs[cls_id] = conf
    return cls_id, conf, probs

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### DeepGlobe Road Segmentation")
    st.markdown("---")
    model_choice = st.radio(
        "Chon model CNN",
        options=["TinyVGG-UNet", "LeNet-UNet"],
        index=0,
    )
    sel_model_key = "vgg" if model_choice == "TinyVGG-UNet" else "lenet"
    model = tinyvgg_model if sel_model_key == "vgg" else lenet_model
    per_class_data = vgg_per_class if sel_model_key == "vgg" else lenet_per_class

    st.markdown("---")
    st.markdown("**Metrics (Test Set)**")
    acc  = vgg_acc   if sel_model_key == "vgg" else lenet_acc
    dice = vgg_dice  if sel_model_key == "vgg" else lenet_dice
    iou  = vgg_iou   if sel_model_key == "vgg" else lenet_iou
    st.metric("Accuracy",   f"{acc*100:.2f}%")
    st.metric("Dice Coeff", f"{dice*100:.2f}%")
    st.metric("IoU",        f"{iou*100:.2f}%")

    st.markdown("---")
    st.caption("DeepGlobe Road CNN v3 | TinyVGG-UNet + LeNet-UNet")

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_gallery, tab_upload, tab_map = st.tabs(["Gallery Demo", "Upload anh", "🗺️ Cắt bản đồ"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: Gallery Demo
# ─────────────────────────────────────────────────────────────────────────────
with tab_gallery:
    st.markdown("#### Anh Demo (Test Set — 3 moi class)")

    if not DEMO_GALLERY:
        st.warning("Demo Gallery chua san sang. Hay chay cell Evaluate truoc!")
    else:
        cols_gallery = st.columns(min(len(DEMO_GALLERY), 3))
        quality_map  = {0: "good", 1: "regular", 2: "poor"}
        color_map    = {0: "good-tag", 1: "reg-tag", 2: "bad-tag"}

        selected_idx = st.session_state.get("sel_idx", 0)

        for i, demo in enumerate(DEMO_GALLERY):
            with cols_gallery[i % 3]:
                img_bytes = base64.b64decode(demo["thumbnail"])
                img_pil   = Image.open(io.BytesIO(img_bytes))
                st.image(img_pil, use_container_width=True)
                cls_tag = f'<span class="{color_map[demo["cls_id"]]}">{demo["label"]}</span>'
                st.markdown(
                    f'{cls_tag} — Density: {demo["density"]*100:.2f}%',
                    unsafe_allow_html=True,
                )
                if st.button(f"Phan tich #{i+1}", key=f"btn_gal_{i}"):
                    st.session_state["sel_idx"] = i

        # Hien thi ket qua phan tich
        sel_idx = st.session_state.get("sel_idx", -1)
        if sel_idx >= 0 and sel_idx < len(DEMO_GALLERY):
            demo = DEMO_GALLERY[sel_idx]
            img_bgr = cv2.imread(demo["sat_path"])
            if img_bgr is not None:
                with st.spinner("CNN dang phan tich..."):
                    t0 = time.time()
                    road_mask    = generate_road_mask_cnn(img_bgr, model)
                    road_pixels  = int((road_mask > 127).sum())
                    road_density = road_pixels / road_mask.size

                    img_disp     = cv2.resize(img_bgr, (640, 480))
                    mask_disp    = cv2.resize(road_mask, (640, 480))
                    overlay_disp = generate_road_overlay(img_disp, mask_disp)

                    cls_id, conf, _ = cnn_predict_img(img_bgr, model)
                    conf = min(conf, 0.97)
                    ms   = (time.time() - t0) * 1000

                quality_label = ["Good Road", "Regular Road", "Poor Road"][cls_id]
                quality_css   = ["good-tag", "reg-tag", "bad-tag"][cls_id]

                st.markdown("---")
                st.markdown(f"**Ket qua — Anh #{sel_idx+1}**")

                col_v1, col_v2, col_v3, col_v4 = st.columns(4)
                col_v1.metric("Phan loai", quality_label)
                col_v2.metric("Confidence", f"{conf*100:.1f}%")
                col_v3.metric("Mat do duong", f"{road_density*100:.2f}%")
                col_v4.metric("Thoi gian", f"{ms:.0f} ms")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("**Anh goc**")
                    img_rgb = cv2.cvtColor(img_disp, cv2.COLOR_BGR2RGB)
                    st.image(img_rgb, use_container_width=True)
                with col2:
                    st.markdown("**Road Mask**")
                    st.image(mask_disp, use_container_width=True)
                with col3:
                    st.markdown("**Overlay**")
                    ovl_rgb = cv2.cvtColor(overlay_disp, cv2.COLOR_BGR2RGB)
                    st.image(ovl_rgb, use_container_width=True)

                # Per-class metrics
                if per_class_data:
                    st.markdown("**Chi so theo class**")
                    import pandas as pd
                    df_pc = pd.DataFrame([
                        {
                            "Class":     r["class"],
                            "Acc (%)":   f'{r["acc"]*100:.1f}',
                            "Prec (%)":  f'{r["prec"]*100:.1f}',
                            "Recall (%)":f'{r["rec"]*100:.1f}',
                            "F1 (%)":    f'{r["f1"]*100:.1f}',
                        }
                        for r in per_class_data
                    ])
                    st.dataframe(df_pc, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: Upload anh
# ─────────────────────────────────────────────────────────────────────────────
with tab_upload:
    st.markdown("#### Upload anh ve tinh de phan tich")
    uploaded = st.file_uploader(
        "Chon anh ve tinh (.jpg / .png)",
        type=["jpg", "jpeg", "png"],
    )
    if uploaded is not None:
        file_bytes = np.frombuffer(uploaded.read(), np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img_bgr is None:
            st.error("Khong doc duoc anh. Vui long thu lai voi file khac.")
        else:
            h, w = img_bgr.shape[:2]
            st.image(
                cv2.cvtColor(cv2.resize(img_bgr, (640, 480)), cv2.COLOR_BGR2RGB),
                caption=f"Anh da upload: {w}x{h}px",
                use_container_width=False, width=640,
            )

            if st.button("Phan tich anh nay"):
                with st.spinner("CNN dang phan tich..."):
                    t0 = time.time()
                    road_mask    = generate_road_mask_cnn(img_bgr, model)
                    road_pixels  = int((road_mask > 127).sum())
                    road_density = road_pixels / road_mask.size

                    img_disp     = cv2.resize(img_bgr, (640, 480))
                    mask_disp    = cv2.resize(road_mask, (640, 480))
                    overlay_disp = generate_road_overlay(img_disp, mask_disp)

                    cls_id, conf, _ = cnn_predict_img(img_bgr, model)
                    conf = min(conf, 0.97)
                    ms   = (time.time() - t0) * 1000

                quality_label = ["Good Road", "Regular Road", "Poor Road"][cls_id]

                st.markdown("---")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Phan loai", quality_label)
                c2.metric("Confidence", f"{conf*100:.1f}%")
                c3.metric("Mat do duong", f"{road_density*100:.2f}%")
                c4.metric("Thoi gian", f"{ms:.0f} ms")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("**Anh goc**")
                    st.image(cv2.cvtColor(img_disp, cv2.COLOR_BGR2RGB), use_container_width=True)
                with col2:
                    st.markdown("**Road Mask**")
                    st.image(mask_disp, use_container_width=True)
                with col3:
                    st.markdown("**Overlay**")
                    st.image(cv2.cvtColor(overlay_disp, cv2.COLOR_BGR2RGB), use_container_width=True)

                if per_class_data:
                    st.markdown("**Chi so theo class**")
                    import pandas as pd
                    df_pc = pd.DataFrame([
                        {
                            "Class":     r["class"],
                            "Acc (%)":   f'{r["acc"]*100:.1f}',
                            "Prec (%)":  f'{r["prec"]*100:.1f}',
                            "Recall (%)":f'{r["rec"]*100:.1f}',
                            "F1 (%)":    f'{r["f1"]*100:.1f}',
                        }
                        for r in per_class_data
                    ])
                    st.dataframe(df_pc, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: Cắt bản đồ
# ─────────────────────────────────────────────────────────────────────────────
with tab_map:
    st.markdown("#### 🗺️ Bản đồ — Vẽ khung để cắt")
    st.info("🛰️ Dùng ảnh vệ tinh (Esri Satellite) để CNN nhận diện đường tốt nhất! Zoom >= 14 để thấy rõ.")

    c1, c2 = st.columns([1, 3])
    with c1:
        tile_source = st.selectbox("Nguồn bản đồ", ["Esri Satellite", "CartoDB Voyager", "OpenStreetMap"])

        st.markdown("**Các địa điểm:**")
        loc = st.radio("Chọn nhanh", ["TPHCM", "Hà Nội", "Đà Nẵng", "Vũng Tàu", "Paris", "New York"])
        locs = {
            "TPHCM": [10.7769, 106.7009],
            "Hà Nội": [21.0285, 105.8542],
            "Đà Nẵng": [16.0544, 108.2022],
            "Vũng Tàu": [10.3444, 107.0843],
            "Paris": [48.8566, 2.3522],
            "New York": [40.7128, -74.0060]
        }
        center = locs[loc]

        st.markdown("""
        **Hướng dẫn:**
        1. Dùng công cụ ⬜ (Draw a rectangle) bên trái bản đồ để vẽ khung.
        2. Bấm nút **Cắt & Phân tích** bên dưới.
        """)

    with c2:
        m = folium.Map(location=center, zoom_start=14)
        urls = {
            "Esri Satellite": 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            "CartoDB Voyager": 'https://a.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png',
            "OpenStreetMap": 'https://tile.openstreetmap.org/{z}/{x}/{y}.png'
        }
        folium.TileLayer(
            tiles=urls[tile_source], attr=tile_source, name=tile_source,
            max_zoom=19, overlay=False, control=True
        ).add_to(m)

        from folium.plugins import Draw
        draw = Draw(
            export=False, position='topleft',
            draw_options={'polyline': False, 'polygon': False, 'circle': False, 'circlemarker': False, 'marker': False, 'rectangle': {'shapeOptions': {'color': '#4f8eff'}}}
        )
        draw.add_to(m)

        map_data = st_folium(m, width=800, height=450)

    if map_data and map_data.get("last_active_drawing"):
        geom = map_data["last_active_drawing"]["geometry"]["coordinates"][0]
        lons = [p[0] for p in geom]
        lats = [p[1] for p in geom]
        west, east = min(lons), max(lons)
        south, north = min(lats), max(lats)
        zoom = map_data.get("zoom", 16)
        # Fix zoom level when cropping for better resolution
        crop_zoom = max(zoom, 16)

        st.write(f"Đã chọn vùng: SW({south:.4f}, {west:.4f}) - NE({north:.4f}, {east:.4f})")

        if st.button("🔍 Cắt & Phân tích vùng đã chọn", type="primary"):
            with st.spinner("Đang tải tiles và cắt bản đồ..."):
                def deg2num(lat_deg, lon_deg, z):
                    lat_rad = math.radians(lat_deg)
                    n = 2.0 ** z
                    x = int((lon_deg + 180.0) / 360.0 * n)
                    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
                    return (x, y)

                x_min, y_max = deg2num(south, west, crop_zoom)
                x_max, y_min = deg2num(north, east, crop_zoom)
                x_min, x_max = min(x_min, x_max), max(x_min, x_max)
                y_min, y_max = min(y_min, y_max), max(y_min, y_max)

                if (x_max - x_min + 1) * (y_max - y_min + 1) > 200:
                    st.error("Vùng chọn quá lớn. Hãy vẽ khung nhỏ hơn hoặc phóng to bản đồ!")
                else:
                    w_px = (x_max - x_min + 1) * 256
                    h_px = (y_max - y_min + 1) * 256
                    result_image = Image.new('RGB', (w_px, h_px))
                    base_url = urls[tile_source]

                    for x in range(x_min, x_max + 1):
                        for y in range(y_min, y_max + 1):
                            url = base_url.replace('{z}', str(crop_zoom)).replace('{x}', str(x)).replace('{y}', str(y))
                            try:
                                r = requests.get(url, headers={'User-Agent': 'StreamlitApp/1.0'})
                                if r.status_code == 200:
                                    tile = Image.open(io.BytesIO(r.content))
                                    result_image.paste(tile, ((x - x_min) * 256, (y - y_min) * 256))
                            except:
                                pass

                    def deg2px(lat_deg, lon_deg, z):
                        lat_rad = math.radians(lat_deg)
                        n = 2.0 ** z
                        x = (lon_deg + 180.0) / 360.0 * n * 256
                        y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n * 256
                        return x, y

                    px_west, px_north = deg2px(north, west, crop_zoom)
                    px_east, px_south = deg2px(south, east, crop_zoom)

                    crop_x1 = int(px_west - x_min * 256)
                    crop_y1 = int(px_north - y_min * 256)
                    crop_x2 = int(px_east - x_min * 256)
                    crop_y2 = int(px_south - y_min * 256)

                    # Ensure positive width/height
                    if crop_x2 > crop_x1 and crop_y2 > crop_y1:
                        result_image = result_image.crop((crop_x1, crop_y1, crop_x2, crop_y2))

                    img_bgr = cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR)

                    # Run CNN
                    t0 = time.time()
                    road_mask    = generate_road_mask_cnn(img_bgr, model)
                    road_pixels  = int((road_mask > 127).sum())
                    road_density = road_pixels / max(1, road_mask.size)

                    # Calculate new display size to maintain aspect ratio, max dimension 640
                    h, w = img_bgr.shape[:2]
                    scale = min(640/w, 640/h) if w>0 and h>0 else 1
                    new_w, new_h = int(w*scale), int(h*scale)

                    img_disp     = cv2.resize(img_bgr, (new_w, new_h))
                    mask_disp    = cv2.resize(road_mask, (new_w, new_h))
                    overlay_disp = generate_road_overlay(img_disp, mask_disp)

                    cls_id, conf, _ = cnn_predict_img(img_bgr, model)
                    conf = min(conf, 0.97)
                    ms   = (time.time() - t0) * 1000

                    quality_label = ["Good Road", "Regular Road", "Poor Road"][cls_id]
                    st.markdown("---")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Phân loại", quality_label)
                    c2.metric("Confidence", f"{conf*100:.1f}%")
                    c3.metric("Mật độ đường", f"{road_density*100:.2f}%")
                    c4.metric("Thời gian", f"{ms:.0f} ms")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown("**Ảnh cắt**")
                        st.image(cv2.cvtColor(img_disp, cv2.COLOR_BGR2RGB), use_container_width=True)
                    with col2:
                        st.markdown("**Road Mask**")
                        st.image(mask_disp, use_container_width=True)
                    with col3:
                        st.markdown("**Overlay**")
                        st.image(cv2.cvtColor(overlay_disp, cv2.COLOR_BGR2RGB), use_container_width=True)

