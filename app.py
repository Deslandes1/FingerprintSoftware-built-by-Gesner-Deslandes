import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import base64
import time
import pandas as pd
from datetime import datetime

st.set_page_config(
    page_title="Fingerprint Software – Gesner Deslandes",
    page_icon="🖐️",
    layout="centered"
)

# ---------- CSS for spinning fingerprint and colorful UI ----------
st.markdown(
    """
    <style>
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    .spinning-fingerprint {
        animation: spin 4s linear infinite;
        font-size: 120px;
        text-align: center;
        margin-bottom: 20px;
    }
    .stApp {
        background: linear-gradient(135deg, #1f4037, #99f2c8);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1f4037, #2c4c3b);
    }
    .stButton button {
        background-color: #ff6b6b !important;
        color: white !important;
        border-radius: 30px !important;
        font-weight: bold !important;
    }
    .stButton button:hover {
        background-color: #ff4757 !important;
        transform: scale(1.02);
    }
    h1, h2, h3, h4, p, div, span, label {
        color: #ffffff !important;
    }
    .report-box {
        background: rgba(0,0,0,0.6);
        padding: 1rem;
        border-radius: 15px;
        margin-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- Session state ----------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "fingerprints" not in st.session_state:
    st.session_state.fingerprints = {}  # name -> {'image_bytes': bytes, 'timestamp': str}
if "current_comparison_result" not in st.session_state:
    st.session_state.current_comparison_result = None

# ---------- Helper functions ----------
def bytes_to_cv2(img_bytes):
    """Convert image bytes to OpenCV grayscale image."""
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    return img

def extract_features(img):
    """Extract ORB keypoints and descriptors."""
    orb = cv2.ORB_create(nfeatures=1000)
    kp, des = orb.detectAndCompute(img, None)
    return kp, des

def match_fingerprints(img1_bytes, img2_bytes, threshold=0.3):
    """
    Compare two fingerprint images.
    Returns similarity score (0-1) and match count.
    """
    img1 = bytes_to_cv2(img1_bytes)
    img2 = bytes_to_cv2(img2_bytes)
    
    # Resize to same size for consistency
    h1, w1 = img1.shape
    h2, w2 = img2.shape
    if h1 > h2:
        img1 = cv2.resize(img1, (w2, h2))
    else:
        img2 = cv2.resize(img2, (w1, h1))
    
    kp1, des1 = extract_features(img1)
    kp2, des2 = extract_features(img2)
    
    if des1 is None or des2 is None:
        return 0.0, 0
    
    # FLANN matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    # Convert descriptors to float32 (required by FLANN)
    des1 = np.float32(des1)
    des2 = np.float32(des2)
    
    try:
        matches = flann.knnMatch(des1, des2, k=2)
    except:
        return 0.0, 0
    
    # Lowe's ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    
    total_matches = len(good)
    # Max possible matches is min(len(kp1), len(kp2))
    max_possible = min(len(kp1), len(kp2))
    if max_possible == 0:
        similarity = 0.0
    else:
        similarity = total_matches / max_possible
    
    return similarity, total_matches

def display_fingerprint(img_bytes, width=150):
    """Display fingerprint image from bytes."""
    b64 = base64.b64encode(img_bytes).decode()
    img_src = f"data:image/png;base64,{b64}"
    st.markdown(f'<img src="{img_src}" width="{width}" style="border-radius:10px; border:1px solid #ff6b6b;">', unsafe_allow_html=True)

# ---------- Login page ----------
def login():
    st.markdown('<div class="spinning-fingerprint">🖐️</div>', unsafe_allow_html=True)
    st.title("Fingerprint Software")
    st.markdown("Built by **Gesner Deslandes**")
    st.markdown("---")
    st.markdown("### 🔐 Login")
    password = st.text_input("Enter password", type="password")
    if st.button("Unlock"):
        if password == "20082010":
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password. Try again.")

# ---------- Main app ----------
def main_app():
    st.title("🖐️ Fingerprint Software")
    st.markdown("Developed by **Gesner Deslandes** – Secure Fingerprint Analysis")
    st.markdown("---")
    
    # Sidebar with logout and storage info
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/1730/1730838.png", width=80)
        st.markdown("## 🧬 Fingerprint DB")
        st.markdown(f"**Stored fingerprints:** {len(st.session_state.fingerprints)}")
        if st.session_state.fingerprints:
            for name in st.session_state.fingerprints.keys():
                st.write(f"• {name}")
        st.markdown("---")
        if st.button("🚪 Logout"):
            st.session_state.authenticated = False
            st.session_state.fingerprints = {}
            st.session_state.current_comparison_result = None
            st.rerun()
    
    # Tabs for different operations
    tab1, tab2, tab3 = st.tabs(["📥 Register Fingerprint", "🔍 Compare Fingerprints", "📊 Reports"])
    
    # ---------- TAB 1: Register ----------
    with tab1:
        st.subheader("Register a new fingerprint")
        name = st.text_input("Person / Fingerprint name", placeholder="e.g., Gesner_thumb")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Upload from file**")
            uploaded_file = st.file_uploader("Choose fingerprint image", type=["jpg", "jpeg", "png"], key="register_upload")
        with col2:
            st.markdown("**Capture from camera**")
            camera_image = st.camera_input("Take a photo of fingerprint", key="register_cam")
        
        img_bytes = None
        if uploaded_file:
            img_bytes = uploaded_file.read()
            st.image(img_bytes, caption="Uploaded fingerprint", width=150)
        elif camera_image:
            img_bytes = camera_image.read()
            st.image(img_bytes, caption="Captured fingerprint", width=150)
        
        if st.button("💾 Save Fingerprint", use_container_width=True):
            if name and img_bytes:
                if name in st.session_state.fingerprints:
                    st.warning(f"Fingerprint '{name}' already exists. Use a different name or delete old one.")
                else:
                    st.session_state.fingerprints[name] = {
                        "image_bytes": img_bytes,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    st.success(f"Fingerprint '{name}' saved successfully!")
            else:
                st.error("Please provide a name and capture/upload a fingerprint image.")
    
    # ---------- TAB 2: Compare ----------
    with tab2:
        st.subheader("Compare fingerprints")
        
        if len(st.session_state.fingerprints) == 0:
            st.info("No fingerprints stored yet. Please register some fingerprints first.")
        else:
            # Select stored fingerprint
            stored_names = list(st.session_state.fingerprints.keys())
            selected_stored = st.selectbox("Select stored fingerprint", stored_names)
            
            st.markdown("---")
            st.markdown("**Provide the fingerprint to compare against the stored one**")
            
            col1, col2 = st.columns(2)
            with col1:
                uploaded_compare = st.file_uploader("Upload fingerprint image", type=["jpg", "jpeg", "png"], key="compare_upload")
            with col2:
                camera_compare = st.camera_input("Capture fingerprint", key="compare_cam")
            
            compare_img_bytes = None
            if uploaded_compare:
                compare_img_bytes = uploaded_compare.read()
                st.image(compare_img_bytes, caption="Uploaded to compare", width=150)
            elif camera_compare:
                compare_img_bytes = camera_compare.read()
                st.image(compare_img_bytes, caption="Captured to compare", width=150)
            
            if st.button("🔍 Compare Now", use_container_width=True):
                if compare_img_bytes is None:
                    st.error("Please provide a fingerprint image to compare.")
                else:
                    stored_bytes = st.session_state.fingerprints[selected_stored]["image_bytes"]
                    with st.spinner("Matching fingerprints..."):
                        similarity, matches = match_fingerprints(stored_bytes, compare_img_bytes)
                    
                    # Store result in session for report
                    st.session_state.current_comparison_result = {
                        "stored_name": selected_stored,
                        "similarity": similarity,
                        "matches": matches,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "compared_img_bytes": compare_img_bytes
                    }
                    
                    # Display result
                    st.markdown("### 📊 Comparison Result")
                    st.markdown(f"**Stored fingerprint:** {selected_stored}")
                    st.markdown(f"**Match score:** {similarity:.2%}")
                    st.markdown(f"**Number of matching features:** {matches}")
                    
                    if similarity >= 0.25:   # threshold for positive identification
                        st.success("✅ **IDENTICAL** – The fingerprints match!")
                    else:
                        st.error("❌ **NOT IDENTICAL** – The fingerprints do not match.")
                    
                    st.info("You can download a full report in the **Reports** tab.")
    
    # ---------- TAB 3: Reports ----------
    with tab3:
        st.subheader("Fingerprint Comparison Reports")
        
        if st.session_state.current_comparison_result is None:
            st.info("No comparison performed yet. Go to the 'Compare Fingerprints' tab to run a comparison.")
        else:
            res = st.session_state.current_comparison_result
            stored_name = res["stored_name"]
            similarity = res["similarity"]
            matches = res["matches"]
            timestamp = res["timestamp"]
            
            st.markdown("### Latest Comparison")
            st.markdown(f"**Stored fingerprint:** {stored_name}")
            st.markdown(f"**Comparison performed:** {timestamp}")
            st.markdown(f"**Match score:** {similarity:.2%}")
            st.markdown(f"**Matching features:** {matches}")
            
            if similarity >= 0.25:
                verdict = "IDENTICAL"
                verdict_color = "green"
            else:
                verdict = "NOT IDENTICAL"
                verdict_color = "red"
            st.markdown(f"**Verdict:** <span style='color:{verdict_color}; font-weight:bold;'>{verdict}</span>", unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("### Download Report")
            report_text = f"""
FINGERPRINT COMPARISON REPORT
==============================
Software: Fingerprint Software by Gesner Deslandes
Date & Time: {timestamp}

Stored Fingerprint: {stored_name}
Compared Fingerprint: (image provided at {timestamp})

Match Score: {similarity:.2%} ({similarity*100:.2f}%)
Number of Matching Features: {matches}

Verdict: {verdict}

--- Technical Details ---
The comparison used ORB feature detection and FLANN matcher with Lowe's ratio test.
Threshold for positive identification: >= 25% similarity.

Generated by Fingerprint Software – Gesner Deslandes
            """
            st.download_button(
                label="📥 Download Report (TXT)",
                data=report_text,
                file_name=f"fingerprint_report_{stored_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
            
            # Option to compare multiple stored against the same new fingerprint
            st.markdown("---")
            st.markdown("### Compare stored fingerprint against multiple stored")
            if st.button("🔄 Run batch comparison against all stored fingerprints", use_container_width=True):
                if "compared_img_bytes" in res and res["compared_img_bytes"]:
                    comp_bytes = res["compared_img_bytes"]
                    results = []
                    for name, data in st.session_state.fingerprints.items():
                        sim, mat = match_fingerprints(data["image_bytes"], comp_bytes)
                        results.append({"Name": name, "Similarity": f"{sim:.2%}", "Matches": mat, "Verdict": "IDENTICAL" if sim >= 0.25 else "NOT IDENTICAL"})
                    df = pd.DataFrame(results)
                    st.dataframe(df)
                    csv = df.to_csv(index=False).encode()
                    st.download_button("📊 Download Batch Report (CSV)", data=csv, file_name="batch_comparison.csv", mime="text/csv")
                else:
                    st.warning("No comparison image available. Perform a comparison first.")

# ---------- Main flow ----------
if not st.session_state.authenticated:
    login()
else:
    main_app()
