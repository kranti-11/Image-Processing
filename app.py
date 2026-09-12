import streamlit as st
from PIL import Image
from rembg import remove, new_session
import numpy as np
import cv2
import pytesseract   # <-- was missing
import io
from compliance_checks import check_white_purity, flag_text

def check_white_purity(img_rgb, tolerance=5):
    """Checks the border pixels of the final image are close to pure white."""
    border_pixels = np.concatenate([
        img_rgb[0, :], img_rgb[-1, :], img_rgb[:, 0], img_rgb[:, -1]
    ])
    diff = np.abs(border_pixels.astype(int) - 255)
    avg_diff = diff.mean(axis=0)
    passed = np.all(avg_diff <= tolerance)
    return passed, avg_diff

def flag_text(img_rgb):
    text = pytesseract.image_to_string(img_rgb).strip()
    return len(text) > 0, text

st.title("Vipto — Product Image Cleaner")
st.write("Upload a product photo to remove the background and get a catalog-ready image.")

session = new_session("isnet-general-use")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGBA")
    st.image(img, caption="Original", width=300)

    with st.spinner("Removing background..."):
        foreground = remove(img, session=session)

        foreground_np = np.array(foreground)
        rgb = foreground_np[:, :, :3]
        alpha = foreground_np[:, :, 3]

        kernel = np.ones((5, 5), np.uint8)
        alpha_clean = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, kernel)
        alpha_clean = cv2.morphologyEx(alpha_clean, cv2.MORPH_CLOSE, kernel)
        alpha_feathered = cv2.GaussianBlur(alpha_clean, (5, 5), 0)

        alpha_norm = alpha_feathered[:, :, None] / 255.0
        white = np.ones_like(rgb) * 255
        result = (rgb * alpha_norm + white * (1 - alpha_norm)).astype(np.uint8)

        result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
        coords = cv2.findNonZero(thresh)
        x, y, w, h = cv2.boundingRect(coords)
        cropped = result_bgr[y:y+h, x:x+w]

        pad = int(max(w, h) * 0.1)
        padded = cv2.copyMakeBorder(cropped, pad, pad, pad, pad,
                                     cv2.BORDER_CONSTANT, value=[255, 255, 255])
        h2, w2 = padded.shape[:2]
        side = max(h2, w2)
        square = cv2.copyMakeBorder(
            padded,
            (side - h2)//2, side - h2 - (side - h2)//2,
            (side - w2)//2, side - w2 - (side - w2)//2,
            cv2.BORDER_CONSTANT, value=[255, 255, 255]
        )
        final = cv2.resize(square, (1000, 1000))
        final_rgb = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)
        white_score = check_white_purity(final_rgb)
text_flag = flag_text(final_rgb)

st.subheader("Compliance Checks")

st.write(f"White background purity: {white_score}%")

if white_score >= 90:
    st.success("✓ White background requirement passed")
else:
    st.warning("⚠ Background may require review")

if text_flag:
    st.warning("⚠ Fine text/details detected — review image")
else:
    st.success("✓ No significant text/detail issue detected")

        # --- Compliance checks (new) ---
        purity_passed, purity_diff = check_white_purity(final_rgb)
        has_text, text_found = flag_text(final_rgb)

    st.image(final_rgb, caption="Catalog-ready result", width=300)

    st.write(f"**White background check:** {'✅ Passed' if purity_passed else '⚠️ Failed'} "
              f"(avg RGB diff from pure white: {purity_diff.round(2)})")
    if has_text:
        st.write(f"**Text detected** (may be legitimate label text, flagged for review): \"{text_found[:80]}\"")
    else:
        st.write("**Text check:** No text detected")

    final_pil = Image.fromarray(final_rgb)
    buf = io.BytesIO()
    final_pil.save(buf, format="JPEG")
    st.download_button("Download result", buf.getvalue(), file_name="processed.jpg", mime="image/jpeg")