import streamlit as st
import cv2
import numpy as np

import src.enhancement as enhancement
import src.detection as detection
from src.classifier import AstroClassifier
from src.solar_classifier import SolarSystemClassifier


PLANET_CONF_THRESHOLD = 0.70
SOLAR_CONF_THRESHOLD  = 0.90
MOON_CONF_THRESHOLD   = 0.90


def estimate_moon_phase(gray_img):
    _, mask = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    moon_pixels = mask > 0

    if np.sum(moon_pixels) < 100:
        return "No clear moon detected"

    ys, xs = np.where(moon_pixels)
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()

    h = y_max - y_min + 1
    w = x_max - x_min + 1

    bounding_area = h * w
    lit_area = np.sum(moon_pixels[y_min:y_max+1, x_min:x_max+1])

    fill_fraction = lit_area / bounding_area

    if fill_fraction > 0.85:
        phase = "Full Moon"
    elif fill_fraction > 0.65:
        phase = "Waxing / Waning Gibbous"
    elif fill_fraction > 0.45:
        phase = "First Quarter / Last Quarter"
    elif fill_fraction > 0.25:
        phase = "Waxing / Waning Crescent"
    else:
        phase = "Very Thin Crescent / New Moon"

    return phase


# Model Loading
@st.cache_resource
def load_models():
    solar_clf = SolarSystemClassifier("models/solar_system_classifier.pth")
    astro_clf = AstroClassifier("models/astro_classifier_spacenet.pth")
    return solar_clf, astro_clf


solar_clf, astro_clf = load_models()


# UI Layout
st.set_page_config(page_title="AstroLens", layout="wide")
st.title("AstroLens: Explore your space images")

moon_tab, obj_tab = st.tabs(["Moon Tools", "Celestial Object Identifier"])

uploaded = st.file_uploader("Upload a space image", type=["jpg", "jpeg", "png"])

if not uploaded:
    with moon_tab:
        st.info("Upload a Moon photo to use enhancement, phase estimation, and crater detection.")
    with obj_tab:
        st.info("Upload any space image (planet, galaxy, nebula, etc.) to classify it.")
else:
    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    astro_label, astro_conf, _ = astro_clf.predict(img_rgb)
    astro_label_clean = astro_label.strip()
    astro_label_lower = astro_label_clean.lower()

    is_planet_like = (astro_label_lower == "planet") and (astro_conf >= PLANET_CONF_THRESHOLD)

    solar_label, solar_conf, _ = solar_clf.predict(img_rgb)
    solar_label_clean = solar_label.strip()
    solar_label_lower = solar_label_clean.lower()

    is_moon = (
        is_planet_like
        and solar_label_lower == "moon"
        and solar_conf >= MOON_CONF_THRESHOLD
    )

    nonzero_pixels = gray[gray > 0]
    mean_brightness = int(np.mean(nonzero_pixels)) if nonzero_pixels.size > 0 else 0
    phase = estimate_moon_phase(gray)

    circles = None
    enhanced_gray = None
    enhanced_rgb = None

# Moon Tools Tab
    with moon_tab:
        st.subheader("Moon Tools")
        st.image(img_rgb, use_container_width=True, caption="Uploaded image")

        if not is_moon:
            st.warning(
                "This image does not look like the **Moon** with high confidence.\n\n"
                f"SpaceNet prediction: **{astro_label_clean}** (conf: {astro_conf:.2f})\n"
                f"Solar System prediction: **{solar_label_clean}** (conf: {solar_conf:.2f})\n\n"
                "Moon tools are only enabled when the image is very likely to be the Moon."
            )
        else:
            left, right = st.columns([2, 1])

            with left:
                st.subheader("Original Image")
                st.image(img_rgb, use_container_width=True)

                st.markdown("### Actions")

                detection_source = st.radio(
                    "Use this image for crater detection:",
                    ["Original", "Enhanced"],
                    index=1,
                    help="Choose whether crater detection runs on the raw image or on an enhanced version."
                )

                col_buttons = st.columns(2)
                with col_buttons[0]:
                    enhance_clicked = st.button("✨ Enhance Image", key="enhance_moon")
                with col_buttons[1]:
                    detect_clicked = st.button("🕳️ Detect Craters", key="detect_craters")

                if enhance_clicked:
                    enhanced_gray = enhancement.enhance_image(img_bgr)
                    enhanced_rgb = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2RGB)

                    st.subheader("Enhanced Image")
                    st.image(enhanced_rgb, use_container_width=True)

                if detect_clicked:
                    if detection_source == "Original":
                        src_for_detection = gray
                        base_bgr = img_bgr.copy()
                        label = "Original"
                    else:
                        if enhanced_gray is None:
                            enhanced_gray = enhancement.enhance_image(img_bgr)
                            enhanced_rgb = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2RGB)

                        src_for_detection = enhanced_gray
                        base_bgr = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
                        label = "Enhanced"

                    circles = detection.detect_craters(src_for_detection)
                    overlay_bgr = detection.draw_craters(base_bgr, circles)
                    output_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

                    st.subheader(f"Crater Detection ({label})")
                    st.image(output_rgb, use_container_width=True)

            with right:
                st.subheader("Moon Info")
                st.write(f"**Estimated Phase:** {phase}")
                st.write(f"**Average Brightness (moon area):** {mean_brightness}")
                st.write(f"**Image Resolution:** {img_rgb.shape[1]} × {img_rgb.shape[0]} px")

                if circles is not None and len(circles) > 0:
                    crater_count = len(circles[0])
                    st.write(f"**Detected Craters:** {crater_count}")
                else:
                    st.write("**Detected Craters:** Not detected yet (click *Detect Craters*).")

                st.caption(
                    "Moon tools work best on high-contrast images of the Moon where it "
                    "appears relatively large in the frame."
                )

# Object Identifier Tab
    with obj_tab:
        st.subheader("Uploaded Image")
        st.image(img_rgb, use_container_width=True)

        st.markdown("### Classification Result")

        if is_planet_like and solar_conf >= SOLAR_CONF_THRESHOLD:
            final_label = solar_label_clean
            final_conf = solar_conf
            final_source = "Solar System model (specific planet / Moon)"
        else:
            final_label = astro_label_clean
            final_conf = astro_conf
            final_source = "SpaceNet model (deep-sky / broad type)"

        st.write(f"**Predicted Object:** {final_label}")
        st.write(f"**Confidence:** {final_conf:.2f}")
        st.write(f"**Model Used:** {final_source}")

        st.markdown("### Basic Image Info")
        st.write(f"**Resolution:** {img_rgb.shape[1]} × {img_rgb.shape[0]} px")

        st.info(
            "AstroLens first uses a broad **SpaceNet classifier** to decide if your image "
            "looks like a planet, galaxy, nebula, etc. If it seems planetary, it then uses "
            "a dedicated **Solar System classifier** to identify which planet (or the Moon) "
            "you’re looking at."
        )