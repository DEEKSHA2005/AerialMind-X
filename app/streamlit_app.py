"""AerialMind-X Streamlit application."""


import streamlit as st

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="AerialMind-X | Autonomous Drone Agent",
    page_icon="assets/drone.png",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# Application Styling
# --------------------------------------------------

st.markdown(
    """
    <style>
        .main {
            background-color: #0e1117;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        .main-title {
            font-size: 2.4rem;
            font-weight: 700;
            color: #60a5fa;
        }

        .subtitle {
            color: #9ca3af;
            font-size: 1rem;
        }

        .status-card {
            background-color: #1e293b;
            padding: 18px;
            border-radius: 10px;
            border: 1px solid #334155;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Header
# --------------------------------------------------

col1, col2 = st.columns(
    [1, 8],
    vertical_alignment="center",
)

with col1:
    st.image("assets/drone.png", width=130)

with col2:
    st.markdown(
        '<div class="main-title">AerialMind-X</div>',
        unsafe_allow_html=True,
    )

st.markdown(
    '<div class="subtitle">'
    "Adaptive Vision-Based Autonomous Drone Navigation "
    "with Dynamic Obstacle Avoidance and Goal-Directed Replanning"
    "</div>",
    unsafe_allow_html=True,
)

st.divider()


# --------------------------------------------------
# Sidebar - Mission Configuration
# --------------------------------------------------

st.sidebar.title("Mission Control")

st.sidebar.subheader("Input Configuration")

input_mode = st.sidebar.radio(
    "Select Visual Input",
    ["Image", "Video", "Webcam"],
)

st.sidebar.divider()

st.sidebar.subheader("Navigation Configuration")

grid_size = st.sidebar.slider(
    "Navigation Grid Size",
    min_value=10,
    max_value=50,
    value=30,
    step=5,
)

confidence = st.sidebar.slider(
    "Detection Confidence",
    min_value=0.10,
    max_value=0.90,
    value=0.40,
    step=0.05,
)

st.sidebar.divider()

start_mission = st.sidebar.button(
    "Start Mission",
    type="primary",
    use_container_width=True,
)

stop_mission = st.sidebar.button(
    "Stop Mission",
    use_container_width=True,
)


# --------------------------------------------------
# Mission Status
# --------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Agent Status", "Standby")

with col2:
    st.metric("Detected Obstacles", "0")

with col3:
    st.metric("Replanning Events", "0")

with col4:
    st.metric("Mission Progress", "0%")


# --------------------------------------------------
# Input Section
# --------------------------------------------------

st.subheader("Visual Perception")

if input_mode == "Image":

    st.info("Upload an image for obstacle detection and route planning.")

    uploaded_image = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
        key="image_upload",
    )

    if uploaded_image is not None:
        st.image(
            uploaded_image,
            caption="Uploaded Environment Image",
            use_container_width=True,
        )

elif input_mode == "Video":

    st.info(
        "Upload a video for sequential frame processing "
        "and dynamic obstacle detection."
    )

    uploaded_video = st.file_uploader(
        "Choose a video",
        type=["mp4", "avi", "mov", "mkv"],
        key="video_upload",
    )

    if uploaded_video is not None:
        st.video(uploaded_video)

elif input_mode == "Webcam":

    st.info("Capture an image using your connected camera.")

    camera_image = st.camera_input(
        "Capture Environment",
        key="webcam_input",
    )

    if camera_image is not None:
        st.image(
            camera_image,
            caption="Captured Camera Image",
            use_container_width=True,
        )


# --------------------------------------------------
# Agent Monitoring Sections
# --------------------------------------------------

st.divider()

st.subheader("Autonomous Agent Monitoring")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Perception",
        "Agent Decisions",
        "Navigation",
        "Safety Monitor",
        "Performance",
    ]
)

with tab1:
    st.write("### Vision Perception")
    st.write("YOLO-based obstacle detection results will appear here.")

with tab2:
    st.write("### Agent Decision Log")
    st.write("The agent's perception and navigation decisions will appear here.")

with tab3:
    st.write("### Navigation Environment")
    st.write("Occupancy grid, A* route, replanned path, and drone trajectory.")

with tab4:
    st.write("### Safety Monitoring")
    st.write("Emergency detection, route availability, and safe-stop information.")

with tab5:
    st.write("### Performance Evaluation")
    st.write("Navigation success, path cost, planning time, and replanning metrics.")


# --------------------------------------------------
# Mission Controls
# --------------------------------------------------

if start_mission:
    st.warning(
        "Mission execution will be enabled after integrating "
        "the perception and navigation agent."
    )

if stop_mission:
    st.session_state["mission_stopped"] = True
    st.warning("Mission stop requested.")
