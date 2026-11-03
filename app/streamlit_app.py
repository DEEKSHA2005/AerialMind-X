"""AerialMind-X Streamlit application."""

import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from aerialmind.core.autonomous_agent import AutonomousAgent

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
# Agent Loading
# --------------------------------------------------

@st.cache_resource
def load_agent(confidence, grid_size):
    """Load and cache the autonomous agent."""
    return AutonomousAgent(
        confidence_threshold=confidence,
        grid_rows=grid_size,
        grid_cols=grid_size,
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
# Input Section
# --------------------------------------------------

st.subheader("Visual Perception")

uploaded_image = None
uploaded_video = None
camera_image = None

if input_mode == "Image":

    st.info(
        "Upload an environment image to start autonomous navigation."
    )

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
        "Video-based sequential perception and replanning "
        "will be integrated in the next stage."
    )

    uploaded_video = st.file_uploader(
        "Choose a video",
        type=["mp4", "avi", "mov", "mkv"],
        key="video_upload",
    )

    if uploaded_video is not None:
        st.video(uploaded_video)

elif input_mode == "Webcam":

    st.info(
        "Capture an environment image using your connected camera."
    )

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
# Mission Execution
# --------------------------------------------------

if stop_mission:
    st.session_state.pop("agent_result", None)
    st.session_state["mission_stopped"] = True
    st.warning("Mission results cleared. No physical drone is controlled.")

if start_mission:

    st.session_state["mission_stopped"] = False

    if input_mode != "Image":
        st.warning(
            "Autonomous execution is currently enabled for Image mode only."
        )

    elif uploaded_image is None:
        st.warning("Please upload an image before starting the mission.")

    else:
        try:
            with st.spinner(
                "Agent is detecting obstacles and planning navigation..."
            ):
                image_bytes = np.frombuffer(
                    uploaded_image.getvalue(),
                    dtype=np.uint8,
                )

                image = cv2.imdecode(
                    image_bytes,
                    cv2.IMREAD_COLOR,
                )

                if image is None:
                    raise ValueError("Uploaded image could not be decoded.")

                agent = load_agent(confidence, grid_size)

                start_time = time.perf_counter()

                result = agent.run_mission(image)

                execution_time = (
                    time.perf_counter() - start_time
                )

                st.session_state["agent_result"] = result
                st.session_state["execution_time"] = execution_time

            st.success("Autonomous agent execution completed.")

        except (FileNotFoundError, ValueError, RuntimeError, cv2.error) as error:
            st.session_state.pop("agent_result", None)
            st.error(f"Mission execution failed: {error}")


# --------------------------------------------------
# Retrieve Mission Results
# --------------------------------------------------

result = st.session_state.get("agent_result")

if st.session_state.get("mission_stopped", False):
    result = None


# --------------------------------------------------
# Mission Status
# --------------------------------------------------

st.divider()
st.subheader("Mission Overview")

if result is not None:

    status = (
        "Goal Reached"
        if result.goal_reached
        else "Safe Stop"
        if result.safe_stop
        else "In Progress"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Agent Status", status)

    with col2:
        st.metric("Detected Obstacles", len(result.detections))

    with col3:
        route_updates = int(
            result.planned_path != result.initial_path
        )
        st.metric("Route Updates", route_updates)

    with col4:
        progress = (
            100.0
            if result.goal_reached
            else 0.0
        )
        st.metric("Mission Progress", f"{progress:.0f}%")

else:

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Agent Status", "Standby")

    with col2:
        st.metric("Detected Obstacles", "0")

    with col3:
        st.metric("Route Updates", "0")

    with col4:
        st.metric("Mission Progress", "0%")


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


# --------------------------------------------------
# Tab 1 - Perception
# --------------------------------------------------

with tab1:

    st.write("### Vision Perception")

    if result is not None:

        st.image(
            cv2.cvtColor(
                result.annotated_image,
                cv2.COLOR_BGR2RGB,
            ),
            caption="YOLO Obstacle Detection",
            use_container_width=True,
        )

        st.write("### Detection Details")

        if result.detections:

            detection_data = [
                {
                    "Object": detection.label,
                    "Confidence": round(
                        detection.confidence, 3
                    ),
                    "Bounding Box": str(
                        detection.bounding_box
                    ),
                }
                for detection in result.detections
            ]

            st.dataframe(
                detection_data,
                use_container_width=True,
            )

        else:
            st.info("No objects detected.")

    else:
        st.info("Start an image mission to view perception results.")


# --------------------------------------------------
# Tab 2 - Agent Decisions
# --------------------------------------------------

with tab2:

    st.write("### Agent Decision Log")

    if result is not None:

        st.success(result.decision)

        st.write(
            f"**Detected objects:** {len(result.detections)}"
        )

        st.write(
            f"**Occupied grid cells:** {len(result.obstacles)}"
        )

        st.write(
            f"**Initial path cells:** {len(result.initial_path)}"
        )

        st.write(
            "**Replanned path cells:** "
            f"{len(result.planned_path) if result.planned_path else 0}"
        )

        st.write(
            f"**Final drone position:** {result.final_position}"
        )

    else:
        st.info("Agent decisions will appear after mission execution.")


# --------------------------------------------------
# Tab 3 - Navigation
# --------------------------------------------------

with tab3:

    st.write("### Navigation Environment")

    if result is not None:

        navigation_grid = np.zeros(
            (grid_size, grid_size)
        )

        for row, col in result.obstacles:
            navigation_grid[row, col] = 1

        fig, ax = plt.subplots(figsize=(9, 7))

        ax.imshow(
            navigation_grid,
            cmap="Greys",
            origin="upper",
        )

        initial_rows = [point[0] for point in result.initial_path]
        initial_cols = [point[1] for point in result.initial_path]

        ax.plot(
            initial_cols,
            initial_rows,
            "--",
            label="Initial A* Path",
            linewidth=2,
        )

        if result.planned_path:

            planned_rows = [
                point[0] for point in result.planned_path
            ]

            planned_cols = [
                point[1] for point in result.planned_path
            ]

            ax.plot(
                planned_cols,
                planned_rows,
                "-",
                label="Replanned Path",
                linewidth=2,
            )

        visited_rows = [
            point[0] for point in result.visited_positions
        ]

        visited_cols = [
            point[1] for point in result.visited_positions
        ]

        ax.plot(
            visited_cols,
            visited_rows,
            "o",
            markersize=3,
            label="Drone Trajectory",
        )

        ax.scatter(
            result.initial_path[0][1],
            result.initial_path[0][0],
            marker="s",
            s=100,
            label="Start",
        )

        ax.scatter(
            result.goal[1],
            result.goal[0],
            marker="*",
            s=180,
            label="Goal",
        )

        ax.set_title("AerialMind-X Navigation Grid")
        ax.set_xlabel("Grid Column")
        ax.set_ylabel("Grid Row")
        ax.legend()
        ax.grid(alpha=0.2)

        st.pyplot(fig)
        plt.close(fig)

    else:
        st.info("Navigation visualization will appear after execution.")


# --------------------------------------------------
# Tab 4 - Safety Monitor
# --------------------------------------------------

with tab4:

    st.write("### Safety Monitoring")

    if result is not None:

        if result.emergency_detected:
            st.error("Emergency detected at the drone's current position.")

        elif result.safe_stop:
            st.warning("Drone entered safe-stop mode.")

        else:
            st.success("No emergency detected. Safe-stop not activated.")

        st.write(
            f"**Emergency detected:** {result.emergency_detected}"
        )

        st.write(
            f"**Safe stop:** {result.safe_stop}"
        )

        st.write(
            f"**Safe-stop reason:** "
            f"{result.safe_stop_reason or 'None'}"
        )

    else:
        st.info("Safety information will appear after mission execution.")


# --------------------------------------------------
# Tab 5 - Performance
# --------------------------------------------------

with tab5:

    st.write("### Performance Evaluation")

    if result is not None:

        execution_time = st.session_state.get(
            "execution_time", 0.0
        )

        st.metric(
            "Execution Time",
            f"{execution_time:.3f} seconds",
        )

        st.metric(
            "Initial Path Length",
            len(result.initial_path),
        )

        st.metric(
            "Replanned Path Length",
            len(result.planned_path)
            if result.planned_path
            else 0,
        )

        st.metric(
            "Visited Positions",
            len(result.visited_positions),
        )

        st.metric(
            "Goal Reached",
            "Yes" if result.goal_reached else "No",
        )

    else:
        st.info("Performance metrics will appear after execution.")
