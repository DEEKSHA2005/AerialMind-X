"""AerialMind-X Streamlit application."""

import tempfile
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from aerialmind.core.autonomous_agent import AutonomousAgent
from aerialmind.core.video_navigation_agent import VideoNavigationAgent

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

@st.cache_resource
def load_video_agent(confidence, grid_size):
    """Load and cache the video navigation agent."""
    return VideoNavigationAgent(
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
        "Upload a video to run sequential obstacle detection, "
        "dynamic replanning and autonomous navigation."
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

    if input_mode == "Video":

        if uploaded_video is None:
            st.warning("Please upload a video before starting the mission.")

        else:
            temp_video_path = None

            try:
                with st.spinner(
                    "Processing video frames and performing autonomous navigation..."
                ):
                    video_agent = load_video_agent(confidence, grid_size)

                    suffix = Path(uploaded_video.name).suffix or ".mp4"

                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=suffix,
                    ) as temp_video:
                        temp_video.write(uploaded_video.getvalue())
                        temp_video_path = Path(temp_video.name)

                    start_time = time.perf_counter()

                    video_results = list(
                        video_agent.process_video(temp_video_path)
                    )

                    execution_time = time.perf_counter() - start_time

                    st.session_state["video_results"] = video_results
                    st.session_state["video_execution_time"] = execution_time

                if video_results:
                    st.success(
                        f"Video navigation completed. "
                        f"{len(video_results)} frames processed."
                    )
                else:
                    st.warning("No video frames were processed.")

            except (FileNotFoundError, ValueError, RuntimeError, cv2.error) as error:
                st.session_state.pop("video_results", None)
                st.error(f"Video mission failed: {error}")

            finally:
                if temp_video_path is not None:
                    temp_video_path.unlink(missing_ok=True)

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
video_results = st.session_state.get("video_results", [])

if st.session_state.get("mission_stopped", False):
    result = None
    video_results = []

# Aggregate detections across the COMPLETE video, not only the last frame.
video_total_detections = sum(len(frame.detections) for frame in video_results)
video_frames_with_detections = sum(
    1 for frame in video_results if frame.detections
)
video_peak_detections = max(
    (len(frame.detections) for frame in video_results),
    default=0,
)
video_obstacle_cells = {
    tuple(cell)
    for frame in video_results
    for cell in frame.obstacles
}
video_latest_detection_frame = next(
    (frame for frame in reversed(video_results) if frame.detections),
    video_results[-1] if video_results else None,
)


# --------------------------------------------------
# Mission Status
# --------------------------------------------------

st.divider()
st.subheader("Mission Overview")

if input_mode == "Video" and video_results:

    latest_result = video_results[-1]

    status = (
        "Goal Reached"
        if latest_result.goal_reached
        else "Safe Stop"
        if latest_result.safe_stop
        else "In Progress"
    )

    route_updates = sum(
        video_results[i].planned_path != video_results[i - 1].planned_path
        for i in range(1, len(video_results))
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Agent Status", status)

    with col2:
        st.metric(
            "Object Detection Events (All Frames)",
            video_total_detections,
        )

    with col3:
        st.metric("Route Updates", route_updates)

    with col4:
        progress = 100 if latest_result.goal_reached else 0
        st.metric("Mission Progress", f"{progress}%")

elif result is not None:

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
        progress = 100 if result.goal_reached else 0
        st.metric("Mission Progress", f"{progress}%")

else:

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Agent Status", "Standby")

    with col2:
        st.metric("Detected Obstacles", 0)

    with col3:
        st.metric("Route Updates", 0)

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

    if input_mode == "Image":

        if result is not None:

            st.image(
                cv2.cvtColor(
                    result.annotated_image,
                    cv2.COLOR_BGR2RGB,
                ),
                caption="YOLO Obstacle Detection",
                use_container_width=True,
            )

            if result.detections:
                detection_data = [
                    {
                        "Object": d.label,
                        "Confidence": round(d.confidence, 3),
                        "Bounding Box": str(d.bounding_box),
                    }
                    for d in result.detections
                ]

                st.dataframe(detection_data, use_container_width=True)
            else:
                st.info("No objects detected.")

        else:
            st.info("Upload an image and start an image mission.")

    elif input_mode == "Video":

        if video_results:

            latest = video_results[-1]
            display_frame = video_latest_detection_frame

            st.image(
                cv2.cvtColor(
                    display_frame.annotated_frame,
                    cv2.COLOR_BGR2RGB,
                ),
                caption=(
                    f"Latest frame containing detections (frame "
                    f"{display_frame.frame_index})"
                    if display_frame.detections
                    else "Latest Processed Video Frame"
                ),
                use_container_width=True,
            )

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Frames Processed", len(video_results))
            col2.metric("Frames With Detections", video_frames_with_detections)
            col3.metric("Total Detection Events", video_total_detections)
            col4.metric("Peak Objects In One Frame", video_peak_detections)

            if display_frame.detections:
                detection_data = [
                    {
                        "Object": d.label,
                        "Confidence": round(d.confidence, 3),
                        "Bounding Box": str(d.bounding_box),
                    }
                    for d in display_frame.detections
                ]
                st.write(f"Detections shown are from frame {display_frame.frame_index}.")
                st.dataframe(detection_data, use_container_width=True)
            else:
                st.warning(
                    "YOLO returned zero detections in every processed frame. "
                    "The detection model/configuration must be checked."
                )

        else:
            st.info("Upload a video and start a video mission.")

    else:
        st.info("Autonomous webcam perception is not yet integrated.")


# --------------------------------------------------
# Tab 2 - Agent Decisions
# --------------------------------------------------

with tab2:

    st.write("### Agent Decision Log")

    if input_mode == "Image" and result is not None:

        st.success(result.decision)

        st.write(f"**Detected objects:** {len(result.detections)}")
        st.write(f"**Occupied grid cells:** {len(result.obstacles)}")
        st.write(f"**Initial path cells:** {len(result.initial_path)}")
        st.write(f"**Replanned path cells:** {len(result.planned_path)}")
        st.write(f"**Final drone position:** {result.final_position}")

    elif input_mode == "Video" and video_results:

        latest = video_results[-1]

        if latest.goal_reached:
            st.success("Navigation goal reached.")
        elif latest.safe_stop:
            st.warning("Agent entered safe-stop mode.")
        else:
            st.info("Agent is continuing simulated navigation.")

        st.write(f"**Frames processed:** {len(video_results)}")
        st.write(f"**Current drone position:** {latest.drone_position}")
        st.write(f"**Detected objects in latest frame:** {len(latest.detections)}")
        st.write(f"**Total detection events across video:** {video_total_detections}")
        st.write(f"**Frames containing detections:** {video_frames_with_detections}")
        st.write(f"**Grid cells marked as obstacles at any point:** {len(video_obstacle_cells)}")
        st.write(f"**Current planned path cells:** {len(latest.planned_path)}")
        st.write(f"**Goal reached:** {latest.goal_reached}")
        st.write(f"**Emergency detected:** {latest.emergency_detected}")

        st.write("### Recent Frame Decisions")

        for frame in video_results[-5:]:
            st.write(
                f"Frame {frame.frame_index} | "
                f"Position: {frame.drone_position} | "
                f"Objects: {len(frame.detections)} | "
                f"Safe stop: {frame.safe_stop}"
            )

    else:
        st.info("Start an Image or Video mission to view agent decisions.")


# --------------------------------------------------
# Tab 3 - Navigation
# --------------------------------------------------

with tab3:

    st.write("### Navigation Environment")

    if input_mode == "Image" and result is not None:

        obstacles = result.obstacles
        initial_path = result.initial_path
        planned_path = result.planned_path
        visited_positions = result.visited_positions
        goal = result.goal

    elif input_mode == "Video" and video_results:

        latest = video_results[-1]

        obstacles = video_obstacle_cells
        initial_path = []
        planned_path = latest.planned_path
        visited_positions = [
            frame.drone_position for frame in video_results
        ]
        goal = None

    else:
        obstacles = None

    if obstacles is not None:

        navigation_grid = np.zeros((grid_size, grid_size))

        for row, col in obstacles:
            navigation_grid[row, col] = 1

        fig, ax = plt.subplots(figsize=(9, 7))

        ax.imshow(
            navigation_grid,
            cmap="Greys",
            origin="upper",
        )

        if initial_path:
            ax.plot(
                [p[1] for p in initial_path],
                [p[0] for p in initial_path],
                "--",
                label="Initial A* Path",
                linewidth=2,
            )

        if planned_path:
            ax.plot(
                [p[1] for p in planned_path],
                [p[0] for p in planned_path],
                "-",
                label="Current Planned Path",
                linewidth=2,
            )

        if visited_positions:
            ax.plot(
                [p[1] for p in visited_positions],
                [p[0] for p in visited_positions],
                "o",
                markersize=3,
                label="Drone Trajectory",
            )

            ax.scatter(
                visited_positions[0][1],
                visited_positions[0][0],
                marker="s",
                s=100,
                label="Start",
            )

        if goal is not None:
            ax.scatter(
                goal[1],
                goal[0],
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
        st.info("Start an Image or Video mission to view navigation.")


# --------------------------------------------------
# Tab 4 - Safety Monitor
# --------------------------------------------------

with tab4:

    st.write("### Safety Monitoring")

    if input_mode == "Image" and result is not None:

        emergency = result.emergency_detected
        safe_stop = result.safe_stop
        reason = result.safe_stop_reason

    elif input_mode == "Video" and video_results:

        latest = video_results[-1]

        emergency = latest.emergency_detected
        safe_stop = latest.safe_stop
        reason = latest.safe_stop_reason

    else:
        emergency = None

    if emergency is not None:

        if emergency:
            st.error("Emergency detected.")

        elif safe_stop:
            st.warning("Agent entered safe-stop mode.")

        else:
            st.success("No emergency detected. Safe-stop not activated.")

        st.write(f"**Emergency detected:** {emergency}")
        st.write(f"**Safe stop:** {safe_stop}")
        st.write(f"**Safe-stop reason:** {reason or 'None'}")

    else:
        st.info("Start an Image or Video mission to view safety information.")


# --------------------------------------------------
# Tab 5 - Performance
# --------------------------------------------------

with tab5:

    st.write("### Performance Evaluation")

    if input_mode == "Image" and result is not None:

        execution_time = st.session_state.get("execution_time", 0.0)

        st.metric("Execution Time", f"{execution_time:.3f} seconds")
        st.metric("Initial Path Length", len(result.initial_path))
        st.metric("Replanned Path Length", len(result.planned_path))
        st.metric("Visited Positions", len(result.visited_positions))
        st.metric("Goal Reached", "Yes" if result.goal_reached else "No")

    elif input_mode == "Video" and video_results:

        latest = video_results[-1]

        execution_time = st.session_state.get(
            "video_execution_time", 0.0
        )

        route_updates = sum(
            video_results[i].planned_path != video_results[i - 1].planned_path
            for i in range(1, len(video_results))
        )

        st.metric("Execution Time", f"{execution_time:.3f} seconds")
        st.metric("Frames Processed", len(video_results))
        st.metric(
            "Average Processing Time per Frame",
            f"{execution_time / len(video_results):.4f} seconds",
        )
        st.metric("Route Updates", route_updates)
        st.metric("Latest Path Length", len(latest.planned_path))
        st.metric(
            "Goal Reached",
            "Yes" if latest.goal_reached else "No",
        )

    else:
        st.info("Start an Image or Video mission to view performance metrics.")
