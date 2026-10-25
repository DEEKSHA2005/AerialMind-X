"""Run real YOLO object detection on a sample image."""

from pathlib import Path

import cv2

from aerialmind.perception.obstacle_detector import ObstacleDetector


def main() -> None:
    image_path = Path("data/samples/obstacle_scene.jpg")

    detector = ObstacleDetector(
        model_path="yolo11n.pt",
        confidence_threshold=0.25,
    )

    detections = detector.detect(image_path)

    image = cv2.imread(str(image_path))

    print(f"Image: {image_path}")
    print(f"Objects detected: {len(detections)}")

    for detection in detections:
        print(
            f"Label: {detection.label}, "
            f"Confidence: {detection.confidence:.2f}, "
            f"Bounding box: {detection.bounding_box}"
        )

        x1, y1, x2, y2 = detection.bounding_box

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image,
            f"{detection.label} {detection.confidence:.2f}",
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    cv2.imshow("AerialMind-X: Obstacle Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
