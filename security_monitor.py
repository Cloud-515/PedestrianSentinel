import argparse
import logging
import sys
import threading
import time
from collections.abc import Hashable
from typing import Union

import cv2
import numpy as np
import supervision as sv
from trackers import ByteTrackTracker
from ultralytics import YOLO

from logging_config import configure_logging

MODEL_PATH = "yolo11n.pt"
DWELL_TIME_THRESHOLD_SECONDS = 2.0
ALARM_COOLDOWN_SECONDS = 10.0
RECONNECT_DELAY_SECONDS = 2.0
MAX_RECONNECT_ATTEMPTS = 5

POLYGON = np.array(
    [
        [200, 200],
        [600, 200],
        [600, 600],
        [200, 600],
    ],
    dtype=np.int32,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class AlarmPlayer:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._is_playing = False

    def trigger(self) -> None:
        with self._lock:
            if self._is_playing:
                return
            self._is_playing = True
        threading.Thread(target=self._play, daemon=True).start()

    def _play(self) -> None:
        try:
            try:
                import winsound

                winsound.Beep(1000, 400)
            except ImportError:
                print("\a", end="", flush=True)
        except RuntimeError as error:
            logger.warning("Unable to play alarm: %s", error)
        finally:
            with self._lock:
                self._is_playing = False


def open_capture(video_source: Union[int, str]) -> cv2.VideoCapture | None:
    capture = cv2.VideoCapture(video_source)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if capture.isOpened():
        return capture

    capture.release()
    return None


def track_id_set(detections: sv.Detections) -> set[Hashable]:
    if detections.tracker_id is None:
        return set()

    ids = set()
    for track_id in detections.tracker_id:
        value = track_id.item() if hasattr(track_id, "item") else track_id
        if value != -1:
            ids.add(value)
    return ids


def labels_for(detections: sv.Detections, in_zone_ids: set[Hashable]) -> list[str]:
    if detections.tracker_id is None:
        return ["Person"] * len(detections)

    labels = []
    for track_id in detections.tracker_id:
        value = track_id.item() if hasattr(track_id, "item") else track_id
        if value == -1:
            labels.append("Person")
        else:
            labels.append(f"ID:{value}" + (" IN ZONE" if value in in_zone_ids else ""))
    return labels


def main(video_source: Union[int, str]) -> int:
    logger.info("Loading model: %s", MODEL_PATH)
    model = YOLO(MODEL_PATH)
    tracker = ByteTrackTracker(
        track_activation_threshold=0.25,
        lost_track_buffer=30,
        frame_rate=30,
        minimum_consecutive_frames=1,
        minimum_iou_threshold=0.1,
        high_conf_det_threshold=0.6,
    )
    zone = sv.PolygonZone(
        polygon=POLYGON,
        triggering_anchors=[sv.Position.BOTTOM_CENTER],
    )
    zone_annotator = sv.PolygonZoneAnnotator(
        zone=zone,
        color=sv.Color.RED,
        thickness=2,
    )
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=0.5)
    alarm_player = AlarmPlayer()

    entry_times: dict[Hashable, float] = {}
    last_alarm_times: dict[Hashable, float] = {}
    reconnect_attempts = 0
    capture = open_capture(video_source)

    if capture is None:
        logger.error("Cannot open video source: %s", video_source)
        return 1

    logger.info("Security monitor started. Press q to exit.")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                reconnect_attempts += 1
                logger.warning(
                    "Video stream interrupted; reconnect attempt %d/%d.",
                    reconnect_attempts,
                    MAX_RECONNECT_ATTEMPTS,
                )
                capture.release()

                if reconnect_attempts >= MAX_RECONNECT_ATTEMPTS:
                    logger.error("Video stream could not be recovered.")
                    return 1

                time.sleep(RECONNECT_DELAY_SECONDS)
                capture = open_capture(video_source)
                if capture is None:
                    continue

                tracker = ByteTrackTracker(
        track_activation_threshold=0.25,
        lost_track_buffer=30,
        frame_rate=30,
        minimum_consecutive_frames=1,
        minimum_iou_threshold=0.1,
        high_conf_det_threshold=0.6,
    )
                entry_times.clear()
                last_alarm_times.clear()
                continue

            reconnect_attempts = 0
            result = model(frame, classes=[0], verbose=False)[0]
            detections = sv.Detections.from_ultralytics(result)
            detections = tracker.update(detections)

            in_zone_mask = zone.trigger(detections=detections)
            in_zone_detections = detections[in_zone_mask]
            in_zone_ids = track_id_set(in_zone_detections)
            now = time.monotonic()

            for track_id in in_zone_ids:
                entered_at = entry_times.setdefault(track_id, now)
                dwell_time = now - entered_at
                last_alarm_at = last_alarm_times.get(track_id)

                if dwell_time < DWELL_TIME_THRESHOLD_SECONDS:
                    continue
                if last_alarm_at is not None and now - last_alarm_at < ALARM_COOLDOWN_SECONDS:
                    continue

                logger.warning(
                    "Intrusion detected: person ID %s has remained in the zone for %.1f seconds.",
                    track_id,
                    dwell_time,
                )
                alarm_player.trigger()
                last_alarm_times[track_id] = now

            departed_ids = set(entry_times) - in_zone_ids
            for track_id in departed_ids:
                logger.info("Person ID %s left the zone.", track_id)
                del entry_times[track_id]

            annotated_frame = box_annotator.annotate(scene=frame, detections=detections)
            annotated_frame = label_annotator.annotate(
                scene=annotated_frame,
                detections=detections,
                labels=labels_for(detections, in_zone_ids),
            )
            annotated_frame = zone_annotator.annotate(scene=annotated_frame)

            cv2.imshow("Security Monitor", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return 0
    finally:
        capture.release()
        cv2.destroyAllWindows()


def parse_video_source(value: str) -> Union[int, str]:
    return int(value) if value.isdecimal() else value


if __name__ == "__main__":
    configure_logging()
    parser = argparse.ArgumentParser(description="Pedestrian zone intrusion monitor")
    parser.add_argument(
        "--source",
        default="0",
        help="Camera index or RTSP/video stream URL. Default: 0",
    )
    arguments = parser.parse_args()
    sys.exit(main(parse_video_source(arguments.source)))
