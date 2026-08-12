from __future__ import annotations

import unittest

import numpy as np
import supervision as sv
from trackers import ByteTrackTracker


class LowPowerTrackingTests(unittest.TestCase):
    def test_skipped_frames_do_not_prune_track_before_next_detection(self) -> None:
        tracker = ByteTrackTracker(
            track_activation_threshold=0.25,
            lost_track_buffer=30,
            frame_rate=30,
            minimum_consecutive_frames=1,
            minimum_iou_threshold=0.1,
            high_conf_det_threshold=0.6,
        )
        detection = sv.Detections(
            xyxy=np.asarray([[10, 10, 30, 50]], dtype=np.float32),
            confidence=np.asarray([0.9], dtype=np.float32),
            class_id=np.asarray([0], dtype=int),
        )

        first = tracker.update(detection, timestamp=0.0)
        self.assertEqual(first.tracker_id.tolist(), [-1])
        self.assertEqual(len(tracker.tracked_objects), 0)

        self.assertEqual(len(tracker.tracked_objects), 0)
        second = tracker.update(detection, timestamp=4.0 / 30.0)
        track_id = second.tracker_id.tolist()[0]
        self.assertGreaterEqual(track_id, 0)

        for _ in range(3):
            tracked = tracker.tracked_objects
            self.assertEqual(tracked.tracker_id.tolist(), [track_id])
            self.assertTrue(np.array_equal(tracked.xyxy, np.asarray([[10, 10, 30, 50]], dtype=np.float32)))


if __name__ == "__main__":
    unittest.main()
