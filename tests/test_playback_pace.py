"""文件播放的节拍：视频位置要跟着墙钟走，而不是跟着推理速度走。

原来的实现是每帧固定睡 ``1/(fps×速度)``，而睡眠**加在推理时间之上** —— 推理一帧
0.1 秒的机器放 30fps 视频，每帧实际耗时 0.13 秒，播放只有约 0.25 倍速，而界面上
写着 1.00×（现场就是这么发现"非低功耗模式播放变慢"的）。

这里钉住三条契约：

* 跟得上时**不跳帧**，也不多睡（把睡眠算进帧间隔里）；
* 跟不上时**跳到该到的位置**，播放速度保持设定值（跳掉的帧不参与检测，
  所以 worker 必须把跳帧数报出来）；
* 暂停恢复、改速度之后**不跳** —— 不然一恢复播放就跳过一大段。
"""

from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

import cv2
import numpy as np

from video_source import VideoSource, VideoSourceSpec

FPS = 30.0
FRAMES = 90
SIZE = (64, 48)


def write_clip(directory: Path) -> Path:
    """写一段真实的小视频（MJPG AVI：编码器到处都有，不用碰 mp4 的编解码依赖）。"""
    path = directory / "clip.avi"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, SIZE
    )
    if not writer.isOpened():
        raise unittest.SkipTest("这台机器没有可用的 MJPG 编码器")
    for index in range(FRAMES):
        writer.write(np.full((SIZE[1], SIZE[0], 3), index % 256, dtype=np.uint8))
    writer.release()
    return path


class PlaybackPaceTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.clip = write_clip(self.directory)

    def source(self, speed: float = 1.0) -> VideoSource:
        source = VideoSource(
            VideoSourceSpec(str(self.clip), operation_mode="video", speed=speed)
        )
        self.assertTrue(source.open(), "小视频应当能打开")
        self.addCleanup(source.close)
        return source

    def test_first_frame_just_anchors_the_clock(self) -> None:
        """打开之后第一次 pace 只建立锚点，不该被判成「落后」而跳帧。"""
        source = self.source()

        self.assertEqual(source.pace(), 0)

    def test_a_consumer_that_keeps_up_never_skips(self) -> None:
        source = self.source()
        source.pace()
        for _ in range(5):
            ok, _, _ = source.read()
            self.assertTrue(ok)
            # 每帧只花掉间隔的五分之一，剩下的时间应当由 pace 睡掉。
            time.sleep(1.0 / FPS / 5)
            self.assertEqual(source.pace(), 0, "跟得上就不该跳帧")

    def test_a_slow_consumer_skips_ahead_to_keep_the_speed(self) -> None:
        """推理比帧间隔慢时：跳到「现在该到的位置」，播放速度不被拉慢。"""
        source = self.source()
        source.pace()
        ok, _, _ = source.read()
        self.assertTrue(ok)

        time.sleep(0.5)  # 模拟一帧推理花了半秒
        skipped = source.pace()

        # 半秒 = 15 帧 @30fps；容差与取整允许差一两帧。
        self.assertGreaterEqual(skipped, 13)
        self.assertLessEqual(skipped, 16)
        # 位置确实跳到了那一带（seek 之后 POS_MSEC 读回的是上一个解出的帧，会差一帧，
        # 所以这里只检查「确实跳过去了」）。
        position = source.position_seconds()
        self.assertGreaterEqual(position, 0.5 - 1.5 / FPS)
        self.assertLessEqual(position, 0.5 + 2.0 / FPS)

    def test_speed_multiplies_the_target_position(self) -> None:
        """2 倍速时，同样的墙钟应当推进两倍的视频位置。"""
        source = self.source(speed=2.0)
        source.pace()
        ok, _, _ = source.read()
        self.assertTrue(ok)

        time.sleep(0.25)
        skipped = source.pace()

        self.assertGreaterEqual(skipped, 12, "2 倍速下 0.25 秒应当跳掉约 15 帧")
        self.assertLessEqual(skipped, 18)

    def test_resuming_from_pause_does_not_jump_forward(self) -> None:
        """暂停期间墙钟还在走，恢复播放时不能把这段算成「落后」。"""
        source = self.source()
        source.pace()
        source.set_paused(True)
        ok, _, _ = source.read()
        self.assertFalse(ok, "暂停时读不到帧")

        time.sleep(0.4)  # 暂停期间的发呆
        source.set_paused(False)

        self.assertEqual(source.pace(), 0, "恢复播放的第一帧不该跳")
        ok, _, _ = source.read()
        self.assertTrue(ok)

    def test_changing_speed_does_not_jump_forward(self) -> None:
        source = self.source(speed=0.5)
        source.pace()
        ok, _, _ = source.read()
        self.assertTrue(ok)

        time.sleep(0.3)
        source.spec.speed = 4.0  # 播放中就改了速度

        self.assertEqual(source.pace(), 0, "改速度那一下只重新起算，不跳帧")

    def test_a_monitor_source_is_never_paced(self) -> None:
        """摄像头/流没有「位置」可对齐：pace 必须是空操作。"""
        source = VideoSource(VideoSourceSpec("0", operation_mode="monitor"))

        self.assertEqual(source.pace(), 0)


if __name__ == "__main__":
    unittest.main()
