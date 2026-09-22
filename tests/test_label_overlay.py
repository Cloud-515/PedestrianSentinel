"""框上那行标签：中文文案、相互不重叠、深浅背景上都读得出来。

渲染好不好看没法断言，但它的**契约**可以，而这个标签的三条契约恰好都是"改坏了看得
出来、但没人盯着就不知道什么时候坏了"的那种：

* 中文必须真的画得出字。它靠的是 opencv 5.0 内置的 Unicode 字体（Hershey 字体本身
  只有 ASCII），**降级 opencv 会让中文变成方块或空白**，而程序不会报任何错；
* 标签之间不能重叠。人一多、一排人站在同一高度上时，各自往自己框上一贴就会叠成一团；
* 文字下面必须有垫底。标签会落在沥青、地砖、人身上，没有垫底时总有一半场合读不出来。

另外两条"别把信息弄丢"的：标签横向不能离开自己的框（离开后就看不出是谁的了），
以及再挤也要给每个框都画出标签（宁可重叠，不能让某个人没有标签）。
"""

from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np
import supervision as sv

from detection_engine import DetectionEngine
from models import ZoneDefinition


def detections(*boxes: tuple[int, int, int, int]) -> sv.Detections:
    return sv.Detections(
        xyxy=np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
        confidence=np.full(len(boxes), 0.9, dtype=np.float32),
        class_id=np.zeros(len(boxes), dtype=int),
        tracker_id=np.arange(len(boxes)),
    )


def build_engine() -> DetectionEngine:
    """只用来测标注的引擎：模型与追踪器都是假的。"""
    with (
        patch("detection_engine.YOLO", return_value=Mock()),
        patch.object(DetectionEngine, "_new_tracker", return_value=Mock()),
    ):
        return DetectionEngine("model.pt", [], "cpu")


def rects_overlap(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> bool:
    """两个矩形是否有**共同像素**（不留间隔，与引擎里那个带 margin 的判据不同）。"""
    return DetectionEngine._rects_overlap(first, second)


class CjkFontTests(unittest.TestCase):
    """中文标签能不能画出来 —— 这一条只在换 opencv 时才会坏，坏了没人看得出来。"""

    def render(self, text: str) -> np.ndarray:
        canvas = np.zeros((80, 400), dtype=np.uint8)
        cv2.putText(
            canvas,
            text,
            (10, 60),
            DetectionEngine.LABEL_FONT,
            1.2,
            255,
            DetectionEngine.LABEL_THICKNESS,
            cv2.LINE_AA,
        )
        return canvas

    def test_each_chinese_character_gets_its_own_glyph(self) -> None:
        """逐字比：字与字之间必须长得不一样。

        这条挡的是「字体里没有中文」这件事的两种表现 —— 一个字都没画出来（空白），
        或者每个字都画成同一个占位符（方块／问号）。两种情况下程序都不报错，只是现场
        看到的标签没法读。
        """
        glyphs = {character: self.render(character) for character in "区域人物秒"}

        for character, canvas in glyphs.items():
            self.assertGreater(
                int(np.count_nonzero(canvas)),
                100,
                f"“{character}”一个像素都没画出来 —— 内置 Unicode 字体是不是没有了？",
            )
        characters = list(glyphs)
        for first, second in zip(characters, characters[1:]):
            self.assertGreater(
                int(np.count_nonzero(glyphs[first] != glyphs[second])),
                50,
                f"“{first}”和“{second}”画出来几乎一样 —— 字形被替换成了同一个占位符",
            )

    def test_a_longer_label_measures_wider(self) -> None:
        """排版按 getTextSize 算尺寸，它对中文必须随字数增长（否则防重叠是假的）。"""
        short, _ = DetectionEngine._label_plate_size("目标1", 1.0)
        long, _ = DetectionEngine._label_plate_size("目标1 0.90 区内1.5秒", 1.0)

        self.assertGreater(long, short)


class PlainLabelTests(unittest.TestCase):
    """默认那版标签：只说看画面的人要判断的三件事。

    它是**默认**（``show_detection_details=False``），所以测试从构造函数与 ``_annotate``
    的默认路径走，而不是直接调 ``_box_label`` 传参 —— 不然「默认值被谁改成了 True」
    这件事就没人挡得住。
    """

    def test_the_default_wording_is_the_plain_one(self) -> None:
        label = DetectionEngine._box_label

        self.assertEqual(label(9, 0.86, None), "行人", "区域外只说这是个人")
        self.assertEqual(
            label(9, 0.86, 1.5), "目标9 区域内 1.5秒", "进了区域、还没到阈值"
        )
        self.assertEqual(
            label(9, 0.86, 2.0, alarmed=True), "目标9 闯入 2.0秒", "已经报过警了"
        )
        self.assertEqual(label(None, None, None), "行人", "没编号也是个人")

    def test_the_zone_label_carries_the_id_that_the_records_use(self) -> None:
        """进了区域就必须带编号。

        报警记录里写的是「目标 9」，而取证截图可能有几个人同时在区域内 —— 只有编号
        能让人认出记录里那条对应画面上的哪个人（这条是用户提的：截图要多目标可分辨）。
        """
        for text in (
            DetectionEngine._box_label(9, 0.86, 1.5),
            DetectionEngine._box_label(9, 0.86, 2.0, alarmed=True),
        ):
            with self.subTest(text=text):
                self.assertIn("目标9", text)

    def test_the_plain_wording_never_leaks_the_confidence(self) -> None:
        """分数不该出现 —— 加回去等于这个开关白做。

        编号反而**要**在区域内的标签里（见上一条），秒数也是（「待了多久」正是操作员
        要判断的三件事之一），所以这里查的是分数。
        """
        for text in (
            DetectionEngine._box_label(9, 0.86, None),
            DetectionEngine._box_label(9, 0.86, 1.5),
            DetectionEngine._box_label(9, 0.86, 2.0, alarmed=True),
        ):
            with self.subTest(text=text):
                self.assertNotIn("0.86", text)
                self.assertNotIn("置信", text)

    def test_a_person_outside_every_zone_has_no_number(self) -> None:
        """区域外的框不带编号：它不在任何记录里，写了只是给画面添字。"""
        self.assertEqual(DetectionEngine._box_label(9, 0.86, None), "行人")

    def test_the_details_switch_brings_the_machine_view_back(self) -> None:
        self.assertEqual(
            DetectionEngine._box_label(9, 0.86, None, show_details=True), "目标9 0.86"
        )
        self.assertEqual(
            DetectionEngine._box_label(9, 0.86, 1.5, show_details=True),
            "目标9 0.86 区内1.5秒",
        )
        self.assertEqual(
            DetectionEngine._box_label(None, 0.24, None, show_details=True), "人物 0.24"
        )

    def test_a_fresh_engine_shows_the_plain_labels(self) -> None:
        self.assertFalse(build_engine().show_detection_details)

    def test_the_switch_reaches_the_frame(self) -> None:
        """整条链路：引擎上那个开关一改，画出来的字就跟着改。"""
        engine = build_engine()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        box = (200, 200, 300, 400)
        zone = ZoneDefinition(
            name="区域 1",
            polygon=[[100.0, 100.0], [600.0, 100.0], [600.0, 460.0], [100.0, 460.0]],
            closed=True,
        )
        engine.zones = [zone]

        def drawn(*, in_zone: bool, alarmed: bool = False) -> list[str]:
            with patch("detection_engine.cv2.putText") as put_text:
                engine._annotate(
                    frame,
                    detections(box),
                    {zone.name: {0}} if in_zone else {},
                    {0: 1.5} if in_zone else {},
                    alarmed_ids={0} if alarmed else set(),
                )
            # 区域名也走 putText（画的就是「区域 1」），按整串排掉它。
            return [call.args[1] for call in put_text.call_args_list if call.args[1] != zone.name]

        self.assertEqual(drawn(in_zone=False), ["行人"])
        self.assertEqual(drawn(in_zone=True), ["目标0 区域内 1.5秒"])
        self.assertEqual(drawn(in_zone=True, alarmed=True), ["目标0 闯入 1.5秒"])

        engine.show_detection_details = True

        self.assertEqual(drawn(in_zone=False), ["目标0 0.90"])
        self.assertEqual(drawn(in_zone=True), ["目标0 0.90 区内1.5秒"])


class LabelLayoutTests(unittest.TestCase):
    """防重叠排版：只动纵向、绝不横向离开自己的框。"""

    def place(
        self,
        boxes: list[tuple[int, int, int, int]],
        frame_size: tuple[int, int] = (1920, 1080),
        occupied: list[tuple[int, int, int, int]] | None = None,
    ) -> list[tuple[int, int, int, int]]:
        scale = DetectionEngine._label_scale(frame_size[1])
        entries = [
            (box, DetectionEngine._label_plate_size(f"目标{index} 0.90 区内1.5秒", scale))
            for index, box in enumerate(boxes)
        ]
        return DetectionEngine._layout_labels(entries, frame_size, occupied)

    def test_a_lone_label_sits_just_above_its_box(self) -> None:
        box = (400, 300, 500, 700)
        (rect,) = self.place([box])

        self.assertEqual(rect[0], box[0], "标签左边与框左边对齐")
        self.assertEqual(
            rect[1] + rect[3], box[1] - DetectionEngine.LABEL_GAP, "标签贴在框上方"
        )

    def test_labels_of_piled_up_boxes_do_not_overlap(self) -> None:
        """十几个人挤在一起（同一高度、横向只差几像素）时不能叠成一团。"""
        boxes = [(200 + index * 6, 300 + index * 2, 300 + index * 6, 700) for index in range(12)]

        rects = self.place(boxes)

        self.assertEqual(len(rects), len(boxes), "每个框都要有标签")
        for index, first in enumerate(rects):
            for second in rects[index + 1 :]:
                self.assertFalse(
                    rects_overlap(first, second),
                    f"标签重叠了：{first} 与 {second}",
                )

    def test_labels_never_leave_their_box_horizontally(self) -> None:
        """让位可以上下挪，但横向必须和框对齐 —— 挪走了就看不出这行字是谁的。"""
        boxes = [(300 + index * 4, 400, 400 + index * 4, 800) for index in range(10)]

        rects = self.place(boxes)

        for box, rect in zip(boxes, rects):
            self.assertEqual(rect[0], box[0])

    def test_labels_stay_inside_the_frame(self) -> None:
        """画面边角上的人（框贴边）不能让标签画到画面外去。"""
        frame_size = (800, 600)
        boxes = [
            (0, 0, 60, 200),
            (740, 0, 800, 200),
            (0, 560, 60, 600),
            (740, 560, 800, 600),
        ]

        rects = self.place(boxes, frame_size)

        for rect in rects:
            x, y, width, height = rect
            self.assertGreaterEqual(x, 0)
            self.assertGreaterEqual(y, 0)
            self.assertLessEqual(x + width, frame_size[0])
            self.assertLessEqual(y + height, frame_size[1])

    def test_a_busy_corner_pushes_labels_down_instead_of_off_screen(self) -> None:
        """一叠框都在画面最上沿：上面没地方了，标签要往框内/下方让，而不是顶出画面。"""
        boxes = [(100 + index * 5, 0, 200 + index * 5, 120) for index in range(6)]

        rects = self.place(boxes, (640, 480))

        self.assertEqual(len(rects), len(boxes))
        for rect in rects:
            self.assertGreaterEqual(rect[1], 0)
            self.assertLessEqual(rect[1] + rect[3], 480)

    def test_occupied_places_are_respected(self) -> None:
        """区域名先占住位置：框标签要躲开它，而不是盖住它。"""
        box = (400, 300, 500, 700)
        scale = DetectionEngine._label_scale(1080)
        size = DetectionEngine._label_plate_size("目标0 0.90 区内1.5秒", scale)
        # 区域名正好占着「框正上方」那一条。
        name_rect = (box[0], box[1] - DetectionEngine.LABEL_GAP - size[1], size[0], size[1])

        (rect,) = self.place([box], occupied=[name_rect])

        self.assertFalse(rects_overlap(rect, name_rect), "标签盖住了区域名")

    def test_every_box_gets_a_label_even_when_there_is_no_room(self) -> None:
        """挤到实在放不下时接受重叠 —— 但一个都不能少（少一个就是一个人没有标签）。"""
        boxes = [(500, 500, 560, 900) for _ in range(30)]

        rects = self.place(boxes)

        self.assertEqual(len(rects), len(boxes))
        for rect in rects:
            self.assertLessEqual(rect[1] + rect[3], 1080)

    def test_labels_avoid_the_zone_name_in_a_real_frame(self) -> None:
        """整条链路：区域名占位 → 标签让位。

        判据是「有人的那一帧」与「没人的那一帧」在名字那块**逐像素一致** —— 标签真躲开
        了才做得到。人正好站在区域第一个顶点（也就是名字）底下，首选位置本来是要撞上的。
        """
        engine = build_engine()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        zone = ZoneDefinition(
            name="区域 1",
            polygon=[[200.0, 200.0], [600.0, 200.0], [600.0, 460.0], [200.0, 460.0]],
            closed=True,
        )
        engine.zones = [zone]
        polygon = np.asarray(zone.polygon, dtype=np.float32)
        name_rect = engine._draw_zone_name(np.zeros_like(frame), zone, polygon)
        assert name_rect is not None

        with_person = engine._annotate(frame, detections((200, 205, 260, 460)), {}, {})
        without_person = engine._annotate(frame, detections(), {}, {})

        x, y, width, height = name_rect
        # 先确认这个测试真的踩到了"要撞"的位置，否则它什么都没验。
        label_rect = engine._layout_labels(
            [
                (
                    (200, 205, 260, 460),
                    engine._label_plate_size("目标0 0.90", engine._label_scale(480)),
                )
            ],
            (640, 480),
        )[0]
        self.assertTrue(
            rects_overlap(label_rect, name_rect),
            "构造的画面没踩到冲突 —— 这个测试就没意义了",
        )
        np.testing.assert_array_equal(
            with_person[y : y + height, x : x + width],
            without_person[y : y + height, x : x + width],
            "区域名那块被框标签动过了 —— 名字没占住位置",
        )


class LabelPlateTests(unittest.TestCase):
    """文字垫底：浅色地面上白字看不见，深色地面上绿字也偏暗，垫一层最深。"""

    def setUp(self) -> None:
        engine = build_engine()
        self.engine = engine
        # 亮底图：不垫底的话标签的文字与底色接近，这一条才测得出来。
        rng = np.random.default_rng(20260922)
        self.frame = rng.integers(180, 240, size=(480, 640, 3), dtype=np.uint8)

    def label_rect(self, box: tuple[int, int, int, int], text: str) -> tuple[int, int, int, int]:
        """复算一下排版应该把标签放在哪 —— 用来在渲染结果里找那块垫底。"""
        scale = self.engine._label_scale(self.frame.shape[0])
        size = self.engine._label_plate_size(text, scale)
        return self.engine._layout_labels([(box, size)], (640, 480))[0]

    def test_the_plate_darkens_the_background_under_the_label(self) -> None:
        box = (200, 200, 300, 400)
        rect = self.label_rect(box, "目标0 0.90")

        rendered = self.engine._annotate(self.frame, detections(box), {}, {})
        x, y, width, height = rect
        before = self.frame[y : y + height, x : x + width].mean()
        after = rendered[y : y + height, x : x + width].mean()

        self.assertLess(after, before - 20, "标签没有垫底，亮地面上会读不出来")

    def test_the_text_is_painted_on_top_of_the_plate(self) -> None:
        """垫底之后字还得在 —— 只压暗不画字的话，这一条就会挂。

        垫底之后底图最亮也到不了 100（亮底 240 × 0.38），而字用的是正的框色
        （绿 200 / 红 255），所以"明显亮于垫底"的像素只可能是字。
        """
        box = (200, 200, 300, 400)
        rect = self.label_rect(box, "目标0 0.90")

        rendered = self.engine._annotate(self.frame, detections(box), {}, {})
        x, y, width, height = rect
        inked = int(np.count_nonzero(rendered[y : y + height, x : x + width].max(axis=2) > 130))

        self.assertGreater(inked, 50, "垫底里没有字 —— 标签把内容也一起压暗了")

    def test_a_person_pressing_against_the_frame_edge_is_still_labelled(self) -> None:
        """框贴到画面最上沿时，上面已经没地方了：标签要挪进画面里，而不是画到画面外。"""
        box = (0, 0, 80, 200)
        rect = self.label_rect(box, "目标0 0.90")

        x, y, width, height = rect
        self.assertGreaterEqual(x, 0)
        self.assertGreaterEqual(y, 0)
        self.assertLessEqual(x + width, 640)
        self.assertLessEqual(y + height, 480)

        rendered = self.engine._annotate(self.frame, detections(box), {}, {})
        self.assertLess(
            rendered[y : y + height, x : x + width].mean(),
            self.frame[y : y + height, x : x + width].mean() - 20,
            "贴边的框也该有标签（整块都在画面内）",
        )


if __name__ == "__main__":
    unittest.main()
