"""程序版本号。

运行期也需要知道版本：启动日志要写它，自检报告要写它，出问题时第一个被问到的也是
它。而 ``execode/version_info.txt`` 只在打包时被 PyInstaller 读走、并不进发布包，
所以这里单独放一份。

**两者必须一致**：``--selftest`` 会比对 —— 源码模式读 ``execode/version_info.txt``，
打包版读 exe 自己的版本资源。不一致就判失败，因为「exe 属性里写着 1.0.0、程序日志
里却写着 1.1.0」正是这种双份版本号最容易出的岔子，而它只在交付之后才被发现。
"""

from __future__ import annotations

VERSION = "1.1.0"
