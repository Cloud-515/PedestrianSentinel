"""让 tests/ 成为一个包，测试才能被标准方式发现。

没有这个文件时 ``python -m unittest discover -s tests -t .`` 会直接抛
``ImportError: Start directory is not importable``，而 ``tests.test_xxx``
也不是可导入的模块名 —— 结果是仓库里 11 个模块 80 多个用例，谁都跑不起来，
只能手工往 ``sys.path`` 里插路径。

测试用的是顶层导入（``from detection_engine import DetectionEngine``），
所以项目根目录必须在 ``sys.path`` 上。两种跑法都已满足这一点：

* ``python -m unittest discover -s tests -t .`` —— ``-t .`` 指定顶层目录为
  项目根，unittest 会把它插入 ``sys.path``；
* ``pytest`` —— 见 pytest.ini，其中显式设了 ``pythonpath = .``。
"""
