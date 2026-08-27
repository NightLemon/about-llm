"""运行本地 A2A 1.0 loopback 协议互操作实验。

底层会启动本地 agent endpoint，发送任务并追踪状态/消息 artifact，不联系外部服务。
这个薄入口固定选择 ``control`` 子命令，其余参数原样交给 package 实现。
"""

from __future__ import annotations

import sys

# 复用经过测试的 A2A CLI，只在项目层提供一条容易发现的学习命令。
from about_llm.agents.a2a_loopback import main

if __name__ == "__main__":
    raise SystemExit(main(["control", *sys.argv[1:]]))
