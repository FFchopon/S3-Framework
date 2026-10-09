> Historical audit dated 2026-09-29. Configuration and validation statements below describe that snapshot, not the current checkout.

# S3 实现检查与修复记录

检查日期：2026-09-29。对照本地论文 `2608.02683v1.pdf` 第 4 节及仓库 README，
将修改限定为路径、导入、状态同步、解析、通信和结果持久化问题。
没有改动阶段划分、攻击任务、规则 JSON、技能内容、模型默认值、记忆采样策略或恢复决策逻辑。

## 已修复

| 问题 | 修复及影响 |
| --- | --- |
| `guardagent` 与实际 `GuardAgent` 大小写不一致 | 修正桥接与过滤器路径，避免大小写敏感文件系统上加载失败。Windows 测试检查目录名精确大小写；未实际运行 Linux。 |
| 过滤器找不到库内规则 | 增加实际 `skill_library` 路径；优先使用已启用技能自己的规则，保留旧路径兼容。 |
| 没有检测技能时静默跳过全部阶段 | 空注册表发出明确警告，保留原有技能启用配置。 |
| 多个环境实例或快照切换后使用错误场景 | 物体解析使用实例的 world profile；快照恢复同步全局 profile 和 benign 标志；Guard 缓存按场景区分，避免旧场景提示词复用。 |
| 记忆排名解析抛异常或截断小数 | 忽略非法、非正、布尔及非整数排名，合法排名保序去重；不改变正常排名的删除语义。 |
| 记忆检索结果误入 ParseData | 有工具元数据的路径也排除 `search_past_conversations`，与已有观察过滤规则保持一致。 |
| 同秒运行覆盖结果文件；写入中断损坏旧 JSON | 文件名增加随机唯一标识；同目录临时文件写完后原子替换，替换失败保留旧结果并清理临时文件。 |
| 写入统计与重新汇总不一致 | 成功字段使用明确布尔判断，未知或缺失危险结果不计为安全成功，保持现有汇总定义。两份 writer 实现同步修复。 |
| 子进程中文通信与 worker 阻塞风险 | 双端明确 UTF-8；worker 错误输出继承父进程，不再写入无人读取的管道。 |
| `deepagent` 包导入失败 | 修复指向不存在的 `deepagent.embodied_env` 和 `deepagent.stage_capture` 的引用。 |

## 配置发现

当前 `GuardAgent/skills/` 只有 `recover`，六个检测技能位于 `GuardAgent/skill_library/`。
因此当前配置不执行任何阶段检测；仅设置 `--guard` 不等于启用了完整 S3 防御。
遵照保留基本设定的要求，本次没有自动启用技能。README 已说明如何选择并复制技能目录，
以及用 `python GuardAgent/agent.py --list-stages` 验证配置。
论文中的 tool execution 在现有 AIR 实现中使用 `post_step` 名称，本次未重命名。

## 验证

- 首轮 14 个回归用例：修复前 12 失败、2 通过，修复后全部通过。
- 扩展后的最终测试：`python -m pytest tests -q`，23 通过。
- 测试覆盖真实 pool/subprocess 的中文错误返回；使用不存在的阶段在模型构建前结束，不发送模型请求。
- Guard 缓存测试替换 agent 构建函数，验证缓存隔离，不验证模型行为。
- 主入口与 GuardAgent 的 `--help` 均成功。
- 全部 43 个 Python 文件通过 AST 语法检查；`pip check` 与 `git diff --check` 通过。
- 测试依赖安装在项目 `.venv`，未修改 requirements：Python 3.12、deepagents 0.7.19、
  langchain 1.4.3、langchain-quickjs 0.3.7、pytest 9.1.1。

未执行真实 LLM 攻防实验，未重跑或修改已有实验结果；以上测试不能作为防御成功率或论文指标的验证。
