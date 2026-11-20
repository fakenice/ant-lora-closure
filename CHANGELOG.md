# Changelog

本仓库演进的如实记录。条目日期 = git 提交日期，`git log` 可逐条对上。

## [Unreleased]

- 2026-11-20 规模复核管线就绪（0.5B 基座 × 四档配比，结果待跑）：新增 `scripts/base_model.py`
  单点解析基座目录（`BASE_MODEL_DIR`）并按登记文件复核权重 sha256，`04_train_lora.py`
  与 `05_eval.py` 改用它；`03c_fetch_weights.py` 支持 `BASE_REPO` 换仓库、把 HF API 报的
  LFS sha256 记进登记文件（不符则改名留证并非零退出）；`run_multiseed_500.sh` 与
  `07_multiseed_summary.py` 增加 `BASE_MODEL_DIR`／`TAG_PREFIX`／`LOG` 环境变量，
  产物按前缀命名空间隔离。前缀为空时已发布汇总逐字节不变（实测），CI drift gate 不受影响。
  新增 `tests/test_scale_check.py`（8 项，含「权重摘要不匹配」与「权重缺失」两条拒跑路径），
  全仓 22 项测试通过。文档：`README.md` §7c 命令与三条隔离保证、`docs/DECISIONS.md` 第 14–17 条。
  当天 hf-mirror 对所有 LFS 大文件返回 502，0.5B 权重改走 ModelScope 分片下载、以 HF LFS
  sha256 校验后落位；GPU 让给 boltz2-eval 的 E1 批次，四档 × 3 seed 尚未开跑，
  本条只登记管线，不登记结果（结果见下一条）。
- 2026-11-20 规模复核跑完并出结论：0.5B 基座 × 四档配比 × 3 seed（12 次训练 + 13 次评测，
  全部 rc=0，串行 5,151 s，01:17:12→02:43:03）。新增 `scripts/10_scale_compare.py`
  （纯 stdlib，只读已发布产物）与 `out/scale_compare_500.{json,csv,md}`：题层面配对
  bootstrap（B=2000、seed 42）给出每档 vs base 与 6 对档间两两的 95% 区间、判分口径分解
  （含撞满 `max_new_tokens=12` 的截断计数）、成本对照。主要数字：0.5B base 41.6%（208/500），
  四档对 base 的区间全部跨 0，6 对档间比较 0 对排除 0；1.5B 上唯一被测出的档间差异
  （`r100_0` −7.3 pt）在 0.5B 不复现（+2.9 pt），且其 158 题不可解析有 93 题是 12-token
  截断造成的，只统计可解析题时收到 −0.6 pt；格式效应（混考题→单字母）在两个基座都复现。
  结论：**「配比无稳定提升」随基座规模成立**（README §5 结论 8、§7c）。
  产物入库 72 份（`logs/` 26 份 = 13 评测 + 12 训练 + 1 运行序列、`out/train_q05b_*` 12 份、
  `out/eval/eval_q05b_*` 26 份、三份 q05b 汇总、0.5B 权重登记 1 份、scale_compare 3 份、
  `out/figures/fig3_scale_compare.png` 1 份）；前缀为空时发布汇总仍逐字节不变。
  CI：`10_scale_compare.py` 进再生成步骤，其三份产物进 drift gate。
  测试：`tests/test_scale_check.py` 新增 `ScaleCompareArtifact`（把结论 8 的关键数字锁成断言，
  数据变了就先炸测试而不是悄悄和 README 对不上），并把 `test_missing_weights_abort` 改为在临时
  models 目录里跑（原先依赖本机是否恰好存在 0.5B 权重）；全仓 27 项测试通过。
  文档：README §4/§5/§6/§7/§7b/§7c/§9 相应条目、`docs/DECISIONS.md` 第 18–21 条。
- 2026-11-20 补上「文档说的话有没有产物支撑」这一层，并改掉两处口径过度：
  ① 新增 `scripts/11_verify_claims.py`（纯 stdlib，24 项断言，把 README 结论 1/2/3/6/8 与
  限制第 1 条逐条对回 `out/` 与 `data/` 的已提交产物；不写进 `tests/` 以免 `compileall`
  报模块名不合法），CI 的 regenerate 作业新增同名步骤。它与 drift gate 的分工是：
  gate 证明产物可逐字节再生，本脚本证明再生出来的产物仍然支持文档的说法。
  ② 新增 `docs/SCRIPTS.md`（每个脚本的输入输出、确定性声明、被哪些测试与文档引用）；
  `03c_fetch_weights.py` 的注释与 `docs/DECISIONS.md` 第 17 条补充说明：登记文件里的
  `endpoint` 是脚本自己的镜像通道，权重的实际来历是仓库外的 ModelScope 兜底脚本，
  仓内仍未实现该通道。
  ③ 修掉两处过度陈述：`scale_compare_500.md` 与 README §7c 原写「档间 CI 半宽都在 ±5 pt 量级」，
  实测半宽是档 vs base 3.1–4.4 pt、档间配对 2.6–3.6 pt，可排除门槛改由脚本按半宽向上取整算出
  （仍为 ≥5 pt，但现在是数据而不是手写的）；README §7c 判读第 4 条补上 0.5B `r30_70` 的
  8 题不可解析（0 题截断），不再让纯对话档看起来是唯一有不可解析的一档。
  ④ README §7c 加 `out/figures/fig3_scale_compare.png`（`09_make_figures.py` 出图前
  从逐题日志重算并与两份汇总＋对照表断言）；README §7b 的 CI 说明从三层改为四层（新增结论口径
  校验一层并写明它与 drift gate 的分工），§9 产物清单补 `scripts/11_verify_claims.py`、
  `docs/SCRIPTS.md`、fig3 与 0.5B 权重登记文件的真实内容。
  ⑤ 用干净克隆模拟 CI，发现口径校验第 4 层读不入库的 `data/clean/dialog_zh.jsonl`（在 CI 里会
  直接 FileNotFoundError）：改为「本机有语料文件时数行数，没有时跳过并改校验漏斗计数自洽
  （`final_kept == after_filters − minhash_duplicates`）」，断言数从 23 变 24。
  ⑥ 同一趟模拟还发现 drift gate 对解释器版本敏感：Python 3.12 起 `sum()` 对 float 用 Neumaier
  补偿求和，`r30_70` 的 `loss_last10` 正好卡在 0.99685 临界上（3.11 → 0.9968，3.13 → 0.9969），
  只有 `summary.json`/`ablation.*` 这一处变，其余 9 份进 gate 的产物两版本逐字节相同。
  已发布产物按 3.11 冻结、不改写；改为在 `ci.yml` 里把 regenerate 作业的 3.11 显式注明原因，
  并登记为 `docs/DECISIONS.md` 第 22 条 + README §7b 与 `docs/SCRIPTS.md` 的说明。
  全仓 27 项 unittest + 24 项口径断言通过（干净克隆里 3.11/3.13 各跑一遍，CI 场景下第 4 层
  为 24 项 1 跳过）。
- ~~计划：0.5B 基座上的配比复现（把「配比收益依赖基座规模」从假设变成实测，GPU ~1 天）~~
  已于 2026-11-20 完成，见上条。
- 2026-11-18 新增 `docs/external_baselines.md`：CMExam / HuatuoGPT-II / CMB / DISC 已发表数字
  逐条带出处核对＋口径差异清单；README 加 §4b「外部对照」节
- 2026-11-18 新增 `docs/DECISIONS.md`：显存四道处理、训练/评测口径一致、三态计分、
  泄漏剔除与 CI 门等决定的仓内依据逐条登记（日期对齐 git）

## [v1.0] - 2026-11-18

发布冻结为可引用版本（Release v1.0，含 tag 与 `.zenodo.json`）。

### Added

- `0d722f5` L1：bootstrap CI、完整依赖锁定、README 图；无 GPU 也能重跑的评测再生成门（CI 门）
- `2c03165` CI 修复：eval500 泄漏检查不再依赖不可再分发的训练池
- `aee5d03` README 记录 `02b` 哈希导出与更新后的数据产物清单
- `634d998` `CITATION.cff` + README 引用节
- `95292d7` `.zenodo.json` 发布元数据

## 2026-11-03

- `b000f04` 阶段 5：多种子 × 500 题评测与阳性对照

## 2026-10-15

- `9cb0032` 阶段 4 收口：中文医疗指令数据配比消融与确定性评测（可复现产物入库）
