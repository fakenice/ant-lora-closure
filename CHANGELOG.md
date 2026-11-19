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
  本条只登记管线，不登记结果。
- 计划：0.5B 基座上的配比复现（把「配比收益依赖基座规模」从假设变成实测，GPU ~1 天）
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
