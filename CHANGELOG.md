# Changelog

本仓库演进的如实记录。条目日期 = git 提交日期，`git log` 可逐条对上。

## [Unreleased]

- 计划：0.5B 基座上的配比复现（把「配比收益依赖基座规模」从假设变成实测，GPU ~1 天）
- 2026-11-18 新增 `docs/external_baselines.md`：CMExam / HuatuoGPT-II / CMB / DISC 已发表数字
  逐条带出处核对＋口径差异清单；README 加 §4b「外部对照」节

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
