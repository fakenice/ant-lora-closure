# 脚本清单（输入 → 输出 → 谁在用）

每个 `scripts/` 条目的职责、读什么、写什么，以及它的产物被哪份文档或哪条测试引用。
目的只有一个：接手的人（或三个月后的自己）能判断「改这个脚本要重跑哪些东西、会不会撬动已发布结论」。
日期对齐 git 提交；口径决定见 `docs/DECISIONS.md`。

## 0. 取数前的探测（一次性，不进任何产物链）

| 脚本 | 输入 | 输出 | 说明 |
|---|---|---|---|
| `00_probe_sources.py` | HF 镜像 API | 只打印，另写 `data/` 下的 JSON manifest | 选定对话语料来源前的人工探测 |
| `00b_probe_mcq.py` | HF 镜像 API | 只打印 | 确认题源是否有 `(question, options, gold)` 三元组 |
| `00c_peek_choices.py` | HF 镜像 API + Range 探测 | 只打印 | CMExam 记录字段与基座分片布局 |

这三个是探索期脚本，重跑只依赖网络，不改变任何已提交产物。

## 1. 数据链

| 脚本 | 输入 | 输出 | 引用它的地方 |
|---|---|---|---|
| `01_fetch_data.py` | hf-mirror | `data/raw/*`、`data/raw_manifest.json` | `tests/test_data.py`、`docs/DATASHEET.md`、README §2（前缀抽样声明的仓内痕迹） |
| `02_clean_dedup.py` | `data/raw/*` | `data/clean/*.jsonl`、`data/stats/quality_report.json` | `tests/test_data.py`、README §2 清洗漏斗 |
| `02b_export_pool_hashes.py` | `data/clean/mcq_train_pool.jsonl` | `data/clean/mcq_train_pool_qhashes.txt` | CI 无池环境下的泄漏比对（DECISIONS 第 11 条） |
| `03_make_ratios.py` | 两份清洗后语料 | `data/ratios/*.jsonl` + `ratios_manifest.json` | `tests/test_data.py`、README §2 配比表 |
| `03b_fetch_model.py` | hf-mirror | `models/<base>/` | 早期整仓下载入口；大文件改由 `03c` 分片，见 DECISIONS 第 13 条 |
| `03c_fetch_weights.py` | hf-mirror + `BASE_REPO` | `models/<base>/`、`out/model_download*.json` | README §1、§7c；`base_model.upstream_sha()` 读的就是这份登记。登记里的 `endpoint` 是本脚本自己的镜像通道，ModelScope 兜底实现在仓外（DECISIONS 第 17 条） |
| `03d_build_eval_500.py` | `data/clean/mcq_train_pool.jsonl`、CMExam test | `data/clean/mcq_eval_500.jsonl`、`data/stats/eval500_report.json` | `tests/test_eval500.py`（扩题先排 172 道泄漏题） |
| `03e_build_positive_control.py` | `mcq_eval_500.jsonl` | `data/ratios/poscontrol_*.jsonl`、`data/stats/positive_control_report.json` | README §4 阳性对照、`tests/test_positive_control.py` |

## 2. 训练与评测（需要 GPU，不在 CI 范围）

| 脚本 | 输入 | 输出 | 说明 |
|---|---|---|---|
| `base_model.py` | `BASE_MODEL_DIR`、登记文件 | 无（被 import） | 基座目录与权重 sha256 的单点解析，训练与评测共用（DECISIONS 第 14、16 条） |
| `04_train_lora.py` | `data/ratios/*.jsonl`、`models/<base>/` | `models/lora_<tag>/`、`out/train_<tag>.jsonl` | 事件日志含 `mask_check`／`loss_check`／`step`／`done`；`done.steps` 必须是计划步数 |
| `05_eval.py` | `data/clean/mcq_eval_{100,500}.jsonl`、adapter | `out/eval/eval_<tag>.json` + `_items.jsonl` | 逐题记录是后面所有再生与断言的唯一真相源 |
| `run_multiseed_500.sh` | 同上 | 12 训 + 13 评 | 开头有显存闸门；`BASE_MODEL_DIR`／`TAG_PREFIX`／`LOG` 三个变量决定规模复核的命名空间 |

## 3. 汇总、对照、图（纯 stdlib／matplotlib，CI 可再生）

| 脚本 | 输入 | 输出 | 确定性 |
|---|---|---|---|
| `06_collect_results.py` | `out/train_*.jsonl`、`out/eval/*.json` | `out/summary.json`、`ablation.{csv,md}`、`resources.md` | 逐字节可再生，进 drift gate |
| `07_multiseed_summary.py` | 12 份 500 题评测 + `TAG_PREFIX` | `out/multiseed_500{_q05b}_summary.{json,csv,md}` | 同上；前缀为空 = 发布件 |
| `08_positive_control_summary.py` | 三份对照评测 | `out/positive_control_summary.{json,md}` | 同上 |
| `09_make_figures.py` | 逐题 jsonl + 三份汇总 | `out/figures/fig{1,2,3}_*.png` | 出图前先与汇总断言；png 字节不进 gate（matplotlib 版本会改写二进制） |
| `10_scale_compare.py` | 两份 multiseed 汇总 + 逐题 jsonl + sweep 日志 | `out/scale_compare_500.{json,csv,md}` | bootstrap `B=2000`、`random.Random(42)`，逐字节可再生；判读文本由表算出，不手写 |
| `11_verify_claims.py` | 上述全部产物 + `data/` 登记 | 无（只断言，非零退出即失败） | README 结论 1/2/3/6/8 与限制第 1 条的机器校验，CI 的 claims gate。24 项里唯一读不入库文件（`data/clean/dialog_zh.jsonl`）的那项在本机执行、在 CI 自动跳过，跳过时改校验漏斗计数自洽 |

为什么不放进 `tests/`：`tests/` 由 `python -m unittest discover -s tests` 跑，
discover 要求模块名是合法标识符，而 `11_verify_claims` 以数字开头。它自带 unittest 结构，
既可直接运行，也能被 `python -m unittest scripts.11_verify_claims` 加载。

**已知的解释器依赖**：`06_collect_results.py` 的 `loss_last10`/`loss_first10` 用 `sum()/n`，
Python 3.12 起 `sum()` 对 float 改用 Neumaier 补偿求和，`r30_70` 的末 10 步均值正好落在
0.99685 这个临界上（3.11 → 0.9968，3.12+ → 0.9969）。已发布产物按 3.11 冻结，CI 的
regenerate 作业因此钉在 3.11；本机用 3.12+ 跑 drift gate 只会在 `summary.json`/`ablation.*`
这一处出 diff，那是解释器而不是数据变了（`docs/DECISIONS.md` 第 22 条）。
`07`/`08`/`10`/`11` 在 3.11 与 3.13 下产物与判定完全一致。

## 4. 改了东西要重跑什么

- 改数据链（§1）：`01→02→02b→03→03d`，之后所有训练与评测都要重来，已发布结论作废。
- 改训练或评测（§2）：跑对应档的 `04`/`05`（产物名走 `--tag`），再 `06`/`07`/`08`/`10`。
- 只改汇总口径（§3）：重跑对应汇总 + `09`，`git diff` 应只出现在你预期的文件里；
  若 `11_verify_claims.py` 报错，说明文档措辞已经不被产物支撑，先改文档再说。
- 换基座做规模复核：`BASE_REPO` 取权重 → `BASE_MODEL_DIR` + `TAG_PREFIX` 跑 `run_multiseed_500.sh`
  → `07`（带前缀）→ `10`。已发布路径不会被写入，见 DECISIONS 第 15 条。
