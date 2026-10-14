# 中文医疗指令数据配比 → LoRA → 确定性评测（单机闭合实验）

一台 RTX 2080 Ti 11 GB 上，把"中文医疗 SFT 语料与单选题语料按不同比例混合，会不会影响下游选择题表现"这个问题做成一个**可复现、可核验、带负结论**的闭合实验：数据取数与登记 → 清洗去重 → 泄漏剔除 → 四档配比 → LoRA 微调 → 100 题确定性评测 → 消融表。

结论先说：**在这个规模上，四档配比的准确率彼此不可区分（69%–74%，全部 95% 区间重叠），而且都不高于未微调的基座（73%）。配比真正改变的是输出格式与训练成本，不是正确率。** 详见下文"结论"和 `docs/DATASHEET.md` 的限制清单。

## 1. 环境与硬件约束（决定了后面所有取舍）

| 项 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 2080 Ti，11,263.7 MiB（Turing sm_75：无 bf16、无 FlashAttention-2） |
| 精度 | fp16 权重 + fp32 累加/loss |
| torch / CUDA | 2.6.0+cu124 / 12.4 |
| 基座 | Qwen2.5-1.5B-Instruct，`model.safetensors` 3,087,467,144 B，sha256 `dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee` |
| 下载 | hf-mirror.com + 分片 Ranged GET；必须 `HF_HUB_DISABLE_XET=1`（镜像下 Xet CAS 返回 401），并按 `Retry-After` 退避（镜像 429 很凶）。见 `out/model_download.json` |

因为显存只有 11 GB，长序列训练会撞 Windows WDDM 的显存溢出（表现为 s/步 从 4 秒跳到 150 秒而不报任何 throttle 标志）。所以脚本里做了四道处理：`max_len=512`、只算答案位置的 cross-entropy（切到监督后缀再过 `lm_head`，避免物化 T×151,936 的 fp32 logits）、每步 `empty_cache()` + `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128`、以及"连续两步超过 25 秒就主动中止"的保护（宁可失败也不要 thrash）。

## 2. 数据链

来源、字节数、sha256、清洗漏斗、抽样偏差声明全部登记在 [`docs/DATASHEET.md`](docs/DATASHEET.md)。要点：

- 对话语料：`shibing624/medical` 的 `finetune/train_zh_0.json`，只取了前 62,914,560 B（全量 1,338,359,048 B 的 4.7%）→ 91,306 条。长度过滤（响应 20–900 字）去掉 29,089 条，低中文占比去掉 1,344 条，MinHash LSH（3-gram、阈值 0.8、128 perm、保留首条）去掉 2,086 条 → **58,787 条**。
- 选择题语料：`fzkuji/CMExam`，train 54,497 / test 6,811 条；结构不合法剔除 8,071 / 972；按归一化题干做 train↔eval 泄漏比对，从训练池剔除 **5 条** → 训练池 46,421 条，评测集 100 条。
- 评测集构造：固定种子 `20261014` 抽样，正确项字母分布 A21/B19/C15/D26/E19，96 题五选项、4 题四选项，最长题干 252 token（上限 900）。
- **必须承认的偏差**：对话语料是按文件顺序取前缀，不是全量随机抽样，所以配比结论只适用于"该文件的前缀样本"。

四档配比（每档 1,200 条，同一 seed 分层抽样）：

| 配置 | 对话 : 考题 | 写入条数 |
|---|---|---|
| `r100_0` | 1200 : 0 | 1200 |
| `r70_30` | 840 : 360 | 1200 |
| `r30_70` | 360 : 840 | 1200 |
| `r0_100` | 0 : 1200 | 1200 |

## 3. 训练与评测口径

- LoRA：r=16、α=32、dropout 0.05、全部 7 类投影 → **可训练 18.46 M / 15.62 亿参数（1.18%）**。
- AdamW，lr 2e-4，warmup 10，micro-batch 1 × 梯度累积 8 × 150 优化步 ≈ 1 个 epoch 的 1,200 行（超长丢弃见下表）。
- Prompt 屏蔽逐条核验：拼接后的前缀 token 必须与模板单独编码结果完全一致才屏蔽，探针 **200/200 通过**（`mask_check` 事件）。
- 自写 response-position CE 与 HF 官方 loss 对齐校验：**绝对差 0.0**（`loss_check` 事件，四档都是）。
- 评测：100 题单选，贪心解码，`max_new_tokens=12`，字母精确匹配；**正确 / 错误 / 不可解析三态分开计**，不合并；无 LLM judge。区间用 Wilson 95%。

## 4. 结果

| 运行 | 对话:考题 | 步数 | s/步 | 训练墙钟 s | 峰值 alloc / reserved MiB | 首→末 loss | 准确率 | Wilson 95% CI | 错答 | 不可解析 | 平均生成 token |
|---|---|---|---|---|---|---|---|---|---|---|---|
| base（无 SFT） | - | - | - | - | - | - | **73.0%** | [63.6%, 80.7%] | 27 | 0 | 8.48 |
| `r100_0` | 1200:0 | 150 | 3.88 | 581.2 | 6421 / 11068 | 2.666 → 2.389 | **69.0%** | [59.4%, 77.2%] | 24 | 7 | 8.28 |
| `r70_30` | 840:360 | 150 | 2.78 | 417.4 | 6439 / 9554 | 1.946 → 1.694 | **74.0%** | [64.6%, 81.6%] | 26 | 0 | 2.0 |
| `r30_70` | 360:840 | 150 | 2.90 | 434.7 | 6456 / 9992 | 1.060 → 0.997 | **71.0%** | [61.5%, 79.0%] | 29 | 0 | 2.0 |
| `r0_100` | 0:1200 | 150 | 2.81 | 421.6 | 4721 / 7012 | 0.434 → 0.325 | **70.0%** | [60.4%, 78.1%] | 30 | 0 | 2.0 |

编码阶段被丢弃的超长样本：`r100_0` 51 条（1149 行可训练）、`r70_30` 38 条（1162）、`r30_70` 26 条（1174）、`r0_100` 0 条（1200）。评测吞吐 0.45 s/题（对话档）vs 0.11–0.12 s/题（含考题档），评测峰值显存 2978 MiB（基座）/ 3128 MiB（挂 adapter）。机器可读版本：`out/summary.json`、`out/ablation.csv`、`out/ablation.md`、`out/resources.md`；逐题记录在 `out/eval/eval_*_items.jsonl`。

## 5. 结论

1. **配比之间没有可区分的准确率差异。** 最大值 `r70_30` 74% 与最小值 `r100_0` 69% 相差 5 个百分点，而 n=100 时 Wilson 半宽约 ±8–9 个百分点，五组区间两两重叠。按"区间重叠的差值不可作为结论"的口径，这里只能说"看不出差异"，不能说"70:30 最优"。
2. **微调没有提高选择题准确率。** 未微调的基座 73%，四档 adapter 落在 69%–74%，全部与基座区间重叠。1.5B 基座在 CMExam 这类题上已接近其上限，1,200 行 × 1 epoch 的 LoRA 不足以撬动知识。
3. **配比真正改变的是格式与成本，这一点可观测且方向一致。** 纯对话档 `r100_0` 保留长答语风格：36/100 走"答案："标记解析、平均生成 8.28 token、**7 题不可解析**（例如输出 `答案是：高位肠梗阻较低位肠梗阻`，没有字母）。凡是混入考题数据的档位都塌缩成"只输出一个字母"（2.0 token，100% first-letter 解析，0 题不可解析）。这是 SFT 语料格式对输出分布的作用，不是知识变化。
4. **训练成本由格式长度驱动，不由配比驱动。** `r100_0` 的 reserved 峰值 11,068 MiB（离 11,263 MiB 只差 195 MiB，正是最容易触发 WDDM 溢出的那一档）与 581 s 墙钟；`r0_100` 只要 7,012 MiB / 422 s。对话样本更长 → 序列更长 → 显存和时间都更高，与"数据里医疗内容占比"无关。
5. 因此这个实验支撑的说法是：**"我在消费级显卡上做了一套带数据登记、泄漏剔除、prompt 屏蔽校验和确定性评测的中文医疗指令消融流程，并给出了负结论与格式效应"**；不支撑"我通过调配数据比例把医学选择题准确率提升了 X 个百分点"。

## 6. 限制（不可回避）

1. 对话语料是 4.7% 前缀样本，配比结论只在该前缀内成立。
2. n=100 → 半宽 ±约 10 个百分点；要做配比排序至少需要 10× 的题量或多 seed 重复。
3. 单一 seed、单 epoch、无重复：模型间的差值无法与随机种子效应分离。
4. 训练目标与评测目标同构（都是"读题→出字母"），所以"混考题→输出更规范"是必然结果，不能读成医学能力提升。
5. `max_len=512` 丢弃了最长的那部分对话样本（`r100_0` 丢 51 条），长度截断对不同档位的影响不等。
6. 五选项题随机基线 20%，字母 D 在评测集正确项里占 26%，模型若有字母偏好会被这个先验放大（本评测未做去偏处理）。
7. fp16 + 1.5B + 11 GB 的绝对值只对本机成立，不能外推到其他硬件。
8. CMExam 数据许可与用途限制见 `docs/DATASHEET.md`，本仓库未重新分发原始数据。

## 7. 复现

```bash
bash setup_env.sh                       # 本机 .venv（torch 2.6.0+cu124 / peft / datasets / datasketch）
python scripts/01_fetch_data.py         # 取数 + 字节/sha256 登记（含前缀抽样声明）
python scripts/02_clean_dedup.py        # 清洗漏斗 + MinHash 去重 + 泄漏剔除
python scripts/03_make_ratios.py        # 四档配比（seed 20261014）
python scripts/03c_fetch_weights.py     # 分片下载基座权重并校验 sha256（需 HF_HUB_DISABLE_XET=1）
bash run_stage4b.sh                     # 训练缺失配比 + 全部评测（含显存闸门，见 §8）
python scripts/06_collect_results.py    # 汇总 out/summary.json / ablation.csv / ablation.md / resources.md
```

单档可控重跑：`python scripts/04_train_lora.py --config r70_30 --steps 150 --accum 8 --max_len 512 --slow_step_abort 25`，评测 `python scripts/05_eval.py --adapter models/lora_r70_30`（或 `base`）。

## 8. 过程中的三次事故与处理（如实记录）

1. **两个训练循环抢同一张卡。** 用后台任务停掉一个循环时，它的 python 子进程和 bash 循环都还活着，于是两组 150-step 训练并发 → 单步时间飙到 154 s，`r30_70`、`r0_100` 被新加的慢步保护中止（rc=3）。处理：确认 `nvidia-smi` 显存回落到 1.3 GB 才重启，并在 `run_stage4b.sh` / `run_stage4c.sh` 开头加硬闸门——启动前显存占用 > 4000 MiB 直接 `exit 9`。教训：`TaskStop` 杀不掉子孙进程，重启前必须查显存和日志。
2. **事件日志被覆盖导致 r70_30 的训练数字丢失。** 18:24 那轮跑出了有效的 150-step adapter，但 18:49 的重启把 `out/train_r70_30.jsonl` 覆盖成只到 step 12 的残档，stdout 又因为块缓冲在中止时整段丢失。处理：先把有效 adapter 备份到 `models/_backup_r70_30_1824`、残档留作 `out/train_r70_30_partial_1856.jsonl`，再用 `run_stage4c.sh` 完整重训一次（417 s，done 事件齐全），并把 `06_collect_results.py` 改成会校验 `steps_logged == 计划步数`，日志不完整就标 `train_log_complete=false` 而不是静默出数。同时把重跑的 python 改成 `-u`。
3. **半成品 adapter 差点被当成正式结果评测。** 早期 smoke 测试（2–4 步）留下的 `models/lora_r30_70`、`models/lora_r0_100` 会被评测循环当成正式档读取。处理：跑评测前把它们移到 `models/_invalid_smoke/` 隔离，只保留四份 150-step 产物。

此外还有两处早期失败已修：镜像的 Xet CAS 返回 401（禁用 Xet）、HEAD 探测触发 429（改用 API 报的字节数 + 1 KiB range 探测 + 6 路错峰分片，238.7 s 全部一次成功）。

## 9. 产物清单

| 路径 | 内容 |
|---|---|
| `data/raw_manifest.json`、`data/stats/quality_report.json` | 原始取数与清洗漏斗计数 |
| `data/clean/dialog_zh.jsonl`（58,787）、`mcq_eval_100.jsonl`、`mcq_train_pool.jsonl`（46,421） | 清洗后语料 |
| `data/ratios/*.jsonl` + `ratios_manifest.json` | 四档配比与 seed |
| `models/Qwen2.5-1.5B-Instruct/` | 基座权重（sha256 已校验） |
| `models/lora_*`（4 份） | 150-step LoRA adapter |
| `out/train_<cfg>.jsonl` | 逐事件：setup / mask_check / loss_check / encode / step / done |
| `out/eval/eval_*.json` + `_items.jsonl` | 五组评测汇总与逐题记录 |
| `out/summary.json`、`ablation.csv`、`ablation.md`、`resources.md` | 消融表与资源账 |
| `docs/DATASHEET.md` | 来源登记、漏斗、抽样偏差、许可与限制 |
| `logs/` | 每档训练与评测的原始日志、`stage4.log` 运行序列 |
