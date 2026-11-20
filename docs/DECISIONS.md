# Decision log

本仓库关键口径与协议决定的记录。每条注明**仓内依据**（文件/代码位置）与可核对的日期
（= git 提交日期，`git log` 可逐条对上）；只收当时已留下书面或代码痕迹的决定。

## 2026-10-15（`9cb0032` 阶段 4 收口）

1. **11 GB 显存约束决定训练实现的四道处理**：`max_len=512`、只对答案位置算
   cross-entropy（切到监督后缀再过 `lm_head`，避免物化 T×151,936 的 fp32 logits）、
   每步 `empty_cache()` + `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128`、
   连续两步 >25 s 主动中止（宁可失败也不 thrash，WDDM 溢出表现为 s/步 4→150 s
   且无 throttle 标志）（`README.md` §1）。
2. **训练目标与评测口径严格一致**：训练只监督答案 token，评测是字母精确匹配
   （贪心解码、`max_new_tokens=12`）——不引入训练/评测错位作为解释变量（`README.md` §3）。
3. **prompt 屏蔽逐条核验**：拼接前缀 token 必须与模板单独编码结果完全一致才屏蔽，
   探针 200/200 通过才继续（`mask_check` 事件；`README.md` §3）。
4. **自写 response-position CE 先与 HF 官方 loss 对齐校验（绝对差 0.0）再用于训练**
   （`loss_check` 事件，四档都是；`README.md` §3）。
5. **评测三态分开计**（正确/错误/不可解析），不合并、无 LLM judge；区间用 Wilson 95%
   （`README.md` §3）。
6. **泄漏剔除在配比构造之前**：train↔eval 按归一化题干比对，从训练池剔除 5 条
   （`README.md` §2；`scripts/02_clean_dedup.py`）。
7. **判读口径**："区间重叠的差值不可作为结论"⇒ 四档配比只写"看不出差异"，
   不写"70:30 最优"；这条直接约束了引用措辞（`README.md` §5 结论 1、
   §引用节"引用时请保留「四档配比不可区分」这一限定"）。

## 2026-11-03（`b000f04` 阶段 5）

8. **扩题到 500 前先做哈希泄漏比对**：400 道新题排除 172 道与训练池重叠者，
   最终 500 题与训练池重叠为 0（`scripts/03d_build_eval_500.py`、
   `data/stats/eval500_report.json`）。原 100 题原样保留，新旧可对照。
9. **用故意泄漏的阳性对照与打乱标签对照来证明评测回路有效**：
   `poscontrol_evaltrain`（背答案）99.2%、`poscontrol_shuffled` 接近随机
   （`scripts/03e_build_positive_control.py`；`README.md` §4 对照表）。
10. **负结论补统计而不改口径**：4 档 × 3 seed × 150 步重跑，配对 McNemar +
    95% t 区间，跨 0 即维持"不能写成提升准确率"（`README.md` §多种子复现、结论条）。

## 2026-11-18（`0d722f5` / `2c03165` / `aee5d03`）

11. **CI 泄漏测试不得依赖不可再分发的训练池**：改为 `02b_export_pool_hashes.py`
    导出题干哈希（46,421 行 → 45,640 唯一），评测再生成门在无池环境下可跑
    （`2c03165`；`README.md` §复现注释）。
12. **drift gate 校验数字一致性而非二进制**：matplotlib 版本会改写 png 字节，
    故图产物本身不进 gate（`README.md` §CI 说明段）。
13. **下载路线固定为 hf-mirror + 分片 Ranged GET，且必须 `HF_HUB_DISABLE_XET=1`**
    （镜像 Xet CAS 返回 401）并按 `Retry-After` 退避；留痕 `out/model_download.json`
    （`README.md` §1）。

## 2026-11-20（规模复核准备：同一配方换 0.5B 基座）

14. **换基座只换基座，其余全部沿用已发布值**：基座由 `scripts/base_model.py` 一处解析
    （`BASE_MODEL_DIR` 选目录），训练（`04_train_lora.py`）与评测（`05_eval.py`）共用同一
    解释器，杜绝「在这个基座上训、在那个基座上评」；步数、accum、`max_len`、seed 列表、
    四档配比、LoRA 超参、评测文件一律不动。
15. **规模复核的产物必须与已发布产物不同名**：所有输出走 `--tag`／`TAG_PREFIX` 命名空间
    （adapter 目录、训练与评测日志、三份汇总文件），前缀为空时逐字节等于发布版；
    `tests/test_scale_check.py` 断言两套文件名互不相交、且发布汇总引用的评测文件仍在，
    CI 的 drift gate 因此不受影响。
16. **权重同一性以 sha256 判据落实，不靠「文件名一样」假设**：`03c_fetch_weights.py` 把
    HF API `/api/models/<repo>?blobs=true` 报的 LFS sha256 记进登记文件，
    `base_model.resolve()` 在训练前复核实际字节（不符即拒跑）。
    *同日补（结果批次）*：`05_eval.py` 也走同一个解析器（`resolve(verify=False)`，只取目录名、
    不复核字节），并把实际使用的基座名写进评测 JSON 的 `base_model` 字段。已发布的 2026-11-03
    那批评测 JSON 早于这个字段，`10_scale_compare.py` 对缺省值回落到 `base_model.DEFAULT_BASE`；
    drift gate 校验的是汇总再生，不受这个新字段影响。字节复核仍只放在训练侧——评测读的是
    已经训好的 adapter，权重错了训练那步就先炸。
17. **镜像 LFS 全线 502 时改走 ModelScope，判据不变**：下载脚本对 ModelScope 的分片
    Ranged GET 产物算 sha256，与 HF API 报的 LFS 摘要逐位比对（其 CDN 对象路径本身就是
    该摘要的分段 `lfs-objects/fd/f7/56fa…`），不吻合就删除并非零退出；换的只是下载通道，
    不是换一份来历不明的权重。兜底脚本留在仓库外（本机 `ms_fetch.py`，不在版本控制里），
    仓库内的可复现入口仍是 `03c_fetch_weights.py`＋登记文件里的摘要。
    *同日补（结果批次）*：落位后跑 `03c_fetch_weights.py` 做登记与复核，登记文件
    `out/model_download_Qwen2.5-0.5B-Instruct.json` 里 `model.safetensors` 那条记的是
    `skipped: already complete on disk`＋从 HF API 取到的 `sha256_upstream`，
    也就是它只负责「字节与上游摘要一致」这一步，不冒充下载来源；
    `endpoint` 字段仍是 03c 自己的镜像地址（小文件与 API 走的通道），不是权重的来历
    （`03c_fetch_weights.py` 写登记文件处留有同一条注释）。
    仓内至今没有 ModelScope 兜底的实现（只有这条决定和那条注释），要在无手工脚本的情况下复现取数，
    得先补那条通道。

## 2026-11-20（`run_multiseed_500.sh` 0.5B 批次跑完后的判读口径）

18. **档间与档-vs-base 的差异改用题层面配对 bootstrap 报告**：重抽单位是 500 道题，
    同一档的三个 seed 先按题平均（`scripts/10_scale_compare.py`，B=2000、`random.Random(42)`）。
    选它而不是继续只报 McNemar 的 p，是因为规模复核要回答的是「效应量可能有多大」，
    需要区间而非二值显著性；配对（同一道题跨规模/跨档同题相减）消掉题目难度这个混杂。
    **代价如实声明**：三个 seed 是被固定的，区间含题目抽样误差但**不含训练随机性**——
    这条写在产物自己的 `boot.caveat` 字段里，`tests/test_scale_check.py` 断言它存在，
    README §7c 与 `scale_compare_500.md` §1 也都各自重复一遍。
19. **不可解析同时报「原始口径」和「仅可解析口径」，并单列撞满 token 上限的题数**：
    纯对话档在 1.5B 上 −7.3 pt 里绝大部分是「答案写成了判分器读不出的样子」
    （3 seed 共 158 题不可解析，其中 93 题是生成用满 `max_new_tokens=12` 被截断的），
    仅可解析口径只剩 −0.6 pt。若不把判分口径与知识变化分开，就会把 1.5B 的长答语风格
    误读成配比的性质。同时保留原始口径为对外口径——用户确实拿不到答案，
    「仅可解析」那一列只用于机制归因（`scale_compare_500.md` §3）。
20. **最好档−最差档只作描述，不作检验**：四档里取最大减最小是挑过的统计量，
    极值选择本身会放大它，所以那个区间不能读成「配比之间有差异」的检验；
    档间差异的判据仍是 §2 那 6 对预先固定的两两比较（1.5B 排除 0 的 3 对全部涉及
    `r100_0`，0.5B 为 0 对）。另注：对 base 的 12 个精确 McNemar p 与 12 对 bootstrap
    区间都未做多重比较校正，单独引用任何一个 p 时必须带上这条（`scale_compare_500.md` §2）。
21. **规模复核的结论只回答「同一套配方换个参数量还成不成立」**：本次只换基座权重文件，
    数据、配比、seed、训练预算、评测集、判分全部沿用发布值，因此它能排除的是
    「某档配比带来 ≥5 pt 的稳定提升」在 0.5B 上也存在；它**不能**分离预训练语料与数据量
    的贡献（两档 base 相差 24 pt 本身就说明可动空间不同），也不能把单 epoch、短预算下
    「测不出差异」升级成「无差异」（同第 10 条的边界）。

## 2026-11-20（用干净克隆模拟 CI 之后）

22. **汇总再生成固定在 Python 3.11，因为已发布产物就是 3.11 的浮点求和结果**：在干净克隆里用
    3.13 跑 `06_collect_results.py`，`out/summary.json` 的 `r30_70.loss_last10` 从 0.9968 变成
    0.9969（`ablation.csv`/`ablation.md` 同处跟着变），另外 9 份进 gate 的产物（含两份 multiseed
    汇总、`positive_control_summary.*`、三份 `scale_compare_500.*`）逐字节不变。原因是 3.12 起
    `sum()` 对 float 改用 Neumaier 补偿求和，而这一档末 10 步的均值恰好落在 0.99685 这个
    四舍五入临界上：3.11 的求和结果是比精确值低 1 ulp 的 `0.9968499999999999`，`round(…, 4)`
    给 0.9968；3.13 的补偿求和给出正好 `9.9685/10`，`round` 给 0.9969。差的只是最后一位显示数字，
    不动任何区间、任何判读。**处理**：不改已发布产物（v1.0 已冻结、且改写它要连带改 README 表格），
    而是把 CI 的 regenerate 作业显式钉在 3.11 并在 `.github/workflows/ci.yml` 里写明原因；
    数据完整性测试仍在 3.11/3.13 双版本跑（那层不产生字节比较）。接手的人如果在本机用 3.12+
    跑 drift gate 看到这一处 diff，应当认定是解释器版本，而不是数据被改动。
    另：口径校验层（第 4 层）唯一依赖不入库文件 `data/clean/dialog_zh.jsonl` 的那条断言改为
    本机执行、CI 跳过并回落到漏斗计数自洽（`final_kept == after_filters − minhash_duplicates`），
    断言数 23→24。

