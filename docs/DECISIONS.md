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
    `base_model.resolve()` 在训练前复核实际字节（不符即拒跑；评测只报目录名，不复核）。
17. **镜像 LFS 全线 502 时改走 ModelScope，判据不变**：下载脚本对 ModelScope 的分片
    Ranged GET 产物算 sha256，与 HF API 报的 LFS 摘要逐位比对（其 CDN 对象路径本身就是
    该摘要的分段 `lfs-objects/fd/f7/56fa…`），不吻合就删除并非零退出；换的只是下载通道，
    不是换一份来历不明的权重。脚本留在仓库外（`E:/Desktop/resume/ms_fetch.py`），
    仓库内的可复现入口仍是 `03c_fetch_weights.py`＋登记文件里的摘要。

