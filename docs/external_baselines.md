# 外部对照：已发表数字（全部照抄原文并核过表号）

目的：给本仓库的负结论一个文献坐标。规则：只抄原文数值并标明表号；每行注明口径；
**绝对值与本仓库结果不可直接比**的原因集中写在 §5。查证方式：2026-11-18 逐条
WebFetch 原文 HTML 核对表内文本（题录红线：不凭记忆转写）。

## 1. 本仓库自己的数字（作为对照行的锚）

| 运行 | 数字 | 口径 |
|---|---|---|
| base（Qwen2.5-1.5B-Instruct，无 SFT） | **65.8%**（329/500，单 seed） | 自建 500 题（CMExam `fzkuji/CMExam` test 抽样、剔除 172 道训练池重叠题）、首字母精确匹配、贪心解码、`max_new_tokens=12` |
| 最优配比 `r70_30` | 68.2%（3-seed 均值，对 base +2.4 pt，McNemar p ≥ 0.089） | 同上 |
| 阳性对照（故意泄漏） | 99.2%；标签打乱 17.6% | 同上；随机字母匹配基线 ≈ 20%（README 限制 5） |

## 2. CMExam 原论文（题源相同，集合与判分不同）

Liu et al., *Benchmarking Large Language Models on CMExam*, arXiv:2306.03030v3, Table 3（已核原文）。
清洗后 68,119 题、8:1:1（test ≈ 6,812），≤5 选项、65,950 单选/2,169 多选；判分为生成答案的精确匹配。

| 数值 | 模型 | 口径 | 出处 |
|---|---|---|---|
| 61.6 ± 0.1 % | GPT-4（zero-shot） | CMExam 全体 test，生成判分 | Table 3 |
| 46.4 ± 0.6 % | GPT-3.5-turbo | 同上 | Table 3 |
| 45.3 ± 1.4 % | ChatGLM-CMExam（6B，CMExam train 微调） | 同上；**在本仓库训练池（46,421 条）之外另微调** | Table 3 |
| 71.6 % | 人类被试加权平均 | — | Table 3 |
| 3.1 ± 0.2 % | Random | **生成/精确匹配口径**，与字母匹配的 ~20% 不可互比 | Table 3 |

## 3. HuatuoGPT-II（7B/13B 医疗 SFT 的同题源参照）

Chen et al., arXiv:2311.09774v2, Table 1（zero-shot、单一简单 prompt，已核原文列序）。

| 数值 | 模型 | 口径 | 出处 |
|---|---|---|---|
| CMExam 65.81 / CMB 60.39 / MedQA(EN) 41.13 | HuatuoGPT-II-7B（Baichuan2-7B 底座、5.25M 文档预训练语料注入） | 各基准**全体**测试集 | Table 1 |
| CMExam 68.98 / CMB 63.34 / MedQA(EN) 45.68 | HuatuoGPT-II-13B | 同上 | Table 1 |
| CMExam 53.33 | Qwen-7B-Chat（通用） | 同上 | Table 1 |
| CMExam 63.57 | Qwen-14B-Chat（通用） | 同上 | Table 1 |
| CMExam 46.51 | ChatGPT | 同上 | Table 1 |

防污染量级参照（同文 Table 2，2023-10-21 国家执业药师资格考试，晚于其数据截止 10-07；及格线口径约 60%）：
GPT-4 48.1、HuatuoGPT-II-13B 52.3、-7B 47.6、ChatGPT 36.2。

## 4. CMB-Exam 与 DISC-MedLLM 自建 MC 基准（多选/抽样口径，只能看量级）

Wang et al., *CMB*, arXiv:2308.08833v2, Table 6/7（zero-shot 平均列，已核原文）：Test = 28 子类 × 400 = 11,200 题，4–6 选项、**一个或多个正确答案**、完全匹配；Yi-6B-Chat 65.87、GPT-4 59.46、HuatuoGPT-II-34B 76.82、-13B 62.75、-7B 59.00、DISC-MedLLM-13B 39.76、ChatGLM3-6B 40.00。

Bao et al., *DISC-MedLLM*, arXiv:2308.14346v1, Table 3（已核原文）：MLEC-QA 抽 test 集 10% 共 1,362 题（5 类）+ NEEP 西医综合 306 取 2020–2021 共 270 题；Average 列：GPT-3.5 few-shot 49.64 / zero-shot 40.22；DISC-MedLLM few-shot 39.79 / zero-shot 37.56；Baichuan-13B-Chat few 29.91 / zero 38.66；HuatuoGPT-13B few 28.87 / zero 24.34。题源与本仓库同为 CNMLE 执医题。

## 5. 口径差异清单（为什么上表不能与本仓库直接比）

1. **子集不同**：本仓库为剔除泄漏后的 100/500 题子集，且旧 100 题已被实测证明偏易（base 73.0% vs 500 题 65.8%）；外部数字均为全体测试集。
2. **判分不同**：本仓库是首字母精确匹配（三态分开计），随机基线 ≈ 20%；CMExam 原文 Random 3.1% 是生成答案精确匹配口径。
3. **题型不同**：CMB-Exam 含多选、完全匹配计分；本仓库子集为单选。
4. **模型侧不同**：外部多数是 7B–34B、全量 SFT 或继续预训练；本仓库是 1.5B 底座 + 1,200 行 × 1 epoch × 150 步 LoRA。
5. **prompt 不同**：HuatuoGPT-II 用单一简单 prompt zero-shot，本仓库用固定模板贪心解码。

因此本节数字只作**量级与方向**参照：base 65.8%（字母匹配子集）与文献中 7B 级医疗模型在 CMExam 全量上的 53–69% 处于同一量级，
与本仓库结论 2（"1.5B 基座在此类题上已接近其上限，小规模 LoRA 撬不动知识"）方向一致；
这不能反过来当作"1.5B ≈ 7B 医疗模型"的证据——集合、判分、题型都不同。

## 6. 未核到项（如实声明）

- **Qwen2.5 技术报告（arXiv:2412.15115v2）没有 MedQA / CMMLU / 任何医疗基准**——全文检索命中数为 0；
  其 Table 10 只报 0.5B–1.5B 档通用/数理/代码基准（LiveCodeBench 14.8 / MMLU-Pro 32.4 / MATH 55.2 等）。
  因此本仓库不能引用"Qwen2.5 官方医疗数字"。
- 「MedQA-ZS」「MedDCA」两个基准名经 arXiv API 全文检索（`totalResults = 0`）**查不到公开出处**，
  故本文件不引用它们；中文执医选择题实际可核的公开基准为 MLEC-QA / CMExam / CMB-Exam（§2–§4）。
- 最接近的通用中文基座数字（非医疗，base 口径，只能当通用能力参照）：Qwen2 技术报告（arXiv:2407.10671v4）
  Table 5：CMMLU 5-shot，Qwen2-1.5B 70.3、Qwen2-0.5B 55.1。

## 7. 文献条目

- Liu et al. 2023, *Benchmarking Large Language Models on CMExam*, arXiv:2306.03030. https://arxiv.org/abs/2306.03030
- Chen et al. 2023, *HuatuoGPT-II, One-stage Training for Medical Adaption of LLMs*, arXiv:2311.09774. https://arxiv.org/abs/2311.09774
- Wang et al. 2023, *CMB: A Comprehensive Medical Benchmark in Chinese*, arXiv:2308.08833. https://arxiv.org/abs/2308.08833
- Bao et al. 2023, *DISC-MedLLM*, arXiv:2308.14346. https://arxiv.org/abs/2308.14346
- Li, Zhong, Chen 2021, *MLEC-QA*, EMNLP 2021, pp. 8862–8874. https://aclanthology.org/2021.emnlp-main.698/
- Qwen2 Technical Report, arXiv:2407.10671; Qwen2.5 Technical Report, arXiv:2412.15115.
