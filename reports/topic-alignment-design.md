# 跨批次稳定的主题编号 —— 设计文档

适用源码树：`/home/wll/uid54`（gensim 4.4.0，基线提交 `c98fe6a9`）
状态：实现 + 测试 + 验收报告
默认行为：**关闭**。不传 `topic_alignment` 时，训练与主题输出与现状逐字节一致。

---

## 0. 需求 → 设计映射

| # | 需求 | 落点 | 测试 / 报告 |
|---|------|------|-------------|
| 一.1 | 两批语料相近主题同号 | `TopicAligner._resolve` 内容相似度匹配；`local_to_uid` | `test_continuation_keeps_the_same_identifier`、`test_two_batches_share_identifiers_and_drift` |
| 一.2 | 旧文档主题分布可重映射、总量守恒 | `AlignmentResult.remap_document_topics` + `document_remap` | `test_document_remap_conserves_mass`、`test_merge_keeps_primary_and_records_destination` |
| 一.3 | 跨批次漂移分数、阈值可配 | `drift_score`（JS 距离），`AlignmentConfig.drift_threshold` | `TestSignatureAndDrift`、报告 `identifier_mapping[].drift` |
| 一.4 | 合并保留主编号、副编号记录去向 | `merges`，`TopicLedger.apply` 写 `merged_into` | `test_merge_keeps_primary_and_records_destination` |
| 一.5 | 分裂：父编号归占比大的一支 | `_resolve` step 4 按 `coverage_old` 选主支 | `test_split_parent_follows_larger_share`、`test_split_parent_moves_when_larger_share_is_second` |
| 一.6 | 消失主题有明确去向 | `_apply_disappearances`：折叠到最相似存活主题，或标记无去向 | `test_disappearance_has_explicit_destination` |
| 一.7 | 增量更新后编号稳定 | `LdaModel.update` → `align_topics`；账本 `next_uid` 单调 | `test_incremental_update_keeps_numbers_stable`、报告多轮验证 |
| 一.8 | 新/旧判定有依据、可人工复核 | 每条匹配带 `similarity/shared_terms/coverage/evidence`；报告含词表 | `to_report()`、`reports/topic-alignment-report.md` |
| 一.9 | 主题数上限：超出按相近并入并留痕 | `_apply_max_topics` + `absorptions` | `test_max_topics_absorbs_and_leaves_trace`、报告 `### max_topics` |
| 一.10 | 关闭时行为一致 | `AlignmentConfig.enabled=False` 默认；`align_topics` 返回 `None` | `TestDisabledBehaviour`、`test_existing_topic_output_unchanged_when_disabled` |
| 二.1 | 编号仅来自内容匹配 | `TopicSignature` 按 `(-prob, term_id)` 排序；贪心按相似度，与训练/初始化顺序无关 | `test_cosine_is_content_only`、`test_continuation_keeps_the_same_identifier`（顺序互换仍同号） |
| 二.2 | 对齐/漂移/文档映射同源 | 三者全部由同一个 `AlignmentResult` 派生 | `remap_document_topics` 直接用 `document_remap` |
| 二.3 | 账本落盘、跨进程一致 | JSON + 版本号 + 内容指纹；原子写 | `TestCrossProcessConsistency`、报告 `## Cross-process ledger read` |
| 二.4 | 现有查询/分布结构不变 | 未改动 `show_topics/get_document_topics/get_topics` 返回类型 | `test_existing_topic_output_unchanged_when_disabled`、原有 `test_ldamodel.py` 49 项 |
| 二.5 | 报告只含编号、词、计数 | `to_report()` 不含任何文档文本 | `reports/topic-alignment-report.md` |
| 三.1 | 账本缺失/主题数对不上要报批次与差异 | `_check_previous`、`LedgerError` 带批次与计数 | `test_missing_ledger_reports_path`、`test_inconsistent_ledger_names_batch_and_counts` |
| 三.2 | 多对一/一对多冲突按可解释规则裁决 | `_resolve` step 2/4 确定性贪心；`reason` 字段 | `test_conflict_resolution_is_recorded` |
| 三.3 | 漂移不可比时明确标注，不给零分 | `drift_score` → `(None, "not_comparable")` | `test_drift_not_comparable_never_zero` |
| 三.4 | 写账本失败不能看起来成功 | `commit` 捕获 `LedgerWriteError` 并回滚内存账本后重抛 | `test_write_failure_raises_and_rolls_back` |
| 四.1 | 增量接口/批大小/迭代语义不变 | 仅在 `update()` 末尾追加一次对齐；参数原样透传 | 原有 `test_ldamodel.py` |
| 四.2 | 词典裁剪与词表映射不变 | 对齐只读 `model.get_topics()/id2word`，不碰 `Dictionary` | 原有 `test_corpora_dictionary.py` |
| 四.3 | 单批训练耗时/内存不翻倍 | 对齐在训练后一次 O(K²·V) 计算，实测 10 主题 8.5 ms / 30 主题 43 ms | 见 §6 |
| 四.4 | 不引入新依赖、离线可跑 | 仅 numpy + 标准库；账本为 JSON | 全文 |

---

## 1. 现状与约束

- 主题编号是**位置编号**：`LdaModel.get_topics()` 返回 `(num_topics, vocab)` 矩阵，行号即主题号；`show_topics`/`get_document_topics` 原样返回行号。重训后行序由随机初始化与语料顺序决定，因此编号不可跨批次解释。
- `update()` 是在线变分贝叶斯，**不改变行数**；这意味着增量更新时行号不变，但一旦用新语料**从零重训**（每月一份报告通常如此），行号就整体重排。
- 模型 `save()` 用 pickle，`save/load` 已能持久化任意新增属性；旧夹具（如 `lda_3_0_1_model`）没有新属性，读取时必须容错（`getattr`）。
- 仓库不允许新增强依赖；因此账本用 JSON，相似度用 numpy 实现。

## 2. 数据结构

```
TopicSignature   一份主题内容指纹：前 topn 个词的概率分布（归一化）+ 批次 + 本地下标
TopicIdentity    一个稳定编号（T0007）+ label + origin_batch + revision + missing + history + signature
TopicMatch       一次旧→新对应：相似度、共享词数、两侧覆盖率、漂移分数与状态、证据
AlignmentResult  单次对齐的完整、自洽结果（matches/births/merges/splits/revivals/
                 disappearances/retirements/absorptions + 三套映射）
TopicLedger      版本化 JSON 账本：batches[]（每批的身份快照）+ next_uid + fingerprint
```

**同源性**：`AlignmentResult.local_to_uid`（编号）、`matches[].drift`（漂移）、
`document_remap`（文档映射）都在 `_resolve`/`_emit` 中一次性产生，不存在第二套计算。
`remap_document_topics()` 只读 `document_remap`。

## 3. 编号算法（内容匹配，与顺序无关）

记上一批主题集合为 P，本批为 C。

1. **候选对**：对每个 `(p, c)` 计算内容相似度 `cos`（前 `topn` 词分布），`cos ≥ similarity_threshold` 才保留。
2. **一对一核心匹配**：所有候选对按 `(-cos, -shared_terms, old_index, new_index)` 排序，贪心取互不冲突的对。排序键里**没有**训练顺序或初始化顺序，因此结果只取决于内容；下标只用于打破完全相同的平局。
3. **休眠复活**：更早批次消失、宽限期未满的身份，可认领无人认领的当前主题；与活跃匹配冲突时活跃者优先。
4. **分裂**：某旧主题若还匹配到未被认领的当前主题，把该旧主题全部候选按 `coverage_old`（旧主题质量落在共享词上的比例）排序，**占比最大的分支保留父编号**，其余分支各领新号。`coverage` 是内容量，不是下标。
5. **合并**：某旧主题没有核心匹配、但其最相近的当前主题已被更强的主张占用时，记为合并；主编号归强者，副编号在账本 `merged_into` 记录去向，其文档质量并入主主题。
6. **消失**：既无匹配也无合并的旧主题，折叠到与其最相似的存活主题（只用于文档重映射），并记录 `reason`；从未找到存活主题时明确记录「无去向」，绝不静默删除。
7. **新生**：仍未分配的当前主题领新号。

`similarity_threshold`、`drift_threshold`、`min_shared_terms`、`topn`、`max_topics`、`retired_grace` 全部来自 `AlignmentConfig`。

## 4. 漂移分数

对同一编号的前后两份签名，取共享词集合，分别归一化后用 Jensen–Shannon 距离，再除以 `sqrt(2)` 归一到 `[0, 1]`。
**共享词少于 `min_shared_terms` 时返回 `(None, "not_comparable")`**，报告里显示为 `not_comparable`，而不是 0 分。
分数超过 `drift_threshold` 时状态标记为 `threshold_exceeded`，但**不改变匹配本身**——匹配由内容相似度决定。

## 5. 账本

```json
{
  "format": "gensim-topic-ledger",
  "version": 1,
  "next_uid": 8,
  "fingerprint": "…sha256…",
  "batches": [
    {
      "batch_id": "2026-11",
      "previous_batch_id": "2026-10",
      "assigned_at": 1795...,
      "num_topics": 5,
      "config": {…},
      "identities": {
        "T0002": {"topic_uid": "T0002", "label": "…", "origin_batch": "2026-10",
                  "local_index": 0, "revision": 2, "missing": 0, "history": […],
                  "signature": {"batch_id": "2026-11", "local_index": 0, "words": [["word", 0.0123], …]}}
      },
      "result": {…完整 AlignmentResult 报告…}
    }
  ]
}
```

- **原子写**：临时文件 → `fsync` → `os.replace`。任何失败抛 `LedgerWriteError`，`commit()` 回滚内存账本后重抛，调用方不可能把失败当成功。
- **版本校验**：`format`/`version` 不符直接 `LedgerError`，消息带路径。
- **跨进程**：纯 JSON 读，无需 gensim；`fingerprint` 对内容做 SHA-256，便于不同进程核对读到同一份历史。
- **一致性自检**：`verify()` 检查批次链、每批 `num_topics` 与活跃身份数是否一致；不一致时 `_check_previous` 报出具体批次与数量差异。

## 6. 对现有行为的影响

- 关闭（默认）：`LdaModel.__init__` 只多存两个属性；`update()` 末尾的 `getattr` 判断为假，直接跳过。`test_ldamodel.py` 49 项在改动前后均通过。
- 开启：在 `update()` 全部训练动作完成后调用一次 `align_topics()`。对齐只读 `get_topics()/id2word`，不写模型状态，不改变主题输出。
- 实测对齐开销（Lee 语料，冻结后单次对齐 + 落盘）：10 主题 8.5 ms，30 主题 42.6 ms；同条件 10 主题 10-pass 训练 1.59 s。开销与训练不在同一量级，不会翻倍。

## 7. 验收

见 `reports/topic-alignment-report.md` / `.json`：

1. 两批分层语料 → 5/5 主题同号，列出对应词表与相似度、漂移。
2. 合并、分裂、消失、复活、`max_topics` 五种场景的编号处置与理由。
3. 文档主题分布重映射前后的质量守恒（`total_conserved=true`）。
4. 独立解释器读账本，编号与写入进程一致，`verify()` 无问题。
5. 原有 `test_ldamodel.py` 49 项通过；新增 `test_topic_alignment.py` 32 项通过。

## 8. 已知边界（诚实声明）

- 相似度阈值是**内容阈值**，不是语义阈值。当语料本身主题退化（例如大量停用词主导，所有主题都由 `will/been/said` 这类词构成）时，任何基于词分布的匹配都会把不同主题判为相近；此时应提高 `similarity_threshold` 或先做停用词/词频裁剪。报告中刻意保留原始词表与相似度，便于人工复核这类情况。
- `max_topics` 的并入依据是「与另一存活主题的内容相似度最高」，不是文档数占比；这样对齐层不需要读语料，代价是无法区分「主题小但与众不同」与「主题大但高度重复」。
- 对齐不改变模型的行顺序，也不重排 `get_topics()`；稳定编号以 `topic_uid` 与 `local_to_uid` 的形式提供，二者在报告与账本中都出现。
