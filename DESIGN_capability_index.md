# MEMCRYST v1 — Capability Index + Skill Format Design

## What this answers

1. **B 路径核心机制**：skill 区分于 compressed memory 的 architectural anchor 是什么。
2. **O(N) cost**：用户提的成本论点怎么落地。
3. **Bidirectional feedback**：capability 怎么让 skill→memory ancestry 比当前更有信号。
4. **Live-Evo 兼容 + 可选 code**：format 走哪条路。

## Proposed skill schema

YAML frontmatter 在原来基础上加 `capability` 段。Body 维持 NL（procedure / when-to-use / anti-patterns）。可选 `script:` 指向 Python 文件做 callable hook，v1 PoC 不强制。

```yaml
---
name: clean-and-place
description: Clean an object and place it at a destination.
category: general

# NEW — capability signature for indexed retrieval
capability:
  verb_family: clean              # closed set: clean | heat | cool | examine | move | search | open | use
  target_type: object             # object | location | abstract
  affordance: sink                # closed set per environment, "none" if unused
  output: object_placed           # closed set: object_placed | object_examined | location_reached | object_processed
  preconditions:                  # tag list, not free NL
    - object_at_source
    - destination_reachable
  task_pattern: "clean <target_object> and place at <destination>"

# Required parameter schema
parameters:
  target_object: {type: object, required: true}
  source_location: {type: receptacle, required: true}
  destination: {type: receptacle, required: true}

# Optional v1.5 hook — Voyager / WebXSkill Grounded mode
script: null                      # path to skill.py if executable, else null
---

## When to use
...

## Procedure
1. go to <source_location>
2. take <target_object> from <source_location>
3. go to sink
4. clean <target_object> with sink
5. go to <destination>
6. put <target_object> in/on <destination>

## Anti patterns
...
```

## Why this format

### A. Live-Evo lineage 兼容

Live-Evo Experience entry 是 6 字段 NL（question / improvement / failure_reason / missed_information / category / weight）。我们的 memory bank 已经直接 fork 了这个 schema。

skill 走的是同一思路但**更结构化**：在 NL body 之上加一层 **closed-set tag**（`verb_family` / `affordance` / `output`）。Reviewer 可以一眼看到这是 Live-Evo episodic memory 的 **procedural compile**，结构层级高出一层。

这条比 MUSE / Voyager 路线更适合"experience-driven self-evolution"的故事：memory 是 atomic experience，skill 是 indexed capability artifact。

### B. 可选 code hook，不强制

`script` 字段 v1 PoC **保留接口但默认 null**。v1.5 / v2 时候：
- 真接 ALFWorld → 把 procedure NL 编译成 action 字符串序列写到 skill.py
- 对应 WebXSkill 的 Grounded mode（skill = callable atomic tool）
- v1 NL mode 对应 WebXSkill 的 Guided mode

这样格式上**预留了 WebXSkill 的 14.3pt 验证机制 + Voyager 的 callable 接口**，但 v1 不被它绑定。Reviewer 问"为啥不直接 code"，答案：因为 v1 验证的是 crystallization mechanism，不是 execution efficiency；code hook 是 v1.5 的 ablation 维度。

### C. 不全代码（不走 Voyager 全 code 路线）

NL procedure 必须留着，因为：
1. 我们整个 paper 的核心是 memory → skill crystallization，**让 LLM 写 NL 比让它写 Python 稳定得多**
2. NL procedure 是 reviewer 看得到的 interpretability anchor
3. 全代码的 skill 在 ALFWorld 之外不通用，会绑死环境

## Capability index 机制

### Retrieval pipeline（agent wake-time）

```
incoming task: "Examine the photo album using the side table lamp"
         │
         ▼
  ┌──────────────────────────────────────────────┐
  │  Stage 1. Task signature extraction          │
  │  - verb: "examine" (closed-set match)        │
  │  - target_type: "object"                     │
  │  - affordance hint: "lamp" (from task NL)    │
  │  - output: "object_examined"                 │
  │  cost: 1 LLM call OR rule-based parsing      │
  └────────────────────┬─────────────────────────┘
                       ▼
  ┌──────────────────────────────────────────────┐
  │  Stage 2. Capability index lookup            │
  │  - hash (verb, affordance, output) → bucket  │
  │  - returns ≤ K skills in O(1) per signature  │
  │  cost: O(K), K typically 1-5                 │
  └────────────────────┬─────────────────────────┘
                       ▼
  ┌──────────────────────────────────────────────┐
  │  Stage 3. Fine-grained rerank (optional)     │
  │  - task_pattern fuzzy match against ≤ K      │
  │  - cosine on description embeddings          │
  │  cost: O(K)                                  │
  └────────────────────┬─────────────────────────┘
                       ▼
                  top-1 or top-3 skill
```

### vs compressed memory baseline

| | compressed memory | MEMCRYST skill |
|---|---|---|
| 检索单位 | episodic summary | indexed capability |
| 检索方法 | embedding cosine top-K | hash(signature) → bucket + rerank |
| 复杂度 | O(N) similarity scan | **O(K) hash + O(K) rerank** |
| Cold start | 必须 N>0 才有 reuse | capability slot 即使为空也存在（可触发 "no skill, fallback to memory"） |
| 失败归因 | "this summary was retrieved, output failed" | "verb_family=clean + affordance=sink failed on `<target_object>` type X"，**axis-level 归因** |

这张表就是答你的 O(N) cost 论点 + 直接给 reviewer 看的 architectural differentiation。

## Bidirectional feedback 怎么对上

### Forward: memory → skill（已有 + 加强）

1. M1 HDBSCAN purity split 已经按 verb_family 切了
2. M2 synthesizer 在生成 SKILL.md 时**同时填 capability 段**——verb_family 直接从 cluster purity 来，affordance / output 从 procedure 抽
3. M2 audit 多一条 check：capability tag 是否 consistent 于 procedure（验证 `verb_family=clean` 的 skill procedure 里有 `clean <obj> with <affordance>` 步骤）
4. ⚠️ **新加一个 check #13**：capability signature 是否完整且 closed-set 合法

### Backward: skill → memory（**这里是真正的 axis-level 反馈**）

当前 M3 是 binary：skill 失败 → 全部 ancestry memory 加 `flagged` 标志。

加 capability 之后变成：
1. skill 失败时记录 failure signature: `{failed_at: step_4, error_type: affordance_unreachable, context: ...}`
2. M3 比对 failure signature 和 skill 的 capability axes
3. 把 ancestry memory **按 capability axis 分组** flag：
   - 如果失败发生在 verb_family 步骤（step 4），flag 那些 verb-相关的 memory
   - 如果失败发生在 affordance 步骤（step 3 go to sink），flag 那些 affordance-相关的 memory
   - 如果失败发生在 placeholder 填充（target_object 类型不匹配），flag 那些 target_type 相关 memory

这就是**真正的 axis-level bidirectional**，比当前的"全部 ancestry 一起 flag"细得多。

Reviewer 看到这条会赞：你不只是"skill 知道 memory 来源"，你还**把失败归因到 capability 维度**，进而精确 reweight memory。这才叫 bidirectional evolve。

### Memory 怎么吸收这个反馈

Live-Evo style weight update：
- `w_new = w_old - α * affected_axis_fail_count`
- 不同于全局 flag：只有"和失败 axis 相关的"memory 被 reweight
- 长期下来 memory bank 自然漂向"安全 axis 的"episode

## v1 PoC 实现范围

| 组件 | v1 范围 |
|---|---|
| Skill schema 扩展 `capability` 段 | ✅ 做 |
| M2 synthesizer 生成 capability tag | ✅ 做（prompt 加 capability 段填写指引） |
| M2 audit 新加 check #13 capability 一致性 | ✅ 做 |
| CapabilityIndex 数据结构 + 写入 | ✅ 做（dict<signature, [skill_id]>） |
| Task signature extractor | ✅ 做（rule-based 优先，LLM fallback） |
| Capability-indexed retrieval | ✅ 做（在 SkillRetriever 里加） |
| O(K) 检索 benchmark | ✅ 做（vs current similarity retrieval 跑个对比） |
| Axis-level M3 failure attribution | ✅ 做 |
| Script hook 字段 | ✅ 留接口，默认 null |
| 真正写 skill.py | ❌ v1.5 |
| Grounded mode 执行 | ❌ v1.5 |

工作量：v1 范围约 1 天 PoC + 半天测试。

## 闭集枚举（v1 PoC 范围，ALFWorld household）

```python
VERB_FAMILY = ["clean", "heat", "cool", "examine", "move", "search", "open", "use"]
AFFORDANCE = ["sink", "microwave", "fridge", "lamp", "desklamp", "floorlamp", "none"]
OUTPUT = ["object_placed", "object_examined", "location_reached", "object_processed", "object_acquired"]
TARGET_TYPE = ["object", "location", "abstract"]
PRECONDITION_TAGS = ["object_at_source", "destination_reachable", "appliance_available", "object_in_inventory"]
```

闭集小，hash table 简单。也是 v1 PoC 的优势——不用 sentence embedding 索引。

## 三个待决策点（写代码前要确认）

### 决策 1. capability 闭集大小

A. 极简（如上）：8 verbs + 7 affordances + 5 outputs + 3 targets + 4 preconditions
B. 中等：每个轴 15-20 个
C. 开放：let LLM emit free string，post-hoc clustering

**推荐 A**。v1 PoC 简单可解释，论文里可以一表展示完整闭集。reviewer 问"扩展性"，答案：v1.5 加 LLM-driven taxonomy expansion。

### 决策 2. task signature extraction 用 rule 还是 LLM

A. 纯 rule（first verb + keyword match）：快、便宜、不稳
B. 纯 LLM：稳、贵、慢
C. **rule 优先 + LLM fallback**：first-verb lookup 成功就走 rule，失败再调 LLM

**推荐 C**。clean / heat / examine 这种 first-verb 90%+ 准确率，剩下 10% 用 LLM。成本平摊每个 task 几 ms + 偶尔 1 个 API call。

### 决策 3. M3 axis-level 反馈是否引入新的 cluster

当 affordance 维度反复失败：是否触发"从 memory bank 重新聚 affordance=X 的 episode"产新 skill？

A. v1 不做：M3 只 reweight，不触发新 cluster
B. v1 做：M3 失败的 axis 触发 M1 重新 mine

**推荐 A**。v1 先把 axis-level reweight 做出来，触发新 cluster 是 v2 的事，不然 PoC 范围爆炸。

---

## 三件你需要拍板的事

1. **format 你 OK 这个 schema 吗**？capability 段 + 闭集 + 可选 script 字段，body 维持 NL。
2. **3 个决策点你都接受推荐吗**？A 极简闭集 + C rule+LLM + A v1 不触发新 cluster。
3. **B 路径 vs C 路径**：上次你说 B 或 C 都可以。这个设计**完整支持 B**（skill = indexed capability artifact）。**也能扩到 C**（如果将来想加 factual compressed memory 层，在 skill bank 之外单独建一个就行，不互相干扰）。我倾向先做 B，C 留 v2 ablation。你 OK 吗？

拍板这三件之后我开始写代码。预计 1-1.5 天能让 synthetic_pilot 跑出"capability-indexed 3 个 skill + O(K) retrieval demo"。
