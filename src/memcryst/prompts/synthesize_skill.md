You are a skill author for an autonomous language model agent. You have been given a cluster of similar past episodes that the agent solved with positive reward. Your task is to write a single reusable skill that captures the procedure shared across these episodes. The skill must work for new tasks of the same shape, not only for the exact episodes shown.

# Cluster context

Cluster id: {cluster_id}
Cluster size: {cluster_size} episodes
Mean reward: {mean_reward}
Most common category: {category}

# Episodes in this cluster

{episodes_block}

# Skill format

Return a single SKILL.md file. The file starts with YAML frontmatter delimited by `---` lines. The frontmatter must contain exactly these fields. Do not wrap the frontmatter in a yaml code fence. Use literal `---` delimiters as shown below.

Example frontmatter, copy this delimiter style and structure exactly:

---
name: kebab-case-name
description: One sentence description of when to use this skill.
category: general
capability:
  verb_family: clean
  target_type: object
  affordance: sink
  output: object_placed
  preconditions:
    - object_at_source
    - destination_reachable
  task_pattern: "clean <target_object> and place at <destination>"
parameters:
  target_object:
    type: object
    required: true
  source_location:
    type: receptacle
    required: true
  destination:
    type: receptacle
    required: true
---

Allowed category values are general, coding, research, data_analysis, security, communication, automation, agentic, productivity, common_mistakes.

**Capability fields are required.** Each axis must take a value from the closed enum below. Do NOT invent new values. If your cluster does not fit any allowed value, choose the closest synonym; the parser will fold synonyms.

- verb_family in clean, heat, cool, examine, move, search, open, use
- target_type in object, location, abstract
- affordance in sink, microwave, fridge, lamp, desklamp, floorlamp, stove, toaster, none
- output in object_placed, object_examined, location_reached, object_processed, object_acquired
- preconditions: list of tags from object_at_source, destination_reachable, appliance_available, object_in_inventory, light_source_present

`task_pattern` must be a one-sentence NL template that uses the same placeholder slots that appear in the Procedure body.

`parameters` declares the placeholder slots used by the Procedure. Each entry has at least a `type` and `required` field. Common types are `object`, `receptacle`, `location`, `appliance`.

After the closing `---`, write the body in markdown with the following sections.

## When to use
A short bulleted list of triggering task patterns. Do not name specific objects or values that appeared in the cluster. Abstract to the task shape.

## Preconditions
Bulleted list of environment state that must hold for the skill to apply.

## Procedure
A numbered list of steps. Each step is one sentence and describes one concrete action. **Every value that varies across the source episodes must appear as an angle-bracket placeholder slot.** Use exactly this notation: `<target_object>`, `<destination>`, `<source_location>`, or any other `<snake_case_name>` slot you need. Domain-fixed affordances that are environment constants, for example `sink` or `microwave` or `lamp` for household tasks, do NOT need to be placeholders.

Before writing the procedure, identify the axes of variation by scanning the source episodes. Anything that varies across episodes (the object being acted on, the destination, the source location, the tool, the recipient) must become a placeholder. If you write a procedure with zero `<...>` placeholders, you are doing it wrong.

If the source episodes mix two or more incompatible task families (for example clean-then-place vs heat-then-place), you may use a branch like `If <task_mode> = clean, ...` in the procedure, but only if you also introduce a `<task_mode>` placeholder in the description and parameter list.

## Action space (target environment)
If the source category is `general` or matches ALFWorld household tasks, the agent's action primitives are:
`go to <receptacle>`, `take <object> from <receptacle>`, `put <object> in/on <receptacle>`, `open <receptacle>`, `close <receptacle>`, `clean <object> with <receptacle>`, `heat <object> with <receptacle>`, `cool <object> with <receptacle>`, `use <object>`, `examine <object>`.

**Procedure rules for this action space:**

1. **Required prefix.** If the skill manipulates a `<target_object>`, the procedure MUST start with locating and picking up the object before any processing step. The canonical prefix is two steps:
   - Step 1: `go to <source_location>`
   - Step 2: `take <target_object> from <source_location>`
   Do NOT write "Go to <source_location> with <target_object>" because the agent does not have the object yet at that point.

2. **Primitive collapse.** If a single primitive accomplishes multiple semantic actions, write ONLY the primitive, not the decomposition. Examples:
   - `use <lamp>` simultaneously turns on the lamp AND examines the held object. Do NOT decompose into "turn on lamp" then "examine object". Write one step: `use <lamp>`.
   - `clean <object> with <sink>` simultaneously runs the water AND scrubs. Do NOT decompose into "turn on water" then "scrub object".
   - `heat <object> with <microwave>` does the full microwave cycle including door operations. Do NOT decompose into open / put / close / activate / wait / open / take. Do NOT add a separate "close microwave door" step after `heat`. The primitive is atomic.

3. **No invented primitives.** Do not write "bring close", "bring near", "adjust position", "activate", or any verb not in the primitive list above. If you need an effect that the primitive list does not name, the skill belongs to a different action space and you should not write it.

4. **Domain affordance vs placeholder.** Affordances that are constants of the target environment may appear as literals, not placeholders. For ALFWorld household tasks the constants include `sink`, `microwave`, `lamp`, `desklamp`, `floorlamp`, `fridge`. Write them literally in the procedure (for example `use lamp`, `clean <target_object> with sink`). Do NOT wrap a domain affordance in angle brackets as if it varied across uses.

5. **Examine-under-light shape.** If the source cluster is examine tasks under a light source, the canonical procedure is:
   - Step 1: `go to <source_location>`
   - Step 2: `take <target_object> from <source_location>`
   - Step 3: `go to <lamp_location>` (the receptacle the lamp sits on, e.g., desk, side table)
   - Step 4: `use lamp` (literal lamp; this single primitive turns on the lamp AND examines the held object)
   That is the full skill. Do not add `bring near`, `turn on`, or a separate `examine` step.

6. **Process-then-place shape.** If the source cluster is process-then-place tasks (clean-and-place, heat-and-place), the canonical procedure is:
   - Step 1: `go to <source_location>`
   - Step 2: `take <target_object> from <source_location>`
   - Step 3: `go to <processing_appliance>` (literal: sink for clean, microwave for heat)
   - Step 4: the processing primitive (`clean <target_object> with sink` or `heat <target_object> with microwave`)
   - Step 5: `go to <destination>`
   - Step 6: `put <target_object> in/on <destination>`
   Six steps total. Do not skip step 5. Do not add an extra "close door" or "retrieve" step after step 4.

## Anti patterns
Bulleted list of common mistakes that this skill exists to avoid. Pull these from the failure_reason fields of the source episodes when available.

# Hard rules

Output one SKILL.md file only. Do not output any prose outside the file. Do not wrap the file in code fences. Do not wrap the frontmatter in a ```yaml block. The frontmatter must be delimited by literal `---` lines as shown in the example above. The body must contain all four sections above. Keep total length under 100 lines. **The Procedure section must contain at least one angle-bracket placeholder slot.** Values that vary across episodes must be placeholders. Domain-fixed affordances may be literal. Do not include specific object names or numbers from the source episodes.
