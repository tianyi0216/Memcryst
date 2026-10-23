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
Write 5-8 CONDITIONAL STRATEGIES — NOT a fixed numbered step sequence. Each is a
tactic paired with an explicit trigger. Use exactly this shape, one per line:

- <Strategy name>: <conditional tactic>. Apply this when <trigger>.

HARD RULES:
1. NEVER write an unconditional manipulation. Every take/put/clean/heat/cool/use
   MUST be conditioned on an observation. Write "If <target_object> is visible on
   the current surface, take it; otherwise keep searching" — NEVER "Take
   <target_object> from <source_location>".
2. Explicitly cover the NON-OBVIOUS moves as triggered strategies: open the
   appliance/container before using it (e.g. open the fridge before `cool`); turn
   the lamp on with `use lamp`; confirm a picked object matches the goal before
   processing it.
3. Cover SEARCH/CONTINGENCY: a strategy for when the target is NOT where expected
   (systematically check unvisited receptacles; never revisit empty ones).
4. The agent is a capable planner. Give it triggered guidance to apply WHEN
   RELEVANT, never a rigid script that overrides its own judgement.

Action primitives (reference inside the conditional strategies, do NOT lay out as
a fixed sequence): `go to <receptacle>`, `take <object> from <receptacle>`,
`put <object> in/on <receptacle>`, `open <receptacle>`, `clean <object> with sink`,
`heat <object> with microwave`, `cool <object> with fridge`, `use lamp`,
`examine <object>`. Variable slots are angle-bracket placeholders
(<target_object>, <destination>, <source_location>); affordances
(sink/microwave/fridge/lamp) are literals.

EXAMPLE (cool-and-place) — copy this conditional STYLE and richness:
- Systematic Object Hunt: Sweep unvisited surfaces and open closed containers until <target_object> is found; never revisit empty spots. Apply this right after parsing the goal, before holding anything.
- Confirm Object Match: After picking something up, verify it is <target_object>; if not, put it back and resume searching. Apply this whenever you acquire an object but the goal object is still unmet.
- Prep Cooling Appliance: Go to the fridge and open it so it is ready. Apply this as soon as the fridge is in view or right after acquiring <target_object>.
- Cool Before Placing: Do not place <target_object> at <destination> until you have run `cool <target_object> with fridge`. Apply this when holding the correct object, before any placement.
- Direct Delivery: Once cooled, go straight to <destination> and put it down with no detours. Apply this right after cooling succeeds.


## Anti patterns
Bulleted list of common mistakes that this skill exists to avoid. Pull these from the failure_reason fields of the source episodes when available.

# Hard rules

Output one SKILL.md file only. Do not output any prose outside the file. Do not wrap the file in code fences. Do not wrap the frontmatter in a ```yaml block. The frontmatter must be delimited by literal `---` lines as shown in the example above. The body must contain all four sections above. Keep total length under 100 lines. **The Procedure section must contain at least one angle-bracket placeholder slot.** Values that vary across episodes must be placeholders. Domain-fixed affordances may be literal. Do not include specific object names or numbers from the source episodes.
