You are a skill author for an autonomous agent. Given a cluster of similar SOLVED episodes, write ONE reusable skill that captures the shared PROCEDURE — the control-flow only.

IMPORTANT DIVISION OF LABOUR: at run time the agent ALSO receives, separately,
  - GROUNDING FACTS about WHERE object types are usually found, and
  - CONSTRAINT FACTS about how appliances/containers work.
So this skill must NOT include search heuristics ("systematically check receptacles", "never revisit") or location guesses — that is the grounding facts' job. Write ONLY the control-flow skeleton. Keep it short.

# Cluster context

Cluster id: {cluster_id}
Cluster size: {cluster_size} episodes
Mean reward: {mean_reward}
Most common category: {category}

# Episodes in this cluster

{episodes_block}

# Skill format

Return a single SKILL.md file. Start with YAML frontmatter delimited by literal `---` lines (no code fence). Exactly these fields:

---
name: kebab-case-name
description: One sentence describing when to use this skill.
category: general
capability:
  verb_family: heat
  target_type: object
  affordance: microwave
  output: object_placed
  preconditions:
    - object_at_source
    - destination_reachable
  task_pattern: "heat <target_object> and place at <destination>"
parameters:
  target_object:
    type: object
    required: true
  destination:
    type: receptacle
    required: true
---

Allowed category values: general, coding, research, data_analysis, security, communication, automation, agentic, productivity, common_mistakes.

**Capability fields are required**, each from the closed enum (do NOT invent):
- verb_family in clean, heat, cool, examine, move, search, open, use
- target_type in object, location, abstract
- affordance in sink, microwave, fridge, lamp, desklamp, floorlamp, stove, toaster, none
- output in object_placed, object_examined, location_reached, object_processed, object_acquired
- preconditions: tags from object_at_source, destination_reachable, appliance_available, object_in_inventory, light_source_present

`task_pattern` uses the same `<placeholder>` slots as the Procedure. `parameters` declares those slots.

After the closing `---`, write EXACTLY these sections.

## When to use
2-3 bullets of triggering task SHAPES. Never a specific object/receptacle/location.

## Procedure
The CONTROL-FLOW SKELETON only — at most 6 lines, one operation per line, using these primitives:

  find(<target_object>)              # locate & take the target (HOW to find it = the grounding facts' job, not yours)
  goto(<place>)                      # by TYPE only (goto(microwave), goto(<destination>)) — never a numbered id
  open(<appliance>)                  # include ONLY where the manipulation needs it (the ordering IS procedure)
  heat/cool/clean(<target_object>)   # the family manipulation (heat=microwave, cool=fridge, clean=sink), run EXACTLY ONCE
  use(<lamp>)                        # examine tasks only
  put(<target_object>, <destination>)

RULES:
1. Write the correct ORDER (e.g. for heat: find → goto(microwave) → open(microwave) → heat → put). The open-before-use ordering belongs here because it is part of the procedure SEQUENCE.
2. Run the family manipulation EXACTLY ONCE. If <destination> IS the appliance, just put (do not re-run).
3. NO search/contingency prose, NO "systematic hunt / never revisit / confirm match", NO location guesses. `find(<target>)` is a single primitive; the agent grounds it from the separately-provided facts.
4. Every room-varying value is an angle-bracket placeholder (<target_object>, <destination>). Affordance literals (microwave/fridge/sink/lamp) are fine.

## Anti patterns
1-2 bullets of control-flow mistakes to avoid (e.g. running the manipulation twice; placing before processing). Pull from the episodes' failure_reason when present. Do NOT restate appliance-open or where-to-look here.

# Hard rules

Output ONE SKILL.md file only, no prose outside it, no code fences, frontmatter delimited by literal `---`. Under 45 lines. The Procedure MUST contain at least one `<placeholder>` and MUST NOT contain a numbered receptacle id or specific object id.
