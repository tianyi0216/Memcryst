"""TaskstreamEnv wraps a materialized event list as an environment so the full
pipeline (trajectories -> writeback -> mining -> synthesis -> audit -> serving)
runs unchanged on conversational QA streams.

Episode shapes for ``MEMCRYST_TASKSTREAM_KIND=qa``:
  session  = read one conversation session (a single step; the trajectory
             carries the session text, so the standard writeback ingests it)
  question = answer from retrieved context; scored by token-F1
             (success = F1 >= 0.5)

The stream file is a JSON list of events ``{"kind": "session", "date", "text"}``
and ``{"kind": "question", "question", "answer", "category"}``.
"""
from __future__ import annotations

import json
import os
import re
from typing import Optional

from memcryst.agent.alfworld_env import ResetResult, StepResult
from memcryst.agent.alfworld_env import register_family_override

_ANSWER_RE = re.compile(r"(the answer is|final answer\s*:?|answer\s*:)", re.IGNORECASE)


def _extract_code(text: str) -> str:
    m = re.findall(r"```(?:python)?\s*(.*?)```", text or "", re.DOTALL)
    return m[-1].strip() if m else ""


def _run_python(code: str, timeout: int = 10, stdin: str = None) -> str:
    import subprocess, sys, tempfile
    from pathlib import Path
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code); path = f.name
    try:
        r = subprocess.run([sys.executable, path], capture_output=True,
                           text=True, timeout=timeout, input=stdin)
        return ((r.stdout or "") + (r.stderr or ""))[:400].strip()
    except Exception as e:
        return f"(execution failed: {e})"
    finally:
        Path(path).unlink(missing_ok=True)




def _sample_test(t, code):
    """Run the FIRST public test case. Returns (passed, feedback). Tasks whose
    tests are unavailable/unparseable pass through (submit directly)."""
    import ast as _ast
    tc = t.get("test_cases")
    if isinstance(tc, str):
        try:
            tc = _ast.literal_eval(tc)
        except Exception:
            return True, ""
    if not tc:
        return True, ""
    first = tc[0]
    if isinstance(first, (list, tuple)) and len(first) == 2:   # [stdin, expected]
        got = _run_python(code, stdin=str(first[0]))
        if got.split() == str(first[1]).split():
            return True, ""
        return False, (f"input:\n{str(first[0])[:300]}\n"
                       f"expected output: {str(first[1])[:200]}\n"
                       f"your output: {got[:250]}")
    if isinstance(first, str) and first.strip().startswith("assert"):
        out = _run_python(code + "\n" + first)
        if out.strip() and ("Error" in out or "Traceback" in out
                            or out.startswith("(execution failed")):
            return False, f"{first[:180]}\n{out[:250]}"
        return True, ""
    return True, ""

_LQA_EMB = {}

def _lqa_embed(texts):
    if "m" not in _LQA_EMB:
        from sentence_transformers import SentenceTransformer
        _LQA_EMB["m"] = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    import numpy as _np
    return _np.asarray(_LQA_EMB["m"].encode(texts, show_progress_bar=False), dtype="float32")


class TaskstreamEnv:
    def __init__(self, split: str = "train", max_steps: int = 30):
        self.tasks = json.loads(open(os.environ["MEMCRYST_TASKSTREAM_DATA"]).read())
        self.kind = os.environ.get("MEMCRYST_TASKSTREAM_KIND", "solve")
        self.eval_kind = os.environ.get("MEMCRYST_TASKSTREAM_EVAL", "math")
        self.max_steps = max_steps
        self._ev = None
        if self.eval_kind != "f1":
            raise ValueError(f"MEMCRYST_TASKSTREAM_EVAL={self.eval_kind!r}: this release ships the token-F1 evaluator "
                             "for conversational QA streams (set MEMCRYST_TASKSTREAM_EVAL=f1)")
        self.mixed = False
        self._ev_by = {}
        self._cur = None
        self._nstep = 0
        self._transcript = ""
        self._last_exec = ""

    def _task_kind(self, t) -> str:
        """Per-task kind. Explicit t['kind'] wins (session/question); a task
        carrying a long 'context' is longqa; otherwise solve."""
        if not self.mixed:
            return t.get("kind") or self.kind
        k = t.get("kind")
        if k in ("session", "question"):
            return k
        if t.get("context"):
            return "longqa"
        return "solve"

    def _task_ev(self, t):
        """Per-task scorer for mixed streams (math vs code)."""
        if not self.mixed:
            return self._ev
        return self._ev_by["hardcode"] if self._is_code_task(t) else self._ev_by["math"]

    def __len__(self):
        return len(self.tasks)

    def _longqa_excerpts(self, t) -> str:
        """Env-side retrieval for longqa tasks so the FULL pipeline (skills/
        facts/raw + sleep crystallization) runs on top. Budget 18000 chars
        leaves prompt room for memory injections."""
        import numpy as np
        ctx = t["context"]
        # token-aware budget: RULER digit-heavy text is ~2.5 chars/token, prose
        # ~4 — a fixed 18000-char budget overflowed the 8192 window for the mem
        # arm (781 BadRequest errors in lc_full, single_1 1.00->0.28 was the
        # alarm). Target ~4200 tokens for the obs, estimated per document.
        words = ctx.count(" ") + 1
        ratio = max(2.2, min(5.0, len(ctx) / (words * 1.35)))
        # MEMCRYST_LQA_BUDGET: obs token target (default 4200). Parity knob for
        # the reader matrix — cap scales with it (3.81x = original 16000/4200).
        _tgt = int(os.environ.get("MEMCRYST_LQA_BUDGET", "4200") or 4200)
        budget = int(_tgt * ratio)
        budget = max(6000, min(int(_tgt * 3.81), budget))
        # MEMCRYST_LQA_MODE=none must win BEFORE the short-document early return:
        # under the reader contract the actor never sees the document itself.
        # (09-03 audit: a 14.5k-char uuid document slipped through this early
        # return into the baseline actor's observation → 10.6k-token prompt,
        # and every baseline task with a document under the char budget was
        # answered from the raw text instead of the method's store.)
        if os.environ.get("MEMCRYST_LQA_MODE", "") == "none":
            return ("[The document was provided to your memory system; "
                    "retrieved content, if any, is in your context blocks.]")
        if len(ctx) <= budget:
            return ctx
        # MEMCRYST_LQA_MODE=tail: bare-baseline presentation — same token budget,
        # tail window, NO retrieval (retrieval reading is the METHOD's substrate,
        # so the vanilla arm must not get it; same definition as ruler_probe
        # vanilla). Default (unset) keeps retrieval excerpts for the system arm.
        if os.environ.get("MEMCRYST_LQA_MODE", "") == "tail":
            return ("[document truncated: showing the final portion]\n"
                    + ctx[-budget:])
        # MEMCRYST_LQA_MODE=none: wave-2 baseline-reader presentation — the
        # document goes to the external method's store (its store IS its
        # reader); the actor sees the question plus the method's injected block.
        if os.environ.get("MEMCRYST_LQA_MODE", "") == "none":
            return ("[The document was provided to your memory system; "
                    "retrieved content, if any, is in your context blocks.]")
        chs = [ctx[i:i + 1200] for i in range(0, len(ctx), 1200)]
        embs = _lqa_embed([c[:800] for c in chs])
        q = _lqa_embed([t["question"][:500]])[0]
        sims = embs @ q / (np.linalg.norm(embs, axis=1) * (float(np.linalg.norm(q)) or 1.0) + 1e-9)
        qt = set(t["question"].lower().split())
        lex = np.asarray([0.08 * min(len(qt & set(c.lower().split())), 4) for c in chs])
        k = max(1, budget // 1200)
        picked = sorted(np.argsort(-(sims + lex))[:k].tolist())
        return "\n---\n".join(
            f"[excerpt {i + 1} of {len(chs)}, in document order]\n{chs[i]}" for i in picked)

    # ----- helpers -----
    def _is_code_task(self, t) -> bool:
        return (t.get("type", "") in ("code_generation", "code_completion")
                or t.get("domain", "") in ("mbpp", "humaneval", "code_contests",
                                           "livecodebench"))

    def _score_solve(self, t, response: str) -> float:
        ev = self._task_ev(t)
        if ev is None:
            return 0.0
        if hasattr(ev, "_evaluate_response"):        # AFlowEvaluator
            return float(ev._evaluate_response(t, response))
        return float(ev._evaluate(t, response))       # HardMathEvaluator

    # ----- env interface -----
    def reset(self, task_idx: Optional[int] = None) -> ResetResult:
        t = self.tasks[(task_idx or 0) % len(self.tasks)]
        from memcryst.envscope import set_subscope
        _kind0 = self._task_kind(t)
        # (a) `or ""` swallowed stream=0 (falsy) — conversation 0 was never
        #     stamped; (b) tasks WITHOUT a stream field (math/code/longqa in
        #     mixed streams) fell to the base stamp, visible to everyone — so
        #     "stream scoping" built walls between conversations (useless,
        #     qa_loco4) and none between domains (the actual hypothesis).
        #     Derive a DOMAIN subscope when the field is absent.
        _sv = t.get("stream", None)
        if _sv is None or _sv == "":
            if _kind0 == "longqa":
                _sv = "lc"
            elif _kind0 in ("session", "question"):
                _sv = "conv"
            else:
                _sv = "code" if self._is_code_task(t) else "math"
        set_subscope(str(_sv))
        # MEMCRYST_MIXED_QA_ADAPTERS=1: conversational-QA retrieval settings for question tasks.
        if os.environ.get("MEMCRYST_MIXED_QA_ADAPTERS", "0") == "1":
            if not hasattr(self, "_adapter_base"):
                self._adapter_base = (
                    os.environ.get("MEMCRYST_FACT_MIN_SUPPORT", "1"),
                    os.environ.get("MEMCRYST_RAW_RECENCY", "0"))
            if _kind0 == "question":
                os.environ["MEMCRYST_FACT_MIN_SUPPORT"] = "1"
                os.environ["MEMCRYST_RAW_RECENCY"] = "4"
            else:
                os.environ["MEMCRYST_FACT_MIN_SUPPORT"] = self._adapter_base[0]
                os.environ["MEMCRYST_RAW_RECENCY"] = self._adapter_base[1]
        self._cur, self._nstep, self._transcript, self._last_exec = t, 0, "", ""
        _tk = _kind0
        self.current_kind = _tk   # exposed for the QA answer-call adapter
        if _tk == "session":
            self._last_session_date = t.get("date", "")   # temporal anchor for questions
            desc = (f"Review the conversation session dated {t.get('date','?')} "
                    f"and remember the important facts about the speakers.")
            register_family_override(desc, "session")
            obs = ("Read this past conversation session and note the important "
                   f"facts (dated {t.get('date','?')}):\n\n{t['text']}\n\n"
                   "When done, reply with: noted")
            return ResetResult(obs=obs, task_description=desc,
                               valid_actions=["noted"], info={"kind": "session"})
        if _tk == "longqa":
            desc = t["question"]
            register_family_override(desc, str(t.get("task", "longqa")))
            obs = (self._longqa_excerpts(t) +
                   f"\n\nQuestion: {t['question']}\n"
                   "Note: when several facts about the same entity conflict, the "
                   "LATEST excerpt reflects the current state.\n"
                   "Answer concisely: Answer: <short answer>")
            return ResetResult(obs=obs, task_description=desc,
                               valid_actions=["Answer: <short answer>"],
                               info={"kind": "longqa"})
        if _tk == "question":
            desc = t.get("desc") or t["question"]
            register_family_override(desc, str(t.get("category", "question")))
            if self.eval_kind == "m2w":
                # Mind2Web offline step: the item text IS the full prompt
                # (goal + history + candidates); desc carries site+goal so the
                # retrieval query stays compact and site-scoped.
                return ResetResult(obs=t["question"], task_description=desc,
                                   valid_actions=["Answer: <letter> <op> [value]"],
                                   info={"kind": "question"})
            _anchor = getattr(self, "_last_session_date", "")
            # Abstention phrasing (MEMCRYST_QA_PUSH=1 selects the more assertive variant).
            _abst = ("Give your best short answer whenever ANY remembered fact "
                     "relates to the question — only if absolutely nothing "
                     "relates, reply: No information available."
                     if os.environ.get("MEMCRYST_QA_PUSH", "0") == "1" else
                     "If the conversations do not contain the answer, reply: "
                     "No information available.")
            obs = (f"Answer this question about the people in the conversations "
                   f"you have reviewed."
                   + (f" Today is {_anchor}." if _anchor else "") + "\n"
                   f"Question: {t['question']}\n"
                   "Reply with ONLY a short phrase (a few words, a date, or a "
                   f"name). {_abst}\nAnswer: <short answer>")
            return ResetResult(obs=obs, task_description=desc,
                               valid_actions=["Answer: <short answer>"],
                               info={"kind": "question"})
        desc = re.sub(r"\s+", " ", t["prompt"])[:220]
        register_family_override(desc, t.get("domain", "task"))
        # Code tasks: state the entry point and one test when the prompt omits them.
        name_hint = ""
        ep = t.get("entry_point")
        if self._is_code_task(t) and ep and ep not in t["prompt"]:
            name_hint = f"\nName the function exactly `{ep}`."
            tc = t.get("test_cases")
            if isinstance(tc, str):
                import ast as _ast
                try:
                    tc = _ast.literal_eval(tc)
                except Exception:
                    tc = [tc]
            if tc:
                name_hint += f"\nYour code should pass tests like: {tc[0]}"
        obs = (f"{t['prompt']}{name_hint}\n\nYou may reason, and you may include ONE "
               "```python block — I will execute it and show you the output. "
               "Finish with: The answer is <answer>"
               + ("" if not self._is_code_task(t) else
                  "\nFor coding tasks, provide the complete solution in the "
                  "```python block (the block itself is your answer)."))
        return ResetResult(obs=obs, task_description=desc,
                           valid_actions=["<your reasoning / python / answer>"],
                           info={"kind": "solve", "domain": t.get("domain")})

    def step(self, action: str) -> StepResult:
        t = self._cur
        self._nstep += 1
        a = (action or "").strip()
        _tk = self._task_kind(t)
        if _tk == "session":
            return StepResult(obs="Noted.", reward=1.0, done=True, success=True,
                              valid_actions=[], info={})
        if _tk == "longqa":
            ans = a
            m = re.search(r"answer\s*:\s*(.+)", a, re.IGNORECASE | re.DOTALL)
            if m:
                ans = m.group(1).strip().splitlines()[0]
            golds = t.get("answers", [])
            sc = (sum(1 for g in golds if str(g).lower() in ans.lower()) / len(golds)) if golds else 0.0
            return StepResult(obs="(answered)", reward=sc, done=True,
                              success=sc >= 0.99, valid_actions=[], info={})
        if _tk == "question" and self.eval_kind == "m2w":
            g = t["gold"]
            m = re.search(r"answer\s*:\s*([A-T])\b[\s.,]*([A-Z]+)?\s*(.*)",
                          a, re.IGNORECASE | re.DOTALL)
            letter = m.group(1).upper() if m else ""
            op = (m.group(2) or "").upper() if m else ""
            val = ((m.group(3) or "").strip().splitlines() or [""])[0] if m else ""
            reward = 0.0
            if letter == g["letter"]:
                reward += 0.5
            if op == g["op"]:
                reward += 0.3
                if g.get("value"):
                    from memcryst.utils.text_metrics import token_f1
                    reward += 0.2 * token_f1(val, g["value"])
                else:
                    reward += 0.2
            success = (letter == g["letter"]) and (op == g["op"])
            return StepResult(obs="(answered)", reward=round(reward, 3), done=True,
                              success=success, valid_actions=[],
                              info={"letter_ok": letter == g["letter"]})
        if _tk == "question":
            ans = a
            m = re.search(r"answer\s*:\s*(.+)", a, re.IGNORECASE | re.DOTALL)
            if m:
                ans = m.group(1).strip().splitlines()[0]
            from memcryst.utils.text_metrics import token_f1
            f1 = token_f1(ans, str(t.get("answer", "")))
            return StepResult(obs=f"(answered)", reward=f1, done=True,
                              success=f1 >= 0.5, valid_actions=[],
                              info={"f1": round(f1, 3)})
        # ----- solve -----
        self._transcript += "\n" + a
        code = _extract_code(a)
        if self._is_code_task(t) and code:
            # self-test loop (MEMCRYST_CODE_SELFTEST=1): run the FIRST public
            # test before submitting; one revision allowed. Turns one-shot
            # generation into an act->observe->correct loop — the surface
            # where retrieved pitfall-memory can actually act.
            _st = os.environ.get("MEMCRYST_CODE_SELFTEST", "0") not in ("", "0")
            if _st and self._nstep == 1:
                ok, fb = _sample_test(t, code)
                if not ok:
                    return StepResult(
                        obs=("Sample test FAILED:\n" + fb +
                             "\nRevise and reply with the COMPLETE corrected "
                             "solution in ONE ```python block."),
                        reward=0.0, done=False, success=False,
                        valid_actions=["<revised ```python block>"], info={})
            score = self._score_solve(t, a)          # evaluator runs unit tests
            return StepResult(obs="(submitted)", reward=score, done=True,
                              success=score >= 0.99, valid_actions=[], info={})
        has_answer = bool(_ANSWER_RE.search(a))
        if code and not has_answer and self._nstep < 3:
            self._last_exec = _run_python(code)
            return StepResult(obs=f"[python output]\n{self._last_exec or '(no output)'}\n"
                                  "Now finish with: The answer is <answer>",
                              reward=0.0, done=False, success=False,
                              valid_actions=["<your reasoning / answer>"], info={})
        response = self._transcript
        if code and not has_answer:                   # ran out of steps: use exec output
            self._last_exec = self._last_exec or _run_python(code)
            response += f"\nThe answer is {self._last_exec}"
        elif self._last_exec and not has_answer:
            response += f"\nThe answer is {self._last_exec}"
        score = self._score_solve(t, response)
        return StepResult(obs="(submitted)", reward=score, done=True,
                          success=score >= 0.99, valid_actions=[], info={})
