# Blackboard and Guard context


Stage middleware writes the current stage information to `agent_state["blackboard"]`
independently of Guard and debug settings. Each task/thread has its own board;
each stage key holds its latest captured snapshot within the current round.
Snapshots are copied before Guard callbacks, so subsequent recovery does not
mutate the captured information. A round consists of one Main Agent model call
and its subsequent tool execution, observation, and post_step checks. After those
checks, only the completed round's `post_step` moves into `history` before the next
model call (including a recovery that skips tool execution). Only the previous
two rounds are retained; older rounds are evicted. Top-level stage keys describe
the current round, identified by `round`. The single top-level `input` is retained
as the global user goal, then refreshed from the bound user message.
Unchanged input refreshes only the snapshot, without another input Guard check
or input stage event. Forced memory retrieval belongs to the first round and
is not archived into history. A halted run retains the
window present at termination.

For example, at round 5, `history` contains rounds 3 and 4 (oldest first).
Each historical entry has `round` and `stages`; its stage keys hold that round's
post-step information only. If post_step was not reached, `stages` is empty;
no evidence is fabricated. History never includes input or recursively contains
older history.

At each invocation entry, Input middleware binds external user input in
`user_input_binding` with `source`, `message_id`, `version`, and `content`.
Round refreshes copy this bound content, rather than rescanning the last
`HumanMessage`. Internal Guard notices and input-recovery copies carry framework
origin metadata and cannot replace the goal. New user messages or genuine edits
to the current user message advance the version; replay and history compaction
retain it. Repeated text still avoids duplicate input checks. The binding and
seen message IDs are per-thread state, and `blackboard.input` remains a string.

Guard user prompts are JSON objects containing `stage`, `stage_payload`, and
`blackboard`. The current stage's current-round snapshot appears only in
`stage_payload`; its top-level key is excluded from the prompt's board. Historical
post-step snapshots remain available. Recovery likewise excludes only
the current-round source stage. Format retries preserve the entire window.
The Guard system prompt defines these fields and treats snapshots as supporting
evidence, not verified facts or authorization. Historical observations may be
stale after later actions; Guard must account for their round and newer evidence.
Pre-filters still inspect the raw
stage payload. The board does not enter the Main Agent prompt or become a tool.

| Key | Captured information |
| --- | --- |
| `round` | Current Main Agent round number |
| `history` | Up to two previous rounds, each with `round` and `stages` containing only available `post_step` |
| `input` | The single global user input text, not repeated in historical stage maps |
| `memory` | Retrieval tool, query/context, and retrieved episodes; MP forced retrieval uses the agent-facing fields without evaluation labels |
| `planning` | The todos submitted to `write_todos`, including content and status |
| `tool_selection` | Pending calls with `name` and `args`; transport `id` is omitted from stage records, Guard payloads and blackboard snapshots |
| `tool_observation` | Only `observation`: a string for one result, or an ordered list of strings for multiple results |
| `post_step` | The completed step's `invocations` (runtime name for tool execution) |
| `output` | Final model output text |

Only reached stages have entries. Ordinary episodic searches and MP forced
retrieval both populate `memory`. The simulated RTE path bypasses middleware and
records one round containing `input` and `post_step` explicitly. With `--save-results`, each task's
result JSON includes a `blackboard` field. Existing `stage_events` retain the
event history; the board stores current-stage values and the previous two rounds'
post-step evidence, rather than the complete task history.


Implementation: `blackboard.py`, `message_provenance.py`, and `stage_capture.py`.

Regression coverage: `tests/test_blackboard.py`, `tests/test_blackboard_window.py`,
`tests/test_guard_blackboard.py`, and `tests/test_user_input_binding.py`.
