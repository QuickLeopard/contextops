# v1.0 — Access-Aware Context + Audit Trail

Status: **Design draft** for Track B (enterprise / on-prem readiness).

## Goal

Make ContextOps safe for multi-user, regulated, or on-prem deployments by
answering two questions for every prompt:

1. **Who is allowed to see this context?** — explicit access labels on prompt
   sections and a clearance-based filter before any reordering or LLM call.
2. **What was shown, to whom, and why?** — an immutable audit trail in the
   local SQLite logger that records access decisions per section per call.

Non-goal: full IAM/OAuth/encryption. This slice introduces the data model and
local audit hooks; enterprise identity integration is deferred.

## Definitions

- **Access level** — a discrete classification label such as `public`,
  `internal`, `confidential`, `restricted`. Levels form a partial order.
- **Actor** — the user/service on whose behalf a prompt is being built. Has an
  `identity` (string) and a `clearance` set of access levels they may read.
- **Access decision** — for each prompt section, either `kept` or `dropped`,
  with the actor/clearance/level that justified the decision.
- **Audit trail** — append-only records in SQLite linking every LLM call to
  the sections that were included and excluded.

## Decisions

| # | Decision | Choice |
|---|---|---|
| 1 | Where access labels live | On each **section** (system/tools/role/context/documents/history/query), not on the whole prompt, so a single prompt can mix public context with restricted documents. |
| 2 | Level ordering | Hard-coded lattice in `contextops.access` (`public < internal < confidential < restricted`) for the MVP. Custom lattices are a future extension. |
| 3 | Default level | `internal` for every section if not specified. This is conservative enough for enterprise defaults without breaking existing public workloads. |
| 4 | Where filtering happens | **Before** `reorder()`/`optimize()`. Reordering operates only on the already-authorized subset; the audit log records what was excluded before the call. |
| 5 | Audit storage | Extend the existing `contextops.logger.Logger` SQLite DB with a new `access_decisions` table, foreign-keyed to `calls`. No new dependencies. |
| 6 | History handling | Each `HistoryMessage` may carry its own `access_level`; the rendered `history` section inherits the highest level among its messages. |

## In scope

### 1. Access model (`contextops/access.py`)

New module with pure functions and dataclasses:

- `AccessLevel` enum: `PUBLIC`, `INTERNAL`, `CONFIDENTIAL`, `RESTRICTED`.
- `Actor(identity: str, clearance: set[AccessLevel])`.
- `can_read(actor: Actor, level: AccessLevel) -> bool` using the lattice.
- `filter_sections_by_access(sections, actor) -> FilterResult`:
  - Returns `kept` and `dropped` section lists.
  - Raises if a section has an unknown level.
  - Preserves original order among kept sections.

### 2. Prompt model changes (`contextops/models.py`)

- Add optional `access_level: Optional[str] = None` to `Prompt` scalar fields
  (`system`, `tools`, `role`, `context`, `documents`, `query`).
- Add optional `access_level: Optional[str] = None` to `HistoryMessage`.
- `Prompt.sections()` returns `(section_name, content, access_level)` triples
  (backwards-compatible: level defaults to `internal`).
- New model `AccessDecision(section, level, decision, reason)`.
- Extend `OptimizationResult` with `access_decisions: list[AccessDecision]`.

### 3. Optimizer integration (`contextops/optimizer.py`)

- `optimize(prompt, *, actor=None, config=None)`:
  - If `actor` is provided, filter sections before reordering.
  - Populate `OptimizationResult.access_decisions`.
  - Add a note for every dropped section.
- `reorder()` signature likewise gains optional `actor`.
- No access arguments → current behavior unchanged.

### 4. Curator integration (`contextops/curator.py`)

- Add optional `access_level` to `DocumentChunk`.
- `curate()` does **not** filter by access (its job is relevance); the caller
  or `Prompt.from_chunks()` handles access later.
- `Prompt.from_chunks()` passes through per-chunk access levels into the
  generated `documents` section (the section level becomes the max of kept
  chunks).

### 5. Audit logger (`contextops/logger.py`)

- New schema migration:

  ```sql
  CREATE TABLE access_decisions (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      call_id INTEGER NOT NULL,
      actor_id TEXT NOT NULL,
      actor_clearance TEXT NOT NULL,
      section_name TEXT NOT NULL,
      section_level TEXT NOT NULL,
      decision TEXT NOT NULL CHECK(decision IN ('kept','dropped')),
      reason TEXT,
      FOREIGN KEY (call_id) REFERENCES calls(id)
  );
  CREATE INDEX idx_access_call ON access_decisions(call_id);
  CREATE INDEX idx_access_actor ON access_decisions(actor_id);
  ```

- `Logger.log_access_decisions(call_id, actor, decisions)`.
- `Logger.audit_query(actor_id=None, limit=100)` returns recent decisions with
  call metadata.

### 6. CLI (`contextops/cli.py`)

- `contextops optimize --actor-id X --actor-clearance public,internal ...`
  passes the actor through.
- New `contextops audit` subcommand:
  - `--actor` filter, `--limit`, `--since`.
  - Renders a table of: timestamp, actor, call model, section, level, decision,
    reason.

### 7. Tests (`tests/test_access.py`)

- Lattice ordering.
- Filtering keeps/drops correct sections for various clearance sets.
- `optimize()` with actor produces authorized section order and records
  decisions.
- Unknown access level raises.
- Logger schema migration + `audit_query()` round-trip.
- CLI `--actor-*` flags end-to-end.

## Out of scope (deferred)

- Custom access-level lattices per tenant.
- Row-level / field-level redaction inside section content.
- Integration with external IdPs (OAuth, SAML, LDAP).
- Encryption of audit logs at rest.
- Distributed / tamper-evident audit logs.
- Per-chunk access inside a single `documents` section (section-level only for
  MVP).

## File layout

```
contextops/
  access.py           NEW     AccessLevel, Actor, can_read, filter_sections_by_access
  models.py           EDIT    add access_level, AccessDecision
  optimizer.py        EDIT    optimize()/reorder() accept actor
  curator.py          EDIT    DocumentChunk.access_level
  logger.py           EDIT    access_decisions schema + helpers
  cli.py              EDIT    --actor-* flags + audit subcommand

tests/
  test_access.py      NEW     access logic tests
  test_logger.py      EDIT    audit table tests
  test_cli.py         EDIT    actor/audit CLI tests

docs/
  PLAN_v1.0.md        NEW     this document
  README.md           EDIT    document --actor-clearance and audit command
```

## Acceptance criteria

- `pytest` stays green (existing tests unaffected when `actor` is omitted).
- `ruff check .` and `mypy contextops contextops_bench` pass on touched files.
- New `tests/test_access.py` covers lattice, filtering, optimizer integration,
  unknown-level error.
- `contextops audit` renders decisions from a local SQLite log.
- README/CHANGELOG updated.

## Security notes

- This slice is **advisory / audit**, not a cryptographically enforced
  boundary. A caller can still bypass access checks by not passing an actor or
  by constructing the prompt manually. Enterprise hardening requires the
  deferred IdP + enforcement work.
- Audit log integrity relies on filesystem permissions of the SQLite DB.
- Access defaults to `internal` to avoid accidentally leaking data in mixed
  prompts.
