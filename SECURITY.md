# Security Policy

## Supported versions

| Version | Supported          |
|---------|--------------------|
| 0.2.x   | ✅ Active          |
| 0.1.x   | ⚠️ Critical fixes only |
| < 0.1   | ❌ Not supported   |

## Reporting a vulnerability

If you discover a security vulnerability in ContextOps, please email
**[security@contextops.dev]** (replace with your real address) instead of
opening a public issue.

We will:

1. Acknowledge receipt within 48 hours.
2. Investigate and produce a fix within 14 days for critical issues.
3. Coordinate disclosure with you before publishing any CVE or advisory.

## What counts as a security issue

- Code execution from a malicious prompt file (`.json` / `.jsonl` dataset).
- SQLite injection via untrusted metadata fields.
- Path traversal in the local logger DB path.
- Sensitive data leakage through the LiteLLM auto-callback (e.g. logging prompts that contain user PII to a world-readable `~/.contextops/calls.db`).
- Dependency vulnerabilities (run `pip-audit` regularly).

## What does NOT count

- Bugs in prompt optimization (wrong section order, bad cache estimates) — file a regular issue.
- API rate-limit issues with Ollama / OpenRouter — those are provider-side.

## Hardening tips for users

- Set `chmod 600 ~/.contextops/calls.db` if you log sensitive prompts.
- Use `EchoJudge` in CI; never run real judges against untrusted input.
- Pin ContextOps in production: `contextops-tool==0.3.0` not `contextops-tool>=0.3`.
- Review the `metadata` field of every `CallLog` before sharing logs.