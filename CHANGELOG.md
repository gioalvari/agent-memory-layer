# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-11-06

### Added

- Bounded conversation history with configurable trimming and stable-prefix
  injection modes (PR #6).
- Hybrid dense and SQLite FTS5 lexical retrieval, fused with reciprocal-rank
  fusion (PR #7).
- Anthropic-compatible `/v1/messages` proxying, including memory injection and
  relevant header forwarding (PR #8).
- Query-aware sentence compression for overlong memory sides (PR #9).
- Temporal metadata, UTC date display, and optional temporal ranking boosts
  (PR #10).
- Asynchronous durable fact extraction, latest-value supersession, facts-first
  injection, and fact administration endpoints (PR #11).
- Italian and Spanish temporal expressions, with accent-tolerant matching.
- Background backfill of sentence embeddings for memories stored before
  `--compress-memories` was enabled.
- Linux (ubuntu-latest) CI job, recursive submodule checkout, and ccache.

### Changed

- Token budgets (history, injection, sticky, facts) use the embedding model's
  tokenizer, with the byte heuristic as fallback.
- Sentence embeddings for `--compress-memories` are computed in batches.
- Metal is enabled only on Apple platforms.

### Fixed

- Composed retrieval and fact handling across the merged features, including
  fact-save synchronization and integration-backend access.
- Test linkage for temporal helpers and formatters used by hybrid and sticky
  retrieval tests.
- Integration and build conflicts while combining history, hybrid retrieval,
  Anthropic Messages, compression, temporal metadata, and fact extraction.
