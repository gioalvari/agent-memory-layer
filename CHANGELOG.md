# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

### Fixed

- Composed retrieval and fact handling across the merged features, including
  fact-save synchronization and integration-backend access.
- Test linkage for temporal helpers and formatters used by hybrid and sticky
  retrieval tests.
- Integration and build conflicts while combining history, hybrid retrieval,
  Anthropic Messages, compression, temporal metadata, and fact extraction.
