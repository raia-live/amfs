---
title: Content Integrity
layout: default
parent: Core Concepts
nav_order: 9
description: "SHA-256 hashing, integrity chains, and verification for tamper detection."
---

# Content Integrity

Every memory entry in AMFS is protected by a **content hash** and an **integrity chain**, allowing you to verify that stored data hasn't been corrupted or tampered with.

## How It Works

### Content Hash

On every write, AMFS computes `SHA-256(canonical_json(value))` and stores it as `content_hash` on the entry. The canonical JSON uses sorted keys, no whitespace, and ASCII-safe encoding for byte-level reproducibility across platforms.

### Integrity Chain

Each version links to its predecessor: `integrity_chain = SHA-256(content_hash + previous_integrity_chain)`. This creates a hash chain similar to a blockchain, where changing any historical version breaks the chain for all subsequent versions.

### Tree Hash

When entries are grouped in an atomic commit, the commit stores a `tree_hash = SHA-256(sorted(entry_hashes))`, sealing the entire batch.

## Verification

### Python SDK

```python
report = mem.verify("myapp/auth")
print(f"Checked: {report['total_checked']}, Valid: {report['valid']}")
if report['corrupted']:
    print(f"CORRUPTED: {report['corrupted']}")
```

### CLI

```bash
amfs verify myapp/auth --verbose
```

### MCP Tool

```
amfs_verify(entity_path="myapp/auth")
```

## Legacy Compatibility

Entries written before content hashing was enabled will have `content_hash = None`. These are treated as valid during verification — the system only flags entries where a hash exists but doesn't match.

## OSS vs Pro

| Feature | OSS | Pro |
|:--------|:---:|:---:|
| SHA-256 content hashing | Yes | Yes |
| Integrity chain linking | Yes | Yes |
| Verification (SDK, CLI, MCP, HTTP) | Yes | Yes |
| Cryptographic signing (HMAC) | — | Yes |
| Scheduled integrity checks | — | Yes |
| Dashboard integrity badges | — | Yes |
