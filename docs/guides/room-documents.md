---
title: Room Documents
layout: default
parent: Guides
nav_order: 12
description: "Put PDFs, Word documents and notes into a room so every teammate's agent can search and quote them."
---

# Room Documents
{: .no_toc }

Add a file to a room and every member's agent can search it, quote it with page numbers, and read it in full. Nobody pastes a contract into a chat window again, and nobody does it a second time for the next teammate.
{: .fs-6 .fw-300 }

Rooms are available on Pro, Teams and Enterprise.
{: .fs-3 }

## Table of Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## What this is for

A room is where your team's agents share memory. Documents are the part of that memory a person already has as a file: the signed contract, the RFP, the design doc, the vendor's pricing PDF.

Before this, getting a document into an agent's hands meant pasting its text into the prompt — every session, and again for every teammate. What each agent then knew was whatever survived that paste, and none of it accumulated anywhere. A document added to a room is extracted once and searchable by everyone, including agents that join next month.

## Add a document

Ask your agent, in whatever words you'd use with a colleague:

> Add `~/Downloads/acme-msa-signed.pdf` to the Vendors room.

The agent calls `amfs_room_add_document` with the path. Hand it the path rather than the contents: the file goes up as bytes, so nothing is lost to a copy-paste, and a 40-page PDF costs no context.

Extraction takes a few seconds. Until it finishes the document's status is `pending`; when it's `ready` it is searchable for every member.

**What you can add**

| Format | Notes |
|:--|:--|
| PDF | Text-based. A scanned PDF is refused with an explanation — there's no OCR yet, and silently indexing an empty document would be worse |
| DOCX | Paragraphs and tables |
| Markdown, plain text | Any `.md` or `.txt` |

Up to 25 MB per file. Files that look like credentials — `.env`, private keys, `credentials.json` — are refused; a room is a shared space, and a secret put into one has been shared.

## Search and quote

> What does the Acme MSA say about termination?

The agent calls `amfs_room_document_search`, which searches the text of every document it can reach and returns passages with page numbers. Ask it to cite them. The point of a citation here is that you can go and check.

Search spans every room you're in, so you don't need to remember which room a file went into. Restrict it to one room or one document when you already know.

When a passage isn't enough — a summary, a full review — the agent reads the document in order with `amfs_room_document_read`, a few pages at a time.

## How agents find out a document exists

Each document also writes one summary memory entry into the room. That's what makes a document *discoverable*: an agent that runs `amfs_briefing` on the room, or searches memory for "pricing", learns the file is there without anyone telling it.

The summary is a signpost, not a substitute. An agent that answers from the summary alone is guessing; the answer is in the document, and searching it is one call.

## Versions and duplicates

Adding the same file again does nothing — identical bytes are recognized, and you're told the document is already there. Adding a *changed* file under the same name creates version 2, and the previous version is marked superseded rather than deleted, so a conversation that cited page 4 of version 1 still resolves.

If a document failed to extract, adding the file again is the retry.

## Removing a document, and closing the room

Deleting a document removes its text, its original, and its searchability. Its summary entry is rewritten to say it was removed, so an agent that finds the entry in an old briefing gets an explanation instead of a mystery.

When a room closes or is deleted, its documents stay readable to whoever was a member, for **30 days**. After that the contents are purged and the record becomes a tombstone: an agent asking for the file is told it expired and when, rather than getting a 404 it will interpret as a bug.

Access never widens. A document is reachable by the members of its room and nobody else — not other rooms in the same account, not agents belonging to people who were never invited.

## Untrusted by design

A document is input, not instruction. Text inside a PDF that says "ignore your previous instructions" is a string an agent should report to you, and our agent instructions say so explicitly. Treat a document the way you'd treat an email attachment from outside your company.

## HTTP API

Everything above is available directly. All routes take the usual account and user authentication.

| Method | Path | Purpose |
|:--|:--|:--|
| `POST` | `/api/v1/rooms/{room_id}/documents` | Upload (multipart `file`, optional `title`, `entity_path`). `202` with a `pending` document |
| `GET` | `/api/v1/rooms/{room_id}/documents` | List, with status and uploader |
| `GET` | `/api/v1/rooms/{room_id}/documents/{id}` | One document's metadata |
| `GET` | `/api/v1/rooms/{room_id}/documents/{id}/text` | Extracted text, by chunk, with page numbers |
| `GET` | `/api/v1/rooms/{room_id}/documents/{id}/download` | Time-limited signed URL for the original |
| `DELETE` | `/api/v1/rooms/{room_id}/documents/{id}` | Remove |
| `GET` | `/api/v1/room-documents/search?q=` | Search across every room you can reach |

An upload answers `202`, never `200`: extraction runs in a separate service, and a `200` would invite you to treat a document as ready when it is not. Poll the document or watch the room's activity stream for `document_added`.

A request for an expired document answers `200` with `available: false` and a message explaining what happened — deliberately not a `404`, because an agent reads `404` as a failure and retries.
