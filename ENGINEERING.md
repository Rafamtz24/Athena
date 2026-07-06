# Athena Engineering Manual

## Purpose

Athena is a modular, local-first cognitive operating system.

The goal is to build clean architecture that can evolve for years.

The architecture is more important than adding features quickly.

## Engineering Principles

- Single Responsibility Principle

- Modular architecture

- Provider agnostic

- Local-first

- Every subsystem must be replaceable

- Brain coordinates but does not reason

- Thought is the object that moves through the system

- Memory is accessed through MemoryManager

- Modules should communicate through events whenever practical

## Workflow

Every task follows this order:

Architecture

↓

Implementation

↓

Verification

↓

Review

↓

Commit

Never skip steps.

## Rules for Implementation

Never delete existing files unless explicitly instructed.

Never rename modules without approval.

Do not inspect the repository unless requested.

Only modify files required by the current ticket.

Prefer small focused changes.

Keep functions small.

Add type hints when practical.

Document public classes.

## Coding Style

Favor readability over cleverness.

Avoid duplication.

Keep abstractions simple.

Do not over-engineer.

## Current Core Architecture

- Brain (`brain/brain.py`) — Coordinates the cognitive pipeline
- Thought (`thought/models.py`) — Temporary cognitive workspace
- Thought Pipeline (`thought/pipeline.py`) — 15-stage processing pipeline
- Cognitive Engine (`cognition/engine.py`) — PromptBuilder + LLM provider
- Context Budget Manager (`context/manager.py`) — Compiles budgeted context packages
- Memory (`memory/`) — Episodic, Semantic, Working
- Knowledge System (`knowledge/`) — Extraction, Validation, Reconciliation
- Tool Planner (`planner/planner.py`) — Decides if a tool is needed
- Tool Router (`tools/router.py`) — Executes tools, produces ToolContext
- Providers (`providers/`) — LlamaCpp, LM Studio (swappable via config)
- PromptBuilder (`prompt/builder.py`) — Renders ReasoningContextPackage
- Event Bus (`events/bus.py`) — Publish-subscribe for module communication
- Hardware Detector (`hardware/detector.py`) — Auto-detects CPU/GPU/RAM
- Debug Manager (`debug/manager.py`) — Debug utilities

## Definition of Done

Every ticket should finish with:

Files created

Files modified

Architecture summary

Verification summary

Design notes