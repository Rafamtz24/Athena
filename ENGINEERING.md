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

Brain

Thought

Thought Pipeline

Memory

Events

Future Cognitive Engine

Future Providers

Future Tools

## Definition of Done

Every ticket should finish with:

Files created

Files modified

Architecture summary

Verification summary

Design notes