                                      USER
                                       │
                                       ▼
                                Create Thought
                                       │
                       ┌───────────────┴────────────────┐
                       │                                │
                       ▼                                ▼
         Working Memory (conversation)      Semantic Memory (Knowledge)
                       │                                │
                       └───────────────┬────────────────┘
                                       │
                                       ▼
                               Tool Planner
                                       │
                                       ▼
                                Tool Router
                                       │
                                       ▼
                               Tool Context
                                       │
                                       ▼
                          ┌──────────────────────┐
                          │ Context Budget       │
                          │ Manager              │
                          │                      │
                          │ Phase 1:             │
                          │  Compute WM budget   │
                          │  → WorkingMemory     │
                          │    .prune()          │
                          │                      │
                          │ Phase 2:             │
                          │  Compile sources     │
                          │  → Priority order    │
                          │  → Budget within     │
                          │    context window    │
                          └──────────┬───────────┘
                                     │
                    ┌────────────────┴────────────────┐
                    │                                 │
                    ▼                                 ▼
         Reasoning Package                 Learning Package
                    │                                 │
                    ▼                                 ▼
          PromptBuilder                    Knowledge Extractor
                    │                                 │
                    ▼                                 ▼
        Reasoning LLM (generate)          Knowledge Candidates
                    │                                 │
                    ▼                                 │
          Generate Response                           │
                    │                                 │
                    ├──────────► Return Response      │
                    │                                 │
                    ▼                                 ▼
          Build Completed Interaction      Knowledge Validator
                    │                                 │
                    │                                 ▼
                    │                    ┌────────────┼────────────┐
                    │                    │            │            │
                    │                    ▼            ▼            ▼
                    │              Exact        New          Possible
                    │           Duplicate    Independent    Conflict
                    │                         Fact
                    │                    │            │            │
                    │                    ▼            ▼            ▼
                    │                Reject    Store in     Memory
                    │                           Semantic    Reconciler
                    │                           Memory          │
                    │                                            ▼
                    │                                  Semantic Memory
                    │                                      Updated
                    │
                    ▼
          WorkingMemory.prune()
          (post-response eviction)

          working_memory.json saved
```

## Pipeline Stages (in order)

| # | Stage | Component | Responsibility |
|---|-------|-----------|----------------|
| 1 | _initialize | ThoughtPipeline | Set metadata, publish ThoughtCreated |
| 2 | _load_knowledge | KnowledgeManager | Retrieve Semantic Memory into thought.knowledge |
| 3 | _plan_tool | Tool Planner | Decide if a tool is needed |
| 4 | _execute_tool | Tool Router | Execute tool, produce ToolContext |
| 5 | _reason | ThoughtPipeline | Publish ReasoningStarted |
| 6 | _plan | ThoughtPipeline | Publish PlanningStarted |
| 7 | _prepare_tools | ThoughtPipeline | Verify tool context, publish ToolsPrepared |
| 8 | _budget_context | ContextBudgetManager | Phase 1: prune WM; Phase 2: compile packages |
| 9 | CognitiveEngine | PromptBuilder + LLM | Render prompt from ReasoningPackage, generate response |
| 10 | _build_response | ThoughtPipeline | Publish ResponseGenerated |
| 11 | _extract_candidates | KnowledgeManager | Extract knowledge from LearningContextPackage |
| 12 | _validate_knowledge | Validator + Reconciler | Quality gate → LLM reconciliation → Semantic Memory |
| 13 | _reflect | ThoughtPipeline | Self-evaluate outcome |
| 14 | _finalize | ThoughtPipeline | Publish ThoughtCompleted |

The conversation window is not loaded by a stage: `AthenaBrain` copies it onto the Thought before the pipeline runs. An episodic store that once supplied a second, verbatim copy of the same turns has been removed.

## Key Components

### Context Budget Manager
- Compiles all context sources into Reasoning and Learning packages
- Uses provider's `get_context_window()` and `count_tokens()`
- Orders sources by priority (100 = User Input, 60 = Chat History)
- Reserves generation budget before prompt allocation
- Computes Working Memory's budget from actual remaining prompt space
- Never rewrites, summarizes, or paraphrases
- No tool-specific hardcoded rules

### Working Memory
- Sliding window of conversation history
- Session-scoped: reset to empty at the start of each session (durable facts live in Semantic Memory)
- Self-prunes via `prune(max_tokens, entries)` instance method
- Pruned in-pipeline (by Context Budget Manager) and post-response (by Brain)
- Also stores temporary knowledge candidates

### Knowledge Extractor
- Consumes only `LearningContextPackage`
- Completely unaware of individual tools
- Builds extraction prompt from the package contents
- Returns KnowledgeCandidates to Working Memory
