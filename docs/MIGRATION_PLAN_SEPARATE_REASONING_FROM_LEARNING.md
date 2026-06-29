# Migration Plan: Separate Reasoning from Learning

## Current Architecture

### Data Flow in ThoughtPipeline.process()

```
User Input
    │
    ▼
Thought(user_input=message)
    │
    ▼
_initialize()                          # Stage 1: Init metadata
_load_memory()                         # Stage 2: Load episodic memories → thought.memories
_load_knowledge()                      # Stage 3: Load semantic memory (retrieved knowledge) → thought.knowledge
_extract_candidates()                  # Stage 3a: Extract candidate facts from conversation
_load_candidates()                     # Stage 3b: Load candidates into thought.candidates
_reason()                              # Stage 4
_plan()                                # Stage 5
_prepare_tools()                       # Stage 6
CognitiveEngine.process()              # Generates LLM response (PROMPT INCLUDES CANDIDATES)
_build_response()                      # Stage 7: Set thought.response
_validate_knowledge()                  # Stage 8: Validate & promote verified knowledge to semantic memory
_reflect()                             # Stage 9
_finalize()                            # Stage 10
```

### Key Observation: Candidates Influence Reasoning

In the current architecture, `_extract_candidates()` and `_load_candidates()` run BEFORE reasoning. This means candidate facts are included in the LLM prompt (via PromptBuilder.build()), directly influencing the generated response.

Evidence in [`athena/prompt/builder.py`](athena/prompt/builder.py:49-60):
```python
candidates = getattr(thought, 'candidates', None)
if not candidates:
    lines.append("(None)")
else:
    for candidate in thought.candidates:
        # Format: statement (confidence=X, category=Y)
        ...
```

This violates the architectural principle: **"Learning must never change the generated response."**

### Current Candidate Lifecycle

1. [`KnowledgeManager.extract_candidates()`](athena/knowledge/manager.py:66-91) - extracts facts from conversation via LLM provider
2. Stored in WorkingMemory as candidate entries (prefixed "CANDIDATE:...")
3. Loaded into `thought.candidates` by `_load_candidates()`
4. Included in prompt for reasoning phase
5. Validated and promoted to semantic memory by `_validate_knowledge()`

## New Architecture

```
                    USER
                      │
                      ▼
               Current Thought (Working Memory)
                      │
                      ▼
           Load Semantic Memory ONLY
                      │
                      ▼
              Reasoning / LLM (Response generated WITHOUT candidates)
                      │
               Generate Response
                 /            \
                /              \
               ▼                ▼
     Return Response     Knowledge Extraction
         to User                  │
                                   ▼
                          Knowledge Validation
                                   │
                                   ▼
                          Update Semantic Memory
```

### Key Changes

1. **Reasoning Phase** (unchanged): Load episodic + semantic memory → Reason → Generate response
2. **Learning Phase** (separated): Extract candidates → Validate → Update semantic memory
3. Candidates are NO LONGER part of the reasoning prompt
4. Learning is a separate responsibility from reasoning

## Migration Plan

### Step 1: Reorder Pipeline Stages (SMALLEST SAFE REFACTOR)

Move `_extract_candidates()` and `_load_candidates()` to AFTER `_build_response()`.

**Files modified:** [`athena/thought/pipeline.py`](athena/thought/pipeline.py)

**Changes:**
- In `process()` method: Move `_extract_candidates()` and `_load_candidates()` calls to after `_build_response()`
- Update docstring to reflect new architecture
- Add comments distinguishing reasoning phase from learning phase

### Step 2 (Future): Clean Up PromptBuilder

After Step 1 is verified, the candidates section in [`athena/prompt/builder.py`](athena/prompt/builder.py:49-60) will always show "(None)" since candidates are loaded after response generation. This can be cleaned up or kept for post-response analysis.

### Step 3 (Future): Add Explicit Learning Method

Add a new method `learn()` to ThoughtPipeline that encapsulates the learning phase:
```python
def learn(self, thought: Thought) -> None:
    """Learning phase: extract, load, validate, and promote knowledge."""
    self._extract_candidates(thought)
    self._load_candidates(thought)
    self._validate_knowledge(thought)
```

## Verification

Run the verification test:
```bash
python athena/thought/full_verification.py
```

This verifies that WorkingMemory candidates are correctly exposed to Thought.candidates during processing.

## Architectural Principles Preserved

- ✅ WorkingMemory preserved as general runtime component
- ✅ KnowledgeManager preserved with its extract/retrieve interface  
- ✅ SemanticMemory preserved with learn/query interface
- ✅ Only moved code that made candidate facts participate in reasoning
- ✅ Removed candidates from pre-reasoning phase
- ✅ Learning is now a separate responsibility

## Files Modified

1. [`athena/thought/pipeline.py`](athena/thought/pipeline.py) - Reorder stages in process() method

## Verification Steps

1. Run `python athena/thought/full_verification.py` - should pass
2. Run `python athena/thought/verify_full_flow.py` - should pass  
3. Verify that response generation no longer includes candidate facts in prompt
4. Verify that knowledge extraction happens after response is generated
