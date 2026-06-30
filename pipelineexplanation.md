```text
                                     USER
                                      │
                                      ▼
                               Create prompt
                                      │
                     ┌────────────────┴────────────────┐
                     │                                 │
                     ▼                                 ▼
          Conversation History          Relevant Semantic Memory
                     │                                 │
                     └────────────────┬────────────────┘
                                      ▼
                          Response Reasoner (LLM)
                                      │
                                      ▼
                           Generate User Response
                                      │
                    ┌─────────────────┴─────────────────┐
                    │                                   │
                    ▼                                   ▼
          Return Response                     Learning Pipeline
              to User                                 │
                                                      ▼
                                     Build Completed Interaction
                              (History + User Input + Assistant Response)
                                                      │
                                                      ▼
                                     Knowledge Extractor (LLM)
                                                      │
                                                      ▼
                                         Knowledge Candidates
                                              (temporary)
                                                      │
                                                      ▼
                                           Simple Validator
                                                      │
              ┌───────────────────────────┼───────────────────────────┐
              │                           │                           │
              ▼                           ▼                           ▼
        Exact Duplicate            New Independent Fact      Possible Conflict
              │                           │                           │
              ▼                           ▼                           ▼
           Reject               Store in Semantic Memory   Memory Reconciler
                                                             (Future)
                                                                  │
                                                                  ▼
                                                        Semantic Memory
                                                              Updated
                                                                  
                                                                  
                                                           
```
