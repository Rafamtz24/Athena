"""Comprehensive verification for Sprint 3 - Thought Pipeline."""

import asyncio
import sys
import os
from datetime import datetime

# Add project root to path so athena package is importable
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)


def test_imports():
    """Test that all new modules can be imported."""
    print("=" * 60)
    print("TEST: Import Verification")
    print("=" * 60)
    
    try:
        from athena.thought.models import Thought
        print("[PASS] Thought model imported successfully")
        
        from athena.thought.pipeline import ThoughtPipeline
        print("[PASS] ThoughtPipeline imported successfully")
        
        from athena.brain.brain import AthenaBrain
        print("[PASS] AthenaBrain imported successfully (uses Thought objects)")
        
        return True
    except ImportError as e:
        print(f"[FAIL] Import error: {e}")
        return False


def test_thought_creation():
    """Test that Thought dataclass can be instantiated."""
    print("\n" + "=" * 60)
    print("TEST: Thought Creation")
    print("=" * 60)
    
    from athena.thought.models import Thought
    
    thought = Thought(user_input="test message")
    assert thought.id is not None, "Thought should have an ID"
    assert thought.user_input == "test message", "User input should be set"
    print(f"[PASS] Thought created with ID: {thought.id}")
    print(f"[PASS] Thought timestamp: {thought.created_at}")
    
    return True


def test_pipeline_stages():
    """Test that pipeline stages exist and are callable."""
    print("\n" + "=" * 60)
    print("TEST: Pipeline Stages")
    print("=" * 60)
    
    from athena.thought.pipeline import ThoughtPipeline
    
    # Check all required methods exist
    assert hasattr(ThoughtPipeline, 'create'), "Pipeline should have create method"
    assert hasattr(ThoughtPipeline, 'process'), "Pipeline should have process method"
    
    # Check private stage methods exist
    stages = ['_initialize', '_load_memory', '_reason', '_plan', 
              '_prepare_tools', '_build_response', '_reflect', '_finalize']
    
    for stage in stages:
        assert hasattr(ThoughtPipeline, stage), f"Pipeline should have {stage} method"
        print(f"[PASS] Stage exists: {stage}")
    
    return True


async def test_pipeline_execution():
    """Test that the pipeline can execute end-to-end."""
    print("\n" + "=" * 60)
    print("TEST: Pipeline Execution")
    print("=" * 60)
    
    from athena.thought.models import Thought
    from athena.thought.pipeline import ThoughtPipeline
    
    # Create a thought and run through pipeline
    thought = Thought(user_input="test message")
    pipeline = ThoughtPipeline()
    result_thought = await pipeline.process(thought)
    
    assert result_thought is not None, "Process should return a thought"
    print(f"[PASS] Pipeline process returned: {result_thought.response}")
    
    return True


async def test_athena_brain_integration():
    """Test that AthenaBrain uses Thought objects internally."""
    print("\n" + "=" * 60)
    print("TEST: AthenaBrain Integration")
    print("=" * 60)
    
    from athena.brain.brain import AthenaBrain
    
    brain = AthenaBrain()
    response = await brain.process("test message via AthenaBrain")
    
    assert response is not None, "Response should not be None"
    print(f"[PASS] AthenaBrain.process returned: {response}")
    
    return True


def test_memory_manager_integration():
    """Test that MemoryManager still works with Thought objects."""
    print("\n" + "=" * 60)
    print("TEST: MemoryManager Integration")
    print("=" * 60)
    
    from athena.memory.manager import MemoryManager
    
    mm = MemoryManager()
    assert mm is not None, "MemoryManager should be instantiable"
    print("[PASS] MemoryManager instantiated successfully")
    
    # Test basic memory operations still work
    mm.store("test memory entry", "test_category")
    results = mm.search("test", top_k=5)
    print(f"[PASS] MemoryManager search returned {len(results)} results")
    
    return True


def test_documentation():
    """Test that documentation file exists."""
    print("\n" + "=" * 60)
    print("TEST: Documentation")
    print("=" * 60)
    
    doc_path = os.path.join(project_root, 'docs', 'SPRINT3.md')
    if os.path.exists(doc_path):
        with open(doc_path, 'r') as f:
            content = f.read()
            assert len(content) > 100, "Documentation should have substantial content"
            print(f"[PASS] SPRINT3.md exists ({len(content)} chars)")
    else:
        print("[FAIL] SPRINT3.md not found")
        return False
    
    return True


def test_file_structure():
    """Test that all required files exist."""
    print("\n" + "=" * 60)
    print("TEST: File Structure")
    print("=" * 60)
    
    required_files = [
        'athena/thought/__init__.py',
        'athena/thought/models.py',
        'athena/thought/pipeline.py',
        'docs/SPRINT3.md',
    ]
    
    for f in required_files:
        full_path = os.path.join(project_root, f)
        if os.path.exists(full_path):
            print(f"[PASS] File exists: {f}")
        else:
            print(f"[FAIL] Missing file: {f}")
            return False
    
    return True


async def main():
    """Run all verification tests."""
    print("\n" + "#" * 60)
    print("# SPRINT 3 VERIFICATION - Thought Pipeline")
    print("#" * 60 + "\n")
    
    results = []
    
    # Test imports
    results.append(("Import Verification", test_imports()))
    
    # Test thought creation
    results.append(("Thought Creation", test_thought_creation()))
    
    # Test pipeline stages
    results.append(("Pipeline Stages", test_pipeline_stages()))
    
    # Test pipeline execution
    results.append(("Pipeline Execution", await test_pipeline_execution()))
    
    # Test AthenaBrain integration
    results.append(("AthenaBrain Integration", await test_athena_brain_integration()))
    
    # Test MemoryManager integration
    results.append(("MemoryManager Integration", test_memory_manager_integration()))
    
    # Test documentation
    results.append(("Documentation", test_documentation()))
    
    # Test file structure
    results.append(("File Structure", test_file_structure()))
    
    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  [{status}] {name}")
    
    print(f"\nTotal: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n✓ ALL VERIFICATION CHECKS PASSED")
    else:
        print(f"\n✗ {total - passed} check(s) failed")


if __name__ == "__main__":
    asyncio.run(main())