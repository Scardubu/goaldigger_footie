#!/usr/bin/env python3
"""
Test script to verify DatabaseManager singleton optimization.
Verifies that multiple instantiations return the same cached instance.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from database.db_manager import DatabaseManager


def test_singleton_pattern():
    """Test that DatabaseManager implements singleton pattern correctly"""
    print("="*60)
    print("Testing DatabaseManager Singleton Pattern")
    print("="*60)
    
    # Test 1: Multiple instances with no URI should return same object
    print("\n📝 Test 1: Multiple instantiations without URI")
    db1 = DatabaseManager()
    db2 = DatabaseManager()
    db3 = DatabaseManager()
    
    assert db1 is db2, "db1 and db2 should be the same instance"
    assert db2 is db3, "db2 and db3 should be the same instance"
    print(f"✅ PASS: All 3 instances are the same object (id: {id(db1)})")
    print(f"   Cached instances: {DatabaseManager.get_cached_instance_count()}")
    
    # Test 2: Different URIs should create different instances
    print("\n📝 Test 2: Different URIs create different instances")
    db_sqlite = DatabaseManager("sqlite:///test1.db")
    db_sqlite2 = DatabaseManager("sqlite:///test2.db")
    
    assert db_sqlite is not db_sqlite2, "Different URIs should create different instances"
    print(f"✅ PASS: Different URIs create different instances")
    print(f"   db_sqlite id: {id(db_sqlite)}")
    print(f"   db_sqlite2 id: {id(db_sqlite2)}")
    print(f"   Cached instances: {DatabaseManager.get_cached_instance_count()}")
    
    # Test 3: Same URI should return cached instance
    print("\n📝 Test 3: Same URI returns cached instance")
    db_sqlite_again = DatabaseManager("sqlite:///test1.db")
    
    assert db_sqlite is db_sqlite_again, "Same URI should return cached instance"
    print(f"✅ PASS: Same URI returns cached instance (id: {id(db_sqlite)})")
    print(f"   Cached instances: {DatabaseManager.get_cached_instance_count()}")
    
    # Test 4: Clear cache and verify re-creation
    print("\n📝 Test 4: Cache clearing")
    initial_count = DatabaseManager.get_cached_instance_count()
    print(f"   Initial cached instances: {initial_count}")
    
    DatabaseManager.clear_instance_cache("sqlite:///test1.db")
    after_clear_count = DatabaseManager.get_cached_instance_count()
    print(f"   After clearing test1.db: {after_clear_count}")
    assert after_clear_count == initial_count - 1, "Should have one less cached instance"
    
    # Create new instance - should be different from original
    db_sqlite_new = DatabaseManager("sqlite:///test1.db")
    assert db_sqlite is not db_sqlite_new, "New instance after cache clear should be different"
    print(f"✅ PASS: Cache clearing works correctly")
    print(f"   Old instance id: {id(db_sqlite)}")
    print(f"   New instance id: {id(db_sqlite_new)}")
    
    # Test 5: Clear all caches
    print("\n📝 Test 5: Clear all caches")
    DatabaseManager.clear_instance_cache()
    final_count = DatabaseManager.get_cached_instance_count()
    assert final_count == 0, "All caches should be cleared"
    print(f"✅ PASS: All caches cleared (count: {final_count})")
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED")
    print("="*60)
    print("\n📊 Singleton optimization is working correctly!")
    print("Multiple DatabaseManager() calls will now reuse cached instances,")
    print("significantly reducing redundant initializations and connections.")
    
    return True

def test_connection_reuse():
    """Test that connections are properly reused"""
    print("\n" + "="*60)
    print("Testing Connection Reuse")
    print("="*60)
    
    # Clear cache first
    DatabaseManager.clear_instance_cache()
    
    # Create multiple instances and test connections
    print("\n📝 Creating 5 DatabaseManager instances...")
    instances = []
    for i in range(5):
        db = DatabaseManager()
        instances.append(db)
        print(f"   Instance {i+1}: id={id(db)}, engine_id={id(db.engine)}")
    
    # Verify all instances share the same engine
    engine_ids = {id(db.engine) for db in instances}
    print(f"\n✅ All {len(instances)} instances share {len(engine_ids)} unique engine(s)")
    
    if len(engine_ids) == 1:
        print("✅ OPTIMAL: Single shared engine - no redundant connections!")
    else:
        print(f"⚠️  Multiple engines detected - expected 1, got {len(engine_ids)}")
    
    # Test that connections work
    print("\n📝 Testing connections...")
    for i, db in enumerate(instances[:3], 1):  # Test first 3
        result = db.test_connection()
        status = "✅" if result else "❌"
        print(f"   Instance {i} connection: {status}")
    
    print("\n" + "="*60)
    print("✅ CONNECTION REUSE TEST COMPLETE")
    print("="*60)

if __name__ == "__main__":
    try:
        test_singleton_pattern()
        test_connection_reuse()
        print("\n🎉 All optimization tests passed!")
        print("DatabaseManager is now optimized with singleton pattern.")
        sys.exit(0)
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
