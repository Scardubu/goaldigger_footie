import runpy
import sys

try:
    runpy.run_path('tests/test_enhanced_aggregator.py', run_name='__main__')
    print('TESTS: OK')
except AssertionError as e:
    print('TESTS: FAILED', e)
    sys.exit(1)
except Exception as e:
    print('TESTS: ERROR', e)
    sys.exit(2)
