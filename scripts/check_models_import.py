import sys

sys.path.insert(0, r'c:/Users/scart/footie')

try:
    import models
    ml = getattr(models, 'ml_integration', None)
    if ml is None:
        print('ml_integration: MISSING')
    else:
        try:
            print('available_models:', ml.get_available_models())
        except Exception as e:
            print('get_available_models error:', e)
        print('create_predictor callable:', hasattr(ml, 'create_predictor'))
        try:
            # Try to create enhanced pipeline if available
            pred = None
            if 'enhanced_pipeline' in ml.get_available_models():
                pred = ml.create_predictor('enhanced_pipeline')
                print('enhanced_pipeline instance created:', type(pred))
        except Exception as e:
            print('create_predictor(enhanced_pipeline) error:', e)
except Exception as e:
    import traceback
    traceback.print_exc()
    print('IMPORT_ERROR:', e)
