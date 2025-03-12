# Run all tests
python -m unittest tests/test_model.py

# Run specific test
python -m unittest tests.test_model.TestHAMF.test_forward_pass

# Run with verbosity
python -m unittest -v tests/test_model.py