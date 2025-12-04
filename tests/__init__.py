"""
Mobile Use Test Suite

This test suite provides comprehensive testing for the mobile-use framework.

Test Structure:
- test_schema.py: Tests for data schemas (EnvState, Action, StepData, etc.)
- test_config.py: Tests for configuration classes
- test_utils.py: Tests for utility functions
- test_vlm.py: Tests for VLM wrapper (requires API credentials)
- test_environment.py: Tests for mobile environment (requires connected device)
- test_agents.py: Tests for agent implementations

Running Tests:
    # Run all tests
    pytest tests/
    
    # Run only unit tests (no external dependencies)
    pytest tests/ -m "not integration"
    
    # Run integration tests (requires device/API)
    pytest tests/ -m "integration"
"""

from dotenv import load_dotenv

load_dotenv()
