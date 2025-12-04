"""
Pytest Configuration and Fixtures

This module provides shared fixtures and configuration for the test suite.
"""

import os
import pytest
from PIL import Image
from unittest.mock import Mock, MagicMock
from typing import Generator

# Mark integration tests
def pytest_configure(config):
    """Add custom markers."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (require external dependencies)"
    )


# ============================================
# Environment Variables and Conditions
# ============================================

def has_vlm_credentials() -> bool:
    """Check if VLM API credentials are available."""
    return bool(os.getenv('VLM_API_KEY')) and bool(os.getenv('VLM_BASE_URL'))


def has_adb_device() -> bool:
    """Check if an ADB device is connected."""
    try:
        import adbutils
        adb = adbutils.AdbClient(host="127.0.0.1", port=5037)
        devices = adb.device_list()
        return len(devices) > 0
    except Exception:
        return False


# Skip conditions
skip_if_no_vlm = pytest.mark.skipif(
    not has_vlm_credentials(),
    reason="VLM_API_KEY and VLM_BASE_URL environment variables required"
)

skip_if_no_device = pytest.mark.skipif(
    not has_adb_device(),
    reason="No ADB device connected"
)


# ============================================
# Mock Fixtures
# ============================================

@pytest.fixture
def mock_image() -> Image.Image:
    """Create a mock PIL Image for testing."""
    return Image.new('RGB', (1080, 1920), color='white')


@pytest.fixture
def mock_image_small() -> Image.Image:
    """Create a small mock PIL Image for testing."""
    return Image.new('RGB', (100, 100), color='blue')


@pytest.fixture
def mock_vlm_response() -> Mock:
    """Create a mock VLM response."""
    response = Mock()
    response.choices = [Mock()]
    response.choices[0].message = Mock()
    response.choices[0].message.content = "Mock response content"
    response.choices[0].delta = Mock()
    response.choices[0].delta.content = "Mock"
    return response


@pytest.fixture
def mock_vlm_wrapper() -> Mock:
    """Create a mock VLMWrapper."""
    vlm = Mock()
    vlm.model_name = "mock-model"
    vlm.max_tokens = 1024
    vlm.temperature = 0.0
    
    # Mock predict method
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message = Mock()
    mock_response.choices[0].message.content = """<thinking>I need to click on the Photos app.</thinking>
<tool_call>{"name": "mobile_use", "arguments": {"action": "click", "point": [540, 960]}}</tool_call>
<conclusion>Clicking on Photos app icon.</conclusion>"""
    vlm.predict = Mock(return_value=mock_response)
    
    return vlm


@pytest.fixture
def mock_env_state(mock_image) -> 'EnvState':
    """Create a mock EnvState for testing."""
    from mobile_use.schema.schema import EnvState
    return EnvState(
        pixels=mock_image,
        package="com.example.app",
        a11y_tree=None,
        device_time="Thu Dec 4 10:00:00 GMT 2025"
    )


@pytest.fixture
def mock_environment(mock_env_state) -> Mock:
    """Create a mock Environment for testing."""
    env = Mock()
    env.host = "127.0.0.1"
    env.port = 5037
    env.serial_no = "emulator-5554"
    env.window_size = (1080, 1920)
    env._action_space = ['open', 'click', 'long_press', 'type', 'key',
                         'swipe', 'press_home', 'press_back', 'wait',
                         'answer', 'system_button', 'clear_text', 'take_note']
    
    # Mock methods
    env.get_state = Mock(return_value=mock_env_state)
    env.reset = Mock()
    env.execute_action = Mock()
    
    return env


# ============================================
# Sample Data Fixtures
# ============================================

@pytest.fixture
def sample_action() -> 'Action':
    """Create a sample Action for testing."""
    from mobile_use.schema.schema import Action
    return Action(name='click', parameters={'coordinate': [540, 960]})


@pytest.fixture
def sample_vlm_config() -> dict:
    """Create sample VLM config dict."""
    return {
        'model_name': 'qwen2.5-vl-72b-instruct',
        'api_key': 'test-api-key',
        'base_url': 'https://api.example.com/v1',
        'max_retry': 3,
        'retry_waiting_seconds': 2,
        'max_tokens': 1024,
        'temperature': 0.0,
    }


@pytest.fixture
def sample_env_config() -> dict:
    """Create sample environment config dict."""
    return {
        'serial_no': 'emulator-5554',
        'host': '127.0.0.1',
        'port': 5037,
        'go_home': False,
        'wait_after_action_seconds': 2.0,
    }


@pytest.fixture
def sample_agent_config(sample_vlm_config, sample_env_config) -> dict:
    """Create sample agent config dict."""
    return {
        'vlm': sample_vlm_config,
        'env': sample_env_config,
        'max_steps': 10,
        'enable_log': False,
        'log_dir': None,
    }


# ============================================
# YAML Config Fixtures
# ============================================

@pytest.fixture
def temp_yaml_config(tmp_path, sample_vlm_config, sample_env_config) -> str:
    """Create a temporary YAML config file for testing."""
    import yaml
    
    config = {
        'vlm': sample_vlm_config,
        'env': sample_env_config,
        'max_steps': 10,
        'enable_log': False,
    }
    
    config_path = tmp_path / "test_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    return str(config_path)


# ============================================
# Test Data Directory Fixtures
# ============================================

@pytest.fixture
def test_data_dir(tmp_path) -> str:
    """Create a temporary directory for test data."""
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()
    return str(data_dir)


@pytest.fixture
def test_log_dir(tmp_path) -> str:
    """Create a temporary directory for logs."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    return str(log_dir)

