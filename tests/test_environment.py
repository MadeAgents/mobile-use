"""
Tests for mobile_use/environment/mobile_environ.py

Tests the Environment class for interacting with Android devices via ADB.
Includes both unit tests with mocks and integration tests that require a connected device.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from PIL import Image

from mobile_use.environment.mobile_environ import Environment
from mobile_use.schema.schema import Action, EnvState


class TestEnvironmentInit:
    """Tests for Environment initialization."""

    def test_init_with_mocked_device(self):
        """Test Environment initialization with mocked adbutils."""
        with patch('mobile_use.environment.mobile_environ.adbutils') as mock_adb:
            # Setup mocks
            mock_device = MagicMock()
            mock_device.window_size.return_value = (1080, 1920)
            mock_adb.AdbClient.return_value.device.return_value = mock_device

            env = Environment(
                serial_no='emulator-5554',
                host='127.0.0.1',
                port=5037,
                go_home=False
            )

            assert env.host == '127.0.0.1'
            assert env.port == 5037
            assert env.serial_no == 'emulator-5554'
            assert env.window_size == (1080, 1920)

    def test_init_with_custom_wait_time(self):
        """Test Environment with custom wait_after_action_seconds."""
        with patch('mobile_use.environment.mobile_environ.adbutils') as mock_adb:
            mock_device = MagicMock()
            mock_device.window_size.return_value = (1080, 1920)
            mock_adb.AdbClient.return_value.device.return_value = mock_device

            env = Environment(
                serial_no='test',
                wait_after_action_seconds=3.5
            )

            assert env.wait_after_action_seconds == 3.5

    def test_init_with_go_home(self):
        """Test Environment initialization with go_home=True."""
        with patch('mobile_use.environment.mobile_environ.adbutils') as mock_adb:
            mock_device = MagicMock()
            mock_device.window_size.return_value = (1080, 1920)
            mock_adb.AdbClient.return_value.device.return_value = mock_device

            env = Environment(serial_no='test', go_home=True)

            # Verify HOME keyevent was called
            mock_device.keyevent.assert_called_with("HOME")


class TestEnvironmentActionSpace:
    """Tests for Environment action_space property."""

    def test_action_space_property(self):
        """Test that action_space returns expected actions."""
        with patch('mobile_use.environment.mobile_environ.adbutils') as mock_adb:
            mock_device = MagicMock()
            mock_device.window_size.return_value = (1080, 1920)
            mock_adb.AdbClient.return_value.device.return_value = mock_device

            env = Environment(serial_no='test')
            
            expected_actions = [
                'open', 'click', 'long_press', 'type', 'key',
                'swipe', 'press_home', 'press_back', 'wait',
                'answer', 'system_button', 'clear_text', 'take_note'
            ]
            
            for action in expected_actions:
                assert action in env.action_space


class TestEnvironmentRegisterAction:
    """Tests for Environment register_action method."""

    def test_register_custom_action(self):
        """Test registering a custom action."""
        with patch('mobile_use.environment.mobile_environ.adbutils') as mock_adb:
            mock_device = MagicMock()
            mock_device.window_size.return_value = (1080, 1920)
            mock_adb.AdbClient.return_value.device.return_value = mock_device

            env = Environment(serial_no='test')
            
            def custom_action(env, text):
                return f"Custom: {text}"
            
            env.register_action('custom', custom_action)
            
            assert 'custom' in env.action_space
            assert env._register_function['custom'] == custom_action

    def test_register_non_callable_raises(self):
        """Test that registering non-callable raises ValueError."""
        with patch('mobile_use.environment.mobile_environ.adbutils') as mock_adb:
            mock_device = MagicMock()
            mock_device.window_size.return_value = (1080, 1920)
            mock_adb.AdbClient.return_value.device.return_value = mock_device

            env = Environment(serial_no='test')
            
            with pytest.raises(ValueError, match="callable"):
                env.register_action('invalid', "not a function")


class TestEnvironmentGetState:
    """Tests for Environment get_state method."""

    def test_get_state_returns_env_state(self):
        """Test that get_state returns an EnvState object."""
        with patch('mobile_use.environment.mobile_environ.adbutils') as mock_adb:
            mock_device = MagicMock()
            mock_device.window_size.return_value = (1080, 1920)
            mock_device.screenshot.return_value = Image.new('RGB', (1080, 1920))
            mock_device.app_current.return_value.package = "com.example.app"
            mock_device.shell.return_value = "Thu Dec 4 10:00:00 GMT 2025"
            mock_adb.AdbClient.return_value.device.return_value = mock_device

            env = Environment(serial_no='test')
            state = env.get_state()

            assert isinstance(state, EnvState)
            assert state.pixels is not None
            assert state.package == "com.example.app"
            assert state.device_time is not None

    def test_get_state_screenshot_error(self):
        """Test get_state raises error on screenshot failure."""
        with patch('mobile_use.environment.mobile_environ.adbutils') as mock_adb:
            mock_device = MagicMock()
            mock_device.window_size.return_value = (1080, 1920)
            mock_device.screenshot.side_effect = Exception("Screenshot failed")
            mock_adb.AdbClient.return_value.device.return_value = mock_device

            env = Environment(serial_no='test')
            
            with pytest.raises(ValueError, match="screenshot"):
                env.get_state()


class TestEnvironmentReset:
    """Tests for Environment reset method."""

    def test_reset_with_go_home(self):
        """Test reset with go_home=True."""
        with patch('mobile_use.environment.mobile_environ.adbutils') as mock_adb:
            mock_device = MagicMock()
            mock_device.window_size.return_value = (1080, 1920)
            mock_adb.AdbClient.return_value.device.return_value = mock_device

            env = Environment(serial_no='test', go_home=False)
            mock_device.keyevent.reset_mock()
            
            env.reset(go_home=True)
            mock_device.keyevent.assert_called_with("HOME")

    def test_reset_without_go_home(self):
        """Test reset with go_home=False."""
        with patch('mobile_use.environment.mobile_environ.adbutils') as mock_adb:
            mock_device = MagicMock()
            mock_device.window_size.return_value = (1080, 1920)
            mock_adb.AdbClient.return_value.device.return_value = mock_device

            env = Environment(serial_no='test', go_home=False)
            mock_device.keyevent.reset_mock()
            
            env.reset(go_home=False)
            mock_device.keyevent.assert_not_called()


class TestEnvironmentExecuteAction:
    """Tests for Environment execute_action method."""

    @pytest.fixture
    def mocked_env(self):
        """Create an Environment with mocked device."""
        with patch('mobile_use.environment.mobile_environ.adbutils') as mock_adb:
            mock_device = MagicMock()
            mock_device.window_size.return_value = (1080, 1920)
            mock_device.get_serialno.return_value = "emulator-5554"
            mock_adb.AdbClient.return_value.device.return_value = mock_device
            
            with patch('mobile_use.environment.mobile_environ.time.sleep'):
                env = Environment(serial_no='test', wait_after_action_seconds=0)
                yield env, mock_device

    def test_execute_click_action(self, mocked_env):
        """Test executing click action."""
        env, mock_device = mocked_env
        action = Action(name='click', parameters={'coordinate': [540, 960]})
        
        env.execute_action(action)
        mock_device.click.assert_called_with(540, 960)

    def test_execute_long_press_action(self, mocked_env):
        """Test executing long_press action."""
        env, mock_device = mocked_env
        action = Action(name='long_press', parameters={'coordinate': [540, 960], 'time': 3.0})
        
        env.execute_action(action)
        mock_device.swipe.assert_called_with(540, 960, 540, 960, duration=3.0)

    def test_execute_type_action_ascii(self, mocked_env):
        """Test executing type action with ASCII text."""
        env, mock_device = mocked_env
        action = Action(name='type', parameters={'text': 'hello'})
        
        with patch('mobile_use.environment.mobile_environ.time.sleep'):
            env.execute_action(action)
        mock_device.shell.assert_called_with(["input", "text", "hello"])

    def test_execute_swipe_action(self, mocked_env):
        """Test executing swipe action."""
        env, mock_device = mocked_env
        action = Action(name='swipe', parameters={
            'coordinate': [540, 960],
            'coordinate2': [540, 500]
        })
        
        env.execute_action(action)
        mock_device.swipe.assert_called_with(540, 960, 540, 500, duration=0.5)

    def test_execute_press_home_action(self, mocked_env):
        """Test executing press_home action."""
        env, mock_device = mocked_env
        mock_device.keyevent.reset_mock()
        action = Action(name='press_home', parameters={})
        
        env.execute_action(action)
        mock_device.keyevent.assert_called_with("HOME")

    def test_execute_press_back_action(self, mocked_env):
        """Test executing press_back action."""
        env, mock_device = mocked_env
        mock_device.keyevent.reset_mock()
        action = Action(name='press_back', parameters={})
        
        env.execute_action(action)
        mock_device.keyevent.assert_called_with("BACK")

    def test_execute_key_action(self, mocked_env):
        """Test executing key action."""
        env, mock_device = mocked_env
        mock_device.keyevent.reset_mock()
        action = Action(name='key', parameters={'text': 'ENTER'})
        
        env.execute_action(action)
        mock_device.keyevent.assert_called_with("ENTER")

    def test_execute_wait_action(self, mocked_env):
        """Test executing wait action."""
        env, mock_device = mocked_env
        action = Action(name='wait', parameters={'time': 5.0})
        
        with patch('mobile_use.environment.mobile_environ.time.sleep') as mock_sleep:
            env.execute_action(action)
            mock_sleep.assert_any_call(5.0)

    def test_execute_open_action(self, mocked_env):
        """Test executing open action."""
        env, mock_device = mocked_env
        action = Action(name='open', parameters={'text': 'com.example.app'})
        
        env.execute_action(action)
        mock_device.app_start.assert_called_with('com.example.app')

    def test_execute_answer_action(self, mocked_env):
        """Test executing answer action."""
        env, mock_device = mocked_env
        action = Action(name='answer', parameters={'text': 'The answer is 42'})
        
        with patch('mobile_use.environment.mobile_environ.os.system') as mock_system:
            result = env.execute_action(action)
            assert result == 'The answer is 42'

    def test_execute_take_note_action(self, mocked_env):
        """Test executing take_note action."""
        env, mock_device = mocked_env
        action = Action(name='take_note', parameters={'text': 'Important note'})
        
        result = env.execute_action(action)
        assert result == 'Important note'

    def test_execute_system_button_back(self, mocked_env):
        """Test executing system_button action with Back."""
        env, mock_device = mocked_env
        mock_device.keyevent.reset_mock()
        action = Action(name='system_button', parameters={'button': 'Back'})
        
        env.execute_action(action)
        mock_device.keyevent.assert_called_with("BACK")

    def test_execute_system_button_home(self, mocked_env):
        """Test executing system_button action with Home."""
        env, mock_device = mocked_env
        mock_device.keyevent.reset_mock()
        action = Action(name='system_button', parameters={'button': 'Home'})
        
        env.execute_action(action)
        mock_device.keyevent.assert_called_with("HOME")

    def test_execute_system_button_menu(self, mocked_env):
        """Test executing system_button action with Menu."""
        env, mock_device = mocked_env
        mock_device.keyevent.reset_mock()
        action = Action(name='system_button', parameters={'button': 'Menu'})
        
        env.execute_action(action)
        mock_device.keyevent.assert_called_with("MENU")

    def test_execute_system_button_enter(self, mocked_env):
        """Test executing system_button action with Enter."""
        env, mock_device = mocked_env
        mock_device.keyevent.reset_mock()
        action = Action(name='system_button', parameters={'button': 'Enter'})
        
        env.execute_action(action)
        mock_device.keyevent.assert_called_with("ENTER")

    def test_execute_unknown_action_raises(self, mocked_env):
        """Test that unknown action raises ValueError."""
        env, mock_device = mocked_env
        action = Action(name='unknown_action', parameters={})
        
        with pytest.raises(ValueError, match="Unknown action"):
            env.execute_action(action)

    def test_execute_registered_custom_action(self, mocked_env):
        """Test executing a registered custom action."""
        env, mock_device = mocked_env
        
        def custom_func(environment, message):
            return f"Custom: {message}"
        
        env.register_action('custom', custom_func)
        action = Action(name='custom', parameters={'message': 'test'})
        
        result = env.execute_action(action)
        assert 'Custom: test' in result


class TestEnvironmentIntegration:
    """Integration tests for Environment that require a connected device.
    
    These tests are skipped if no ADB device is connected.
    """

    @pytest.fixture
    def real_env(self):
        """Create an Environment with actual device connection."""
        try:
            import adbutils
            adb = adbutils.AdbClient(host="127.0.0.1", port=5037)
            devices = adb.device_list()
            if not devices:
                pytest.skip("No ADB device connected")
            
            serial_no = devices[0].serial
            return Environment(
                serial_no=serial_no,
                host="127.0.0.1",
                port=5037,
                go_home=False,
                wait_after_action_seconds=1.0
            )
        except Exception as e:
            pytest.skip(f"Cannot connect to ADB: {e}")

    @pytest.mark.integration
    def test_get_state_real_device(self, real_env):
        """Test getting state from real device."""
        state = real_env.get_state()
        
        assert isinstance(state, EnvState)
        assert state.pixels is not None
        assert isinstance(state.pixels, Image.Image)
        assert state.package is not None
        assert len(state.package) > 0

    @pytest.mark.integration
    def test_execute_click_real_device(self, real_env):
        """Test executing click on real device."""
        action = Action(name='click', parameters={'coordinate': [540, 960]})
        
        # Should not raise
        real_env.execute_action(action)

    @pytest.mark.integration
    def test_reset_real_device(self, real_env):
        """Test resetting real device to home."""
        real_env.reset(go_home=True)
        
        state = real_env.get_state()
        # After going home, should be on launcher
        assert state.package is not None

    @pytest.mark.integration
    def test_action_space_contains_all(self, real_env):
        """Test that action space contains all expected actions."""
        expected = ['click', 'type', 'swipe', 'press_home', 'press_back']
        for action in expected:
            assert action in real_env.action_space

