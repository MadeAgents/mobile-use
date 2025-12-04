"""
Tests for mobile_use/agents/

Tests the Agent base class and various agent implementations including
ReActAgent, QwenAgent, MultiAgent, and HierarchicalAgent.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import json

from mobile_use.agents.base import Agent
from mobile_use.agents.agent_react import ReActAgent, parse_reason_and_action
from mobile_use.agents.agent_qwen import QwenAgent, _parse_response, _parse_response_qwen3, slim_messages
from mobile_use.agents.multi_agent import MultiAgent
from mobile_use.schema.schema import (
    Action, EnvState, AgentState, AgentStatus,
    BaseStepData, SingleAgentStepData, MobileUseStepData,
    BaseEpisodeData, MobileUseEpisodeData
)


# ============================================
# Helper Functions Tests
# ============================================

class TestParseReasonAndAction:
    """Tests for ReAct agent's parse_reason_and_action function."""

    def test_parse_click_action(self):
        """Test parsing a click action."""
        content = """Thought: I need to click on the Photos icon to open it.
Action: click(point=[540, 960])"""
        
        size = (1080, 1920)
        raw_size = (1080, 1920)
        
        reason, action, action_s = parse_reason_and_action(content, size, raw_size)
        
        assert reason is not None
        assert "click" in reason.lower() or "Photos" in reason
        assert action.name == 'click'

    def test_parse_type_action(self):
        """Test parsing a type action."""
        content = """Thought: I need to type the search query.
Action: type(text="hello world")"""
        
        size = (1080, 1920)
        raw_size = (1080, 1920)
        
        reason, action, action_s = parse_reason_and_action(content, size, raw_size)
        
        assert action.name == 'type'
        assert action.parameters['text'] == "hello world"

    def test_parse_finished_action(self):
        """Test parsing a finished action."""
        content = """Thought: The task is complete.
Action: finished(answer="Task done")"""
        
        size = (1080, 1920)
        raw_size = (1080, 1920)
        
        reason, action, action_s = parse_reason_and_action(content, size, raw_size)
        
        assert action.name == 'finished'

    def test_parse_invalid_action_raises(self):
        """Test that invalid action format raises exception."""
        content = "This is not a valid action format"
        
        size = (1080, 1920)
        raw_size = (1080, 1920)
        
        with pytest.raises(Exception):
            parse_reason_and_action(content, size, raw_size)


class TestQwenParseResponse:
    """Tests for Qwen agent's _parse_response function."""

    def test_parse_click_response(self):
        """Test parsing Qwen click response."""
        content = """<thinking>I need to click on the button.</thinking>
<tool_call>{"name": "mobile_use", "arguments": {"action": "left_click", "point": [540, 960]}}</tool_call>
<conclusion>Clicking the button.</conclusion>"""
        
        size = (1080, 1920)
        raw_size = (1080, 1920)
        
        thought, action, action_s, summary = _parse_response(content, size, raw_size)
        
        assert thought == "I need to click on the button."
        assert action.name == 'click'
        assert summary == "Clicking the button."

    def test_parse_swipe_response(self):
        """Test parsing Qwen swipe response."""
        content = """<thinking>Need to scroll down.</thinking>
<tool_call>{"name": "mobile_use", "arguments": {"action": "scroll", "start_point": [540, 1500], "end_point": [540, 500]}}</tool_call>
<conclusion>Scrolling down.</conclusion>"""
        
        size = (1080, 1920)
        raw_size = (1080, 1920)
        
        thought, action, action_s, summary = _parse_response(content, size, raw_size)
        
        assert action.name == 'swipe'
        assert 'coordinate' in action.parameters
        assert 'coordinate2' in action.parameters

    def test_parse_type_response(self):
        """Test parsing Qwen type response."""
        content = """<thinking>Enter the text.</thinking>
<tool_call>{"name": "mobile_use", "arguments": {"action": "type", "content": "Hello"}}</tool_call>
<conclusion>Typing Hello.</conclusion>"""
        
        size = (1080, 1920)
        raw_size = (1080, 1920)
        
        thought, action, action_s, summary = _parse_response(content, size, raw_size)
        
        assert action.name == 'type'
        assert action.parameters['text'] == "Hello"

    def test_parse_missing_tool_call_raises(self):
        """Test that missing tool_call raises exception."""
        content = "<thinking>Something</thinking>"
        
        with pytest.raises(Exception, match="extract action"):
            _parse_response(content, (100, 100), (100, 100))


class TestQwen3ParseResponse:
    """Tests for Qwen3 agent's _parse_response_qwen3 function."""

    def test_parse_qwen3_format(self):
        """Test parsing Qwen3 format response."""
        content = """Thought: I need to click the button.
Action: Click on the submit button.
<tool_call>{"name": "mobile_use", "arguments": {"action": "left_click", "point": [500, 500]}}</tool_call>"""
        
        size = (999, 999)  # Qwen3 uses relative coordinates
        raw_size = (1080, 1920)
        
        thought, action, action_s, summary = _parse_response_qwen3(content, size, raw_size)
        
        assert thought == "I need to click the button."
        assert action.name == 'click'
        # Coordinates should be scaled from 999 to raw_size
        assert action.parameters['coordinate'][0] <= raw_size[0]
        assert action.parameters['coordinate'][1] <= raw_size[1]


class TestSlimMessages:
    """Tests for slim_messages function."""

    def test_slim_messages_keeps_latest(self):
        """Test that slim_messages keeps only the latest images."""
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": "Image 1"},
                {"type": "image_url", "image_url": {"url": "url1"}}
            ]},
            {"role": "user", "content": [
                {"type": "text", "text": "Image 2"},
                {"type": "image_url", "image_url": {"url": "url2"}}
            ]},
            {"role": "user", "content": [
                {"type": "text", "text": "Image 3"},
                {"type": "image_url", "image_url": {"url": "url3"}}
            ]},
        ]
        
        result = slim_messages(messages, num_image_limit=2)
        
        # Count images in result
        image_count = sum(
            1 for msg in result
            for content in msg['content']
            if 'image' in content.get('type', '')
        )
        assert image_count == 2

    def test_slim_messages_preserves_text(self):
        """Test that slim_messages preserves text content."""
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": "Hello"},
                {"type": "image_url", "image_url": {"url": "url1"}}
            ]},
        ]
        
        result = slim_messages(messages, num_image_limit=0)
        
        # Text should be preserved
        assert any(
            content.get('text') == "Hello"
            for msg in result
            for content in msg['content']
        )


# ============================================
# Agent Base Class Tests
# ============================================

class TestAgentRegistration:
    """Tests for Agent registration mechanism."""

    def test_agent_registered_types(self):
        """Test that agents are properly registered."""
        # These should be registered via @Agent.register decorator
        assert 'ReAct' in Agent.list_available() or Agent.by_name('ReAct') is not None
        assert 'Qwen' in Agent.list_available() or Agent.by_name('Qwen') is not None

    def test_create_agent_from_params(self):
        """Test creating agent from params dict."""
        with patch('mobile_use.agents.base.Environment'):
            with patch('mobile_use.agents.base.VLMWrapper'):
                with patch('mobile_use.agents.agent_react.load_prompt') as mock_prompt:
                    mock_prompt.return_value = MagicMock()
                    
                    agent = Agent.from_params({
                        'type': 'ReAct',
                        'max_steps': 5
                    })
                    
                    assert agent is not None
                    assert agent.max_steps == 5


# ============================================
# ReActAgent Tests
# ============================================

class TestReActAgent:
    """Tests for ReActAgent class."""

    @pytest.fixture
    def mock_react_agent(self, mock_environment, mock_vlm_wrapper, mock_env_state):
        """Create a mocked ReActAgent."""
        with patch('mobile_use.agents.agent_react.load_prompt') as mock_prompt:
            mock_prompt_obj = MagicMock()
            mock_prompt_obj.system_prompt_en = "You are an assistant."
            mock_prompt_obj.system_prompt_zh = "你是一个助手。"
            mock_prompt_obj.task_prompt = "Complete: {goal}"
            mock_prompt.return_value = mock_prompt_obj
            
            with patch('mobile_use.agents.base.Environment', return_value=mock_environment):
                with patch('mobile_use.agents.base.VLMWrapper', return_value=mock_vlm_wrapper):
                    agent = ReActAgent(max_steps=3)
                    agent.env = mock_environment
                    agent.vlm = mock_vlm_wrapper
                    
                    # Setup VLM response
                    mock_response = MagicMock()
                    mock_response.choices = [MagicMock()]
                    mock_response.choices[0].message.content = """Thought: I need to click the Photos app.
Action: click(point=[540, 960])"""
                    mock_vlm_wrapper.predict.return_value = mock_response
                    
                    return agent

    def test_react_agent_init(self, mock_react_agent):
        """Test ReActAgent initialization."""
        assert mock_react_agent.max_steps == 3
        assert mock_react_agent.state == AgentState.READY

    def test_react_agent_reset(self, mock_react_agent):
        """Test ReActAgent reset."""
        mock_react_agent.reset(goal="Open Photos", max_steps=5)
        
        assert mock_react_agent.goal == "Open Photos"
        assert mock_react_agent.max_steps == 5
        assert mock_react_agent.curr_step_idx == 0
        assert len(mock_react_agent.trajectory) == 0

    def test_react_agent_step(self, mock_react_agent, mock_env_state):
        """Test ReActAgent step execution."""
        mock_react_agent.reset(goal="Open Photos")
        mock_react_agent.env.get_state.return_value = mock_env_state
        
        step_data = mock_react_agent.step()
        
        assert step_data is not None
        assert step_data.step_idx == 0
        # VLM should have been called
        mock_react_agent.vlm.predict.assert_called()


# ============================================
# QwenAgent Tests
# ============================================

class TestQwenAgent:
    """Tests for QwenAgent class."""

    @pytest.fixture
    def mock_qwen_agent(self, mock_environment, mock_vlm_wrapper, mock_env_state):
        """Create a mocked QwenAgent."""
        with patch('mobile_use.agents.agent_qwen.load_prompt') as mock_prompt:
            mock_prompt_obj = MagicMock()
            mock_prompt_obj.system_prompt = "System: {width}x{height}"
            mock_prompt_obj.task_prompt = "Task: {goal}"
            mock_prompt_obj.history_prompt = "History: {history}"
            mock_prompt_obj.thinking_prompt = "Think step by step."
            mock_prompt.return_value = mock_prompt_obj
            
            with patch('mobile_use.agents.base.Environment', return_value=mock_environment):
                with patch('mobile_use.agents.base.VLMWrapper', return_value=mock_vlm_wrapper):
                    agent = QwenAgent(max_steps=3)
                    agent.env = mock_environment
                    agent.vlm = mock_vlm_wrapper
                    
                    # Setup VLM response
                    mock_response = MagicMock()
                    mock_response.choices = [MagicMock()]
                    mock_response.choices[0].message.content = """<thinking>I need to click.</thinking>
<tool_call>{"name": "mobile_use", "arguments": {"action": "left_click", "point": [540, 960]}}</tool_call>
<conclusion>Clicking.</conclusion>"""
                    mock_vlm_wrapper.predict.return_value = mock_response
                    
                    return agent

    def test_qwen_agent_init(self, mock_qwen_agent):
        """Test QwenAgent initialization."""
        assert mock_qwen_agent.max_steps == 3
        assert mock_qwen_agent.enable_think is True

    def test_qwen_agent_reset(self, mock_qwen_agent):
        """Test QwenAgent reset."""
        mock_qwen_agent.reset(goal="Test task", max_steps=10)
        
        assert mock_qwen_agent.goal == "Test task"
        assert mock_qwen_agent.max_steps == 10

    def test_qwen_agent_config_options(self):
        """Test QwenAgent configuration options."""
        with patch('mobile_use.agents.agent_qwen.load_prompt') as mock_prompt:
            mock_prompt.return_value = MagicMock()
            with patch('mobile_use.agents.base.Environment'):
                with patch('mobile_use.agents.base.VLMWrapper'):
                    agent = QwenAgent(
                        enable_think=False,
                        message_type='chat',
                        coordinate_type='relative',
                        num_image_limit=5
                    )
                    
                    assert agent.enable_think is False
                    assert agent.message_type == 'chat'
                    assert agent.coordinate_type == 'relative'
                    assert agent.num_image_limit == 5


# ============================================
# MultiAgent Tests
# ============================================

class TestMultiAgent:
    """Tests for MultiAgent class."""

    @pytest.fixture
    def mock_multi_agent(self, mock_environment, mock_vlm_wrapper):
        """Create a mocked MultiAgent."""
        with patch('mobile_use.agents.multi_agent.Planner'):
            with patch('mobile_use.agents.multi_agent.Operator') as MockOperator:
                with patch('mobile_use.agents.multi_agent.Reflector'):
                    with patch('mobile_use.agents.base.Environment', return_value=mock_environment):
                        with patch('mobile_use.agents.base.VLMWrapper', return_value=mock_vlm_wrapper):
                            from mobile_use.schema.config import MultiAgentConfig, OperatorConfig
                            
                            # Create minimal config
                            agent = MultiAgent(max_steps=3)
                            agent.env = mock_environment
                            agent.vlm = mock_vlm_wrapper
                            
                            return agent

    def test_multi_agent_init(self, mock_multi_agent):
        """Test MultiAgent initialization."""
        assert mock_multi_agent.max_steps == 3

    def test_multi_agent_reset(self, mock_multi_agent):
        """Test MultiAgent reset."""
        mock_multi_agent.reset(goal="Complex task", max_steps=20)
        
        assert mock_multi_agent.goal == "Complex task"
        assert mock_multi_agent.max_steps == 20
        assert len(mock_multi_agent.trajectory) == 0

    def test_multi_agent_episode_data_type(self, mock_multi_agent):
        """Test that MultiAgent uses MobileUseEpisodeData."""
        mock_multi_agent.reset(goal="Test")
        
        assert isinstance(mock_multi_agent.episode_data, MobileUseEpisodeData)


# ============================================
# Agent State Management Tests
# ============================================

class TestAgentStateManagement:
    """Tests for agent state management."""

    def test_agent_state_transitions(self, mock_environment, mock_vlm_wrapper):
        """Test agent state transitions."""
        with patch('mobile_use.agents.agent_react.load_prompt') as mock_prompt:
            mock_prompt.return_value = MagicMock()
            with patch('mobile_use.agents.base.Environment', return_value=mock_environment):
                with patch('mobile_use.agents.base.VLMWrapper', return_value=mock_vlm_wrapper):
                    agent = ReActAgent(max_steps=5)
                    agent.env = mock_environment
                    agent.vlm = mock_vlm_wrapper
                    
                    # Initial state
                    assert agent.state == AgentState.READY
                    
                    # After reset
                    agent.reset(goal="Test")
                    assert agent.state == AgentState.READY

    def test_set_max_steps(self, mock_environment, mock_vlm_wrapper):
        """Test setting max steps."""
        with patch('mobile_use.agents.agent_react.load_prompt') as mock_prompt:
            mock_prompt.return_value = MagicMock()
            with patch('mobile_use.agents.base.Environment', return_value=mock_environment):
                with patch('mobile_use.agents.base.VLMWrapper', return_value=mock_vlm_wrapper):
                    agent = ReActAgent(max_steps=5)
                    agent.set_max_steps(10)
                    
                    assert agent.max_steps == 10


# ============================================
# Integration Tests
# ============================================

class TestAgentIntegration:
    """Integration tests for agents that require real VLM and device.
    
    These tests are skipped if credentials/device are not available.
    """

    @pytest.fixture
    def real_react_agent(self):
        """Create a ReActAgent with real credentials."""
        import os
        
        api_key = os.getenv('VLM_API_KEY')
        base_url = os.getenv('VLM_BASE_URL')
        
        if not api_key or not base_url:
            pytest.skip("VLM credentials not available")
        
        try:
            import adbutils
            adb = adbutils.AdbClient(host="127.0.0.1", port=5037)
            devices = adb.device_list()
            if not devices:
                pytest.skip("No ADB device connected")
            serial_no = devices[0].serial
        except Exception:
            pytest.skip("Cannot connect to ADB")
        
        from mobile_use.schema.config import VLMConfig, MobileEnvConfig
        
        vlm_config = VLMConfig(
            model_name='qwen2.5-vl-72b-instruct',
            api_key=api_key,
            base_url=base_url,
            max_tokens=512
        )
        env_config = MobileEnvConfig(
            serial_no=serial_no,
            go_home=False,
            wait_after_action_seconds=1.0
        )
        
        return ReActAgent(
            vlm=vlm_config,
            env=env_config,
            max_steps=1
        )

    @pytest.mark.integration
    def test_react_agent_single_step(self, real_react_agent):
        """Test ReActAgent executing a single step."""
        real_react_agent.reset(goal="Describe what you see on the screen")
        
        step_data = real_react_agent.step()
        
        assert step_data is not None
        assert step_data.curr_env_state is not None
        assert step_data.curr_env_state.pixels is not None

    @pytest.fixture
    def real_qwen_agent(self):
        """Create a QwenAgent with real credentials."""
        import os
        
        api_key = os.getenv('VLM_API_KEY')
        base_url = os.getenv('VLM_BASE_URL')
        
        if not api_key or not base_url:
            pytest.skip("VLM credentials not available")
        
        try:
            import adbutils
            adb = adbutils.AdbClient(host="127.0.0.1", port=5037)
            devices = adb.device_list()
            if not devices:
                pytest.skip("No ADB device connected")
            serial_no = devices[0].serial
        except Exception:
            pytest.skip("Cannot connect to ADB")
        
        from mobile_use.schema.config import VLMConfig, MobileEnvConfig
        
        vlm_config = VLMConfig(
            model_name='qwen2.5-vl-72b-instruct',
            api_key=api_key,
            base_url=base_url,
            max_tokens=512
        )
        env_config = MobileEnvConfig(
            serial_no=serial_no,
            go_home=False,
            wait_after_action_seconds=1.0
        )
        
        return QwenAgent(
            vlm=vlm_config,
            env=env_config,
            max_steps=1,
            enable_think=True
        )

    @pytest.mark.integration
    def test_qwen_agent_single_step(self, real_qwen_agent):
        """Test QwenAgent executing a single step."""
        real_qwen_agent.reset(goal="Open the Photos app")
        
        step_data = real_qwen_agent.step()
        
        assert step_data is not None
        assert step_data.curr_env_state is not None

