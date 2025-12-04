"""
Tests for mobile_use/schema/schema.py

Tests the data models including EnvState, Action, AgentState, AgentStatus,
and various step/episode data classes.
"""

import pytest
from PIL import Image
from dataclasses import fields

from mobile_use.schema.schema import (
    EnvState,
    Action,
    AgentState,
    AgentStatus,
    VLMCallingData,
    BaseStepData,
    SingleAgentStepData,
    MobileUseStepData,
    BaseEpisodeData,
    MobileUseEpisodeData,
    HierarchicalAgentTaskData,
)


class TestEnvState:
    """Tests for EnvState dataclass."""

    def test_create_env_state_with_pixels_only(self, mock_image):
        """Test creating EnvState with only required fields."""
        state = EnvState(pixels=mock_image)
        assert state.pixels == mock_image
        assert state.package is None
        assert state.a11y_tree is None
        assert state.device_time is None

    def test_create_env_state_with_all_fields(self, mock_image):
        """Test creating EnvState with all fields."""
        state = EnvState(
            pixels=mock_image,
            package="com.example.app",
            a11y_tree={"node": "root"},
            device_time="Thu Dec 4 10:00:00 GMT 2025"
        )
        assert state.pixels == mock_image
        assert state.package == "com.example.app"
        assert state.a11y_tree == {"node": "root"}
        assert state.device_time == "Thu Dec 4 10:00:00 GMT 2025"

    def test_env_state_is_frozen(self, mock_image):
        """Test that EnvState is immutable (frozen)."""
        state = EnvState(pixels=mock_image)
        with pytest.raises(AttributeError):
            state.package = "new.package"


class TestAction:
    """Tests for Action dataclass."""

    def test_create_action_with_name_only(self):
        """Test creating Action with only name."""
        action = Action(name='press_home')
        assert action.name == 'press_home'
        assert action.parameters is None

    def test_create_action_with_parameters(self):
        """Test creating Action with parameters."""
        action = Action(
            name='click',
            parameters={'coordinate': [540, 960]}
        )
        assert action.name == 'click'
        assert action.parameters == {'coordinate': [540, 960]}

    def test_action_repr(self):
        """Test Action string representation."""
        action = Action(name='click', parameters={'coordinate': [540, 960]})
        repr_str = repr(action)
        assert 'click' in repr_str
        assert 'coordinate' in repr_str

    def test_action_str(self):
        """Test Action __str__ method."""
        action = Action(name='type', parameters={'text': 'hello'})
        str_val = str(action)
        assert 'type' in str_val
        assert 'text=hello' in str_val

    def test_action_repr_with_none_params(self):
        """Test Action repr with None parameters."""
        action = Action(name='press_home')
        repr_str = repr(action)
        assert 'press_home()' == repr_str

    def test_action_repr_filters_none_values(self):
        """Test that repr filters out None parameter values."""
        action = Action(name='click', parameters={'coordinate': [100, 200], 'extra': None})
        repr_str = repr(action)
        assert 'coordinate' in repr_str
        assert 'None' not in repr_str


class TestAgentState:
    """Tests for AgentState enum."""

    def test_agent_state_values(self):
        """Test AgentState enum values."""
        assert AgentState.READY.value == 'READY'
        assert AgentState.RUNNING.value == 'RUNNING'
        assert AgentState.CALLUSER.value == 'CALLUSER'

    def test_agent_state_comparison(self):
        """Test AgentState comparison."""
        state = AgentState.READY
        assert state == AgentState.READY
        assert state != AgentState.RUNNING


class TestAgentStatus:
    """Tests for AgentStatus enum."""

    def test_agent_status_values(self):
        """Test AgentStatus enum values."""
        assert AgentStatus.FINISHED.value == 'FINISHED'
        assert AgentStatus.FAILED.value == 'FAILED'

    def test_agent_status_comparison(self):
        """Test AgentStatus comparison."""
        status = AgentStatus.FINISHED
        assert status == AgentStatus.FINISHED
        assert status != AgentStatus.FAILED


class TestVLMCallingData:
    """Tests for VLMCallingData dataclass."""

    def test_create_vlm_calling_data(self):
        """Test creating VLMCallingData."""
        messages = [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
        data = VLMCallingData(messages=messages, response="Hello there!")
        assert data.messages == messages
        assert data.response == "Hello there!"


class TestBaseStepData:
    """Tests for BaseStepData dataclass."""

    def test_create_base_step_data(self, mock_env_state):
        """Test creating BaseStepData with required fields."""
        step_data = BaseStepData(
            step_idx=0,
            curr_env_state=mock_env_state
        )
        assert step_data.step_idx == 0
        assert step_data.curr_env_state == mock_env_state
        assert step_data.content is None
        assert step_data.action is None
        assert step_data.exec_env_state is None
        assert step_data.vlm_call_history == []

    def test_create_base_step_data_with_action(self, mock_env_state, sample_action):
        """Test creating BaseStepData with action."""
        step_data = BaseStepData(
            step_idx=1,
            curr_env_state=mock_env_state,
            action=sample_action,
            content="VLM response"
        )
        assert step_data.action == sample_action
        assert step_data.content == "VLM response"


class TestSingleAgentStepData:
    """Tests for SingleAgentStepData dataclass."""

    def test_create_single_agent_step_data(self, mock_env_state):
        """Test creating SingleAgentStepData."""
        step_data = SingleAgentStepData(
            step_idx=0,
            curr_env_state=mock_env_state,
            thought="I need to click the button",
            action_s="click(coordinate=[540, 960])",
            action_desc="Clicking the Photos button",
            answer=None,
            summary="Opened Photos app"
        )
        assert step_data.thought == "I need to click the button"
        assert step_data.action_s == "click(coordinate=[540, 960])"
        assert step_data.summary == "Opened Photos app"

    def test_single_agent_step_data_inherits_from_base(self, mock_env_state):
        """Test that SingleAgentStepData inherits from BaseStepData."""
        step_data = SingleAgentStepData(
            step_idx=0,
            curr_env_state=mock_env_state
        )
        # Check inherited fields
        assert hasattr(step_data, 'step_idx')
        assert hasattr(step_data, 'curr_env_state')
        assert hasattr(step_data, 'vlm_call_history')


class TestMobileUseStepData:
    """Tests for MobileUseStepData dataclass."""

    def test_create_mobile_use_step_data(self, mock_env_state):
        """Test creating MobileUseStepData with all fields."""
        step_data = MobileUseStepData(
            step_idx=0,
            curr_env_state=mock_env_state,
            thought="Analyzing the screen",
            plan="1. Open app 2. Click button",
            sub_goal="Open the app",
            progress="Step 1 of 3",
            memory="User wants to check photos",
            reflection_outcome="C",
            reflection_error=None,
            step_duration=1.5,
            exec_duration=0.5
        )
        assert step_data.plan == "1. Open app 2. Click button"
        assert step_data.sub_goal == "Open the app"
        assert step_data.step_duration == 1.5

    def test_mobile_use_step_data_default_values(self, mock_env_state):
        """Test MobileUseStepData default values."""
        step_data = MobileUseStepData(
            step_idx=0,
            curr_env_state=mock_env_state
        )
        assert step_data.action_type_tokens is None
        assert step_data.action_type_logprobs is None
        assert step_data.knowledge is None


class TestBaseEpisodeData:
    """Tests for BaseEpisodeData dataclass."""

    def test_create_base_episode_data(self):
        """Test creating BaseEpisodeData."""
        episode = BaseEpisodeData(goal="Open Photos app")
        assert episode.goal == "Open Photos app"
        assert episode.num_steps is None
        assert episode.status is None
        assert episode.message is None
        assert episode.trajectory is None

    def test_create_base_episode_data_with_all_fields(self, mock_env_state, sample_action):
        """Test creating BaseEpisodeData with all fields."""
        step = BaseStepData(step_idx=0, curr_env_state=mock_env_state, action=sample_action)
        episode = BaseEpisodeData(
            goal="Complete the task",
            num_steps=5,
            status="FINISHED",
            message="Task completed",
            trajectory=[step]
        )
        assert episode.num_steps == 5
        assert len(episode.trajectory) == 1


class TestMobileUseEpisodeData:
    """Tests for MobileUseEpisodeData dataclass."""

    def test_create_mobile_use_episode_data(self):
        """Test creating MobileUseEpisodeData."""
        episode = MobileUseEpisodeData(goal="Test task")
        assert episode.goal == "Test task"
        assert episode.trajectory == []
        assert episode.finish_count == 0
        assert episode.memory == ""

    def test_mobile_use_episode_data_with_trajectory(self, mock_env_state):
        """Test MobileUseEpisodeData with trajectory."""
        step = MobileUseStepData(step_idx=0, curr_env_state=mock_env_state)
        episode = MobileUseEpisodeData(
            goal="Test task",
            trajectory=[step],
            finish_count=1,
            memory="Some notes"
        )
        assert len(episode.trajectory) == 1
        assert episode.finish_count == 1
        assert episode.memory == "Some notes"


class TestHierarchicalAgentTaskData:
    """Tests for HierarchicalAgentTaskData dataclass."""

    def test_create_hierarchical_task_data(self):
        """Test creating HierarchicalAgentTaskData."""
        episode = MobileUseEpisodeData(goal="Main task")
        task_data = HierarchicalAgentTaskData(
            task="Complete complex task",
            episode_data=episode
        )
        assert task_data.task == "Complete complex task"
        assert task_data.episode_data == episode
        assert task_data.sub_tasks is None

    def test_hierarchical_task_data_with_subtasks(self):
        """Test HierarchicalAgentTaskData with subtasks."""
        episode = MobileUseEpisodeData(goal="Main task")
        sub_episode = MobileUseEpisodeData(goal="Sub task 1")
        
        task_data = HierarchicalAgentTaskData(
            task="Complex task",
            episode_data=episode,
            task_type="navigation",
            sub_tasks=["Step 1", "Step 2", "Step 3"],
            sub_tasks_return=["Done", "Done"],
            sub_tasks_episode_data=[sub_episode],
            current_sub_task_idx=1
        )
        assert task_data.task_type == "navigation"
        assert len(task_data.sub_tasks) == 3
        assert task_data.current_sub_task_idx == 1


class TestDataclassFields:
    """Tests to verify dataclass field definitions."""

    def test_env_state_fields(self):
        """Verify EnvState has expected fields."""
        field_names = {f.name for f in fields(EnvState)}
        expected = {'pixels', 'package', 'a11y_tree', 'device_time'}
        assert field_names == expected

    def test_action_fields(self):
        """Verify Action has expected fields."""
        field_names = {f.name for f in fields(Action)}
        expected = {'name', 'parameters'}
        assert field_names == expected

    def test_mobile_use_step_data_fields(self):
        """Verify MobileUseStepData has all expected fields."""
        field_names = {f.name for f in fields(MobileUseStepData)}
        expected_fields = {
            'step_idx', 'curr_env_state', 'content', 'action',
            'exec_env_state', 'vlm_call_history', 'thought', 'action_s',
            'action_desc', 'answer', 'summary', 'action_type_tokens',
            'action_type_logprobs', 'plan', 'sub_goal', 'progress',
            'memory', 'reflection_outcome', 'reflection_error',
            'trajectory_reflection_outcome', 'trajectory_reflection_error',
            'evaluation_result', 'evaluation_reason', 'knowledge',
            'step_duration', 'exec_duration'
        }
        assert expected_fields.issubset(field_names)

