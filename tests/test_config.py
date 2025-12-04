"""
Tests for mobile_use/schema/config.py

Tests the configuration classes including BaseConfig, VLMConfig,
MobileEnvConfig, AgentConfig and various sub-agent configs.
"""

import pytest
import yaml
import tempfile
import os

from mobile_use.schema.config import (
    BaseConfig,
    MobileEnvConfig,
    VLMConfig,
    SubAgentConfig,
    PlannerConfig,
    KnowledgeConfig,
    OperatorConfig,
    AnswerAgentConfig,
    ReflectorConfig,
    TrajectoryReflectorConfig,
    GlobalReflectorConfig,
    ProgressorConfig,
    NoteTakerConfig,
    AgentConfig,
    ReActAgentConfig,
    QwenAgentConfig,
    MultiAgentConfig,
    HierarchicalAgentConfig,
)


class TestBaseConfig:
    """Tests for BaseConfig class."""

    def test_from_yaml_basic(self, tmp_path):
        """Test loading config from YAML file."""
        # Create a simple config class for testing
        class SimpleConfig(BaseConfig):
            name: str = "default"
            value: int = 0

        yaml_content = """
name: test
value: 42
"""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml_content)

        config = SimpleConfig.from_yaml(str(config_path))
        assert config.name == "test"
        assert config.value == 42


class TestMobileEnvConfig:
    """Tests for MobileEnvConfig class."""

    def test_default_values(self):
        """Test MobileEnvConfig default values."""
        config = MobileEnvConfig()
        assert config.serial_no is None
        assert config.host == "127.0.0.1"
        assert config.port == 5037
        assert config.go_home is False
        assert config.wait_after_action_seconds == 2.0

    def test_custom_values(self):
        """Test MobileEnvConfig with custom values."""
        config = MobileEnvConfig(
            serial_no="emulator-5554",
            host="192.168.1.100",
            port=5038,
            go_home=True,
            wait_after_action_seconds=1.5
        )
        assert config.serial_no == "emulator-5554"
        assert config.host == "192.168.1.100"
        assert config.port == 5038
        assert config.go_home is True
        assert config.wait_after_action_seconds == 1.5

    def test_from_yaml(self, tmp_path):
        """Test loading MobileEnvConfig from YAML."""
        yaml_content = """
serial_no: device123
host: localhost
port: 5039
go_home: true
wait_after_action_seconds: 3.0
"""
        config_path = tmp_path / "env_config.yaml"
        config_path.write_text(yaml_content)

        config = MobileEnvConfig.from_yaml(str(config_path))
        assert config.serial_no == "device123"
        assert config.go_home is True


class TestVLMConfig:
    """Tests for VLMConfig class."""

    def test_required_fields(self):
        """Test VLMConfig with required fields."""
        config = VLMConfig(
            model_name="qwen2.5-vl-72b-instruct",
            api_key="test-key",
            base_url="https://api.example.com/v1"
        )
        assert config.model_name == "qwen2.5-vl-72b-instruct"
        assert config.api_key == "test-key"
        assert config.base_url == "https://api.example.com/v1"

    def test_default_values(self):
        """Test VLMConfig default values."""
        config = VLMConfig(
            model_name="test-model",
            api_key="key",
            base_url="url"
        )
        assert config.max_retry == 3
        assert config.retry_waiting_seconds == 2
        assert config.max_tokens == 1024
        assert config.temperature == 0.0

    def test_extra_fields_allowed(self):
        """Test that VLMConfig allows extra fields."""
        config = VLMConfig(
            model_name="test",
            api_key="key",
            base_url="url",
            custom_param="custom_value"
        )
        assert config.custom_param == "custom_value"

    def test_from_yaml(self, tmp_path):
        """Test loading VLMConfig from YAML."""
        yaml_content = """
model_name: gpt-4-vision
api_key: secret-key
base_url: https://api.openai.com/v1
max_tokens: 2048
temperature: 0.5
"""
        config_path = tmp_path / "vlm_config.yaml"
        config_path.write_text(yaml_content)

        config = VLMConfig.from_yaml(str(config_path))
        assert config.model_name == "gpt-4-vision"
        assert config.max_tokens == 2048
        assert config.temperature == 0.5


class TestSubAgentConfig:
    """Tests for SubAgentConfig and derived classes."""

    def test_sub_agent_config_defaults(self):
        """Test SubAgentConfig default values."""
        config = SubAgentConfig()
        assert config.enabled is False
        assert config.vlm is None
        assert config.prompt_config is None

    def test_planner_config(self):
        """Test PlannerConfig inherits from SubAgentConfig."""
        config = PlannerConfig(enabled=True)
        assert config.enabled is True

    def test_reflector_config(self):
        """Test ReflectorConfig."""
        config = ReflectorConfig(enabled=True)
        assert config.enabled is True


class TestKnowledgeConfig:
    """Tests for KnowledgeConfig class."""

    def test_default_values(self):
        """Test KnowledgeConfig default values."""
        config = KnowledgeConfig()
        assert config.embedding_model_path is None
        assert config.knowledge_database_dir is None
        assert config.explored_knowledge_path is None

    def test_custom_values(self):
        """Test KnowledgeConfig with custom values."""
        config = KnowledgeConfig(
            embedding_model_path="/models/embedding",
            knowledge_database_dir="/data/knowledge",
            explored_knowledge_path="/data/explored.json"
        )
        assert config.embedding_model_path == "/models/embedding"


class TestOperatorConfig:
    """Tests for OperatorConfig class."""

    def test_default_values(self):
        """Test OperatorConfig default values."""
        config = OperatorConfig()
        assert config.name == "Operator"
        assert config.num_histories is None
        assert config.include_device_time is True
        assert config.include_tips is True
        assert config.include_a11y_tree is False
        assert config.max_pixels is None
        assert config.knowledge is None

    def test_with_knowledge(self):
        """Test OperatorConfig with knowledge config."""
        knowledge = KnowledgeConfig(embedding_model_path="/path")
        config = OperatorConfig(enabled=True, knowledge=knowledge)
        assert config.knowledge is not None
        assert config.knowledge.embedding_model_path == "/path"


class TestAnswerAgentConfig:
    """Tests for AnswerAgentConfig class."""

    def test_default_values(self):
        """Test AnswerAgentConfig default values."""
        config = AnswerAgentConfig()
        assert config.name == "AnswerAgent"
        assert config.include_tips is False


class TestTrajectoryReflectorConfig:
    """Tests for TrajectoryReflectorConfig class."""

    def test_default_values(self):
        """Test TrajectoryReflectorConfig default values."""
        config = TrajectoryReflectorConfig()
        assert config.evoke_every_steps == 5
        assert config.cold_steps == 3
        assert config.detect_error is True
        assert config.num_histories == 'auto'
        assert config.num_latest_screenshots == 0
        assert config.max_repeat_action == 3
        assert config.max_repeat_action_series == 2
        assert config.max_repeat_screen == 3
        assert config.max_fail_count == 3

    def test_custom_values(self):
        """Test TrajectoryReflectorConfig with custom values."""
        config = TrajectoryReflectorConfig(
            evoke_every_steps=3,
            num_histories=5,
            max_repeat_action=5
        )
        assert config.evoke_every_steps == 3
        assert config.num_histories == 5


class TestGlobalReflectorConfig:
    """Tests for GlobalReflectorConfig class."""

    def test_default_values(self):
        """Test GlobalReflectorConfig default values."""
        config = GlobalReflectorConfig()
        assert config.num_latest_screenshots == 3


class TestAgentConfig:
    """Tests for AgentConfig class."""

    def test_default_values(self):
        """Test AgentConfig default values."""
        config = AgentConfig()
        assert config.vlm is None
        assert config.env is None
        assert config.max_steps == 10
        assert config.enable_log is False
        assert config.log_dir is None

    def test_with_sub_configs(self, sample_vlm_config, sample_env_config):
        """Test AgentConfig with VLM and env configs."""
        vlm = VLMConfig(**sample_vlm_config)
        env = MobileEnvConfig(**sample_env_config)
        
        config = AgentConfig(
            vlm=vlm,
            env=env,
            max_steps=20,
            enable_log=True,
            log_dir="/logs"
        )
        assert config.vlm.model_name == sample_vlm_config['model_name']
        assert config.env.serial_no == sample_env_config['serial_no']
        assert config.max_steps == 20

    def test_from_yaml(self, temp_yaml_config):
        """Test loading AgentConfig from YAML."""
        config = AgentConfig.from_yaml(temp_yaml_config)
        assert config.max_steps == 10
        assert config.vlm is not None
        assert config.env is not None


class TestReActAgentConfig:
    """Tests for ReActAgentConfig class."""

    def test_default_values(self):
        """Test ReActAgentConfig default values."""
        config = ReActAgentConfig()
        assert config.num_latest_screenshots == 3
        assert config.max_action_retry == 3
        assert config.prompt_config is None

    def test_custom_values(self):
        """Test ReActAgentConfig with custom values."""
        config = ReActAgentConfig(
            num_latest_screenshots=5,
            max_action_retry=5,
            max_steps=15
        )
        assert config.num_latest_screenshots == 5
        assert config.max_steps == 15


class TestQwenAgentConfig:
    """Tests for QwenAgentConfig class."""

    def test_default_values(self):
        """Test QwenAgentConfig default values."""
        config = QwenAgentConfig()
        assert config.max_action_retry == 3
        assert config.enable_think is True
        assert config.prompt_config is None
        assert config.min_pixels == 3136
        assert config.max_pixels == 10035200
        assert config.message_type == 'single'
        assert config.num_image_limit == 2
        assert config.coordinate_type == 'absolute'

    def test_message_type_literal(self):
        """Test QwenAgentConfig message_type values."""
        config_single = QwenAgentConfig(message_type='single')
        assert config_single.message_type == 'single'
        
        config_chat = QwenAgentConfig(message_type='chat')
        assert config_chat.message_type == 'chat'

    def test_coordinate_type_literal(self):
        """Test QwenAgentConfig coordinate_type values."""
        config_abs = QwenAgentConfig(coordinate_type='absolute')
        assert config_abs.coordinate_type == 'absolute'
        
        config_rel = QwenAgentConfig(coordinate_type='relative')
        assert config_rel.coordinate_type == 'relative'


class TestMultiAgentConfig:
    """Tests for MultiAgentConfig class."""

    def test_default_values(self):
        """Test MultiAgentConfig default values."""
        config = MultiAgentConfig()
        assert config.planner is None
        assert config.operator is None
        assert config.answer_agent is None
        assert config.reflector is None
        assert config.trajectory_reflector is None
        assert config.global_reflector is None
        assert config.progressor is None
        assert config.note_taker is None
        assert config.max_action_retry == 3
        assert config.reflect_on_demand is False
        assert config.logprob_threshold == -0.01
        assert config.enable_pre_reflection is True

    def test_with_sub_agents(self):
        """Test MultiAgentConfig with sub-agent configs."""
        config = MultiAgentConfig(
            operator=OperatorConfig(enabled=True, name="Operator"),
            reflector=ReflectorConfig(enabled=True),
            max_steps=15
        )
        assert config.operator.enabled is True
        assert config.reflector.enabled is True
        assert config.max_steps == 15


class TestHierarchicalAgentConfig:
    """Tests for HierarchicalAgentConfig class."""

    def test_default_values(self):
        """Test HierarchicalAgentConfig default values."""
        config = HierarchicalAgentConfig()
        assert config.task_classifier is None
        assert config.task_orchestrator is None
        assert config.task_extractor is None
        assert config.task_rewriter is None
        assert config.enable_hierarchical_planning is True

    def test_with_task_configs(self):
        """Test HierarchicalAgentConfig with task configs."""
        config = HierarchicalAgentConfig(
            task_classifier=SubAgentConfig(enabled=True),
            task_orchestrator=SubAgentConfig(enabled=True),
            enable_hierarchical_planning=False
        )
        assert config.task_classifier.enabled is True
        assert config.task_orchestrator.enabled is True
        assert config.enable_hierarchical_planning is False

    def test_inherits_from_multi_agent_config(self):
        """Test that HierarchicalAgentConfig inherits from MultiAgentConfig."""
        config = HierarchicalAgentConfig(
            operator=OperatorConfig(enabled=True),
            max_action_retry=5
        )
        assert config.operator.enabled is True
        assert config.max_action_retry == 5


class TestConfigYAMLRoundTrip:
    """Tests for config YAML serialization and deserialization."""

    def test_full_config_from_yaml(self, tmp_path):
        """Test loading a complete configuration from YAML."""
        yaml_content = """
vlm:
  model_name: qwen2.5-vl-72b-instruct
  api_key: test-api-key
  base_url: https://api.example.com/v1
  max_tokens: 2048
  temperature: 0.1

env:
  serial_no: emulator-5554
  host: 127.0.0.1
  port: 5037
  go_home: false
  wait_after_action_seconds: 2.0

max_steps: 20
enable_log: true
log_dir: /tmp/logs
"""
        config_path = tmp_path / "full_config.yaml"
        config_path.write_text(yaml_content)

        config = AgentConfig.from_yaml(str(config_path))
        
        assert config.vlm.model_name == "qwen2.5-vl-72b-instruct"
        assert config.vlm.max_tokens == 2048
        assert config.env.serial_no == "emulator-5554"
        assert config.max_steps == 20
        assert config.enable_log is True
        assert config.log_dir == "/tmp/logs"

    def test_qwen_agent_config_from_yaml(self, tmp_path):
        """Test loading QwenAgentConfig from YAML."""
        yaml_content = """
vlm:
  model_name: qwen2.5-vl
  api_key: key
  base_url: url

max_steps: 15
max_action_retry: 5
enable_think: false
message_type: chat
coordinate_type: relative
"""
        config_path = tmp_path / "qwen_config.yaml"
        config_path.write_text(yaml_content)

        config = QwenAgentConfig.from_yaml(str(config_path))
        
        assert config.max_action_retry == 5
        assert config.enable_think is False
        assert config.message_type == 'chat'
        assert config.coordinate_type == 'relative'

    def test_multi_agent_config_from_yaml(self, tmp_path):
        """Test loading MultiAgentConfig from YAML."""
        yaml_content = """
vlm:
  model_name: test-model
  api_key: key
  base_url: url

max_steps: 25

operator:
  enabled: true
  name: Operator
  include_device_time: true
  include_tips: true

reflector:
  enabled: true

trajectory_reflector:
  enabled: true
  evoke_every_steps: 3
  max_repeat_action: 4

reflect_on_demand: true
logprob_threshold: -0.05
"""
        config_path = tmp_path / "multi_agent_config.yaml"
        config_path.write_text(yaml_content)

        config = MultiAgentConfig.from_yaml(str(config_path))
        
        assert config.operator.enabled is True
        assert config.operator.name == "Operator"
        assert config.reflector.enabled is True
        assert config.trajectory_reflector.enabled is True
        assert config.trajectory_reflector.evoke_every_steps == 3
        assert config.reflect_on_demand is True
        assert config.logprob_threshold == -0.05

