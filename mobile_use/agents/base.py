from abc import ABC, abstractmethod
from typing import Iterator, List
import logging
import os
from datetime import datetime
from uuid import uuid4

from pyregister import Registrable
from mobile_use.schema.schema import BaseStepData, AgentState, BaseEpisodeData
from mobile_use.environment.mobile_environ import Environment
from mobile_use.utils.vlm import VLMWrapper
from mobile_use.schema.config import AgentConfig

logger = logging.getLogger(__name__)


class Agent(ABC, Registrable):
    def __init__(self, config_path: str = None, **kwargs):
        super().__init__()
        if config_path is not None:
            self.config = AgentConfig.from_yaml(config_path)
        else:
            self.config = AgentConfig(**kwargs)
        
        self.env = Environment(**self.config.env.model_dump()) if self.config.env else None
        self.vlm = VLMWrapper(**self.config.vlm.model_dump()) if self.config.vlm else None
        self.max_steps = self.config.max_steps

        self.enable_log = self.config.enable_log
        self.log_dir = self.config.log_dir
        self._episode_log_dir = None

        self._init_data()

    def _init_data(self, goal: str=''):
        self.goal = goal
        self.status = None
        self.state = AgentState.READY
        self.messages = []
        self.curr_step_idx = 0
        self.trajectory: List[BaseStepData] = []
        self.episode_data: BaseEpisodeData = BaseEpisodeData(goal=goal, num_steps=0, trajectory=self.trajectory)

        if self.enable_log:
            # reset episode log directory
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:6]
            self._episode_log_dir = os.path.join(self.log_dir, run_id)

    def set_max_steps(self, max_steps: int) -> None:
        self.max_steps = max_steps

    @abstractmethod
    def reset(self, *args, **kwargs) -> None:
        """Reset Agent to init state"""

    @abstractmethod
    def step(self) -> BaseStepData:
        """Get the next step action based on the current environment state.

        Returns: BaseStepData
        """

    @abstractmethod
    def iter_run(self, input_content: str) -> Iterator[BaseStepData]:
        """Execute all step with maximum number of steps base on user input content.

        Returns: The content is an iterator for BaseStepData
        """
