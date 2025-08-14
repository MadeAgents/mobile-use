from abc import ABC, abstractmethod
from typing import Iterator, List

from pyregister import Registrable
from mobile_use.schema.schema import BaseStepData, AgentState, BaseEpisodeData
from mobile_use.environment.mobile_environ import Environment
from mobile_use.utils.vlm import VLMWrapper
from mobile_use.schema.config import AgentConfig


class Agent(ABC, Registrable):
    def __init__(self, config_path: str):
        super().__init__()
        config = AgentConfig.from_yaml(config_path)
        self.env = Environment(**config.env.model_dump())
        self.vlm = VLMWrapper(**config.vlm.model_dump())
        self._init_data()

    def _init_data(self, goal: str='', max_steps: int=10):
        self.goal = goal
        self.max_steps = max_steps
        self.status = None
        self.state = AgentState.READY
        self.messages = []
        self.curr_step_idx = 0
        self.trajectory: List[BaseStepData] = []
        self.episode_data: BaseEpisodeData = BaseEpisodeData(goal=goal, num_steps=0, trajectory=self.trajectory)

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
