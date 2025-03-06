from abc import ABC, abstractmethod
from pyregister import Registrable
from typing import Iterator, List

from mobile_use.scheme import StepData, AgentState, EpisodeData
from mobile_use.environ import Environment
from mobile_use.vlm import VLMWrapper


class Agent(ABC, Registrable):
    def __init__(self, env: Environment, vlm: VLMWrapper, max_steps: int=10):
        super().__init__()
        self.env = env
        self.vlm = vlm
        self.max_steps = max_steps
        self._init_data()

    def _init_data(self, goal: str=''):
        self.goal = goal
        self.status = None
        self.state = AgentState.READY
        self.messages = []
        self.curr_step_idx = 0
        self.trajectory: List[StepData] = []
        self.episode_data: EpisodeData = EpisodeData(num_steps=0, trajectory=self.trajectory)

    @abstractmethod
    def reset(self, *args, **kwargs) -> None:
        """Reset Agent to init state"""

    @abstractmethod
    def step(self) -> StepData:
        """Get the next step action based on the current environment state.

        Returns: StepData
        """

    @abstractmethod
    def iter_run(self, input_content: str) -> Iterator[StepData]:
        """Execute all step with maximum number of steps base on user input content.

        Returns: The content is an iterator for StepData
        """
