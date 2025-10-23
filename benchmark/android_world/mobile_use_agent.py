"""MobileUse agent for AndroidWorld."""
import logging
import traceback

import mobile_use
from android_world.agents import base_agent
from android_world.env import interface

logger = logging.getLogger(__name__)


class MobileUse(base_agent.EnvironmentInteractingAgent):
  def __init__(
          self, 
          env: interface.AsyncEnv,
          agent: mobile_use.Agent,
          name: str = "MobileUse",
      ):
    super().__init__(env, name)
    self.agent = agent
    self.agent.reset()

  def reset(self, go_home: bool = False) -> None:
    super().reset(go_home)
    self.env.hide_automation_ui()
    self.agent.reset()

  def step(self, goal: str) -> base_agent.AgentInteractionResult:
    if self.agent.goal != goal:
      self.agent.reset(goal=goal)

    answer = None
    step_data = None
    try:
      step_data = self.agent.step()
      answer = step_data.answer
    except Exception as e:
      logger.info("Some error happened during the MobileUse agent run.")
      traceback.print_exc()
      self.agent.status = mobile_use.AgentStatus.FAILED
      self.agent.episode_data.status = self.agent.status
      self.agent.episode_data.message = str(e)
      return base_agent.AgentInteractionResult(True, {"step_data": step_data})

    self.agent.episode_data.num_steps = self.agent.curr_step_idx + 1
    self.agent.episode_data.status = self.agent.status

    if answer is not None:
       logger.info("Agent interaction cache is updated: %s" % answer)
       self.env.interaction_cache = answer

    if self.agent.status == mobile_use.AgentStatus.FINISHED:
        logger.info("Agent indicates task is done.")
        self.agent.episode_data.message = 'Agent indicates task is done.'
        return base_agent.AgentInteractionResult(True, {"step_data": step_data})
    elif self.agent.state == mobile_use.AgentState.CALLUSER:
        logger.warning("CALLUSER is not supported in AdroidWorld evaluation.")
        return base_agent.AgentInteractionResult(True, {"step_data": step_data})
    else:
        self.agent.curr_step_idx += 1
        logger.info("Agent indicates one step is done.")
        return base_agent.AgentInteractionResult(False, {"step_data": step_data})
