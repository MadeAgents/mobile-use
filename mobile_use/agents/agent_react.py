import logging
import re
from typing import Iterator
import json
import copy

from mobile_use.schema.schema import *
from mobile_use.environment.mobile_environ import Environment
from mobile_use.utils.vlm import VLMWrapper
from mobile_use.utils.utils import encode_image_url, smart_resize, generate_message, contains_chinese
from mobile_use.agents import Agent
from mobile_use.agents.agent_qwen import slim_messages
from mobile_use.schema.config import ReActAgentConfig
from mobile_use.utils.constants import IMAGE_PLACEHOLDER
from mobile_use.default_prompts.prompt_type import load_prompt, ReActAgentPrompt

ACTION_SPACE = ['click', 'long_press', 'type', 'scroll', 'press_home', 'press_back', 'finished', 'call_user', 'wait']

logger = logging.getLogger(__name__)


def parse_reason_and_action(content: str, size: tuple[float, float], raw_size: tuple[float, float]) -> Action:
    reason = re.search(r'Thought:(.*)Action:', content, flags=re.DOTALL)
    if reason:
        reason_s = reason.group(1).strip()
    else:
        reason_s = None
    
    action_name = '|'.join(ACTION_SPACE)
    search_res = re.search(fr'Action: *({action_name})\((.*)\)', content, flags=re.DOTALL)

    if not search_res:
        raise Exception("Action is undefined")

    name = search_res.group(1).strip()
    params = eval(f"dict({search_res.group(2)})")

    for k, v in params.items():
        if k in ['click', 'long_press', 'scroll']:
            try:
                x = round(v[0] / size[0] * raw_size[0])
                y = round(v[1] / size[1] * raw_size[1])
                params[k] = (x, y)
            except:
                pass
    action_a = Action(name=name, parameters=params)
    action_s = f'{name}({search_res.group(2)})'     # raw action
    return reason_s, action_a, action_s


@Agent.register('SingleAgent')
@Agent.register('ReAct')
class ReActAgent(Agent):
    def __init__(self, config_path: str = None, **kwargs):
        super().__init__(config_path, **kwargs)
        if config_path is not None:
            self.config = ReActAgentConfig.from_yaml(config_path)
        else:
            self.config = ReActAgentConfig(**kwargs)
        
        self.num_latest_screenshots = self.config.num_latest_screenshots
        self.max_action_retry = self.config.max_action_retry
        self.prompt: ReActAgentPrompt = load_prompt('react_agent', self.config.prompt_config)

    def _init_data(self, goal: str=''):
        super()._init_data(goal)
        self.trajectory: List[SingleAgentStepData] = []
        self.episode_data: BaseEpisodeData = BaseEpisodeData(goal=goal, num_steps=0, trajectory=self.trajectory)
        self.messages: List[Dict[str,Any]] = []

    def reset(self, goal: str='', max_steps: int = None) -> None:
        """Reset the state of the agent.
        """
        self._init_data(goal=goal)
        if isinstance(max_steps, int):
            self.set_max_steps(max_steps)

    def _get_curr_step_data(self) -> SingleAgentStepData:
        if len(self.trajectory) > self.curr_step_idx:
            return self.trajectory[self.curr_step_idx]
        else:
            return None

    def step(self) -> SingleAgentStepData:
        """Execute the task with maximum number of steps.

        Returns: StepData
        """
        logger.info("Step %d ... ..." % self.curr_step_idx)

        # Get the current environment screen
        env_state = self.env.get_state()
        pixels = env_state.pixels
        raw_size = pixels.size
        resized_height, resized_width = smart_resize(
            height=pixels.height,
            width=pixels.width)
        pixels = pixels.resize((resized_width, resized_height))

        # Add new step data
        self.trajectory.append(MobileUseStepData(
            step_idx=self.curr_step_idx,
            curr_env_state=env_state,
        ))
        step_data = self.trajectory[-1]

        if self.curr_step_idx == 0:
            # Add system prompt
            if contains_chinese(self.goal):
                system_prompt = self.prompt.system_prompt_zh
            else:
                system_prompt = self.prompt.system_prompt_en
            logger.info(f"System Prompt:\n{system_prompt}")
            system_message = generate_message("system", system_prompt)
            self.messages.append(system_message)

            # Add user prompt
            user_prompt = self.prompt.task_prompt.format(goal=self.episode_data.goal)
            user_message = generate_message("user", user_prompt)
            self.messages.append(user_message)

        if self.state == AgentState.CALLUSER:
            user_message = generate_message("user", self._user_input)
            self.messages.append(user_message)
        else:
            user_message = generate_message("user", IMAGE_PLACEHOLDER, images=[pixels])
            self.messages.append(user_message)

        self.messages = slim_messages(self.messages, num_image_limit=self.num_latest_screenshots)

        # Call VLM
        response = self.vlm.predict(self.messages)

        # parse the response
        thought_s, action, action_s = None, None, None
        counter = self.max_action_retry
        while counter > 0:
            try:
                raw_action = response.choices[0].message.content
                logger.info("Action from VLM:\n%s" % raw_action)
                step_data.content = raw_action
                thought_s, action, action_s = parse_reason_and_action(raw_action, (resized_width, resized_height), raw_size)
                logger.info(f"Thought: {thought_s}")
                logger.info(f"Action: {action}")
                logger.info(f"Action string: {action_s}")
                break
            except Exception as e:
                logger.warning(f"Failed to parse the action. Error is {e.args}")
                error_prompt = f"Failed to parse the action. Error is {e.args}\nPlease follow the output format to provide a valid action:"
                msg = {"role": "user", "content": [{"type": "text", "text": error_prompt}]}
                self.messages.append(msg)
                response = self.vlm.predict(self.messages)
                counter -= 1
        if counter != self.max_action_retry:
            self.messages = self.messages[:-(self.max_action_retry - counter)]

        if action is None:
            logger.warning("Action parse error after max retry.")
        else:
            if action.name.upper() == 'FINISHED':
                logger.info(f"Finished: {action}")
                self.status = AgentStatus.FINISHED
                step_data.answer = action.parameters.get('answer')
            elif action.name.upper() == 'CALL_USER':
                logger.info(f"Call for help from user:{action}")
                self.state = AgentState.CALLUSER
            else:
                logger.info(f"Execute the action: {action}")
                self.env.execute_action(action)
                step_data.exec_env_state = self.env.get_state()

        if action is not None:
            step_data.action = action
            step_data.thought = thought_s
            step_data.action_s = action_s

        return step_data


    def iter_run(self, input_content: str) -> Iterator[SingleAgentStepData]:
        """Execute the agent with user input content.

        Returns: Iterator[SingleAgentStepData]
        """

        if self.state == AgentState.READY:
            self.reset(goal=input_content)
            logger.info("Start task: %s, with at most %d steps" % (self.goal, self.max_steps))
        elif self.state == AgentState.CALLUSER:
            self._user_input = input_content      # user answer
            self.state = AgentState.RUNNING       # reset agent state
            logger.info("Continue task: %s, with user input %s" % (self.goal, input_content))
        else:
            raise Exception('Error agent state')

        for step_idx in range(self.curr_step_idx, self.max_steps):
            self.curr_step_idx = step_idx
            # show init environment
            yield SingleAgentStepData(
                step_idx=self.curr_step_idx,
                curr_env_state=self.env.get_state(),
                vlm_call_history=[]
            )
            try:
                self.step()
            except Exception as e:
                self.status = AgentStatus.FAILED
                self.episode_data.status = self.status
                self.episode_data.message = str(e)
                yield self._get_curr_step_data()
                return

            self.episode_data.num_steps = step_idx + 1
            self.episode_data.status = self.status

            if self.status == AgentStatus.FINISHED:
                logger.info("Agent indicates task is done.")
                self.episode_data.message = 'Agent indicates task is done'
                yield self._get_curr_step_data()
                return
            elif self.state == AgentState.CALLUSER:
                logger.info("Agent indicates to ask user for help.")
                yield self._get_curr_step_data()
                return
            else:
                logger.info("Agent indicates one step is done.")
            yield self._get_curr_step_data()
        logger.warning(f"Agent reached max number of steps: {self.max_steps}.")

    def run(self, input_content: str) -> BaseEpisodeData:
        """Execute the agent with user input content.

        Returns: EpisodeData
        """
        for _ in self.iter_run(input_content, stream=False):
            pass
        return self.episode_data
