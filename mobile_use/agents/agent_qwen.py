import logging
import re
from typing import Iterator
import json
import copy

from mobile_use.schema.schema import *
from mobile_use.environment.mobile_environ import Environment
from mobile_use.utils.vlm import VLMWrapper
from mobile_use.utils.utils import encode_image_url, smart_resize, show_message, generate_message
from mobile_use.agents import Agent
from mobile_use.schema.config import QwenAgentConfig
from mobile_use.utils.constants import IMAGE_PLACEHOLDER
from mobile_use.default_prompts.prompt_type import load_prompt, QwenAgentPrompt


logger = logging.getLogger(__name__)


def _parse_response(content: str, size: tuple[float, float], raw_size: tuple[float, float]) -> Action:
    def map_names(name: str) -> str:
        maps = {
            "left_click": "click",
            "point": "coordinate",
            "start_point": "coordinate",
            "start_box": "coordinate",
            "end_point": "coordinate2",
            "end_box": "coordinate2",
            "scroll": "swipe",
            "content": "text",
            "open_app": "open",
        }
        return maps.get(name, name)
    thought = re.search(r'<thinking>(.*?)</thinking>', content, flags=re.DOTALL)
    if thought:
        thought_s = thought.group(1).strip()
    else:
        thought_s = None
    summary = re.search(r'<conclusion>(.*?)</conclusion>', content, flags=re.DOTALL)
    if summary:
        summary_s = summary.group(1).strip()
    else:
        summary_s = None
    action = re.search(r'<tool_call>(.*?)</tool_call>', content, flags=re.DOTALL)
    if not action:
        raise Exception("Cannot extract action in the content.")
    action_s = action.group(1).strip()
    action = json.loads(action_s)
    name = action['arguments']['action']

    # Remove the 'action' key and map the other keys in the arguments
    action['arguments'].pop('action')
    params = {}

    for k, v in action['arguments'].items():
        mapped_key = map_names(k)  # Map the key name
        if mapped_key in ['coordinate', 'coordinate2']:
            try:
                x = round(v[0] / size[0] * raw_size[0])
                y = round(v[1] / size[1] * raw_size[1])
                params[mapped_key] = (x, y)
            except:
                pass
        else:
            params[mapped_key] = v

    action_a = Action(name=name, parameters=params)
    return thought_s, action_a, action_s, summary_s

def slim_messages(messages, num_image_limit = 5):
    keep_image_index = []
    image_ptr = 0
    messages = copy.deepcopy(messages)
    for msg in messages:
        for content in msg['content']:
            if 'image' in content['type'] or 'image_url' in content['type']:
                keep_image_index.append(image_ptr)
                image_ptr += 1
    keep_image_index = keep_image_index[-num_image_limit:]

    image_ptr = 0
    for msg in messages:
        new_content = []
        for content in msg['content']:
            if 'image' in content['type'] or 'image_url' in content['type']:
                if image_ptr not in keep_image_index:
                    pass
                else:
                    new_content.append(content)
                image_ptr += 1
            else:
                new_content.append(content)
        msg['content'] = new_content
    return messages


@Agent.register('Qwen')
class QwenAgent(Agent):
    def __init__(self, config_path: str):
        super().__init__(config_path)
        config = QwenAgentConfig.from_yaml(config_path)
        self.config = config

        self.max_action_retry = config.max_action_retry
        self.enable_think = config.enable_think
        self.min_pixels = config.min_pixels
        self.max_pixels = config.max_pixels
        self.message_type = config.message_type
        self.num_image_limit = config.num_image_limit
        self.prompt: QwenAgentPrompt = load_prompt("qwen_agent", config.prompt_config)

    def _init_data(self, goal: str='', max_steps: int=10):
        super()._init_data(goal, max_steps)
        self.trajectory: List[SingleAgentStepData] = []
        self.episode_data: BaseEpisodeData = BaseEpisodeData(goal=goal, num_steps=0, trajectory=self.trajectory)
        self.messages: List[Dict[str,Any]] = []

    def reset(self, goal: str='', max_steps: int = 10) -> None:
        """Reset the state of the agent.
        """
        self._init_data(goal=goal, max_steps=max_steps)

    def _get_curr_step_data(self) -> SingleAgentStepData:
        if len(self.trajectory) > self.curr_step_idx:
            return self.trajectory[self.curr_step_idx]
        else:
            return None

    def step(self) -> SingleAgentStepData:
        """Execute the task with maximum number of steps.

        Returns: Answer
        """
        logger.info("Step %d ... ..." % self.curr_step_idx)
        show_step = [0,4]

        # Get the current environment screen
        env_state = self.env.get_state()
        pixels = env_state.pixels
        raw_size = pixels.size
        resized_height, resized_width = smart_resize(
            height=pixels.height,
            width=pixels.width,
            factor=28,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,)
        pixels = pixels.resize((resized_width, resized_height))

        # Add new step data
        self.trajectory.append(MobileUseStepData(
            step_idx=self.curr_step_idx,
            curr_env_state=env_state,
        ))
        step_data = self.trajectory[-1]

        if self.curr_step_idx == 0:
            # Add system prompt
            system_prompt = self.prompt.system_prompt.format(
                width = resized_width,
                height = resized_height,
            )
            system_message = generate_message("system", system_prompt)
            self.messages.append(system_message)

            # Add user prompt
            user_prompt = self.prompt.task_prompt.format(goal=self.episode_data.goal)
            if self.enable_think:
                user_prompt += self.prompt.thinking_prompt
            user_prompt += f"\n{IMAGE_PLACEHOLDER}"
            user_message = generate_message("user", user_prompt, images=[pixels])
            self.messages.append(user_message)

        if self.message_type == 'single':
            user_prompt = self.prompt.task_prompt.format(goal=self.episode_data.goal)
            history = [str(step.summary) for step in self.trajectory[:-1]]
            history = ''.join([f'Step {si+1}: {_}; 'for si, _ in enumerate(history)])
            user_prompt += self.prompt.history_prompt.format(history=history)
            if self.enable_think:
                user_prompt += self.prompt.thinking_prompt
            user_prompt += f"\n{IMAGE_PLACEHOLDER}"
            user_message = generate_message("user", user_prompt, images=[pixels])
            self.messages[1] = user_message

        if self.message_type == 'chat' and self.curr_step_idx > 0:
            last_step = self.trajectory[self.curr_step_idx - 1]
            assistant_message = generate_message("assistant", last_step.content)
            self.messages.append(assistant_message)

            user_prompt = ""
            if self.enable_think:
                user_prompt += self.prompt.thinking_prompt
            user_prompt += f"\n{IMAGE_PLACEHOLDER}"
            user_message = generate_message("user", user_prompt, images=[pixels])
            self.messages.append(user_message)
            
            self.messages = slim_messages(self.messages, num_image_limit=self.num_image_limit)

        # Call VLM
        if self.curr_step_idx in show_step:
            show_message(self.messages, "Qwen")
        response = self.vlm.predict(self.messages)

        # parse the response
        thought_s, action, action_s, summary_s = None, None, None, None
        counter = self.max_action_retry
        while counter > 0:
            try:
                raw_action = response.choices[0].message.content
                logger.info("Action from VLM:\n%s" % raw_action)
                step_data.content = raw_action
                thought_s, action, action_s, summary_s = _parse_response(raw_action, (resized_width, resized_height), raw_size)
                logger.info(f"Thought: {thought_s}")
                logger.info(f"Action: {action}")
                logger.info(f"Action string: {action_s}")
                logger.info(f"Summary: {summary_s}")
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
            if action.name == 'terminate':
                if action.parameters['status'] == 'success':
                    logger.info(f"Finished: {action}")
                    self.status = AgentStatus.FINISHED
                elif action.parameters['status'] == 'failure':
                    logger.info(f"Failed: {action}")
                    self.status = AgentStatus.FAILED
            elif action.name == 'answer':
                logger.info(f"Answer: {action}")
                answer = action.parameters['text'].strip()
                step_data.answer = answer
                logger.info("Terminate the task after answering question.")
                self.status = AgentStatus.FINISHED
            else:
                logger.info(f"Execute the action: {action}")
                try:
                    self.env.execute_action(action)
                except Exception as e:
                    logger.warning(f"Failed to execute the action: {action}. Error: {e}")
                    action = None
                step_data.exec_env_state = self.env.get_state()

        if action is not None:
            step_data.action = action
            step_data.thought = thought_s
            step_data.action_s = action_s
            step_data.summary = summary_s

        return step_data


    def iter_run(self, input_content: str, stream: bool=False) -> Iterator[SingleAgentStepData]:
        """Execute the agent with user input content.

        Returns: Iterator[StepData]
        """

        if self.state == AgentState.READY:
            self.reset(goal=input_content)
            logger.info("Start task: %s, with at most %d steps" % (self.goal, self.max_steps))
        else:
            raise Exception('Error agent state')

        for step_idx in range(self.curr_step_idx, self.max_steps):
            self.curr_step_idx = step_idx
            try:
                self.step()
                yield self._get_curr_step_data()
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
            elif self.status == AgentStatus.FAILED:
                logger.info("Agent indicates task is failed.")
                self.episode_data.message = 'Agent indicates task is failed'
                yield self._get_curr_step_data()
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
