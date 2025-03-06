"""
Following the design of mobile agent in Qwen2.5-VL, 
we implement the agent in QwenAgent class.

Note: Call user is not supported in this version.
"""

import logging
import re
from typing import Iterator
import json

from mobile_use.scheme import *
from mobile_use.environ import Environment
from mobile_use.vlm import VLMWrapper
from mobile_use.utils import encode_image_url, smart_resize
from mobile_use.agents import Agent


logger = logging.getLogger(__name__)


IMAGE_PLACEHOLDER = '<|vision_start|><|image_pad|><|vision_end|>'

SYSTEM_PROMPT = """
You are a helpful assistant.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{{"type": "function", "function": {{"name_for_human": "mobile_use", "name": "mobile_use", "description": "Use a touchscreen to interact with a mobile device, and take screenshots.
* This is an interface to a mobile device with touchscreen. You can perform actions like clicking, typing, swiping, etc.
* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.
* The screen's resolution is {width}x{height}.
* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.", "parameters": {{"properties": {{"action": {{"description": "The action to perform. The available actions are:
* `key`: Perform a key event on the mobile device.
    - This supports adb's `keyevent` syntax.
    - Examples: "volume_up", "volume_down", "power", "camera", "clear".
* `click`: Click the point on the screen with coordinate (x, y).
* `long_press`: Press the point on the screen with coordinate (x, y) for specified seconds.
* `swipe`: Swipe from the starting point with coordinate (x, y) to the end point with coordinates2 (x2, y2).
* `type`: Input the specified text into the activated input box.
* `answer`: Output the answer.
* `system_button`: Press the system button.
* `open`: Open an app on the device.
* `wait`: Wait specified seconds for the change to happen.
* `terminate`: Terminate the current task and report its completion status.", "enum": ["key", "click", "long_press", "swipe", "type", "answer", "system_button", "open", "wait", "terminate"], "type": "string"}}, "coordinate": {{"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=click`, `action=long_press`, and `action=swipe`.", "type": "array"}}, "coordinate2": {{"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=swipe`.", "type": "array"}}, "text": {{"description": "Required only by `action=key`, `action=type`, `action=answer`, and `action=open`.", "type": "string"}}, "time": {{"description": "The seconds to wait. Required only by `action=long_press` and `action=wait`.", "type": "number"}}, "button": {{"description": "Back means returning to the previous interface, Home means returning to the desktop, Menu means opening the application background menu, and Enter means pressing the enter. Required only by `action=system_button`", "enum": ["Back", "Home", "Menu", "Enter"], "type": "string"}}, "status": {{"description": "The status of the task. Required only by `action=terminate`.", "type": "string", "enum": ["success", "failure"]}}}}, "required": ["action"], "type": "object"}}, "args_format": "Format the arguments as a JSON object."}}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>
""".strip()


# Not used, because the agent will not follow the output format. Put it here for reference.
THINK_AND_SUMMARY_PROMPT = "Before answering, explain your reasoning step-by-step in <thinking></thinking> tags, and insert them before the <tool_call></tool_call> XML tags.\nAfter answering, summarize your action in <conclusion></conclusion> tags, and insert them after the <tool_call></tool_call> XML tags."


ACTION_SPACE = ["key", "click", "left_click", "long_press", "swipe", "scroll", "type", "answer", "system_button", "open", "wait", "terminate"]

def _parse_response(content: str, size: tuple[float, float], raw_size: tuple[float, float]) -> Action:
    reason = re.search(r'<thinking>(.*?)</thinking>', content, flags=re.DOTALL)
    if reason:
        reason_s = reason.group(1).strip()
    else:
        reason_s = None
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
    if name not in ACTION_SPACE:
        raise Exception(f"Action {name} is not in the action space.")
    action['arguments'].pop('action')
    params = action['arguments']

    for k, v in params.items():
        if k in ['coordinate', 'coordinate2', 'point', 'start_point', 'end_point']:
            try:
                x = round(v[0] / size[0] * raw_size[0])
                y = round(v[1] / size[1] * raw_size[1])
                params[k] = (x, y)
            except:
                pass
    action_a = Action(name=name, parameters=params)
    return reason_s, action_a, action_s, summary_s


@Agent.register('Qwen')
class QwenAgent(Agent):
    def __init__(
            self, 
            env: Environment,
            vlm: VLMWrapper,
            max_steps: int=10,
            num_latest_screenshot: int=10,
            max_reflection_action: int=3,
            reflection_action_waiting_seconds: float=1.0,
            max_retry_vlm: int=3,
            retry_vlm_waiting_seconds: float=1.0,
        ):
        super().__init__(env=env, vlm=vlm, max_steps=max_steps)
        self.num_latest_screenshot = num_latest_screenshot
        self.max_reflection_action = max_reflection_action
        self.reflection_action_waiting_seconds = reflection_action_waiting_seconds
        self.max_retry_vlm = max_retry_vlm
        self.retry_vlm_waiting_seconds = retry_vlm_waiting_seconds

    def reset(self, goal: str='') -> None:
        """Reset the state of the agent.
        """
        self._init_data(goal=goal)

    def _get_curr_step_data(self) -> StepData:
        if len(self.trajectory) > self.curr_step_idx:
            return self.trajectory[self.curr_step_idx]
        else:
            return None

    def step(self) -> StepData:
        """Execute the task with maximum number of steps.

        Returns: StepData
        """
        logger.info("Step %d ... ..." % self.curr_step_idx)

        # Init messages
        if self.curr_step_idx == 0:
            env_state = self.env.get_state()
            pixels = env_state.pixels.copy()
            resized_height, resized_width = smart_resize(height=pixels.height, width=pixels.width)
            self.messages.append({
                'role': 'system', 
                'content': [
                    {"type": "text", "text": SYSTEM_PROMPT.format(width=resized_width, height=resized_height)},
                ]
            })
            self.messages.append({
                'role': 'user', 
                'content': [
                    {'type': 'text','text': f'The user query: {self.goal}\nTask progress (You have done the following operation on the current device): None'},
                    {}, # Place holder for observation
                    {}, # Place holder for image
                    # {'type': 'text','text': THINK_AND_SUMMARY_PROMPT},
                ]
            })
        user_message_length = len(self.messages[1]['content'])

        # Fixed Picture sequence inconsistency problem in vllm0.7.2 
        # and Compatible QwenAPI error: '<400> InternalError.Algo.InvalidParameter: Invalid text: <|image_pad|>'
        observation = '' if 'dashscope.aliyuncs.com' in str(self.vlm.client.base_url) else IMAGE_PLACEHOLDER
        self.messages[-1]['content'][1] = {
            'type': 'text',
            'text': f'Observation: {observation}'
        }

        # Get the current environment screen
        env_state = self.env.get_state()
        pixels = env_state.pixels.copy()
        resized_height, resized_width = smart_resize(height=pixels.height, width=pixels.width)
        img_msg = {
            "type": "image_url",
            "image_url": {"url": encode_image_url(pixels)}
        }
        self.messages[-1]['content'][2] = img_msg

        # Add new step data
        self.trajectory.append(StepData(
            step_idx=self.curr_step_idx,
            curr_env_state=env_state,
            vlm_call_history=[]
        ))

        step_data = self.trajectory[-1]
        response = self.vlm.predict(self.messages)
        counter = self.max_reflection_action
        reason, action = None, None
        while counter > 0:
            try:
                content = response.choices[0].message.content
                step_data.content = content
                logger.info("Content from VLM:\n%s" % step_data.content)
                step_data.vlm_call_history.append(VLMCallingData(self.messages, response))
                reason, action, action_s, summary = _parse_response(content, (resized_width, resized_height), env_state.pixels.size)
                logger.info("REASON: %s" % reason)
                logger.info("ACTION: %s" % str(action))
                logger.info("SUMMARY: %s" % summary)
                break
            except Exception as e:
                logger.warning(f"Failed to parse the action from: {content}.Error is {e.args}")
                msg = {
                    'type': 'text', 
                    'text': f"Failed to parse the action from: {content}.Error is {e.args}"
                }
                self.messages[-1]['content'].append(msg)
                response = self.vlm.predict(self.messages)
                counter -= 1

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
            else:
                logger.info(f"Execute the action: {action}")

                try:
                    self.env.execute_action(action)
                except Exception as e:
                    logger.warning(f"Failed to execute the action: {action}. Error: {e}")
                    action = None
                step_data.exec_env_state = self.env.get_state()

        self.messages[-1]['content'] = self.messages[-1]['content'][:user_message_length]
        if self.curr_step_idx == 0:
            self.messages[-1]['content'][0]['text'] = f'The user query: {self.goal}\nTask progress (You have done the following operation on the current device): '
        
        if action is None:
            self.messages[-1]['content'][0]['text'] += f'\nStep {self.curr_step_idx + 1}: None'
        else:
            # self.messages[-1]['content'][0]['text'] += f'\nStep {self.curr_step_idx + 1}: <thinking>\n{reason}\n</thinking>\n<tool_call>\n{action_s}\n</tool_call>\n<conclusion>\n{summary}\n</conclusion>'
            self.messages[-1]['content'][0]['text'] += f'\nStep {self.curr_step_idx + 1}: <tool_call>\n{action_s}\n</tool_call> '

        step_data.action = action
        step_data.thought = reason
        return step_data


    def iter_run(self, input_content: str) -> Iterator[StepData]:
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

    def run(self, input_content: str) -> EpisodeData:
        """Execute the agent with user input content.

        Returns: EpisodeData
        """
        for _ in self.iter_run(input_content, stream=False):
            pass
        return self.episode_data
