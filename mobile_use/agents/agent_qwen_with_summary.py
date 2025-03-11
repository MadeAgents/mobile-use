"""
Based on the design of mobile agent in Qwen2.5-VL, 
we improve it by adding some guidance and summary part.
This agent achieves better performance in the online 
benchmark AndroidWorld.

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

PROMPT_PREFIX = """You are a helpful assistant.

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
* `system_button`: Press the system button.
* `open`: Open an app on the device.
* `wait`: Wait specified seconds for the change to happen.
* `terminate`: Terminate the current task and report its completion status.", "enum": ["key", "click", "long_press", "swipe", "type", "system_button", "open", "wait", "terminate"], "type": "string"}}, "coordinate": {{"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=click`, `action=long_press`, and `action=swipe`.", "type": "array"}}, "coordinate2": {{"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=swipe`.", "type": "array"}}, "text": {{"description": "Required only by `action=key`, `action=type`, and `action=open`.", "type": "string"}}, "time": {{"description": "The seconds to wait. Required only by `action=long_press` and `action=wait`.", "type": "number"}}, "button": {{"description": "Back means returning to the previous interface, Home means returning to the desktop, Menu means opening the application background menu, and Enter means pressing the enter. Required only by `action=system_button`", "enum": ["Back", "Home", "Menu", "Enter"], "type": "string"}}, "status": {{"description": "The status of the task. Required only by `action=terminate`.", "type": "string", "enum": ["success", "failure"]}}}}, "required": ["action"], "type": "object"}}, "args_format": "Format the arguments as a JSON object."}}}}
</tools>

Here are some useful guidelines you need to follow:
- If the task is finished, you should terminate the task in time! Do not repeat the action!
- Action click, long_press and swipe must contain coordinates within.
- You may be given some history plan and actions, this is the response from the previous loop.
- You should carefully consider your plan base on the task, screenshot, and history actions.
- When something does not work as expected (dueto various reasons), sometimes a simple retry can solve the problem, but if it doesn't (you can see that from the history), you should SWITCH to other solutions.
- Sometimes there is some default text in the text field you want to type in, remember to delete them before typing.
- To delete some text, you can place the cursor at the right place and long press the backspace to delete all the text.
"""

SYSTEM_PROMPT = PROMPT_PREFIX + """
First, write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.
Then execute an action in the form of function. For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:

## Format
Thought: The process of thinking.
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>"""

MEMORY_PROMPT = """
New observation: {observation}
Carefully examine the information above to identify any important content that needs to be recorded.
IMPORTANT: Do not take notes on low-level actions! Do not take notes on action plans! Do not take notes on progress! Do not take notes on current status! 
Only keep track of **significant textual or visual information on the screenshot** relevant to the user's request that might be useful for future actions!
The information you record is usually some numbers or texts displayed on the screen relevant to the user's request!
If the information is only useful for the next action, you should not record it.
Output the updated important notes, combining the old and new ones. If nothing new to record, copy the existing important notes.

## Format
Important notes:
- ...
- ...
"""

SUMMARY_PROMPT_TEMPLATE = (
    '\nThe (overall) user goal/request is: {goal}\n'
    'Now I want you to summerize the latest step.\n'
    'You will be given the screenshot before you performed the action,'
    ' the action you chose (together with the reason) and the screenshot'
    ' after the action was performed.\n'
    'This is the screenshot before the action:{image_before_action}\n'
    'This is the screenshot after the action:{image_after_action}\n'
    'This is the action you picked: {action}\n'
    'Based on the reason: {reason}\n\n'
    'By comparing the two screenshots and the'
    ' action performed, give a brief summary of this step. This summary'
    ' will be added to action history and used in future action selection,'
    ' so try to include essential information you think that will be most'
    ' useful for future action selections like what you'
    ' intended to do, why, if it worked as expected, if not'
    ' what might be the reason (be critical, the action/reason might be'
    ' wrong), what should/should not be done next and so on. Some more'
    ' rules/tips you should follow:\n'
    '- Keep it short (better less than 50 words) and in a single line\n'
    "- Some actions (like `wait`) don't involve screen change,"
    ' you can just assume they work as expected.\n'
    '- Given this summary will be added into action history, it can be used as'
    ' memory to include information that needs to be remembered, or shared'
    ' between different apps.\n\n'
    'Summary of this step: '
)

ANSWER_PROMPT_TEMPLATE = """
The (overall) user query is: {goal}
Now you have finished the task. I want you to provide an answer to the user query.
Answer with the following format:

## Format
<tool_call>
{{"name": "mobile_use", "arguments": {{"action": "answer", "text": <your-answer>}}}}
</tool_call>"""

ACTION_SPACE = ["key", "click", "left_click", "long_press", "swipe", "scroll", "type", "answer", "system_button", "open", "wait", "terminate"]

def _parse_response(content: str, size: tuple[float, float], raw_size: tuple[float, float]) -> Action:
    reason = re.search(r"Thought:(.*?)(?=\n|Action:|<tool_call>|\{\"name\": \"mobile_use\",)", content, flags=re.DOTALL)
    if reason:
        reason_s = reason.group(1).strip()
    else:
        reason_s = None
    summary = re.search(r'Summary:(.*)', content, flags=re.DOTALL)
    if summary:
        summary_s = summary.group(1).strip()
    else:
        summary_s = None
    action = re.search(r'{"name": "mobile_use",(.*?)}}', content, flags=re.DOTALL)
    if not action:
        raise Exception("Cannot extract action in the content.")
    action_s = '{"name": "mobile_use",' + action.group(1).strip() + '}}'
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


@Agent.register('QwenWithSummary')
class QwenWithSummaryAgent(Agent):
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

        self.memory = 'None'

    def reset(self, goal: str='') -> None:
        """Reset the state of the agent.
        """
        self._init_data(goal=goal)
        self.memory = 'None'

    def _get_curr_step_data(self) -> StepData:
        if len(self.trajectory) > self.curr_step_idx:
            return self.trajectory[self.curr_step_idx]
        else:
            return None

    def step(self):
        """Execute the task with maximum number of steps.

        Returns: Answer
        """
        logger.info("Step %d ... ..." % self.curr_step_idx)
        answer = None

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
                    {'type': 'text','text': f'The user query: {self.goal}\nTask progress (You have done the following operation on the current device): None\n'},
                    {'type': 'text','text': f'Important notes (During the operations, you record the following contents on the screenshot for use in subsequent operations): {self.memory}\n'},
                    {}, # Place holder for observation
                    {}, # Place holder for image
                ]
            })
        user_message_length = len(self.messages[1]['content'])

        # Fixed Picture sequence inconsistency problem in vllm0.7.2 
        # and Compatible QwenAPI error: '<400> InternalError.Algo.InvalidParameter: Invalid text: <|image_pad|>'
        observation = '' if 'dashscope.aliyuncs.com' in str(self.vlm.client.base_url) else IMAGE_PLACEHOLDER
        self.messages[-1]['content'][2] = {
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
        self.messages[-1]['content'][3] = img_msg

        # Add new step data
        self.trajectory.append(StepData(
            step_idx=self.curr_step_idx,
            curr_env_state=env_state,
            vlm_call_history=[]
        ))

        step_data = self.trajectory[-1]
        response = self.vlm.predict(self.messages, stop=['Summary'])

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
                    'text': f"Failed to parse the action from: {content}.\nError is {e.args}\nPlease follow the format to provide a valid action:\n" + \
                """
## Format
Thought: The process of thinking.
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
                """.strip()
                }
                self.messages[-1]['content'].append(msg)
                response = self.vlm.predict(self.messages, stop=['Summary'])
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

        # Summary
        summary = None
        summary_messages = []
        summary_messages.append({
            'role': 'user',
            'content': [
                {"type": "text", "text": PROMPT_PREFIX.format(width=resized_width, height=resized_height) + SUMMARY_PROMPT_TEMPLATE.format(
                    goal=self.goal,
                    image_before_action=observation,
                    image_after_action=observation,
                    action=action,
                    reason=reason
                )},
                {
                "type": "image_url",
                "image_url": {"url": encode_image_url(pixels)}
                },
                {
                "type": "image_url",
                "image_url": {"url": encode_image_url(step_data.exec_env_state.pixels)}
                }
            ]
        })
        try:
            response = self.vlm.predict(summary_messages)
            summary = response.choices[0].message.content
            logger.info("Summary from VLM:\n%s" % summary)
        except Exception as e:
            logger.warning(f"Failed to get the summary. Error: {e}")

        self.messages[-1]['content'] = self.messages[-1]['content'][:user_message_length]
        if self.curr_step_idx == 0:
            self.messages[-1]['content'][0]['text'] = f'The user query: {self.goal}\nTask progress (You have done the following operation on the current device): '

        if action is None:
            self.messages[-1]['content'][0]['text'] += f'\nStep {self.curr_step_idx + 1}: None'
        else:
            self.messages[-1]['content'][0]['text'] += f'\nStep {self.curr_step_idx + 1}: Thought: {reason}\n<tool_call>\n{action_s}\n</tool_call>\nSummary: {summary}'

        # Memory
        memory_messages = []
        memory_messages.append({
            'role': 'system',
            'content': [
                {"type": "text", "text": "You are a helpful AI assistant for operating mobile phones. Your goal is to take notes of important content relevant to the user's request."}
            ]
        })
        memory_messages.append({
            'role': 'user',
            'content': [
                self.messages[-1]['content'][0],
                self.messages[-1]['content'][1],
                {"type": "text", "text": MEMORY_PROMPT.format(observation=observation)},
                {
                "type": "image_url",
                "image_url": {"url": encode_image_url(step_data.exec_env_state.pixels)}
                }
            ]
        })
        print("MEMORY MESSAGES")
        print(memory_messages[0])
        print(memory_messages[1]['content'][:3])
        try:
            response = self.vlm.predict(memory_messages)
            memory = response.choices[0].message.content
            logger.info("Memory from VLM:\n%s" % memory)
            self.memory = memory
            self.messages[-1]['content'][1]['text'] = f'Important notes (During the operations, you record the following contents on the screenshot for use in subsequent operations): {self.memory}\n'
        except Exception as e:
            logger.warning(f"Failed to get the memory. Error: {e}")

        # Answer
        if self.status == AgentStatus.FINISHED:
            msg = {
                'type': 'text', 'text': ANSWER_PROMPT_TEMPLATE.format(goal=self.goal)
            }
            self.messages[-1]['content'].append(msg)
            try:
                response = self.vlm.predict(self.messages)
                content = response.choices[0].message.content
                logger.info("Answer from VLM:\n%s" % content)
                _, answer, _, _ = _parse_response(content, (resized_width, resized_height), env_state.pixels.size)
                answer = answer.parameters['text']
                step_data.answer = answer
                logger.info("Answer from VLM:\n%s" % answer)
            except Exception as e:
                logger.warning(f"Failed to get the answer. Error: {e}")

        step_data.action = action
        step_data.thought = reason
        step_data.additional_info['summary'] = summary

        return answer


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
