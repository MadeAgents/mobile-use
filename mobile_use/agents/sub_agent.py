from abc import ABC, abstractmethod
import re
import json
import logging

from mobile_use.scheme import *
from mobile_use.utils import encode_image_url, smart_resize, remove_img_placeholder, is_same_image, diff_image

__all__ = ['Planner', 'Operator', 'Reflector', 'LongReflector', 'NoteTaker', 'Processor', 'Evaluator', 'TaskSummarizer', 'ExperienceExtractor', 'Evolutor', 'UITARSOperator']

logger = logging.getLogger(__name__)

# Fix Picture sequence inconsistency problem in vllm0.7.2 
# If you are using QwenAPI from 'dashscope.aliyuncs.com', replace IMAGE_PLACEHOLDER with ''
IMAGE_PLACEHOLDER = '<|vision_start|><|image_pad|><|vision_end|>'

ACTION_SPACE = ["key", "click", "left_click", "long_press", "swipe", "scroll", "type", "clear_text", "answer", "system_button", "wait", "terminate", "take_note"]


def get_history(trajectory: List[StepData], num_histories=None):
    start_idx = 0 if num_histories is None else max(0, len(trajectory) - num_histories)
    history = []
    for i in range(start_idx, len(trajectory)):
        step_list = []
        step_list.append(f"Action: {trajectory[i].action_desc}")
        step_list.append(f"<tool_call> {trajectory[i].action_s} </tool_call>")
        if hasattr(trajectory[i], "summary") and trajectory[i].summary is not None:
            step_list.append(f"Summary: {trajectory[i].summary}")
        if hasattr(trajectory[i], "reflection_outcome") and trajectory[i].reflection_outcome is not None:
            if trajectory[i].reflection_outcome == "A":
                step_list.append("Successful")
            elif trajectory[i].reflection_outcome in ["B", "C"]:
                step_list.append("Failed")
                step_list.append(f"Feedback: {trajectory[i].reflection_error}")
        elif hasattr(trajectory[i], "long_reflaction_outcome") and trajectory[i].long_reflection_outcome is not None:
            if trajectory[i].long_reflection_outcome == "A":
                step_list.append("Successful")
            elif trajectory[i].long_reflection_outcome in ["B"]:
                step_list.append("Failed")
                step_list.append(f"Feedback: {trajectory[i].long_reflection_error}")
        history.append(f"Step-{i+1}: {'; '.join(step_list)}")
    return history


class SubAgent(ABC):
    @abstractmethod
    def get_message(self, episodedata: EpisodeData) -> list:
        pass
    @abstractmethod
    def parse_response(self, response: str):
        pass


"""
Call in the beginning of each step.
"""
class Planner(SubAgent):
    def get_message(self, episodedata: EpisodeData) -> list:
        messages = []
        trajectory = episodedata.trajectory
        current_step = trajectory[-1]

        pixels = current_step.curr_env_state.pixels.copy()
        resized_height, resized_width = smart_resize(height=pixels.height, width=pixels.width)
        
        # Add system prompt
        messages.append({
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a helpful AI assistant for operating mobile phones. Your goal is to track progress and devise high-level plans to achieve the user's requests. Think as if you are a human user operating the phone."
                }
            ]
        })

        # Add user prompt
        prompt = "### User Instruction ###\n"
        prompt += f"{episodedata.goal}\n\n"

        prompt += "### Current Screenshot ###\n"
        prompt += f"{IMAGE_PLACEHOLDER}\n"
        prompt += (
            f"The image is a screenshot showing the current state of the phone. "
            f"Its width and height are {resized_width} and {resized_height} pixels, respectively.\n\n"
        )

        if len(trajectory) == 1:
            # first time planning
            prompt += "---\n"
            prompt += "Think step by step and make an high-level plan to achieve the user's instruction. If the request is complex, break it down into subgoals. If the request involves exploration, include concrete subgoals to quantify the investigation steps. The screenshot displays the starting state of the phone.\n\n"
            prompt += "---\n"

            prompt += "Provide your output in the following format which contains three parts:\n\n"
            prompt += "### Thought ###\n"
            prompt += "A detailed explanation of your rationale for the plan and subgoals.\n\n"
            prompt += "### Plan ###\n"
            prompt += "1. first subgoal\n"
            prompt += "2. second subgoal\n"
            prompt += "...\n\n"
            prompt += "### Current Subgoal ###\n"
            prompt += "The first subgoal you should work on.\n\n"
        else:
            previous_step = trajectory[-2]
            # continue planning
            prompt += "### Current Plan ###\n"
            prompt += f"{previous_step.plan}\n\n"
            prompt += "### Previous Subgoal ###\n"
            prompt += f"{previous_step.sub_goal}\n\n"

            prompt += "---\n"
            prompt += "The sections above provide an overview of the plan you are following, the current subgoal you are working on. The screenshot displays the current state of the phone.\n"
            prompt += "Carefully assess the current status to determine if the task has been fully completed. If the user's request involves exploration, ensure you have conducted sufficient investigation. If you are confident that no further actions are required, mark the task as \"Finished\" in your output. If the task is not finished, outline the next steps. If you are stuck with errors, think step by step about whether the overall plan needs to be revised to address the error.\n"
            prompt += "NOTE: If the current situation prevents proceeding with the original plan or requires clarification from the user, make reasonable assumptions and revise the plan accordingly. Act as though you are the user in such cases.\n\n"
            
            prompt += "---\n"
            prompt += "Provide your output in the following format, which contains three parts:\n\n"
            prompt += "### Thought ###\n"
            prompt += "Provide a detailed explanation of your rationale for the plan and subgoals.\n\n"
            prompt += "### Plan ###\n"
            prompt += "If an update is required for the high-level plan, provide the updated plan here. Otherwise, keep the current plan and copy it here.\n\n"
            prompt += "### Current Subgoal ###\n"
            prompt += "The next subgoal to work on. If the previous subgoal is not yet complete, copy it here. If all subgoals are completed, write \"Finished\".\n"
        
        messages.append({
            "role": "user",
            "content": [
                {"type": "text","text": prompt},
                {"type": "image_url","image_url": {"url": encode_image_url(pixels)}, "resized_height": resized_height, "resized_width": resized_width}
            ]
        })

        return messages

    def parse_response(self, response: str):
        thought = response.split("### Thought ###")[-1].split("### Plan ###")[0].replace("\n", " ").replace("  ", " ").strip()
        plan = response.split("### Plan ###")[-1].split("### Current Subgoal ###")[0].replace("\n", " ").replace("  ", " ").strip()
        current_subgoal = response.split("### Current Subgoal ###")[-1].replace("\n", " ").replace("  ", " ").strip()
        return thought, plan, current_subgoal


class Operator(SubAgent):
    def __init__(self, num_histories: int = None):
        super().__init__()
        self.num_histories = num_histories

    def get_message(self, episodedata: EpisodeData, device_time: str = None, is_answer: bool = False) -> list:
        messages = []
        trajectory = episodedata.trajectory
        current_step = trajectory[-1]
        
        pixels = current_step.curr_env_state.pixels.copy()
        resized_height, resized_width = smart_resize(height=pixels.height, width=pixels.width)

        if not is_answer:
            # Add system prompt
            messages.append({
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": f"""
You are a helpful AI assistant for operating mobile phones. Your goal is to choose the correct actions to complete the user's instruction. Think as if you are a human user operating the phone.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{{"type": "function", "function": {{"name_for_human": "mobile_use", "name": "mobile_use", "description": "Use a touchscreen to interact with a mobile device, and take screenshots.
* This is an interface to a mobile device with touchscreen. You can perform actions like clicking, typing, swiping, etc.
* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.
* The screen's resolution is {resized_width}x{resized_height}.
* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.", "parameters": {{"properties": {{"action": {{"description": "The action to perform. The available actions are:
* `key`: Perform a key event on the mobile device.
    - This supports adb's `keyevent` syntax.
    - Examples: "volume_up", "volume_down", "power", "camera", "clear".
* `click`: Click the point on the screen with coordinate (x, y).
* `long_press`: Press the point on the screen with coordinate (x, y) for specified seconds.
* `swipe`: Swipe from the starting point with coordinate (x, y) to the end point with coordinates2 (x2, y2).
* `type`: Input the specified text into the activated input box.
* `clear_text`: Clear the text in the activated input box.
* `system_button`: Press the system button.
* `wait`: Wait specified seconds for the change to happen.
* `take_note`: Extract and save important texts, numbers, or images on the current screen for future use. put information in the `text` parameter.
* `terminate`: Terminate the current task and report its completion status.", "enum": ["key", "click", "long_press", "swipe", "type", "clear_text", "system_button", "wait", "take_note", "terminate"], "type": "string"}}, "coordinate": {{"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=click`, `action=long_press`, and `action=swipe`.", "type": "array"}}, "coordinate2": {{"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=swipe`.", "type": "array"}}, "text": {{"description": "Required only by `action=key`, `action=type`, and `action=take_note`.", "type": "string"}}, "time": {{"description": "The seconds to wait. Required only by `action=long_press` and `action=wait`.", "type": "number"}}, "button": {{"description": "Back means returning to the previous interface, Home means returning to the desktop, Menu means opening the application background menu, and Enter means pressing the enter. Required only by `action=system_button`", "enum": ["Back", "Home", "Menu", "Enter"], "type": "string"}}, "status": {{"description": "The status of the task. Required only by `action=terminate`.", "type": "string", "enum": ["success", "failure"]}}}}, "required": ["action"], "type": "object"}}, "args_format": "Format the arguments as a JSON object."}}}}
</tools>
"""
# * `terminate`: Terminate the current task and report its completion status.", "enum": ["key", "click", "long_press", "swipe", "type", "clear_text", "system_button", "open", "wait", "terminate"], "type": "string"}}, "coordinate": {{"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=click`, `action=long_press`, and `action=swipe`.", "type": "array"}}, "coordinate2": {{"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=swipe`.", "type": "array"}}, "text": {{"description": "Required only by `action=key`, `action=type`, and `action=open`.", "type": "string"}}, "time": {{"description": "The seconds to wait. Required only by `action=long_press` and `action=wait`.", "type": "number"}}, "button": {{"description": "Back means returning to the previous interface, Home means returning to the desktop, Menu means opening the application background menu, and Enter means pressing the enter. Required only by `action=system_button`", "enum": ["Back", "Home", "Menu", "Enter"], "type": "string"}}, "status": {{"description": "The status of the task. Required only by `action=terminate`.", "type": "string", "enum": ["success", "failure"]}}}}, "required": ["action"], "type": "object"}}, "args_format": "Format the arguments as a JSON object."}}}}
                    }
                ]
            })
        else:
            messages.append({
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": f"""
You are a helpful AI assistant for operating mobile phones. Your goal is to choose the correct actions to complete the user's instruction. Think as if you are a human user operating the phone.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{{"type": "function", "function": {{"name_for_human": "mobile_use", "name": "mobile_use", "description": "Use a touchscreen to interact with a mobile device, and take screenshots.
* This is an interface to a mobile device with touchscreen. You can perform actions like clicking, typing, swiping, etc.
* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.
* The screen's resolution is {resized_width}x{resized_height}.
* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.", "parameters": {{"properties": {{"action": {{"description": "The action to perform. The available actions are:
* `answer`: Answer the user query. ", "enum": ["answer"], "type": "string"}}, "text": {{"description": "Required only by `action=answer`.", "type": "string"}}}}, "required": ["action"], "type": "object"}}, "args_format": "Format the arguments as a JSON object."}}}}
</tools>
"""
                    }
                ]
            })

        # Add user prompt
        prompt = "### User Instruction ###\n"
        prompt += f"{episodedata.goal}\n\n"

        if device_time is not None:
            prompt += "### Current Time ###\n"
            prompt += f"{device_time}\n\n"

        if hasattr(current_step, "plan") and current_step.plan is not None:
            prompt += "### Overall Plan ###\n"
            prompt += f"{current_step.plan}\n\n"

        if hasattr(current_step, "sub_goal") and current_step.sub_goal is not None:
            prompt += "### Current Subgoal ###\n"
            prompt += f"{current_step.sub_goal}\n\n"

        prompt += "### Latest History Operations ###\n"
        prompt += "You have done the following operation on the current device):\n"
        if len(trajectory) > 1 and (self.num_histories is None or self.num_histories > 0):
            if is_answer:
                history = get_history(trajectory[:-1], self.num_histories)
            else:
                history = get_history(trajectory[:-1], self.num_histories)
            prompt += "\n".join(history)
            prompt += "\n\n"
        else:
            prompt += "No actions have been taken yet.\n\n"

        if len(trajectory) > 1:
            previous_step = trajectory[-2]
            if hasattr(previous_step, "progress") and previous_step.progress is not None:
                prompt += "### Progress ###\n"
                prompt += "After completing the history operations, you have the following thoughts about the progress of user\'s instruction completion:\n"
                prompt += f"Completed contents:\n{previous_step.progress}\n\n"

            if hasattr(previous_step, "memory") and previous_step.memory is not None:
                prompt += "### Memory ###\n"
                prompt += "During the operations, you record the following contents on the screenshot for use in subsequent operations:\n"
                prompt += f"{previous_step.memory}\n\n"

            if hasattr(episodedata, "memory"):
                prompt += "### Memory ###\n"
                prompt += "During previous operations, you have used the action `take_note` to record the following contents on the screenshot:\n"
                if episodedata.memory == "":
                    prompt += "None\n\n"
                else:
                    prompt += f"{episodedata.memory}\n\n"

            if not is_answer:
                if hasattr(previous_step, "reflection_outcome") and previous_step.reflection_outcome is not None and previous_step.reflection_outcome in ['B', 'C']:
                    # prompt += "### Latest operation ###\n"
                    # prompt += f"You previously wanted to perform the operation \"{previous_step.action_desc}\" on this page and executed the Action \"{previous_step.action_s}\". But you find that this operation does not meet your expectation.\nFeedback:{previous_step.reflection_error}\n You need to reflect and revise your operation this time."
                    # prompt += "\n\n"
                    prompt += "### Latest operation ###\n"
                    prompt += f"You previously wanted to perform the operation \"{previous_step.action_desc}\" on this page and executed the Action \"{previous_step.action_s}\". But the reflector find that this operation may not meet your expectation.\nFeedback:{previous_step.reflection_error}\n If you think it is reasonable, you need to reflect and revise your operation this time. If you think the reflector is not correct, you can ignore the feedback."
                    prompt += "\n\n"
                
                if hasattr(previous_step, "long_reflection_outcome") and previous_step.long_reflection_outcome is not None and previous_step.long_reflection_outcome in ['B']:
                    prompt += "### Reflection ###\n"
                    prompt += "According to your history operations, you have the following reflection:\n"
                    prompt += f"Reflection: {previous_step.long_reflection_error}\n"
                    prompt += "If you think the reflection is reasonable, you need to reflect, reschedule, and revise your operation this time. Try to avoid the same mistake."
                    prompt += "\n\n"

            if hasattr(previous_step, "evaluation_result") and previous_step.evaluation_result is not None and "Failed" in previous_step.evaluation_result:
                prompt += "### Evaluation Result ###\n"
                prompt += "In the last step, you think the task is finished. However, the evaluator find that the task is not finished. This is the evaluation result:\n"
                prompt += f"Evaluation: {previous_step.evaluation_reason}\n"
                prompt += "If you think the evaluation is reasonable, perform more actions to finish the task. Otherwise, you can terminate the task.\n"
                prompt += "\n"

        prompt += "### Observation ###\n"
        prompt += f"This is the current screenshot of the phone. The screen's resolution is {resized_width}x{resized_height}."
        prompt += f"{IMAGE_PLACEHOLDER}\n\n"

#         prompt += "### Guidance ###\n"
#         prompt += """Here are some useful guidelines you need to follow:
# - If the task is finished, you should terminate the task in time!
# - If you stuck in an action, you should try to change the action or the correspoinding parameters. Do not always repeat the same action!
# - Sometimes there is some default text in the text field you want to type in, remember to delete them before typing.
# - To delete some text, you can place the cursor at the right place and long press the backspace to delete all the text.

# """
        if hasattr(episodedata, "input_tips") and episodedata.input_tips is not None:
            prompt += "### Tips ###\n"
            prompt += "From previous experience interacting with the device, you have collected the following tips that might be useful for deciding what to do next:\n"
            prompt += f"{episodedata.input_tips}\n\n"

            if hasattr(episodedata, "retrieved_tips") and episodedata.retrieved_tips is not None:
                prompt += "### Retrieved Tips ###\n"
                prompt += "You have also retrieved the following tips from similar tasks that might be useful for deciding what to do next:\n"
                prompt += f"{episodedata.retrieved_tips}\n\n"

        prompt += "### Response Requirements ###\n"
        prompt += """First, think about the requirements that have been completed in previous operations and the requirements that need to be completed in the next one operation. Put your thinking process in one sentence in `Thought` part.
Secend, provide a brief description of the chosen action in `Action` part. Only describe the current ONE action. Don't describe the future ones or the whole plan.
Last, execute an action in the form of function. For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:

### Format ###
Thought: ... (Your thinking process)
Action: ... (Your action description)
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>"""

        if is_answer:
            prompt += """

The (overall) user query is: {goal}
Now you have finished the task. I want you to provide an answer to the user query.
Answer with the following format:

## Format
<tool_call>
{{"name": "mobile_use", "arguments": {{"action": "answer", "text": <your-answer>}}}}
</tool_call>""".format(goal=episodedata.goal)

        messages.append({
            "role": "user",
            "content": [
                {"type": "text","text": prompt},
                {"type": "image_url","image_url": {"url": encode_image_url(pixels)}, "resized_height": resized_height, "resized_width": resized_width}
            ]
        })

        return messages
    
    def parse_response(self, content: str, size: tuple[float, float], raw_size: tuple[float, float]):
        thought = re.search(r"Thought:(.*?)(?=\n|Action:|<tool_call>|\{\"name\": \"mobile_use\",)", content, flags=re.DOTALL)
        if thought:
            thought_s = thought.group(1).strip()
        else:
            thought_s = None
        action_desc = re.search(r"Action:(.*?)(?=\n|<tool_call>|\{\"name\": \"mobile_use\",)", content, flags=re.DOTALL)
        if action_desc:
            action_desc_s = action_desc.group(1).strip()
        else:
            action_desc_s = None
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

        return thought_s, action_a, action_s, action_desc_s


"""
Call after executing each action.
"""
class Reflector(SubAgent):
    def __init__(self):
        super().__init__()
        self.valid_options = ['A', 'B', 'C', 'D']

    def get_message(self, episodedata: EpisodeData) -> list:
        messages = []
        trajectory = episodedata.trajectory
        current_step = trajectory[-1]

        pixels_before = current_step.curr_env_state.pixels.copy()
        resized_height, resized_width = smart_resize(height=pixels_before.height, width=pixels_before.width)
        pixels_after = current_step.exec_env_state.pixels.copy()

        diff_flag = False
        new_img1, new_img2 = diff_image(pixels_before, pixels_after)
        if new_img1 is not None:
            pixels_before, pixels_after = new_img1, new_img2
            diff_flag = True
        
        # Add system prompt
        messages.append({
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a helpful AI assistant for operating mobile phones. Your goal is to verify whether the latest action produced the expected behavior."
                }
            ]
        })

        # Add user prompt
        prompt = ""
        prompt += "### User Instruction ###\n"
        prompt += f"{episodedata.goal}\n\n"

        if hasattr(current_step, "sub_goal") and current_step.sub_goal is not None:
            prompt += "### Current Subgoal ###\n"
            prompt += f"{current_step.sub_goal}\n\n"

        prompt += "---\n"
        prompt += f"Screenshot before latest action: {IMAGE_PLACEHOLDER}\n"
        prompt += f"Screenshot after latest action: {IMAGE_PLACEHOLDER}\n"
        prompt += f"The two images are two phone screenshots before and after your latest action. " 
        prompt += f"The width and height are {resized_width} and {resized_height} pixels, respectively.\n"
        if diff_flag:
            logger.info("The last action successfully produces some changes. The difference between the two images is highlighted in red boxes.")
            prompt += "The last action successfully produces some observable changes. The difference between the two images is highlighted in red boxes. You can find it on the images.\n"
        prompt += "\n"

        prompt += "---\n"
        prompt += "### Latest Action ###\n"
        prompt += f"Action: {current_step.action_s}\n"
        prompt += f"Expectation: {current_step.action_desc}\n\n"

        prompt += "---\n"
        prompt += "Carefully examine the information provided above to determine whether the last action meets the expectation. If not, identify the failure mode and provide reasoning on the potential reason causing this failure. Note that for the “Swipe” action, it may take multiple attempts to display the expected content. Thus, for a \"Swipe\" action, if the screen shows new content, it usually meets the expectation.\n\n"

        prompt += "Provide your output in the following format containing three parts:\n\n"
        prompt += "### Outcome ###\n"
        prompt += "Choose from the following options. Give your answer as \"A\", \"B\",\"C\" or \"D\":\n"
        prompt += "A: Successful or Partially Successful. The result of the last action meets the expectation, or on the right path to meet the expectation.\n"
        prompt += "B: Failed. The last action results in a wrong page. I need to return to the previous state.\n"
        prompt += "C: Failed. The last action produces no changes.\n"
        prompt += "D: Uncertain. Can't determine whether the last action meets the expectation.\n"
        prompt += "NOTE: In some cases, the action may not produce any observable feedback, such as click a `save` or `add` button. You can't determine whether the action meets the expectation. In this case, you can choose \"D\".\n"
        prompt += "\n"

        prompt += "### Error Description ###\n"
        prompt += "If the action failed, provide a detailed description of the error and the potential reason causing this failure. If the action succeeded, put \"None\" here.\n\n"

        messages.append({
            "role": "user",
            "content": [
                {"type": "text","text": prompt},
                {"type": "image_url","image_url": {"url": encode_image_url(pixels_before)}, "resized_height": resized_height, "resized_width": resized_width},
                {"type": "image_url","image_url": {"url": encode_image_url(pixels_after)}, "resized_height": resized_height, "resized_width": resized_width}
            ]
        })

        return messages

    def parse_response(self, response: str) -> dict:
        outcome = response.split("### Outcome ###")[-1].split("### Error Description ###")[0].replace("\n", " ").replace("  ", " ").strip()
        error_description = response.split("### Error Description ###")[-1].split("### Explanation ###")[0].replace("\n", " ").replace("  ", " ").strip()
        return outcome, error_description


class LongReflector(SubAgent):
    def __init__(
        self,
        evoke_every_steps: int = 5,
        cold_steps: int = 3,
        detect_error: bool = True,
        num_histories = 'auto',
        num_latest_screenshots: int = 0
    ):
        super().__init__()
        self.valid_options = ['A', 'B']
        self.evoke_every_steps = evoke_every_steps
        self.cold_steps = cold_steps
        self.sleep_count = 0
        self.detect_error = detect_error
        if num_histories == 'auto':
            self.num_histories = evoke_every_steps
        else:
            self.num_histories = num_histories
        self.num_latest_screenshots = num_latest_screenshots
    
    def detect(
        self, 
        episodedata: EpisodeData,
        max_repeat_action: int = 3,
        max_repeat_action_series: int = 2,
        max_repeat_screen: int = 3,
        max_fail_count: int = 3
    ) -> str:
        error = ''
        trajectory = episodedata.trajectory
        if len(trajectory) < min(max_repeat_action, max_repeat_screen, max_fail_count):
            return error
        current_step = trajectory[-1]

        # detect repeated actions
        repeat_action = 1
        if current_step.action not in ["swipe", "scroll"]:
            for step in trajectory[:-1][::-1]:
                if step.action == current_step.action:
                    repeat_action += 1
                else:
                    break
                if repeat_action >= max_repeat_action:
                    error += f"The action `{current_step.action_s}` has repeated more than {max_repeat_action} times.\n"
                    break

        # detect repeated action series
        if len(trajectory) >= 4:
            if trajectory[-1].action == trajectory[-3].action and trajectory[-2].action == trajectory[-4].action:
                error += f"The latest two actions have repeated more than {max_repeat_action_series} times.\n"
        if len(trajectory) >= 6:
            if trajectory[-1].action == trajectory[-4].action and trajectory[-2].action == trajectory[-5].action and trajectory[-3].action == trajectory[-6].action:
                error += f"The latest three actions have repeated more than {max_repeat_action_series} times.\n"

        # detect repeated screenshots
        repeat_screen = 1
        for step in trajectory[:-1][::-1]:
            if is_same_image(step.exec_env_state.pixels.copy(), current_step.exec_env_state.pixels.copy()):
                repeat_screen += 1
            else:
                break
            if repeat_screen >= max_repeat_screen:
                error += f"The screen has kept unchanged for more than {max_repeat_screen} times.\n"
                break

        # detect fail reflection
        if hasattr(current_step, "reflection_outcome") and current_step.reflection_outcome is not None:
            fail_count = 0
            for step in trajectory[::-1]:
                if step.reflection_outcome in ['B', 'C']:
                    fail_count += 1
                else:
                    break
                if fail_count >= max_fail_count:
                    error += f"You have encountered several failed attempts.\n"
                    break

        if error != '':
            logger.warning(f"Long Reflector detects error: {error}")

        return error
        

    def get_message(self, episodedata: EpisodeData) -> list:
        error = ''
        if self.detect_error:
            error = self.detect(episodedata)
        step_idx = len(episodedata.trajectory)

        if step_idx % self.evoke_every_steps != 0 and error == '':
            self.sleep_count += 1
            return None
        if self.sleep_count < self.cold_steps:
            self.sleep_count += 1
            return None

        self.sleep_count = 0
        messages = []
        trajectory = episodedata.trajectory
        current_step = trajectory[-1]

        num_latest_screenshots = min(self.num_latest_screenshots, len(trajectory))
        if num_latest_screenshots > 0:
            screenshots = [step.exec_env_state.pixels.copy() for step in trajectory[-num_latest_screenshots:]]
            resized_height, resized_width = smart_resize(height=screenshots[0].height, width=screenshots[0].width)

        # Add system prompt
        messages.append({
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a helpful AI assistant for operating mobile phones. Your goal is to verify whether the latest action produced the expected behavior."
                }
            ]
        })

        # Add user prompt
        prompt = "### User Instruction ###\n"
        prompt += f"{episodedata.goal}\n\n"

        if hasattr(current_step, "plan") and current_step.plan is not None:
            prompt += "### Overall Plan ###\n"
            prompt += "This is the high-level plan you made to achieve the user's instruction:\n"
            prompt += f"{current_step.plan}\n\n"
        
        prompt += "### Latest History Operations ###\n"
        prompt += "You have done the following operation on the current device):\n"
        history = get_history(trajectory, self.num_histories)
        prompt += "\n".join(history)
        if hasattr(current_step, "answer") and current_step.answer is not None:
            prompt += f"\nFinal answer: {current_step.answer}"
        prompt += "\n\n"
            
        if hasattr(current_step, "progress") and current_step.progress is not None:
            prompt += "### Progress ###\n"
            prompt += "After completing the history operations, you have the following thoughts about the progress of user\'s instruction completion:\n"
            prompt += f"Completed contents:\n{current_step.progress}\n\n"
        
        if num_latest_screenshots > 0:
            prompt += "### Latest Screenshots\n"
            prompt += f"This is the latest screenshots when you are performing operations to complete user\'s instruction. The width and height are {resized_width} and {resized_height} pixels, respectively.\n"
            for i in range(len(screenshots)):
                prompt += f"{IMAGE_PLACEHOLDER}"
            prompt += "\n\n"
        
        if error != '':
            prompt += "### Warning Information ###\n"
            prompt += f"{error}\n\n"

        prompt += "---\n"
        prompt += "Carefully examine the information provided above to determine whether the latest history operations are on the right path to completing the user's instruction.\n\n"

        prompt += "Provide your output in the following format containing three parts:\n\n"
        prompt += "### Outcome ###\n"
        prompt += "Choose from the following options. Give your answer as \"A\" or \"B\":\n"
        prompt += "A: Successful or Partially Successful. The latest history operations meets the expectation. Or although there is something wrong, it has been corrected.\n"
        prompt += "B: Failed. The latest history operations are on the wrong path.  It may result in a wrong page, or produce no changes, or being meaninglessly repeating.\n"

        prompt += "### Error Description ###\n"
        prompt += "If the actions failed, provide a detailed description of the error and the potential reason causing the failure. If the action succeeded, put \"None\" here.\n\n"

        message_content = [{"type": "text","text": prompt}]
        if num_latest_screenshots > 0:
            for screenshot in screenshots:
                message_content.append({"type": "image_url","image_url": {"url": encode_image_url(screenshot)}, "resized_height": resized_height, "resized_width": resized_width})
        messages.append({"role": "user","content": message_content})

        return messages

    def parse_response(self, response: str) -> dict:
        outcome = response.split("### Outcome ###")[-1].split("### Error Description ###")[0].replace("\n", " ").replace("  ", " ").strip()
        error_description = response.split("### Error Description ###")[-1].split("### Explanation ###")[0].replace("\n", " ").replace("  ", " ").strip()
        return outcome, error_description


"""
Gemerate memory
"""
class NoteTaker(SubAgent):
    def get_message(self, episodedata: EpisodeData) -> list:
        messages = []
        trajectory = episodedata.trajectory
        current_step = trajectory[-1]

        pixels = current_step.exec_env_state.pixels.copy()
        resized_height, resized_width = smart_resize(height=pixels.height, width=pixels.width)

        # Add system prompt
        messages.append({
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a helpful AI assistant for operating mobile phones. Your goal is to take notes of important content relevant to the user's request."
                }
            ]
        })

        # Add user prompt
        prompt = "### User Instruction ###\n"
        prompt += f"{episodedata.goal}\n\n"

        if hasattr(current_step, "plan") and current_step.plan is not None:
            prompt += "### Overall Plan ###\n"
            prompt += f"{current_step.plan}\n\n"

        if hasattr(current_step, "sub_goal") and current_step.sub_goal is not None:
            prompt += "### Current Subgoal ###\n"
            prompt += f"{current_step.sub_goal}\n\n"

        prompt += "### Existing Important Notes ###\n"
        if len(trajectory) > 1:
            previous_step = trajectory[-2]
            prompt += f"{previous_step.memory}\n\n"
        else:
            prompt += "No important notes recorded.\n\n"

        prompt += "### Current Screenshot ###\n"
        prompt += f"{IMAGE_PLACEHOLDER}\n"
        prompt += (
            f"The image is a screenshot showing the current state of the phone. "
            f"Its width and height are {resized_width} and {resized_height} pixels, respectively.\n\n"
        )

        prompt += "---\n"
        prompt += "Carefully examine the information above to identify any important content that needs to be recorded. IMPORTANT: Do not take notes on low-level actions; only keep track of significant textual or visual information relevant to the user's request.\n\n"

        prompt += "Provide your output in the following format:\n"
        prompt += "### Important Notes ###\n"
        prompt += "The updated important notes, combining the old and new ones. If nothing new to record, copy the existing important notes. If you think some information in the existing important notes is no longer useful, you can remove it.\n"

        messages.append({
            "role": "user",
            "content": [
                {"type": "text","text": prompt},
                {"type": "image_url","image_url": {"url": encode_image_url(pixels)}, "resized_height": resized_height, "resized_width": resized_width}
            ]
        })

        return messages
    
    def parse_response(self, response: str) -> str:
        return response.split("### Important Notes ###")[-1].replace("\n", " ").replace("  ", " ").strip()


"""
Call in the end of each step.
"""
class Processor(SubAgent):
    def get_message(self, episodedata: EpisodeData) -> list:
        messages = []
        trajectory = episodedata.trajectory
        current_step = trajectory[-1]
        
        # Add system prompt
        messages.append({
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a helpful AI assistant for operating mobile phones. Your goal is to summarize the completed contents based on the history operations."
                }
            ]
        })

        # Add user prompt
        prompt = ""
        # prompt += "### Background ###\n"
        # prompt += f"There is an user\'s instruction which is: {episodedata.goal}. You are a mobile phone operating assistant and are operating the user\'s mobile phone.\n\n"

        if len(trajectory) > 1:
            prompt += "### History operations ###\n"
            prompt += "To complete the requirements of user\'s instruction, you have performed a series of operations. These operations are as follow:\n"
            history = get_history(trajectory[:-1])
            prompt += "\n".join(history)
            prompt += "\n"
            
            previous_step = trajectory[-2]
            prompt += "### Progress thinking ###\n"
            prompt += "After completing the history operations, you have the following thoughts about the progress:\n"
            prompt += f"Completed contents:\n{previous_step.progress}\n\n"

            prompt += "### Current operation ###\n"
            prompt += f"Action description: {current_step.action_desc}\n"
            prompt += f"Action: {current_step.action}\n\n"

            if hasattr(current_step, "reflection_outcome") and current_step.reflection_outcome is not None:
                if current_step.reflection_outcome in ['B', 'C']:
                    prompt += "### Reflection ###\n"
                    prompt += "According to your current operation, you have the following reflection:\n"
                    prompt += f"Reflection: {current_step.reflection_error}\n"
                    prompt += "\n"
            
            prompt += "### Response requirements ###\n"
            prompt += "Now you need to update the \"Completed contents\". Completed contents is a general summary of the current contents that have been completed based on the provided information.\n"
            prompt += "Note: Only descripe the actially performed action. The action purpose may be incouded in the action description. Don't include it in completed contents!\n"
            prompt += "Sometimes the action description describes more than one action, such as \"Click the text field and type in text\". You should only summarize the action that has been actually performed.\n"
            prompt += "\n"
            
            prompt += "### Output format ###\n"
            prompt += "Your output format is:\n"
            prompt += "### Completed contents ###\nUpdated Completed contents. Don\'t output the purpose of any operation. Just summarize the contents that have been actually completed."
            
        else:
            prompt += "### Current operation ###\n"
            prompt += "To complete the requirements of user\'s instruction, you have performed an operation. Your operation thought and action of this operation are as follows:\n"
            prompt += f"Action thought: {current_step.thought}\n"
            prompt += f"Action description: {current_step.action_desc}\n"
            prompt += f"Action: {current_step.action}\n\n"
            
            prompt += "### Response requirements ###\n"
            prompt += "Now you need to update the \"Completed contents\". Completed contents is a general summary of the current contents that have been completed based on the provided information.\n"
            prompt += "Note: Only descripe the actially performed action. The action purpose may be incouded in the action description. Don't include it in completed contents!\n"
            prompt += "\n"

            prompt += "### Output format ###\n"
            prompt += "Your output format is:\n"
            prompt += "### Completed contents ###\nGenerated Completed contents. Don\'t output the purpose of any operation. Just summarize the contents that have been actually completed.\n"
            prompt += "(Please use English to output)"
            
        messages.append({
            "role": "user",
            "content": [{"type": "text","text": prompt}]
        })

        return messages
    
    def parse_response(self, response: str):
        return response.split("### Completed contents ###")[-1].replace("\n", " ").replace("  ", " ").strip()


class OldProcessor(SubAgent):
    def get_message(self, episodedata: EpisodeData) -> list:
        messages = []
        trajectory = episodedata.trajectory
        current_step = trajectory[-1]
        
        # Add system prompt
        messages.append({
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a helpful AI mobile phone operating assistant."
                }
            ]
        })

        # Add user prompt
        prompt = "### Background ###\n"
        prompt += f"There is an user\'s instruction which is: {episodedata.goal}. You are a mobile phone operating assistant and are operating the user\'s mobile phone.\n\n"
        
        if len(trajectory) > 1:
            prompt += "### History operations ###\n"
            prompt += "To complete the requirements of user\'s instruction, you have performed a series of operations. These operations are as follow:\n"
            history = get_history(trajectory)
            prompt += "\n".join(history)
            prompt += "\n"
            
            previous_step = trajectory[-2]
            prompt += "### Progress thinking ###\n"
            prompt += "After completing the history operations, you have the following thoughts about the progress of user\'s instruction completion:\n"
            prompt += f"Completed contents:\n{previous_step.progress}\n\n"
            
            prompt += "### Response requirements ###\n"
            prompt += "Now you need to update the \"Completed contents\". Completed contents is a general summary of the current contents that have been completed based on the ### History operations ###.\n\n"
            
            prompt += "### Output format ###\n"
            prompt += "Your output format is:\n"
            prompt += "### Completed contents ###\nUpdated Completed contents. Don\'t output the purpose of any operation. Just summarize the contents that have been actually completed in the ### History operations ###."
            
        else:
            prompt += "### Current operation ###\n"
            prompt += "To complete the requirements of user\'s instruction, you have performed an operation. Your operation thought and action of this operation are as follows:\n"
            prompt += f"Operation thought: {current_step.thought}\n"
            prompt += f"Operation action: {current_step.action_desc}\n\n"
            
            prompt += "### Response requirements ###\n"
            prompt += "Now you need to combine all of the above to generate the \"Completed contents\".\n"
            prompt += "Completed contents is a general summary of the current contents that have been completed. You need to first focus on the requirements of user\'s instruction, and then summarize the contents that have been completed.\n\n"
            
            prompt += "### Output format ###\n"
            prompt += "Your output format is:\n"
            prompt += "### Completed contents ###\nGenerated Completed contents. Don\'t output the purpose of any operation. Just summarize the contents that have been actually completed in the ### Current operation ###.\n"
            prompt += "(Please use English to output)"
            
        messages.append({
            "role": "user",
            "content": [{"type": "text","text": prompt}]
        })

        return messages
    
    def parse_response(self, response: str):
        return response.split("### Completed contents ###")[-1].replace("\n", " ").replace("  ", " ").strip()


"""
Decide whether the user's instruction has been completed successfully.
"""
class Evaluator(SubAgent):
    def __init__(self, num_latest_screenshots: int = 3):
        super().__init__()
        self.num_latest_screenshots = num_latest_screenshots

    def get_message(self, episodedata: EpisodeData) -> list:
        messages = []
        trajectory = episodedata.trajectory
        last_step = trajectory[-1]

        num_latest_screenshots = min(self.num_latest_screenshots, len(trajectory))
        screenshots = [step.exec_env_state.pixels.copy() for step in trajectory[-num_latest_screenshots:]]
        resized_height, resized_width = smart_resize(height=screenshots[0].height, width=screenshots[0].width)
        
        # Add system prompt
        messages.append({
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a helpful AI assistant for operating mobile phones. Your goal is to evaluate whether the user's instruction has been completed successfully."
                }
            ]
        })

        # Add user prompt
        prompt = "### User Instruction ###\n"
        prompt += f"{episodedata.goal}\n\n"

        if hasattr(last_step, "plan") and last_step.plan is not None:
            prompt += "### Overall Plan ###\n"
            prompt += "This is the high-level plan you made to achieve the user's instruction:\n"
            prompt += f"{last_step.plan}\n\n"
        
        prompt += "### History operations ###\n"
        prompt += "To complete the requirements of user\'s instruction, you have performed a series of operations. These operations are as follow:\n"
        history = get_history(trajectory)
        prompt += "\n".join(history)
        if hasattr(last_step, "answer") and last_step.answer is not None:
            prompt += f"\nFinal answer: {last_step.answer}"
        prompt += "\n\n"
            
        # if hasattr(last_step, "progress") and last_step.progress is not None:
        #     prompt += "### Progress ###\n"
        #     prompt += "After completing the history operations, you have the following thoughts about the progress of user\'s instruction completion:\n"
        #     prompt += f"Completed contents:\n{last_step.progress}\n\n"
        
        prompt += "### Latest Screenshots\n"
        prompt += f"This is the latest screenshots when you are performing operations to complete user\'s instruction. The width and height are {resized_width} and {resized_height} pixels, respectively.\n"
        for i in range(len(screenshots)):
            prompt += f"{IMAGE_PLACEHOLDER}"
        prompt += "\n\n"

        prompt += "### Response requirements ###\n"
        prompt += "Carefully examine the latest screenshots and the information provided above to determine whether the user's instruction is successfully completed or not. Provide your output in the following format:\n\n"

        prompt += "### Result ###\n"
        prompt += "Choose from the following options. Give your answer as \"Success\", \"Failed\" or \"Uncertain\":\n"
        prompt += "Success: The task is successfully finished.\n"
        prompt += "Failed: The task is not finished.\n"
        prompt += "Uncertain: Can't determain whether the task is finished or not.\n"
        prompt += "NOTE: Be careful when judging the task as failed. If there is not enough evidence to determine whether the task is failed, you should choose \"Uncertain\".\n\n"

        prompt += "### Reason ###\n"
        prompt += "Provide reason for your answer.\n"

        message_content = [{"type": "text","text": prompt}]
        for screenshot in screenshots:
            message_content.append({"type": "image_url","image_url": {"url": encode_image_url(screenshot)}, "resized_height": resized_height, "resized_width": resized_width})
        messages.append({"role": "user","content": message_content})

        return messages
    
    def parse_response(self, response: str):
        result = response.split("### Result ###")[-1].split("### Reason ###")[0].replace("\n", " ").replace("  ", " ").strip()
        reason = response.split("### Reason ###")[-1].strip()
        return result, reason, None


class TaskSummarizer(SubAgent):
    def get_message(self, episodedata: EpisodeData, result: str) -> list:
        messages = []
        trajectory = episodedata.trajectory
        last_step = trajectory[-1]

        # Add system prompt
        messages.append({
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a helpful AI assistant for operating mobile phones. Your goal is to analyze and summarize a trajectory of mobile task execution."
                }
            ]
        })

        # Add user prompt
        prompt = ""
        prompt += "---\n"
        prompt += "The following is a mobile use task you have done.\n\n"
        prompt += "### User Instruction ###\n"
        prompt += f"{episodedata.goal}\n\n"

        if hasattr(last_step, "plan") and last_step.plan is not None:
            prompt += "### Overall Plan ###\n"
            prompt += f"{last_step.plan}\n\n"

        prompt += "### History operations ###\n"
        prompt += "To complete the requirements of user\'s instruction, you have performed a series of operations. These operations are as follow:\n"
        history = get_history(trajectory)
        prompt += "\n".join(history)
        if hasattr(last_step, "answer") and last_step.answer is not None:
            prompt += f"\nFinal answer: {last_step.answer}"
        prompt += "\n\n"

        if hasattr(last_step, "progress") and last_step.progress is not None:
            prompt += "### Progress ###\n"
            prompt += "After completing the history operations, you have the following thoughts about the progress of user\'s instruction completion:\n"
            prompt += f"Completed contents:\n{last_step.progress}\n\n"
        
        prompt += "Evaluation Result\n"
        prompt += "You have evaluated the task and the result is:\n"
        prompt += f"{result}\n\n"

        prompt += "---\n"
        prompt += "Carefully examine the information of the finished task to summarize it.\n"
        prompt += "Your summarized information will be referred to by another agent when performing new tasks.\n"
        prompt += "You should follow the below instructions:\n"
        prompt += "1. If the task is successfully executed, you should summarize the successful plan based on the whole trajectory to finish the task.\n"
        prompt += "2. Otherwise, provide the reasons why the task is failed and potential suggestions that may avoid this failure.\n"
        prompt += "\n"

        prompt += "### Attention ###\n"
        prompt += " 1. Only extract the correct plan and do not provide redundant steps.\n"
        prompt += " 2. The suggestions are for another agent not human, so they must be doable through the agent's action.\n"
        prompt += "\n"

        prompt += "Provide your output in the following format:\n\n"
        prompt += "### Summary ###\n"
        prompt += "Summary of the finished task.\n"

        messages.append({
            "role": "user",
            "content": [{"type": "text","text": prompt}]
        })
        return messages

    def parse_response(self, response: str) -> str:
        summary = response.split("### Summary ###")[-1].strip()
        return summary



class ExperienceExtractor(SubAgent):
    def get_message(self, current_goal: str, finished_goal: str, summary: str) -> list:
        messages = []

        # Add system prompt
        messages.append({
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a helpful AI assistant for operating mobile phones. Your goal is to provide useful information as requested, to help another agent follow the instruction and perform the mobile use task."
                }
            ]
        })

        # Add user prompt
        prompt = ""
        prompt += "### Current Task's User Instruction ###\n"
        prompt += f"{current_goal}\n\n"

        prompt += "### Retrieved similar task experience ###\n"
        prompt += "This is a similar task you have done.\n"
        prompt += f"User Instruction: {finished_goal}\n"
        prompt += f"Experience: {summary}\n\n"

        prompt += "---\n"
        prompt += "Based on the retrieved similar task experience, if you think it is indeed useful to the current task, provide the final knowledge in a numbered list. "
        prompt += "Your output will be referred to by another agent when performing the new task.\n"

        prompt += "Provide your output in the following format:\n\n"
        prompt += "### Knowledge ###\n"
        prompt += "1. ...\n"

        messages.append({
            "role": "user",
            "content": [{"type": "text","text": prompt}]
        })
        return messages

    def parse_response(self, response: str) -> str:
        knowledge = response.split("### Knowledge ###")[-1].strip()
        return knowledge


class Evolutor(SubAgent):
    def get_message(self, episodedata: EpisodeData) -> list:
        messages = []
        trajectory = episodedata.trajectory
        last_step = trajectory[-1]

        # Add system prompt
        messages.append({
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a helpful AI assistant specializing in mobile phone operations. Your goal is to reflect on past experiences and provide insights to improve future interactions."
                }
            ]
        })

        prompt = "### Current Task ###\n"
        prompt += f"{episodedata.goal}\n\n"

        if hasattr(last_step, "plan") and last_step.plan is not None:
            prompt += "### Overall Plan ###\n"
            prompt += f"{last_step.plan}\n\n"

        if hasattr(last_step, "progress") and last_step.progress is not None:
            prompt += "### Progress ###\n"
            prompt += f"{last_step.progress}\n\n"
    
        prompt += "### Existing Tips from Past Experience ###\n"
        if episodedata.input_tips is not None:
            prompt += f"{episodedata.input_tips}\n\n"
        else:
            prompt += "No tips recorded.\n\n"

        prompt += "### Full Action History ###\n"

        if len(trajectory) > 0:
            history = get_history(trajectory)
            prompt += "\n".join(history)
            if hasattr(last_step, "answer") and last_step.answer is not None:
                prompt += f"\nFinal answer: {last_step.answer}"
            prompt += "\n\n"
        else:
            prompt += "No actions have been taken yet.\n\n"
            
        # if len(info_pool.future_tasks) > 0:
        #     prompt += "---\n"
        #     # if the setting provides future tasks explicitly
        #     prompt += "### Future Tasks ###\n"
        #     prompt += "Here are some tasks that you might be asked to do in the future:\n"
        #     for task in info_pool.future_tasks:
        #         prompt += f"- {task}\n"
        #     prompt += "\n"

        prompt += "---\n"
        prompt += "Carefully reflect on the interaction history of the current task. Check if there are any general tips that might be useful for handling future tasks, such as advice on preventing certain common errors?\n\n"

        prompt += "Provide your output in the following format:\n\n"

        prompt += "### Updated Tips ###\n"
        prompt += "If you have any important new tips to add (not already included in the existing tips), combine them with the current list. If there are no new tips, simply copy the existing tips here. Keep your tips concise and general.\n"
        # prompt += "If there are more than 10 tips, keep the most important 10 ones.\n"

        messages.append({
            "role": "user",
            "content": [{"type": "text","text": prompt}]
        })

        return messages

    def parse_response(self, response: str) -> dict:
        updated_tips = response.split("### Updated Tips ###")[-1].strip()
        return updated_tips


class UITARSOperator(SubAgent):
    def __init__(self, enable_multi_model=False, num_latest_screenshot: int = 5):
        super().__init__()
        self.enable_multi_model = enable_multi_model
        self.num_latest_screenshot = num_latest_screenshot

        self.system_prompt = r"""You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. 

## Output Format
```\nThought: ...
Action: ...\n```

## Action Space
click(start_box='<|box_start|>(x1,y1)<|box_end|>')
long_press(start_box='<|box_start|>(x1,y1)<|box_end|>', time='')
type(content='')
scroll(start_box='<|box_start|>(x1,y1)<|box_end|>', end_box='<|box_start|>(x3,y3)<|box_end|>')
press_home()
press_back()
finished(content='') # Submit the task regardless of whether it succeeds or fails.

## Note
- Use English in `Thought` part.

- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.

## User Instruction
"""

        self.ACTION_SPACE = ['click', 'long_press', 'type', 'scroll', 'press_home', 'press_back', 'finished', 'answer']

        self.messages = []

    def get_message(self, episodedata: EpisodeData) -> list:
        trajectory = episodedata.trajectory
        current_step = trajectory[-1]
        
        pixels = current_step.curr_env_state.pixels.copy()

        if len(trajectory) == 1:
            self.messages.append({
                "role": "user",
                "content": [{"type": "text","text": self.system_prompt + f"{episodedata.goal}\n"}]
            })
        else:
            previous_step = trajectory[-2]
            self.messages.append({
                "role": "assistant",
                "content": [
                    {"type": "text","text": f"Thought: {previous_step.thought}\nAction: {previous_step.action_s}\n"},
                    # {"type": "text","text": f"Action: {previous_step.action_s}\n"},
                ]
            })

        if self.enable_multi_model:
            prompt = ""
            if hasattr(current_step, "plan") and current_step.plan is not None:
                prompt += "### Overall Plan ###\n"
                prompt += f"{current_step.plan}\n\n"

            if hasattr(current_step, "sub_goal") and current_step.sub_goal is not None:
                prompt += "### Current Subgoal ###\n"
                prompt += f"{current_step.sub_goal}\n\n"

            if len(trajectory) > 1:
                previous_step = trajectory[-2]
                if hasattr(previous_step, "progress") and previous_step.progress is not None:
                    prompt += "### Progress ###\n"
                    prompt += "After completing the history operations, you have the following thoughts about the progress of user\'s instruction completion:\n"
                    prompt += f"Completed contents:\n{previous_step.progress}\n\n"

                if hasattr(previous_step, "memory") and previous_step.memory is not None:
                    prompt += "### Memory ###\n"
                    prompt += "During the operations, you record the following contents on the screenshot for use in subsequent operations:\n"
                    prompt += f"{previous_step.memory}\n\n"

                if hasattr(previous_step, "reflection_outcome") and previous_step.reflection_outcome is not None and previous_step.reflection_outcome in ['B', 'C']:
                    prompt += "### Latest operation ###\n"
                    prompt += f"You previously wanted to perform the operation \"{previous_step.action_desc}\" on this page and executed the Action \"{previous_step.action_s}\". But you find that this operation does not meet your expectation.\nFeedback:{previous_step.reflection_error}\n You need to reflect and revise your operation this time."
                    prompt += "\n\n"

            if hasattr(episodedata, "input_tips") and episodedata.input_tips is not None:
                prompt += "### Tips ###\n"
                prompt += "From previous experience interacting with the device, you have collected the following tips that might be useful for deciding what to do next:\n"
                prompt += f"{episodedata.input_tips}\n\n"

        self.messages.append({ "role": "user", "content": [{"type": "text","text": IMAGE_PLACEHOLDER}]})
        self.messages.append({ "role": "user", "content": [{"type": "image_url","image_url": {"url": encode_image_url(pixels)}}]})

        messages = remove_img_placeholder(self.messages, num_latest_screenshot=self.num_latest_screenshot)

        if self.enable_multi_model:
            messages.insert(-1, { "role": "user", "content": [{"type": "text","text": prompt}]})

        return messages
    
    def parse_response(self, content: str, size: tuple[float, float], raw_size: tuple[float, float]):
        thought = re.search(r"Thought:(.*?)(?=\n|Action:)", content, flags=re.DOTALL)
        if thought:
            thought_s = thought.group(1).strip()
        else:
            thought_s = None

        action = re.search(r'Action:(.*)', content, flags=re.DOTALL)
        # action = re.search(r'Action:(.*?)(?=\n|Thought|Action)', content, flags=re.DOTALL)
        action_s = action.group(1).strip()
        name = re.search(r'([a-z_]+)', action_s.lower()).group(1).strip()

        params = {}
        action_params = re.findall(r"(\w+)='([^']*)'", action_s)
        if len(action_params):
            for k, v in action_params:
                try:
                    x, y = eval(v)
                    x = round(x / 1000 * raw_size[0])
                    y = round(y / 1000 * raw_size[1])
                    params[k] = (x, y)
                except:
                    params[k] = v.strip()
        
        if name not in self.ACTION_SPACE:
            raise Exception(f"Action {name} is not in the action space.")
        if name in ['click', 'long_press', 'scroll']:
            if 'start_box' not in params:
                raise Exception(f"Action {name} requires 'start_box' parameter.")
        if name == 'scroll' and 'end_box' not in params:
            raise Exception(f"Action {name} requires 'end_box' parameter.")
        if name == 'type' and 'content' not in params:
            raise Exception(f"Action {name} requires 'content' parameter.")

        action_a = Action(name=name, parameters=params)
        return thought_s, action_a, action_s
