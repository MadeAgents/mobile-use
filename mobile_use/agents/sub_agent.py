from abc import ABC, abstractmethod
import re
import json

from mobile_use.scheme import *
from mobile_use.utils import encode_image_url, smart_resize

# Fix Picture sequence inconsistency problem in vllm0.7.2 
# If you are using QwenAPI from 'dashscope.aliyuncs.com', replace IMAGE_PLACEHOLDER with ''
IMAGE_PLACEHOLDER = '<|vision_start|><|image_pad|><|vision_end|>'

ACTION_SPACE = ["key", "click", "left_click", "long_press", "swipe", "scroll", "type", "answer", "system_button", "open", "wait", "call_user", "terminate"]


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
                {"type": "image_url","image_url": {"url": encode_image_url(pixels)}}
            ]
        })

        return messages

    def parse_response(self, response: str):
        thought = response.split("### Thought ###")[-1].split("### Plan ###")[0].replace("\n", " ").replace("  ", " ").strip()
        plan = response.split("### Plan ###")[-1].split("### Current Subgoal ###")[0].replace("\n", " ").replace("  ", " ").strip()
        current_subgoal = response.split("### Current Subgoal ###")[-1].replace("\n", " ").replace("  ", " ").strip()
        return thought, plan, current_subgoal


# class Operator(SubAgent):
#     def __init__(self, num_histories: int = None):
#         super().__init__()
#         self.num_histories = num_histories

#     def get_message(self, episodedata: EpisodeData) -> list:
#         messages = []
#         trajectory = episodedata.trajectory
#         current_step = trajectory[-1]
        
#         pixels = current_step.curr_env_state.pixels.copy()
#         resized_height, resized_width = smart_resize(height=pixels.height, width=pixels.width)

#         # Add system prompt
#         messages.append({
#             "role": "system",
#             "content": [
#                 {
#                     "type": "text",
#                     "text": f"""
# You are a helpful AI assistant for operating mobile phones. Your goal is to choose the correct actions to complete the user's instruction. Think as if you are a human user operating the phone.

# # Tools

# You may call one or more functions to assist with the user query.

# You are provided with function signatures within <tools></tools> XML tags:
# <tools>
# {{"type": "function", "function": {{"name_for_human": "mobile_use", "name": "mobile_use", "description": "Use a touchscreen to interact with a mobile device, and take screenshots.
# * This is an interface to a mobile device with touchscreen. You can perform actions like clicking, typing, swiping, etc.
# * Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.
# * The screen's resolution is {resized_width}x{resized_height}.
# * Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.", "parameters": {{"properties": {{"action": {{"description": "The action to perform. The available actions are:
# * `key`: Perform a key event on the mobile device.
#     - This supports adb's `keyevent` syntax.
#     - Examples: "volume_up", "volume_down", "power", "camera", "clear".
# * `click`: Click the point on the screen with coordinate (x, y).
# * `long_press`: Press the point on the screen with coordinate (x, y) for specified seconds.
# * `swipe`: Swipe from the starting point with coordinate (x, y) to the end point with coordinates2 (x2, y2).
# * `type`: Input the specified text into the activated input box.
# * `system_button`: Press the system button.
# * `open`: Open an app on the device.
# * `wait`: Wait specified seconds for the change to happen.
# * `call_user`: Call user when you need help or need additional information from the user.
# * `terminate`: Terminate the current task and report its completion status.", "enum": ["key", "click", "long_press", "swipe", "type", "system_button", "open", "wait", "call_user", "terminate"], "type": "string"}}, "coordinate": {{"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=click`, `action=long_press`, `action=wait`, and `action=swipe`.", "type": "array"}}, "coordinate2": {{"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=swipe`.", "type": "array"}}, "text": {{"description": "Required only by `action=key`, `action=type`, `action=call_user`, and `action=open`.", "type": "string"}}, "time": {{"description": "The seconds to wait. Required only by `action=long_press` and `action=wait`.", "type": "number"}}, "button": {{"description": "Back means returning to the previous interface, Home means returning to the desktop, Menu means opening the application background menu, and Enter means pressing the enter. Required only by `action=system_button`", "enum": ["Back", "Home", "Menu", "Enter"], "type": "string"}}, "status": {{"description": "The status of the task. Required only by `action=terminate`.", "type": "string", "enum": ["success", "failure"]}}}}, "required": ["action"], "type": "object"}}, "args_format": "Format the arguments as a JSON object."}}}}
# </tools>
# """
#                 }
#             ]
#         })

#         # Add user prompt
#         prompt = "### User Instruction ###\n"
#         prompt += f"{episodedata.goal}\n\n"

#         if hasattr(current_step, "plan") and current_step.plan is not None:
#             prompt += "### Overall Plan ###\n"
#             prompt += f"{current_step.plan}\n\n"

#         # if hasattr(current_step, "progress_status") and current_step.progress_status is not None:
#         #     prompt += "### Progress Status ###\n"
#         #     prompt += f"{current_step.progress_status}\n\n"

#         if hasattr(current_step, "sub_goal") and current_step.sub_goal is not None:
#             prompt += "### Current Subgoal ###\n"
#             prompt += f"{current_step.sub_goal}\n\n"

#         prompt += "### Latest History Operations ###\n"
#         prompt += "You have done the following operation on the current device):\n"
#         if len(trajectory) > 1 and (self.num_histories is None or self.num_histories > 0):
#             start_idx = 0 if self.num_histories is None else max(0, len(trajectory) - 1 - self.num_histories)
#             for i in range(start_idx, len(trajectory) - 1):
#                 step_list = []
#                 step_list.append(f"Action: {trajectory[i].action_desc}")
#                 step_list.append(f"<tool_call> {trajectory[i].action_s} </tool_call>")
#                 if hasattr(trajectory[i], "summary") and trajectory[i].summary is not None:
#                     step_list.append(f"Summary: {trajectory[i].summary}")
#                 # if hasattr(trajectory[i], "reflection_outcome") and trajectory[i].reflection_outcome is not None:
#                 #     if trajectory[i].reflection_outcome == "A":
#                 #         step_list.append("Successful")
#                 #     else:
#                 #         step_list.append("Failed")
#                 #         step_list.append(f"Feedback: {trajectory[i].reflection_error}")
#                 prompt += f"Step-{i+1}: {'; '.join(step_list)}\n"
#             prompt += "\n"
#         else:
#             prompt += "No actions have been taken yet.\n\n"

#         if len(trajectory) > 1:
#             previous_step = trajectory[-2]
#             if hasattr(previous_step, "progress") and previous_step.progress is not None:
#                 prompt += "### Progress ###\n"
#                 prompt += "After completing the history operations, you have the following thoughts about the progress of user\'s instruction completion:\n"
#                 prompt += f"Completed contents:\n{previous_step.progress}\n\n"

#             if hasattr(previous_step, "memory") and previous_step.memory is not None:
#                 prompt += "### Memory ###\n"
#                 prompt += "During the operations, you record the following contents on the screenshot for use in subsequent operations:\n"
#                 prompt += f"{previous_step.memory}\n\n"

#         if hasattr(current_step, "reflection_outcome") and current_step.reflection_outcome is not None and current_step.reflection_outcome != "A":
#             prompt += "### Latest operation ###\n"
#             prompt += f"You previously wanted to perform the operation \"{current_step.action_desc}\" on this page and executed the Action \"{current_step.action_s}\". But you find that this operation does not meet your expectation.\nFeedback:{current_step.reflection_error}\n You need to reflect and revise your operation this time."
#             prompt += "\n\n"

#         prompt += "### Observation ###\n"
#         prompt += f"This is the current screenshot of the phone. The screen's resolution is {resized_width}x{resized_height}."
#         prompt += f"{IMAGE_PLACEHOLDER}\n\n"

#         prompt += "### Guidance ###\n"
#         prompt += "Here are some useful guidelines you need to follow:\n"
#         if len(trajectory) == 1:
#             prompt += "- The app is opening, you don't need to open it. Wait if it is loading. \n"
#             prompt += """
# "- If it shows an advertisement, Use the action wait, do not use the action click!!! If it can be closed, add the coordinate of the close button in wait action: 
# <tool_call>
# {{"name": "mobile_use", "arguments": {"action": "wait", "time": 2, "coordinate": [x, y]}}}
# </tool_call>
# """
#         prompt += """- If the task is finished, you should terminate the task in time!
# - If you stuck in an action, you should try to change the action or the correspoinding parameters. Do not always repeat the same action!
# - If you want to type in some text, remember to click the input field first.
# - When there is ad, countdown, or progress bar on the screen, you should wait until the ad, countdown, or progress bar disappears.

# """

#         prompt += "### Response Requirements ###\n"
#         prompt += """First, think about the requirements that have been completed in previous operations and the requirements that need to be completed in the next one operation. Put your thinking process in one sentence in `Thought` part.
# Secend, provide a brief description of the chosen action and the expected outcome in `Action` part.
# Last, execute an action in the form of function. For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:

# ### Format ###
# Thought: ... (Your thinking process)
# Action: ... (Your action description)
# <tool_call>
# {{"name": <function-name>, "arguments": <args-json-object>}}
# </tool_call>"""

#         messages.append({
#             "role": "user",
#             "content": [
#                 {"type": "text","text": prompt},
#                 {"type": "image_url","image_url": {"url": encode_image_url(pixels)}}
#             ]
#         })

#         return messages
    
#     def parse_response(self, content: str, size: tuple[float, float], raw_size: tuple[float, float]):
#         thought = re.search(r"Thought:(.*?)(?=\n|Action:|<tool_call>|\{\"name\": \"mobile_use\",)", content, flags=re.DOTALL)
#         if thought:
#             thought_s = thought.group(1).strip()
#         else:
#             thought_s = None
#         action_desc = re.search(r"Action:(.*?)(?=\n|<tool_call>|\{\"name\": \"mobile_use\",)", content, flags=re.DOTALL)
#         if action_desc:
#             action_desc_s = action_desc.group(1).strip()
#         else:
#             action_desc_s = None
#         action = re.search(r'{"name": "mobile_use",(.*?)}}', content, flags=re.DOTALL)
#         if not action:
#             raise Exception("Cannot extract action in the content.")
#         action_s = '{"name": "mobile_use",' + action.group(1).strip() + '}}'
#         action = json.loads(action_s)
#         name = action['arguments']['action']
#         if name not in ACTION_SPACE:
#             raise Exception(f"Action {name} is not in the action space.")
#         action['arguments'].pop('action')
#         params = action['arguments']

#         for k, v in params.items():
#             if k in ['coordinate', 'coordinate2', 'point', 'start_point', 'end_point']:
#                 try:
#                     x = round(v[0] / size[0] * raw_size[0])
#                     y = round(v[1] / size[1] * raw_size[1])
#                     params[k] = (x, y)
#                 except:
#                     pass
#         action_a = Action(name=name, parameters=params)

#         return thought_s, action_a, action_s, action_desc_s


class Operator(SubAgent):
    def __init__(self, num_histories: int = None):
        super().__init__()
        self.num_histories = num_histories

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
* `system_button`: Press the system button.
* `open`: Open an app on the device.
* `wait`: Wait specified seconds for the change to happen.
* `call_user`: Call user when you need help or need additional information from the user.
* `terminate`: Terminate the current task and report its completion status.", "enum": ["key", "click", "long_press", "swipe", "type", "system_button", "open", "wait", "call_user", "terminate"], "type": "string"}}, "coordinate": {{"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=click`, `action=long_press`, `action=wait`, and `action=swipe`.", "type": "array"}}, "coordinate2": {{"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=swipe`.", "type": "array"}}, "text": {{"description": "Required only by `action=key`, `action=type`, `action=call_user`, and `action=open`.", "type": "string"}}, "time": {{"description": "The seconds to wait. Required only by `action=long_press` and `action=wait`.", "type": "number"}}, "button": {{"description": "Back means returning to the previous interface, Home means returning to the desktop, Menu means opening the application background menu, and Enter means pressing the enter. Required only by `action=system_button`", "enum": ["Back", "Home", "Menu", "Enter"], "type": "string"}}, "status": {{"description": "The status of the task. Required only by `action=terminate`.", "type": "string", "enum": ["success", "failure"]}}}}, "required": ["action"], "type": "object"}}, "args_format": "Format the arguments as a JSON object."}}}}
</tools>
"""
                }
            ]
        })

        # Add user prompt
        prompt = "### User Instruction ###\n"
        prompt += f"{episodedata.goal}\n\n"

        if hasattr(current_step, "plan") and current_step.plan is not None:
            prompt += "### Overall Plan ###\n"
            prompt += f"{current_step.plan}\n\n"

        # if hasattr(current_step, "progress_status") and current_step.progress_status is not None:
        #     prompt += "### Progress Status ###\n"
        #     prompt += f"{current_step.progress_status}\n\n"

        if hasattr(current_step, "sub_goal") and current_step.sub_goal is not None:
            prompt += "### Current Subgoal ###\n"
            prompt += f"{current_step.sub_goal}\n\n"

        prompt += "### Latest History Operations ###\n"
        prompt += "You have done the following operation on the current device):\n"
        if len(trajectory) > 1 and (self.num_histories is None or self.num_histories > 0):
            start_idx = 0 if self.num_histories is None else max(0, len(trajectory) - 1 - self.num_histories)
            for i in range(start_idx, len(trajectory) - 1):
                step_list = []
                step_list.append(f"Action: {trajectory[i].action_desc}")
                step_list.append(f"<tool_call> {trajectory[i].action_s} </tool_call>")
                if hasattr(trajectory[i], "summary") and trajectory[i].summary is not None:
                    step_list.append(f"Summary: {trajectory[i].summary}")
                if hasattr(trajectory[i], "reflection_outcome") and trajectory[i].reflection_outcome is not None:
                    if trajectory[i].reflection_outcome == "A":
                        step_list.append("Successful")
                    else:
                        step_list.append("Failed")
                        step_list.append(f"Feedback: {trajectory[i].reflection_error}")
                prompt += f"Step-{i+1}: {'; '.join(step_list)}\n"
            prompt += "\n"
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

            if hasattr(previous_step, "reflection_outcome") and previous_step.reflection_outcome is not None and previous_step.reflection_outcome != "A":
                prompt += "### Latest operation ###\n"
                prompt += f"You previously wanted to perform the operation \"{previous_step.action_desc}\" on this page and executed the Action \"{previous_step.action_s}\". But you find that this operation does not meet your expectation.\nFeedback:{previous_step.reflection_error}\n You need to reflect and revise your operation this time."
                prompt += "\n\n"

        prompt += "### Observation ###\n"
        prompt += f"This is the current screenshot of the phone. The screen's resolution is {resized_width}x{resized_height}."
        prompt += f"{IMAGE_PLACEHOLDER}\n\n"

        prompt += "### Guidance ###\n"
        prompt += "Here are some useful guidelines you need to follow:\n"
        if len(trajectory) == 1:
            prompt += "- The app is opening, you don't need to open it. Wait if it is loading. \n"
        prompt += """- If you want to type in some text, remember to click the input field first.
- If it shows an advertisement, you must use the action `wait`. Do not click!!!
- If it requests any permissions, you must use the action `wait`. Do not click!!!
- If the screenshot shows any terms and conditions, you must use the action `wait`. Do not click!!!
- If the screenshot shows any popups, you must use the action `wait`. Do not click!!!
- If the screenshot asks user preference or profile, you must use the action `wait`. Do not click!!!
- If the screenshot shows any other information that you think requires user attention, you must use the action `wait`. Do not click!!!

"""

#         if len(trajectory) == 1:
#             prompt += "- The app is opening, you don't need to open it. Wait if it is loading. \n"
#             prompt += """
# "- If it shows an advertisement, Use the action wait, do not use the action click!!! If it can be closed, add the coordinate of the close button in wait action: 
# <tool_call>
# {{"name": "mobile_use", "arguments": {"action": "wait", "time": 2, "coordinate": [x, y]}}}
# </tool_call>
# """
#         prompt += """- If the task is finished, you should terminate the task in time!
# - If you stuck in an action, you should try to change the action or the correspoinding parameters. Do not always repeat the same action!
# - If you want to type in some text, remember to click the input field first.
# - When there is ad, countdown, or progress bar on the screen, you should wait until the ad, countdown, or progress bar disappears.

# """

        prompt += "### Response Requirements ###\n"
        prompt += """First, think about the requirements that have been completed in previous operations and the requirements that need to be completed in the next one operation. Put your thinking process in one sentence in `Thought` part.
Secend, provide a brief description of the chosen action and the expected outcome in `Action` part.
Last, execute an action in the form of function. For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:

### Format ###
Thought: ... (Your thinking process)
Action: ... (Your action description)
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>"""

        messages.append({
            "role": "user",
            "content": [
                {"type": "text","text": prompt},
                {"type": "image_url","image_url": {"url": encode_image_url(pixels)}}
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


class ReflectorBeforeExecution(SubAgent):
    def get_message(self, episodedata: EpisodeData) -> list:
        messages = []
        trajectory = episodedata.trajectory
        current_step = trajectory[-1]

        pixels_before = current_step.curr_env_state.pixels.copy()
        resized_height, resized_width = smart_resize(height=pixels_before.height, width=pixels_before.width)
        
        # Add system prompt
        messages.append({
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a helpful AI assistant for operating mobile phones. Your goal is to verify whether the latest action is reasonable."
                }
            ]
        })

        # Add user prompt
        prompt = "### User Instruction ###\n"
        prompt += f"{episodedata.goal}\n\n"

        if hasattr(current_step, "sub_goal") and current_step.sub_goal is not None:
            prompt += "### Current Subgoal ###\n"
            prompt += f"{current_step.sub_goal}\n\n"

        prompt += "### History Operations ###\n"
        prompt += "You have done the following operation on the current device):\n"
        if len(trajectory) > 1:
            for i in range(len(trajectory) - 1):
                step_list = []
                step_list.append(f"Action: {trajectory[i].action_desc}")
                step_list.append(f"<tool_call> {trajectory[i].action_s} </tool_call>")
                if hasattr(trajectory[i], "summary") and trajectory[i].summary is not None:
                    step_list.append(f"Summary: {trajectory[i].summary}")
                # if hasattr(trajectory[i], "reflection_outcome") and trajectory[i].reflection_outcome is not None:
                #     if trajectory[i].reflection_outcome == "A":
                #         step_list.append("Successful")
                #     else:
                #         step_list.append("Failed")
                #         step_list.append(f"Feedback: {trajectory[i].reflection_error}")
                prompt += f"Step-{i+1}: {'; '.join(step_list)}\n"
            prompt += "\n"
        else:
            prompt += "No actions have been taken yet.\n\n"

        prompt += "### Guidance ###\n"
        prompt += "Here are some useful guidelines you need to follow:\n"
        if len(trajectory) == 1:
            prompt += "- The app is opening, you don't need to open it. Wait if it is loading. \n"
            prompt += """
"- If it shows an advertisement, Use the action wait, do not use the action click!!! If it can be closed, add the coordinate of the close button in wait action: 
<tool_call>
{{"name": "mobile_use", "arguments": {"action": "wait", "time": <time>, "coordinate": [x, y]}}}
</tool_call>
"""
        prompt += """- If the task is finished, you should terminate the task in time!
- If you stuck in an action, you should try to change the action or the correspoinding parameters. Do not always repeat the same action!
- If you want to type in some text, remember to click the input field first.
- When there is ad, countdown, or progress bar on the screen, you should wait until the ad, countdown, or progress bar disappears.

"""

        prompt += "---\n"
        prompt += f"{IMAGE_PLACEHOLDER}\n"
        prompt += f"This is the screenshot before executing the latest action." 
        prompt += f"The width and height are {resized_width} and {resized_height} pixels, respectively.\n\n"

        prompt += "---\n"
        prompt += "### Latest Action ###\n"
        prompt += f"Action Thought: {current_step.thought}\n"
        prompt += f"Action Description: {current_step.action_desc}\n"
        prompt += f"Action: {current_step.action_s}\n\n"

        prompt += "---\n"
        prompt += "Carefully examine the information provided above to determine whether the last action is reasonable and can produce the expected behavior. If you think the action is incorrect, provide your reason.\n\n"

        prompt += "Provide your output in the following format containing three parts:\n\n"
        prompt += "### Outcome ###\n"
        prompt += "Choose from the following options. Give your answer as \"A\" or \"B\":\n"
        prompt += "A: Correct. The last action is reasonable.\n"
        prompt += "B: Wrong. The last action will results in a wrong page or produce no changes. It can't meet the goal.\n"

        prompt += "### Error Description ###\n"
        prompt += "If the action is wrong, provide a detailed description of the error and the potential reason causing the failure. If the action is correct, put \"None\" here.\n\n"

        messages.append({
            "role": "user",
            "content": [
                {"type": "text","text": prompt},
                {"type": "image_url","image_url": {"url": encode_image_url(pixels_before)}},
            ]
        })

        return messages

    def parse_response(self, response: str) -> dict:
        outcome = response.split("### Outcome ###")[-1].split("### Error Description ###")[0].replace("\n", " ").replace("  ", " ").strip()
        error_description = response.split("### Error Description ###")[-1].replace("\n", " ").replace("  ", " ").strip()
        return outcome, error_description


"""
Call after executing each action.
"""
class Reflector(SubAgent):
    def get_message(self, episodedata: EpisodeData) -> list:
        messages = []
        trajectory = episodedata.trajectory
        current_step = trajectory[-1]

        pixels_before = current_step.curr_env_state.pixels.copy()
        resized_height, resized_width = smart_resize(height=pixels_before.height, width=pixels_before.width)
        pixels_after = current_step.exec_env_state.pixels.copy()
        
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

        if hasattr(current_step, "sub_goal") and current_step.sub_goal is not None:
            prompt += "### Current Subgoal ###\n"
            prompt += f"{current_step.sub_goal}\n\n"

        prompt += "---\n"
        prompt += f"Screenshot before latest action: {IMAGE_PLACEHOLDER}\n"
        prompt += f"Screenshot after latest action: {IMAGE_PLACEHOLDER}\n"
        prompt += f"The two images are two phone screenshots before and after your latest action. " 
        prompt += f"The width and height are {resized_width} and {resized_height} pixels, respectively.\n\n"

        prompt += "---\n"
        prompt += "### Latest Action ###\n"
        prompt += f"Action: {current_step.action_s}\n"
        prompt += f"Expectation: {current_step.action_desc}\n\n"

        prompt += "---\n"
        prompt += "Carefully examine the information provided above to determine whether the last action produced the expected behavior. If the action failed, identify the failure mode and provide reasoning on the potential reason causing this failure. Note that for the “Swipe” action, it may take multiple attempts to display the expected content. Thus, for a \"Swipe\" action, if the screen shows new content, it usually meets the expectation.\n\n"

        prompt += "Provide your output in the following format containing three parts:\n\n"
        prompt += "### Outcome ###\n"
        prompt += "Choose from the following options. Give your answer as \"A\", \"B\" or \"C\":\n"
        prompt += "A: Successful or Partially Successful. The result of the last action meets the expectation.\n"
        prompt += "B: Failed. The last action results in a wrong page. I need to return to the previous state.\n"
        prompt += "C: Failed. The last action produces no changes.\n\n"

        prompt += "### Error Description ###\n"
        prompt += "If the action failed, provide a detailed description of the error and the potential reason causing this failure. If the action succeeded, put \"None\" here.\n\n"

        messages.append({
            "role": "user",
            "content": [
                {"type": "text","text": prompt},
                {"type": "image_url","image_url": {"url": encode_image_url(pixels_before)}},
                {"type": "image_url","image_url": {"url": encode_image_url(pixels_after)}}
            ]
        })

        return messages

    def parse_response(self, response: str) -> dict:
        outcome = response.split("### Outcome ###")[-1].split("### Error Description ###")[0].replace("\n", " ").replace("  ", " ").strip()
        error_description = response.split("### Error Description ###")[-1].replace("\n", " ").replace("  ", " ").strip()
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
                {"type": "image_url","image_url": {"url": encode_image_url(pixels)}}
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
            for i in range(len(trajectory)):
                step_list = []
                step_list.append(f"Action: {trajectory[i].action_desc}")
                step_list.append(f"<tool_call> {trajectory[i].action_s} </tool_call>")
                if hasattr(trajectory[i], "summary") and trajectory[i].summary is not None:
                    step_list.append(f"Summary: {trajectory[i].summary}")
                if hasattr(trajectory[i], "reflection_outcome") and trajectory[i].reflection_outcome is not None:
                    if trajectory[i].reflection_outcome == "A":
                        step_list.append("Successful")
                    else:
                        step_list.append("Failed")
                        step_list.append(f"Feedback: {trajectory[i].reflection_error}")
                prompt += f"Step-{i+1}: {'; '.join(step_list)}\n"
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
    def __init__(self, num_latest_screenshots: int = 0):
        super().__init__()
        self.num_latest_screenshots = num_latest_screenshots

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
                    "text": "You are a helpful AI assistant for operating mobile phones. Your goal is to evaluate whether the user's instruction has been completed."
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
        if len(trajectory) > 0:
            for i in range(len(trajectory)):
                step_list = []
                step_list.append(f"Action: {trajectory[i].action_desc}")
                step_list.append(f"<tool_call> {trajectory[i].action_s} </tool_call>")
                if hasattr(trajectory[i], "summary") and trajectory[i].summary is not None:
                    step_list.append(f"Summary: {trajectory[i].summary}")
                if hasattr(trajectory[i], "reflection_outcome") and trajectory[i].reflection_outcome is not None:
                    if trajectory[i].reflection_outcome == "A":
                        step_list.append("Successful")
                    else:
                        step_list.append("Failed")
                        step_list.append(f"Feedback: {trajectory[i].reflection_error}")
                prompt += f"Step-{i+1}: {'; '.join(step_list)}\n"
            prompt += "\n"
        else:
            prompt += "No actions have been taken yet.\n\n"

        prompt += "### Response requirements ###\n"
        prompt += "Carefully examine the information provided above to determine whether the user's instruction is completed or not. Provide your output in the following format:\n\n"

        prompt += "### Result ###\n"
        prompt += "Give your answer as \"Finished\" or \"Not Finished\".\n"

        messages.append({"role": "user","content": [{"type": "text","text": prompt}]})

        return messages
    
    def parse_response(self, response: str):
        result = response.split("### Result ###")[-1].strip()
        return result

