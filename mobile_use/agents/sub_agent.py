from abc import ABC, abstractmethod
import re
import json
import logging
from typing import Union, Literal

from mobile_use.default_prompts.prompt_type import *
from mobile_use.utils.vlm import VLMWrapper
from mobile_use.schema.schema import *
from mobile_use.utils.constants import IMAGE_PLACEHOLDER
from mobile_use.utils.utils import *
from mobile_use.schema.config import *

# __all__ = ['Planner', 'Operator', 'Reflector', 'LongReflector', 'NoteTaker', 'Processor', 'Evaluator', 'TaskSummarizer', 'ExperienceExtractor', 'Evolutor', 'UITARSOperator']

logger = logging.getLogger(__name__)
# import logging

# logging.basicConfig(level=logging.INFO, format='%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s')
# logger = logging.getLogger()

ACTION_SPACE = ["key", "click", "left_click", "long_press", "swipe", "scroll", "type", "clear_text", "answer", "system_button", "open", "wait", "terminate", "take_note"]


def get_history(trajectory: List[MobileUseStepData], num_histories=None):
    start_idx = 0 if num_histories is None else max(0, len(trajectory) - num_histories)
    history = []
    for i in range(start_idx, len(trajectory)):
        step_list = []
        step_list.append(f"Action: {trajectory[i].action_desc}")
        step_list.append(f"<tool_call> {trajectory[i].action_s} </tool_call>")
        if trajectory[i].summary is not None:
            step_list.append(f"Summary: {trajectory[i].summary}")
        if trajectory[i].reflection_outcome is not None:
            if trajectory[i].reflection_outcome == "A":
                step_list.append("Successful")
            elif trajectory[i].reflection_outcome in ["B", "C"]:
                step_list.append("Failed")
                step_list.append(f"Feedback: {trajectory[i].reflection_error}")
        elif trajectory[i].long_reflection_outcome is not None:
            if trajectory[i].long_reflection_outcome == "A":
                step_list.append("Successful")
            elif trajectory[i].long_reflection_outcome in ["B"]:
                step_list.append("Failed")
                step_list.append(f"Feedback: {trajectory[i].long_reflection_error}")
        history.append(f"Step-{i+1}: {'; '.join(step_list)}")
    return history


class SubAgent(ABC):
    def __init__(self, config: SubAgentConfig):
        super().__init__()
        self.vlm = VLMWrapper(**config.vlm.model_dump())

    @abstractmethod
    def get_message(self, episodedata: MobileUseEpisodeData) -> list:
        pass

    @abstractmethod
    def parse_response(self, response: str):
        pass


"""
Call in the beginning of each step.
"""
class Planner(SubAgent):
    def __init__(self, vlm: VLMWrapper, prompt_config = None):
        super().__init__(vlm)
        self.prompt: PlannerPrompt = load_prompt("planner", prompt_config)

    def get_message(self, episodedata: MobileUseEpisodeData) -> list:
        messages = []
        trajectory = episodedata.trajectory
        current_step = trajectory[-1]

        pixels = current_step.curr_env_state.pixels.copy()
        resized_height, resized_width = smart_resize(height=pixels.height, width=pixels.width)
        
        # Add system prompt
        system_message = generate_message("system", self.prompt.system_prompt)
        messages.append(system_message)

        # Add user prompt
        prompt_list = []
        task_prompt = self.prompt.task_prompt.format(
            task_description = episodedata.goal,
            screenshot = IMAGE_PLACEHOLDER,
            resized_width = resized_width,
            resized_height = resized_height,
        )
        prompt_list.append(task_prompt)

        if len(trajectory) == 1:
            init_plan = self.prompt.init_plan
            prompt_list.append(init_plan)
        else:
            previous_step = trajectory[-2]
            continue_plan = self.prompt.continue_plan.format(
                current_plan = previous_step.plan,
                previous_subgoal = previous_step.sub_goal,
            )
            prompt_list.append(continue_plan)

        prompt = "\n\n".join(prompt_list)
        user_message = generate_message("user", prompt, images=[pixels])
        messages.append(user_message)

        return messages

    def parse_response(self, response: str):
        thought = response.split("### Thought ###")[-1].split("### Plan ###")[0].replace("\n", " ").replace("  ", " ").strip()
        plan = response.split("### Plan ###")[-1].split("### Current Subgoal ###")[0].replace("\n", " ").replace("  ", " ").strip()
        current_subgoal = response.split("### Current Subgoal ###")[-1].replace("\n", " ").replace("  ", " ").strip()
        return thought, plan, current_subgoal


class Operator(SubAgent):
    def __init__(
        self, 
        vlm: VLMWrapper, 
        prompt_config = None, 
        num_histories: int = None,
        include_device_time: bool = True,
        include_tips: bool = True,
    ):
        super().__init__(vlm)
        self.prompt: OperatorPrompt = load_prompt("operator", prompt_config)
        self.num_histories = num_histories
        self.include_device_time = include_device_time
        self.include_tips = include_tips

    def get_message(self, episodedata: MobileUseEpisodeData) -> list:
        messages = []
        trajectory = episodedata.trajectory
        current_step = trajectory[-1]
        
        pixels = current_step.curr_env_state.pixels.copy()
        resized_height, resized_width = smart_resize(height=pixels.height, width=pixels.width)
        self.raw_size = (pixels.width, pixels.height)
        self.resized_size = (resized_width, resized_height)

        # Add system prompt
        system_prompt = self.prompt.system_prompt.format(
            resized_width = resized_width,
            resized_height = resized_height,
        )
        system_message = generate_message("system", system_prompt)
        messages.append(system_message)

        # Add user prompt
        prompt_list = []
        task_prompt = self.prompt.task_prompt.format(
            task_description = episodedata.goal,
        )
        prompt_list.append(task_prompt)

        if self.include_device_time:
            device_time = current_step.curr_env_state.device_time
            # # Remove the hour-minute-second and the timezone 
            # device_time = ' '.join(device_time.split()[:3] + device_time.split()[-2:])
            device_time_prompt = self.prompt.device_time_prompt.format(
                device_time = device_time
            )
            prompt_list.append(device_time_prompt)

        if current_step.plan is not None:
            plan_prompt = self.prompt.plan_prompt(
                plan = current_step.plan,
            )
            prompt_list.append(plan_prompt)

        if current_step.sub_goal is not None:
            subgoal_prompt = self.prompt.subgoal_prompt(
                subgoal = current_step.sub_goal,
            )
            prompt_list.append(subgoal_prompt)

        if len(trajectory) > 1 and (self.num_histories is None or self.num_histories > 0):
            history = get_history(trajectory[:-1], self.num_histories)
            history = "\n".join(history)
        else:
            history = "No actions have been taken yet."
        history_prompt = self.prompt.history_prompt.format(
            history = history,
        )
        prompt_list.append(history_prompt)

        if len(trajectory) > 1:
            previous_step = trajectory[-2]
            if previous_step.progress is not None:
                progress_prompt = self.prompt.progress_prompt.format(
                    progress = previous_step.progress,
                )
                prompt_list.append(progress_prompt)

            if previous_step.memory is not None:
                memory_prompt = self.prompt.memory_prompt.format(
                    memory = previous_step.memory,
                )
                prompt_list.append(memory_prompt)

            if previous_step.reflection_outcome is not None and previous_step.reflection_outcome in ['B', 'C']:
                reflection_prompt = self.prompt.reflection_prompt.format(
                    action_desc = previous_step.action_desc,
                    action_s = previous_step.action_s,
                    reflection_error = previous_step.reflection_error,
                )
                prompt_list.append(reflection_prompt)

            if previous_step.long_reflection_outcome is not None and previous_step.long_reflection_outcome in ['B']:
                trajectory_reflection_prompt = self.prompt.trajectory_reflection_prompt.format(
                    trajectory_reflection_error = previous_step.long_reflection_error,
                )
                prompt_list.append(trajectory_reflection_prompt)

            if previous_step.evaluation_result is not None and "Failed" in previous_step.evaluation_result:
                global_reflection_prompt = self.prompt.global_reflection_prompt.format(
                    global_reflection_error = previous_step.evaluation_reason,
                )
                prompt_list.append(global_reflection_prompt)

        observation_prompt = self.prompt.observation_prompt(
            resized_width = resized_width,
            resized_height = resized_height,
            image_placeholder = IMAGE_PLACEHOLDER,
        )
        prompt_list.append(observation_prompt)

        if self.include_tips:
            init_tips = self.prompt.init_tips
            prompt_list.append(init_tips)

        response_prompt = self.prompt.response_prompt
        prompt_list.append(response_prompt)

        prompt = "\n\n".join(prompt_list)
        user_message = generate_message("user", prompt, images=[pixels])
        messages.append(user_message)

        return messages
    
    def parse_response(self, content: str, size: tuple[float, float] = None, raw_size: tuple[float, float] = None):
        if size is None:
            size = self.resized_size
        if raw_size is None:
            raw_size = self.raw_size
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


class AnswerAgent(SubAgent):
    """
    This agent is used to answer the user query.
    It is a special case of the Operator, which only supports the `answer` action.
    """
    def __init__(
        self, 
        vlm: VLMWrapper, 
        prompt_config = None, 
        num_histories: int = None,
        include_device_time: bool = True,
    ):
        super().__init__(vlm)
        self.prompt: AnswerAgentPrompt = load_prompt("answer_agent", prompt_config)
        self.num_histories = num_histories
        self.include_device_time = include_device_time

    def get_message(self, episodedata: MobileUseEpisodeData) -> list:
        messages = []
        trajectory = episodedata.trajectory
        current_step = trajectory[-1]
        
        pixels = current_step.curr_env_state.pixels.copy()
        resized_height, resized_width = smart_resize(height=pixels.height, width=pixels.width)

        # Add system prompt
        system_prompt = self.prompt.system_prompt.format(
            resized_width = resized_width,
            resized_height = resized_height,
        )
        system_message = generate_message("system", system_prompt)
        messages.append(system_message)

        # Add user prompt
        prompt_list = []
        task_prompt = self.prompt.task_prompt.format(
            task_description = episodedata.goal,
        )
        prompt_list.append(task_prompt)

        if self.include_device_time:
            device_time = current_step.curr_env_state.device_time
            # # Remove the hour-minute-second and the timezone 
            # device_time = ' '.join(device_time.split()[:3] + device_time.split()[-2:])
            device_time_prompt = self.prompt.device_time_prompt.format(
                device_time = device_time
            )
            prompt_list.append(device_time_prompt)

        if current_step.plan is not None:
            plan_prompt = self.prompt.plan_prompt(
                plan = current_step.plan,
            )
            prompt_list.append(plan_prompt)

        if current_step.sub_goal is not None:
            subgoal_prompt = self.prompt.subgoal_prompt(
                subgoal = current_step.sub_goal,
            )
            prompt_list.append(subgoal_prompt)

        if len(trajectory) > 1 and (self.num_histories is None or self.num_histories > 0):
            history = get_history(trajectory, self.num_histories)
            history = "\n".join(history)
        else:
            history = "No actions have been taken yet."
        history_prompt = self.prompt.history_prompt.format(
            history = history,
        )
        prompt_list.append(history_prompt)

        if len(trajectory) > 1:
            previous_step = trajectory[-2]
            if previous_step.progress is not None:
                progress_prompt = self.prompt.progress_prompt.format(
                    progress = previous_step.progress,
                )
                prompt_list.append(progress_prompt)

            if previous_step.memory is not None:
                memory_prompt = self.prompt.memory_prompt.format(
                    memory = previous_step.memory,
                )
                prompt_list.append(memory_prompt)

        observation_prompt = self.prompt.observation_prompt(
            resized_width = resized_width,
            resized_height = resized_height,
            image_placeholder = IMAGE_PLACEHOLDER,
        )
        prompt_list.append(observation_prompt)

        response_prompt = self.prompt.response_prompt
        prompt_list.append(response_prompt)

        prompt = "\n\n".join(prompt_list)
        user_message = generate_message("user", prompt, images=[pixels])
        messages.append(user_message)

        return messages



class Reflector(SubAgent):
    def __init__(self, vlm, prompt_config = None):
        super().__init__(vlm)
        self.prompt: ReflectorPrompt = load_prompt("reflector", prompt_config)
        self.valid_options = ['A', 'B', 'C', 'D']

    def get_message(self, episodedata: MobileUseEpisodeData) -> list:
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
        system_message = generate_message("system", self.prompt.system_prompt)
        messages.append(system_message)

        # Add user prompt
        prompt_list = []

        task_prompt = self.prompt.task_prompt.format(
            task_description = episodedata.goal,
        )
        prompt_list.append(task_prompt)

        subgoal_prompt = self.prompt.subgoal_prompt(
            subgoal = current_step.sub_goal,
        )
        prompt_list.append(subgoal_prompt)

        observation_prompt = self.prompt.observation_prompt(
            screenshot1 = IMAGE_PLACEHOLDER,
            screenshot2 = IMAGE_PLACEHOLDER,
            resized_width = resized_width,
            resized_height = resized_height,
        )
        if diff_flag:
            logger.info("The last action successfully produces some changes. The difference between the two images is highlighted in red boxes.")
            observation_prompt += "\n" + self.prompt.diff_image_prompt
        prompt_list.append(observation_prompt)

        expection_prompt = self.prompt.expection_prompt(
            action_s = current_step.action_s,
            action_desc = current_step.action_desc,
        )
        prompt_list.append(expection_prompt)

        response_prompt = self.prompt.response_prompt
        prompt_list.append(response_prompt)

        prompt = "\n\n".join(prompt_list)
        user_message = generate_message("user", prompt, images=[pixels_before, pixels_after])
        messages.append(user_message)

        return messages

    def parse_response(self, response: str) -> dict:
        outcome = response.split("### Outcome ###")[-1].split("### Error Description ###")[0].replace("\n", " ").replace("  ", " ").strip()
        error_description = response.split("### Error Description ###")[-1].split("### Explanation ###")[0].replace("\n", " ").replace("  ", " ").strip()
        return outcome, error_description


class TrajectoryReflector(SubAgent):
    def __init__(
        self,
        vlm,
        prompt_config = None,
        evoke_every_steps: int = 5,  # how often to evoke the reflector
        cold_steps: int = 3,  # how many steps to wait before the last evoke
        detect_error: bool = True,  # whether to detect error patterns in the trajectory
        num_histories: Union[Literal['auto'], int] = 'auto',  # how many histories to include in the prompt, 'auto' means use evoke_every_steps
        num_latest_screenshots: int = 0,  # how many latest screenshots to include in the prompt
        max_repeat_action: int = 3,  # error detection parameters, how many times an action is repeated will cause an error
        max_repeat_action_series: int = 2,  # error detection parameters, how many times a series of actions is repeated will cause an error
        max_repeat_screen: int = 3,  # error detection parameters, how many times a screenshot is repeated will cause an error
        max_fail_count: int = 3  # error detection parameters, how many times the actions are failed will cause an error
    ):
        super().__init__(vlm)
        self.prompt: TrajectoryReflectorPrompt = load_prompt("trajectory_reflector", prompt_config)
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

        self.max_repeat_action = max_repeat_action
        self.max_repeat_action_series = max_repeat_action_series
        self.max_repeat_screen = max_repeat_screen
        self.max_fail_count = max_fail_count
    
    def detect(
        self, 
        episodedata: MobileUseEpisodeData,
    ) -> list:
        error = []
        trajectory = episodedata.trajectory
        if len(trajectory) < min(self.max_repeat_action, self.max_repeat_screen, self.max_fail_count):
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
                if repeat_action >= self.max_repeat_action:
                    error.append(f"The action `{current_step.action_s}` has repeated more than {self.max_repeat_action} times.")
                    break

        # detect repeated action series
        if len(trajectory) >= 4:
            if trajectory[-1].action == trajectory[-3].action and trajectory[-2].action == trajectory[-4].action:
                error.append(f"The latest two actions have repeated more than {self.max_repeat_action_series} times.")
        if len(trajectory) >= 6:
            if trajectory[-1].action == trajectory[-4].action and trajectory[-2].action == trajectory[-5].action and trajectory[-3].action == trajectory[-6].action:
                error.append(f"The latest three actions have repeated more than {self.max_repeat_action_series} times.")

        # detect repeated screenshots
        repeat_screen = 1
        for step in trajectory[:-1][::-1]:
            if is_same_image(step.exec_env_state.pixels.copy(), current_step.exec_env_state.pixels.copy()):
                repeat_screen += 1
            else:
                break
            if repeat_screen >= self.max_repeat_screen:
                error.append(f"The screen has kept unchanged for more than {self.max_repeat_screen} times.")
                break

        # detect fail reflection
        if current_step.reflection_outcome is not None:
            fail_count = 0
            for step in trajectory[::-1]:
                if step.reflection_outcome in ['B', 'C']:
                    fail_count += 1
                else:
                    break
                if fail_count >= self.max_fail_count:
                    error.append(f"You have encountered several failed attempts.")
                    break

        return error
        

    def get_message(self, episodedata: MobileUseEpisodeData) -> list:
        error = []
        if self.detect_error:
            error = self.detect(episodedata)
        step_idx = len(episodedata.trajectory)

        if step_idx % self.evoke_every_steps != 0 and len(error) == 0:
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
        else:
            screenshots = None

        # Add system prompt
        system_message = generate_message("system", self.prompt.system_prompt)
        messages.append(system_message)

        # Add user prompt
        prompt_list = []

        task_prompt = self.prompt.task_prompt.format(
            task_description = episodedata.goal,
        )
        prompt_list.append(task_prompt)

        if current_step.plan is not None:
            plan_prompt = self.prompt.plan_prompt(
                plan = current_step.plan,
            )
            prompt_list.append(plan_prompt)
        
        history = get_history(trajectory, self.num_histories)
        history = "\n".join(history)
        if current_step.answer is not None:
            history += f"\nFinal answer: {current_step.answer}"
        history_prompt = self.prompt.history_prompt(
            history = history,
        )
        prompt_list.append(history_prompt)
            
        if current_step.progress is not None:
            progress_prompt = self.prompt.progress_prompt(
                progress = current_step.progress,
            )
            prompt_list.append(progress_prompt)

        if num_latest_screenshots > 0:
            image_placeholders = IMAGE_PLACEHOLDER * num_latest_screenshots
            observation_prompt = self.prompt.observation_prompt(
                resized_width = resized_width,
                resized_height = resized_height,
                image_placeholders = image_placeholders,
            )
            prompt_list.append(observation_prompt)
        
        if len(error) > 0:
            error = '\n'.join(error)
            logger.info(f"Trajectory Reflector detects error: {error}")
            error_info_prompt = self.prompt.error_info_prompt(
                error = error,
            )
            prompt_list.append(error_info_prompt)

        response_prompt = self.prompt.response_prompt
        prompt_list.append(response_prompt)

        prompt = "\n\n".join(prompt_list)
        user_message = generate_message("user", prompt, images=screenshots)
        messages.append(user_message)

        return messages

    def parse_response(self, response: str) -> dict:
        outcome = response.split("### Outcome ###")[-1].split("### Error Description ###")[0].replace("\n", " ").replace("  ", " ").strip()
        error_description = response.split("### Error Description ###")[-1].split("### Explanation ###")[0].replace("\n", " ").replace("  ", " ").strip()
        return outcome, error_description


class GlobalReflector(SubAgent):
    def __init__(self, vlm: VLMWrapper, prompt_config = None, num_latest_screenshots: int = 3):
        super().__init__(vlm)
        self.prompt: GlobalReflectorPrompt = load_prompt("global_reflector", prompt_config)
        self.num_latest_screenshots = num_latest_screenshots

    def get_message(self, episodedata: MobileUseEpisodeData) -> list:
        messages = []
        trajectory = episodedata.trajectory
        last_step = trajectory[-1]

        num_latest_screenshots = min(self.num_latest_screenshots, len(trajectory))
        if num_latest_screenshots > 0:
            screenshots = [step.exec_env_state.pixels.copy() for step in trajectory[-num_latest_screenshots:]]
            resized_height, resized_width = smart_resize(height=screenshots[0].height, width=screenshots[0].width)
        else:
            screenshots = None

        # Add system prompt
        system_message = generate_message("system", self.prompt.system_prompt)
        messages.append(system_message)

        # Add user prompt
        prompt_list = []

        task_prompt = self.prompt.task_prompt.format(
            task_description = episodedata.goal,
        )
        prompt_list.append(task_prompt)

        if last_step.plan is not None:
            plan_prompt = self.prompt.plan_prompt.format(
                plan = last_step.plan,
            )
            prompt_list.append(plan_prompt)

        history = get_history(trajectory)
        history = "\n".join(history)
        if last_step.answer is not None:
            history += f"\nFinal answer: {last_step.answer}"
        history_prompt = self.prompt.history_prompt.format(
            history = history,
        )
        prompt_list.append(history_prompt)

        progress_prompt = self.prompt.progress_prompt.format(
            progress = last_step.progress
        )
        prompt_list.append(progress_prompt)

        if num_latest_screenshots > 0:
            image_placeholders = IMAGE_PLACEHOLDER * num_latest_screenshots
            observation_prompt = self.prompt.observation_prompt.format(
                resized_width = resized_width,
                resized_height = resized_height,
                image_placeholders = image_placeholders,
            )
            prompt_list.append(observation_prompt)

        response_prompt = self.prompt.response_prompt
        prompt_list.append(response_prompt)

        prompt = "\n\n".join(prompt_list)
        user_message = generate_message("user", prompt, images=screenshots)
        messages.append(user_message)

        return messages
    
    def parse_response(self, response: str):
        result = response.split("### Result ###")[-1].split("### Reason ###")[0].replace("\n", " ").replace("  ", " ").strip()
        reason = response.split("### Reason ###")[-1].strip()
        return result, reason, None


"""
Gemerate memory
"""
class NoteTaker(SubAgent):
    def get_message(self, episodedata: MobileUseEpisodeData) -> list:
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

        if current_step.plan is not None:
            prompt += "### Overall Plan ###\n"
            prompt += f"{current_step.plan}\n\n"

        if current_step.sub_goal is not None:
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
class Progressor(SubAgent):
    def __init__(self, vlm, prompt_config = None):
        super().__init__(vlm)
        self.prompt: ProgressorPrompt = load_prompt("progressor", prompt_config)

    def get_message(self, episodedata: MobileUseEpisodeData) -> list:
        messages = []
        trajectory = episodedata.trajectory
        current_step = trajectory[-1]
        
        # Add system prompt
        system_message = generate_message("system", self.prompt.system_prompt)
        messages.append(system_message)

        # Add user prompt
        prompt_list = []

        # task_prompt = self.prompt.task_prompt.format(
        #     task_description = episodedata.goal,
        # )
        # prompt_list.append(task_prompt)

        if len(trajectory) > 1:
            history = get_history(trajectory[:-1])
            history = "\n".join(history)
            previous_step = trajectory[-2]
            continue_progress_start = self.prompt.continue_progress_start.format(
                history = history,
                progress = previous_step.progress,
                action_desc = current_step.action_desc,
                action = current_step.action,
            )
            prompt_list.append(continue_progress_start)

            if current_step.reflection_outcome is not None:
                if current_step.reflection_outcome in ['B', 'C']:
                    continue_progress_reflection = self.prompt.continue_progress_reflection.format(
                        reflection_error = current_step.reflection_error,
                    )
                    prompt_list.append(continue_progress_reflection)

            continue_progress_response = self.prompt.continue_progress_response
            prompt_list.append(continue_progress_response)

        else:
            init_progress = self.prompt.init_progress.format(
                thought = current_step.thought,
                action_desc = current_step.action_desc,
                action = current_step.action,
            )
            prompt_list.append(init_progress)
            
        prompt = "\n\n".join(prompt_list)
        user_message = generate_message("user", prompt)
        messages.append(user_message)

        return messages
    
    def parse_response(self, response: str):
        return response.split("### Completed contents ###")[-1].replace("\n", " ").replace("  ", " ").strip()





if __name__ == "__main__":
    vlm = VLMWrapper(
        model_name="qwen2.5-vl-72b-instruct",
        api_key='EMPTY',
        base_url='http://hammer-llm.oppo.test/v1'
    )
    pixels = Image.open(r"/home/notebook/data/personal/S9057346/tmp/mobile-use/benchmark/android_world/ref_image.png")
    step_data = MobileUseStepData(step_idx=0, curr_env_state=EnvState(pixels=pixels, package=""))
    episodedata = MobileUseEpisodeData(goal="Open Wifi", num_steps=0, trajectory=[step_data])
    sub_agent = Planner(vlm)
    messages = sub_agent.get_message(episodedata)
    show_message(messages)
    # response = sub_agent.predict(messages)
    # content = sub_agent.get_content(response)
    # print(f"Response Content: {content}")
    # thought, plan, current_subgoal = sub_agent.parse_response(content)
    # print(f"Thought: {thought}")
    # print(f"Plan: {plan}")
    # print(f"Current Subgoal: {current_subgoal}")
