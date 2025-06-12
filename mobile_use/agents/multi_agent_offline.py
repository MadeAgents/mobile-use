import logging
import re
from typing import Iterator
import json

from mobile_use.scheme import *
from mobile_use.environ import Environment
from mobile_use.vlm import VLMWrapper
from mobile_use.utils import encode_image_url, smart_resize
from mobile_use.agents import Agent

from mobile_use.agents.sub_agent import Planner, Operator, NoteTaker, Processor, ReflectorBeforeExecution, Evaluator, AdvertisementDetector


logger = logging.getLogger(__name__)


ANSWER_PROMPT_TEMPLATE = """
The (overall) user query is: {goal}
Now you have finished the task. I want you to provide an answer to the user query.
Answer with the following format:

## Format
<tool_call>
{{"name": "mobile_use", "arguments": {{"action": "answer", "text": <your-answer>}}}}
</tool_call>"""

def show_message(messages: List[dict], name: str = None):
    name = f"{name} " if name is not None else ""
    logger.info(f"==============={name}MESSAGE==============")
    for message in messages:
        logger.info(f"ROLE: {message['role']}")
        for content in message['content']:
            if content['type'] == 'text':
                logger.info(f"TEXT:")
                logger.info(content['text'])
    logger.info(f"==============={name}MESSAGE END==============")

@Agent.register('MultiAgentOffline')
class MultiAgentOffline(Agent):
    def __init__(
            self, 
            env: Environment,
            vlm: VLMWrapper,
            max_steps: int=10,
            num_latest_screenshot: int=10,
            num_histories: int = None,
            max_reflection_action: int=3,
            reflection_action_waiting_seconds: float=1.0,
            max_retry_vlm: int=3,
            retry_vlm_waiting_seconds: float=1.0,
            use_planner: bool=False,
            use_reflector: bool=False,
            use_note_taker: bool=False,
            use_processor: bool=False,
            use_evaluator: bool=False,
            use_operator: bool=True,
            use_advertisement_detector: bool=False,
        ):
        super().__init__(env=env, vlm=vlm, max_steps=max_steps)
        self.num_latest_screenshot = num_latest_screenshot
        self.num_histories = num_histories
        self.max_reflection_action = max_reflection_action
        self.reflection_action_waiting_seconds = reflection_action_waiting_seconds
        self.max_retry_vlm = max_retry_vlm
        self.retry_vlm_waiting_seconds = retry_vlm_waiting_seconds

        self.use_planner = use_planner
        self.use_reflector = use_reflector
        self.use_note_taker = use_note_taker
        self.use_processor = use_processor
        self.use_evaluator = use_evaluator
        self.use_operator = use_operator
        self.use_advertisement_detector = use_advertisement_detector

        self.planner = Planner()
        self.operator = Operator(num_histories=self.num_histories)
        self.reflector = ReflectorBeforeExecution()
        self.note_taker = NoteTaker()
        self.processor = Processor()
        self.evaluator = Evaluator()
        self.advertisement_detector = AdvertisementDetector()

    def reset(self, goal: str='') -> None:
        """Reset the state of the agent.
        """
        self._init_data(goal=goal)
        self.planner = Planner()
        self.operator = Operator(num_histories=self.num_histories)
        self.reflector = ReflectorBeforeExecution()
        self.note_taker = NoteTaker()
        self.processor = Processor()
        self.evaluator = Evaluator()
        self.advertisement_detector = AdvertisementDetector()

    def _get_curr_step_data(self) -> StepData:
        if len(self.trajectory) > self.curr_step_idx:
            return self.trajectory[self.curr_step_idx]
        else:
            return None

    def step(self, pixels: Image.Image, show_step = ()) -> None:
        """Execute the task with maximum number of steps.

        Returns: Action
        """
        logger.info("Step %d ... ..." % self.curr_step_idx)
        is_finish = False

        # Get the current environment screen
        env_state = EnvState(pixels=pixels, package="")
        resized_height, resized_width = smart_resize(height=pixels.height, width=pixels.width)
        logger.info("Get the current screenshot.")

        # Add new step data
        self.trajectory.append(StepData(
            step_idx=self.curr_step_idx,
            curr_env_state=env_state,
            vlm_call_history=[]
        ))
        step_data = self.trajectory[-1]

        # Call planner
        if self.use_planner:
            logger.info("Start call planner.")
            plan_messages = self.planner.get_message(self.episode_data)
            if self.curr_step_idx in show_step:
                show_message(plan_messages, "Planner")
            response = self.vlm.predict(plan_messages)
            try:
                raw_plan = response.choices[0].message.content
                logger.info("Plan from VLM:\n%s" % raw_plan)
                plan_thought, plan, current_subgoal = self.planner.parse_response(raw_plan)
                logger.info("PLAN THOUGHT: %s" % plan_thought)
                logger.info("PLAN: %s" % plan)
                logger.info("CURRENT SUBGOAL: %s" % current_subgoal)
                step_data.plan = plan
                step_data.sub_goal = current_subgoal
            except Exception as e:
                logger.warning(f"Failed to parse the plan. Error: {e}")
            logger.info("finish call planner.")

        if self.use_advertisement_detector:
            logger.info("Start call advertisement detector.")
            ad_messages = self.advertisement_detector.get_message(self.episode_data)
            if self.curr_step_idx in show_step:
                show_message(ad_messages, "Advertisement Detector")
            response = self.vlm.predict(ad_messages)
            try:
                content = response.choices[0].message.content
                logger.info("Advertisement Detection from VLM:\n%s" % content)
                resized_size = (resized_width, resized_height)
                ad_thought, ad_action_name, ad_bounding_box = self.advertisement_detector.parse_response(content, resized_size, pixels.size)
                step_data.ad_detection_thought = ad_thought
                step_data.ad_detection_action_name = ad_action_name
                step_data.ad_detection_bounding_box = str(ad_bounding_box)
            except Exception as e:
                logger.warning(f"Failed to parse the advertisement detection. Error: {e}")
            logger.info("Finish call advertisement detector.")

        action, is_finish = None, None
        if self.use_operator:
            # Call Operator
            logger.info("Start call operator.")
            action_thought, action, action_s, action_desc = None, None, None, None
            operator_messages = self.operator.get_message(self.episode_data)
            if self.curr_step_idx in show_step:
                show_message(operator_messages, "Operator")
            response = self.vlm.predict(operator_messages, stop=['Summary'])

            for counter in range(self.max_reflection_action):
                try:
                    raw_action = response.choices[0].message.content
                    logger.info("Action from VLM:\n%s" % raw_action)
                    step_data.content = raw_action
                    resized_size = (resized_width, resized_height)
                    action_thought, action, action_s, action_desc, finish_thought, finish_result = self.operator.parse_response(raw_action, resized_size, pixels.size)
                    logger.info("ACTION THOUGHT: %s" % action_thought)
                    logger.info("ACTION: %s" % str(action))
                    logger.info("ACTION DESCRIPTION: %s" % action_desc)
                    logger.info("FINISH THOUGHT: %s" % finish_thought)
                    logger.info("FINISH RESULT: %s" % finish_result)
                    step_data.thought = action_thought
                    step_data.action_desc = action_desc
                    step_data.action_s = action_s
                    step_data.action = action
                    step_data.finish_thought = finish_thought
                    step_data.finish_result = finish_result
                    break
                except Exception as e:
                    logger.warning(f"Failed to parse the action. Error is {e.args}")
                    msg = {
                        'type': 'text', 
                        'text': f"Failed to parse the action.\nError is {e.args}\nPlease follow the output format to provide a valid action:"
                    }
                    operator_messages[-1]['content'].append(msg)
                    response = self.vlm.predict(operator_messages, stop=['Summary'])
            if counter > 0:
                operator_messages[-1]['content'] = operator_messages[-1]['content'][:-counter]

            if action is None:
                logger.warning("Action parse error after max retry.")

            logger.info("finish call operator.")

        # Call Reflector
        if self.use_reflector and action is not None:
            logger.info("Start call reflector.")
            outcome = None
            reflection_messages = self.reflector.get_message(self.episode_data)
            if self.curr_step_idx in show_step:
                show_message(reflection_messages, "Reflector")
            response = self.vlm.predict(reflection_messages)
            try:
                content = response.choices[0].message.content
                logger.info("Reflection from VLM:\n%s" % content)
                outcome, error_description = self.reflector.parse_response(content)
                if outcome in ['A', 'B']:
                    logger.info("Outcome: %s" % outcome)
                    logger.info("Error Description: %s" % error_description)
                    step_data.reflection_outcome = outcome
                    step_data.reflection_error = error_description
            except Exception as e:
                logger.warning(f"Failed to parse the reflection. Error: {e}")
            logger.info("Finish call reflector.")

            if outcome == 'B':
                logger.info("Start to refactor the operation.")
                action_thought, action, action_s, action_desc = None, None, None, None
                operator_messages = self.operator.get_message(self.episode_data)
                if self.curr_step_idx in show_step:
                    show_message(operator_messages, "Refact Operator")
                response = self.vlm.predict(operator_messages, stop=['Summary'])

                for counter in range(self.max_reflection_action):
                    try:
                        raw_action = response.choices[0].message.content
                        logger.info("Action from VLM:\n%s" % raw_action)
                        step_data.content = raw_action
                        resized_size = (resized_width, resized_height)
                        action_thought, action, action_s, action_desc = self.operator.parse_response(raw_action, resized_size, pixels.size)
                        logger.info("ACTION THOUGHT: %s" % action_thought)
                        logger.info("ACTION: %s" % str(action))
                        logger.info("ACTION DESCRIPTION: %s" % action_desc)
                        break
                    except Exception as e:
                        logger.warning(f"Failed to parse the action. Error is {e.args}")
                        msg = {
                            'type': 'text', 
                            'text': f"Failed to parse the action.\nError is {e.args}\nPlease follow the output format to provide a valid action:"
                        }
                        operator_messages[-1]['content'].append(msg)
                        response = self.vlm.predict(operator_messages, stop=['Summary'])
                if counter > 0:
                    operator_messages[-1]['content'] = operator_messages[-1]['content'][:-counter]

                if action is None:
                    logger.warning("Action parse error after max retry.")
                
                if action is not None:
                    step_data.thought = action_thought
                    step_data.action_desc = action_desc
                    step_data.action_s = action_s
                    step_data.action = action

                logger.info("finish to refactor the operation.")


        # Call Processor
        if self.use_processor:
            logger.info("Start call processor.")
            processor_messages = self.processor.get_message(self.episode_data)
            if self.curr_step_idx in show_step:
                show_message(processor_messages, "Processor")
            response = self.vlm.predict(processor_messages)
            try:
                content = response.choices[0].message.content
                logger.info("Progress from VLM:\n%s" % content)
                progress = self.processor.parse_response(content)
                logger.info("Progress: %s" % progress)
                step_data.progress = progress
            except Exception as e:
                logger.warning(f"Failed to parse the progress. Error: {e}")
            logger.info("Finish call processor.")
        
        # Call Evaluator
        if self.use_evaluator:
            logger.info("Start call evaluator.")
            evaluator_messages = self.evaluator.get_message(self.episode_data)
            show_message(evaluator_messages, "Evaluator")
            response = self.vlm.predict(evaluator_messages)
            try:
                content = response.choices[0].message.content
                logger.info("Evaluation from VLM:\n%s" % content)
                evaluation = self.evaluator.parse_response(content)
                logger.info("Evaluation: %s" % evaluation)
                if evaluation.startswith("Finished"):
                    step_data.answer = evaluation
                    is_finish = True
            except Exception as e:
                logger.warning(f"Failed to parse the evaluation. Error: {e}")
            logger.info("Finish call evaluator.")
        
        return action, is_finish

    def iter_run(self, input_content):
        pass

    def run(
            self,
            input_content: str, 
            screenshots: List[Image.Image],
            show: bool = False
    ) -> List[Action]:
        """Execute the agent with user input content.

        Returns: List[Action]
        """
        actions = []

        self.reset(goal=input_content)
        max_steps = len(screenshots)

        logger.info(f"Running task: {input_content}.")
        if show:
            show = (0, max_steps - 1)
        else:
            show = ()
        for step_idx in range(max_steps):
            self.curr_step_idx = step_idx

            self.step(screenshots[step_idx], show_step=show)
            action = self.trajectory[-1].action
            is_finish = self.trajectory[-1].finish_result
            actions.append((action, is_finish))

            self.episode_data.num_steps = step_idx + 1
            self.episode_data.status = self.status

        return actions


    def run_ad_detector(
            self,
            input_content: str, 
            screenshots: List[Image.Image]
    ):
        """Execute the agent with user input content.

        Returns: List[Action]
        """
        thoughts, action_names, bboxes = [], [], []

        self.reset(goal=input_content)
        max_steps = len(screenshots)

        logger.info(f"Running task: {input_content}.")

        for step_idx in range(max_steps):
            self.curr_step_idx = step_idx

            self.step(screenshots[step_idx])
            thoughts.append(self.trajectory[-1].ad_detection_thought)
            action_names.append(self.trajectory[-1].ad_detection_action_name)
            bboxes.append(self.trajectory[-1].ad_detection_bounding_box)

            self.episode_data.num_steps = step_idx + 1
            self.episode_data.status = self.status

        return thoughts, action_names, bboxes
