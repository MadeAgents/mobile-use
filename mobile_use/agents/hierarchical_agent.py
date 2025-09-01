import logging
from typing import Iterator
import os
import pickle
import gzip
import io
import json
import time
import traceback

from mobile_use.schema.schema import *
from mobile_use.environment.mobile_environ import Environment
from mobile_use.utils.vlm import VLMWrapper
from mobile_use.utils.utils import encode_image_url, smart_resize, show_message, generate_message
from mobile_use.agents import Agent
from mobile_use.agents.sub_agent import *
from mobile_use.schema.config import SubAgentConfig, HierarchicalAgentConfig

logger = logging.getLogger(__name__)


@Agent.register('HierarchicalAgent')
class HierarchicalAgent(Agent):
    def __init__(self, config_path: str):
        super().__init__(config_path)
        config = HierarchicalAgentConfig.from_yaml(config_path)
        self.config = config

        self.subagent_map = {
            'Operator': Operator,
            'TrainedOperator': TrainedOperator,
            'OperatorQwen': OperatorQwen,
            'AnswerAgent': AnswerAgent,
            'TrainedAnswerAgent': TrainedAnswerAgent,
            'AnswerAgentQwen': AnswerAgentQwen,
        }

        self._init_sub_agents()

        self.max_action_retry = self.config.max_action_retry
        self.reflect_on_demand = self.config.reflect_on_demand
        self.logprob_threshold = self.config.logprob_threshold
        self.enable_pre_reflection = self.config.enable_pre_reflection
        self.enable_hierarchical_planning = self.config.enable_hierarchical_planning

    def _init_data(self, goal: str='', max_steps: int=10):
        super()._init_data(goal, max_steps)
        self.trajectory: List[MobileUseStepData] = []
        self.episode_data: MobileUseEpisodeData = MobileUseEpisodeData(goal=goal, num_steps=0, trajectory=self.trajectory)
        self.task_data = HierarchicalAgentTaskData(task=goal, episode_data=self.episode_data)


    def _init_sub_agent(self, sub_agent_class: SubAgent, sub_agent_config: Optional[SubAgentConfig]) -> Optional[SubAgent]:
        if sub_agent_config and sub_agent_config.enabled:
            if sub_agent_config.vlm is None:
                sub_agent_config.vlm = self.config.vlm
            return sub_agent_class(sub_agent_config)
        return None

    def _init_sub_agents(self):
        operator_class = self.subagent_map[self.config.operator.name] if self.config.operator else None
        answer_agent_class = self.subagent_map[self.config.answer_agent.name] if self.config.answer_agent else None
        self.planner = self._init_sub_agent(Planner, self.config.planner)
        self.operator = self._init_sub_agent(operator_class, self.config.operator)
        self.answer_agent = self._init_sub_agent(answer_agent_class, self.config.answer_agent)
        self.reflector = self._init_sub_agent(Reflector, self.config.reflector)
        self.trajectory_reflector = self._init_sub_agent(TrajectoryReflector, self.config.trajectory_reflector)
        self.global_reflector = self._init_sub_agent(GlobalReflector, self.config.global_reflector)
        self.progressor = self._init_sub_agent(Progressor, self.config.progressor)
        self.note_taker = self._init_sub_agent(NoteTaker, self.config.note_taker)
        self.task_classifier = self._init_sub_agent(TaskClassifier, self.config.task_classifier)
        self.task_orchestrator = self._init_sub_agent(TaskOrchestrator, self.config.task_orchestrator)
        self.task_extractor = self._init_sub_agent(TaskExtractor, self.config.task_extractor)
        self.task_rewriter = self._init_sub_agent(TaskRewriter, self.config.task_rewriter)

    def reset(self, goal: str='', max_steps: int = 10) -> None:
        """Reset the state of the agent.
        """
        self._init_data(goal=goal, max_steps=max_steps)
        self._init_sub_agents()

    def _get_curr_step_data(self):
        if len(self.trajectory) > self.curr_step_idx:
            return self.trajectory[self.curr_step_idx]
        else:
            return None

    def get_action_type_logprobs(self, response):
        action_type_tokens, action_type_logprobs = None, None
        tokens, logprobs = [], []
        for item in response.choices[0].logprobs.content:
            tokens.append(item.token)
            logprobs.append(item.logprob)

        start_index = next((i for i in range(len(tokens) - 1, -1, -1) if 'action' in tokens[i]), None)
        if start_index is not None:
            end_index = next((i for i in range(start_index + 1, len(tokens)) if ',' in tokens[i]), None)
            if end_index is not None:
                action_type_idxs = [i for i in range(start_index + 1, end_index) if any(c.isalpha() for c in tokens[i])]
                action_type_tokens = [tokens[i] for i in action_type_idxs]
                action_type_logprobs = [logprobs[i] for i in action_type_idxs]
                logger.info("Action type tokens: %s" % action_type_tokens)
                logger.info("Action type logprobs: %s" % action_type_logprobs)
        return action_type_tokens, action_type_logprobs

    def step(self):
        """Execute the task with maximum number of steps.

        Returns: Answer
        """
        start_time = time.time()
        logger.info("Step %d ... ..." % self.curr_step_idx)
        show_step = [0,4]

        # Task classification and planning
        if self.enable_hierarchical_planning and self.curr_step_idx == 0:
            import json
            task_type_json = json.load(open("benchmark/android_world/tasks_type.json", "r"))
            for task_info in task_type_json:
                if task_info['goal'] == self.goal:
                    if task_info['type'] in ['A', 'C']:
                        task_type = task_info['type']
                        sub_tasks = task_info['sub_tasks']
                        logger.info("Task Type: %s" % task_type)
                        logger.info("Sub Tasks: %s" % str(sub_tasks))
                        self.task_data.task_type = task_type
                        self.task_data.sub_tasks = sub_tasks
                        self.task_data.sub_tasks_return = [None] * len(sub_tasks)
                        self.task_data.sub_tasks_episode_data = [None] * len(sub_tasks)
                        self.task_data.current_sub_task_idx = 0
                        self.episode_data.goal = self.task_data.sub_tasks[0]
        # if self.enable_hierarchical_planning and self.curr_step_idx == 0:
        #     task_classification_messages = self.task_classifier.get_message(self.task_data)
        #     show_message(task_classification_messages, "TaskClassifier")
        #     response = self.task_classifier.vlm.predict(task_classification_messages)
        #     try:
        #         content = response.choices[0].message.content
        #         logger.info("Task Classification from VLM:\n%s" % content)
        #         task_type = self.task_classifier.parse_response(content)
        #         logger.info("Task Type: %s" % task_type)
        #         self.task_data.task_type = task_type
        #     except Exception as e:
        #         logger.warning(f"Failed to parse the task type. Error: {e}")
        #     if self.task_data.task_type in ['A', 'C']:
        #         task_orchestrator_messages = self.task_orchestrator.get_message(self.task_data)
        #         show_message(task_orchestrator_messages, "TaskOrchestrator")
        #         response = self.task_orchestrator.vlm.predict(task_orchestrator_messages)
        #         try:
        #             content = response.choices[0].message.content
        #             logger.info("Task Orchestration from VLM:\n%s" % content)
        #             sub_tasks = self.task_orchestrator.parse_response(content)
        #             if sub_tasks is not None and len(sub_tasks) > 0:
        #                 logger.info("Sub Tasks: %s" % str(sub_tasks))
        #                 self.task_data.sub_tasks = sub_tasks
        #                 self.task_data.sub_tasks_return = [None] * len(sub_tasks)
        #                 self.task_data.sub_tasks_episode_data = [None] * len(sub_tasks)
        #                 self.task_data.current_sub_task_idx = 0
        #                 self.goal = self.task_data.sub_tasks[0]
        #                 self.episode_data.goal = self.goal
        #                 logger.info(f"Update the goal to the first sub task: {self.goal}")
        #         except Exception as e:
        #             logger.warning(f"Failed to parse the sub tasks. Error: {e}")

        # Get the current environment screen
        env_state = self.env.get_state()
        pixels = env_state.pixels
        resized_height, resized_width = smart_resize(height=pixels.height, width=pixels.width)

        # Add new step data
        self.trajectory.append(MobileUseStepData(
            step_idx=self.curr_step_idx,
            curr_env_state=env_state,
        ))
        step_data = self.trajectory[-1]

        # Call planner
        if self.planner:
            plan_messages = self.planner.get_message(self.episode_data)
            if self.curr_step_idx in show_step:
                show_message(plan_messages, "Planner")
            response = self.planner.vlm.predict(plan_messages)
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

        # Call Operator
        action_thought, action, action_s, action_desc = None, None, None, None
        skip_reflector = False
        operator_messages = self.operator.get_message(self.episode_data)
        if self.curr_step_idx in show_step:
            show_message(operator_messages, "Operator")
        response = self.operator.vlm.predict(operator_messages, logprobs=self.reflect_on_demand)

        for counter in range(self.max_action_retry):
            try:
                raw_action = response.choices[0].message.content
                logger.info("Action from VLM:\n%s" % raw_action)
                resized_size = (resized_width, resized_height)
                action_thought, action, action_s, action_desc = self.operator.parse_response(raw_action)
                logger.info("ACTION THOUGHT: %s" % action_thought)
                logger.info("ACTION: %s" % str(action))
                logger.info("ACTION DESCRIPTION: %s" % action_desc)
                break
            except Exception as e:
                logger.warning(f"Failed to parse the action. Error is {e.args}")
                error_prompt = f"Failed to parse the action. Error is {e.args}\nPlease follow the output format to provide a valid action:"
                msg = {"role": "user", "content": [{"type": "text", "text": error_prompt}]}
                operator_messages.append(msg)
                response = self.operator.vlm.predict(operator_messages)
        if counter > 0:
            operator_messages = operator_messages[:-counter]

        if action is None:
            logger.warning("Action parse error after max retry.")
        else:
            try:
                if action.name == 'terminate':
                    if action.parameters['status'] == 'success':
                        logger.info(f"Finished: {action}")
                        self.status = AgentStatus.FINISHED
                        self.episode_data.finish_count += 1
                    elif action.parameters['status'] == 'failure':
                        logger.info(f"Failed: {action}")
                        self.status = AgentStatus.FAILED
                elif action.name == 'take_note':
                    logger.info(f"Take note: {action}")
                    self.episode_data.memory += action.parameters['text'].strip()
                    self.episode_data.memory += "\n"
                    logger.info(f"Current Memory: {self.episode_data.memory}")
                    skip_reflector = True
                elif action.name == 'answer':
                    logger.info(f"Answer: {action}")
                    answer = action.parameters['text'].strip()
                    step_data.answer = answer
                    self.status = AgentStatus.FINISHED
                    logger.info("Terminate the task after answering question.")
                elif self.enable_pre_reflection and action.name == 'type' and \
                        len(self.trajectory) > 1 and self.trajectory[-2].action.name == 'type' and \
                        'coordinate' not in action.parameters:
                        skip_reflector = True
                        step_data.reflection_outcome = 'C'
                        step_data.reflection_error = "Action executed failed. You should first click the corresponding text field before typing in text."
                        logger.info(f"Skip the reflector since there is continuous type action.")
                else:
                    logger.info(f"Execute the action: {action}")
                    start_exec_time = time.time()
                    self.env.execute_action(action)
                    step_data.exec_duration = time.time() - start_exec_time
            except Exception as e:
                logger.warning(f"Failed to execute the action: {action}. Error: {traceback.format_exc()}")
                action = None

        if action is not None:
            step_data.thought = action_thought
            step_data.action_desc = action_desc
            step_data.action_s = action_s
            step_data.action = action

            if self.reflector and self.reflect_on_demand:
                action_type_tokens, action_type_logprobs = None, None
                try:
                    action_type_tokens, action_type_logprobs = self.get_action_type_logprobs(response)
                except Exception as e:
                    logger.warning(f"Failed to get the logprobs. Error: {e}")
                if action_type_tokens is not None and action_type_logprobs is not None:
                    avg_logprob = sum(action_type_logprobs) / len(action_type_logprobs)
                    logger.info(f"Average action type logprobs: {avg_logprob}")
                    step_data.action_type_tokens = action_type_tokens
                    step_data.action_type_logprobs = action_type_logprobs
                    if avg_logprob > self.logprob_threshold:
                        logger.info(f"Skip the reflector since the action type logprobs is lower than the threshold.")
                        skip_reflector = True

        step_data.exec_env_state = self.env.get_state()

        if self.status not in [AgentStatus.FINISHED, AgentStatus.FAILED] and action is not None:
            # Call NoteTaker
            if self.note_taker:
                note_messages = self.note_taker.get_message(self.episode_data)
                if self.curr_step_idx in show_step:
                    show_message(note_messages, "NoteTaker")
                response = self.note_taker.vlm.predict(note_messages)
                try:
                    content = response.choices[0].message.content
                    logger.info("Note from VLM:\n%s" % content)
                    note = self.note_taker.parse_response(content)
                    if note is not None:
                        logger.info("Note: %s" % note)
                        self.episode_data.memory = note.strip()
                        logger.info(f"Current Memory: {self.episode_data.memory}")
                except Exception as e:
                    logger.warning(f"Failed to parse the note. Error: {e}")

            # Call Reflector
            if self.reflector and not skip_reflector:
                reflection_messages = self.reflector.get_message(self.episode_data)
                if self.curr_step_idx in show_step:
                    show_message(reflection_messages, "Reflector")
                response = self.reflector.vlm.predict(reflection_messages)
                try:
                    content = response.choices[0].message.content
                    logger.info("Reflection from VLM:\n%s" % content)
                    outcome, error_description = self.reflector.parse_response(content)
                    if outcome in self.reflector.valid_options:
                        logger.info("Reflection Outcome: %s" % outcome)
                        logger.info("Reflection Error: %s" % error_description)
                        step_data.reflection_outcome = outcome
                        step_data.reflection_error = error_description
                except Exception as e:
                    logger.warning(f"Failed to parse the reflection. Error: {e}")

            # Call Progressor
            if self.progressor:
                progressor_messages = self.progressor.get_message(self.episode_data)
                if self.curr_step_idx in show_step:
                    show_message(progressor_messages, "Progressor")
                response = self.progressor.vlm.predict(progressor_messages)
                try:
                    content = response.choices[0].message.content
                    logger.info("Progress from VLM:\n%s" % content)
                    progress = self.progressor.parse_response(content)
                    logger.info("Progress: %s" % progress)
                    step_data.progress = progress
                except Exception as e:
                    logger.warning(f"Failed to parse the progress. Error: {e}")

            # Call TrajectoryReflector
            if self.trajectory_reflector:
                detected_error, trajectory_reflection_messages = self.trajectory_reflector.get_message(self.episode_data)
                if trajectory_reflection_messages is not None:
                    if self.curr_step_idx in [4,9]:
                        show_message(trajectory_reflection_messages, "TrajectoryReflector")
                    response = self.trajectory_reflector.vlm.predict(trajectory_reflection_messages)
                    try:
                        content = response.choices[0].message.content
                        logger.info("Trajectory Reflection from VLM:\n%s" % content)
                        outcome, error_description = self.trajectory_reflector.parse_response(content)
                        if detected_error is not None:
                            error_description = detected_error + "\n" + error_description
                        if outcome in self.trajectory_reflector.valid_options:
                            logger.info("Trajectory Reflection Outcome: %s" % outcome)
                            logger.info("Trajectory Reflection Error: %s" % error_description)
                            step_data.trajectory_reflection_outcome = outcome
                            step_data.trajectory_reflection_error = error_description
                    except Exception as e:
                        logger.warning(f"Failed to parse the trajectory reflection. Error: {e}")

        # Call AnswerAgent
        if self.status == AgentStatus.FINISHED:
            answer_messages = self.answer_agent.get_message(self.episode_data)
            show_message(answer_messages, "Answer")
            response = self.answer_agent.vlm.predict(answer_messages)
            try:
                content = response.choices[0].message.content
                logger.info("Answer from VLM:\n%s" % content)
                _, answer_action, _, _ = self.answer_agent.parse_response(content)
                answer = answer_action.parameters['text']
                step_data.answer = answer
                logger.info("Answer: %s" % answer)
            except Exception as e:
                logger.warning(f"Failed to get the answer. Error: {e}")
            
            # Call GlobalReflector
            if self.global_reflector and self.episode_data.finish_count == 1:
                evaluator_messages = self.global_reflector.get_message(self.episode_data)
                show_message(evaluator_messages, "Evaluator")
                logger.info("Evaluating...")
                response = self.global_reflector.vlm.predict(evaluator_messages)
                result, reason = None, None
                try:
                    content = response.choices[0].message.content
                    logger.info("Evaluation from VLM:\n%s" % content)
                    result, reason = self.global_reflector.parse_response(content)
                    logger.info("Evaluation Result: %s" % result)
                    logger.info("Evaluation Reason: %s" % reason)
                except Exception as e:
                    logger.warning(f"Failed to parse the evaluation. Error: {e}")
                if result is not None and 'Failed' in result:
                    logger.info("Evaluator determines that the task is not completed for the first time. Will remove the FINISH status.")
                    self.status = None
                    step_data.evaluation_result = result
                    step_data.evaluation_reason = reason
        
        # Next sub task
        if self.status == AgentStatus.FINISHED and self.task_data.sub_tasks is not None:
            self.task_data.sub_tasks_episode_data[self.task_data.current_sub_task_idx] = self.episode_data
            if self.task_data.current_sub_task_idx + 1 < len(self.task_data.sub_tasks):
                if self.task_data.task_type == 'A':
                    # extract sub task info
                    sub_task_info_messages = self.task_extractor.get_message(self.task_data)
                    show_message(sub_task_info_messages, "TaskExtractor")
                    response = self.task_extractor.vlm.predict(sub_task_info_messages)
                    try:
                        content = response.choices[0].message.content
                        logger.info("Sub Task Info from VLM:\n%s" % content)
                        sub_task_info = self.task_extractor.parse_response(content)
                        if sub_task_info is not None:
                            logger.info("Sub Task Info: %s" % str(sub_task_info))
                            self.task_data.sub_tasks_return[self.task_data.current_sub_task_idx] = sub_task_info
                    except Exception as e:
                        logger.warning(f"Failed to parse the sub task info. Error: {e}")

                    # rewrite the next sub task
                    sub_task_rewrite_messages = self.task_rewriter.get_message(self.task_data)
                    show_message(sub_task_rewrite_messages, "TaskRewriter")
                    response = self.task_rewriter.vlm.predict(sub_task_rewrite_messages)
                    try:
                        content = response.choices[0].message.content
                        logger.info("Rewritten Sub Task from VLM:\n%s" % content)
                        rewritten_sub_task = self.task_rewriter.parse_response(content)
                        if rewritten_sub_task is not None:
                            logger.info("Rewritten Sub Task: %s" % rewritten_sub_task)
                            self.task_data.sub_tasks[self.task_data.current_sub_task_idx + 1] = rewritten_sub_task
                    except Exception as e:
                        logger.warning(f"Failed to parse the rewritten sub task. Error: {e}")
                    
                    # Go back to home
                    for i in range(3):
                        self.env.execute_action(Action(name="press_back"))
                    self.env.execute_action(Action(name="press_home"))

                self.task_data.current_sub_task_idx += 1
                new_goal = self.task_data.sub_tasks[self.task_data.current_sub_task_idx]
                new_trajectory: List[MobileUseStepData] = []
                new_episodedata = MobileUseEpisodeData(goal=new_goal, num_steps=0, trajectory=new_trajectory)
                self.trajectory = new_trajectory
                self.episode_data = new_episodedata
                self.task_data.episode_data = new_episodedata
                logger.info(f"Update the goal to the next sub task: {self.goal}")
                self.status = None
                # self.status = AgentStatus.FINISHED

        step_data.memory = self.episode_data.memory
        step_data.step_duration = time.time() - start_time
        return step_data


    def iter_run(self, input_content: str, stream: bool=False) -> Iterator[MobileUseStepData]:
        """Execute the agent with user input content.

        Returns: Iterator[StepData]
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
            elif self.state == AgentState.CALLUSER:
                logger.info("Agent indicates to ask user for help.")
                yield self._get_curr_step_data()
                return
            else:
                logger.info("Agent indicates one step is done.")
            yield self._get_curr_step_data()
        logger.warning(f"Agent reached max number of steps: {self.max_steps}.")

    def run(self, input_content: str) -> HierarchicalAgentTaskData:
        """Execute the agent with user input content.

        Returns: EpisodeData
        """
        for _ in self.iter_run(input_content, stream=False):
            pass
        return self.episode_data
