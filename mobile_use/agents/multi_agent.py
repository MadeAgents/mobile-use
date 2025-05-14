import logging
from typing import Iterator
import os
import pickle
import gzip
import io
import json
import time

from mobile_use.scheme import *
from mobile_use.environ import Environment
from mobile_use.vlm import VLMWrapper
from mobile_use.utils import encode_image_url, smart_resize
from mobile_use.agents import Agent

from mobile_use.agents.sub_agent import *


logger = logging.getLogger(__name__)


with open(os.path.join(os.path.dirname(__file__), 'assets', 'explored_knowledge.json'), 'r') as f:
    EXPLORED_SUMMARY: dict = json.load(f)


INIT_TIPS = """- Click the correct text field before typing!
- If the task is finished, you should terminate the task in time!
- If you stuck in an action, you should try to change the action or the correspoinding parameters. Do not always repeat the same action!
- If you want to open an app, always first try to use the `open` action with app name to open the app.
- If you want to delete, move, copy, or rename a file, always first try to long press the file and select the corresponding action.
- Remember to add or change the correct suffix when naming a file.
- Always remember to save the file after you create or modify it.
- When you want to swipe the screen, try to avoid the keyboard area.

### Tips for typing text ###
- When you want to paste text, you should use long press and then click paste. Don't use the clipboard button on the keyboard.
- Before type in text, remember to first click the correct text field.
- Always use the action `clear_text` to clear the text in the text field."""


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
            else:
                logger.info(f"{content['type']}: SKIP.")
    logger.info(f"==============={name}MESSAGE END==============")

def _unzip_and_read_pickle(file_path: str) -> Any:
  """Reads a gzipped pickle file using 'with open', unzips, and unpickles it.

  Args:
      file_path: The path to the gzipped pickle file.

  Returns:
      The original Python object that was pickled and gzipped.
  """
  with open(file_path, 'rb') as f:
    compressed = f.read()

  with gzip.open(io.BytesIO(compressed), 'rb') as f_in:
    return pickle.load(f_in)

def recover_tips(log_dir: str):
    if not log_dir:
        logger.info("Load the initial tips since the log directory is not provided.")
        return INIT_TIPS
    if not os.path.exists(log_dir):
        logger.info(f"Load the initial tips since the log directory {log_dir} does not exist.")
        return INIT_TIPS
    files = os.listdir(log_dir)
    files = [file for file in files if file.endswith('.pkl.gz')]
    if not files:
        logger.info(f"Load the initial tips since the log directory {log_dir} is empty.")
        return INIT_TIPS
    t = time.time()
    files = [file for file in files if os.path.getsize(os.path.join(log_dir, file)) >= 10*1024]
    files.sort(key=lambda x: os.path.getmtime(os.path.join(log_dir, x)), reverse=True)
    for file in files:
        data = _unzip_and_read_pickle(os.path.join(log_dir, file))
        latest_tips = data[0]['episode_data'].get('output_tips', None)
        latest_tips = latest_tips[0] if latest_tips else None
        if latest_tips:
            logger.info(f"Load the latest tips from the log file {file}.")
            logger.info(f"TIPS: {latest_tips}")
            logger.info(f"Tips loading time: {time.time()-t}")
            return latest_tips
    logger.info(f"Load the initial tips since no valid tips are found in the log directory {log_dir}.")
    return INIT_TIPS




@Agent.register('MultiAgent')
class MultiAgent(Agent):
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
            use_long_reflector: bool=False,
            use_note_taker: bool=False,
            use_processor: bool=False,
            use_evaluator: bool=False, # new self-evolution
            use_evolutor: bool=False, # old self-evolution
            evaluate_when_finish: bool=False,
            reflect_on_demand: bool=False,
            logprob_threshold: float=-0.01,
            include_time: bool=True,
            log_dir: str=None,
        ):
        super().__init__(env=env, vlm=vlm, max_steps=max_steps)
        self.num_latest_screenshot = num_latest_screenshot
        self.num_histories = num_histories
        self.max_reflection_action = max_reflection_action
        self.reflection_action_waiting_seconds = reflection_action_waiting_seconds
        self.max_retry_vlm = max_retry_vlm
        self.retry_vlm_waiting_seconds = retry_vlm_waiting_seconds
        self.log_dir = log_dir

        self.use_planner = use_planner
        self.use_reflector = use_reflector
        self.use_long_reflector = use_long_reflector
        self.use_note_taker = use_note_taker
        self.use_processor = use_processor
        self.use_evaluator = use_evaluator
        self.use_evolutor = use_evolutor
        self.evaluate_when_finish = evaluate_when_finish
        self.reflect_on_demand = reflect_on_demand
        self.logprob_threshold = logprob_threshold
        self.include_time = include_time

        self.planner = Planner()
        self.operator = Operator()
        self.reflector = Reflector()
        self.long_reflector = LongReflector()
        self.note_taker = NoteTaker()
        self.processor = Processor()
        self.evaluator = Evaluator()
        self.evolutor = Evolutor()
        self.task_summarizer = TaskSummarizer()
        self.experience_extractor = ExperienceExtractor()

        if self.use_evolutor:
            self.tips = recover_tips(self.log_dir)
        else:
            self.tips = INIT_TIPS

        self.device_time = None
        if self.include_time:
            self.device_time = self._get_device_time()

    def reset(self, goal: str='') -> None:
        """Reset the state of the agent.
        """
        self._init_data(goal=goal)
        self.device_time = None
        if self.include_time:
            self.device_time = self._get_device_time()
        self.planner = Planner()
        self.operator = Operator()
        self.reflector = Reflector()
        self.long_reflector = LongReflector()
        self.note_taker = NoteTaker()
        self.processor = Processor()
        self.evolutor = Evolutor()
        self.task_summarizer = TaskSummarizer()
        self.experience_extractor = ExperienceExtractor()
        self.goal_pre = ''
        
    
    def _get_device_time(self) -> str:
        date_str = self.env.get_time()
        # # Remove the hour-minute-second and the timezone 
        # date_str = ' '.join(date_str.split()[:3] + date_str.split()[-1:])
        return date_str

    def _get_curr_step_data(self) -> StepData:
        if len(self.trajectory) > self.curr_step_idx:
            return self.trajectory[self.curr_step_idx]
        else:
            return None

    def _run_evaluator(self, file):
        logger.info(f"Evaluating file {file}.")
        show = True
        task = file.split('_')[0]
        logger.info(f"Task: {task}")
        data = _unzip_and_read_pickle(os.path.join(self.log_dir, file))
        goal = data[0]['goal']
        logger.info(f"Goal: {goal}")
        is_successful = data[0]['is_successful']
        logger.info(f"Is successful: {is_successful}")
        trajectory = data[0]['episode_data']['step_data']
        episodedata = EpisodeData(goal=goal, num_steps=len(trajectory), trajectory=trajectory)

        evaluator_messages = self.evaluator.get_message(episodedata)
        show_message(evaluator_messages, "Evaluator")
        logger.info("Evaluating...")
        response = self.vlm.predict(evaluator_messages)
        result, reason, tips = None, None, None
        try:
            content = response.choices[0].message.content
            logger.info("Evaluation from VLM:\n%s" % content)
            result, reason, tips = self.evaluator.parse_response(content)
            logger.info("Evaluation Result: %s" % result)
            logger.info("Evaluation Reason: %s" % reason)
            logger.info("Evaluation Tips: %s" % tips)
        except Exception as e:
            logger.warning(f"Failed to parse the evaluation. Error: {e}")

        summary = None
        if result is not None:
            logger.info("Summarizing...")
            summary_messages = self.task_summarizer.get_message(episodedata, result)
            show_message(summary_messages, "TaskSummarizer")
            response = self.vlm.predict(summary_messages)
            try:
                content = response.choices[0].message.content
                logger.info("Task Summary from VLM:\n%s" % content)
                summary = self.task_summarizer.parse_response(content)
                logger.info("Task Summary: %s" % summary)
            except Exception as e:
                logger.warning(f"Failed to parse the summary. Error: {e}")

        return {
            'file': file,
            'task': task,
            'modify_time': os.path.getmtime(os.path.join(self.log_dir, file)),
            'is_successful': is_successful,
            'result': result,
            'goal': goal,
            'reason': reason,
            'tips': tips,
            'summary': summary,
        }

    def evaluate(self, check=False):
        if not os.path.exists(self.log_dir):
            logger.warning(f"EVALUATOR: The log directory {self.log_dir} does not exist.")
            return False
        files = os.listdir(self.log_dir)
        files = [file for file in files if file.endswith('.pkl.gz')]
        files = [file for file in files if os.path.getsize(os.path.join(self.log_dir, file)) >= 10*1024]
        if len(files) == 0:
            logger.warning(f"EVALUATOR: No valid task files are found in the log directory {self.log_dir}.")
            return False
        files.sort(key=lambda x: os.path.getmtime(os.path.join(self.log_dir, x)), reverse=True)
        if check:
            if os.path.exists(os.path.join(self.log_dir, 'evaluator.jsonl')):
                with open(os.path.join(self.log_dir, 'evaluator.jsonl'), 'r') as f:
                    data = f.readlines()
            else:
                data = []
            data = [json.loads(d.strip()) for d in data]
            new_data = []
            for file in files[::-1]:
                modify_time = os.path.getmtime(os.path.join(self.log_dir, file))
                is_find = False
                for d in data:
                    if d['file'] == file:
                        if modify_time != d['modify_time']:
                            logger.warning(f"EVALUATOR: Find unmatched modify time for task file {file}.")
                        else:
                            new_data.append(d)
                            is_find = True
                if not is_find:
                    logger.warning(f"EVALUATOR: The task file {file} is not found in the evaluator, Will re-evaluate it.")
                    new_data.append(self._run_evaluator(file))
            with open(os.path.join(self.log_dir, 'evaluator.jsonl'), 'w') as f:
                for d in new_data:
                    json.dump(d, f)
                    f.write('\n')
        else:
            evaluate_result = self._run_evaluator(files[0])
            with open(os.path.join(self.log_dir, 'evaluator.jsonl'), 'a') as f:
                json.dump(evaluate_result, f)
                f.write('\n')
        return True

    def retrieve_tips(self, gemerate_experience: bool=False):
        logger.info("Retrieving tips...")
        retrieved_tips = None
        t = time.time()
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device="cpu")

        goal = self.episode_data.goal
        with open (os.path.join(self.log_dir, 'evaluator.jsonl'), 'r') as f:
            data = f.readlines()
        data = [json.loads(d.strip()) for d in data]
        candidate_goals = [d['goal'] for d in data]
        embeddings1 = model.encode([goal])
        embeddings2 = model.encode(candidate_goals)
        similarities = model.similarity(embeddings1, embeddings2)
        similarities = similarities[0]
        sorted_indices = similarities.argsort(descending=True)
        retrieved_task = data[sorted_indices[0]]
        logger.info(f"Retrieved file: {retrieved_task['file']}")
        logger.info(f"Retrieving time: {time.time()-t}")

        if gemerate_experience:
            experience_extract_message = self.experience_extractor.get_message(goal, retrieved_task['goal'], retrieved_task['summary'])
            show_message(experience_extract_message, "ExperienceExtractor")
            logger.info("Extracting experience...")
            response = self.vlm.predict(experience_extract_message)
            try:
                content = response.choices[0].message.content
                logger.info("Experience from VLM:\n%s" % content)
                retrieved_tips = self.experience_extractor.parse_response(content)
            except Exception as e:
                logger.warning(f"Failed to parse the experience. Error: {e}")
        else:
            retrieved_tips = retrieved_task['tips']

        logger.info(f"Retrieved tips: {retrieved_tips}")
        return retrieved_tips
    
    def get_action_type_logprobs(self, response):
        action_type_tokens, action_type_logprobs = None, None
        tokens, logprobs = [], []
        for item in response.choices[0].logprobs.content:
            tokens.append(item.token)
            logprobs.append(item.logprob)
        logger.info("Tokens: %s" % tokens)
        logger.info("Logprobs: %s" % logprobs)

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
        
        if self.goal_pre != self.episode_data.goal:  # = is origin, != is using exp
            self.goal_pre = self.episode_data.goal
            explored_summary = ''
            if hasattr(self, 'explored_summary_key'):
                print('self.explored_summary_key is set to:', self.explored_summary_key)
                explored_summary = EXPLORED_SUMMARY.get(self.explored_summary_key, '')
            else:
                print("self.explored_summary_key is not set.")
            if not explored_summary:
                print("No explored summary found.")
            self.episode_data.experience = self.vlm.predict([ { "role": "user", "content": [ {"type": "text", "text": "The summary of pages explored:\n"+explored_summary+"\nuser query:\n"+self.episode_data.goal+"\nPlease extract the text snippet that helps complete the user's request without exceeding 100 tokens and must keeping the original description. Don't answer user query! Extract only!"} ] }]).choices[0].message.content
            print("experience=================", self.episode_data.experience)
            
        start_time = time.time()
        logger.info("Step %d ... ..." % self.curr_step_idx)
        answer = None
        show_step = [0,4]

        # Get the current environment screen
        env_state = self.env.get_state()
        pixels = env_state.pixels
        resized_height, resized_width = smart_resize(height=pixels.height, width=pixels.width)
        

        # Add new step data
        if len(self.trajectory) == 0:
            self.episode_data.input_tips = self.tips
            # Evaluator
            if self.use_evaluator:
                if self.evaluate(check=True):
                    self.episode_data.retrieved_tips = self.retrieve_tips(gemerate_experience=True)

        self.trajectory.append(StepData(
            step_idx=self.curr_step_idx,
            curr_env_state=env_state,
            vlm_call_history=[]
        ))
        step_data = self.trajectory[-1]

        # Call planner
        if self.use_planner:
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

        # Call Operator
        action_thought, action, action_s, action_desc = None, None, None, None
        skip_reflector = False
        # operator_messages = self.operator.get_message(self.episode_data)
        
        operator_messages = self.operator.get_message(self.episode_data, device_time=self.device_time)
        if self.curr_step_idx in show_step:
            show_message(operator_messages, "Operator")
        response = self.vlm.predict(operator_messages, stop=['Summary'], logprobs=self.reflect_on_demand)

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
        else:
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
            else:
                logger.info(f"Execute the action: {action}")
                if action.name == 'type':
                    if len(self.trajectory) > 1 and self.trajectory[-2].action.name == 'type' and 'coordinate' not in action.parameters:
                        skip_reflector = True
                if skip_reflector:
                    step_data.reflection_outcome = 'C'
                    step_data.reflection_error = "Action executed failed. You should first click the corresponding text field before typing in text."
                    logger.info(f"Skip the reflector since there is continuous type action.")
                else:
                    try:
                        start_exec_time = time.time()
                        self.env.execute_action(action)
                        step_data.exec_duration = time.time() - start_exec_time
                    except Exception as e:
                        logger.warning(f"Failed to execute the action: {action}. Error: {e}")
                        action = None
        
        if action is not None:
            step_data.thought = action_thought
            step_data.action_desc = action_desc
            step_data.action_s = action_s
            step_data.action = action

            if self.reflect_on_demand:
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
            # Call Reflector
            if self.use_reflector and not skip_reflector:
                reflection_messages = self.reflector.get_message(self.episode_data)
                if self.curr_step_idx in show_step:
                    show_message(reflection_messages, "Reflector")
                response = self.vlm.predict(reflection_messages)
                try:
                    content = response.choices[0].message.content
                    logger.info("Reflection from VLM:\n%s" % content)
                    outcome, error_description = self.reflector.parse_response(content)
                    if outcome in self.reflector.valid_options:
                        logger.info("Outcome: %s" % outcome)
                        logger.info("Error Description: %s" % error_description)
                        step_data.reflection_outcome = outcome
                        step_data.reflection_error = error_description
                except Exception as e:
                    logger.warning(f"Failed to parse the reflection. Error: {e}")

            # Call NoteTaker
            if self.use_note_taker:
                note_messages = self.note_taker.get_message(self.episode_data)
                if self.curr_step_idx in show_step:
                    show_message(note_messages, "NoteTaker")
                response = self.vlm.predict(note_messages)
                try:
                    content = response.choices[0].message.content
                    logger.info("Memory from VLM:\n%s" % content)
                    memory = self.note_taker.parse_response(content)
                    logger.info("Memory: %s" % memory)
                    step_data.memory = memory
                except Exception as e:
                    logger.warning(f"Failed to parse the memory. Error: {e}")

            # Call Processor
            if self.use_processor:
                skip_processor = False
                # if self.use_reflector and step_data.reflection_outcome in ['B', 'C']:
                #     skip_processor = True
                # if self.use_long_reflector and step_data.long_reflection_outcome in ['B']:
                #     skip_processor = True
                if skip_processor:
                    if len(self.trajectory) > 1:
                        step_data.progress = self.trajectory[-2].progress
                else:
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

            # Call LongReflector
            if self.use_long_reflector:
                long_reflection_messages = self.long_reflector.get_message(self.episode_data)
                if long_reflection_messages is not None:
                    if self.curr_step_idx in [4,9]:
                        show_message(long_reflection_messages, "LongReflector")
                    response = self.vlm.predict(long_reflection_messages)
                    try:
                        content = response.choices[0].message.content
                        logger.info("Long Reflection from VLM:\n%s" % content)
                        outcome, error_description = self.long_reflector.parse_response(content)
                        if outcome in self.long_reflector.valid_options:
                            logger.info("Long Outcome: %s" % outcome)
                            logger.info("Long Error Description: %s" % error_description)
                            step_data.long_reflection_outcome = outcome
                            step_data.long_reflection_error = error_description
                    except Exception as e:
                        logger.warning(f"Failed to parse the long reflection. Error: {e}")

        if self.status == AgentStatus.FINISHED:
            # Answer
            # msg = {
            #     'type': 'text', 'text': ANSWER_PROMPT_TEMPLATE.format(goal=self.goal)
            # }
            # operator_messages[-1]['content'].append(msg)
            # show_message(operator_messages, "Answer")
            # response = self.vlm.predict(operator_messages)
            # answer_messages = self.operator.get_message(self.episode_data, is_answer=True)
            answer_messages = self.operator.get_message(self.episode_data, device_time=self.device_time, is_answer=True)
            show_message(answer_messages, "Answer")
            response = self.vlm.predict(answer_messages)
            try:
                content = response.choices[0].message.content
                logger.info("Answer from VLM:\n%s" % content)
                _, answer, _, _ = self.operator.parse_response(content, resized_size, pixels.size)
                answer = answer.parameters['text']
                step_data.answer = answer
                logger.info("Answer: %s" % answer)
            except Exception as e:
                logger.warning(f"Failed to get the answer. Error: {e}")
            
            # Evaluator
            if self.evaluate_when_finish and self.episode_data.finish_count == 1:
                evaluator_messages = self.evaluator.get_message(self.episode_data)
                show_message(evaluator_messages, "Evaluator")
                logger.info("Evaluating...")
                response = self.vlm.predict(evaluator_messages)
                result, reason, tips = None, None, None
                try:
                    content = response.choices[0].message.content
                    logger.info("Evaluation from VLM:\n%s" % content)
                    result, reason, tips = self.evaluator.parse_response(content)
                    logger.info("Evaluation Result: %s" % result)
                    logger.info("Evaluation Reason: %s" % reason)
                    logger.info("Evaluation Tips: %s" % tips)
                except Exception as e:
                    logger.warning(f"Failed to parse the evaluation. Error: {e}")
                if result is not None and 'Failed' in result:
                    logger.info("Evaluator determines that the task is not completed for the first time. Will remove the FINISH status.")
                    self.status = None
                    step_data.evaluation_result = result
                    step_data.evaluation_reason = reason
            
            # Evolutor
            if self.use_evolutor:
                evolutor_messages = self.evolutor.get_message(self.episode_data)
                show_message(evolutor_messages, "Evolutor")
                response = self.vlm.predict(evolutor_messages)
                try:
                    content = response.choices[0].message.content
                    logger.info("Updated tips from VLM:\n%s" % content)
                    updated_tips = self.evolutor.parse_response(content)
                    logger.info("Updated tips: %s" % updated_tips)
                    self.episode_data.output_tips = updated_tips
                    self.tips = updated_tips
                except Exception as e:
                    logger.warning(f"Failed to parse the updated tips. Error: {e}")

        step_data.step_duration = time.time() - start_time
        return answer


    def iter_run(self, input_content: str, stream: bool=False) -> Iterator[StepData]:
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

    def run(self, input_content: str) -> EpisodeData:
        """Execute the agent with user input content.

        Returns: EpisodeData
        """
        for _ in self.iter_run(input_content, stream=False):
            pass
        return self.episode_data
