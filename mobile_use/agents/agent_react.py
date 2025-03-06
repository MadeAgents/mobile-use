import re
import logging
from typing import Iterator

from mobile_use.action import ACTION_SPACE
from mobile_use.scheme import *
from mobile_use.environ import Environment
from mobile_use.vlm import VLMWrapper
from mobile_use.utils import encode_image_url, contains_chinese, smart_resize
from mobile_use.agents import Agent


logger = logging.getLogger(__name__)


SYSTEM_PROMPT_EN = """
You are using a Mobile device. You are able to use a Action Space Operator to interact with the mobile based on the given task and screenshot.

## Action Space
Your available "Next Action" only include:
- click(point=[x,y]): Click on the coordinate point specified on the screen (x,y).
- long_press(point=[x,y]): Long press the screen to specify coordinates (x,y).
- type(text='hello world'): Types a string of text.
- scroll(start_point=[x1,y1], end_point=[x2,y2]): Scroll the screen, (x1,y1) is the starting coordinate position, (x2,y2) is the end coordinate position. In particular, when y1=y2, you can swipe left and right on the desktop to switch pages, which is very helpful for finding a specific application.
- press_home(): Back to Home page.
- press_back(): Back to previous page.
- finished(answer=''): Submit the task regardless of whether it succeeds or fails. The answer parameter is to summarize the content of the reply to the user.
- call_user(question=''): Submit the task and call the user when the task is unsolvable, or when you need the user's help.
- wait(): Wait for loading to complete.

## Note
- Action click, long_press and scroll must contain coordinates within.
- You may be given some history plan and actions, this is the response from the previous loop.
- You should carefully consider your plan base on the task, screenshot, and history actions.
- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.

## Suggestions
- If you need to open an APP, when the home page is not available, you can scroll down to the search page to find the corresponding APP.
- When the screen of the previous operation is not responsive, you need to avoid performing the same action in the next step.
- Shopping or life services apps, you should make use of the in-app search function as much as possible to find quickly.
- Reduce the execution steps as much as possible, and find the optimal execution path to achieve the task goal.

## Format
Task: The task description.
Observation: The mobile screenshot or user response.
Thought: The process of thinking.
Action: The next action. Must be one of the Action Space.

**Be aware that Observation, Thought, and Action will be repeated.**

Now, let's begin!
""".strip()


SYSTEM_PROMPT_ZH = """
你正在使用移动设备。你可以根据给定的任务和截图使用给定的操作动作与移动设备进行交互。

## 操作动作
你可以使用一下任一动作作为下一步的操作：
- click(point=[x,y]): 点击屏幕上指定的坐标点(x,y)
- long_press(point=[x,y]): 长按屏幕指定的坐标点(x,y)
- type(text='hello world'): 输入一段文本字符串
- scroll(start_point=[x1,y1], end_point=[x2,y2]): 滚动屏幕，(x1,y1) 为起始坐标位置，(x2,y2) 为结束坐标位置。特别地，当y1=y2时可以实现桌面左右滑动切换页面，这对寻找指定应用非常有帮助。
- press_home(): 回到主页
- press_back(): 返回上一页
- finished(answer=''): 任务结束标志， 无论任务成功与否，都需要通过该操作结束，answer 参数值则为总结回复用户的内容
- call_user(question=''): 当任务无法解决或需要用户帮助时，向用户发起帮助提问，例如需要输入登录相关信息
- wait():等待加载完成

## 注意事项
- 操作click、long_press和scroll必须包含其中的坐标。
- 可能会提供一些历史计划和操作，这是前一个循环的响应。
- 您应基于任务、截图和历史操作仔细考虑您的计划。
- 编写一个小计划，并最终将您下一个行动（及其目标元素）用一句话总结在“Thought”部分中。

## 操作建议
- 如果你需要打开某个APP应用，当主页没有时，你可以通过下滑屏幕进入搜索页查找相应的APP
- 当前前后操作的屏幕无响应时，你需要避免在下一步再执行相同的操作
- 购物或者生活服务类APP，你应该尽可能利用APP内的搜索功能从而能够快速找到
- 尽可能减少执行步骤，寻找最优的执行路径已达成任务目标

## 格式说明
Task: 任务描述。
Observation: 移动设备截屏或用户响应。
Thought: 思考或总结的过程，思考是一定要结合历史的操作。
Action: 下一步操作。必须是操作动作中的一个。
Finally: 如果任务已完成，输出最终回复结果

**请注意 Observation Thought 和 Action 将会重复出现，直至任务结束**

现在，让我们开始吧！
""".strip()


IMAGE_PLACEHOLDER = '<|vision_start|><|image_pad|><|vision_end|>'


def parse_reason_and_action(content: str, size: tuple[float, float], raw_size: tuple[float, float]) -> Action:
    reason = re.search(r'Thought:(.*)Action:', content, flags=re.DOTALL)
    if reason:
        reason_s = reason.group(1).strip()
    else:
        reason_s = None
    
    action_name = '|'.join(ACTION_SPACE.keys())
    search_res = re.search(fr'Action: *({action_name})\((.*)\)', content, flags=re.DOTALL)

    if not search_res:
        raise Exception("Action is undefined")

    name = search_res.group(1).strip()
    params = eval(f"dict({search_res.group(2)})")

    for k, v in params.items():
        if ACTION_SPACE[name].get('parameters', {}).get(k, {}).get('type') == 'array':
            try:
                x = round(v[0] / size[0] * raw_size[0])
                y = round(v[1] / size[1] * raw_size[1])
                params[k] = (x, y)
            except:
                pass
    action_a = Action(name=name, parameters=params)
    action_r = f'{name}({search_res.group(2)})'     # raw action
    return reason_s, action_a, action_r


@Agent.register('ReAct')
class ReActAgent(Agent):
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

    def _remain_most_recent_images(self):
        couter = 0
        for i in range(len(self.messages)-1, -1, -1):
            message = self.messages[i]
            if isinstance(message['content'], list):
                j = len(message['content']) - 1
                while j >= 0:
                    cnt = message['content'][j]
                    if cnt['type'] == 'image_url':
                        if couter >= self.num_latest_screenshot:
                            message['content'].pop(j)
                            message['content'][j-1]['text'] = message['content'][j-1]['text'].replace(IMAGE_PLACEHOLDER, 'None')
                        else:
                            couter += 1
                    j -= 1

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
            system_prompt = SYSTEM_PROMPT_ZH if contains_chinese(self.goal) else SYSTEM_PROMPT_EN
            logger.info(f"system_prompt:\n{system_prompt}")
            self.messages.append({
                'role': 'system', 
                'content': system_prompt
            })
            self.messages.append({
                'role': 'user', 
                'content': [
                    {
                        'type': 'text',
                        'text': f'Task: {self.goal}'
                    }
                ]
            })
        if self.state == AgentState.CALLUSER:
            observation = self._user_input
            img_msg = None
        else:
            # Fixed Picture sequence inconsistency problem in vllm0.7.2 
            # and Compatible QwenAPI error: '<400> InternalError.Algo.InvalidParameter: Invalid text: <|image_pad|>'
            observation = '' if 'dashscope.aliyuncs.com' in str(self.vlm.client.base_url) else IMAGE_PLACEHOLDER

            # Get the current environment screen
            env_state = self.env.get_state()
            pixels = env_state.pixels.copy()
            pixels.thumbnail((1024, 1024))
            h, w = smart_resize(height=pixels.height, width=pixels.width)
            pixels = pixels.resize([w, h])
            img_msg = {
                "type": "image_url",
                "image_url": {"url": encode_image_url(pixels)}
            }
            # Add new step data
            self.trajectory.append(StepData(
                step_idx=self.curr_step_idx,
                curr_env_state=env_state,
                vlm_call_history=[]
            ))
        self.messages[-1]['content'].append({
            'type': 'text',
            'text': f'Observation: {observation}'
        })
        if img_msg:
            self.messages[-1]['content'].append(img_msg)

        step_data = self.trajectory[-1]

        self._remain_most_recent_images()
        
        response = self.vlm.predict(self.messages, stop=['Observation'])

        counter = self.max_reflection_action
        reason, action = None, None
        while counter > 0:
            try:
                content = response.choices[0].message.content
                step_data.content = content
                logger.info("Content from VLM:\n%s" % step_data.content)
                step_data.vlm_call_history.append(VLMCallingData(self.messages, response))
                reason, action, action_r = parse_reason_and_action(content, pixels.size, env_state.pixels.size)
                logger.info("REASON: %s" % reason)
                logger.info("ACTION: %s" % str(action))
                self.messages[-1]['content'].append({
                    'type': 'text',
                    'text': f'Thought: {reason}\nAction: {action_r}'
                })
                break
            except Exception as e:
                logger.warning(f"Failed to parse the action from: {content}.")
                msg = {
                    'type': 'text', 
                    'text': f"Failed to parse the action from: {content}.Error is {e.args}"
                }
                self.messages[-1]['content'].append(msg)
                self._remain_most_recent_images()
                response = self.vlm.predict(self.messages, stop=['Observation'])
                counter -= 1
        if action is None:
            raise Exception("Action parse error after max retry")

        step_data.action = action
        step_data.thought = reason

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

        return step_data

    def iter_run(self, input_content: str) -> Iterator[StepData]:
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
