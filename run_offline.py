"""
安装：
git clone https://github.com/MadeAgents/mobile-use.git
cd mobile-use
git checkout test
pip install -e .

说明：
run_offline.py 提供了一个离线运行的例子，可以用来测试模拟点击的效果；
Action_map 函数对输出的action进行了调整，如需进一步调整可以按需修改；
当前的agent是一个MultiAgentOffline，可以根据需要选择其他的agent；
当前agent对于广告页的action预测效果不是特别好，容易点击跳过或不输出跳过广告的坐标。
"""

import os
from PIL import Image
from typing import Dict
import logging

from mobile_use import VLMWrapper, Action
from mobile_use import MultiAgentOffline


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()


def Action_map(action: Action) -> Dict:
    """
    map the MobileUse Action space to the test action space
    Returns: 
    Dict = {
        "action_type": str,
        "args": Dict = {
            "coordinates": {"startPoint": {"x": integer,"y": integer},"endPoint": {"x": integer,"y": integer}},
            "text": str,
            "duration": integer,
            "autoClosable": boolean,
        }
    }
    """
    result = {}
    try:
        if action.name in ['click', 'left_click']:
            x, y = action.parameters['coordinate']
            result['action_type'] = 'Click'
            result['args'] = {"coordinates": {"startPoint": {"x": x, "y": y}, "endPoint": {"x": x, "y": y}}}
        elif action.name == 'long_press':
            x, y = action.parameters['coordinate']
            duration = action.parameters['time'] * 1000
            result['action_type'] = 'LONGPRESS'
            result['args'] = {"coordinates": {"startPoint": {"x": x, "y": y}, "endPoint": {"x": x, "y": y}}, "duration": duration}
        elif action.name == 'swipe':
            x1, y1 = action.parameters['coordinate']
            x2, y2 = action.parameters['coordinate2']
            result['action_type'] = 'SWIPE'
            result['args'] = {"coordinates": {"startPoint": {"x": x1, "y": y1}, "endPoint": {"x": x2, "y": y2}}}
        elif action.name == 'type':
            text = action.parameters['text']
            result['action_type'] = 'Type'
            result['args'] = {"text": text}
        elif action.name == 'system_button':
            button = action.parameters['button']
            if button == "Back":
                result['action_type'] = 'BACK'
                result['args'] = {}
            elif button == "Home":
                result['action_type'] = 'HOME'
                result['args'] = {}
            else:
                result['action_type'] = 'Abort'
                result['args']['abortReason'] = "Unknown system button."
        elif action.name == 'open':
            text = action.parameters['text']
            result['action_type'] = 'Awake'
            result['args'] = {"text": text}
        elif action.name == 'wait':
            duration = action.parameters['time'] * 1000
            result['action_type'] = 'Wait'
            result['args'] = {"duration": duration}
            if 'coordinate' in action.parameters:
                x, y = action.parameters['coordinate']
                result['args']['autoClosable'] = True
                result['args']['coordinates'] = {"startPoint": {"x": x, "y": y}, "endPoint": {"x": x, "y": y}}
            else:
                result['args']['autoClosable'] = False
        elif action.name == 'call_user':
            text = action.parameters['text']
            result['action_type'] = 'Pop'
            result['args'] = {"text": text}
        elif action.name == 'terminate':
            status = action.parameters['status']
            if status == 'success':
                result['action_type'] = 'Complete'
                result['args'] = {}
            else:
                result['action_type'] = 'Abort'
                result['args']['abortReason'] = "Agent thinks the task is failed."
        else:
            result['action_type'] = 'Abort'
            result['args']['abortReason'] = "Unknown action name."
    except Exception as e:
        result['action_type'] = 'Abort'
        result['args']['abortReason'] = f"Failed to parse the action. Error: {e}"
    return result
        


VLM_API_KEY="EMPTY"
VLM_BASE_URL="http://hammer-llm.oppo.test/v1/"


vlm = VLMWrapper(
            model_name="qwen2.5-vl-72b-instruct", 
            api_key=VLM_API_KEY,
            base_url=VLM_BASE_URL,
            max_tokens=1024,
            max_retry=1,
            temperature=0.0
        )
agent = MultiAgentOffline(
    env=None,
    vlm=vlm,
    use_planner=False,
    use_reflector=False,
    use_note_taker=False,
    use_processor=False,
)

agent.reset()
screenshots = []

goal = "用网易有道词典搜考研英语一"
img_dir = "/home/notebook/data/personal/S9057346/test-mobile/模拟点击调试数据/网易有道词典-query15"
img_files = ['网易有道词典-query15-01.jpg', '网易有道词典-query15-02.jpg', '网易有道词典-query15-03.jpg', '网易有道词典-query15-03-type.jpg']

for img_file in img_files:
    img_path = os.path.join(img_dir, img_file)
    img = Image.open(img_path)
    screenshots.append(img)

actions = agent.run(goal, screenshots=screenshots)
print(actions)

actions = [Action_map(action) for action in actions]
print(actions)
