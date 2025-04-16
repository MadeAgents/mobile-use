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
import pickle
import logging
import pandas as pd
from PIL import Image
from typing import Dict
from dotenv import load_dotenv

from mobile_use import VLMWrapper, Action
from mobile_use import MultiAgentOffline


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def save(data, path):
    with open(path, "wb") as f:
        pickle.dump(data, f)
    
def load(path):
    with open(path, "rb") as f:
        loaded = pickle.load(f)
    return loaded

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
            result['action_type'] = 'click'
            result['args'] = {"coordinates": {"startPoint": {"x": x, "y": y}, "endPoint": {"x": x, "y": y}}}
        elif action.name == 'long_press':
            x, y = action.parameters['coordinate']
            duration = action.parameters['time'] * 1000
            result['action_type'] = 'longpress'
            result['args'] = {"coordinates": {"startPoint": {"x": x, "y": y}, "endPoint": {"x": x, "y": y}}, "duration": duration}
        elif action.name == 'swipe':
            x1, y1 = action.parameters['coordinate']
            x2, y2 = action.parameters['coordinate2']
            result['action_type'] = 'swipe'
            result['args'] = {"coordinates": {"startPoint": {"x": x1, "y": y1}, "endPoint": {"x": x2, "y": y2}}}
        elif action.name == 'type':
            text = action.parameters['text']
            result['action_type'] = 'type'
            result['args'] = {"text": text}
        elif action.name == 'system_button':
            button = action.parameters['button']
            if button == "Back":
                result['action_type'] = 'back'
                result['args'] = {}
            elif button == "Home":
                result['action_type'] = 'home'
                result['args'] = {}
            else:
                result['action_type'] = 'abort'
                result['args']['abortReason'] = "Unknown system button."
        elif action.name == 'open':
            text = action.parameters['text']
            result['action_type'] = 'awake'
            result['args'] = {"text": text}
        elif action.name == 'wait':
            duration = action.parameters['time'] * 1000
            result['action_type'] = 'wait'
            result['args'] = {"duration": duration}
            if 'coordinate' in action.parameters:
                x, y = action.parameters['coordinate']
                result['args']['autoClosable'] = True
                result['args']['coordinates'] = {"startPoint": {"x": x, "y": y}, "endPoint": {"x": x, "y": y}}
            else:
                result['args']['autoClosable'] = False
        elif action.name == 'call_user':
            text = action.parameters['text']
            result['action_type'] = 'pop'
            result['args'] = {"text": text}
        elif action.name == 'terminate':
            status = action.parameters['status']
            if status == 'success':
                result['action_type'] = 'complete'
                result['args'] = {}
            else:
                result['action_type'] = 'abort'
                result['args']['abortReason'] = "Agent thinks the task is failed."
        else:
            result['action_type'] = 'abort'
            result['args']['abortReason'] = "Unknown action name."
    except Exception as e:
        result['action_type'] = 'abort'
        result['args']['abortReason'] = f"Failed to parse the action. Error: {e}"
    return result


load_dotenv()
print(os.getenv('VLM_API_KEY'))
print(os.getenv('VLM_BASE_URL'))
vlm = VLMWrapper(
            model_name="qwen2.5-vl-72b-instruct", 
            api_key=os.getenv('VLM_API_KEY'),
            base_url=os.getenv('VLM_BASE_URL'),
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
    use_evaluator=True
)

excel_path = r"E:\模拟点击调试数据\4月11日数据.xlsx"
data_path = r"E:\模拟点击调试数据\0411模拟点击数据"
log_path = r"E:\模拟点击调试数据\0411模拟点击数据输出"
output_path = r"E:\模拟点击调试数据\result.xlsx"

df = pd.read_excel(excel_path)
print(df.columns)
grouped = df.groupby(['app', '屏幕尺寸', 'query_id'])
for (app, screen_size, query_id), group in grouped:
    try:
        logger.info(f"Processing query {(app, screen_size, query_id)}...")
        group = group.reset_index(drop=True)

        query = group.iloc[0]['query']

        log_dir = os.path.join(log_path, app, screen_size)
        log_file = os.path.join(log_dir, f"{query_id}.pkl")
        if os.path.exists(log_file):
            logger.info(f"Task {(app, screen_size, query_id)} already exists, skipping...")
            continue

        agent.reset(query)
        max_step = len(group)
        imgs = []

        for index, row in group.iterrows():
            img_name = row['截图(名称）']
            img_path = os.path.join(data_path, app, screen_size, query_id, img_name)
            img = Image.open(img_path)
            imgs.append(img)
        
        actions = agent.run(query, screenshots=imgs)
        logger.info(f"Actions: {actions}")
        maped_actions = [Action_map(action[0]) for action in actions]
        logger.info(f"Maped Actions: {maped_actions}")

    except Exception as e:
        logger.error(f"Error processing query {query_id}: {e}")
        continue

    # Save the episode
    episode_data = agent.episode_data
    os.makedirs(log_dir, exist_ok=True)
    save(episode_data, log_file)

    try:
        # Save the result to a new excel file
        try:
            existing_df = pd.read_excel(output_path, sheet_name='Sheet1', engine='openpyxl')
        except (ValueError, FileNotFoundError):
            existing_df = pd.DataFrame()
        updated_rows = []
        for index, row in group.iterrows():
            new_data = row.to_dict()
            new_data.update({'action': actions[index][0], 'action2': actions[index][1], 'maped_action': maped_actions[index]})
            updated_rows.append(new_data)
        updated_df = pd.concat([existing_df, pd.DataFrame(updated_rows)], ignore_index=True)
        with pd.ExcelWriter(output_path, mode='w', engine='openpyxl') as writer:
            updated_df.to_excel(writer, sheet_name='Sheet1', index=False)

    except Exception as e:
        logger.error(f"Error saving query {query_id} to excel: {e}")
        continue
