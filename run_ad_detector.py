

import os
import pickle
import logging
import pandas as pd
from PIL import Image
from typing import Dict
from dotenv import load_dotenv

from mobile_use import VLMWrapper, Action
from mobile_use import MultiAgentOffline
from mobile_use.agents.sub_agent import AdvertisementDetector


logging.basicConfig(level=logging.INFO, format='%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s')
logger = logging.getLogger()

def save(data, path):
    with open(path, "wb") as f:
        pickle.dump(data, f)
    
def load(path):
    with open(path, "rb") as f:
        loaded = pickle.load(f)
    return loaded

excel_path = r"E:\模拟点击调试数据\一键执行数据\一键执行-步骤合并数据集-1.24.xlsx"
data_path = r"E:\模拟点击调试数据\一键执行数据"
output_path = r"E:\模拟点击调试数据\一键执行数据\ad_detector_result_with_thought.xlsx"

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
    use_evaluator=False,
    use_operator=False,
    use_advertisement_detector=True,
)

df = pd.read_excel(excel_path, dtype=str)
print(df.columns)
grouped = df.groupby(['数据日期', 'app', '屏幕尺寸', 'query_id'])
for (date, app, screen_size, query_id), group in grouped:
    try:
        logger.info(f"Processing query {(date, app, screen_size, query_id)}...")
        group = group.reset_index(drop=True)

        query = group.iloc[0]['query']

        agent.reset(query)
        imgs = []

        for index, row in group.iterrows():
            img_name = row['截图(名称）']
            img_path = os.path.join(data_path, f"{date}一键执行Agent工具采集数据", "Agent工具采集数据", app, screen_size, query_id, img_name)
            img = Image.open(img_path)
            imgs.append(img)
        
        ad_detector_thoughts, ad_detector_action_names, ad_detector_bboxes = agent.run_ad_detector(query, screenshots=imgs)
        logger.info(f"Advertisement Detector Thoughts: {ad_detector_thoughts}")
        logger.info(f"Advertisement Detector Action Names: {ad_detector_action_names}")
        logger.info(f"Advertisement Detector BBoxes: {ad_detector_bboxes}")

    except Exception as e:
        logger.error(f"Error processing query {query_id}: {e}")
        continue

    try:
        # Save the result to a new excel file
        try:
            existing_df = pd.read_excel(output_path, sheet_name='Sheet1', engine='openpyxl')
        except (ValueError, FileNotFoundError):
            existing_df = pd.DataFrame()
        updated_rows = []
        for index, row in group.iterrows():
            new_data = row.to_dict()
            new_data.update({'ad_detector_thought': ad_detector_thoughts[index],
                             'ad_detector_action_name': ad_detector_action_names[index], 
                             'ad_detector_bbox': ad_detector_bboxes[index]})
            updated_rows.append(new_data)
        updated_df = pd.concat([existing_df, pd.DataFrame(updated_rows)], ignore_index=True)
        with pd.ExcelWriter(output_path, mode='w', engine='openpyxl') as writer:
            updated_df.to_excel(writer, sheet_name='Sheet1', index=False)

    except Exception as e:
        logger.error(f"Error saving query {query_id} to excel: {e}")
        continue
