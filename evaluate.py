import pandas as pd
import numpy as np
import re
from collections import defaultdict
from typing import List
from sklearn.metrics import confusion_matrix

def process_action_type(action_type: str):
    return action_type.lower().strip()

def process_bboxes(bboxs_str: str):
    if pd.isna(bboxs_str) or bboxs_str == '':
        return None
    bboxs = bboxs_str.split('\n')
    result = []
    for bbox in bboxs:
        if bbox == '':
            continue
        coordinates = re.findall(r'\[(\d+),(\d+)\]', bbox)
        coordinates = [(int(x), int(y)) for x, y in coordinates]
        x1, y1 = coordinates[0]
        x2, y2 = coordinates[1]
        result.append((x1, y1, x2, y2))
    if len(result) == 0:
        return None
    return result

def process_coordinate(coordinates_str: str):
    if pd.isna(coordinates_str) or coordinates_str == '':
        return None
    coordinate = eval(coordinates_str)
    return coordinate

def process_text(s):
    if pd.isna(s) or s == '':
        return None
    s = s.strip()
    return re.sub(r'[\u200B-\u200D\uFEFF]', '', s)

def process_gt_action2(action2: str):
    if pd.isna(action2):
        return 'incomplete'
    elif action2 == 'complete':
        return 'complete'
    else:
        raise ValueError(f"Unknown action2 type: {action2}")

def process_pred_action2(action2):
    if action2 == True or action2 == 'Finish':
        return 'complete'
    elif action2 == False or action2 == 'Not Finish':
        return 'incomplete'
    else:
        raise ValueError(f"Unknown action2 type: {action2}")

def is_in_box(bboxs: List[List], coordinate):
    if coordinate is None:
        return False
    if len(coordinate) == 4:
        x = (coordinate[0] + coordinate[2]) / 2
        y = (coordinate[1] + coordinate[3]) / 2
    elif len(coordinate) == 2:
        x, y = coordinate
    else:
        raise ValueError(f"Invalid coordinate format: {coordinate}")
    for bbox in bboxs:
        x1, y1, x2, y2 = bbox
        if x1 <= x <= x2 and y1 <= y <= y2:
            return True
    return False



output_path = r"E:\模拟点击调试数据\一键执行数据\result_in_one_finish_thought2.xlsx"
print(output_path)
df = pd.read_excel(output_path)

gt_types, pred_types = [], []
gt_bbox1s, pred_coordinate1s = [], []
gt_text1s, pred_text1s = [], []
gt_completes, pred_completes = [], []
tasks = defaultdict(list)
idx = 0
grouped = df.groupby(['数据日期', 'app', '屏幕尺寸', 'query_id'])
for (date, app, screen_size, query_id), group in grouped:
    for index, row in group.iterrows():
        gt_types.append(process_action_type(row['事件1']))
        pred_types.append(process_action_type(row['action1']))
        gt_bbox1s.append(process_bboxes(row['坐标1']))
        pred_coordinate1s.append(process_coordinate(row['coordinate1']))
        gt_text1s.append(process_text(row['搜索槽位（searchKeyword）']))
        pred_text1s.append(process_text(row['text1']))
        gt_completes.append(process_gt_action2(row['事件2']))
        pred_completes.append(process_pred_action2(row['action2']))

        tasks[(date, app, screen_size, query_id)].append(idx)
        idx += 1

gt_types = np.array(gt_types)
pred_types = np.array(pred_types)
type_matches = gt_types == pred_types

coordinate_matches = [True] * len(gt_bbox1s)
for i in range(len(gt_bbox1s)):
    if gt_types[i] in ['click', 'wait'] and gt_bbox1s[i] is not None:
        coordinate_matches[i] = is_in_box(gt_bbox1s[i], pred_coordinate1s[i])
coordinate_matches = np.array(coordinate_matches)

text_matches = [True] * len(gt_text1s)
for i in range(len(gt_text1s)):
    if gt_types[i] in ['type'] and gt_text1s[i] is not None:
        text_matches[i] = gt_text1s[i] == pred_text1s[i]
text_matches = np.array(text_matches)

gt_completes = np.array(gt_completes)
pred_completes = np.array(pred_completes)
complete_matches = gt_completes == pred_completes

action_matches = type_matches & coordinate_matches & text_matches
step_matches = action_matches & complete_matches

step_results = defaultdict(lambda: defaultdict(int))
for i, gt_type in enumerate(gt_types):
    step_results[gt_type][pred_types[i]] += 1
    step_results[gt_type]['total'] += 1
    if type_matches[i]:
        step_results[gt_type]['type_match'] += 1
    if coordinate_matches[i]:
        step_results[gt_type]['coordinate_match'] += 1
    if text_matches[i]:
        step_results[gt_type]['text_match'] += 1
    if action_matches[i]:
        step_results[gt_type]['action_match'] += 1
    if step_matches[i]:
        step_results[gt_type]['step_match'] += 1
    if gt_type == 'wait' and gt_bbox1s[i] is not None and type_matches[i]:
        step_results[gt_type]['wait_grounding_type_match'] += 1
        if coordinate_matches[i]:
            step_results[gt_type]['wait_grounding_coordinate_match'] += 1

# Step-level metric
for gt_type, results in step_results.items():
    print(f"Action Type: {gt_type}, Total: {results['total']}, Type Match: {results['type_match']}, Accuracy: {results['type_match'] / results['total']:.2%}")
    print(f"Action Type: {gt_type}, Pred click: {results['click']}, Pred type: {results['type']}, Pred wait: {results['wait']}, Pred abort: {results['abort']}")
    print(f"Action Type: {gt_type}, Total: {results['total']}, Action Match: {results['action_match']}, Accuracy: {results['action_match'] / results['total']:.2%}")
    print(f"Action Type: {gt_type}, Total: {results['total']}, Step Match: {results['step_match']}, Accuracy: {results['step_match'] / results['total']:.2%}")

# Action parameter metric
results = step_results['click']
print(f"Action Type: click, Type Match: {results['type_match']}, Coordinate Match: {results['coordinate_match']}, Accuracy: {results['coordinate_match'] / results['type_match']:.2%}")

results = step_results['type']
print(f"Action Type: type, Type Match: {results['type_match']}, Text Match: {results['text_match']}, Accuracy: {results['text_match'] / results['type_match']:.2%}")
# for i in range(len(gt_types)):
#     if gt_types[i] == 'type' and type_matches[i] and not text_matches[i]:
#         print(f"Row {i}: GT Text: {gt_text1s[i]}, Pred Text: {pred_text1s[i]}")

results = step_results['wait']
print(f"Action Type: wait, Wait Grounding Type Match: {results['wait_grounding_type_match']}, Wait Grounding Coordinate Match: {results['wait_grounding_coordinate_match']}, Accuracy: {results['wait_grounding_coordinate_match'] / results['wait_grounding_type_match']:.2%}")

print()

# Task-level metric
task_results = defaultdict(int)
for task, indices in tasks.items():
    task_results['total'] += 1
    task_results['task_success'] += all(step_matches[i] for i in indices)
    task_results['task_success_wo_wait_abort'] += all(step_matches[i] for i in indices if gt_types[i] not in ['wait', 'abort'])
    # num_type_actions = sum(1 for i in indices if gt_types[i] == 'type')
    # if num_type_actions != 1:
    #     print(f"Task {task} has {num_type_actions} type actions, expected 1.")
    # if gt_types[indices[-2]] != 'type' or gt_types[indices[-1]] != 'click':
    #     print(f"Task {task} does not end with type and click actions: {gt_types[indices[-2]]}, {gt_types[indices[-1]]}")

print(f"Total Tasks: {task_results['total']}, Task Success: {task_results['task_success']}, Task Success Rate: {task_results['task_success'] / task_results['total']:.2%}")
print(f"Total Tasks: {task_results['total']}, Task Success (without wait/abort): {task_results['task_success_wo_wait_abort']}, Task Success Rate (without wait/abort): {task_results['task_success_wo_wait_abort'] / task_results['total']:.2%}")

from sklearn.metrics import classification_report, confusion_matrix
print("Confusion Matrix for Complete:")
print(confusion_matrix(gt_completes, pred_completes))
print(classification_report(gt_completes, pred_completes))


"""
E:\模拟点击调试数据\一键执行数据\result_in_one_new.xlsx
Action Type: click, Total: 498, Type Match: 495, Accuracy: 99.40%
Action Type: click, Total: 498, Action Match: 443, Accuracy: 88.96%
Action Type: click, Total: 498, Step Match: 254, Accuracy: 51.00%
Action Type: type, Total: 233, Type Match: 232, Accuracy: 99.57%
Action Type: type, Total: 233, Action Match: 223, Accuracy: 95.71%
Action Type: type, Total: 233, Step Match: 223, Accuracy: 95.71%
Action Type: wait, Total: 225, Type Match: 202, Accuracy: 89.78%
Action Type: wait, Total: 225, Action Match: 196, Accuracy: 87.11%
Action Type: wait, Total: 225, Step Match: 196, Accuracy: 87.11%
Action Type: abort, Total: 16, Type Match: 0, Accuracy: 0.00%
Action Type: abort, Total: 16, Action Match: 0, Accuracy: 0.00%
Action Type: abort, Total: 16, Step Match: 0, Accuracy: 0.00%
Action Type: click, Type Match: 495, Coordinate Match: 443, Accuracy: 89.49%
Action Type: type, Type Match: 232, Text Match: 223, Accuracy: 96.12%
Action Type: wait, Wait Grounding Type Match: 155, Wait Grounding Coordinate Match: 149, Accuracy: 96.13%

Total Tasks: 230, Task Success: 20, Task Success Rate: 8.70%
Total Tasks: 230, Task Success (without wait/abort): 24, Task Success Rate (without wait/abort): 10.43%
Confusion Matrix for Complete:
[[ 30 203]
 [  2 737]]
              precision    recall  f1-score   support

    complete       0.94      0.13      0.23       233
  incomplete       0.78      1.00      0.88       739

    accuracy                           0.79       972
   macro avg       0.86      0.56      0.55       972
weighted avg       0.82      0.79      0.72       972
"""


"""
E:\模拟点击调试数据\一键执行数据\result_in_one_finish_thought_his.xlsx
Action Type: click, Total: 498, Type Match: 461, Accuracy: 92.57%
Action Type: click, Total: 498, Action Match: 408, Accuracy: 81.93%
Action Type: click, Total: 498, Step Match: 379, Accuracy: 76.10%
Action Type: type, Total: 233, Type Match: 202, Accuracy: 86.70%
Action Type: type, Total: 233, Action Match: 193, Accuracy: 82.83%
Action Type: type, Total: 233, Step Match: 192, Accuracy: 82.40%
Action Type: wait, Total: 225, Type Match: 194, Accuracy: 86.22%
Action Type: wait, Total: 225, Action Match: 186, Accuracy: 82.67%
Action Type: wait, Total: 225, Step Match: 186, Accuracy: 82.67%
Action Type: abort, Total: 16, Type Match: 0, Accuracy: 0.00%
Action Type: abort, Total: 16, Action Match: 0, Accuracy: 0.00%
Action Type: abort, Total: 16, Step Match: 0, Accuracy: 0.00%
Action Type: click, Type Match: 461, Coordinate Match: 408, Accuracy: 88.50%
Action Type: type, Type Match: 202, Text Match: 193, Accuracy: 95.54%
Row 27: GT Text: 会员DIY, Pred Text: 用中国移动搜索会员DIY
Row 116: GT Text: pdf, Pred Text: PDF
Row 236: GT Text: 学好物理, Pred Text: 用作业帮搜索学好物理
Row 387: GT Text: 使用量分析, Pred Text: 用中国电信搜索使用量分析
Row 392: GT Text: 彩铃限时福利, Pred Text: 用中国电信搜索彩铃限时福利
Row 433: GT Text: 视频权益免费领, Pred Text: 用中国联通搜索视频权益免费领
Row 711: GT Text: 时停五百年, Pred Text: 停五百年
Row 789: GT Text: 课程, Pred Text: 课程名称
Row 930: GT Text: flash教学课件, Pred Text: 用腾讯文档搜索flash教学课件
Action Type: wait, Wait Grounding Type Match: 148, Wait Grounding Coordinate Match: 140, Accuracy: 94.59%

Total Tasks: 230, Task Success: 101, Task Success Rate: 43.91%
Total Tasks: 230, Task Success (without wait/abort): 132, Task Success Rate (without wait/abort): 57.39%
Confusion Matrix for Complete:
[[197  36]
 [ 28 711]]
              precision    recall  f1-score   support

    complete       0.88      0.85      0.86       233
  incomplete       0.95      0.96      0.96       739

    accuracy                           0.93       972
   macro avg       0.91      0.90      0.91       972
weighted avg       0.93      0.93      0.93       972
"""


"""
E:\模拟点击调试数据\一键执行数据\result_in_one_finish_thought.xlsx
Action Type: click, Total: 498, Type Match: 494, Accuracy: 99.20%
Action Type: click, Total: 498, Action Match: 431, Accuracy: 86.55%
Action Type: click, Total: 498, Step Match: 396, Accuracy: 79.52%
Action Type: type, Total: 233, Type Match: 231, Accuracy: 99.14%
Action Type: type, Total: 233, Action Match: 219, Accuracy: 93.99%
Action Type: type, Total: 233, Step Match: 219, Accuracy: 93.99%
Action Type: wait, Total: 225, Type Match: 197, Accuracy: 87.56%
Action Type: wait, Total: 225, Action Match: 192, Accuracy: 85.33%
Action Type: wait, Total: 225, Step Match: 192, Accuracy: 85.33%
Action Type: abort, Total: 16, Type Match: 0, Accuracy: 0.00%
Action Type: abort, Total: 16, Action Match: 0, Accuracy: 0.00%
Action Type: abort, Total: 16, Step Match: 0, Accuracy: 0.00%
Action Type: click, Type Match: 494, Coordinate Match: 431, Accuracy: 87.25%
Action Type: type, Type Match: 231, Text Match: 219, Accuracy: 94.81%
Action Type: wait, Wait Grounding Type Match: 151, Wait Grounding Coordinate Match: 146, Accuracy: 96.69%

Total Tasks: 230, Task Success: 109, Task Success Rate: 47.39%
Total Tasks: 230, Task Success (without wait/abort): 138, Task Success Rate (without wait/abort): 60.00%
Confusion Matrix for Complete:
[[190  43]
 [  6 733]]
              precision    recall  f1-score   support

    complete       0.97      0.82      0.89       233
  incomplete       0.94      0.99      0.97       739

    accuracy                           0.95       972
   macro avg       0.96      0.90      0.93       972
weighted avg       0.95      0.95      0.95       972
"""


"""
E:\模拟点击调试数据\一键执行数据\result_in_one_finish_thought2.xlsx
Action Type: click, Total: 498, Type Match: 495, Accuracy: 99.40%
Action Type: click, Pred click: 495, Pred type: 3, Pred wait: 0, Pred abort: 0
Action Type: click, Total: 498, Action Match: 426, Accuracy: 85.54%
Action Type: click, Total: 498, Step Match: 314, Accuracy: 63.05%
Action Type: type, Total: 233, Type Match: 231, Accuracy: 99.14%
Action Type: type, Pred click: 2, Pred type: 231, Pred wait: 0, Pred abort: 0
Action Type: type, Total: 233, Action Match: 223, Accuracy: 95.71%
Action Type: type, Total: 233, Step Match: 222, Accuracy: 95.28%
Action Type: wait, Total: 225, Type Match: 190, Accuracy: 84.44%
Action Type: wait, Pred click: 35, Pred type: 0, Pred wait: 190, Pred abort: 0
Action Type: wait, Total: 225, Action Match: 184, Accuracy: 81.78%
Action Type: wait, Total: 225, Step Match: 184, Accuracy: 81.78%
Action Type: abort, Total: 16, Type Match: 5, Accuracy: 31.25%
Action Type: abort, Pred click: 8, Pred type: 0, Pred wait: 3, Pred abort: 5
Action Type: abort, Total: 16, Action Match: 5, Accuracy: 31.25%
Action Type: abort, Total: 16, Step Match: 5, Accuracy: 31.25%
Action Type: click, Type Match: 495, Coordinate Match: 426, Accuracy: 86.06%
Action Type: type, Type Match: 231, Text Match: 223, Accuracy: 96.54%
Action Type: wait, Wait Grounding Type Match: 143, Wait Grounding Coordinate Match: 137, Accuracy: 95.80%

Total Tasks: 230, Task Success: 60, Task Success Rate: 26.09%
Total Tasks: 230, Task Success (without wait/abort): 74, Task Success Rate (without wait/abort): 32.17%
Confusion Matrix for Complete:
[[102 131]
 [  5 734]]
              precision    recall  f1-score   support

    complete       0.95      0.44      0.60       233
  incomplete       0.85      0.99      0.92       739

    accuracy                           0.86       972
   macro avg       0.90      0.72      0.76       972
weighted avg       0.87      0.86      0.84       972
"""