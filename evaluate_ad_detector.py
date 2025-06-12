import pandas as pd
import numpy as np
import re
from sklearn.metrics import classification_report, confusion_matrix

def convert_bbox_to_coordinates(bboxs_str: str):
    bboxs = bboxs_str.split('\n')
    reult = None
    for bbox in bboxs:
        if bbox == '':
            continue
        coordinates = re.findall(r'\[(\d+),(\d+)\]', bbox)
        coordinates = [(int(x), int(y)) for x, y in coordinates]
        x1, y1 = coordinates[0]
        x2, y2 = coordinates[1]
        if reult is not None:
            raise ValueError("每个框只能包含一个坐标对")
        reult = [x1, y1, x2, y2]
    return reult


def compute_iou(box1, box2):
    """
    计算两个矩形框的 IoU
    box1, box2: tuple 或 list，格式为 (x1, y1, x2, y2)
    """

    if len(box1) != 4 or len(box2) != 4:
        raise ValueError("每个框必须是一个包含四个元素的列表或元组，格式为 (x1, y1, x2, y2)")
    if box1[2] < box1[0] or box1[3] < box1[1] or box2[2] < box2[0] or box2[3] < box2[1]:
        raise ValueError("框的坐标不合法，右下角坐标必须大于左上角坐标")

    # 获取交集的坐标
    x_left   = max(box1[0], box2[0])
    y_top    = max(box1[1], box2[1])
    x_right  = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    # 计算交集面积
    if x_right < x_left or y_bottom < y_top:
        return 0.0  # 没有重叠

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # 计算两个框的面积
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # 计算并集面积
    union_area = box1_area + box2_area - intersection_area

    # 计算 IoU
    iou = intersection_area / union_area
    return iou


def compare_bbox(box1, box2):
    if len(box1) != 4 or len(box2) != 4:
        raise ValueError("每个框必须是一个包含四个元素的列表或元组，格式为 (x1, y1, x2, y2)")
    if box1[2] < box1[0] or box1[3] < box1[1] or box2[2] < box2[0] or box2[3] < box2[1]:
        raise ValueError("框的坐标不合法，右下角坐标必须大于左上角坐标")
    width = box1[2] - box1[0]
    height = box1[3] - box1[1]
    # box1[0] = max(0, box1[0]-width*0.1)
    # box1[1] = max(0, box1[1]-height*0.1)
    # box1[0] = box1[0]+width*0.4
    # box1[1] = box1[1]-height*0.1
    # box1[2] = box1[2]-width*0.4
    # box1[3] = box1[3]+height*0.1
    x = (box2[0] + box2[2]) / 2
    y = (box2[1] + box2[3]) / 2
    # print(box1)
    # print(box2)
    # print(x,y)
    # print()
    if box1[0] <= x <= box1[2] and box1[1] <= y <= box1[3]:
        return True
    return False


output_path = r"E:\模拟点击调试数据\一键执行数据\ad_detector_result_with_thought.xlsx"
df = pd.read_excel(output_path)
new_df = pd.DataFrame()

grouped = df.groupby(['数据日期', 'app', '屏幕尺寸', 'query_id'])
gt_action_types, gt_bboxes, pred_action_types, pred_bboxes = [], [], [], []

for (date, app, screen_size, query_id), group in grouped:
    for index, row in group.iterrows():
        gt_action_type = row['事件1']
        gt_bbox = row['坐标1']
        pred_action_type = row['ad_detector_action_name']
        pred_bbox = row['ad_detector_bbox']
        gt_action_types.append(gt_action_type)
        gt_bboxes.append(gt_bbox)
        pred_action_types.append(pred_action_type)
        pred_bboxes.append(pred_bbox)
print("gt_action_types:", np.unique(gt_action_types))
print("pred_action_types:", np.unique(pred_action_types))

gt_action_types = ['wait' if action == 'wait' else 'abort' if action == 'Abort' else 'other' for action in gt_action_types]
print(classification_report(gt_action_types, pred_action_types))
print(confusion_matrix(gt_action_types, pred_action_types))

gt_action_types_is_wait = [action in ['wait'] for action in gt_action_types]
pred_action_types_is_wait = [action in ['wait'] for action in pred_action_types]
print("Confusion Matrix for 'wait':")
print(confusion_matrix(gt_action_types_is_wait, pred_action_types_is_wait))
# print(classification_report(gt_action_types_is_wait, pred_action_types_is_wait))

gt_action_types_is_abort = [action in ['abort'] for action in gt_action_types]
pred_action_types_is_abort = [action in ['abort'] for action in pred_action_types]
print("Confusion Matrix for 'Abort':")
print(confusion_matrix(gt_action_types_is_abort, pred_action_types_is_abort))

# gt_action_types_wo_abort, pred_action_types_wo_abort = [], []
# for gt_action, pred_action in zip(gt_action_types, pred_action_types):
#     if gt_action != 'Abort':
#         gt_action_types_wo_abort.append(gt_action)
#         pred_action_types_wo_abort.append(pred_action)
# gt_action_types_wo_abort_is_wait = [action in ['wait'] for action in gt_action_types_wo_abort]
# pred_action_types_wo_abort_is_wait = [action in ['wait'] for action in pred_action_types_wo_abort]
# print(confusion_matrix(gt_action_types_wo_abort_is_wait, pred_action_types_wo_abort_is_wait))
# print(classification_report(gt_action_types_wo_abort_is_wait, pred_action_types_wo_abort_is_wait))

gt_bboxes = [gt_bboxes[i] if gt_action_types[i] == 'wait' else None for i in range(len(gt_bboxes))]
gt_bboxes = [None if pd.isna(bbox) else bbox for bbox in gt_bboxes]
gt_bboxes = [convert_bbox_to_coordinates(bbox) if bbox else None for bbox in gt_bboxes]
pred_bboxes = [eval(bbox) if not pd.isna(bbox) else None for bbox in pred_bboxes]
print(len(gt_bboxes), len(pred_bboxes))
ious = []
practice_metrics = []
for gt_bbox, pred_bbox in zip(gt_bboxes, pred_bboxes):
    if gt_bbox is not None and pred_bbox is not None:
        iou = compute_iou(gt_bbox, pred_bbox)
        ious.append(iou)
        practice_metrics.append(compare_bbox(gt_bbox, pred_bbox))
print(len([bbox for bbox in gt_bboxes if bbox is not None]), len([bbox for bbox in pred_bboxes if bbox is not None]), len(ious), len(practice_metrics))
print(f"Average IoU: {np.mean(ious):.4f}")
print(f"Practice Metrics: {np.mean(practice_metrics):.4f}")
