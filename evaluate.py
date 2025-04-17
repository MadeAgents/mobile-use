import pandas as pd
import re

def is_in_box(bboxs_str: str, x, y):
    bboxs = bboxs_str.split('\n')
    for bbox in bboxs:
        if bbox == '':
            continue
        coordinates = re.findall(r'\[(\d+),(\d+)\]', bbox)
        coordinates = [(int(x), int(y)) for x, y in coordinates]
        x1, y1 = coordinates[0]
        x2, y2 = coordinates[1]
        if x1 <= x <= x2 and y1 <= y <= y2:
            return True
    return False

def compare_action2(a, b):
    if a == 'complete':
        return b == True
    else:
        return b == False

output_path = r"E:\模拟点击调试数据\result.xlsx"
df = pd.read_excel(output_path)
new_df = pd.DataFrame()

grouped = df.groupby(['app', '屏幕尺寸', 'query_id'])
task_count, task_success_count = 0, 0
no_wait_abort_task_count, no_wait_abort_task_success_count = 0, 0
click_count, click_success_count = 0, 0
wait_count, wait_success_count = 0, 0
type_count, type_success_count = 0, 0
abort_count, abort_success_count = 0, 0
grounding_count, grounding_success_count = 0, 0
text_count, text_success_count = 0, 0

for (app, screen_size, query_id), group in grouped:
    # if screen_size == "2268x2440":
    #     continue
    task_success = True
    is_no_wait_abort_task = True
    for index, row in group.iterrows():
        step_success = False
        pred = eval(row['maped_action'])
        gt_action_type = row['事件1']
        pred_action_type = pred['action_type']
        gt_action2 = row['事件2']
        pred_action2 = row['action2']
        if gt_action_type == 'click':
            click_count += 1
            if pred_action_type == gt_action_type:
                grounding_count += 1
                bboxs_str = row['坐标1']
                x = pred['args']['coordinates']['startPoint']['x']
                y = pred['args']['coordinates']['startPoint']['y']
                if is_in_box(bboxs_str, x, y):
                    grounding_success_count += 1
                    is_action2_equal = compare_action2(gt_action2, pred_action2)
                    if is_action2_equal:
                        click_success_count += 1
                        step_success = True
        elif gt_action_type == 'wait':
            wait_count += 1
            is_no_wait_abort_task = False
            if pred_action_type == gt_action_type:
                is_action2_equal = compare_action2(gt_action2, pred_action2)
                if is_action2_equal:
                    wait_success_count += 1
                    step_success = True
        elif gt_action_type == 'type':
            type_count += 1
            if pred_action_type == gt_action_type:
                text_count += 1
                gt_text = row['搜索槽位（searchKeyword）']
                pred_text = pred['args']['text']
                if gt_text == pred_text:
                    text_success_count += 1
                    is_action2_equal = compare_action2(gt_action2, pred_action2)
                    if is_action2_equal:
                        type_success_count += 1
                        step_success = True
        elif gt_action_type == 'Abort':
            abort_count += 1
            is_no_wait_abort_task = False
            if pred_action_type == gt_action_type:
                is_action2_equal = compare_action2(gt_action2, pred_action2)
                if is_action2_equal:
                    abort_success_count += 1
                    step_success = True
        else:
            raise ValueError(f"Unknown action type: {gt_action_type}")
        if step_success == False:
            task_success = False

        new_row = row.copy()
        new_row['step_success'] = step_success
        new_df = pd.concat([new_df, pd.DataFrame([new_row])], ignore_index=True)
    
    task_count += 1
    if task_success:
        task_success_count += 1
    if is_no_wait_abort_task:
        no_wait_abort_task_count += 1
        if task_success:
            no_wait_abort_task_success_count += 1

print(f"task_count: {task_count}, task_success_count: {task_success_count}")
print(f"task_success_rate: {task_success_count / task_count:.2%}")
print(f"no_wait_abort_task_count: {no_wait_abort_task_count}, no_wait_abort_task_success_count: {no_wait_abort_task_success_count}")
print(f"no_wait_abort_task_success_rate: {no_wait_abort_task_success_count / no_wait_abort_task_count:.2%}")

print(f"click_count: {click_count}, click_success_count: {click_success_count}, click_success_rate: {click_success_count / click_count:.2%}")
print(f"wait_count: {wait_count}, wait_success_count: {wait_success_count}, wait_success_rate: {wait_success_count / wait_count:.2%}")
print(f"type_count: {type_count}, type_success_count: {type_success_count}, type_success_rate: {type_success_count / type_count:.2%}")
print(f"abort_count: {abort_count}, abort_success_count: {abort_success_count}, abort_success_rate: {abort_success_count / abort_count:.2%}")
print(f"total_success_rate: {(click_success_count + wait_success_count + type_success_count + abort_success_count) / (click_count + wait_count + type_count + abort_count):.2%}")
print(f"grounding_count: {grounding_count}, grounding_success_count: {grounding_success_count}, grounding_success_rate: {grounding_success_count / grounding_count:.2%}")
print(f"text_count: {text_count}, text_success_count: {text_success_count}, text_success_rate: {text_success_count / text_count:.2%}")

new_df.to_excel(r"E:\模拟点击调试数据\result_with_step_success.xlsx", index=False)
