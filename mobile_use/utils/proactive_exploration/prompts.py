reward_prompt = '''Please reflect on whether there are any problems with your exploration based on the pages you recently explored.

requirements:
1. Output actions that are different from the history!!!
2. Do not click the button that will leave the App!!!Do not click on the <Help> page.
3. Avoid clicking where you need to type!!! Do not stay on the keyboard page for too long. If you need to type, please call the type function.
4. You must explore the process of creating different type content, such as adding files, folders, and entries.


## Output format
<think>...</think><advice>...</advice>

Please think about whether the exploration paths in the last few steps violate the above requirements and give appropriate suggestions for trajectory correction.'''
summary_prompt = '''You are exploring what functions an App has. Your task is to constantly record the experience gained from the latest exploration and continuously update and summarize the experience gained. And keep recording the functions supported by the page.

### Requirements
1. The recorded experience needs to be concise, but the key points should be highlighted, such as the specific meanings of different colors or buttons;
2. It is necessary to record what actions were performed and what effects were produced and the functions supported by the page;
3. The description of the previous and next pages can be detailed or brief.
4. Do not include words such as "possible issue" in the experience records.


The latest explored page is as follows:
<|vision_start|><|image_pad|><|vision_end|>
After taking the following action {action_1}, the following page is obtained:
<|vision_start|><|image_pad|><|vision_end|>

Summary of experience of previously explored content -before_summary:
{before_summary}

You need to summarize all main functions on the page and determine whether to update before_summary based on the most recent exploration step.

### Output format
1. If the experience summarized in the most recent step has similar points recorded in before_summary, there is no need to update it and output False
2. If before_summary is None or you think it needs to be updated, please output the updated experience summary in the format of <exp>1, ...\n2, ...</exp>.
3. Note! After the update, do not lose the important function description information in before_summary! !
4. Do not use the word "page" directly. You need a more specific description such as: App main page, page after clicking setting, etc.'''

exploration_prompt = r"""You are an intelligent agent who is familiarizing with the functions and operations of an app by yourself. Your task is to become familiar with the general functions of an app by operating your mobile phone, so in most cases you only need to type, click or swipe.

Attention:
1. You must explore the process of creating different type content, such as adding files, folders, and entries.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{{"type": "function", "function": {{"name_for_human": "mobile_use", "name": "mobile_use", "description": "Use a touchscreen to interact with a mobile device, and take screenshots.
* This is an interface to a mobile device with touchscreen. You can perform actions like clicking, typing, swiping, etc.
* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.
* The screen's resolution is {width}x{height}.
* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.", "parameters": {{"properties": {{"action": {{"description": "The action to perform. The available actions are:
* `key`: Perform a key event on the mobile device.
    - This supports adb's `keyevent` syntax.
    - Examples: "volume_up", "volume_down", "power", "camera", "clear".
* `click`: Click the point on the screen with coordinate (x, y).
* `long_press`: Press the point on the screen with coordinate (x, y) for specified seconds.
* `swipe`: Swipe from the starting point with coordinate (x, y) to the end point with coordinates2 (x2, y2).
* `type`: Input the specified text into the activated input box.
* `system_button`: Press the system button.
* `open`: Open an app on the device.", "enum": ["key", "click", "long_press", "swipe", "type","system_button", "open"], "type": "string"}}, "coordinate": {{"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=click`, `action=long_press`, and `action=swipe`.", "type": "array"}}, "coordinate2": {{"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=swipe`.", "type": "array"}}, "text": {{"description": "Required only by `action=open`, `action=type`.", "type": "string"}}, "time": {{"description": "The seconds to wait. Required only by `action=long_press`.", "type": "number"}}, "button": {{"description": "Back means returning to the previous interface, Required only by `action=system_button`", "enum": ["Back"], "type": "string"}} }}, "required": ["action"], "type": "object"}}, "args_format": "Format the arguments as a JSON object."}}}}
</tools>

For each function call, return a json object with function name and arguments within ```JSON``` XML tags:
```JSON
{{"name": <function-name>, "arguments": <args-json-object>}}
```

You may call one functions to explorate.
""".strip()
