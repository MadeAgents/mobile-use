import re
import os
import time
import json
import pprint
import logging
import gradio as gr

from dataclasses import asdict
from dotenv import load_dotenv
from gradio import ChatMessage
from typing import Dict, Any
from PIL import Image, ImageDraw, ImageFont
from mobile_use.logger import setup_logger
from mobile_use.scheme import AgentState
from mobile_use import Environment, VLMWrapper, Agent


load_dotenv()
setup_logger(name='mobile_use')
logger = logging.getLogger('mobile_use')

disable_btn = gr.Button(interactive=False)
enable_btn = gr.Button(interactive=True, visible=True)
PARAMS_NAME = []
PARAMS_COMPONENT = []
VIEW_IMAGE_SIZE = (750, 750)
IMAGE_OUTPUT = 'logs/history'
os.makedirs(IMAGE_OUTPUT, exist_ok=True)


class Worker:
    def __init__(self):
        self._agent: Agent = None
        self._stop = False
        self._images = []
        self._history_path = None

    def save_image(self, image: Image.Image, filename: str):
        idx = self._images.index(filename) if filename in self._images else -1
        if idx >= 0:
            self._images[idx] = filename
        else:
            self._images.append(filename)
        file_path = os.path.join(self._history_path, filename)
        image.save(file_path)

    def reset(self, env: Dict[str, Any], vlm: Dict[str, Any], agent: Dict[str, Any], goal: str):
        logger.info("Reset Agent and Environment")
        env = Environment(**env)
        vlm = VLMWrapper(**vlm)
        self._images.clear()
        self._agent = Agent.from_params({'type': 'ReAct', 'env': env, 'vlm': vlm, **agent})
        i = 0
        name = re.sub(r'[^\w\u4e00-\u9fff\s-]', '', goal[:128])
        history_path = os.path.join(IMAGE_OUTPUT, name)
        while os.path.exists(history_path):
            i += 1
            history_path = os.path.join(IMAGE_OUTPUT, name) + f'_{i}'
        self._history_path = history_path
        os.makedirs(self._history_path)

    def run(self, input_content: str):
        self._stop = False
        img_file = None
        for step_data in self._agent.iter_run(input_content):
            if step_data is None:
                break
            if step_data.curr_env_state is not None:
                r = 20
                if step_data.action:
                    image = step_data.curr_env_state.pixels.copy()
                    draw = ImageDraw.Draw(image)
                    if step_data.action.name == 'click':
                        x, y = step_data.action.parameters['point']
                        draw.ellipse([x-r, y-r, x+r, y+r], fill=(255, 0, 0), outline='black', width=2)
                    elif step_data.action.name == 'scroll':
                        x, y = step_data.action.parameters['start_point']
                        draw.ellipse([x-r, y-r, x+r, y+r], fill=(255, 0, 0), outline='black', width=2)
                        x, y = step_data.action.parameters['end_point']
                        draw.ellipse([x-r, y-r, x+r, y+r], fill=(255, 0, 0), outline='black', width=2)
                    draw.text((200, 10), f"Step {step_data.step_idx}", font=ImageFont.load_default().font_variant(size=30), fill=(255, 0, 0))
                else:
                    image = step_data.curr_env_state.pixels
                img_file = f'step_{step_data.step_idx}_0.png'
                self.save_image(image, img_file)

            text = ''
            if step_data.thought:
                text += f'Step {step_data.step_idx}\nThought: {step_data.thought}'

            if step_data.action:
                text += f'\nAction: {step_data.action}'
                a = step_data.action
                if a.name.upper() == 'FINISHED' and a.parameters.get('answer'):
                    text += f"\n\nTask Finished: {a.parameters.get('answer')}"

            yield dict(text=text, img_file=img_file)

            if step_data.exec_env_state is not None:
                img_file = f'step_{step_data.step_idx}_1.png'
                self.save_image(step_data.exec_env_state.pixels, img_file)
                yield dict(text=text, img_file=img_file)

            if self._stop:
                text = "\n\n**The task has been canceled!**"
                yield dict(text=text.strip(), img_file=img_file)
                break
        if self._agent.curr_step_idx == self._agent.max_steps:
            text = f"\n\n**The task has stopped because the maximum number of steps({self._agent.max_steps}) has been reached**"
            yield dict(text=text.strip(), img_file=img_file)

    def stop(self):
        self._stop = True

class SessionWorkers:

    def __init__(self, active_duration: int=300):
        self._workers_ = {}
        self.active_duration = active_duration
    
    def get_worker(self, session_id: str) -> Worker:
        if session_id not in self._workers_:
            self._workers_[session_id] = (time.time(), Worker())
        return self._workers_[session_id][1]

    def _clear_(self):
        for sid in list(self._workers_.keys()):
            if time.time() - self._workers_[sid][0] > self.active_duration:
                self._workers_.pop(sid)


session_workers = SessionWorkers()


def get_button_state(i_run: bool=None, i_stop: bool=None, i_clear: bool=None, stop_value: str=None):
    buttons = []
    if i_run is not None:
        buttons.append(gr.update(
            value='▶️ Run' if i_run else '▶️ Running',
            interactive=i_run
        ))
    if i_stop is not None:
        if stop_value is not None:
            buttons.append(gr.update(interactive=i_stop, value=stop_value))
        else:
            buttons.append(gr.update(interactive=i_stop))
    if i_clear is not None:
        buttons.append(gr.update(
            interactive=i_clear
        ))
    return buttons



def run_agent(request: gr.Request, input_content, messages, image, *args):
    session_id = request.session_hash
    logger.info(f"instruction: {input_content}")
    messages.append(ChatMessage(role='user', content=input_content))
    yield [messages, image] + get_button_state(False, True, False)

    params = {}
    for name, value in zip(PARAMS_NAME, args):
        prefix, name = name.split('/')
        if prefix not in params:
            params[prefix] = {}
        params[prefix][name] = value
    print('============== params ==============')
    pprint.pprint(params)

    # Try to get the base_url and api_key from the env if it is not available
    if not params['vlm']['base_url']:
        params['vlm']['base_url'] = os.getenv('VLM_BASE_URL', None)
        params['vlm']['api_key'] = os.getenv('VLM_API_KEY', None)
    if not params['vlm']['base_url']:
        messages.append(ChatMessage(role="assistant", content=f'Missing vlm base url'))
        yield [messages, image] + get_button_state(True, False, True)
        return

    worker = session_workers.get_worker(session_id)
    if worker._agent is None or worker._agent.state != AgentState.CALLUSER:
        try:
            worker.reset(goal=input_content, **params)
        except Exception as e:
            logger.error(e)
            messages.append(ChatMessage(role="assistant", content=f'The agent initialization fails: {e}'))
            yield [messages, image] + get_button_state(True, False, True)
            return

    show_image = image
    step_idx = -1
    for msg in worker.run(input_content):
        img_file = msg.get('img_file')
        if img_file:
            show_image = os.path.join(worker._history_path, img_file)
        if step_idx != worker._agent.curr_step_idx:
            step_idx = worker._agent.curr_step_idx
            messages.append(ChatMessage(role="assistant", content=msg["text"]))
        else:
            messages[-1].content = msg["text"]
        yield [messages, show_image] + get_button_state(False, True, False)
    yield [messages, show_image] + get_button_state(True, False, True, stop_value='⏹️ Stop')

    # save the history
    messages_dict = []
    for msg in messages:
        try:
            msg = asdict(msg)
        except:
            pass      
        if isinstance(msg, dict):
            messages_dict.append(msg)
        else:
            logger.error(f"Error message format: {type(msg)} {msg}")
    with open(os.path.join(worker._history_path, 'messages.json'), 'w', encoding='utf-8') as writer:
        json.dump(messages_dict, writer, ensure_ascii=False, indent=4)


def clear_history(request: gr.Request):
    return ([], "", None) + (disable_btn,) * 2


def get_previous_image(request: gr.Request, curr_image_path):
    session_id = request.session_hash
    global session_workers
    worker = session_workers.get_worker(session_id)
    logger.info(f'curr_image_path: {curr_image_path}')
    curr_file = os.path.basename(curr_image_path)

    curr_index = worker._images.index(curr_file)
    if curr_index - 1 < 0:
        return curr_image_path
    else:
        pre_image_path = os.path.join(worker._history_path, worker._images[curr_index-1])
        return pre_image_path


def get_next_image(request: gr.Request, curr_image_path):
    session_id = request.session_hash
    global session_workers
    worker = session_workers.get_worker(session_id)
    logger.info(f'curr_image_path: {curr_image_path}')
    curr_file = os.path.basename(curr_image_path)

    curr_index = worker._images.index(curr_file)
    if curr_index + 1 >= len(worker._images):
        return curr_image_path
    else:
        next_image_path = os.path.join(worker._history_path, worker._images[curr_index+1])
        return next_image_path


def stop_worker(request: gr.Request):
    session_id = request.session_hash
    global session_workers
    session_workers.get_worker(session_id).stop()
    return gr.update(value="⏹️ Stopping...", interactive=False)


def add_text(instruction, messages, request: gr.Request):
    logger.info(f"instruction: {instruction}")
    messages.append(ChatMessage(role='user', content=instruction))
    return (messages, "") + (disable_btn, enable_btn, disable_btn)


def add_params_component(prefix, name, component):
    PARAMS_NAME.append(prefix+'/'+name)
    PARAMS_COMPONENT.append(component)


def build_agent_ui_demo():
    with gr.Blocks(title="Mobile Use WebUI", theme=gr.themes.Default()) as demo:
        with gr.Row():
            gr.Markdown(
                """
                # 📱 Mobile Use WebUI
                ### Control your mobile with AI assistance
                """,
                elem_classes=["header-text"],
            )
        with gr.Group():
            with gr.Row():
                with gr.Column(scale=2):
                    with gr.Accordion("📱 Mobile Settings", open=False):
                        with gr.Group():
                            host = gr.Textbox(
                                label="Android ADB Server Host",
                                placeholder='127.0.0.1',
                                info="Android ADB server host, support remote device.",
                            )
                            port = gr.Number(
                                label="Android ADB Server Port",
                                value=5037,
                                info="Android ADB server port",
                            )
                            serial_no = gr.Textbox(
                                label="Device Serial No.",
                                placeholder='a22d0110',
                                info="Serial No. for connected device",
                            )
                            reset_to_home = gr.Checkbox(
                                label="Reset to HOME",
                                value=True,
                                interactive=True,
                                info="Reset the device to HOME screen",
                            )
                            add_params_component('env', 'host', host)
                            add_params_component('env', 'port', port)
                            add_params_component('env', 'serial_no', serial_no)
                            add_params_component('env', 'go_home', reset_to_home)
                    with gr.Accordion("⚙️ Agent Settings", open=False):
                        with gr.Group():
                            with gr.Column():
                                max_steps = gr.Slider(
                                    minimum=1,
                                    maximum=50,
                                    value=30,
                                    step=1,
                                    interactive=True,
                                    label="Max Run Steps",
                                    info="Maximum number of steps the agent will take",
                                )
                                num_latest_screenshot = gr.Slider(
                                    minimum=1,
                                    maximum=10,
                                    value=3,
                                    step=1,
                                    interactive=True,
                                    label="Maximum Latest Screenshot",
                                    info="Maximum latest screenshot for per vllm request",
                                )
                                max_reflection_action = gr.Slider(
                                    minimum=1,
                                    maximum=5,
                                    value=1,
                                    step=1,
                                    interactive=True,
                                    label="Maximum Reflection Action",
                                    info="Maximum reflection action for per request",
                                )
                                add_params_component('agent', 'max_steps', max_steps)
                                add_params_component('agent', 'num_latest_screenshot', num_latest_screenshot)
                                add_params_component('agent', 'max_reflection_action', max_reflection_action)
                    with gr.Accordion("🔧 VLM Configuration", open=False):
                        with gr.Group():
                            vlm_base_url = gr.Textbox(
                                label="Base URL",
                                placeholder='http://127.0.0.1:8000/v1',
                                interactive=True,
                                info="API endpoint URL"
                            )
                            add_params_component('vlm', 'base_url', vlm_base_url)
                            vlm_api_key = gr.Textbox(
                                label="API Key",
                                type="password",
                                value='EMPTY',
                                interactive=True,
                                info="Your API key"
                            )
                            add_params_component('vlm', 'api_key', vlm_api_key)
                            vlm_model_name = gr.Dropdown(
                                label="Model Name",
                                choices=['qwen2.5-vl-7b-instruct', 'qwen2.5-vl-72b-instruct'],
                                value='qwen2.5-vl-72b-instruct',
                                interactive=True,
                                allow_custom_value=True,  # Allow users to input custom model names
                                info="Select a model from the dropdown or type a custom model name"
                            )
                            add_params_component('vlm', 'model_name', vlm_model_name)
                            vlm_max_retry = gr.Slider(
                                minimum=1,
                                maximum=5,
                                value=1,
                                step=1,
                                interactive=True,
                                label="Max Retry per Request",
                                info="Maximum number of request to VLM",
                            )
                            add_params_component('vlm', 'max_retry', vlm_max_retry)
                            vlm_temperature = gr.Slider(
                                minimum=0.0,
                                maximum=1.0,
                                value=0.3,
                                step=0.1,
                                interactive=True,
                                label="Temperature",
                                info="Controls randomness in model outputs"
                            )
                            add_params_component('vlm', 'temperature', vlm_temperature)
                    with gr.Column():
                        chatbot = gr.Chatbot(
                            elem_id="chatbot",
                            type="messages",
                            label="ToolAgent",
                            show_label=False,
                            height=550,
                        )
                        textbox = gr.Textbox(
                            lines=1,
                            label="指令",
                            show_label=False,
                            placeholder="👉 Please enter your task description",
                        )
                        with gr.Row():
                            run_button = gr.Button("▶️ Run", variant="primary")
                            stop_button = gr.Button("⏹️ Stop", variant="stop", interactive=False)
                            clear_btn = gr.Button(value="🗑️ Clear", interactive=False)

                with gr.Column(scale=1):
                    image_view = gr.Image(type="filepath", label="Screenshot", interactive=False, height=732)
                    with gr.Row():
                        previous_image = gr.Button(value="Previous", interactive=True)
                        next_image = gr.Button(value="Next", interactive=True)

        # register listeners
        btn_list = [run_button, stop_button, clear_btn]
    
        run_button.click(
            fn=run_agent,
            inputs=[textbox, chatbot, image_view] + PARAMS_COMPONENT,
            outputs=[chatbot, image_view] + btn_list,
        )
        stop_button.click(
            fn=stop_worker, inputs=None, outputs=stop_button
        )
        clear_btn.click(
            fn=clear_history, inputs=None, outputs=[chatbot, textbox, image_view, stop_button, clear_btn]
        )
        previous_image.click(
            fn=get_previous_image,
            inputs=[image_view],
            outputs=[image_view]
        )
        next_image.click(
            fn=get_next_image,
            inputs=[image_view],
            outputs=[image_view]
        )
    return demo


if __name__ == "__main__":
    demo = build_agent_ui_demo()
    demo.launch()
