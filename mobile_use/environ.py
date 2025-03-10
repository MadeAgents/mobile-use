import os
import base64
import time
import logging
import adbutils
from .scheme import Action, EnvState
from mobile_use.utils import contains_chinese

logger = logging.getLogger(__name__)


class Environment:

    def __init__(
            self,
            serial_no: str=None,
            host: str="127.0.0.1",
            port: int=5037,
            go_home: bool = True,
            wait_after_action_seconds: float=2.0
        ):
        self._d = self._setup_device(serial_no, host, port)
        self.reset(go_home=go_home)
        self.window_size = self._d.window_size(landscape=False)
        self.wait_after_action_seconds = wait_after_action_seconds

    def _setup_device(self, serial_no: str, host: str, port: int):
        try:
            adb = adbutils.AdbClient(host=host, port=port)
            device = adb.device(serial_no)
        except Exception as e:
            logger.error(f"Failed to connect to the device: {serial_no}.")
            raise e
        return device

    def close(self):
        self._d.close()

    def reset(self, go_home: bool = True):
        if go_home:
            self._d.keyevent("HOME")

    def get_state(self):
        pixels = self._d.screenshot(error_ok=False)
        package = self._d.app_current().package
        state = EnvState(pixels=pixels, package=package)
        return state
    
    def execute_action(self, action: Action):
        answer = None
        if action.name == 'open_app' or action.name == 'open':
            package_name = action.parameters['package_name']
            self._d.app_start(package_name)
        elif action.name == 'click' or action.name == 'left_click':
            if 'coordinate' in action.parameters:       # QwenAgent
                x, y = action.parameters['coordinate']
            else:
                x, y = action.parameters['point']
            self._d.click(x, y)
        elif action.name == 'long_press':
            if 'coordinate' in action.parameters:       # QwenAgent
                x, y = action.parameters['coordinate']
            else:
                x, y = action.parameters['point']
            duration = action.parameters.get('time', 2.0)
            self._d.swipe(x, y, x, y, duration=duration)
        elif action.name == 'type':
            text = action.parameters['text']
            # self._d.send_keys(text)
            if contains_chinese(text):
                charsb64 = str(base64.b64encode(text.encode('utf-8')))[1:]
                self._d.shell(["ime", "enable", 'com.android.adbkeyboard/.AdbIME'])
                self._d.shell(["ime", "set", 'com.android.adbkeyboard/.AdbIME'])
                os.system(f"adb -s {self._d.get_serialno()} shell am broadcast -a ADB_INPUT_B64 --es msg %s" %charsb64)
                self._d.shell(["ime", "disable", 'com.android.adbkeyboard/.AdbIME'])
            else:
                self._d.shell(["input", "text", text])
        elif action.name == 'key':
            text = action.parameters['text']
            self._d.keyevent(text)
        elif action.name == 'scroll':
            x1, y1 = action.parameters['start_point']
            x2, y2 = action.parameters['end_point']
            self._d.swipe(x1, y1, x2, y2, duration=0.5)
        elif action.name == 'swipe':       # QwenAgent
            x1, y1 = action.parameters['coordinate']
            x2, y2 = action.parameters['coordinate2']
            self._d.swipe(x1, y1, x2, y2, duration=0.5)
        elif action.name == 'press_home':
            self._d.keyevent("HOME")
        elif action.name == 'press_back':
            self._d.keyevent("BACK")
        elif action.name == 'wait':
            duration = action.parameters.get('time', 5.0)
            time.sleep(duration)
        elif action.name == 'answer':
            answer = action.parameters['text']
            os.system(f'adb -s {self._d.get_serialno()} shell am broadcast com.example.ACTION_UPDATE_OVERLAY --es task_type_string "Agent answered:" --es goal_string "{text}"')
        elif action.name == 'system_button':
            button = action.parameters['button']
            if button == 'Back':
                self._d.keyevent("BACK")
            elif button == 'Home':
                self._d.keyevent("HOME")
            elif button == 'Menu':
                self._d.keyevent("MENU")
            elif button == 'Enter':
                self._d.keyevent("ENTER")
        else:
            raise ValueError(f"Unknown action: {action.name}")
        time.sleep(self.wait_after_action_seconds)
        return answer
