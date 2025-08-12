import os
import time
import base64
import logging
import traceback
from typing import Optional

import adbutils
from mobile_use.schema.schema import Action, EnvState
from mobile_use.utils.utils import contains_chinese

logger = logging.getLogger(__name__)


class Environment:
    def __init__(
        self,
        serial_no: str=None,
        host: str="127.0.0.1",
        port: int=5037,
        wait_after_action_seconds: float=2.0
    ):
        self.host = host
        self.port = port
        self.serial_no = serial_no
        self.wait_after_action_seconds = wait_after_action_seconds

        self._action_space =  ['open', 'open_app', 'click', 'long_press', 'type', 'key',
                'scroll', 'swipe', 'press_home', 'press_back', 'wait',
                'time', 'answer', 'system_button', 'clear_text', 'take_note']
        self._register_function = {}

        self._d = self._setup_device(serial_no, host, port)
        self.window_size = self._d.window_size(landscape=False)

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

    def get_state(self, display_id: int=-1) -> EnvState:
        try:
            pixels = self._d.screenshot(display_id, error_ok=False)
        except Exception as e:
            raise ValueError(f"Get screenshot error, {traceback.format_exc()}") from e

        package = self._d.app_current().package
        device_time = self._d.shell('date')
        state = EnvState(pixels=pixels, package=package, device_time=device_time)
        return state


    @property
    def action_space(self):
        return self._action_space


    def register_action(self, action_name: str, action_func):
        if action_name in self.action_space:
            raise ValueError(f"Action {action_name} is already registered.")
        if not callable(action_func):
            raise ValueError(f"Action function for {action_name} must be callable.")

        self._action_space.append(action_name)
        self._register_function[action_name] = action_func


    def execute_action(self, action: Action) -> Optional[str]:
        answer = None

        if action.name not in self.action_space:
            raise ValueError(f"Action {action.name} is not in the action space.")

        match action.name:
            case 'open_app':
                package_name = action.parameters['package_name']
                self._d.app_start(package_name)

            case 'click':
                x, y = action.parameters['coordinate']
                self._d.click(x, y)

            case 'long_press':
                x, y = action.parameters['coordinate']
                duration = action.parameters.get('time', 2.0)
                self._d.swipe(x, y, x, y, duration=duration)

            case 'type':
                text = action.parameters['text']

                if contains_chinese(text):
                    logger.info("TYPE: Chinese detected.")
                    charsb64 = str(base64.b64encode(text.encode('utf-8')))[1:]
                    re = self._d.shell(["ime", "enable", 'com.android.adbkeyboard/.AdbIME'])
                    logger.info(re)

                    self._d.shell(["ime", "set", 'com.android.adbkeyboard/.AdbIME'])
                    os.system(f"adb -P {self.port} -s {self._d.get_serialno()} shell am broadcast -a ADB_INPUT_B64 --es msg %s" %charsb64)
                    self._d.shell(["ime", "disable", 'com.android.adbkeyboard/.AdbIME'])

                else:
                    self._d.shell(["input", "text", text])

            case 'key':
                text = action.parameters['text']
                self._d.keyevent(text)

            case 'scroll':
                if 'start_box' in action.parameters:
                    x1, y1 = action.parameters['start_box']
                    x2, y2 = action.parameters['end_box']
                else:
                    x1, y1 = action.parameters['start_point']
                    x2, y2 = action.parameters['end_point']
                self._d.swipe(x1, y1, x2, y2, duration=0.5)

            case 'swipe':
                x1, y1 = action.parameters['coordinate']
                x2, y2 = action.parameters['coordinate2']
                self._d.swipe(x1, y1, x2, y2, duration=0.5)

            case 'press_home':
                self._d.keyevent("HOME")

            case 'press_back':
                self._d.keyevent("BACK")

            case 'wait':
                duration = action.parameters.get('time', 5.0)
                time.sleep(duration)

            case 'time':
                answer = self._d.shell('date')

            case 'answer':
                answer = action.parameters['text']
                os.system(f'adb -P {self.port} -s {self._d.get_serialno()} shell am broadcast com.example.ACTION_UPDATE_OVERLAY --es task_type_string "Agent answered:" --es goal_string "{answer}"')

            case 'system_button':
                button = action.parameters['button']
                if button == 'Back':
                    self._d.keyevent("BACK")
                elif button == 'Home':
                    self._d.keyevent("HOME")
                elif button == 'Menu':
                    self._d.keyevent("MENU")
                elif button == 'Enter':
                    self._d.keyevent("ENTER")

            case 'clear_text':
                re = self._d.shell(["ime", "enable", 'com.android.adbkeyboard/.AdbIME'])
                logger.info(re)
                re = self._d.shell(["ime", "set", 'com.android.adbkeyboard/.AdbIME'])
                logger.info(re)
                time.sleep(1)
                os.system(f"adb -P {self.port} -s {self._d.get_serialno()} shell am broadcast -a ADB_CLEAR_TEXT")
                re = self._d.shell(["ime", "disable", 'com.android.adbkeyboard/.AdbIME'])
                logger.info(re)
                re = self._d.shell(["input", "text", " "])
                logger.info(re)
            case 'take_note':
                note = action.parameters['text']
                return note
            case _ if action.name in self.register_function:
                answer = self._register_function[action.name](**action.parameters)
            case _:
                raise ValueError(f"Unknown action: {action.name}")

        time.sleep(self.wait_after_action_seconds)
        return str(answer)
