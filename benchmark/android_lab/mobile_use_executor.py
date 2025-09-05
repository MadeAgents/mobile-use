import time
import logging
from PIL import Image
from mobile_use.schema.schema import Action
from mobile_use.environment.mobile_environ import Environment, EnvState
from third_party.android_lab.page_executor.text_executor import TextOnlyExecutor
from third_party.android_lab.utils_mobile.and_controller import AndroidController
from third_party.android_lab.utils_mobile.utils import print_with_color


logger = logging.getLogger(__name__)


class MobileUseExecutor(TextOnlyExecutor):

    def __call__(self, code_snippet):
        '''
        self.new_page_captured = False
        self.controller.on("page", self.__capture_new_page__)
        self.current_return = None'''

        current_return = self.current_return
        local_context = self.__get_class_methods__()
        local_context.update(**{'self': self})
        return current_return

    def execute_action(self, action: Action):
        print_with_color(f"Execute Action {action}", "green")
        answer = None
        if action.name in ["open", "open_app"]:
            package_name = action.parameters['text']
            self.controller.launch(package_name)
            self.current_return = {"operation": "do", "action": 'Launch', "kwargs": {"package": package_name}}
        elif action.name == 'click':
            x, y = action.parameters['coordinate']
            self.controller.tap(x, y)
            self.current_return = {"operation": "do", "action": 'Tap', "kwargs": {"element": [x, y]}}
        elif action.name == 'long_press':
            x, y = action.parameters['coordinate']
            self.controller.long_press(x, y)
            self.current_return = {"operation": "do", "action": 'Long Press', "kwargs": {"element": [x, y]}}
        elif action.name == 'type':
            text = action.parameters['text']
            self.controller.text(text)
            self.current_return = {"operation": "do", "action": 'Type', "kwargs": {"text": text}}
        elif action.name == 'key':
            text = action.parameters['text']
            self.controller.run_command(f'adb shell input keyevent {text}')
            self.current_return = {"operation": "do", "action": f'Press {text}'}
        elif action.name == 'swipe':
            x1, y1 = action.parameters['coordinate']
            x2, y2 = action.parameters['coordinate2']
            self.controller.run_command(f'adb shell input swipe {x1} {y1} {x2} {y2} 500')
            self.current_return = {"operation": "do", "action": 'Swipe', "kwargs": {"start": [x1, y1], "end": [x2, y2]}}
        elif action.name == 'press_home':
            self.controller.home()
            self.current_return = {"operation": "do", "action": 'Press Home'}
        elif action.name == 'press_back':
            self.controller.back()
            self.current_return = {"operation": "do", "action": 'Press Back'}
        elif action.name == 'wait':
            duration = action.parameters.get('time', 5.0)
            time.sleep(duration)
            self.current_return = {"operation": "do", "action": 'Wait'}
        elif action.name == 'answer':
            answer = action.parameters['text']
        elif action.name == 'system_button':
            button = action.parameters['button']
            if button == 'Back':
                self.controller.back()
                self.current_return = {"operation": "do", "action": 'Press Back'}
            elif button == 'Home':
                self.controller.home()
                self.current_return = {"operation": "do", "action": 'Press Home'}
            elif button == 'Menu':
                self.controller.run_command('adb shell input keyevent Menu')
                self.current_return = {"operation": "do", "action": 'Press Menu'}
            elif button == 'Enter':
                self.controller.enter()
                self.current_return = {"operation": "do", "action": 'Press Enter'}
        elif action.name == 'clear_text':
            # The default keyboard is ADBKeyBoard
            self.controller.run_command('adb shell am broadcast -a ADB_CLEAR_TEXT')
            self.current_return = {"operation": "do", "action": 'Clear Text'}
        elif action.name == 'take_note':
            note = action.parameters['text']
            self.current_return = {"operation": "do", "action": 'Take Note', "kwargs": {"note": note}}
            return note
        else:
            raise ValueError(f"Unknown action: {action.name}")
        time.sleep(2)       # wait action ready
        return answer


class AndroidLabEnvironment(Environment):
    def __init__(self, controller: AndroidController, config, page_executor: MobileUseExecutor):
        self.config = config
        self.controller = controller
        self.executor = page_executor

    def reset(self, go_home: bool = False):
        if go_home:
            self.controller.home()

    def get_state(self):
        if self.executor.current_screenshot is None:
            self.executor.update_screenshot()
        try:
            pixels = Image.open(self.executor.current_screenshot)
        except Exception as e:
            logger.error(f"Failed to get screenshot: {e}.")
            raise(e)
        state = EnvState(pixels=pixels, package='')
        return state

    def get_time(self) -> str:
        re = self.controller.run_command('adb shell date')
        return re
    
    def execute_action(self, action):
        return self.executor.execute_action(action=action)
