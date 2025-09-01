import os
import time
import base64
import logging
import traceback
from typing import Optional
import adbutils

from android_world.env import interface
from android_world.agents.m3a import _generate_ui_elements_description_list

from mobile_use.environment.mobile_environ import Environment
from mobile_use.schema.schema import Action, EnvState

logger = logging.getLogger(__name__)


def get_a11y_tree(aw_env: interface.AsyncEnv) -> str:
    state = aw_env.get_state(wait_to_stabilize=False)
    ui_elements = state.ui_elements
    logical_screen_size = aw_env.logical_screen_size
    a11y_tree = _generate_ui_elements_description_list(ui_elements, logical_screen_size)
    return a11y_tree

class MobileUseEnvironment(Environment):
    def __init__(
        self,
        serial_no: str=None,
        host: str="127.0.0.1",
        port: int=5037,
        wait_after_action_seconds: float=2.0,
        aw_env: interface.AsyncEnv=None,
    ):
        super().__init__(
            serial_no=serial_no,
            host=host,
            port=port,
            wait_after_action_seconds=wait_after_action_seconds,
        )
        self.aw_env = aw_env

    def get_state(self, display_id: int=-1, return_a11y_tree: bool=True) -> EnvState:
        try:
            pixels = self._d.screenshot(display_id, error_ok=False)
        except Exception as e:
            raise ValueError(f"Get screenshot error, {traceback.format_exc()}") from e
        
        a11y_tree = None
        if return_a11y_tree:
            if self.aw_env is None:
                logger.warning("AndroidWorld environment is not provided, cannot get a11y_tree.")
            try:
                logger.info("Getting a11y tree...")
                a11y_tree = get_a11y_tree(self.aw_env)
            except Exception as e:
                logger.warning(f"Failed to get a11y_tree: {e}")

        package = self._d.app_current().package
        device_time = self._d.shell('date')
        state = EnvState(pixels=pixels, package=package, a11y_tree=a11y_tree, device_time=device_time)
        return state
