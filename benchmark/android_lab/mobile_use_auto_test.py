import logging
import traceback
import mobile_use
from third_party.android_lab.evaluation.evaluation import AutoTask
from third_party.android_lab.evaluation.auto_test import AutoTest
from benchmark.android_lab.mobile_use_executor import AndroidLabEnvironment, MobileUseExecutor


logger = logging.getLogger(__name__)


class MobileUseAgent:
    def __init__(self, vllm_config, agent_config):
        self.vllm_config = vllm_config
        self.agent_config = agent_config

    def construct(self, instruction, controller, config, page_executor):
        params = {
            **self.agent_config,
            "vlm": mobile_use.VLMWrapper(**self.vllm_config),
            "env": AndroidLabEnvironment(controller, config, page_executor)
        }
        agent = mobile_use.Agent.from_params(params=params)
        agent.reset(goal=instruction)
        return agent


class MobileUse_AutoTask(AutoTask):
    def set_system_prompt(self, instruction):
        pass

    def run_step(self, round_count):
        self.record.update_before(controller=self.controller, need_screenshot=True, ac_status=self.accessibility)
        step_data = None
        try:
            step_data = self.agent.step()
            if self.agent.status == mobile_use.scheme.AgentStatus.FINISHED:
                self.page_executor.is_finish = True
                message = self.agent.trajectory[-1].thought
                for step_data in self.agent.trajectory[::-1]:
                    if step_data.answer is not None:
                        message = step_data.answer
                        break
                self.page_executor.current_return = {"operation": "finish", "action": 'finish', "kwargs": {"message": message}}
            rsp = self.agent.trajectory[-1].content
        except Exception as e:
            logger.info("Some error happened during the MobileUse agent run.")
            traceback.print_exc()
            rsp = str(e)
        print(f"response = {rsp}")
        exe_res = self.page_executor('')
        print(f"Exe res = {exe_res}")
        self.record.update_after(exe_res, rsp)
        self.record.turn_number += 1


class MobileUse_AutoTest(AutoTest):
    def get_agent(self):
        agent = self.llm_agent.construct(self.instruction, self.controller, self.config, self.page_executor)
        task_agent = MobileUse_AutoTask(
            self.instruction,
            self.controller,
            self.page_executor,
            agent,
            self.record,
            self.command_per_step)
        return task_agent

    def get_executor(self):
        return MobileUseExecutor(self.controller, self.config)
