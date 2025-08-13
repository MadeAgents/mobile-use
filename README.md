# Mobile Use 📱
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


<h2 style="text-align: center;">Mobile Use​: Your AI assistant for mobile - Any app, any task.</h2>

![](docs/assets/framework.svg)


[ English | [中文](docs/README_zh.md) ]

https://github.com/user-attachments/assets/5c4d3ce8-0135-4e6e-b003-b20f81f834d4

The user inputs the task description on the Web interface, and the Mobile Use automatically operates the mobile phone and completes the task.


## 🎉 News
- **[2025/06/13]**: Our [paper](paper/MobileUse_2025_06_13.pdf) "MobileUse: A Hierarchical Reflection-Driven GUI Agent for Autonomous Mobile Operation" now is released!
- **[2025/05/13]**: Mobile Use v0.3.0 now is released! AndroidLab dynamic environment now is support! Significant improvements have been achieved on the two evaluation benchmarks of [AndroidLab](https://github.com/THUDM/Android-Lab) and [AndroidWorld](https://github.com/google-research/android_world).
- **[2025/03/28]**: The [document](benchmark/android_world/README.md) for running Mobile Use in the AndroidWorld dynamic environment now is released!
- **[2025/03/17]**: Mobile Use now supports the [multi-agent](mobile_use/agents/multi_agent.py) framework! Equipped with planning, reflection, memorization and progress mechanisms, Mobile Use achieves impressive performance on AndroidWorld!
- **[2025/03/04]**: Mobile Use is released! We have also released v0.1.0 of [mobile-use](https://github.com/MadeAgents/mobile-use) library, providing you an AI assistant for mobile - Any app, any task!

## 📊 Benchmark
![](docs/assets/androidworld_benchmark.png)

In the [AndroidWorld](https://github.com/google-research/android_world) dynamic evaluation environment, we evaluated the multi-agent version of Mobile Use agent with the multimodal large language model Qwen2.5-VL-72B-Instruct and achieved a 61.2% success rate.


![](docs/assets/androidlab_benchmark.png)

In the [AndroidLab](https://github.com/THUDM/Android-Lab) dynamic evaluation environment, we evaluated the multi-agent version of Mobile Use agent with the multimodal large language model Qwen2.5-VL-72B-Instruct and achieved a 44.2% success rate.


## ✨ Key Features
- **Auto-operating the phone**: Automatically operate the UI to complete tasks based on user input descriptions.
- **Smart Element Recognition**: Automatically parses GUI layouts and identifies operational targets.
- **Complex Task Processing**: Supports decomposition of complex task and multi-step operations.

## 🚀 Quick Start
`mobile-use` requires [ADB](https://developer.android.com/tools/adb) to control the phone, which necessitates the prior installation of the relevant tools and connecting the phone to the computer via USB.

### 1. Install SDK Platform-Tools
- Step 1. Download SDK Platform-Tools for Desktop, click [here](https://developer.android.com/tools/releases/platform-tools#downloads).
- Step 2. Unzip the downloaded file and add the platform-tools path to the environment variables.

    - Windows

        In Windows, you can add the `platform-tools` PATH to the ` Path` environment variable on the graphical interface (see [here](https://www.architectryan.com/2018/03/17/add-to-the-path-on-windows-10)) or through the command line as follows:
        ```
        setx PATH "%PATH%;D:\your\download\path\platform-tools"
        ```

    - Mac/Linux
        ```
        $ echo 'export PATH=/your/downloads/path/platform-tools:$PATH' >> ~/.bashrc
        $ source ~/.bashrc
        ```
- Step 3. Open the command line and enter `adb devices` (Windows: `adb.exe devices`) to verify adb is available or not.


### 2. Enable developer mode and open USB debugging on your phone
<img src="docs/assets/usb_debug_en.png" style="width:30%; height:auto;">

For HyperOS or MIUI, you need to turn on USB Debugging (Security Settings) at the same time.

### 3. Connect your computer and phone using a USB cable. And verify the adb is connected.
Run the `adb devices` (Windows: `adb.exe devices`) command on the command line terminal. If the device serial_no is listed, the connection is successful. The correct log is as follows:
```
List of devices attached
a22d0110        device
```

### 4: Install mobile-use
#### Option 1:  Install package directly (Recommended)
With pip (Python>=3.10):
```
pip install mobile-use
```

#### Option 2:  Install from source code
```
# Clone github repo
git clone https://github.com/MadeAgents/mobile-use.git

# Change directory into project directory
cd mobile-use

# Install uv if you don't have it already
pip install uv

# Create a virtual environment and install dependencies
# We support using Python 3.10, 3.11, 3.12
uv venv .venv --python=3.10

# Activate the virtual environment
# For macOS/Linux
source .venv/bin/activate
# For Windows
.venv\Scripts\activate

# Install mobile-use with all dependencies (pip >= 21.1)
uv pip install -e .
```


### 5. Launch the webui service
```
python -m mobile_use.webui
```

### 6. Usage
Once the service starts successfully, open the address http://127.0.0.1:7860 in your browser to access the WebUI page, as shown below:

![](docs/assets/webui.png)

Click VLM Configuration to set the Base URL and API Key of the multimodal large language model. It is recommended to use the multimodal large language model of Qwen2.5-VL series.

![alt text](docs/assets/vlm_configuration.png)

Input task descriptions in the input box at the lower left corner, click start to execute tasks.

### 7. Support Chinese characters (Optional)

If you want to input Chinese characters to your phone, for example, to let MobileUse execute a command like this: search for 
"咖啡" in the Meituan app, you need

- Step 1. Download ADBKeyBoard apk, click [here](https://github.com/senzhk/ADBKeyBoard).
- Step 2. Install ADBKeyBoard to your phone.
  ```
  adb install <path-to-ADBKeyboard.apk>
  ```

**⚠️ Special Reminder**: The actions are autonomously decided by the intelligent agent, which may pose uncontrollable operational risks. It is recommended that during the experience, you constantly monitor your phone's status. If you encounter any operational risks, promptly terminate the task or use a test phone for the experience to avoid issues caused by accidental operations.


## 🎉 More Demo
Case1：Search the latest news of DeepSeek-R2 in Xiaohongshu APP and forward one of the news to the Weibo App

https://github.com/user-attachments/assets/c44ddf8f-5d3f-4ace-abb3-fab4838b68a4


Case2：Order 2 Luckin coffees with Meituan, 1 hot raw coconut latte standard sweet, and 1 cold light jasmine

https://github.com/user-attachments/assets/6130e87e-dd07-4ddf-a64d-051760dbe6b3


Case3：用美团点一杯咖啡，冰的，标准糖

https://github.com/user-attachments/assets/fe4847ba-f94e-4baa-b4df-857cadae5b07


Case4：用美团帮我点2杯瑞幸咖啡，要生椰拿铁标准糖、热的

https://github.com/user-attachments/assets/5c4d3ce8-0135-4e6e-b003-b20f81f834d4


Case5：在浏览器找一张OPPO Find N5图片，询问DeepSeek应用该手机介绍信息，将找到的图片和介绍信息通过小红书发布

https://github.com/user-attachments/assets/4c3d8800-78b7-4323-aad2-8338fe81cb81


Case6：帮我去OPPO商城、京东、以及淘宝分别看一下oppofind n5售价是多少

https://github.com/user-attachments/assets/84990487-f2a3-4921-a20e-fcdebfc8fc60


Case7: Turn on Bluetooth and WIFI

https://github.com/user-attachments/assets/c82ae51e-f0a2-4c7b-86e8-e3411d9749bb


## ⚙️ Advance

### Advance Settings
**📱 Mobile Settings**

The `Android ADB Server Host` and `Android ADB Server Port` allow you to specify the address and port of the android ADB service, which can be used for remote device connections or local android ADB services on non-default port. When multiple devices exist, you need to specify the `Device Serial No`. The `Reset to HOME` parameter indicates whether to return the phone to the home page before executing the task. If you continue the previous task, you need to cancel this option.

![alt text](docs/assets/mobile_settings.png)

**⚙️ Agent Settings**

The `Max Run Steps` parameter specifies the maximum number of iteration steps for the Agent. If the current task exceeds the maximum number of iteration steps, the task will be stopped. Therefore, you are advised to set a larger value for complex tasks with more operation steps. The `Maximum Latest Screenshot` is to control the number of latest screenshots that the Agent can see. Because pictures consume more tokens, when the task has more steps, Appropriately take a Screenshot of the latest `Maximum Latest Screenshot` and send it to VLM to generate the next operation accordingly. The `Maximum Reflection Action` is to control the maximum number of reflection times of the Agent. The greater the value, the higher the fault tolerance rate of the Agent, but the longer the processing time of the task. 

![alt text](docs/assets/agent_settings.png)


**🔧 VLM Configuration**

Click `VLM Configuration` to specify the Base URL and API Key of the multimodal large language model, as well as the model name and temperature coefficient. It is recommended to use the multimodal large language model of Qwen2.5-VL series.

![alt text](docs/assets/vlm_configuration.png)


### Use agent with code
```python
import os
from dotenv import load_dotenv
from mobile_use.scheme import AgentState
from mobile_use import Environment, VLMWrapper, Agent
from mobile_use.logger import setup_logger

load_dotenv()
setup_logger(name='mobile_use')

# Create environment controller
env = Environment(serial_no='a22d0110')
vlm = VLMWrapper(
    model_name="qwen2.5-vl-72b-instruct", 
    api_key=os.getenv('VLM_API_KEY'),
    base_url=os.getenv('VLM_BASE_URL'),
    max_tokens=128,
    max_retry=1,
    temperature=0.0
)

agent = Agent.from_params(dict(type='default', env=env, vlm=vlm, max_steps=3))

going = True
input_content = goal
while going:
    going = False
    for step_data in agent.iter_run(input_content=input_content):
        print(step_data.action, step_data.thought)
```


### Running Mobile Use in AndoirdWorld
See [AndroidWorld.md](docs/AndroidWorld.md).

## 🗺️ Roadmap
- [x] Improve agent memory and reflection (summarize, compress.)
- [x] Provide multi-agent implementation 
- [x] Provide an evaluation process about AndroidWorld dynamic environment 
- [ ] Develop an APP that can be installed directly on the phone


## 🌱Contributing
We welcome all forms of contributions! Please read our contribution guide to learn about:
- How to submit an issue to report problems.
- The process of participating in feature development, See [Developer Document](docs/develop_en.md).
- Code style and quality standards, See [Developer Document](docs/develop_en.md).
- Methods for suggesting documentation improvements.


## 📜 License
This project is licensed under the MIT License, which permits free use and modification of the code but requires retaining the original copyright notice.

## 📚 Citation
If you have used this project in your research or work, please cite:
```
@article{li2025mobileuse,
  title={MobileUse: A GUI Agent with Hierarchical Reflection for Autonomous Mobile Operation},
  author={Li, Ning and Qu, Xiangmou and Zhou, Jiamu and Wang, Jun and Wen, Muning and Du, Kounianhua and Lou, Xingyu and Peng, Qiuying and Wang, Jun and Zhang, Weinan},
  journal={arXiv preprint arXiv:2507.16853},
  year={2025},
  url={https://arxiv.org/abs/2507.16853}
}
```

## 🤝 Acknowledgements
This project benefits from the contributions of:
- Inspiration from [browser-use](https://github.com/browser-use/browser-use)
- The multimodal large language model for the agent is based on [Qwen2.5-VL](https://huggingface.co/collections/Qwen/qwen25-vl-6795ffac22b334a837c0f9a5)
- The multi-agent implementation is based on [Mobile-Agent](https://github.com/X-PLUG/MobileAgent)
- The Web UI is built on [Gradio](https://www.gradio.app)

Thanks for their wonderful works.
