# Proactive Exploration

This module provides an automated proactive exploration framework for interacting with Android applications in the AndroidWorld environment.

## File Structure

- `main.py`: Entry point for running the exploration process.
- `utils.py`: Utility functions for image comparison and response parsing.
- `prompts.py`: Predefined prompts for guiding the LLM during exploration.

## Requirements

- Android emulator or device with ADB enabled.
- Python environment with `android_world` and `mobile_use` package.


## Usage

Open the Android Virtual Device or connect to the Android device. Use the following command to check if the device is available.
```bash
adb devices
```

Run the exploration script with the following command:

```bash
python main.py <Arguments>
```

### Arguments

- `--log-dir`: Directory for saving logs and screenshots (default: `./`).
- `--app-name`: Name of the app to explore (must be in the predefined `APPS` dictionary).
- `--base-url`: Base URL for the LLM API.
- `--api-key`: API key for the LLM (defaults to the `OPENAI_API_KEY` environment variable).
- `--model-name`: Name of the LLM model to use (default: `qwen2.5-vl-72b-instruct`).
- `--iterations`: Number of exploration iterations (default: 100).
- `--critic-interval`: Interval for critic agent  evaluation (default: 3).

## Workflow

1. **AndroidWorld Environment Setup**: Initializes the AndroidWorld environment and the app status. If you want to apply proactive exploration in your own device, please omit this process.

2. **More Setup**: Initializes the MobileUse environment and the llm client.

3. **Open app**: Open the app to be explored.

4. **Exploration Loop**: Action, Summary, Critic loop to proactively explore the app and accumulate knowledge.


## Outputs

- **App Folder**: Screenshots and the outputs of each agent.
- **explored_knowledge.json**: A JSON file summarizing the explored knowledge for each app. It can be used by the Operator to enhance the task execution process.
