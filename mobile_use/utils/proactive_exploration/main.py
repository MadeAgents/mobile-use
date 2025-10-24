#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import time
import base64
import random
import argparse
from typing import List, Tuple

import tqdm
from openai import OpenAI

from io import BytesIO
from mobile_use.utils import encode_image_url, smart_resize
import android_world.env
import android_world.env.adb_utils
from prompts import reward_prompt, summary_prompt, exploration_prompt
from utils import compare_image, parse_response

import android_world
from android_world import registry, suite_utils
from android_world.env import env_launcher
from android_world.env.setup_device import apps  # noqa: F401

import mobile_use


APPS = {
    "tasks": [
        "TasksCompletedTasksForDate",
        "TasksDueNextWeek",
        "TasksDueOnDate",
        "TasksHighPriorityTasks",
        "TasksHighPriorityTasksDueOnDate",
        "TasksIncompleteTasksOnDate"
    ],
    "clock": ["ClockTimerEntry"],
    "contacts": ["ContactsAddContact"],
    "files": ["FilesDeleteFile"],
    "joplin": ["NotesRecipeIngredientCount"],
    "simple draw pro": ["SimpleDrawProCreateDrawing"],
    "markor": ["MarkorCreateFolder"],
    "broccoli": ["RecipeAddSingleRecipe"],
    "camera": ["CameraTakePhoto"],
    "pro expense": ["ExpenseDeleteDuplicates"],
    "simple sms": ["SimpleSmsReply"],
    "retro music": [
        "RetroCreatePlaylist",
        "RetroPlayingQueue",
        "RetroPlaylistDuration",
        "RetroSavePlaylist"
    ],
    "vlc": ["VlcCreatePlaylist", "VlcCreateTwoPlaylists"],
    "simple calendar": ["SimpleCalendarDeleteEvents"],
    "settings": ["SystemBrightnessMax"],
    "audio recorder": ["AudioRecorderRecordAudio", "AudioRecorderRecordAudioWithFileName"],
    "chrome": ["BrowserDraw", "BrowserMaze", "BrowserMultiply"],
    "open tracks": [
        "SportsTrackerActivitiesOnDate",
        "SportsTrackerActivityDuration",
        "SportsTrackerLongestDistanceActivity",
        "SportsTrackerTotalDistanceForCategoryOverInterval",
        "SportsTrackerTotalDurationForCategoryThisWeek"
    ],
    "osmand": ["OsmAndTrack", "OsmAndMarker", "OsmAndFavorite"],
}


def run_exploration(
    log_dir: str,
    app_name: str,
    critic_interval: int,
    base_url: str,
    api_key: str,
    model_name: str,
    iterations: int,
) -> None:
    """
    Execute automated exploration tasks in AndroidWorld + mobile_use environment.

    Args:
        log_dir: Directory for logs and intermediate JSON/image outputs.
        critic_interval: Interval of steps to perform reward (critic) evaluation.
        base_url: Base URL for the LLM service.
        api_key: API key for the LLM service.
        model_name: Name of the LLM model to use.
        iterations: Total number of exploration iterations.
    """

    # Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)
    app_log_dir = os.path.join(log_dir, app_name.replace(" ", "_"))
    os.makedirs(app_log_dir, exist_ok=True)

    # Initialize AndroidWorld environment
    env = env_launcher.load_and_setup_env(
        console_port=5554, 
        emulator_setup=False,
        adb_path=r'C:\adb\platform-tools\adb.exe'
    )

    # Register available tasks
    task_registry = registry.TaskRegistry()
    for task_name in APPS[app_name]:
        suite = suite_utils.create_suite(
            task_registry.get_registry(family="android_world"),
            n_task_combinations=1,
            seed=30,
            tasks=[task_name],
            use_identical_params=False,
        )
        try:
            suite[task_name][0].initialize_task(env=env)
        except Exception:
            print("Failed to initialize the app", task_name)

    # Initialize mobile_use environment
    random.seed(42)
    mobile_use_env = mobile_use.Environment(serial_no="emulator-5554", port=5038)
    mobile_use_env.reset()
    time.sleep(2)

    # Initialize LLM client
    client = OpenAI(base_url=base_url, api_key=api_key)

    actions: List[str] = []
    summary_s: List[str] = []
    reward_s: List[Tuple[int, str]] = []
    llm_judge = ""

    # Open the app to explore
    android_world.env.adb_utils.launch_app(app_name, env.controller)

    # Main exploration loop
    for i in tqdm.tqdm(range(iterations)):
        pixels = mobile_use_env.get_state().pixels
        img_path = rf"{app_log_dir}/shot{i}.png"
        pixels.save(img_path)

        # ====== Critic ======
        if i > 0 and (i % critic_interval == 0):
            msg = [{"role": "user", "content": [{"type": "text", "text": reward_prompt}]}]
            # Attach the last N frames for visual feedback
            for j in range(i - critic_interval, i):
                with open(img_path.replace(f"shot{i}.png", f"shot{j}.png"), "rb") as image_file:
                    encoded = base64.b64encode(image_file.read()).decode("utf-8")
                msg[0]["content"].append(
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}}
                )

            time.sleep(2)
            judge_raw = client.chat.completions.create(
                model=model_name,
                messages=msg,
                temperature=0,
                max_tokens=1024,
            ).choices[0].message.content

            # Extract <advice>...</advice> section
            advice_matches = re.findall(r"<advice>(.*)</advice>", judge_raw, re.DOTALL)
            if advice_matches:
                llm_judge = advice_matches[0].strip().replace("user|User", "agent")
                reward_s.append((i, llm_judge))
                with open(rf"{app_log_dir}/advices.json", "w", encoding="utf-8") as fp:
                    json.dump(reward_s, fp, indent=2, ensure_ascii=False)

        # ====== Action and Summary ======
        while True:
            try:
                time.sleep(2)
                resized_h, resized_w = smart_resize(height=pixels.height, width=pixels.width)

                # Ask the LLM what to do next
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": [
                                {
                                    "type": "text",
                                    "text": exploration_prompt
                                    .replace("{width}", str(resized_w))
                                    .replace("{height}", str(resized_h)),
                                },
                            ],
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text":
                                        "The summary of pages explored:\n"
                                        + str(summary_s[-1:])
                                        + "\nSome exploration advice:\n"
                                        + llm_judge
                                        + "\n\nLet's start exploring the app features.",
                                },
                                {"type": "image_url", "image_url": {"url": encode_image_url(pixels)}},
                            ],
                        },
                    ],
                    temperature=1,
                    max_tokens=256,
                ).choices[0].message.content.replace(r'{{"name"', r'{"name"')

                # Parse LLM response into executable action
                action_name, params = parse_response(response, (resized_w, resized_h), pixels.size)
                action = mobile_use.Action(name=action_name, parameters=params)

                before_img = mobile_use_env.get_state().pixels
                time.sleep(2)
                mobile_use_env.execute_action(action)
                time.sleep(3)
                pixels = mobile_use_env.get_state().pixels

                # Compare frames to detect redundant exploration
                img_equal = compare_image(before_img, pixels)
                if img_equal > 0.99:
                    llm_judge += (
                        f"Note! After executing {response}, "
                        f"the previous and next pages are almost identical. Try again."
                    )
                    continue
                else:
                    llm_judge = re.sub(
                        r"Note! After execute[\S\s]* The previous and next pages are the same. Please try again\.",
                        "",
                        llm_judge,
                    )

                actions.append(response)
                with open(rf"{app_log_dir}/actions.json", "w", encoding="utf-8") as fp:
                    json.dump(actions, fp, indent=2, ensure_ascii=False)

                # ====== Summary ======
                before_summary = summary_s[-1] if summary_s else "None"
                filled = (
                    summary_prompt
                    .replace("{before_summary}", before_summary)
                    .replace("{action_1}", actions[-1])
                )
                parts = filled.split("<|vision_start|><|image_pad|><|vision_end|>")

                summary_response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": parts[0]},
                                {"type": "image_url", "image_url": {"url": encode_image_url(before_img)}},
                                {"type": "text", "text": parts[1]},
                                {"type": "image_url", "image_url": {"url": encode_image_url(pixels)}},
                                {"type": "text", "text": parts[2]},
                            ],
                        },
                    ],
                    temperature=0,
                    max_tokens=2048,
                ).choices[0].message.content

                if "false" not in summary_response.lower():
                    summary_s.append(summary_response)

                with open(rf"{app_log_dir}/summary.json", "w", encoding="utf-8") as f:
                    json.dump(summary_s, f, indent=2, ensure_ascii=False)

                break  # Exit the exploration loop for this iteration

            except Exception as e:
                print("Error during exploration:", e)
                break
    
    # Save explored knowledge
    knowledge_path = os.path.join(log_dir, "explored_knowledge.json")
    if os.path.exists(knowledge_path):
        with open(knowledge_path, "r", encoding="utf-8") as f:
            explored_knowledge = json.load(f)
    else:
        explored_knowledge = {}
    explored_knowledge[app_name] = summary_s[-1] if summary_s else ""
    with open(knowledge_path, "w", encoding="utf-8") as f:
        json.dump(explored_knowledge, f, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run automated mobile exploration with configurable parameters."
    )
    parser.add_argument("--log-dir", type=str, default="./", help="Directory for saving logs and images.")
    parser.add_argument("--app-name", type=str, required=True, help="Name of the app to explore.")
    parser.add_argument("--base-url", type=str, required=True, help="Base URL for the LLM API.")
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        help="API key for the LLM (defaults to environment variable OPENAI_API_KEY).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="qwen2.5-vl-72b-instruct",
        help="Model name for the LLM (default: qwen2.5-vl-72b-instruct).",
    )
    parser.add_argument("--iterations", type=int, default=100, help="Number of exploration iterations (default: 100).")
    parser.add_argument("--critic-interval", type=int, default=3, help="Interval for reward evaluation (default: 3).")
    return parser.parse_args()


def main() -> None:
    """Entry point for command-line execution."""
    args = parse_args()

    assert args.app_name in APPS, f"App {args.app_name} not found in predefined APPS."

    run_exploration(
        log_dir=args.log_dir,
        app_name=args.app_name,
        critic_interval=args.critic_interval,
        base_url=args.base_url,
        api_key=args.api_key,
        model_name=args.model_name,
        iterations=args.iterations,
    )


if __name__ == "__main__":
    main()
