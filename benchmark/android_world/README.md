# Benchmark MobileUse in AndroidWorld

## Step 1: Environment Setup

**Fetch the AndroidWorld submodule**

We fix several issues in AndroidWorld: When conducting multiple experiments using the same Android Virtual Device (AndroidWorldAVD), we observed that the internal state of certain apps could vary depending on tasks executed in previous runs (for example, the camera might remain in photo or video mode). Although a robust agent should ideally handle all possible states, such variations can introduce instability and inconsistency in evaluation. To mitigate this issue, we implemented a modification: before executing any task involving `Audio Recorder`, `Camera`, `Tasks`, `Markor`, `Simple Calendar Pro`, or `Chrome` apps, we reset the corresponding app to ensure a consistent internal state across all task runs. 

To apply these fixes, you need fetch our AndroidWorld fork:
```
git submodule sync
git submodule update --init --remote --recursive
```

**Set up the AndroidWorld environment**

Follow the guidance in [README.md](https://github.com/google-research/android_world).

**Install mobile-use**

Follow the guidance in [README.md](../../../README.md).

We recommand you to install mobile-use in the same environment created for AndroidWorld.

📌 **Note**: To run AndroidWorld on the Windows platform, you should use python>=3.12.



## Step 2: Perform the benchmark
**Config setup**

Copy the template config file in `configs` and set up the missing api_key and base_url, for example:
```
cp benchmark/android_world/configs/mobileuse_template.yaml benchmark/android_world/configs/mobileuse.yaml
vim benchmark/android_world/configs/mobileuse.yaml
# set up the missing api_key and base_url
```

**Environment variable setup**

You can set `ANDROID_ADB_SERVER_PORT` to change the adb server port if the default 5037 port is not available. AndroidWorld sets a default maximum step for each task. However, due to the misalignment of the action space, using Mobile Use may require more steps to complete a task. You can modify the maximum number of steps by setting `ANDROID_MAX_STEP`:
```
# Change the adb server port to 5038.
# Set the maximum number of steps to 1.2 times the original amount.
echo "ANDROID_ADB_SERVER_PORT=5038
ANDROID_MAX_STEP=1.2" > .env
```

**Start evaluation**

Choose the Agent type and the corresponding config file. All agents can be found in `mobile_use/agents` folder.

```
python benchmark/android_world/run.py \
     --mobileuse_agent_name=MultiAgent \
     --mobileuse_config_path=benchmark/android_world/configs/mobileuse.yaml
```
