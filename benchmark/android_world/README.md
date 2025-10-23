# Benchmark MobileUse in AndroidWorld

## Step 1: Environment Setup

Install AndroidWorld by following the guidance:
```
# Install the AndroidEnv

cd third_party/android_env
python setup.py install


# Install AndroidWorld. Note: Python 3.11 or above is required.

cd third_party/android_world
pip install -r requirements.txt
python setup.py install
```

Install mobile-use by following the guidance in [README.md](../README.md).

We recommand you to install mobile-use in the same environment created for AndroidWorld.

📌 **Note**: To run AndroidWorld on the Windows platform, you should use python>=3.12.



## Step 2: Perform the benchmark
```
python benchmark/android_world/run.py \
     --mobileuse_agent_name=MultiAgent \
     --mobileuse_config_path=benchmark/android_world/configs/MultiAgent_template.yaml
```
