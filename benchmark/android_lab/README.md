# Benchmark MobileUse in AndroidLab

## Step 1: Environment Setup

**Fetch the AndroidLab submodule**

We fix several issues in AndroidLab: When performing tasks related to `Clock`, we fixed an issue that prevented the correct extraction of information from the clock interface. For `Settings` related tasks, we addressed problems that caused failures in retrieving the app storage, system brightness, and app notification information. 

To apply these fixes, you need fetch our AndroidLab fork:
```
git submodule sync
git submodule update --init --remote --recursive
```

**Install AndroidLab requirements**

```
uv pip install -r third_party/android_lab/requirements.txt
```


**Set up the AVD environment**

Set up detail see [Android_Lab document](https://github.com/THUDM/Android-Lab).

We recommand use Docker on Linux (x86_64).


**Install mobile-use**

Follow the guidance in [README.md](../../../README.md).


## Step 2: Perform the benchmark
**Config setup**

Copy the template config file in `configs` and set up the missing api_key and base_url, for example:
```
cp benchmark/android_lab/configs/mobileuse_template.yaml benchmark/android_lab/configs/mobileuse.yaml
vim benchmark/android_lab/configs/mobileuse.yaml
# set up the missing api_key and base_url
```

**Start evaluation**
```
python benchmark/android_lab/eval.py -n test_name -c benchmark/android_lab/configs/mobileuse.yaml
```

**Calculate the metrics**
```
python benchmark/android_lab/generate_result.py --input_folder logs/evaluation_mobile_use
```
