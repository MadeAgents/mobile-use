# Benchmark MobileUse in AndroidLab

## Step 1: Environment Setup
**Install AndroidLab requirements**
```
pip install -r third_party/Android_Lab/requirements.txt
```


**Set up the AVD environment**

Set up detail see [Android_Lab document](https://github.com/THUDM/Android-Lab?tab=readme-ov-file).

We recommand use Docker on Linux (x86_64).


**Install mobile-use**
```
pip install -e .
```


## Step 2: Perform the benchmark
1. Copy the template config file and set your api_key and base_url in the config file
```
cp benchmark/android_lab/configs/mobile-use-MultiAgent_template.yaml benchmark/android_lab/configs/mobile-use-MultiAgent.yaml
```

2. Start evaluation
```
python eval.py -n test_name -c benchmark/android_lab/configs/mobile-use-MultiAgent.yaml
```

3. Calculate the metrics
```
python benchmark/android_lab/generate_result.py --input_folder logs/evaluation_mobile_use
```
