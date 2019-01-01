## 基于视听信息的音源分离与定位

### 目录

0. [文件结构](#文件)
1. [训练&测试](#训练)
2. [分离结果](#分离)
3. [结果评价](#评价)
4. [Reference](#reference)

### 文件结构

代码文件树如下所示：

```c
Deep-MIML-Network
├── checkpoints/ (模型存放路径，含有预训练好的模型)
├── data/ (预处理后数据读取接口)
├── models/ (MIML网络模型)
├── options/ (训练及测试参数定义)
├── util/
|   ├── audio_import.py (音频导入、STFT和NMF)
|   ├── audio_save.py (ISTFT、保存音频文件)
|   ├── extract_key_bases.py (提取key bases)
|   ├── feat_extractor.py (图像识别与定位)
|   ├── nmf_calculate.py (NMF算法接口)
|   ├── NMF_model.py (从sklearn修改而来，实现fixed NMF)
|   ├── preprocess.py (训练集、测试集数据预处理函数)
|   └── util.py (文件目录相关函数)
├── bases.json(训练过程中生成)
├── labels.json(ImageNet类别标签)
├── test(测试脚本)
├── test.bat(windows平台测试批处理文件，命令行运行)
├── test.py
├── train(训练脚本)
├── train.bat(windows平台训练批处理文件，命令行运行)
├── train.py
└── tsbd.bat(windows平台查看训练过程，双击后打开浏览器查看)
```

训练数据集文件结构如下所示

```C
Train_Dataset
├── audios/
|   ├── duet/
|   └── solo/
├── all.h5(训练过程中生成)
├── train.h5(训练过程中生成)
└── val.h5(训练过程中生成)
```

测试数据集文件结构如下所示
```C
Test_Dataset
├── gt_audio/
|   ├── accordion_1_saxophone_1.wav
|   ├── accordion_1_saxophone_1_bases.npy(测试过程中生成)
|   ├── ...
├── testimage/
|   ├── accordion_1_saxophone_1/
|   |   ├── 00001.jpg
|   |   ├── labels.npy(测试过程中生成)
|   |   ├── ...
├── result_audio/(音源分离生成的音频文件)
|   ├── accordion_1_saxophone_1_seg1.wav
|   ├── accordion_1_saxophone_1_seg2.wav
|   ├── ...
├── result_json/ 
├── └── result.json(测试结果)
├── locations.json(测试过程中生成)
└── test.h5(测试过程中生成)
```

### 训练&测试

可用如下 shell 命令进行训练，或在 windows 控制台运行 train.bat

>  **注意**
>
>  1. 各参数的含义可以在 options/ 下三个文件中查看
>  2. 尽管代码支持多线程，但是我们发现在 windows 下多线程运行速度还要慢于单线程
>  2. 下面的参数`name`是本次训练的代号名称，没有本质意义，但是决定了训练过程中模型的保存路径（文件夹名称）。我们在提交的代码中文件夹 checkpoints 下已经存放有一个 deepMIML 文件夹，其中含有我们预训练好的模型。因此如果希望不训练直接进行测试，可以在测试脚本中将`name`设为同样的 “deepMIML”，就可以直接调用该模型。如果希望重新训练，可以更改训练脚本中的`name`参数，或者删去原有的 checkpoints/deepMIML/ 文件夹

```sh
python train.py --dataset_root path_of_train_dataset \
  --name deepMIML --checkpoints_dir checkpoints \
  --model MIML --batchSize 64 \
  --dataset musicInstruments \
  --L 8 --num_of_bases 16 \
  --num_of_fc 1 \
  --learning_rate 0.001 \
  --learning_rate_decrease_itr 5 \
  --decay_factor 0.94 \
  --display_freq 20 \
  --save_epoch_freq 20 \
  --save_latest_freq 500 \
  --gpu_ids -1 --nThreads 0 \
  --with_batchnorm \
  --with_softmax \
  --continue_train \
  --niter 300 \
  --validation_on \
  --validation_freq 50 \
  --validation_batches 10 \
  --measure_time \
  --selected_classes \
  --using_multi_labels |& tee -a train.log
```

可用如下命令（在浏览器中）查看训练日志
```shell
tensorboard --logdir=runs
```

可用如下shell命令进行测试，或在windows控制台运行test.bat
```shell
python test.py --dataset_root path_of_test_dataset \
  --name deepMIML --checkpoints_dir checkpoints \
  --model MIML --batchSize 64 \
  --dataset musicInstruments \
  --gpu_ids -1 --nThreads 0 \
  --num_of_fc 1 \
  --with_batchnorm \
  --L 8  --num_of_bases 16 \
  --selected_classes \
  --dataset musicInstruments \
  --using_multi_labels |& tee -a test.log
```

### 分离结果

我们已将分离后的音频上传至清华[云盘](https://cloud.tsinghua.edu.cn/d/6e22c41df8ae4ee2bb4b/)

### 结果评价

```python
Evaluate(jsonpath,ResultAudioPath,gtAudioPath)
```
我们调用 Evaluate 评价某次训练得到结果如下（被测试的音频文件即在清华[云盘](https://cloud.tsinghua.edu.cn/d/6e22c41df8ae4ee2bb4b/)中）：
```
Location:
   [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
SDR:
    9.7774  -10.0925   10.3236    3.0574    9.5298
   -0.6161    3.1117    6.1900    2.9565    0.0434
    3.4566   12.5439  -13.1518    9.1149   -4.7662
    9.8254    6.0613    6.8480    3.2044    2.2414
   -3.1581    5.3332    3.3113    4.0279    9.9413
   -8.1165   -5.8993    6.6886    5.1155   -1.8460
   -2.1518    3.8127   -0.8400    3.1811  -10.6738
    9.3080    9.8159    1.9693    6.0704   -0.6313
    1.6346   -1.0661  -11.8851   14.3333    8.3415
    4.9627    8.2003   -8.4008    8.6957    1.5058
mean SDR:
    2.6248
location accuracy:
    0.92
```

### Reference

Learning to Separate Object Sounds by Watching Unlabeled Video: [[Project Page]](http://vision.cs.utexas.edu/projects/separating_object_sounds/)    [[arXiv]](https://arxiv.org/abs/1712.04109)