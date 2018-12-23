## Learning to Separate Object Sounds by Watching Unlabeled Video

Use the following command to train the deep MIML network:
```
python train.py --HDF5FileRoot path_of_train.h5 \
  --name deepMIML --checkpoints_dir checkpoints \
  --model MIML --batchSize 64 \
  --dataset musicInstruments \
  --L 8 --K 1 --num_of_bases 16 \
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

Use the following command to see the train log:
```
tensorboard --logdir=runs
```

Use the following command to test the deep MIML network:
```
python test.py --HDF5FileRoot path_of_test.h5 \
  --name deepMIML --checkpoints_dir checkpoints \
  --model MIML --batchSize 64 \
  --dataset musicInstruments \
  --gpu_ids -1 --nThreads 0 \
  --num_of_fc 1 \
  --with_batchnorm \
  --L 8 --K 1 --num_of_bases 16 \
  --selected_classes \
  --dataset musicInstruments \
  --using_multi_labels |& tee -a test.log
  ```

Learning to Separate Object Sounds by Watching Unlabeled Video: [[Project Page]](http://vision.cs.utexas.edu/projects/separating_object_sounds/)    [[arXiv]](https://arxiv.org/abs/1712.04109)<br/>

This repository contains the deep MIML network implementation for our [ECCV 2018 paper](http://www.cs.utexas.edu/~grauman/papers/sound-sep-eccv2018.pdf).
