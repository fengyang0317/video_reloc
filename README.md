# Video Re-localization
by [Yang Feng](http://cs.rochester.edu/u/yfeng23/), Lin Ma, Wei Liu, Tong 
Zhang, [Jiebo Luo](http://cs.rochester.edu/u/jluo)

### Introduction
Video Re-localization aims to accurately localize a segment in a reference 
video such that the segment semantically corresponds to a query video. For 
more details, please refer to our [paper](https://arxiv.org/abs/1808.01575).

![alt text](http://cs.rochester.edu/u/yfeng23/eccv18/framework.png "Framework")

### Citation

    @InProceedings{feng2018video,
      author = {Feng, Yang and Ma, Lin and Liu, Wei and Zhang, Tong and Luo, 
      Jiebo},
      title = {Video Re-localization},
      booktitle = {ECCV},
      year = {2018}
    }

### Requirements
```
sudo apt install python-opencv
pip install tf-nightly-gpu==1.11.0.dev20180821
```

### Dataset
1. Download the ActivityNet features at 
[link](http://activity-net.org/challenges/2016/download.html). You will get 
**activitynet_v1-3.part-00** to **activitynet_v1-3.part-05**.

2. Merge and unzip the files. You'll get **sub_activitynet_v1-3.c3d.hdf5**.
    ```
    cat activitynet_v1-3.part-0* > temp.zip
    unzip temp.zip
    ```

3. Get the code and split the features.
    ```
    git clone https://github.com/fengyang0317/video_reloc.git
    cd video_reloc
    ln -s /path/to/sub_activitynet_v1-3.c3d.hdf5 data/
    wget "http://ec2-52-11-11-89.us-west-2.compute.amazonaws.com/files/\
    activity_net.v1-3.min.json" -P data/
    python split_feat.py
    ```

4. [Optional] Download the all the videos into data/videos and get the number
    of frames in each video.
    ```
    python get_frame_num.py
    ```

5. Generate the dataset json.
    ```
    python create_dataset.py
    ```
### Model
1. Train the model.
    ```
    python match.py --data_dir data
    ```
    
2. Eval the model. The results may be slightly different from the values reported in the
paper.
    ```
    python eval.py --data_dir data --ckpt_path saving/model.ckpt-(best val ckpt)
    ```
