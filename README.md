# Kaggle Mammography Breast Cancer Detection Competition 2023: Solution Attempts
## Code rules by Kaggle
Solution attempts for RSNA Screening Mammography Breast Cancer Detection competition.  https://www.kaggle.com/competitions/rsna-breast-cancer-detection/overview

_As per Kaggle competition rules copied below from https://www.kaggle.com/competitions/rsna-breast-cancer-detection/rules, the discussion/code would be uploaded to kaggle website as well. Unfortunately, the code currently doesn't run on python 3.7 environment._

> B. Public Code Sharing. You are permitted to publicly share Competition Code, provided that such public sharing does not violate the intellectual property rights of any third party. If you do choose to share Competition Code or other such code, you are required to share it on Kaggle.com on the discussion forum or notebooks associated specifically with the Competition for the benefit of all competitors. By so sharing, you are deemed to have licensed the shared code under an Open Source Initiative-approved license (see www.opensource.org) that in no event limits commercial use of such Competition Code or model containing or depending on such Competition Code.

The code can be replicated using following commands, including, in a virtual docker container.

## Create docker image and container and execute training and prediction commands
Here are the steps to run the scripts on Docker:

1. Create Docker Image

Replace username and mypassword with your Docker Hub username and password respectively.
> docker build --build-arg USERNAME=username --build-arg PASSWORD=mypassword -t kg_rsna_bc_docker_cuda12.1_ubuntu22.04_username .

2. Create and Run Container
```
docker run -dit --name kg_rsna_bc_gpu1_name --shm-size=1.5g --gpus '"device=1"' \
--mount type=bind,source="$(pwd)"/input/rsna-breast-cancer-detection,target=/home/username/input/rsna-breast-cancer-detection \
--mount type=bind,source="$(pwd)"/AutogluonModels,target=/home/username/AutogluonModels \
--user username \
kg_rsna_bc_docker_cuda12.1_ubuntu22.04_username tail -f /dev/null
```

3. Login
Use the following command to log in to the container:
```
docker exec -it kg_rsna_bc_gpu0_name /bin/bash
docker exec -it kg_rsna_bc_gpu1_name /bin/bash
```

4. Create PNG Files of a Certain Size
```
time python ./autogluon_beginner_multimodal_v2.py --createpng --uselock --png_size 768 --croptype 0
```

5. Train and Predict on Original PNGs
```
Command for 256x256 resolution png files
nohup time python ./autogluon_beginner_multimodal_v2.py --png_size 256 --train --predict --croptype 0 --use_train_to_test > ./output/runlog/kaggle_bc_rsna_autoguon_256_20230403_1305.txt 2>&1 &

Command for 512x512 resolution png files
nohup time python ./autogluon_beginner_multimodal_v2.py --png_size 512 --train --predict --croptype 0 --use_train_to_test > ./output/runlog/kaggle_bc_rsna_autoguon_512_20230403_1305.txt 2>&1 &

Command for 768x768 resolution png files
nohup time python ./autogluon_beginner_multimodal_v2.py --png_size 768 --train --predict --croptype 0 --use_train_to_test > ./output/runlog/kbcr_auto_768_crop0_ctftt_$(date +%Y%m%d_%H%M%S).txt 2>&1 &

Command for 1024x1024 resolution png files
nohup time python ./autogluon_beginner_multimodal_v2.py --png_size 1024 --train --predict --croptype 0 --use_train_to_test > ./output/runlog/kaggle_bc_rsna_autoguon_1024_20230403_1305.txt 2>&1 &
```

6. Create Focused Region Cropped PNG Files
```
time python ./convert_png_to_better_bbox.py
```

7. Train and Predict on Cropped PNGs
```
nohup time python ./autogluon_beginner_multimodal_v2.py --png_size 256 --train --predict --croptype 3 --use_train_to_test > ./output/runlog/kaggle_bc_rsna_autoguon_256_20230403_1305.txt 2>&1 &
nohup time python ./autogluon_beginner_multimodal_v2.py --png_size 512 --train --predict --croptype 3 --use_train_to_test > ./output/runlog/kaggle_bc_rsna_autoguon_512.txt 2>&1 &
nohup time python ./autogluon_beginner_multimodal_v2.py --png_size 768 --train --predict --croptype 3 --use_train_to_test > ./output/runlog/kaggle_bc_rsna_autoguon_768.txt 2>&1 &
nohup time python ./autogluon_beginner_multimodal_v2.py --png_size 1024 --train --predict --croptype 3 --use_train_to_test > ./output/runlog/kaggle_bc_rsna_autoguon_1024.txt 2>&1 &
```

## Results

### Probabalistic F1 Scores
| Size | Image Type | Probabilistic F1 Score |
|------|------------|------------------------|
| **256**  | **Cropped**    | **0.964**                  |
| 256  | Original   | 0.362                  |
| 512  | Cropped    | 0.439                  |
| 512  | Original   | 0.053                  |
| 768  | Cropped    | 0.313                  |
| 768  | Original   | 0.694                  |
| 1024 | Cropped    | 0.040                  |
| 1024 | Original   | 0.517                  |


