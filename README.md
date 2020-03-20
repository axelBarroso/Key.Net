# Key.Net: Keypoint Detection by Handcrafted and Learned CNN Filters
Code for the ICCV19 paper:

```text
"Key.Net: Keypoint Detection by Handcrafted and Learned CNN Filters".
Axel Barroso-Laguna, Edgar Riba, Daniel Ponsa, Krystian Mikolajczyk. ICCV 2019.
```
[[Paper on arxiv](https://arxiv.org/abs/1904.00889)]


## Update on March 20 2020

We have updated the descriptor part. Before, we were using a TensorFlow implementation of the HardNet descriptor, which we switched to the official [HardNet model in Pytorch](https://github.com/DagnyT/hardnet). 
This change provides better results on the matching step, and thus, all that follows.

## Prerequisite

Python 3.7 is required for running Key.Net code. Use Conda to install the dependencies:

```bash
conda create --name keyNet_environment tensorflow-gpu=1.13.1
conda activate keyNet_environment 
conda install -c conda-forge opencv tqdm
conda install -c conda-forge scikit-image
conda install pytorch==1.2.0 -c pytorch
```

## Feature Extraction

`extract_multiscale_features.py` can be used to extract Key.Net features for a given list of images. The list of images must contain the full path to them, if they do not exist, an error will raise. 

The script generates two numpy files, one '.kpt' for keypoints, and a '.dsc' for descriptors. The descriptor used together with Key.Net is [HardNet](https://github.com/DagnyT/hardnet). The output format of the keypoints is as follow:

- `keypoints` [`N x 4`] array containing the positions of keypoints `x, y`, scales `s` and their scores `sc`. 


Arguments:

  * list-images: File containing the image paths for extracting features.
  * results-dir: The output path to save the extracted features.
  * checkpoint-det-dir: The path to the checkpoint file to load the detector weights. Default: Pretrained Key.Net.
  * checkpoint-desc-dir: The path to the checkpoint file to load the HardNet descriptor weights.
  * num-points: The number of desired features to extract. Default: 1500.
  * extract-MS: Set to True if you want to extract multi-scale features. Default: True.


Run the following script to generate the keypoint and descriptor numpy files from the image allocated in `test_im` directory. 

```bash
python extract_multiscale_features.py --list-images test_im/image.txt --results-dir test_im/
``` 


## HSequences Benchmark

We also provide the benchmark to compute [HSequences](https://github.com/hpatches/hpatches-dataset) repeatability (single- and multi-scale), and MMA metrics. To do so, first download full images (HSequences) from [HPatches repository](http://icvl.ee.ic.ac.uk/vbalnt/hpatches/hpatches-sequences-release.tar.gz). Once downloaded, place it on the root directory of the project. We provide a file `HSequences_bench/HPatches_images.txt` containing the list of images inside HSequences.


Run the next script to compute the features from HSequences:

```bash
python extract_multiscale_features.py --list-images HSequences_bench/HPatches_images.txt --results-dir extracted_features
```

Once all features have been extracted, to compute repeatability and MMA metrics run:

```bash
python hsequeces_bench.py --results-dir extracted_features --results-bench-dir HSequences_bench/results --split full
```

Use arguments to set different options:

  * results-bench-dir: The output path to save the results in a pickle file.
  * results-dir: The output path to load the extracted features.
  * split: The name of the HPatches (HSequences) split. Use full, view or illum. 
  * top-k-points: The number of top points to use for evaluation. Set to None to use all points.
  * pixel-threshold: The distance of pixels for a matching correspondence to be considered correct.
  * overlap: The overlap threshold for a correspondence to be considered correct.
  * detector-name: Set the name of the detector for which you desire to compute the benchmark (and features have been already extracted).


## Training Key.Net 

Before training Key.Net a synthetic dataset must be generated. In our paper, we downloaded ImageNet and used it to generate synthetic pairs of images, however, any other dataset would work if it is big enough. Therefore, the first time you run the `train_network.py` script, two tfrecord will be generated, one for training and another for validation. This is only done when the code couldn't find them, thus, the next runs of the script will skip this part.

```bash
python train_network.py --data-dir /path/to/ImageNet --network-version KeyNet_default
```

Check the arguments to customize your training, some parameters you might want to change are:

  * Dataset parameters:

      * max-angle: The max angle value for generating a synthetic view to train Key.Net.
      * max-scale: The max scale value for generating a synthetic view to train Key.Net.
      * max-shearing: The max shearing value for generating a synthetic view to train Key.Net.

  * Network Architecture:

      * num-filters: The number of filters in each learnable block. 
      * num-learnable-blocks: The number of learnable blocks after handcrafted block.
      * num-levels-within-net: The number of pyramid levels inside the architecture. 
      * factor-scaling-pyramid: The scale factor between the multi-scale pyramid levels in the architecture.
      * conv-kernel-size: The size of the convolutional filters in each of the learnable blocks.


## BibTeX

If you use this code in your research, please cite our paper:

```bibtex
@InProceedings{Barroso-Laguna2019ICCV,
    author = {Barroso-Laguna, Axel and Riba, Edgar and Ponsa, Daniel and Mikolajczyk, Krystian},
    title = {{Key.Net: Keypoint Detection by Handcrafted and Learned CNN Filters}},
    booktitle = {Proceedings of the 2019 IEEE/CVF International Conference on Computer Vision},
    year = {2019},
}

