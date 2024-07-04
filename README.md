# DrumClassifier

A convolutional neural-network (CNN) for classifying drum instruments (hi-hats, kicks, and snares).

## **About**

`DrumClassifier` is the classifier model used for my [`DrumTracker`](https://github.com/xFiveRivers/drum-tracker) MIDI transcription program built with `PyTorch` and `TorchAudio`.

## **Installation**

Instructions for installation of the `drum-tracker` environment are the same as the ones found [here](https://github.com/xFiveRivers/drum-tracker).

## **Repository Structure**

The repository has been structured in a way to attempt to minimize nested dependencies while retaining clarity of format. The folder structure tree is as follows:

![Folder Structure](img/file_structure.png)

Where the root folder hosts all of the sub-folders and all files pertaining to the main usage of the project.

### Data Folder

The data folder contains the archives of the raw and clean data, as well as the extracted data folders (`raw/` and `clean/`) to be used for data processing and model training.

### Img Folder

Any images used in notebooks or other files are stored here.

### Results Folder

All results from the model training process are stored here. The `models/` sub-folder hosts the `state_dict` resulting from a completed training run.

### Src Folder

The core files for running the pipeline can be found here.

The `_legacy/` sub-folder contains old scripts or revisions that are no longer useful for the core function of the project, however hold some value to me. 

The `classes/` sub-folder contains the classes used for the core pipeline of training the model.

The `models/` sub-folder contains various model architectures explored during the creation of the project with `model_02` being the final revision.

Finally, the `utility/` subfolder contains any useful scripts that were needed for file management or notebooks.

## **Usage**

> **Note:** For all project uses, activate the `drum-tracker` environment in the root of the repository.

### Compressing and Extracting Data

The `archive_data.py` script found under `src/` is used to compress or extract the raw or clean data used for training. There are two arguments, the first specifies whether to compress or extract, and the second specifies either raw or clean data.

For example. to compress raw data, run the following command from the root of the repository:

```
python src/archive_data.py compress raw
```

And to extract clean data, run the following:

```
python src/archive_data.py extract clean
```

### Generating Data

The `generate_data.py` script found under `src/` is used to generate the clean data from the raw data. After extracting the raw data, run the following from the root of the repository:

```
python src/generate_data.py
```

and the generated data will be found under `data/clean/`.

### Training the Model

The `train_model.py` script found under `src/` is used for model training. It takes in three arguments, the number of `epochs` to train for, dataloader batch size (`batch`), and learning rate (`lr`). Run the following command from the root of the repository:

```
python src/train_model.py --epochs=100 --lr=0.1 --batch=32
```

The model's `state_dict` will be saved under the `results/models/` folder.

## **Data**

The classifier is trained on percussion samples that I frequently use in hip-hop production. They are low-fidelity in nature and are what you would typically hear in old-school or boom-bap hip-hop tracks. For simplicity only hi-hats, kicks, and snares are used for training with hopes of extending the model's capabilities further down the line.

### **Data Processing**

In order to do classification that is some-what robust to generalization and accurate, some data processing is required. A flowchart of the pre-processing steps can be found below.

1. The data is first converted from stereo to mono to reduce dimensionality.

2. Then down-sampled from 44100kHz to 16000kHz to decrease memory while still retaining sonic information.

3. Each sample is then segmented into $\frac{1}{10}$ of a second chunks to synthetically increase the total amount of training data.

4. Then each chunk that falls below an average decibel threshold is discarded to remove quiet samples.

5. All data transformed up to this point is now considered `clean` data.

6. A `mel-spectrogram` transformation is then applied to the clean data to be used with a `PyTorch` dataloader for the model.

![Data Flowchart](img/data_flowchart.png)
