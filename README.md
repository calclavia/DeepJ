# DeepJ: A model for style-specific music generation
https://arxiv.org/abs/1801.00887

## Abstract
Recent advances in deep neural networks have enabled algorithms to compose music that is comparable to music composed by humans. However, few algorithms allow the user to generate music with tunable parameters. The ability to tune properties of generated music will yield more practical benefits for aiding artists, filmmakers, and composers in their creative tasks. In this paper, we introduce DeepJ - an end-to-end generative model that is capable of composing music conditioned on a specific mixture of composer styles. Our innovations include methods to learn musical style and music dynamics. We use our model to demonstrate a simple technique for controlling the style of generated music as a proof of concept. Evaluation of our model using human raters shows that we have improved over the Biaxial LSTM approach.

## Requirements
- Python 3.5

Clone Python MIDI (https://github.com/vishnubob/python-midi)
`cd python-midi`
then install using
`python3 setup.py install`.

Then, install other dependencies of this project.
```
pip install -r requirements.txt
```

The dataset is not provided in this repository. To train a custom model, you will need to include a MIDI dataset in the `data/` folder.

## Usage
To train a new model, run the following command:
```
python train.py
```

To generate music, run the following command:
```
python generate.py
```

Use the help command to see CLI arguments:
```
python generate.py --help
```
