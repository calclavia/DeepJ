# DeepJ: A model for style-specific music generation
https://arxiv.org/abs/1801.00887

## Abstract
Recent advances in deep neural networks have enabled algorithms to compose music that is comparable to music composed by humans. However, few algorithms allow the user to generate music with tunable parameters. The ability to tune properties of generated music will yield more practical benefits for aiding artists, filmmakers, and composers in their creative tasks. In this paper, we introduce DeepJ - an end-to-end generative model that is capable of composing music conditioned on a specific mixture of composer styles. Our innovations include methods to learn musical style and music dynamics. We use our model to demonstrate a simple technique for controlling the style of generated music as a proof of concept. Evaluation of our model using human raters shows that we have improved over the Biaxial LSTM approach.

## Setup Environment
This project uses NVIDIA Docker build its environment.

```
docker-compose build
```

To run the Docker environment:
