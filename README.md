# One Pixel Attack

[![Who would win?](images/who-would-win.jpg "one thicc boi that's who")](https://www.reddit.com/r/ProgrammerHumor/comments/79g0m6/one_pixel_attack_for_fooling_deep_neural_networks/?ref=share&ref_source=link)

How simple is it to cause a deep neural network to misclassify an image if we are only allowed to modify the color of one pixel and only see the prediction probability? Turns out it is very simple. In many cases, we can even cause the network to return any answer we want.

The following project is a Keras reimplementation and tutorial of ["One pixel attack for fooling deep neural networks"](https://arxiv.org/abs/1710.08864).

## How It Works

For this attack, we will use the [Cifar10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html). The task of the dataset is to correctly classify a 32x32 pixel image in 1 of 10 categories (e.g., bird, deer, truck). The black-box attack requires only the probability labels (the probability value for each category) that get outputted by the neural network. We generate adversarial images by selecting a pixel and modifying it to a certain color.

By using an Evolutionary Algorithm called Differential Evolution (DE), we can iteratively generate adversarial images to try to minimize the confidence (probability) of the neural network's classification.

[![Ackley GIF](images/Ackley.gif)](https://en.wikipedia.org/wiki/Differential_evolution)

First, generate several adversarial samples that modify a random pixel and run the images through the neural network. Next, combine the previous pixels' positions and colors together, generate several more adversarial samples from them, and run the new images through the neural network. If there were pixels that lowered the confidence of the network from the last step, replace them as the current best known solutions. Repeat these steps for a few iterations; then on the last step return the adversarial image that reduced the network's confidence the most. If successful, the confidence would be reduced so much that a new (incorrect) category now has the highest classification confidence.

See below for some examples of successful attacks:

[![Examples](images/pred.png "thicc indeed")](one-pixel-attack.ipynb)

## Getting Started

Just want to read? [View the tutorial notebook on GitHub](one-pixel-attack.ipynb).

To run the code in the tutorial, a dedicated GPU suitable for running with Keras (`tensorflow-gpu`) is recommended. Python 3.5+ required.

1. Clone the repository.

```bash
git clone https://github.com/Hyperparticle/one-pixel-attack-keras
cd ./one-pixel-attack-keras
```

2. Install the python packages in requirements.txt if you don't have them already.

```bash
pip install -r ./requirements.txt
```

3. Run the iPython tutorial notebook with Jupyter.

```bash
jupyter notebook ./one-pixel-attack.ipynb
```

## Training and Testing

To train a model, run `train.py`. The model will be checkpointed (saved) after each epoch to the `networks/models` directory.

For example, to train a ResNet with 200 epochs and a batch size of 128:

```bash
python train.py --model resnet --epochs 200 --batch_size 128
```

To perform  attack, run `attack.py`. By default this will run all models with default parameters. To specify the types of models to test, use `--model`.

```bash
python attack.py --model densenet capsnet
```

The available models currently are:
- `lecun_net` - [LeNet, first CNN model](http://yann.lecun.com/exdb/lenet/)
- `pure_cnn` - [A NN with just convolutional layers](https://en.wikipedia.org/wiki/Convolutional_neural_network)
- `net_in_net` - [Network in Network](https://arxiv.org/abs/1312.4400)
- `resnet` - [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- `densenet` - [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)
- `wide_resnet` - [Wide Residual Networks](https://arxiv.org/abs/1605.07146)
- `capsnet` - [Dynamic Routing Between Capsules](https://arxiv.org/abs/1710.09829)

## Results

Preliminary results after running several experiments on various models. Each experiment generates 100 adversarial images and calculates the attack success rate, i.e., the ratio of images that successfully caused the model to misclassify an image over the total number of images. For a given model, multiple experiments are run based on the number of pixels that may be modified in an image (1,3, or 5). The differential algorithm was run with a population size of 400 and a max iteration count of 75.

### Untargeted Attack

**Untargeted attack on 1,3,5 pixel perturbations (100 samples)**

| model              | parameters | test accuracy | pixels | attack success rate  |
| ------------------ | ---------- | ------------- | ------ | -------------------- |
| Lecun Net          | 62K        | 74.9%         | 1      | 63%                  |
|                    |            |               | 3      | 92%                  |
|                    |            |               | 5      | 93%                  |
|                    |            |               |        |                      |
| Pure CNN           | 1.4M       | 88.8%         | 1      | 13%                  |
|                    |            |               | 3      | 58%                  |
|                    |            |               | 5      | 63%                  |
|                    |            |               |        |                      |
| Network in Network | 970K       | 90.8%         | 1      | 34%                  |
|                    |            |               | 3      | 73%                  |
|                    |            |               | 5      | 73%                  |
|                    |            |               |        |                      |
| ResNet             | 470K       | 92.3%         | 1      | 34%                  |
|                    |            |               | 3      | 79%                  |
|                    |            |               | 5      | 79%                  |
|                    |            |               |        |                      |
| DenseNet           | 850K       | 94.7%         | 1      | 31%                  |
|                    |            |               | 3      | 71%                  |
|                    |            |               | 5      | 69%                  |
|                    |            |               |        |                      |
| Wide ResNet        | 11M        | 95.3%         | 1      | 19%                  |
|                    |            |               | 3      | 58%                  |
|                    |            |               | 5      | 65%                  |
|                    |            |               |        |                      |
| CapsNet            | 12M        | 79.8%         | 1      | 19%                  |
|                    |            |               | 3      | 39%                  |
|                    |            |               | 5      | 36%                  |

It appears that the capsule network CapsNet, while more resilient to the one pixel attack than all other CNNs, is still vulnerable.

## Milestones

- [x] Cifar10 dataset
- [x] Tutorial notebook
- [x] Lecun Network, Network in Network, Residual Network, DenseNet models
- [x] CapsNet (capsule network) model
- [x] Configurable command-line interface
- [x] Efficient differential evolution implementation
- [ ] MNIST dataset
- [ ] ImageNet dataset
- [ ] Graph plot of all results
