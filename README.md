
# Running Code

## Create Environment
```bash
conda create --name sative -y python=3.8
conda activate sative
python -m pip install --upgrade pip
```

## Installing sative package and dependencies
```bash
pip install -e .
```

## Running feature extraction on images
```bash
python sative/scripts/extract_features.py --data data/train_images_0
```

## Train SparseAutoEncoder on features
```bash
python sative/scripts/train_sae.py
```

## Extract Features from Trained Model
```bash
python sative/scripts/extract_features.py
```



# Write-Up

## Data Preperation
I decided to amortize the task of extracting the class token features for *much* quicker training. Loading images is expensive and costly (2hrs for ~1 mil images) but loading the features makes training very quick (~2 mins/epoch)

I thought about keeping the images to their original sizes so they don't get squeezed, or do a random square crop to ameliorate the same issue. However, this is my first time *ever* using a ViT so I thought it would be safer to do the easy thing, I can experiment more later. 

## SAE architecture and design
I closely followed the Anthropic Blog on [Learning Features from Sparse Autoencoders](https://transformer-circuits.pub/2023/monosemantic-features]). This provided a really clear explanation of what to do. I also referred to [Andrew Ng's notes on Sparse Autoencoders](https://web.stanford.edu/class/cs294a/sparseAutoencoder.pdf) for intuition and to frankly just understand what it was since [Hugo Fry's Blogpost](https://www.lesswrong.com/posts/bCtbuWraqYTDtuARg/towards-multimodal-interpretability-learning-sparse-2) was informative on what you can do with a SAE but wasn't very clear on what the architecture actually was. He also has a [repo](https://github.com/HugoFry/mats_sae_training_for_ViTs) with his work but it uses a lot of custom tools I'm not comfortable with. I ended up using MLX rather than pytorch anyways.

## Training process
Setting up training was very straightforward, I used considerations from [Anthropic's Blog Post](https://transformer-circuits.pub/2023/monosemantic-features]) including the "Pre encoder bias". I did not implement Neuron resampling but it's something I would like to learn more about. I used the batch size and learning rate considerations from [Hugo Fry's Blogpost](https://www.lesswrong.com/posts/bCtbuWraqYTDtuARg/towards-multimodal-interpretability-learning-sparse-2). I also referred heavily to the [MLX Documentation](https://ml-explore.github.io/mlx/build/html/index.html) since I have never used the framework before. It was a pleasure to find out how easy and efficient writing it is!

Unfortunately, I did not have time to implement Ghost Gradients or Neuron Resampling. These would have helped with training the autoencoder neurons since these procedures are designed to resuscitate dead neurons. In order to aleviate this, I tried increasing the batch size drastically in hopes that with more variance in the data per optimization step the latent representation would keep more neurons alive. This did happen, I increased the batch size from 2048 to 65536 and the number of dead neurons decreased from 5127 to 4202.

## Evaluation
I compared multiple graphs from [Hugo Fry's Blogpost](https://www.lesswrong.com/posts/bCtbuWraqYTDtuARg/towards-multimodal-interpretability-learning-sparse-2) with my own. Including sparsity and identifying the ultra-low density cluster. Unfortunately, my graphs differ from the ones on the blog. I believe this is because I did not implement a routine to resuscitate dead neurons.

## Future things to do
1. I **really** like the idea of applying Feature localization to this interpretation framework
2. In order to view a representative category image, I want to do something like [Neural Style Transfer](https://arxiv.org/abs/1508.06576) where the optimization variable is an input image and the optimization objective is to maximize activation in the SAE for a single neuron while pressing the rest to 0
3. The results of SAE remind me of the nearest neighbor evaluations in [Learning to See by Looking at Noise](https://arxiv.org/pdf/2106.05963). I'm curious to apply this framwork to a ViT trained on just noise then develop self-supervised class categories with SAE, no fine-tuning required. 
4. I wonder if you apply the SAE on earlier layers whether the categories learned would be more similar to geometric features, like squares and circles, rather than category features. With that, you could possibly develop an interpretable taxonomy of features layer by layer!