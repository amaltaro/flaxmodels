<div align="center"><img src="docs/img/flax.png" alt="flax" width="200" height="200"></div>
<div align="center"><h3>Flax Models</h3></div>
<div align="center">A collection of pretrained models in <a href="https://github.com/google/flax">Flax</a>.</div>

</br>

<!-- ABOUT -->
### About
The goal of this project is to make current deep learning models more easily available for the awesome <a href="https://github.com/google/jax">Jax</a>/<a href="https://github.com/google/flax">Flax</a> ecosystem.

### Models
* [GPT2](flaxmodels/gpt2)  
* [StyleGAN2](flaxmodels/stylegan2)  
* [ResNet{18, 34, 50, 101, 152}](flaxmodels/resnet)  
* [VGG{16, 19}](flaxmodels/vgg)  

### Example Notebooks to play with on Colab
* <a href="https://colab.research.google.com/drive/1j58Bnt1n-k4UJRQI9jnJAJIxME8ZDZjj?usp=sharing">GPT2</a>
* <a href="https://colab.research.google.com/drive/1klNP4LbrXK5P3KwFM9_PqCVx5MwwilCI?usp=sharing">StyleGAN2</a>
* <a href="https://colab.research.google.com/drive/1hjOV3_3OT5xz0iaj4fdCJurL7XWBJUWc?usp=sharing">ResNet</a>
* <a href="https://colab.research.google.com/drive/1wIzRnxlxJmrZNsUthtjKWPKULKzvacPD?usp=sharing">VGG</a>

### Installation
You will need Python 3.7 or later.
 
1. For GPU usage, follow the <a href="https://github.com/google/jax#installation">Jax</a> installation with CUDA.
2. Then install:
   ```sh
   > pip install --upgrade git+https://github.com/matthias-wright/flaxmodels.git
   ```
For CPU-only you can skip step 1.

### Documentation
The documentation for the models is on the individual model pages.

### Testing
To run the tests, pytest needs to be installed. 
```sh
> git clone https://github.com/matthias-wright/flaxmodels.git
> cd flaxmodels
> python -m pytest tests/
```

### Acknowledgments
Thank you to the developers of Jax and Flax. The title image is a photograph of a flax flower, kindly made available by <a href="https://unsplash.com/@matyszczyk">Marta Matyszczyk</a>. 

### License
Each model has an individual license.
