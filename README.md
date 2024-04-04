# imgGEN using GAN model
Design and Implement a Generative Adversarial Network (GAN) to generate new images
## Objective
To build and deploy a Deep Learning Model that generates new images of a specific category
– in this case we have, “CAT”, using a GAN model.
## Requirements
### Dataset:
The [dataset](https://www.kaggle.com/datasets/azmeenasiraj/cat-faces-data-set) used is sourced from the Cat Faces dataset obtained from [Kaggle](https://www.kaggle.com/datasets). This dataset 
contains 29,843 coloured images of cat faces, each of dimension `64 x 64`.

### Data Preprocessing
Before training, the samples in the dataset have to be preprocessed to ensure uniformity in data and improve the overall model performance.
#### 1. Resizing:
- Resizing the images to a specific resolution is beneficial as it ensures that the input dimensions of all samples are uniform.
- Consistent input dimensions help the model better understand the spatial information, scale, and hidden characteristics of the samples, both individually and relatively with each other.
- For this model, all the input samples have been resized to a `64 x 64` dimension with `3 colour` channels (resizing has been done in the dataset itself).
- The input shape of each sample is therefore: (64, 64, 3)
#### 2. Normalization: 
- Normalization is done to map the values of input samples to a specific range like `[0,1]` or `[-1, 1]`.
- This is done to reduce bias and stabilize the training process.
- The range to which normalization is done depends on the activation function used.
In this model, tanh activation is used for the most part.
- Therefore, the `input` has been mapped to the `range [-1, 1]`.
- This range ensures that the hyperbolic tangent can capture both positive and negative relationships in data, covering a broad spectrum and thereby improving model performance. 
#### 3. Data Augmentation:
- Data Augmentation is commonly used for image processing tasks.
- By performing various transformations such as `flipping`, `rotating`, `zooming`, and `resizing` to existing data, we can synthetically generate more data samples.
- This improves the quality and diversity of data available for training.
- Data augmentation is thus beneficial in preventing overfitting and also improving the quality of generation.
- Due to computational resource limitations and a sufficiently large size of the training dataset, it has been decided to not perform data augmentation for this model.
#### 4. Change of Colour Space:
- The `OpenCV` package has been used for all image manipulation tasks. 
- By default, OpenCV reads images in the BGR format. The colour space of the input dataset has been converted from BGR to Grayscale.
- Grayscale images have only one channel as opposed to the three channels (Red, Green, Blue) of the BGR format.
- This conversion significantly reduces the data size and aids in efficient computation.
- Due to the availability of limited computational resources, having lower storage requirements facilitates faster and simplified processing.
- Grayscale images are less prone to noise, unlike colour channels.
Additionally, data visualization of generated images can be enhanced in terms of features such as shapes and textures.
##### Sample Image from Dataset

<img src="https://github.com/iamrajharshit/imgGEN/blob/main/img/Sample%20Image%20from%20Dataset.png" title="Sample Image from Dataset" alt="Sample Image from Dataset" width="200" height="200"/>&nbsp;

##### Image after Loading  
<img src="https://github.com/iamrajharshit/imgGEN/blob/main/img/Image%20after%20loading.png" title="Image after loading" alt="Image after loading" width="200" height="200"/>&nbsp;

### Model Design and Model Training
#### GAN Model Architecture:
##### The Generator:
- The Generator is part of the GAN that generates realistic-looking images.
- These generated images should be similar to real images from the dataset used for training.
- In this case, our generator model should be able to generate images of cat faces.
##### Training Objective:
Over time, after training the generator should be able to generate data that is indistinguishable from real data.
- The generator model is first defined using the `Sequential()` model.
- This acts as the backbone framework upon which all the other layers are built.
Following this, a `Dense()` layer is defined with `256 x 8 x 8` number of input `neurons`.
- This is the initial Dense layer unit which will be later upscaled to the dimension of the real image in the dataset.`256` represents the filter number, and `8 x 8` is the initial pixel dimension which will be upscaled to `64 x 64`, which is the required image dimension.
- This Dense layer is then reshaped to `8 x 8 x 256` to prepare for the transposed convolution operations. Transposed convolution is the layer that upscales noise to the required output.
- To perform transposed convolution, the input shape should be transposed from `256 x 8 x 8` to `8 x 8 x 256`.
- `LeakyReLu` is an activation function and a variant of `ReLU` (Rectified Linear Units). `ReLU` maps output values to 0 (if input is negative) or x (if input is positive).
- The disadvantage of this is it leads to the dying `ReLU` problem during backpropagation. If a large negative gradient is present, the output is mapped to 0, and therefore learning and updation do not take place during backpropagation because of the very low value of the gradient.
- `Leaky ReLU` on the other hand uses a small alpha value (0.1 be default) that is multiplied by negative input values. `LeakyReLU` therefore maps the output to alpha. x(negative) or x(positive), thus preventing the dying `ReLu` problem.
- Similarly, the `tanh` (Hyperbolic tangent) activation function is used to map the input to the `range [-1, 1]`. This covers a broad spectrum and can capture both positive and negative relationships in data.
- `BatchNormalization()` layer is used to stabilize the training process and prevent overfitting in the network.
- The `Conv2DTranspose()` layer performs the Transposed Convolutional operation. This is the opposite of Regular convolution. Here, the input slides over the convolutional filter, followed by which the elementwise multiplication and addition are performed to upscale the image to the required dimensions. Here we perform up-sampling.
- `kernel_size` indicates the dimensions of the convolutional filters. Here, we use `3 x 3`filters.
- `strides` show the number of steps taken in each convolution operation. For transposed convolutional operation, strides also define the up-sampling factor.
- We use `strides = 2 x 2`and`padding = 'same'` indicates that the input and output will be of the same dimension for `stride = 1 / 2`.
##### Generator Summary
<img src="https://github.com/iamrajharshit/imgGEN/blob/main/img/Generator%20Summary%20(model%20sequential).png" title="Generator Summary " alt="Generator Summary " />&nbsp;

##### The Discriminator:
The `Discriminator model` is the part of `GAN` that differentiates between real and generated (fake) images. 
The discriminator is trained using both real images from the dataset as well as fake images generated by the generator. 
It is a binary classifier which can predict whether the input image is real or fake.
##### Training Objective: 
- After sufficient training, the discriminator should be able to accurately distinguish between real and fake images.
The discriminator here resembles a regular Convolutional Neural Network (CNN).
- The `Conv2D()` layer performs the Convolutional operation. The convolution operation involves sliding an `n` number of filters over the input image, followed by elementwise multiplication and addition. This operation extracts import features and structures of the image.
- This hidden information is used to train the model to understand the difference between real and fake images.
- `Flatten()` layer breaks down the `2D image` into a `1D vector` to serve as input to the fully connected layer. The dense layer acts as the fully connected layer.
- The output of this layer is a binary value `0/1` which indicates whether the input is real or fake.
- The `Sigmoid` activation function maps the output between `0` and `1`. This is ideal for a binary classification problem. `0`indicates that the input is a `fake image` generated by the discriminator whereas `1` indicates the input is a `real image` from the dataset.
- Loss Functions are used to calculate the loss between actual and predicted output.
- The goal is to minimize loss between actual and predicted output, thereby increasing classification accuracy.
- This loss is backpropagated through the network to update the weights of the network.
- The gradient of loss is a significant hyperparameter for training. Here, we use `binary_crossentropy` to guide model training since the problem is a binary classification task.
- Optimizers are used during compilation and training to improve the performance of the model. 
- Optimizers guide weight updation in the network during backpropagation to ensure the best possible performance and faster convergence.
Here, we use the Adam optimizer with an initial `learning rate` of 0.001. Adam has an adaptive learning rate which is adjusted for each parameter based on the gradient values. This leads to accuracy in updation and faster convergence.
##### Discriminator Summary
<img src="https://github.com/iamrajharshit/imgGEN/blob/main/img/Discriminator%20summary.png" title="Discriminator Summary" alt="Discriminator Summary" />&nbsp;
##### GAN:
- The GAN architecture is a combination of a `generator` and a `discriminator` network.
- GANs are excellent generative models that are useful for several applications like text generation, and image generation.
- The model architecture used is the Deep Convolutional GAN (DCGAN) architecture since the task is image generation.
- DCGAN is similar to GAN. However, both the generator and discriminator are designed as Convolutional Neural Networks (CNN) and their equivalents.
- The use of CNN in GAN ensures the generation of realistic and high-quality images.
- CNNs can effectively capture the spatial hierarchies in data and therefore learn to generate images with complex structures effectively.
- GAN training is computationally very expensive and training is also intensive.
- The higher the training, the better the quality of generation. GAN training also happens in an adversarial manner. This means that the generator and discriminator compete during training.
- The generator tries to produce images that are indistinguishable from real images whereas the discriminator tries to accurately differentiate between real and fake images.
- The goal is to train the generator to produce an image that is so real that the discriminator is forced to classify it as real.
#### GAN (Combined Generator and Discriminator) Summary
<img src="https://github.com/iamrajharshit/imgGEN/blob/main/img/Discriminator%20summary.png" title="GAN Summary" alt="GAN Summary" />&nbsp;

### Visual Inspection:
These are samples of images generated after `100 epochs`. Clearly, the images are not realistic. 
#### Results:
#### Generated Image after 100 Epoch:
<img src="https://github.com/iamrajharshit/imgGEN/blob/main/img/Image%20generated%20after%20100%20epochs.%20Using%20CPU.png" title="Generated Image after 100 Epochs" alt="Generated Image after 100 Epochs" />&nbsp;
##### Saving The Image after 100 Epochs
<img src="https://github.com/iamrajharshit/imgGEN/blob/main/img/Saving%20the%20image%20after%20100%20Epochs.png" title="Image after 100 Epochs" alt="Image after 100 Epochs" />&nbsp;
- However, most images have vague shapes of the cat faces. Some have well-defined shapes of the eyes, nose and mouth as well.
- Training with more number of epochs is required (1000 epochs).
- More training with better parameter values and updated network design has to be done for improved image quality.

### Observations
#### Loss Curve:
<img src="https://github.com/iamrajharshit/imgGEN/blob/main/img/Loss%20Over%20Time%20(Generator%20vs%20Discriminator)%20here%20we%20see%20our%20loss%20is%20high.png" title="Loss Over Time" alt="Loss Over Time " />&nbsp;
- The generator loss is usually high in the beginning as you can see, we have done only 100 epochs here, therefore the loss is high through out he graph.
- In next phase, will have around 1000 plus epochs and we expect to see gradually reduce in loss and the curve will stabilizes to a steady value as it learns to produce high-quality, realistic data over time. 
- A well-trained generator can produce hyper-realistic images and fool the discriminator into thinking it is real.

## Phase 2 Mini Project
### Experiment Setup and Training:
- The program uses Python 3 as the programming language and is executed in the Google 
Colaboratory environment over `T4 GPU` for intensive image processing and training support.
- The file environment is in the interactive mode.
- The model will be trained for 1000 epochs. Since GANS are computationally very expensive they require more training. Considering the limitations of computational resources available as well as the quality of generation required, 1000 epochs seemed like an ideal trade-off.
- In caseof higher GPU availability, more epochs would be beneficial in generating higher-quality of images.

### Model Evaluation
#### Frechet Inception Distance:
- Frechet Inception Distance (FID) is one of the most significant Performance evaluation metrics in GANs. 
- The FID score assesses the quality of image generation relative to the real images from the training dataset.
- Usually, a lower value of FID is desired since it indicates that the generated image has statistically similar features to the real images. This implies that the generated image is more realistic and has high visual fidelity.
(Yet to be calculated!)
#### Inception Score:
- Inception Score is one of the most important evaluation metrics for images generated through GANs. 
- The Inception score relies on the output of the Inception network to which generated images are passed. This network outputs the probability distribution of the generated image which represents the likelihood of the sample belonging to a particular class (real/fake). The probability distributions of Real and generated images are compared to get the Inception Score.
- Inception score analyses images based on both quality and diversity. It uses Kullback-Leibler distance to assess the Quality and entropy to assess the Diversity of generated images.
- A higher value of inception score indicated better generation with a balance between quality and diversity in generation. However, an image with a good inception score value may not always be realistic in terms of human perception.
(Yet to be calculated!)

### Model Deployment
- To deploy the model to a cloud-based on API, we have to first choose a suitable cloud service like Azure Machine Learning, GCP Ai platform, or AWS SageMaker, followed by which a compute instance has to be made for execution. Setting up the environment with adequate resources, processing units, and dependencies is done at this time.










