# Image-denoising
Image denoising using K Nearest Neighbors filter implemented in CUDA technology

## Overview
This application uses a K Nearest Neighbors filter to remove noise from images. The project was based on [this paper](http://developer.download.nvidia.com/compute/cuda/1_1/Website/projects/imageDenoising/doc/imageDenoising.pdf).

## Compilation
In order to compile this program you need a suitable NVIDIA GPU and CUDA development environment.
You can compile the application using a command:

``` 
nvcc ./knn_image_denoiser.cu -lpng -o knn_image_denoiser.out
```

## Running the application
In order to run the program, you need to specify two arguments:
- input image path
- output image path

For example:
```
./knn_image_denoiser.out input.png output.png
```

## Example images before and after denoising process:

<img src="images_noise/medusa_noise.png" alt="before" width="400"/>
<img src="images_fixed/medusa_fixed.png" alt="after" width="400"/>

<img src="images_noise/portrait_noise.png" alt="before" width="400"/>
<img src="images_fixed/portrait_fixed.png" alt="after" width="400"/>

<img src="images_noise/yoda_noise.png" alt="before" width="400"/>
<img src="images_fixed/yoda_fixed.png" alt="after" width="400"/>