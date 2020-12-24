#!/bin/bash

for i in {1..10}; do ./knn_image_denoiser.out portrait_noise.png portrait_fixed.png; done
for i in {1..10}; do ./knn_image_denoiser.out medusa_noise.png medusa_fixed.png; done
for i in {1..10}; do ./knn_image_denoiser.out yoda_noise.png yoda_fixed.png; done
for i in {1..10}; do ./knn_image_denoiser.out road_noise.png road_fixed.png; done
for i in {1..10}; do ./knn_image_denoiser.out 4k_noise.png 4k_fixed.png; done
