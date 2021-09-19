#!/bin/bash

for i in {1..10}; do ./knn_omp portrait_noise.png portrait_fixed.png; done
for i in {1..10}; do ./knn_omp medusa_noise.png medusa_fixed.png; done
for i in {1..10}; do ./knn_omp yoda_noise.png yoda_fixed.png; done
for i in {1..10}; do ./knn_omp road_noise.png road_fixed.png; done
for i in {1..10}; do ./knn_omp 4k_noise.png 4k_fixed.png; done
