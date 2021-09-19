#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <png.h>
#include <math.h>

#define FILTER_RADIUS     3                                                         // M
#define FILTER_AREA       ( (2 * FILTER_RADIUS + 1) * (2 * FILTER_RADIUS + 1) )     // (N ^ 2)
#define INV_FILTER_AREA   ( 1.0f / (float)FILTER_AREA )                             // (1 / r ^ 2)
#define WEIGHT_THRESHOLD  0.02f
#define LERP_THRESHOLD    0.66f
#define NOISE_VAL         0.32f
#define NOISE             ( 1.0f / (NOISE_VAL * NOISE_VAL) )                        // (1 / h ^ 2)
#define LERPC             0.16f


int width, height;
png_byte color_type;
png_byte bit_depth;
png_bytep *row_pointers = NULL;

__host__
void read_png_file(char *filename) 
{
  FILE *fp = fopen(filename, "rb");

  png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  if(!png) abort();

  png_infop info = png_create_info_struct(png);
  if(!info) abort();

  if(setjmp(png_jmpbuf(png))) abort();

  png_init_io(png, fp);

  png_read_info(png, info);

  width      = png_get_image_width(png, info);
  height     = png_get_image_height(png, info);
  color_type = png_get_color_type(png, info);
  bit_depth  = png_get_bit_depth(png, info);

  // Read any color_type into 8bit depth, RGBA format.
  // See http://www.libpng.org/pub/png/libpng-manual.txt

  if(bit_depth == 16)
    png_set_strip_16(png);

  if(color_type == PNG_COLOR_TYPE_PALETTE)
    png_set_palette_to_rgb(png);

  // PNG_COLOR_TYPE_GRAY_ALPHA is always 8 or 16bit depth.
  if(color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
    png_set_expand_gray_1_2_4_to_8(png);

  if(png_get_valid(png, info, PNG_INFO_tRNS))
    png_set_tRNS_to_alpha(png);

  // These color_type don't have an alpha channel then fill it with 0xff.
  if(color_type == PNG_COLOR_TYPE_RGB ||
     color_type == PNG_COLOR_TYPE_GRAY ||
     color_type == PNG_COLOR_TYPE_PALETTE)
    png_set_filler(png, 0xFF, PNG_FILLER_AFTER);

  if(color_type == PNG_COLOR_TYPE_GRAY ||
     color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
    png_set_gray_to_rgb(png);

  png_read_update_info(png, info);

  if (row_pointers) abort();

  row_pointers = (png_bytep*)malloc(sizeof(png_bytep) * height);
  for(int y = 0; y < height; y++) {
    row_pointers[y] = (png_byte*)malloc(png_get_rowbytes(png,info));
  }

  png_read_image(png, row_pointers);

  fclose(fp);

  png_destroy_read_struct(&png, &info, NULL);
}

__host__
void write_png_file(char *filename) 
{
  FILE *fp = fopen(filename, "wb");
  if(!fp) abort();

  png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  if (!png) abort();

  png_infop info = png_create_info_struct(png);
  if (!info) abort();

  if (setjmp(png_jmpbuf(png))) abort();

  png_init_io(png, fp);

  // Output is 8bit depth, RGBA format.
  png_set_IHDR(
    png,
    info,
    width, height,
    8,
    PNG_COLOR_TYPE_RGBA,
    PNG_INTERLACE_NONE,
    PNG_COMPRESSION_TYPE_DEFAULT,
    PNG_FILTER_TYPE_DEFAULT
  );
  png_write_info(png, info);

  // To remove the alpha channel for PNG_COLOR_TYPE_RGB format,
  // Use png_set_filler().
  //png_set_filler(png, 0, PNG_FILLER_AFTER);

  if (!row_pointers) abort();

  png_write_image(png, row_pointers);
  png_write_end(png, NULL);

  for(int y = 0; y < height; y++) {
    free(row_pointers[y]);
  }
  free(row_pointers);

  fclose(fp);

  png_destroy_write_struct(&png, &info);
}

__host__
void image_to_array(png_byte *image)
{
  for(int y = 0; y < height; y++) {
    png_byte *row = row_pointers[y];
    for(int x = 0; x < width; x++) {
      png_byte *px = &(row[x * 4]);
      image[(y * width + x) * 4] = px[0];
      image[(y * width + x) * 4 + 1] = px[1];
      image[(y * width + x) * 4 + 2] = px[2];
      image[(y * width + x) * 4 + 3] = px[3];
    }
  }
}

__host__
void array_to_image(png_byte *image)
{
  for(int y = 0; y < height; y++) {
    png_byte *row = row_pointers[y];
    for(int x = 0; x < width; x++) {
      png_byte *px = &(row[x * 4]);
      px[0] = image[(y * width + x) * 4];
      px[1] = image[(y * width + x) * 4 + 1];
      px[2] = image[(y * width + x) * 4 + 2];
      px[3] = image[(y * width + x) * 4 + 3];
    }
  }
}

__device__ float colorDistance(float4 a, float4 b)
{
  return (
      (b.x - a.x) / 255.0f * (b.x - a.x) / 255.0f + (b.y - a.y) / 255.0f * (b.y - a.y) / 255.0f + (b.z - a.z) / 255.0f * (b.z - a.z) / 255.0f
  );
}

__device__ float pixelDistance(float x, float y)
{
  return (
      x * x + y * y
  );
}

__device__ float lerpf(float a, float b, float c){
  return a + (b - a) * c;
}

__host__
void errorexit(const char *s) 
{
    printf("\n%s",s);	
    exit(EXIT_FAILURE);	 	
}

__host__
long getTime(struct timeval *start,struct timeval *stop) 
{
  long time=1000000*(stop->tv_sec-start->tv_sec)+stop->tv_usec-start->tv_usec;

  return time / 1000;
}

__global__ 
void knn_filter(png_byte *img, png_byte *img_out, int width, int height)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx >= width || idy >= height) return;

    // Normalized counter for the weight threshold
    float fCount = 0;

    // Total sum of pixel weights
    float sum_weights = 0;

    // Result accumulator
    float3 color = {0, 0, 0};

    // Center of the filter
    int pos = (idy * width + idx) * 4;
    float4 color_center = {(float)img[pos], (float)img[pos + 1], (float)img[pos + 2], (float)img[pos + 3]};

    for (int y = -FILTER_RADIUS; y <= FILTER_RADIUS; y++)
    {
      for (int x = -FILTER_RADIUS; x <= FILTER_RADIUS; x++)
      {
        if (idy + y < 0 || idy + y >= height || idx + x < 0 || idx + x >= width)
          continue;

        int curr_pos = ((idy + y) * width + (idx + x)) * 4;
        float4 color_xy = {(float)img[curr_pos], (float)img[curr_pos + 1], (float)img[curr_pos + 2], (float)img[curr_pos + 3]};

        float pixel_distance = pixelDistance((float)x, (float)y);

        float color_distance = colorDistance(color_center, color_xy);
        
        // Denosing
        float weight_xy = expf(-(pixel_distance * INV_FILTER_AREA + color_distance * NOISE));
        
        color.x += color_xy.x * weight_xy;
        color.y += color_xy.y * weight_xy;
        color.z += color_xy.z * weight_xy;

        sum_weights += weight_xy;
        fCount += (weight_xy > WEIGHT_THRESHOLD) ? INV_FILTER_AREA : 0;
      }
    }

    // Normalize result color
    sum_weights = 1.0f / sum_weights;
    color.x *= sum_weights;
    color.y *= sum_weights;
    color.z *= sum_weights;

    float lerpQ = (fCount > LERP_THRESHOLD) ? LERPC : 1.0f - LERPC;

    color.x = lerpf(color.x, color_center.x, lerpQ);
    color.y = lerpf(color.y, color_center.y, lerpQ);
    color.z = lerpf(color.z, color_center.z, lerpQ);

    // Result to memory
    img_out[pos] = (png_byte)color.x;
    img_out[pos+1] = (png_byte)color.y;
    img_out[pos+2] = (png_byte)color.z;
    img_out[pos+3] = img[pos+3];
}

int main(int argc,char **argv) {

  struct timeval start, stop;
  gettimeofday(&start, NULL);

  png_byte *host_img;

  // read png file to an array
  read_png_file(argv[1]);

  // allocate memory
  int size = width * height * sizeof(png_byte) * 4;

  cudaMallocHost((void**)&host_img, size);
  
  png_byte *device_img = NULL;
  if (cudaSuccess!=cudaMalloc((void **)&device_img, size))
    errorexit("Error allocating memory on the GPU 1");

  png_byte *device_output = NULL;
  if (cudaSuccess!=cudaMalloc((void **)&device_output, size))
    errorexit("Error allocating memory on the GPU 2");

  // copy image array to allocated memory
  image_to_array(host_img);

  // copy image array to device
  if (cudaSuccess!=cudaMemcpy(device_img, host_img, size, cudaMemcpyHostToDevice))
    errorexit("Error copying data to device");

  // kernel block/thread configuration
  dim3 threadsPerBlock(8, 8);
  dim3 numBlocks(ceil((float)width / threadsPerBlock.x), ceil((float)height / threadsPerBlock.y));

  // kernel
  knn_filter<<<numBlocks,threadsPerBlock>>>(device_img, device_output, width, height);
  if (cudaSuccess!=cudaGetLastError())
    errorexit("Error during kernel launch");
  
  cudaDeviceSynchronize();

  // copy memory back to host
  if (cudaSuccess!=cudaMemcpy(host_img, device_output, size, cudaMemcpyDeviceToHost))
      errorexit("Error copying results to host");
  
  // prepare array to write png
  array_to_image(host_img);

  // release resources
  if (cudaSuccess!=cudaFreeHost(host_img))
    errorexit("Error when deallocating space on host");
  if (cudaSuccess!=cudaFree(device_img))
    errorexit("Error when deallocating space on the GPU");
  if (cudaSuccess!=cudaFree(device_output))
    errorexit("Error when deallocating output space on the GPU");

  // write array to new png file
  write_png_file(argv[2]);
  gettimeofday(&stop, NULL);
  long timeElapsed = getTime(&start, &stop);

    // printf("Size: %dx%d Time elapsed: %ld ms\n", width, height, timeElapsed);

  FILE *pFile;
  pFile = fopen("knn_tests.txt", "a");
  fprintf(pFile, "Size: %dx%d Time elapsed: %ld ms\n", width, height, timeElapsed);
  fclose(pFile);
  printf("Success.\n");

  return 0;
}
