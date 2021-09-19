#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <png.h>
#include <math.h>
#include <omp.h>

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

float colorDistance(float* a, float* b)
{
  return (
      (b[0] - a[0]) / 255.0f * (b[0] - a[0]) / 255.0f + (b[1] - a[1]) / 255.0f * (b[1] - a[1]) / 255.0f + (b[2] - a[2]) / 255.0f * (b[2] - a[2]) / 255.0f
  );
}

float pixelDistance(float x, float y)
{
  return (
      x * x + y * y
  );
}

float lerpf(float a, float b, float c){
  return a + (b - a) * c;
}

void errorexit(const char *s) 
{
    printf("\n%s",s);	
    exit(EXIT_FAILURE);	 	
}

long getTime(struct timeval *start,struct timeval *stop) 
{
  long time=1000000*(stop->tv_sec-start->tv_sec)+stop->tv_usec-start->tv_usec;

  return time / 1000;
}

void knn_filter(png_byte *img, png_byte *img_out, int width, int height)
{
    int threadId = omp_get_thread_num();
    int threadCount = omp_get_num_threads();

    //#pragma omp parallel for schedule(dynamic,2) shared(idx,idy)
    //#pragma omp for schedule(dynamic)
    for (int idy = 0; idy < height; idy++) {
        for (int idx = threadId; idx < width; idx += threadCount) {

            // Normalized counter for the weight threshold
            float fCount = 0;

            // Total sum of pixel weights
            float sum_weights = 0;

            // Result accumulator
            float color[] = { 0, 0, 0 };

            // Center of the filter
            int pos = (idy * width + idx) * 4;
            float color_center[] = { (float)img[pos], (float)img[pos + 1], (float)img[pos + 2], (float)img[pos + 3] };

            for (int y = -FILTER_RADIUS; y <= FILTER_RADIUS; y++)
            {
                for (int x = -FILTER_RADIUS; x <= FILTER_RADIUS; x++)
                {
                    int idy_y = idy + y;
                    int idx_x = idx + x;

                    if (idy_y < 0 || idy_y >= height || idx_x < 0 || idx_x >= width)
                        continue;

                    int curr_pos = (idy_y * width + idx_x) * 4;
                    float color_xy[] = { (float)img[curr_pos], (float)img[curr_pos + 1], (float)img[curr_pos + 2], (float)img[curr_pos + 3] };

                    float pixel_distance = pixelDistance((float)x, (float)y);

                    float color_distance = colorDistance(color_center, color_xy);

                    // Denoising
                    float weight_xy = expf(-(pixel_distance * INV_FILTER_AREA + color_distance * NOISE));

                    color[0] += color_xy[0] * weight_xy;
                    color[1] += color_xy[1] * weight_xy;
                    color[2] += color_xy[2] * weight_xy;

                    sum_weights += weight_xy;
                    fCount += (weight_xy > WEIGHT_THRESHOLD) ? INV_FILTER_AREA : 0;
                }
            }

            // Normalize result color
            sum_weights = 1.0f / sum_weights;
            color[0] *= sum_weights;
            color[1] *= sum_weights;
            color[2] *= sum_weights;

            float lerpQ = (fCount > LERP_THRESHOLD) ? LERPC : 1.0f - LERPC;

            color[0] = lerpf(color[0], color_center[0], lerpQ);
            color[1] = lerpf(color[1], color_center[1], lerpQ);
            color[2] = lerpf(color[2], color_center[2], lerpQ);

            // Result to memory
            img_out[pos] = (png_byte)color[0];
            img_out[pos + 1] = (png_byte)color[1];
            img_out[pos + 2] = (png_byte)color[2];
            img_out[pos + 3] = img[pos + 3];
        }
    }
}

int main(int argc,char **argv){
    struct timeval start, stop;
    int no_threads = 2;

    if (argc > 3) {
        no_threads = atoi(argv[3]);
    }

    gettimeofday(&start, NULL);

    // read png file to an array
    read_png_file(argv[1]);

    // allocate memory
    int size = width * height * sizeof(png_byte) * 4;

    png_byte* input_img = (png_byte*)malloc(size);
    if (!input_img) errorexit("Error allocating memory for input");

    png_byte* output_img = (png_byte*)malloc(size);
    if (!output_img) errorexit("Error allocating memory for output");

    // copy image array to allocated memory
    image_to_array(input_img);

    // execute the algorithm
    #pragma omp parallel num_threads(no_threads)
    knn_filter(input_img, output_img, width, height);
    
    // prepare array to write png
    array_to_image(output_img);

    // release resources
    free(input_img);
    free(output_img);

    /* rewrite */
    
    // write array to new png file
    write_png_file(argv[2]);
    gettimeofday(&stop, NULL);
    long timeElapsed = getTime(&start, &stop);

    // printf("Size: %dx%d Time elapsed: %ld ms\n", width, height, timeElapsed);

    FILE *pFile;
    pFile = fopen("knn_tests_omp.txt", "a");
    fprintf(pFile, "Size: %dx%d Time elapsed: %ld ms\n", width, height, timeElapsed);
    fclose(pFile);
	
    printf("Success.\n");

    return 0;
}