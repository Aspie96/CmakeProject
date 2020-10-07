#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <stdlib.h>
#include <time.h>
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"

#define S_R_SHIFT(A, B)	(((B) >= 0) ? ((A) >> (B)) : (A) << -(B))
#define APPROX_DIVIDE1(A, B) (S_R_SHIFT(A, B) + (S_R_SHIFT(A, (B) - 1) & 1))
#define APPROX_DIVIDE2(A, B) (((A) >> (B)) + (((A) >> ((B) - 1)) & 1))
#ifndef N
#define N 13
#endif
#ifndef WIDTH
#define WIDTH 0
#endif
#ifndef HEIGHT
#define HEIGHT WIDTH
#endif
#ifndef SAVED
#define SAVED (N - 1)
#endif

void pascal(int *p, int n) {
	int k;
	n--;
	p[0] = 1;
	for(k = 0; k < n; k++) {
		p[k + 1] = p[k] * (n - k) / (k + 1);
	}
}

void kernel1a(const stbi_uc *img, int width, int height, int n, int *kernel, unsigned short *result) {
	int i, j, z, k, l, c;
	for(i = 0; i < width; i++)
	for(j = 0; j < height; j++)
	for(z = 0; z < 3; z++) {
		c = 0;
		for(k = 0; k < n; k++) {
			l = i + k - n / 2;
			if(0 <= l && l < width) {
				c += kernel[k] * img[(j * width + l) * 3 + z];
			}
		}
		result[(z * height + j) * width + i] = APPROX_DIVIDE1(c, n - 9);
	}
}

void kernel1b(unsigned short *img, int width, int height, int n, int *kernel, unsigned short *result) {
	int i, j, k, z, c, l;
	for(i = 0; i < width; i++)
	for(j = 0; j < height; j++)
	for(z = 0; z < 3; z++) {
		c = 0;
		for(k = 0; k < n; k++) {
			l = i + k - n / 2;
			if(0 <= l && l < width) {
				c += kernel[k] * img[(z * height + j) * width + l];
			}
		}
		result[(z * height + j) * width + i] = APPROX_DIVIDE2(c, n - 1);
	}
}
void kernel2a(unsigned short *img, int width, int height, int n, int *kernel, unsigned short *result) {
	int i, j, z, k, l, c;
	for(i = 0; i < width; i++)
	for(j = 0; j < height; j++)
	for(z = 0; z < 3; z++) {
		c = 0;
		for(k = 0; k < n; k++) {
			l = j + k - n / 2;
			if(0 <= l && l < height) {
				c += kernel[k] * img[(z * height + l) * width + i];
			}
		}
		result[(z * height + j) * width + i] = APPROX_DIVIDE2(c, n - 1);
	}
}

void kernel2b(unsigned short *img, int width, int height, int n, int *kernel, stbi_uc *result) {
	int i, j, z, k, l, c;
	for(i = 0; i < width; i++)
	for(j = 0; j < height; j++)
	for(z = 0; z < 3; z++) {
		c = 0;
		for(k = 0; k < n; k++) {
			l = j + k - n / 2;
			if(0 <= l && l < height) {
				c += kernel[k] * img[(z * height + l) * width + i];
			}
		}
		result[(j * width + i) * 3 + z] = APPROX_DIVIDE1(c, n + 7);
	}
}

void blur(int n, int width, int height, stbi_uc *img, unsigned short *aux1, unsigned short *aux2) {
	int *kernel1;
	int *kernel2;
	int n_init;
	int i;
	if(n <= 15 || (n - 1) % 14 == 0) {
		n_init = n;
	} else {
		n_init = ((n - 1) % 14) + 1;
	}
	kernel1 = (int*)malloc(sizeof(int) * n_init);
	kernel2 = (int*)malloc(sizeof(int) * 15);
	pascal(kernel1, n_init);
	pascal(kernel2, 15);
	kernel1a(img, width, height, n_init, kernel1, aux1);
	for(i = n_init; i < (n - 1); i += 14) {
		kernel2a(aux1, width, height, 15, kernel2, aux2);
		kernel1b(aux2, width, height, 15, kernel2, aux1);
	}
	kernel2b(aux1, width, height, n_init, kernel1, img);
	free(kernel1);
	free(kernel2);
}

double test_blur_time(int n, int width, int height, stbi_uc *img_d, unsigned short *aux1_d, unsigned short *aux2_d) {
	clock_t begin = clock();
	blur(n, width, height, img_d, aux1_d, aux2_d);
	clock_t end = clock();
	return (double)(end - begin) / CLOCKS_PER_SEC;
}

int main(void) {
	int nk = N;
	int *ns = (int*)malloc(sizeof(int) * nk);
	int i;
	for(i = 0; i < nk; i++) {
		ns[i] = (1 << (i + 1)) + 1;
	}
	const char fname[] = "./CmakeProject/img2.png";
	int width, height, chn;
	stbi_uc *img = stbi_load(fname, &width, &height, &chn, 3);
	stbi_uc *img_c;
	if(WIDTH != 0) {
		stbi_uc *img_r = (stbi_uc *)malloc(sizeof(stbi_uc) * WIDTH * HEIGHT * 3);
		stbir_resize_uint8(img, width, height, 0, img_r, WIDTH, HEIGHT, 0, 3);
		width = WIDTH;
		height = HEIGHT;
		img = img_r;
	}
	img_c = (stbi_uc*)malloc(sizeof(stbi_uc) * width * height * 3);
	unsigned short *aux1, *aux2;
	aux1 = (unsigned short*)malloc(sizeof(unsigned short) * width * height * 3);
	aux2 = (unsigned short*)malloc(sizeof(unsigned short) * width * height * 3);
	printf("Size of image: %dx%d\n", width, height);
	for(i = 0; i < nk; i++) {
		memcpy(img_c, img, sizeof(stbi_uc) * width * height * 3);
		printf("Blurring with kernel size %d...", ns[i]);
		double time = test_blur_time(ns[i], width, height, img_c, aux1, aux2);
		printf(" Blurred in %f seconds!\n", time);
		if(i == SAVED) {
			memcpy(img, img_c, sizeof(stbi_uc) * width * height * 3);
			const char fname2[] = "image2.bmp";
			stbi_write_bmp(fname2, width, height, 3, img);
		}
	}
	return 0;
}
