﻿#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <cstdlib>
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"

#define S_R_SHIFT(A, B)	(((B) >= 0) ? ((A) >> (B)) : (A) << -(B))
#define APPROX_DIVIDE1(A, B) (S_R_SHIFT(A, B) + (S_R_SHIFT(A, (B) - 1) & 1))
#define APPROX_DIVIDE2(A, B) (((A) >> (B)) + (((A) >> ((B) - 1)) & 1))
#ifndef N
#define N 11
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
	n--;
	p[0] = 1;
	for(int k = 0; k < n; k++) {
		p[k + 1] = p[k] * (n - k) / (k + 1);
	}
}

void checkCudaErrors(cudaError_t error) {

}

__global__
void kernel1a(const stbi_uc *img, int width, int height, int n, int *kernel, unsigned short *result) {
	int i, j, z, k, l, c;
	i = blockIdx.x * blockDim.x + threadIdx.x;
	j = blockIdx.y * blockDim.y + threadIdx.y;
	z = blockIdx.z;
	if(i < width && j < height) {
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

__global__
void kernel1b(unsigned short *img, int width, int height, int n, int *kernel, unsigned short *result) {
	int i, j, z, k, l, c;
	i = blockIdx.x * blockDim.x + threadIdx.x;
	j = blockIdx.y * blockDim.y + threadIdx.y;
	z = blockIdx.z;
	if(i < width && j < height) {
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

__global__
void kernel2a(unsigned short *img, int width, int height, int n, int *kernel, unsigned short *result) {
	int i, j, z, k, l, c;
	i = blockIdx.x * blockDim.x + threadIdx.x;
	j = blockIdx.y * blockDim.y + threadIdx.y;
	z = blockIdx.z;
	if(i < width && j < height) {
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

__global__
void kernel2b(unsigned short *img, int width, int height, int n, int *kernel, stbi_uc *result) {
	int i, j, z, k, l, c;
	i = blockIdx.x * blockDim.x + threadIdx.x;
	j = blockIdx.y * blockDim.y + threadIdx.y;
	z = blockIdx.z;
	if(i < width && j < height) {
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

void applyKernel1(const stbi_uc *img, int width, int height, int n, int *kernel, int *result) {
	int i, j, k, l;
	for(i = 0; i < width; i++) {
		for(j = 0; j < height; j++) {
			result[(j * width + i) * 3] = 0;
			result[(j * width + i) * 3 + 1] = 0;
			result[(j * width + i) * 3 + 2] = 0;
			for(k = 0; k < n; k++) {
				l = i + k - n / 2;
				if(0 <= l && l < width) {
					result[(j * width + i) * 3] += kernel[k] * img[(j * width + l) * 3];
					result[(j * width + i) * 3 + 1] += kernel[k] * img[(j * width + l) * 3 + 1];
					result[(j * width + i) * 3 + 2] += kernel[k] * img[(j * width + l) * 3 + 2];
				}
			}
		}
	}
}

void applyKernel2(int *img, int width, int height, int n, int *kernel, stbi_uc *result) {
	int i, j, k, l, r, g, b;
	for(i = 0; i < width; i++) {
		for(j = 0; j < height; j++) {
			r = 0;
			g = 0;
			b = 0;
			for(k = 0; k < n; k++) {
				l = j + k - n / 2;
				if(0 <= l && l < height) {
					r += kernel[k] * img[(l * width + i) * 3];
					g += kernel[k] * img[(l * width + i) * 3 + 1];
					b += kernel[k] * img[(l * width + i) * 3 + 2];
				}
			}
			result[(j * width + i) * 3] = r >> (n - 1 + n - 1);
			result[(j * width + i) * 3 + 1] = g >> (n - 1 + n - 1);
			result[(j * width + i) * 3 + 2] = b >> (n - 1 + n - 1);
		}
	}
}

__global__
void saxpy(int n, stbi_uc *y) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < n) y[i] = 0;
}

void blur(int n, int width, int height, stbi_uc *img_d, unsigned short *aux1_d, unsigned short *aux2_d) {
	int *kernel1;
	int *kernel2;
	int n_init;
	int *kernel1_d;
	int *kernel2_d;
	if(n <= 15 || (n - 1) % 14 == 0) {
		n_init = 15;
	} else {
		n_init = ((n - 1) % 14) + 1;
	}
	kernel1 = (int*)malloc(sizeof(int) * n_init);
	kernel2 = (int*)malloc(sizeof(int) * 15);
	pascal(kernel1, n_init);
	pascal(kernel2, 15);
	dim3 blocks((width + 14) / 15, (height + 14) / 15, 3);
	dim3 threadsPerBlock(15, 15, 1);
	cudaMalloc(&kernel1_d, sizeof(int) * n_init);
	cudaMalloc(&kernel2_d, sizeof(int) * 15);
	cudaMemcpy(kernel1_d, kernel1, sizeof(int) * n_init, cudaMemcpyHostToDevice);
	cudaMemcpy(kernel2_d, kernel2, sizeof(int) * 15, cudaMemcpyHostToDevice);
	kernel1a<<<blocks, threadsPerBlock>>>(img_d, width, height, n_init, kernel1_d, aux1_d);
	for(int i = n_init; i < (n - 1); i += 14) {
		kernel2a<<<blocks, threadsPerBlock>>>(aux1_d, width, height, 15, kernel2_d, aux2_d);
		kernel1b<<<blocks, threadsPerBlock>>>(aux2_d, width, height, 15, kernel2_d, aux1_d);
		cudaDeviceSynchronize();
	}
	kernel2b<<<blocks, threadsPerBlock>>>(aux1_d, width, height, n_init, kernel1_d, img_d);
	cudaFree(kernel1_d);
	cudaFree(kernel2_d);
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
	for(int i = 0; i < nk; i++) {
		ns[i] = (1 << (i + 1)) + 1;
	}
	const char fname[] = "../../../img2.png";
	int width, height, chn;
	stbi_uc *img = stbi_load(fname, &width, &height, &chn, 3);
	stbi_uc *img_d;
	if(WIDTH != 0) {
		stbi_uc *img_r = (stbi_uc *)malloc(sizeof(stbi_uc) * WIDTH * HEIGHT * 3);
		stbir_resize_uint8(img, width, height, 0, img_r, WIDTH, HEIGHT, 0, 3);
		width = WIDTH;
		height = HEIGHT;
		img = img_r;
	}
	checkCudaErrors(cudaMalloc(&img_d, sizeof(stbi_uc) * width * height * 3));
	unsigned short *aux1_d, *aux2_d;
	checkCudaErrors(cudaMalloc(&aux1_d, sizeof(unsigned short) * width * height * 3));
	checkCudaErrors(cudaMalloc(&aux2_d, sizeof(unsigned short) * width * height * 3));
	printf("Size of image: %dx%d\n", width, height);
	for(int i = 0; i < nk; i++) {
		checkCudaErrors(cudaMemcpy(img_d, img, sizeof(stbi_uc) * width * height * 3, cudaMemcpyHostToDevice));
		cudaDeviceSynchronize();
		printf("Blurring with kernel size %d...", ns[i]);
		double time = test_blur_time(ns[i], width, height, img_d, aux1_d, aux2_d);
		printf(" Blurred in %f seconds!\n", time);
		if(i == SAVED) {
			checkCudaErrors(cudaMemcpy(img, img_d, sizeof(stbi_uc) * width * height * 3, cudaMemcpyDeviceToHost));
			const char fname2[] = "image2.bmp";
			stbi_write_bmp(fname2, width, height, 3, img);
		}
	}
	cudaFree(aux1_d);
	cudaFree(aux2_d);
	cudaFree(img_d);
	cudaError_t b = cudaGetLastError();
	return 0;
}
