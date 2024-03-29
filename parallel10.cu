﻿#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <cstdlib>
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"
#include <unistd.h>

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
#ifndef SAVE_OUTPUT
#define SAVE_OUTPUT 0
#endif
#ifdef __cplusplus
//#ifndef _MSC_VER
#define restrict __restrict__
//#endif
#endif
#ifndef IMGPATH
#define IMGPATH "../../../img2.png"
#endif
#ifndef IT
#define IT 1
#endif

__constant__ int filter1_d[9];
__constant__ int filter2_d[9];

void pascal(int *p, int n) {
	n--;
	p[0] = 1;
	for(int k = 0; k < (n >> 1); k++) {
		p[k + 1] = p[k] * (n - k) / (k + 1);
	}
}

__global__
void kernel1a(const stbi_uc *restrict img, int width, int height, size_t result_pitc, size_t img_pitch, int n, unsigned short *restrict result) {
	int i, j, z, k, l, m, c;
	z = blockIdx.x;
	i = blockIdx.y * blockDim.y + threadIdx.y;
	j = blockIdx.z * blockDim.z + threadIdx.z;
	if(i < width && j < height) {
		c = 0;
		for(k = 0; k < n >> 1; k++) {
			m = 0;
			l = i + k - n / 2;
			if(0 <= l && l < width) {
				m = img[(j * img_pitch + l * 3) + z];
			}
			l = i + n - 1 - k - n / 2;
			if(0 <= l && l < width) {
				m += img[(j * img_pitch + l * 3) + z];
			}
			c += filter1_d[k] * m;
		}
		l = i + k - n / 2;
		if(0 <= l && l < width) {
			c += filter1_d[k] * img[(j * img_pitch + l * 3) + z];
		}
		result[(z * height + j) * result_pitc + i] = APPROX_DIVIDE1(c, n - 9);
	}
}

__global__
void kernel1b(const unsigned short *restrict img, int width, int height, size_t result_pitc, size_t img_pitch, int n, unsigned short *restrict result) {
	int i, j, z, k, l, c, m;
	z = blockIdx.x;
	i = blockIdx.y * blockDim.y + threadIdx.y;
	j = blockIdx.z * blockDim.z + threadIdx.z;
	if(i < width && j < height) {
		c = 0;
		for(k = 0; k < n >> 1; k++) {
			l = i + k - n / 2;
			m = 0;
			if(0 <= l && l < width) {
				m = img[(z * height + j) * img_pitch + l];
			}
			l = i + n - 1 - k - n / 2;
			if(0 <= l && l < width) {
				m += img[(z * height + j) * img_pitch + l];
			}
			c += filter2_d[k] * m;
		}
		l = i + k - n / 2;
		if(0 <= l && l < width) {
			c += filter2_d[k] * img[(z * height + j) * img_pitch + l];
		}
		result[(z * height + j) * result_pitc + i] = APPROX_DIVIDE2(c, n - 1);
	}
}

__global__
void kernel2a(const unsigned short *img, int width, int height, size_t result_pitc, size_t img_pitch, int n, unsigned short *restrict result) {
	int i, j, z, k, l, m, c;
	z = blockIdx.x;
	i = blockIdx.y * blockDim.y + threadIdx.y;
	j = blockIdx.z * blockDim.z + threadIdx.z;
	if(i < width && j < height) {
		c = 0;
		for(k = 0; k < n >> 1; k++) {
			l = j + k - n / 2;
			m = 0;
			if(0 <= l && l < height) {
				m = img[(z * height + l) * img_pitch + i];
			}
			l = j + n - 1 - k - n / 2;
			if(0 <= l && l < height) {
				m += img[(z * height + l) * img_pitch + i];
			}
			c += filter2_d[k] * m;
		}
		l = j + k - n / 2;
		if(0 <= l && l < height) {
			c += filter2_d[k] * img[(z * height + l) * img_pitch + i];
		}
		result[(z * height + j) * result_pitc + i] = APPROX_DIVIDE2(c, n - 1);
	}
}

__global__
void kernel2b(const unsigned short *restrict img, int width, int height, size_t result_pitc, size_t img_pitch, int n, stbi_uc *restrict result) {
	int i, j, z, k, l, m, c;
	z = blockIdx.x;
	i = blockIdx.y * blockDim.y + threadIdx.y;
	j = blockIdx.z * blockDim.z + threadIdx.z;
	if(i < width && j < height) {
		c = 0;
		for(k = 0; k < n >> 1; k++) {
			l = j + k - n / 2;
			m = 0;
			if(0 <= l && l < width) {
				m = img[(z * height + l) * img_pitch + i];
			}
			l = j + n - 1 - k - n / 2;
			if(0 <= l && l < width) {
				m += img[(z * height + l) * img_pitch + i];
			}
			c += filter1_d[k] * m;
		}
		l = j + k - n / 2;
		if(0 <= l && l < width) {
			c += filter1_d[k] * img[(z * height + l) * img_pitch + i];
		}
		result[(j * result_pitc + i * 3) + z] = APPROX_DIVIDE2(c, n + 7);
	}
}

__global__ void ki() {
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	float ia, ib;
	ia = ib = 0.0f;
	ib += ia + tid;
}

void blur(int n, int width, int height, stbi_uc *restrict img) {
	int *restrict filter1, *restrict filter2, n_init, i;
	unsigned short *restrict aux1_d, *restrict aux2_d;
	stbi_uc *restrict img_d;
	dim3 blocks(3, (width + 31) / 32, (height + 31) / 32);
	dim3 threadsPerBlock(1, 32, 32);
	size_t aux1_pitch, aux2_pitch, img_pitch;

	if(n <= 17 || (n - 1) % 16 == 0) {
		n_init = 17;
	} else {
		n_init = ((n - 1) % 16) + 1;
	}
	filter1 = (int *)malloc(sizeof(int) * ((n_init >> 1) + 1));
	filter2 = (int *)malloc(sizeof(int) * 9);
	cudaMallocPitch((void **)&aux1_d, &aux1_pitch, sizeof(unsigned short) * width, height * 3);
	aux1_pitch /= sizeof(unsigned short);
	cudaMallocPitch((void **)&aux2_d, &aux2_pitch, sizeof(unsigned short) * width, height * 3);
	aux2_pitch /= sizeof(unsigned short);
	cudaMallocPitch((void **)&img_d, &img_pitch, sizeof(stbi_uc) * width * 3, height);
	pascal(filter1, n_init);
	pascal(filter2, 17);
	cudaMemcpyToSymbol(filter1_d, filter1, sizeof(int) * ((n_init >> 1) + 1));
	cudaMemcpyToSymbol(filter2_d, filter2, sizeof(int) * 9);
	cudaMemcpy2D(img_d, img_pitch, img, sizeof(stbi_uc) * width * 3, sizeof(stbi_uc) * width * 3, height, cudaMemcpyHostToDevice);
	kernel1a << <blocks, threadsPerBlock >> > (img_d, width, height, aux1_pitch, img_pitch / sizeof(stbi_uc), n_init, aux1_d);
	for(i = n_init; i < (n - 1); i += 16) {
		kernel2a << <blocks, threadsPerBlock >> > (aux1_d, width, height, aux2_pitch, aux1_pitch, 17, aux2_d);
		kernel1b << <blocks, threadsPerBlock >> > (aux2_d, width, height, aux1_pitch, aux2_pitch, 17, aux1_d);
	}
	kernel2b << <blocks, threadsPerBlock >> > (aux1_d, width, height, img_pitch / sizeof(stbi_uc), aux1_pitch, n_init, img_d);
	cudaMemcpy2D(img, sizeof(stbi_uc) * width * 3, img_d, img_pitch, sizeof(stbi_uc) * width * 3, height, cudaMemcpyDeviceToHost);
	free(filter1);
	free(filter2);
	cudaFree(aux1_d);
	cudaFree(aux2_d);
	cudaFree(img_d);
	cudaDeviceSynchronize();
}

double test_blur_time(int n, int width, int height, stbi_uc *img) {
	int i;
	clock_t begin, end;

	begin = clock();
	for(i = 0; i < IT; i++) {
		blur(n, width, height, img);
	}
	end = clock();
	return (double)(end - begin) / CLOCKS_PER_SEC;
}

int main(void) {
	int nk, *restrict ns, i, width, height, chn, f;
	stbi_uc *restrict img, *restrict img_c, *restrict img_r;
	double time;
	const char fname[] = IMGPATH;
	const char fname2[] = "image2.bmp";

	printf("Parallel version - no constant memory - no shared memory\n");
	nk = N;
	ns = (int *)malloc(sizeof(int) * nk);
	for(i = 0; i < nk; i++) {
		ns[i] = (1 << (i + 1)) + 1;
	}
	img = stbi_load(fname, &width, &height, &chn, 3);
	if(WIDTH != 0) {
		img_r = (stbi_uc *)malloc(sizeof(stbi_uc) * WIDTH * HEIGHT * 3);
		stbir_resize_uint8(img, width, height, 0, img_r, WIDTH, HEIGHT, 0, 3);
		free(img);
		width = WIDTH;
		height = HEIGHT;
		img = img_r;
	}
	img_c = (stbi_uc *)malloc(sizeof(stbi_uc) * width * height * 3);
	printf("Size of image: %dx%d\n", width, height);

	dim3 blocks(3, (width + 31) / 32, (height + 31) / 32);
	dim3 threadsPerBlock(1, 32, 32);
	ki << <blocks, threadsPerBlock >> > ();
	cudaDeviceSynchronize();

	f = 0;
	for(i = 0; i < nk && !f; i++) {
		memcpy(img_c, img, sizeof(stbi_uc) * width * height * 3);
		printf("Blurring with kernel size %d...", ns[i]);
		time = test_blur_time(ns[i], width, height, img_c);
		if(time < 0) {
			time = -time;
			f = 1;
		}
		printf(" Blurred in %f seconds!\n", time);
		if(SAVE_OUTPUT) {
			memcpy(img, img_c, sizeof(stbi_uc) * width * height * 3);
			stbi_write_bmp(fname2, width, height, 3, img);
		}
		sleep(5);
	}
	printf("\n");
	free(ns);
	free(img);
	free(img_c);
	return 0;
}
