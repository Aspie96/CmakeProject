#define STB_IMAGE_IMPLEMENTATION
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
#define NBLOCK 8
#define NBLOCKH 2
#define NBLOCKH 2

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
	int i, j, z, k, l, c, b;
	extern __shared__ unsigned short tile[];
	j = blockIdx.y * blockDim.y + threadIdx.y;
	z = blockIdx.z;
	for(b = 0; b < NBLOCKH; b++) {
		i = (blockIdx.x * NBLOCKH + b) * blockDim.x + threadIdx.x;
		tile[threadIdx.y * (n - 1 + NBLOCKH * blockDim.x) + (n >> 1) + threadIdx.x + blockDim.x * b] = img[(z * height + j) * width + i];
	}
	if(!((n >> 1) <= threadIdx.x && threadIdx.x < blockDim.x - (n >> 1))) {
		/*int aux = (threadIdx.x < n >> 1) ? (blockIdx.x + 1) * NBLOCKH * blockDim.x + threadIdx.x : (blockIdx.x * NBLOCKH - 1) * blockDim.x + threadIdx.x;
		int aux2 = (threadIdx.x < n >> 1) ? (n >> 1) + blockDim.x * NBLOCKH : (n >> 1) - blockDim.x;*/
		int aux = (threadIdx.x < n >> 1) ? blockIdx.x * NBLOCKH * blockDim.x + threadIdx.x - (n >> 1) : i + (n >> 1);
		int aux2 = (threadIdx.x < n >> 1) ? 0 : n - 1 + blockDim.x * (NBLOCKH - 1);
		tile[threadIdx.y * (n - 1 + NBLOCKH * blockDim.x) + threadIdx.x + aux2] = (0 <= aux && aux < width) ? img[(z * height + j) * width + aux] : 0;
	}
	if(threadIdx.y == 0 && threadIdx.x < (n >> 1) + 1) {
		tile[blockDim.y * (blockDim.x * NBLOCKH + n - 1) + threadIdx.x] = kernel[threadIdx.x];
	}
	__syncthreads();
	for(b = 0; b < NBLOCKH; b++) {
		i = (blockIdx.x * NBLOCKH + b) * blockDim.x + threadIdx.x;
		if(i < width && j < height) {
			c = 0;
			for(k = 0; k < n >> 1; k++) {
				c += tile[blockDim.y * (blockDim.x * NBLOCKH + n - 1) + k] * (tile[threadIdx.y * (n - 1 + NBLOCKH * blockDim.x) + b * blockDim.x + threadIdx.x + k] + tile[threadIdx.y * (n - 1 + NBLOCKH * blockDim.x) + b * blockDim.x + threadIdx.x + n - 1 - k]);
			}
			c += tile[blockDim.y * (blockDim.x * NBLOCKH + n - 1) + k] * tile[threadIdx.y * (n - 1 + NBLOCKH * blockDim.x) + b * blockDim.x + threadIdx.x + k];
			result[(z * height + j) * width + i] = APPROX_DIVIDE2(c, n - 1);
		}
	}
}

__global__
void kernel2a(unsigned short *img, int width, int height, int n, int *kernel, int nblock, unsigned short *result) {
	int i, j, z, k, l, c, b;
	extern __shared__ unsigned short tile[];
	i = blockIdx.x * blockDim.x + threadIdx.x;
	z = blockIdx.z;
	for(b = 0; b < nblock; b++) {
		j = (blockIdx.y * nblock + b) * blockDim.y + threadIdx.y;
		tile[(threadIdx.y + (n >> 1) + blockDim.y * b) * blockDim.x + threadIdx.x] = img[(z * height + j) * width + i];
	}
	if(!((n >> 1) <= threadIdx.y && threadIdx.y < blockDim.y - (n >> 1))) {
		int aux = (threadIdx.y < n >> 1) ? blockIdx.y * nblock * blockDim.y + threadIdx.y - (n >> 1) : j + (n >> 1);
		int aux2 = (threadIdx.y < n >> 1) ? 0 : n - 1 + blockDim.y * (nblock - 1);
		tile[(threadIdx.y + aux2) * blockDim.x + threadIdx.x] = (0 <= aux && aux < height) ? img[(z * height + aux) * width + i] : 0;
	} else if(threadIdx.y == (n >> 1) + 1 && threadIdx.x < (n >> 1) + 1) {
		tile[blockDim.x * (blockDim.y * nblock + n - 1) + threadIdx.x] = kernel[threadIdx.x];
	}
	__syncthreads();
	for(b = 0; b < nblock; b++) {
		j = (blockIdx.y * nblock + b) * blockDim.y + threadIdx.y;
		if(i < width && j < height) {
			c = 0;
			for(k = 0; k < n >> 1; k++) {
				c += tile[blockDim.x * (blockDim.y * nblock + n - 1) + k] * (tile[(threadIdx.y + k + blockDim.y * b) * blockDim.x + threadIdx.x] + tile[(threadIdx.y + n - 1 - k + blockDim.y * b) * blockDim.x + threadIdx.x]);
			}
			c += tile[blockDim.x * (blockDim.y * nblock + n - 1) + k] * tile[(threadIdx.y + k + blockDim.y * b) * blockDim.x + threadIdx.x];
			result[(z * height + j) * width + i] = APPROX_DIVIDE2(c, n - 1);
		}
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
		result[(j * width + i) * 3 + z] = APPROX_DIVIDE2(c, n + 7);
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
	int *filter1;
	int *filter2;
	int *filter1_d;
	int *filter2_d;
	int n_init;
	if(n <= 15 || (n - 1) % 14 == 0) {
		n_init = 15;
	} else {
		n_init = ((n - 1) % 14) + 1;
	}
	filter1 = (int *)malloc(sizeof(int) * n_init);
	filter2 = (int *)malloc(sizeof(int) * 15);
	pascal(filter1, n_init);
	pascal(filter2, 15);
	dim3 blocks((width + 31) / 32, (height + 31) / 32, 3);
	dim3 blocks2((width + 31) / 32, (height + 31) / 32 / NBLOCK, 3);
	dim3 blocks3((width + 1024) / 1024 / NBLOCKH, height, 3);
	dim3 threadsPerBlock(32, 32, 1);
	dim3 threadsPerBlock2(1024, 1, 1);
	cudaMalloc(&filter1_d, sizeof(int) * n_init);
	cudaMalloc(&filter2_d, sizeof(int) * 15);
	cudaMemcpy(filter1_d, filter1, sizeof(int) * n_init, cudaMemcpyHostToDevice);
	cudaMemcpy(filter2_d, filter2, sizeof(int) * 15, cudaMemcpyHostToDevice);
	//cudaError_t b = cudaGetLastError();
	kernel1a << <blocks, threadsPerBlock >> > (img_d, width, height, n_init, filter1_d, aux1_d);
	cudaDeviceSynchronize();
	//cudaError_t dd = cudaGetLastError();
	for(int i = n_init; i < (n - 1); i += 14) {
		kernel2a << <blocks2, threadsPerBlock, sizeof(unsigned short) *((32) *(32 * NBLOCK + 15 - 1) + 8) >> > (aux1_d, width, height, 15, filter2_d, NBLOCK, aux2_d);
		cudaDeviceSynchronize();
		kernel1b << <blocks3, threadsPerBlock2, sizeof(unsigned short) *((32) * (32 * NBLOCK + 15 - 1) + 8) >> > (aux2_d, width, height, 15, filter2_d, aux1_d);
		cudaDeviceSynchronize();
	}
	kernel2b << <blocks, threadsPerBlock >> > (aux1_d, width, height, n_init, filter1_d, img_d);
	free(filter1);
	free(filter2);
}

double test_blur_time(int n, int width, int height, stbi_uc *img_d, unsigned short *aux1_d, unsigned short *aux2_d) {
	clock_t begin = clock();
	blur(n, width, height, img_d, aux1_d, aux2_d);
	clock_t end = clock();
	return (double)(end - begin) / CLOCKS_PER_SEC;
}

int main(void) {
	printf("Parallel version - no constant memory - yes shared memory\n");
	int nk = N;
	const char fname[] = "./CmakeProject/img2.png";
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
		int ks = (1 << (i + 1)) + 1;
		printf("Blurring with kernel size %d...", ks);
		double time = test_blur_time(ks, width, height, img_d, aux1_d, aux2_d);
		printf(" Blurred in %f seconds!\n", time);
		if(i == SAVED) {
			checkCudaErrors(cudaMemcpy(img, img_d, sizeof(stbi_uc) * width * height * 3, cudaMemcpyDeviceToHost));
			//cudaError_t b = cudaGetLastError();
			const char fname2[] = "image2.bmp";
			stbi_write_bmp(fname2, width, height, 3, img);
		}
	}
	cudaFree(aux1_d);
	cudaFree(aux2_d);
	cudaFree(img_d);
	cudaError_t b = cudaGetLastError();
	printf("\n");
	return 0;
}
