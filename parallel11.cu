#define STB_IMAGE_IMPLEMENTATION
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
#define NBLOCK 8
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
void kernel1a(const stbi_uc *restrict img, int width, int height, size_t result_pitc, size_t img_pitch, int n, int nblock, unsigned short *restrict result) {
	int i, j, z, k, l, c, b, m, f;
	extern __shared__ unsigned short tile[];
	z = blockIdx.x;
	j = blockIdx.z * blockDim.z + threadIdx.z;
	for(b = 0; b < nblock; b++) {
		i = (blockIdx.y * nblock + b) * blockDim.y + threadIdx.y;
		tile[threadIdx.z * (n - 1 + nblock * blockDim.y) + (n >> 1) + threadIdx.y + blockDim.y * b] = img[(j * img_pitch + i * 3) + z];
	}
	if(threadIdx.y < n - 1) {
		int aux = (threadIdx.y < n >> 1) ? blockIdx.y * nblock * blockDim.y + threadIdx.y - (n >> 1) : (threadIdx.y - (n >> 1)) + blockDim.y * nblock * (blockIdx.y + 1);
		int aux2 = (threadIdx.y < n >> 1) ? 0 : blockDim.y * nblock;
		tile[threadIdx.z * (n - 1 + nblock * blockDim.y) + threadIdx.y + aux2] = (0 <= aux && aux < width) ? img[(j * img_pitch + aux * 3) + z] : 0;
	}
	if(blockDim.y <= 32) {
		f = threadIdx.y;
	} else {
		f = threadIdx.y - 32;
	}
	if(threadIdx.z == 0 && f >= 0 && f < (n >> 1) + 1) {
		tile[blockDim.z * (blockDim.y * nblock + n - 1) + f] = filter1_d[f];
	}
	__syncthreads();
	for(b = 0; b < nblock; b++) {
		i = (blockIdx.y * nblock + b) * blockDim.y + threadIdx.y;
		if(i < width && j < height) {
			c = 0;
			for(k = 0; k < n >> 1; k++) {
				c += tile[blockDim.z * (blockDim.y * nblock + n - 1) + k] * (tile[threadIdx.z * (n - 1 + nblock * blockDim.y) + b * blockDim.y + threadIdx.y + k] + tile[threadIdx.z * (n - 1 + NBLOCK * blockDim.y) + b * blockDim.y + threadIdx.y + n - 1 - k]);
			}
			c += tile[blockDim.z * (blockDim.y * nblock + n - 1) + k] * tile[threadIdx.z * (n - 1 + nblock * blockDim.y) + b * blockDim.y + threadIdx.y + k];
			result[(z * height + j) * result_pitc + i] = APPROX_DIVIDE1(c, n - 9);
		}
	}
}

__global__
void kernel1b(unsigned short *img, int width, int height, size_t result_pitc, size_t img_pitch, int n, int nblock, unsigned short *restrict result) {
	int i, j, z, k, l, c, b, m, f;
	extern __shared__ unsigned short tile[];
	z = blockIdx.x;
	j = blockIdx.z * blockDim.z + threadIdx.z;
	for(b = 0; b < nblock; b++) {
		i = (blockIdx.y * nblock + b) * blockDim.y + threadIdx.y;
		tile[threadIdx.z * (n - 1 + nblock * blockDim.y) + (n >> 1) + threadIdx.y + blockDim.y * b] = img[(z * height + j) * img_pitch + i];
	}
	if(threadIdx.y < n - 1) {
		int aux = (threadIdx.y < n >> 1) ? blockIdx.y * nblock * blockDim.y + threadIdx.y - (n >> 1) : (threadIdx.y - (n >> 1)) + blockDim.y * nblock * (blockIdx.y + 1);
		int aux2 = (threadIdx.y < n >> 1) ? 0 : blockDim.y * nblock;
		tile[threadIdx.z * (n - 1 + nblock * blockDim.y) + threadIdx.y + aux2] = (0 <= aux && aux < width) ? img[(z * height + j) * img_pitch + aux] : 0;
	}
	if(blockDim.y <= 32) {
		f = threadIdx.y;
	} else {
		f = threadIdx.y - 32;
	}
	if(threadIdx.z == 0 && f >= 0 && f < (n >> 1) + 1) {
		tile[blockDim.z * (blockDim.y * nblock + n - 1) + f] = filter2_d[f];
	}
	__syncthreads();
	for(b = 0; b < nblock; b++) {
		i = (blockIdx.y * nblock + b) * blockDim.y + threadIdx.y;
		if(i < width && j < height) {
			c = 0;
			for(k = 0; k < n >> 1; k++) {
				c += tile[blockDim.z * (blockDim.y * nblock + n - 1) + k] * (tile[threadIdx.z * (n - 1 + nblock * blockDim.y) + b * blockDim.y + threadIdx.y + k] + tile[threadIdx.z * (n - 1 + NBLOCK * blockDim.y) + b * blockDim.y + threadIdx.y + n - 1 - k]);
			}
			c += tile[blockDim.z * (blockDim.y * nblock + n - 1) + k] * tile[threadIdx.z * (n - 1 + nblock * blockDim.y) + b * blockDim.y + threadIdx.y + k];
			result[(z * height + j) * result_pitc + i] = APPROX_DIVIDE2(c, n - 1);
		}
	}
}

__global__
void kernel2a(unsigned short *img, int width, int height, size_t result_pitc, size_t img_pitch, int n, int nblock, unsigned short *restrict result) {
	int i, j, z, k, l, c, b;
	extern __shared__ unsigned short tile[];
	z = blockIdx.x;
	i = blockIdx.y * blockDim.y + threadIdx.y;
	for(b = 0; b < nblock; b++) {
		j = (blockIdx.z * nblock + b) * blockDim.z + threadIdx.z;
		tile[(threadIdx.z + (n >> 1) + blockDim.z * b) * blockDim.y + threadIdx.y] = img[(z * height + j) * img_pitch + i];
	}
	if(!((n >> 1) <= threadIdx.z && threadIdx.z < blockDim.z - (n >> 1))) {
		int aux = (threadIdx.z < n >> 1) ? blockIdx.z * nblock * blockDim.z + threadIdx.z - (n >> 1) : j + (n >> 1);
		int aux2 = (threadIdx.z < n >> 1) ? 0 : n - 1 + blockDim.z * (nblock - 1);
		tile[(threadIdx.z + aux2) * blockDim.y + threadIdx.y] = (0 <= aux && aux < height) ? img[(z * height + aux) * img_pitch + i] : 0;
	} else if(threadIdx.z == (n >> 1) + 1 && threadIdx.y < (n >> 1) + 1) {
		tile[blockDim.y * (blockDim.z * nblock + n - 1) + threadIdx.y] = filter2_d[threadIdx.y];
	}
	__syncthreads();
	for(b = 0; b < nblock; b++) {
		j = (blockIdx.z * nblock + b) * blockDim.z + threadIdx.z;
		if(i < width && j < height) {
			c = 0;
			for(k = 0; k < n >> 1; k++) {
				c += tile[blockDim.y * (blockDim.z * nblock + n - 1) + k] * (tile[(threadIdx.z + k + blockDim.z * b) * blockDim.y + threadIdx.y] + tile[(threadIdx.z + n - 1 - k + blockDim.z * b) * blockDim.y + threadIdx.y]);
			}
			c += tile[blockDim.y * (blockDim.z * nblock + n - 1) + k] * tile[(threadIdx.z + k + blockDim.z * b) * blockDim.y + threadIdx.y];
			result[(z * height + j) * result_pitc + i] = APPROX_DIVIDE2(c, n - 1);
		}
	}
}

__global__
void kernel2b(const unsigned short *restrict img, int width, int height, size_t result_pitc, size_t img_pitch, int n, int nblock, stbi_uc *restrict result) {
	int i, j, z, k, l, c, b;
	extern __shared__ unsigned short tile[];
	z = blockIdx.x;
	i = blockIdx.y * blockDim.y + threadIdx.y;
	for(b = 0; b < nblock; b++) {
		j = (blockIdx.z * nblock + b) * blockDim.z + threadIdx.z;
		tile[(threadIdx.z + (n >> 1) + blockDim.z * b) * blockDim.y + threadIdx.y] = img[(z * height + j) * img_pitch + i];
	}
	if(!((n >> 1) <= threadIdx.z && threadIdx.z < blockDim.z - (n >> 1))) {
		int aux = (threadIdx.z < n >> 1) ? blockIdx.z * nblock * blockDim.z + threadIdx.z - (n >> 1) : j + (n >> 1);
		int aux2 = (threadIdx.z < n >> 1) ? 0 : n - 1 + blockDim.z * (nblock - 1);
		tile[(threadIdx.z + aux2) * blockDim.y + threadIdx.y] = (0 <= aux && aux < height) ? img[(z * height + aux) * img_pitch + i] : 0;
	} else if(threadIdx.z == (n >> 1) + 1 && threadIdx.y < (n >> 1) + 1) {
		tile[blockDim.y * (blockDim.z * nblock + n - 1) + threadIdx.y] = filter1_d[threadIdx.y];
	}
	__syncthreads();
	for(b = 0; b < nblock; b++) {
		j = (blockIdx.z * nblock + b) * blockDim.z + threadIdx.z;
		if(i < width && j < height) {
			c = 0;
			for(k = 0; k < n >> 1; k++) {
				c += tile[blockDim.y * (blockDim.z * nblock + n - 1) + k] * (tile[(threadIdx.z + k + blockDim.z * b) * blockDim.y + threadIdx.y] + tile[(threadIdx.z + n - 1 - k + blockDim.z * b) * blockDim.y + threadIdx.y]);
			}
			c += tile[blockDim.y * (blockDim.z * nblock + n - 1) + k] * tile[(threadIdx.z + k + blockDim.z * b) * blockDim.y + threadIdx.y];
			result[(j * result_pitc + i * 3) + z] = APPROX_DIVIDE2(c, n + 7);
		}
	}
}

__global__ void ki() {
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	float ia, ib;
	ia = ib = 0.0f;
	ib += ia + tid;
}

void blur(int n, int width, int height, stbi_uc *restrict img) {
	int *restrict filter1, *restrict filter2, n_init, i, nBlocksV, nBlocksH, threadsH;
	unsigned short *restrict aux1_d, *restrict aux2_d;
	stbi_uc *restrict img_d;
	size_t aux1_pitch, aux2_pitch, img_pitch;

	nBlocksV = NBLOCK;
	if(width < 32 * NBLOCK) {
		nBlocksV = width / 32;
	}
	threadsH = 256;
	if(width < 256) {
		threadsH = width;
	}
	nBlocksH = NBLOCK;
	if(width < threadsH * NBLOCK) {
		nBlocksH = width / threadsH;
	}
	dim3 blocks1(3, (width + threadsH * nBlocksH - 1) / (threadsH * nBlocksH), height);
	dim3 blocks2(3, (width + 31) / 32, (height + 31 * nBlocksV - 1) / (32 * nBlocksV));
	dim3 threadsPerBlock1(1, threadsH, 1);
	dim3 threadsPerBlock2(1, 32, 32);

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
	cudaMalloc((void **)&filter1_d, sizeof(int) * ((n_init >> 1) + 1));
	cudaMalloc((void **)&filter2_d, sizeof(int) * 9);
	cudaMemcpyToSymbol(filter1_d, filter1, sizeof(int) * ((n_init >> 1) + 1));
	cudaMemcpyToSymbol(filter2_d, filter2, sizeof(int) * 9);
	cudaMemcpy2D(img_d, img_pitch, img, sizeof(stbi_uc) * width * 3, sizeof(stbi_uc) * width * 3, height, cudaMemcpyHostToDevice);
	kernel1a << <blocks1, threadsPerBlock1, sizeof(unsigned short) *(threadsH * nBlocksH + 16 + (n_init >> 1)) >> > (img_d, width, height, aux1_pitch, img_pitch / sizeof(stbi_uc), n_init, nBlocksH, aux1_d);
	for(i = n_init; i < (n - 1); i += 16) {
		kernel2a << <blocks2, threadsPerBlock2, sizeof(unsigned short) *(48 * (32 * nBlocksV + 16) + 9) >> > (aux1_d, width, height, aux2_pitch, aux1_pitch, 17, nBlocksV, aux2_d);
		kernel1b << <blocks1, threadsPerBlock1, sizeof(unsigned short) *(threadsH * nBlocksH + 25) >> > (aux2_d, width, height, aux1_pitch, aux2_pitch, 17, nBlocksH, aux1_d);
	}
	kernel2b << <blocks2, threadsPerBlock2, sizeof(unsigned short) *((32 + n_init - 1) * (32 * nBlocksV + n_init - 1) + (n_init >> 1)) >> > (aux1_d, width, height, img_pitch / sizeof(stbi_uc), aux1_pitch, n_init, nBlocksV, img_d);
	cudaMemcpy2D(img, sizeof(stbi_uc) * width * 3, img_d, img_pitch, sizeof(stbi_uc) * width * 3, height, cudaMemcpyDeviceToHost);
	free(filter1);
	free(filter2);
	cudaFree(aux1_d);
	cudaFree(aux2_d);
	cudaFree(img_d);
	cudaDeviceSynchronize();
	cudaError_t a = cudaGetLastError();
}

double test_blur_time(int n, int width, int height, stbi_uc *img) {
	clock_t begin, end;

	begin = clock();
	for(i = 0; i < IT; i++) {
		blur(n, width, height, img);
	}
	end = clock();
	return (double)(end - begin) / CLOCKS_PER_SEC;
}

int main(void) {
	int nk, *restrict ns, i, width, height, chn, f, nBlocksV, nBlocksH, threadsH;
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

	nBlocksV = NBLOCK;
	if(width < 32 * NBLOCK) {
		nBlocksV = width / 32;
	}
	threadsH = 256;
	if(width < 256) {
		threadsH = width;
	}
	nBlocksH = NBLOCK;
	if(width < threadsH * NBLOCK) {
		nBlocksH = width / threadsH;
	}
	dim3 blocks1(3, (width + threadsH * nBlocksH - 1) / (threadsH * nBlocksH), height);
	dim3 blocks2(3, (width + 31) / 32, (height + 31 * nBlocksV - 1) / (32 * nBlocksV));
	dim3 threadsPerBlock1(1, threadsH, 1);
	dim3 threadsPerBlock2(1, 32, 32);
	ki << <threadsPerBlock1, threadsPerBlock1 >> > ();
	ki << <threadsPerBlock2, threadsPerBlock2 >> > ();
	cudaDeviceSynchronize();

	f = 0;
	for(i = 0; i < nk && !f; i++) {
		memcpy(img_c, img, sizeof(stbi_uc) * width * height * 3);
		printf("Blurring with filter size %d...", ns[i]);
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
	}
	printf("\n");
	free(ns);
	free(img);
	free(img_c);
	return 0;
}
