#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <cstdlib>
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"

#define APPROX_DIVIDE(A, B) (((A) >> (B)) + (((A) >> ((B) - 1)) & 1))
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
	extern __shared__ unsigned short tile[];
	int tileW = blockDim.x + n - 1;
	int tileH = blockDim.y;
	int blockS = blockDim.x * blockDim.y;
	for(k = 0; k < (tileW * tileH) / blockS; k++) {
		int pos = k + (threadIdx.y * blockDim.x + threadIdx.x) * ((tileW * tileH) / blockS);
		int imgX = blockDim.x * blockIdx.x - n / 2 +pos % tileW;
		int imgY = blockDim.y * blockIdx.y+pos / tileW;
		tile[pos] = (0 <= imgX && width > imgX && 0 <= imgY && height > imgY) ? img[(imgY * width + imgX) * 3 + z] << 8 : 0;
	}
	int pos = blockDim.x * blockDim.y * k + threadIdx.y * blockDim.x + threadIdx.x;
	if(pos < tileW * tileH) {
		int imgX = blockDim.x * blockIdx.x - n / 2 + pos % tileW;
		int imgY = blockDim.y * blockIdx.y+ pos / tileW;
		tile[pos] = (0 <= imgX && width > imgX && 0 <= imgY && height > imgY) ? img[(imgY * width + imgX) * 3 + z] << 8 : 0;
	}
	__syncthreads();
	if(i < width && j < height) {
		c = 0;
		for(k = 0; k < n; k++) {
			c += kernel[k] * tile[(threadIdx.y) * tileW + threadIdx.x + k];
		}
		result[(z * height + j) * width + i] = APPROX_DIVIDE(c, n - 1);
	}
}

__global__
void kernel1b(unsigned short *img, int width, int height, int n, int *kernel, unsigned short *result) {
	int i, j, z, k, l, c;
	i = blockIdx.x * blockDim.x + threadIdx.x;
	j = blockIdx.y * blockDim.y + threadIdx.y;
	z = blockIdx.z;
	extern __shared__ unsigned short tile[];
	int tileW = blockDim.x + n - 1;
	int tileH = blockDim.y;
	int blockS = blockDim.x * blockDim.y;
	for(k = 0; k < (tileW * tileH) / blockS; k++) {
		int pos = k + (threadIdx.y * blockDim.x + threadIdx.x) * ((tileW * tileH) / blockS);
		int imgX = blockDim.x * blockIdx.x - n / 2 + pos % tileW;
		int imgY = blockDim.y * blockIdx.y + pos / tileW;
		tile[pos] = (0 <= imgX && width > imgX && 0 <= imgY && height > imgY) ? img[(z * height + imgY) * width + imgX] : 0;
	}
	int pos = blockDim.x * blockDim.y * k + threadIdx.y * blockDim.x + threadIdx.x;
	if(pos < tileW * tileH) {
		int imgX = blockDim.x * blockIdx.x - n / 2 + pos % tileW;
		int imgY = blockDim.y * blockIdx.y + pos / tileW;
		tile[pos] = (0 <= imgX && width > imgX && 0 <= imgY && height > imgY) ? img[(z * height + imgY) * width + imgX] : 0;
	}
	__syncthreads();
	if(i < width && j < height) {
		c = 0;
		for(k = 0; k < n; k++) {
			c += kernel[k] * tile[(threadIdx.y) * tileW + threadIdx.x + k];
		}
		result[(z * height + j) * width + i] = APPROX_DIVIDE(c, n - 1);
	}
}

__global__
void kernel2a(unsigned short *img, int width, int height, int n, int *kernel, unsigned short *result) {
	int i, j, z, k, l, c;
	i = blockIdx.x * blockDim.x + threadIdx.x;
	j = blockIdx.y * blockDim.y + threadIdx.y;
	z = blockIdx.z;
	extern __shared__ unsigned short tile[];
	int tileW = blockDim.x;
	int tileH = blockDim.y + n - 1;
	int blockS = blockDim.x * blockDim.y;
	for(k = 0; k < (tileW * tileH) / blockS; k++) {
		int pos = k + (threadIdx.y * blockDim.x + threadIdx.x) * ((tileW * tileH) / blockS);
		int imgX = blockDim.x * blockIdx.x + pos % tileW;
		int imgY = blockDim.y * blockIdx.y - n / 2 + pos / tileW;
		tile[pos] = (0 <= imgX && width > imgX && 0 <= imgY && height > imgY) ? img[(z * height + imgY) * width + imgX] : 0;
	}
	int pos = blockDim.x * blockDim.y * k + threadIdx.y * blockDim.x + threadIdx.x;
	if(pos < tileW * tileH) {
		int imgX = blockDim.x * blockIdx.x + pos % tileW;
		int imgY = blockDim.y * blockIdx.y - n / 2 + pos / tileW;
		tile[pos] = (0 <= imgX && width > imgX && 0 <= imgY && height > imgY) ? img[(z * height + imgY) * width + imgX] : 0;
	}
	__syncthreads();
	if(i < width && j < height) {
		c = 0;
		for(k = 0; k < n; k++) {
			c += kernel[k] * tile[(threadIdx.y + k) * tileW + (threadIdx.x)];
		}
		result[(z * height + j) * width + i] = APPROX_DIVIDE(c, n - 1);
	}
}

__global__
void kernel2b(unsigned short *img, int width, int height, int n, int *kernel, stbi_uc *result) {
	int i, j, z, k, l, c;
	i = blockIdx.x * blockDim.x + threadIdx.x;
	j = blockIdx.y * blockDim.y + threadIdx.y;
	z = blockIdx.z;
	extern __shared__ unsigned short tile[];
	int tileW = blockDim.x;
	int tileH = blockDim.y + n - 1;
	int blockS = blockDim.x * blockDim.y;
	for(k = 0; k < (tileW * tileH) / blockS; k++) {
		int pos = k + (threadIdx.y * blockDim.x + threadIdx.x) * ((tileW * tileH) / blockS);
		int imgX = blockDim.x * blockIdx.x+ pos % tileW;
		int imgY = blockDim.y * blockIdx.y - n / 2 + pos / tileW;
		tile[pos] = (0 <= imgX && width > imgX && 0 <= imgY && height > imgY) ? img[(z * height + imgY) * width + imgX] : 0;
	}
	int pos = blockDim.x * blockDim.y * k + threadIdx.y * blockDim.x + threadIdx.x;
	if(pos < tileW * tileH) {
		int imgX = blockDim.x * blockIdx.x + pos % tileW;
		int imgY = blockDim.y * blockIdx.y - n / 2 + pos / tileW;
		tile[pos] = (0 <= imgX && width > imgX && 0 <= imgY && height > imgY) ? img[(z * height + imgY) * width + imgX] : 0;
	}
	__syncthreads();
	if(i < width && j < height) {
		c = 0;
		for(k = 0; k < n; k++) {
			c += kernel[k] * tile[(threadIdx.y + k) * tileW + (threadIdx.x)];
		}
		result[(j * width + i) * 3 + z] = APPROX_DIVIDE(c, n + 7);
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
	dim3 threadsPerBlock(32, 32, 1);
	cudaMalloc(&filter1_d, sizeof(int) * n_init);
	cudaMalloc(&filter2_d, sizeof(int) * 15);
	cudaMemcpy(filter1_d, filter1, sizeof(int) * n_init, cudaMemcpyHostToDevice);
	cudaMemcpy(filter2_d, filter2, sizeof(int) * 15, cudaMemcpyHostToDevice);
	//cudaError_t b = cudaGetLastError();
	kernel1a << <blocks, threadsPerBlock, sizeof(int) *(32 + n_init / 2) *(32 + n_init / 2) >> > (img_d, width, height, n_init, filter1_d, aux1_d);
	cudaDeviceSynchronize();
	//cudaError_t dd = cudaGetLastError();
	for(int i = n_init; i < (n - 1); i += 14) {
		kernel2a << <blocks, threadsPerBlock, sizeof(int) *(32 + 15 / 2) *(32 + 15 / 2) >> > (aux1_d, width, height, 15, filter2_d, aux2_d);
		cudaDeviceSynchronize();
		kernel1b << <blocks, threadsPerBlock, sizeof(int) *(32 + 15 / 2) *(32 + 15 / 2) >> > (aux2_d, width, height, 15, filter2_d, aux1_d);
		cudaDeviceSynchronize();
	}
	kernel2b<<<blocks, threadsPerBlock, sizeof(int) *(32 + n_init / 2) *(32 + n_init / 2) >>>(aux1_d, width, height, n_init, filter1_d, img_d);
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
	const char fname[] = "../../../img2.png";
	int width, height, chn;
	stbi_uc *img = stbi_load(fname, &width, &height, &chn, 3);
	stbi_uc *img_d;
	if(WIDTH != 0) {
		stbi_uc *img_r = (stbi_uc*)malloc(sizeof(stbi_uc) * WIDTH * HEIGHT * 3);
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
