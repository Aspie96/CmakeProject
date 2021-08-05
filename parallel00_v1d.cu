#include <stdio.h>

#ifdef __cplusplus
//#ifndef _MSC_VER
#define restrict __restrict__
//#endif
#endif

void pascal(int *p, int n) {
	n--;
	p[0] = 1;
	for(int k = 0; k < (n >> 1); k++) {
		p[k + 1] = p[k] * (n - k) / (k + 1);
	}
}

#define APPROX_DIVIDE2(A, B) (((A) >> (B)) + (((A) >> ((B) - 1)) & 1))

//#ifndef _MSC_VER
#define restrict __restrict__
//#endif

__global__
void kernel1b1(const unsigned short *restrict img, int width, int height, size_t result_pitch, size_t img_pitch, int n, unsigned short *restrict result) {
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
			c += m;
		}
		l = i + k - n / 2;
		if(0 <= l && l < width) {
			c += img[(z * height + j) * img_pitch + l];
		}
		result[(z * height + j) * result_pitch + i] = c / n;
	}
}

__global__
void kernel2a1(const unsigned short *restrict img, int width, int height, size_t result_pitch, size_t img_pitch, int n, const int *restrict filter, unsigned short *restrict result) {
	int i, j, z, k, l, m, c;
	i = blockIdx.x * blockDim.x + threadIdx.x;
	j = blockIdx.y * blockDim.y + threadIdx.y;
	z = blockIdx.z;
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
			c += filter[k] * m;
		}
		l = j + k - n / 2;
		if(0 <= l && l < height) {
			c += filter[k] * img[(z * height + l) * img_pitch + i];
		}
		result[(z * height + j) * result_pitch + i] = APPROX_DIVIDE2(c, n - 1);
	}
}

__global__
void kernel1b3(const unsigned short *restrict img, int width, int height, size_t result_pitch, size_t img_pitch, int n, unsigned short *restrict result) {
	int i, j, z, k, l, c, m;
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
			c += m;
		}
		l = i + k - n / 2;
		if(0 <= l && l < width) {
			c += img[(j * img_pitch + l * 3) + z];
		}
		result[(j * result_pitch + i * 3) + z] = c / n;
	}
}

__global__
void kernel2a3(const unsigned short *restrict img, int width, int height, size_t result_pitch, size_t img_pitch, int n, const int *restrict filter, unsigned short *restrict result) {
	int i, j, z, k, l, c, m;
	z = blockIdx.x;
	i = blockIdx.y * blockDim.y + threadIdx.y;
	j = blockIdx.z * blockDim.z + threadIdx.z;
	if(i < width && j < height) {
		c = 0;
		for(k = 0; k < n >> 1; k++) {
			l = j + k - n / 2;
			m = 0;
			if(0 <= l && l < width) {
				m = img[(l * img_pitch + i * 3) + z];
			}
			l = j + n - 1 - k - n / 2;
			if(0 <= l && l < width) {
				m += img[(l * img_pitch + i * 3) + z];
			}
			c += filter[k] * m;
		}
		l = j + k - n / 2;
		if(0 <= l && l < width) {
			c += filter[k] * img[(l * img_pitch + i * 3) + z];
		}
		result[(j * result_pitch + i * 3) + z] = APPROX_DIVIDE2(c, n - 1);
	}
}

void blur1(int width, int height) {
	int i;
	size_t img1_pitch, img2_pitch;
	unsigned short *restrict img1, *restrict img2;
	dim3 blocks(3, (width + 31) / 32, (height + 31) / 32);
	dim3 threadsPerBlock(1, 32, 32);

	cudaMallocPitch((void **)&img1, &img1_pitch, sizeof(unsigned short) * width, height * 3);
	img1_pitch /= sizeof(unsigned short);
	cudaMallocPitch((void **)&img2, &img2_pitch, sizeof(unsigned short) * width, height * 3);
	img2_pitch /= sizeof(unsigned short);
	for(i = 0; i < 1000; i++) {
		kernel1b1 << <blocks, threadsPerBlock >> > (img1, width, height, img2_pitch, img1_pitch, 17, img2);
	}
	cudaFree(img1);
	cudaFree(img2);
	cudaDeviceSynchronize();
}

void blur3(int width, int height) {
	int i;
	size_t img1_pitch, img2_pitch;
	unsigned short *restrict img1, *restrict img2;
	dim3 blocks(3, (width + 31) / 32, (height + 31) / 32);
	dim3 threadsPerBlock(1, 32, 32);

	cudaMallocPitch((void **)&img1, &img1_pitch, sizeof(unsigned short) * width * 3, height);
	img1_pitch /= sizeof(unsigned short);
	cudaMallocPitch((void **)&img2, &img2_pitch, sizeof(unsigned short) * width * 3, height);
	img2_pitch /= sizeof(unsigned short);
	for(i = 0; i < 1000; i++) {
		kernel1b3 << <blocks, threadsPerBlock >> > (img1, width, height, img2_pitch, img1_pitch, 17, img2);
	}
	cudaFree(img1);
	cudaFree(img2);
	cudaDeviceSynchronize();
}

int main(void) {
	clock_t begin, end;
	begin = clock();
	blur1(4096, 4096);
	end = clock();
	printf("Time: %f", (double)(end - begin) / CLOCKS_PER_SEC);
	return 0;
}
