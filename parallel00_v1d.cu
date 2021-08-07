#include <stdio.h>

void pascal(int *p, int n) {
	n--;
	p[0] = 1;
	for(int k = 0; k < (n >> 1); k++) {
		p[k + 1] = p[k] * (n - k) / (k + 1);
	}
}

__global__
void kernel1(unsigned short *__restrict__ result,const unsigned short *__restrict__ img, int width, int height, size_t result_pitch, size_t img_pitch, int n, const int *__restrict__ filter) {
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
			c += filter[k] * m;
		}
		l = i + k - n / 2;
		if(0 <= l && l < width) {
			c += filter[k] * img[(z * height + j) * img_pitch + l];
		}
		result[(z * height + j) * result_pitch + i] = c >> (n - 1);
	}
}

__global__
void kernel2(unsigned short *__restrict__ result, const unsigned short *__restrict__ img, int width, int height, size_t result_pitch, size_t img_pitch, int n, const int *__restrict__ filter) {
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
			c += filter[k] * m;
		}
		l = i + k - n / 2;
		if(0 <= l && l < width) {
			c += filter[k] * img[(j * img_pitch + l * 3) + z];
		}
		result[(j * result_pitch + i * 3) + z] = c >> (n - 1);
	}
}

void blur(int width, int height) {
	int i, *__restrict__ filter, *__restrict__ filter_d;
	size_t img1_pitch, img2_pitch;
	unsigned short *__restrict__ img1, *__restrict__ img2;
	dim3 blocks(3, (width + 31) / 32, (height + 31) / 32);
	dim3 threadsPerBlock(1, 32, 32);

	filter = (int *)malloc(sizeof(int) * 9);
	pascal(filter, 17);
	cudaMalloc((void **)&filter_d, sizeof(int) * 9);
	cudaMemcpy(filter_d, filter, sizeof(int) * 9, cudaMemcpyHostToDevice);

	cudaMallocPitch((void **)&img1, &img1_pitch, sizeof(unsigned short) * width, height * 3);
	img1_pitch /= sizeof(unsigned short);
	cudaMallocPitch((void **)&img2, &img2_pitch, sizeof(unsigned short) * width, height * 3);
	img2_pitch /= sizeof(unsigned short);
	for(i = 0; i < 100; i++) {
		// CALL kerne1 OR kernel2
		kernel2 << <blocks, threadsPerBlock >> > (img2, img1, width, height, img2_pitch, img1_pitch, 17, filter_d);
	}
	cudaFree(img1);
	cudaFree(img2);
	cudaFree(filter_d);
	free(filter);
	cudaDeviceSynchronize();
}

int main(void) {
	clock_t begin, end;
	begin = clock();
	blur(4096, 4096);
	end = clock();
	printf("Time: %f", (double)(end - begin) / CLOCKS_PER_SEC);
	return 0;
}
