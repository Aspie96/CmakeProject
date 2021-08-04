#include <stdio.h>

//#ifndef _MSC_VER
#define restrict __restrict__
//#endif

__global__
void kernel1b(const unsigned short *restrict img, int width, int height, size_t result_pitch, size_t img_pitch, int n, unsigned short *restrict result) {
	int i, j, z, k, l, c;
	i = blockIdx.x * blockDim.x + threadIdx.x;
	j = blockIdx.y * blockDim.y + threadIdx.y;
	z = blockIdx.z;
	if(i < width && j < height) {
		c = 0;
		for(k = 0; k < n; k++) {
			l = i + k - n / 2;
			if(0 <= l && l < width) {
				c += img[(z * height + j) * img_pitch + i];
			}
		}
		result[(z * height + j) * result_pitch + i] = c / 17;
	}
}

void blur(int width, int height) {
	int i;
	size_t img1_pitch, img2_pitch;
	unsigned short *restrict img1, *restrict img2;
	dim3 blocks((width + 31) / 32, (height + 31) / 32, 3);
	dim3 threadsPerBlock(32, 32, 1);
	cudaError_t a;

	cudaMallocPitch((void **)&img1, &img1_pitch, sizeof(unsigned short) * width, height * 3);
	img1_pitch /= sizeof(unsigned short);
	cudaMallocPitch((void **)&img2, &img2_pitch, sizeof(unsigned short) * width, height * 3);
	img2_pitch /= sizeof(unsigned short);
	for(i = 0; i < 1000; i++) {
		kernel1b << <blocks, threadsPerBlock >> > (img1, width, height, img2_pitch, img1_pitch, 17, img2);
	}
	cudaFree(img1);
	cudaFree(img2);
	cudaDeviceSynchronize();
	a = cudaGetLastError();
}

int main(void) {
	clock_t begin, end;
	begin = clock();
	blur(4096, 4096);
	end = clock();
	printf("Time: %f", (double)(end - begin) / CLOCKS_PER_SEC);
	return 0;
}
