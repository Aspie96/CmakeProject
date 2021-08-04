#include <stdio.h>

//#ifndef _MSC_VER
#define restrict __restrict__
//#endif

__global__
void kernel1b(const unsigned short *restrict img, int width, int height, size_t result_pitch, size_t img_pitch, int n, unsigned short *restrict result) {
	int i, j, z, k, l, c;
	z = blockIdx.x;
	i = blockIdx.y * blockDim.y + threadIdx.y;
	j = blockIdx.z * blockDim.z + threadIdx.z;
	if(i < width && j < height) {
		c = 0;
		for(k = 0; k < n; k++) {
			l = i + k - n / 2;
			if(0 <= l && l < width) {
				c += img[(j * img_pitch + l * 3) + z];
			}
		}
		result[(j * result_pitch + i * 3) + z] = c / 17;
	}
}

void blur(int width, int height) {
	int i;
	size_t img1_pitch, img2_pitch;
	unsigned short *restrict img1, *restrict img2;
	dim3 blocks(3, (width + 31) / 32, (height + 31) / 32);
	dim3 threadsPerBlock(1, 32, 32);

	cudaMallocPitch((void **)&img1, &img1_pitch, sizeof(unsigned short) * width, height * 3);
	img1_pitch /= sizeof(unsigned short);
	cudaMallocPitch((void **)&img2, &img2_pitch, sizeof(unsigned short) * width, height * 3);
	img2_pitch /= sizeof(unsigned short);
	for(i = 0; i < 1; i++) {
		kernel1b << <blocks, threadsPerBlock >> > (img1, width, height, img2_pitch, img1_pitch, 17, img2);
	}
	cudaFree(img1);
	cudaFree(img2);
	cudaDeviceSynchronize();
}

int main(void) {
	clock_t begin, end;
	begin = clock();
	blur(32, 32);
	end = clock();
	printf("Time: %f", (double)(end - begin) / CLOCKS_PER_SEC);
	return 0;
}
