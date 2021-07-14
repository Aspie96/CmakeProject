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
#define WIDTH 32
#endif
#ifndef HEIGHT
#define HEIGHT WIDTH
#endif
#ifndef SAVE_OUTPUT
#define SAVE_OUTPUT 1
#endif
#ifdef __cplusplus
#ifndef _MSC_VER
#define restrict __restrict__
#endif
#endif
#ifndef IMGPATH
#define IMGPATH "../../../img2.png"
#endif

void pascal(int *restrict p, int n) {
	int k;
	n--;
	p[0] = 1;
	for(k = 0; k < (n >> 1); k++) {
		p[k + 1] = p[k] * (n - k) / (k + 1);
	}
}

void kernel1a(const stbi_uc *restrict img, int width, int height, int n, const int *restrict filter, unsigned short *restrict result) {
	int i, j, z, k, l, c, m;
	for(j = 0; j < height; j++)
	for(i = 0; i < width; i++)
	for(z = 0; z < 3; z++) {
		c = 0;
		for(k = 0; k < n >> 1; k++) {
			l = i + k - (n >> 1);
			m = 0;
			if(0 <= l && l < width) {
				m += img[(j * width + l) * 3 + z];
			}
			l = i + (n - k - 1) - (n >> 1);
			if(0 <= l && l < width) {
				m += img[(j * width + l) * 3 + z];
			}
			c += filter[k] * m;
		}
		l = i + k - (n >> 1);
		if(0 <= l && l < width) {
			c += filter[k] * img[(j * width + l) * 3 + z];
		}
		result[(z * height + j) * width + i] = APPROX_DIVIDE1(c, n - 9);
	}
}

void kernel1b(const unsigned short *restrict img, int width, int height, int n, const int *restrict filter, unsigned short *restrict result) {
	int i, j, k, z, c, l, m;
	for(z = 0; z < 3; z++)
	for(j = 0; j < height; j++)
	for(i = 0; i < width; i++) {
		c = 0;
		for(k = 0; k < (n >> 1); k++) {
			l = i + k - (n >> 1);
			m = 0;
			if(0 <= l && l < width) {
				m += img[(z * height + j) * width + l];
			}
			l = i + (n - k - 1) - (n >> 1);
			if(0 <= l && l < width) {
				m += img[(z * height + j) * width + l];
			}
			c += filter[k] * m;
		}
		if(0 <= l && l < width) {
			c += filter[k] * img[(z * height + j) * width + l];
		}
		result[(z * height + j) * width + i] = APPROX_DIVIDE2(c, n - 1);
	}
}
void kernel2a(const unsigned short *restrict img, int width, int height, int n, const int *restrict filter, unsigned short *restrict result) {
	int i, j, z, k, l, c, m;
	for(z = 0; z < 3; z++)
	for(j = 0; j < height; j++)
	for(i = 0; i < width; i++) {
		c = 0;
		for(k = 0; k < (n >> 1); k++) {
			l = j + k - (n >> 1);
			m = 0;
			if(0 <= l && l < height) {
				m += img[(z * height + l) * width + i];
			}
			l = j + (n - k - 1) - (n >> 1);
			if(0 <= l && l < height) {
				m += img[(z * height + l) * width + i];
			}
			c += filter[k] * m;
		}
		l = j + k - (n >> 1);
		if(0 <= l && l < height) {
			c += filter[k] * img[(z * height + l) * width + i];
		}
		result[(z * height + j) * width + i] = APPROX_DIVIDE2(c, n - 1);
	}
}

void kernel2b(const unsigned short *restrict img, int width, int height, int n, const int *restrict filter, stbi_uc *restrict result) {
	int i, j, z, k, l, c, m;
	for(i = 0; i < width; i++)
	for(j = 0; j < height; j++)
	for(z = 0; z < 3; z++) {
		c = 0;
		for(k = 0; k < (n >> 1); k++) {
			l = j + k - (n >> 1);
			m = 0;
			if(0 <= l && l < height) {
				m += img[(z * height + l) * width + i];
			}
			l = j + k - (n >> 1);
			if(0 <= l && l < height) {
				m += img[(z * height + l) * width + i];
			}
			c += filter[k] * m;
		}
		l = j + k - (n >> 1);
		if(0 <= l && l < height) {
			c += filter[k] * img[(z * height + l) * width + i];
		}
		result[(j * width + i) * 3 + z] = APPROX_DIVIDE1(c, n + 7);
	}
}

int blur(clock_t begin, int n, int width, int height, stbi_uc *restrict img) {
	int *restrict kernel1, *restrict kernel2, n_init, i;
	unsigned short *restrict aux1, *restrict aux2;

	n_init = ((n - 1) % 16) + 1;
	if(n_init == 1) {
		n_init = 17;
	}
	kernel1 = (int*)malloc(sizeof(int) * ((n_init >> 1) + 1));
	kernel2 = (int*)malloc(sizeof(int) * 9);
	aux1 = (unsigned short *)malloc(sizeof(unsigned short) * width * height * 3);
	aux2 = (unsigned short *)malloc(sizeof(unsigned short) * width * height * 3);
	pascal(kernel1, n_init);
	pascal(kernel2, 17);
	kernel1a(img, width, height, n_init, kernel1, aux1);
	if((clock() - begin) / CLOCKS_PER_SEC > 100) {
		free(kernel1);
		free(kernel2);
		free(aux1);
		free(aux2);
		return 1;
	}
	for(i = n_init; i < (n - 1); i += 16) {
		kernel2a(aux1, width, height, 17, kernel2, aux2);
		if((clock() - begin) / CLOCKS_PER_SEC > 100) {
			free(kernel1);
			free(kernel2);
			free(aux1);
			free(aux2);
			return 1;
		}
		kernel1b(aux2, width, height, 17, kernel2, aux1);
		if((clock() - begin) / CLOCKS_PER_SEC > 100) {
			free(kernel1);
			free(kernel2);
			free(aux1);
			free(aux2);
			return 1;
		}
	}
	kernel2b(aux1, width, height, n_init, kernel1, img);
	free(kernel1);
	free(kernel2);
	free(aux1);
	free(aux2);
	if((clock() - begin) / CLOCKS_PER_SEC > 100) {
		return 1;
	}
	return 0;
}

double test_blur_time(int n, int width, int height, stbi_uc *restrict img_d) {
	clock_t begin, end;
	
	begin = clock();
	if(blur(begin, n, width, height, img_d)) {
		end = clock();
		printf("Program failed");
		return -(double)(end - begin) / CLOCKS_PER_SEC;
	} else {
		end = clock();
		printf("Success");
		return (double)(end - begin) / CLOCKS_PER_SEC;
	}
}

int main(void) {
	int nk, *restrict ns, i, width, height, chn, f;
	stbi_uc *restrict img, *restrict img_c, *restrict img_r;
	double time;
	const char fname[] = IMGPATH;
	const char fname2[] = "image2.bmp";

	printf("%d\n", N);
	print("%d\n", WIDTH);
	print("%s\n", fname);

	return 0;
}
