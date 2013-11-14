#define NUM 7
//Size of MASK: NUMxNUM. Should be always odd

WeightedMedian(struct Image *input, struct Image *output, int MASK[][N] {
	int x, y;
	int i, j;
	int m, z;
	int a;
	int AR[150]
	n = num;
	
	for(y = n/2; y < (input->Rows - n/2); y++)
		for(x = n/2; x < (input->Cols - n/2); x++)
		{
			z = 0;
			for(j = -n/2; j <= n/2; j++)
				for(i = -n/2; i <= n/2, i++)
				{
					for(m = 1; m <= MASK[(int) (n/2 + i)][(int) (n/2 + j)]; m++)
					{
						AR[z] = *(input->Data + x + i + ((long) (y+j) * input->Cols));
						z++;
					}
				}
			
			for(j = 1; j < (z-1); j++)
			{
				a = AR[j];
				i = j-1;
				while(i >= 0 && AR[i] > a)
				{
					AR[i+1] = AR[i];
					i = i-1;
				}
				AR[i+1] = a;
			}
			
			*(output->Data + x + ((long) y * input->Cols)) = AR[z/2];
		}
}

//# in mask = # of times each pixel is repeated in the median calc
//sum of all numbers in mask == odd #
//n == size of filtering operation (must be odd)
//total # vals in median calc < 150.