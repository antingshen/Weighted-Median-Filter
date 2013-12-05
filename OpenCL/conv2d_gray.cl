// Assumes Kernel is 5x5

__kernel void convolve(
	const __global float * pad, 
	__global float * kern, 
	__global float * out, 
	const int pad_num_col,
	const float median_index) 
{ 
	const int NUM_ITERATIONS = 8;

	const int out_num_col = get_global_size(0);
	const int out_col = get_global_id(0); 
	const int out_row = get_global_id(1);

	float buffer[25];

	int pad_row_head;
	int index = 0;
	int i = 0;

	// copy into buffer
	pad_row_head = out_row * pad_num_col + out_col;
	buffer[0] = pad[pad_row_head];
	buffer[1] = pad[pad_row_head+1];
	buffer[2] = pad[pad_row_head+2];
	buffer[3] = pad[pad_row_head+3];
	buffer[4] = pad[pad_row_head+4];

	pad_row_head += pad_num_col;
	buffer[5] = pad[pad_row_head];
	buffer[6] = pad[pad_row_head+1];
	buffer[7] = pad[pad_row_head+2];
	buffer[8] = pad[pad_row_head+3];
	buffer[9] = pad[pad_row_head+4];

	pad_row_head += pad_num_col;
	buffer[10] = pad[pad_row_head];
	buffer[11] = pad[pad_row_head+1];
	buffer[12] = pad[pad_row_head+2];
	buffer[13] = pad[pad_row_head+3];
	buffer[14] = pad[pad_row_head+4];

	pad_row_head += pad_num_col;
	buffer[15] = pad[pad_row_head];
	buffer[16] = pad[pad_row_head+1];
	buffer[17] = pad[pad_row_head+2];
	buffer[18] = pad[pad_row_head+3];
	buffer[19] = pad[pad_row_head+4];

	pad_row_head += pad_num_col;
	buffer[20] = pad[pad_row_head];
	buffer[21] = pad[pad_row_head+1];
	buffer[22] = pad[pad_row_head+2];
	buffer[23] = pad[pad_row_head+3];
	buffer[24] = pad[pad_row_head+4];

	// find median with binary search
	float estimate = 128.0f;
	float lower = 0.0f;
	float upper = 255.0f;
	float higher;

	for (int _ = 0; _ < NUM_ITERATIONS; _++){
		higher = 0;
		higher += ((float)(estimate < buffer[0])) * kern[0];
		higher += ((float)(estimate < buffer[1])) * kern[1];
		higher += ((float)(estimate < buffer[2])) * kern[2];
		higher += ((float)(estimate < buffer[3])) * kern[3];
		higher += ((float)(estimate < buffer[4])) * kern[4];
		higher += ((float)(estimate < buffer[5])) * kern[5];
		higher += ((float)(estimate < buffer[6])) * kern[6];
		higher += ((float)(estimate < buffer[7])) * kern[7];
		higher += ((float)(estimate < buffer[8])) * kern[8];
		higher += ((float)(estimate < buffer[9])) * kern[9];
		higher += ((float)(estimate < buffer[10])) * kern[10];
		higher += ((float)(estimate < buffer[11])) * kern[11];
		higher += ((float)(estimate < buffer[12])) * kern[12];
		higher += ((float)(estimate < buffer[13])) * kern[13];
		higher += ((float)(estimate < buffer[14])) * kern[14];
		higher += ((float)(estimate < buffer[15])) * kern[15];
		higher += ((float)(estimate < buffer[16])) * kern[16];
		higher += ((float)(estimate < buffer[17])) * kern[17];
		higher += ((float)(estimate < buffer[18])) * kern[18];
		higher += ((float)(estimate < buffer[19])) * kern[19];
		higher += ((float)(estimate < buffer[20])) * kern[20];
		higher += ((float)(estimate < buffer[21])) * kern[21];
		higher += ((float)(estimate < buffer[22])) * kern[22];
		higher += ((float)(estimate < buffer[23])) * kern[23];
		higher += ((float)(estimate < buffer[24])) * kern[24];
		if (higher > median_index){
			lower = estimate;
		} else {
			upper = estimate;
		}
		estimate = 0.5 * (upper + lower);
	}

	out[out_row*out_num_col+out_col] = estimate;
} 













