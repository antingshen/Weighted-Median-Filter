// Assumes Kernel is 5x5

__kernel void convolve(
	const __global float * pad, 
	__global float * kern, 
	__global float * out, 
	const int pad_num_col,
	__local float * buffer,
	const int median_index) 
{ 
	const int KER_SIZE = 5;
	const int NUM_ITERATIONS = 8;

	const int out_num_col = get_global_size(0);
	const int out_col = get_global_id(0); 
	const int out_row = get_global_id(1);

	int buffer_row_head;
	int pad_row_head;

	// copy into buffer
	for (int row = 0; row < KER_SIZE; row++) {
		buffer_row_head = row * KER_SIZE;
		pad_row_head = row * pad_num_col;

		for (int col = 0; col < KER_SIZE; col++) { 
			buffer[buffer_row_head+col] = pad[pad_row_head+col];
		}
	}

	// find median with binary search
	float estimate = 128.0f;
	float lower = 0.0f;
	float upper = 255.0f;
	uint higher;

	for (int _ = 0; _ < NUM_ITERATIONS; _++){
		higher = 0;
		for (int i = 0; i < KER_SIZE*KER_SIZE; i++){
			higher += (estimate < buffer[i]) * kern[i];
		}
		if (higher > median_index){
			lower = estimate;
		} else {
			upper = estimate;
		}
		estimate = 0.5 * (upper + lower);
	}

	out[out_row*out_num_col+out_col];
} 













