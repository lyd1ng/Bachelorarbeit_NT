kernel void buffer_test_4(global float* dataA)
{
	const int id = get_global_id(0) + 1;
	dataA[id] = 4;
}

kernel void buffer_test_4_2d(global float* dataA, int width)
{
	const int id = (get_global_id(0) + 1) + (get_global_id(1) + 1) * width;
	dataA[id] = 4;
}
