kernel void array1d_add(
		   global float* data1,
		   global float* data2,
		   global float* result)
{
	int index = get_global_id(0);
	result[index] = data1[index] + data2[index];
}

kernel void array2d_add(
		   global float* data1,
		   global float* data2,
		   global float* result)
{
	int index = get_global_id(0) + get_global_id(1) * get_global_size(0);
	result[index] = data1[index] + data2[index];
}

kernel void array3d_add(
		   global float* data1,
		   global float* data2,
		   global float* result)
{
	int index = get_global_id(0) + get_global_id(1) * get_global_size(0)
		    + get_global_id(2) * get_global_size(0) * get_global_size(1);
	result[index] = data1[index] + data2[index];
}
