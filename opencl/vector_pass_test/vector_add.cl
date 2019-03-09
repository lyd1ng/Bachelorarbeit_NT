kernel void vector_add(global float3* vector_in,
		       global float* scalar_out)
{
	const int id = get_global_id(0);
	scalar_out[id] = vector_in[id].z;
}
