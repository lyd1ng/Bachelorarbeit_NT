kernel void vector_add(global float* dataC)
{
	const int id = get_global_id(0);
	dataC[id] = id;
}

