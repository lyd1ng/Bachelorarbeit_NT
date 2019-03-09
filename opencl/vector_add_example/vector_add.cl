kernel void vector_add(global float* dataA,
		       global float* dataB,
		       global float* dataC)
{
	const int id = get_global_id(0);
	dataC[id] = dataA[id] + dataB[id];
}

kernel void vector_add_batched(global float* dataA,
		       global float* dataB,
		       global float* dataC)
{
	const int batch_size = 64;
	const int id = get_global_id(0) * batch_size;
	for (int i=id; i < id + batch_size; i++)
	{
		dataC[i] = dataA[i] + dataB[i];
	}
}

kernel void group_id(global float* dataA,
		       global float* dataB,
		       global float* dataC)
{
	const int batch_size = 4;
	const int id = get_global_id(0) * batch_size;
	for (int i=id; i < id + batch_size; i++)
	{
		dataC[i] = get_group_id(0);
	}

}


