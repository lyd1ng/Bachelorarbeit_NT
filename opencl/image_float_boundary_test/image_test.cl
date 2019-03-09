kernel void max_value_test(write_only image2d_t dst_image,
		       int width,
		       int height)
{
	const float4 color = (float4)(100.0, 200.0, 300.0, 400.0);
	int2 coord = (int2)(get_global_id(0) , get_global_id(1));
	write_imagef(dst_image, coord, color);
}

kernel void times2_test(read_only image2d_t src_image,
			write_only image2d_t dst_image,
			int width,
			int height)
{
	const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP
				 | CLK_FILTER_NEAREST;
	int2 coord = (int2)(get_global_id(0) , get_global_id(1));
	float4 color = read_imagef(src_image, sampler, coord);
	write_imagef(dst_image, coord, color);
}
