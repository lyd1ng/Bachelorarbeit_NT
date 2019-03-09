#pragma OPENCL EXTENSION cl_khr_fp64 : enable

inline float ramped_sin_fp32(float t, float ramp_len, float frequency)
{
	//return smoothstep(0, ramp_len, t) * sin(2.0 * M_PI * frequency * t);
	return sin(2.0 * M_PI * frequency * t);
}

inline double ramped_sin_fp64(double t, double ramp_len, double frequency)
{
	//return smoothstep(0, ramp_len, t) * sin(2.0 * M_PI * frequency * t);
	return sin(2.0 * M_PI * frequency * t);
}


kernel void fdtd3d_noiter_fp32(
			       global float3* old_e,
		   	       global float3* old_h,
		   	       global float3* current_e,
		   	       global float3* current_h,
			       global float3* e_curl_integral,
		   	       global float3* e_field_integral,
		   	       global float3* h_curl_integral,
			       global float3* h_field_integral,
		   	       constant float4* ex_factor,
		   	       constant float4* ey_factor,
		   	       constant float4* ez_factor,
		   	       constant float4* hx_factor,
		   	       constant float4* hy_factor,
		   	       constant float4* hz_factor,
			       constant char* geometry,
			       float grid_width_x,
			       float grid_width_y,
			       float grid_width_z,
			       int uwidth,
			       int udepth,
			       float max_time,
			       float time_step,
			       int batch_size_x,
			       int batch_size_y,
			       int batch_size_z,
			       constant char* source_factor,
			       float source_amplitude,
			       float source_ramp_len,
			       float source_frequency)

{
	// get the actual id
	const int  ID = get_global_id(0) * batch_size_x + get_global_id(1) * batch_size_y * uwidth + get_global_id(2) * batch_size_z * uwidth * udepth + 1 + uwidth + uwidth * udepth;

	int id = ID;
	
	for (float current_time = 0; current_time < max_time; current_time += time_step)
	{
		for (int z = 0; z < batch_size_z; z++)
		{
			for (int y = 0; y < batch_size_y; y++)
			{
				for (int x = 0; x < batch_size_x; x++)
				{
					id = ID + x + y * uwidth + z * uwidth * udepth;
					// calculate the current magnetic field from the old electric field
					float hx_curl_term = ((old_e[id + uwidth].z - old_e[id].z) / grid_width_y) - ((old_e[id + uwidth * udepth].y - old_e[id].y) / grid_width_z);
					float hy_curl_term = ((old_e[id + uwidth * udepth].x - old_e[id].x) / grid_width_z) - ((old_e[id + 1].z - old_e[id].z) / grid_width_x);
					float hz_curl_term = ((old_e[id + 1].y - old_e[id].y) / grid_width_x) - ((old_e[id + uwidth].x - old_e[id].x) / grid_width_y);
					h_curl_integral[id].x += hx_curl_term;
					h_curl_integral[id].y += hy_curl_term;
					h_curl_integral[id].z += hz_curl_term;
					h_field_integral[id] += old_h[id];
					current_h[id].x = + hx_factor[id].x * old_h[id].x - hx_factor[id].y * hx_curl_term
							 - hx_factor[id].z * h_curl_integral[id].x - hx_factor[id].w * h_field_integral[id].x;
					current_h[id].y = + hy_factor[id].x * old_h[id].y - hy_factor[id].y * hy_curl_term
							 - hy_factor[id].z * h_curl_integral[id].y - hy_factor[id].w * h_field_integral[id].y;
					current_h[id].z = + hz_factor[id].x * old_h[id].z - hz_factor[id].y * hz_curl_term
							 - hz_factor[id].z * h_curl_integral[id].z - hz_factor[id].w * h_field_integral[id].z;
				}
			}
		}

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		// calculate the current electric field from the current magnetic field
		for (int z = 0; z < batch_size_z; z++)
		{
			for (int y = 0; y < batch_size_y; y++)
			{
				for (int x = 0; x < batch_size_x; x++)
				{
					id = ID + x + y * uwidth + z * uwidth * udepth;
					float ex_curl_term = ((current_h[id].z - current_h[id - uwidth].z) / grid_width_y) - ((current_h[id].y - current_h[id - uwidth * udepth].y) / grid_width_z);
					float ey_curl_term = ((current_h[id].x - current_h[id - uwidth * udepth].x) / grid_width_z) - ((current_h[id].z - current_h[id - 1].z) / grid_width_x);
					float ez_curl_term = ((current_h[id].y - current_h[id - 1].y) / grid_width_x) - ((current_h[id].x - current_h[id - uwidth].x) / grid_width_y);
					e_curl_integral[id].x += ex_curl_term;
					e_curl_integral[id].y += ey_curl_term;
					e_curl_integral[id].z += ez_curl_term;
					e_field_integral[id] += old_e[id];
					current_e[id].x = ex_factor[id].x * old_e[id].x - ex_factor[id].y * e_field_integral[id].x
							 + ex_factor[id].z * ex_curl_term + ex_factor[id].w * e_curl_integral[id].x;
					current_e[id].y = ey_factor[id].x * old_e[id].y - ey_factor[id].y * e_field_integral[id].y
							 + ey_factor[id].z * ey_curl_term + ey_factor[id].w * e_curl_integral[id].y;
					current_e[id].z = ez_factor[id].x * old_e[id].z - ez_factor[id].y * e_field_integral[id].z
							 + ez_factor[id].z * ez_curl_term + ez_factor[id].w * e_curl_integral[id].z;

					// zero ez out if the soure_factor is 1
					current_e[id].z -= abs(source_factor[id]) * current_e[id].z;
					// than add the source. this way a hard source is created!
					current_e[id].z += source_factor[id] * source_amplitude * ramped_sin_fp32(current_time, source_ramp_len, source_frequency);
					current_e[id] *= geometry[id];
				}
			}
		}

		// synchronize the workitems after calculating the current fields
		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		// now memory consistency is assured and the fields can be copied
		for (int z = 0; z < batch_size_z; z++)
		{
			for (int y = 0; y < batch_size_y; y++)
			{
				for (int x = 0; x < batch_size_x; x++)
				{
					id = ID + x + y * uwidth + z * uwidth * udepth;
					old_h[id] = current_h[id];
					old_e[id] = current_e[id];
				}
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	}
}


kernel void fdtd3d_noiter_fp64(
			       global double3* old_e,
		   	       global double3* old_h,
		   	       global double3* current_e,
		   	       global double3* current_h,
			       global double3* e_curl_integral,
		   	       global double3* e_field_integral,
		   	       global double3* h_curl_integral,
			       global double3* h_field_integral,
		   	       constant double4* ex_factor,
		   	       constant double4* ey_factor,
		   	       constant double4* ez_factor,
		   	       constant double4* hx_factor,
		   	       constant double4* hy_factor,
		   	       constant double4* hz_factor,
			       constant char* geometry,
			       double grid_width_x,
			       double grid_width_y,
			       double grid_width_z,
			       int uwidth,
			       int udepth,
			       double max_time,
			       double time_step,
			       int batch_size_x,
			       int batch_size_y,
			       int batch_size_z,
			       constant char* source_factor,
			       double source_amplitude,
			       double source_ramp_len,
			       double source_frequency)

{
	// get the actual id
	const int ID = get_global_id(0) * batch_size_x + get_global_id(1) * batch_size_y * uwidth + get_global_id(2) * batch_size_z * uwidth * udepth + 1 + uwidth + uwidth * udepth;

	int id = ID;
	
	for (double current_time = 0; current_time < max_time; current_time += time_step)
	{
		for (int z = 0; z < batch_size_z; z++)
		{
			for (int y = 0; y < batch_size_y; y++)
			{
				for (int x = 0; x < batch_size_x; x++)
				{
					id = ID + x + y * uwidth + z * uwidth * udepth;
					// calculate the current magnetic field from the old electric field
					double hx_curl_term = ((old_e[id + uwidth].z - old_e[id].z) / grid_width_y) - ((old_e[id + uwidth * udepth].y - old_e[id].y) / grid_width_z);
					double hy_curl_term = ((old_e[id + uwidth * udepth].x - old_e[id].x) / grid_width_z) - ((old_e[id + 1].z - old_e[id].z) / grid_width_x);
					double hz_curl_term = ((old_e[id + 1].y - old_e[id].y) / grid_width_x) - ((old_e[id + uwidth].x - old_e[id].x) / grid_width_y);
					h_curl_integral[id].x += hx_curl_term;
					h_curl_integral[id].y += hy_curl_term;
					h_curl_integral[id].z += hz_curl_term;
					h_field_integral[id] += old_h[id];
					current_h[id].x = + hx_factor[id].x * old_h[id].x - hx_factor[id].y * hx_curl_term
							 - hx_factor[id].z * h_curl_integral[id].x - hx_factor[id].w * h_field_integral[id].x;
					current_h[id].y = + hy_factor[id].x * old_h[id].y - hy_factor[id].y * hy_curl_term
							 - hy_factor[id].z * h_curl_integral[id].y - hy_factor[id].w * h_field_integral[id].y;
					current_h[id].z = + hz_factor[id].x * old_h[id].z - hz_factor[id].y * hz_curl_term
							 - hz_factor[id].z * h_curl_integral[id].z - hz_factor[id].w * h_field_integral[id].z;
				}
			}
		}

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
		// calculate the current electric field from the current magnetic field
		for (int z = 0; z < batch_size_z; z++)
		{
			for (int y = 0; y < batch_size_y; y++)
			{
				for (int x = 0; x < batch_size_x; x++)
				{
					id = ID + x + y * uwidth + z * uwidth * udepth;
					double ex_curl_term = ((current_h[id].z - current_h[id - uwidth].z) / grid_width_y) - ((current_h[id].y - current_h[id - uwidth * udepth].y) / grid_width_z);
					double ey_curl_term = ((current_h[id].x - current_h[id - uwidth * udepth].x) / grid_width_z) - ((current_h[id].z - current_h[id - 1].z) / grid_width_x);
					double ez_curl_term = ((current_h[id].y - current_h[id - 1].y) / grid_width_x) - ((current_h[id].x - current_h[id - uwidth].x) / grid_width_y);
					e_curl_integral[id].x += ex_curl_term;
					e_curl_integral[id].y += ey_curl_term;
					e_curl_integral[id].z += ez_curl_term;
					e_field_integral[id] += old_e[id];
					current_e[id].x = ex_factor[id].x * old_e[id].x - ex_factor[id].y * e_field_integral[id].x
							 + ex_factor[id].z * ex_curl_term + ex_factor[id].w * e_curl_integral[id].x;
					current_e[id].y = ey_factor[id].x * old_e[id].y - ey_factor[id].y * e_field_integral[id].y
							 + ey_factor[id].z * ey_curl_term + ey_factor[id].w * e_curl_integral[id].y;
					current_e[id].z = ez_factor[id].x * old_e[id].z - ez_factor[id].y * e_field_integral[id].z
							 + ez_factor[id].z * ez_curl_term + ez_factor[id].w * e_curl_integral[id].z;

					// zero ez out if the soure_factor is 1
					current_e[id].z -= abs(source_factor[id]) * current_e[id].z;
					// than add the source. this way a hard source is created!
					current_e[id].z += source_factor[id] * source_amplitude * ramped_sin_fp64(current_time, source_ramp_len, source_frequency);
					current_e[id] *= geometry[id];
				}
			}
		}

		// synchronize the workitems after calculating the current fields
		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		// now memory consistency is assured and the fields can be copied
		for (int z = 0; z < batch_size_z; z++)
		{
			for (int y = 0; y < batch_size_y; y++)
			{
				for (int x = 0; x < batch_size_x; x++)
				{
					id = ID + x + y * uwidth + z * uwidth * udepth;
					old_h[id] = current_h[id];
					old_e[id] = current_e[id];
				}
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	}
}

