#pragma OPENCL EXTENSION cl_khr_fp64 : enable

inline float ramped_sin_fp32(float t, float ramp_len, float frequency)
{
	return smoothstep(0, ramp_len, t) * sin(2.0 * M_PI * frequency * t);
}

inline double ramped_sin_fp64(double t, double ramp_len, double frequency)
{
	return smoothstep(0, ramp_len, t) * sin(2.0 * M_PI * frequency * t);
}


kernel void fdtd2d_noiter_fp32(global float* ez_old,
		   	       global float* hx_old,
		   	       global float* hy_old,
		   	       global float* ez_current,
		   	       global float* hx_current,
		   	       global float* hy_current,
		   	       global float* ez_integral,
		   	       global float* hx_integral,
		   	       global float* hy_integral,
		   	       constant float* ez_factor1,
		   	       constant float* ez_factor2,
		   	       constant float* ez_factor3,
		   	       constant float* hx_factor1,
		   	       constant float* hx_factor2,
		   	       constant float* hx_factor3,
		   	       constant float* hy_factor1,
		   	       constant float* hy_factor2,
		   	       constant float* hy_factor3,
			       constant char* geometry,
			       float grid_width_x,
			       float grid_width_y,
			       int uwidth,
			       float max_time,
			       float time_step,
			       int batch_size_x,
			       int batch_size_y,
			       constant char* source_factor,
			       float source_amplitude,
			       float source_ramp_len,
			       float source_frequency)
{
	// Get the actual id
	const int ID = get_global_id(0) * batch_size_x + get_global_id(1) * batch_size_y * uwidth + 1 + uwidth;

	int id = ID;


	for (float current_time = 0; current_time < max_time; current_time += time_step)
	{
		// Calculate the current magnetic field from the old electric field
		for (int y = 0; y < batch_size_y; y++)
		{
			for (int x = 0; x < batch_size_x; x++)
			{
				id = ID + x + y * uwidth;
				// Calculate the curl terms to simplify the actual calculation
				const float hx_curl_term = ((ez_old[id + uwidth] - ez_old[id]) / grid_width_y);
				const float hy_curl_term = ((ez_old[id + 1] - ez_old[id]) / grid_width_x);

				// Euler integration of the curl terms
				hx_integral[id] += hx_curl_term;
				hy_integral[id] += hy_curl_term;

				// Update the magnetic field
				hx_current[id] = hx_factor1[id] * hx_old[id]
						 - hx_factor2[id] * hx_curl_term
						 - hx_factor3[id] * hx_integral[id];
				hy_current[id] = hy_factor1[id] * hy_old[id]
						 + hy_factor2[id] * hy_curl_term
						 + hy_factor3[id] * hy_integral[id];
			}
		}

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
		// Calculate the current electric field from the current magnetic field
		for (int y = 0; y < batch_size_y; y++)
		{
			for (int x = 0; x < batch_size_x; x++)
			{
				id = ID + x + y * uwidth;
				// Calculate the curl term to simplify the actual calculation
				const float ez_curl_term = ((hy_current[id] - hy_current[id - 1]) / grid_width_x) - ((hx_current[id] - hx_current[id - uwidth]) / grid_width_y);

				// Euler integration of the curl term
				ez_integral[id] += ez_old[id];
				// Calculate the electric field
				ez_current[id] = ez_factor1[id] * ez_old[id]
						 - ez_factor2[id] * ez_integral[id]
						 + ez_factor3[id] * ez_curl_term;
				// Zero ez out if the soure_factor is 1
				ez_current[id] -= abs(source_factor[id]) * ez_current[id];
				// Than add the source. This way a hard source is created!
				ez_current[id] += source_factor[id] * source_amplitude * ramped_sin_fp32(current_time, source_ramp_len, source_frequency);
			}
		}
		// Synchronize the workitems after calculating the current fields
		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
		// Now memory consistency is assured and the fields can be copied
		for (int y = 0; y < batch_size_y; y++)
		{
			for (int x = 0; x < batch_size_x; x++)
			{
				id = ID + x + y * uwidth;
				hx_old[id] = hx_current[id];
				hy_old[id] = hy_current[id];
				ez_old[id] = ez_current[id];
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	}
}

kernel void fdtd2d_noiter_fp64(global double* ez_old,
		   	       global double* hx_old,
		   	       global double* hy_old,
		   	       global double* ez_current,
		   	       global double* hx_current,
		   	       global double* hy_current,
		   	       global double* ez_integral,
		   	       global double* hx_integral,
		   	       global double* hy_integral,
		   	       constant double* ez_factor1,
		   	       constant double* ez_factor2,
		   	       constant double* ez_factor3,
		   	       constant double* hx_factor1,
		   	       constant double* hx_factor2,
		   	       constant double* hx_factor3,
		   	       constant double* hy_factor1,
		   	       constant double* hy_factor2,
		   	       constant double* hy_factor3,
			       constant char* geometry,
			       double grid_width_x,
			       double grid_width_y,
			       int uwidth,
			       double max_time,
			       double time_step,
			       int batch_size_x,
			       int batch_size_y,
			       constant char* source_factor,
			       double source_amplitude,
			       double source_ramp_len,
			       double source_frequency)
{
	// Get the actual id
	const int ID = get_global_id(0) * batch_size_x + get_global_id(1) * batch_size_y * uwidth + 1 + uwidth;

	int id = ID;


	for (double current_time = 0; current_time < max_time; current_time += time_step)
	{
		// Calculate the current magnetic field from the old electric field
		for (int y = 0; y < batch_size_y; y++)
		{
			for (int x = 0; x < batch_size_x; x++)
			{
				id = ID + x + y * uwidth;
				// Calculate the curl terms to simplify the actual calculation
				const double hx_curl_term = ((ez_old[id + uwidth] - ez_old[id]) / grid_width_y);
				const double hy_curl_term = ((ez_old[id + 1] - ez_old[id]) / grid_width_x);

				// Euler integration of the curl terms
				hx_integral[id] += hx_curl_term;
				hy_integral[id] += hy_curl_term;

				// Update the magnetic field
				hx_current[id] = hx_factor1[id] * hx_old[id]
						 - hx_factor2[id] * hx_curl_term
						 - hx_factor3[id] * hx_integral[id];
				hy_current[id] = hy_factor1[id] * hy_old[id]
						 + hy_factor2[id] * hy_curl_term
						 + hy_factor3[id] * hy_integral[id];
			}
		}

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
		// Calculate the current electric field from the current magnetic field
		for (int y = 0; y < batch_size_y; y++)
		{
			for (int x = 0; x < batch_size_x; x++)
			{
				id = ID + x + y * uwidth;
				// Calculate the curl term to simplify the actual calculation
				const double ez_curl_term = ((hy_current[id] - hy_current[id - 1]) / grid_width_x) - ((hx_current[id] - hx_current[id - uwidth]) / grid_width_y);

				// Euler integration of the curl term
				ez_integral[id] += ez_old[id];
				// Calculate the electric field
				ez_current[id] = ez_factor1[id] * ez_old[id]
						 - ez_factor2[id] * ez_integral[id]
						 + ez_factor3[id] * ez_curl_term;
				// Zero ez out if the soure_factor is 1
				ez_current[id] -= abs(source_factor[id]) * ez_current[id];
				// Than add the source. This way a hard source is created!
				ez_current[id] += source_factor[id] * source_amplitude * ramped_sin_fp64(current_time, source_ramp_len, source_frequency);
			}
		}
		// Synchronize the workitems after calculating the current fields
		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
		// Now memory consistency is assured and the fields can be copied
		for (int y = 0; y < batch_size_y; y++)
		{
			for (int x = 0; x < batch_size_x; x++)
			{
				id = ID + x + y * uwidth;
				hx_old[id] = hx_current[id];
				hy_old[id] = hy_current[id];
				ez_old[id] = ez_current[id];
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	}
}
