#pragma OPENCL EXTENSION cl_khr_fp64 : enable

inline float ramped_sin_fp32(float t, float ramp_len, float frequency)
{
	return smoothstep(0, ramp_len, t) * sin(2.0 * M_PI * frequency * t);
}

inline double ramped_sin_fp64(double t, double ramp_len, double frequency)
{
	return smoothstep(0, ramp_len, t) * sin(2.0 * M_PI * frequency * t);
}


kernel void fdtd1d_noiter_fp32(global float* ey_old,
		   	       global float* hx_old,
		   	       global float* ey_current,
		   	       global float* hx_current,
		   	       constant float* ey_factor1,
			       constant float* ey_factor2,
		   	       constant float* hx_factor1,
			       constant float* hx_factor2,
			       constant char* geometry_factor,
			       float grid_width_z,
			       float max_time,
			       float time_step,
			       int batch_size,
			       constant char* source_factor,
			       float source_amplitude,
			       float source_ramp_len,
			       float source_frequency)
{
	// Get the actual id depending on batch_size
	const int ID = get_global_id(0) * batch_size + 1;

	for (float current_time = 0; current_time < max_time; current_time += time_step)
	{
		for (int id = ID; id < ID + batch_size; id++)
		{
			// Calculate the current magnetic field from the OLD electric field
			hx_current[id] = hx_factor1[id] * hx_old[id] + hx_factor2[id]
				* ((ey_old[id + 1] - ey_old[id]) / grid_width_z);
		}
		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
		for (int id = ID; id < ID + batch_size; id++)
		{
			// Calculate the current electric field from the CURRENT magnetic field
			ey_current[id] = geometry_factor[id] * (ey_factor1[id] * ey_old[id] + ey_factor2[id]
				* ((hx_current[id] - hx_current[id - 1]) / grid_width_z));

			// Zero ey out if the soure_factor is 1
			ey_current[id] -= abs(source_factor[id]) * ey_current[id];
			// Than add the source. This way a hard source is created!
			ey_current[id] += source_factor[id] * source_amplitude * ramped_sin_fp32(current_time, source_ramp_len, source_frequency);
		}
		// Synchronize the workitems after calculating the current fields
		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
		// Now memory consistency is assured and the fields can be copied
		for (int id = ID; id < ID + batch_size; id++)
		{
			hx_old[id] = hx_current[id];
			ey_old[id] = ey_current[id];
		}
		// Again the workitems have to be synchronized
		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	}
}

kernel void fdtd1d_noiter_fp64(global double* ey_old,
		   	       global double* hx_old,
		   	       global double* ey_current,
		   	       global double* hx_current,
		   	       constant double* ey_factor1,
			       constant double* ey_factor2,
		   	       constant double* hx_factor1,
			       constant double* hx_factor2,
			       constant char* geometry_factor,
			       double grid_width_z,
			       double max_time,
			       double time_step,
			       int batch_size,
			       constant char* source_factor,
			       double source_amplitude,
			       double source_ramp_len,
			       double source_frequency)
{
	// Get the actual id depending on batch_size
	const int ID = get_global_id(0) * batch_size + 1;

	for (double current_time = 0; current_time < max_time; current_time += time_step)
	{
		for (int id = ID; id < ID + batch_size; id++)
		{
			// Calculate the current magnetic field from the OLD electric field
			hx_current[id] = hx_factor1[id] * hx_old[id] + hx_factor2[id]
				* ((ey_old[id + 1] - ey_old[id]) / grid_width_z);
		}
		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
		for (int id = ID; id < ID + batch_size; id++)
		{
			// Calculate the current electric field from the CURRENT magnetic field
			ey_current[id] = geometry_factor[id] * (ey_factor1[id] * ey_old[id] + ey_factor2[id]
				* ((hx_current[id] - hx_current[id - 1]) / grid_width_z));

			// Zero ey out if the soure_factor is 1
			ey_current[id] -= abs(source_factor[id]) * ey_current[id];
			// Than add the source. This way a hard source is created!
			ey_current[id] += source_factor[id] * source_amplitude * ramped_sin_fp64(current_time, source_ramp_len, source_frequency);
		}
		// Synchronize the workitems after calculating the current fields
		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
		// Now memory consistency is assured and the fields can be copied
		for (int id = ID; id < ID + batch_size; id++)
		{
			hx_old[id] = hx_current[id];
			ey_old[id] = ey_current[id];
		}
		// Again the workitems have to be synchronized
		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	}
}


