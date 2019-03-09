#pragma OPENCL EXTENSION cl_khr_fp64 : enable

kernel void fdtd1d_noiter_fp32(global float* ey_old,
		   	       global float* hx_old,
		   	       global float* ey_current,
		   	       global float* hx_current,
		   	       constant float* ey_factor1,
			       constant float* ey_factor2,
		   	       constant float* hx_factor1,
			       constant float* hx_factor2,
			       constant float* geometry_factor,
			       float grid_width_z)
{
	// Get the actual id
	const int id = get_global_id(0) + 1;

	// Calculate the current magnetic field from the OLD electric field
	hx_current[id] = hx_factor1[id] * hx_old[id] + hx_factor2[id]
		* ((ey_old[id + 1] - ey_old[id]) / grid_width_z);
	
	// Calculate the current electric field from the CURRENT magnetic field
	ey_current[id] = geometry_factor[id] * (ey_factor1[id] * ey_old[id] + ey_factor2[id]
		* ((hx_current[id] - hx_current[id - 1]) / grid_width_z));
}

kernel void fdtd1d_noiter_fp64(global double* ey_old,
		   	       global double* hx_old,
		   	       global double* ey_current,
		   	       global double* hx_current,
		   	       constant double* ey_factor1,
			       constant double* ey_factor2,
		   	       constant double* hx_factor1,
			       constant double* hx_factor2,
			       constant double* geometry_factor,
			       double grid_width_z)
{
	// Get the actual id
	const int id = get_global_id(0) + 1;

	// Calculate the current magnetic field from the OLD electric field
	hx_current[id] = hx_factor1[id] * hx_old[id] + hx_factor2[id]
		* ((ey_old[id + 1] - ey_old[id]) / grid_width_z);
	
	// Calculate the current electric field from the CURRENT magnetic field
	ey_current[id] = geometry_factor[id] * (ey_factor1[id] * ey_old[id] + ey_factor2[id]
		* ((hx_current[id] - hx_current[id - 1]) / grid_width_z));
}
