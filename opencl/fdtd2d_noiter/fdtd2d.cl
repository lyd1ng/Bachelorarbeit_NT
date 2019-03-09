#pragma OPENCL EXTENSION cl_khr_fp64 : enable

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
			       int uwidth)
{
	// Get the actual id
	const int id = get_global_id(0) + get_global_id(1) * uwidth + 1 + uwidth;

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
	

	// Calculate the curl term to simplify the actual calculation
	const float ez_curl_term = ((hy_current[id] - hy_current[id - 1]) / grid_width_x)
				   - ((hx_current[id] - hx_current[id - uwidth]) / grid_width_y);

	// Euler integration of the curl term
	ez_integral[id] += ez_old[id];
	// Calculate the electric field
	ez_current[id] = geometry[id] * (ez_factor1[id] * ez_old[id]
			 - ez_factor2[id] * ez_integral[id]
			 + ez_factor3[id] * ez_curl_term);
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
			       int uwidth)
{
	// Get the actual id
	const int id = get_global_id(0) + get_global_id(1) * uwidth + 1 + uwidth;

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
	

	// Calculate the curl term to simplify the actual calculation
	const double ez_curl_term = ((hy_current[id] - hy_current[id - 1]) / grid_width_x)
				   - ((hx_current[id] - hx_current[id - uwidth]) / grid_width_y);

	// Euler integration of the curl term
	ez_integral[id] += ez_old[id];
	// Calculate the electric field
	ez_current[id] = ez_factor1[id] * ez_old[id]
			 - ez_factor2[id] * ez_integral[id]
			 + ez_factor3[id] * ez_curl_term;
}
