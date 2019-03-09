#pragma OPENCL EXTENSION cl_khr_fp64 : enable

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
			       int udepth)
{
	// Get the actual id
	const int id = get_global_id(0) + get_global_id(1) * uwidth + get_global_id(2) * uwidth * udepth + 1 + uwidth + uwidth * udepth;

	// Calculate the current magnetic field from the old electric field
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

	// Calculate the current electric field from the current magnetic field
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
	current_e[id] *= geometry[id];
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
			       int udepth)
{
	// Get the actual id
	const int id = get_global_id(0) + get_global_id(1) * uwidth + get_global_id(2) * uwidth * udepth + 1 + uwidth + uwidth * udepth;

	// Calculate the current magnetic field from the old electric field
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

	// Calculate the current electric field from the current magnetic field
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
	current_e[id] *= geometry[id];
}
