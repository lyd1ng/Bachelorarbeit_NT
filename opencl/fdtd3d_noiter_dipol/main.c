#include <math.h>
#include <stdio.h>
#include <alloca.h>
#include <getopt.h>
#include <CL/opencl.h>
#include "c_helper.h"
#include "fdtd_helper.h"

#ifndef fp
#define fp float
#endif

int USIZE;

// Es mus float4 verwendet werden auch wenn im Kernel
// float3 verwendet wird...
typedef struct __attribute__((__packed__)) fp4 {
	fp x;
	fp y;
	fp z;
	fp w;
} fp4;


fp abs4(fp4 x)
{
	return sqrt(x.x * x.x + x.y * x.y + x.z * x.z + x.w * x.w);
}


void verbose(int error, char* msg)
{
	if (error) { printf("%s\n", msg); }
}



void set_pml_geometry(const int uwidth, const int uheight, const int udepth, const fp time_step, fp* sigmax, fp* sigmay, fp* sigmaz)
{
    int pml_width = 20;
    int pml_height = 20;
	int pml_depth = 20;
    fp f = vacuum_permittivity / (2.0 * time_step);
	for (int z=1; z < udepth - 1; z++)
	{
    	for (int y=1; y < uheight - 1; y++)
    	{
    	    for (int x=1; x < pml_width; x++)
			{
				int id = y * USIZE + z * USIZE * USIZE;
    	        sigmax[id + pml_width - x - 1] = f * pow((fp)x / (fp)pml_width, 3);
    	        sigmax[id + uwidth - pml_width + x] = f * pow((fp)x / (fp)pml_width, 3);
			}
    	}
	}
	for (int z=1; z < udepth - 1; z++)
	{
    	for (int x=1; x < uwidth - 1; x++)
    	{
    	    for (int y=1; y < pml_height; y++)
			{
				int id = x + z * USIZE * USIZE;
    	        sigmay[id + (pml_height - y - 1) * USIZE] = f * pow((fp)y / (fp)pml_height, 3);
    	        sigmay[id + (uheight - pml_height + y) * USIZE] = f * pow((fp)y / (fp)pml_height, 3);
			}
    	}
	}
	for (int z=1; z < pml_depth; z++)
	{
    	for (int x=1; x < uwidth - 1; x++)
    	{
    	    for (int y=1; y < uheight; y++)
			{
				int id = x + y * USIZE;
    	        sigmaz[id + (pml_depth - z - 1) * USIZE * USIZE] = f * pow((fp)z / (fp)pml_depth, 3);
    	        sigmaz[id + (USIZE - pml_depth + z) * USIZE * USIZE] = f * pow((fp)z / (fp)pml_depth, 3);
			}
    	}
	}
}

void calculate_factors_universe(const int uwidth,
		       const int uheight,
		       fp time_step,
		       fp* permittivity,
		       fp* permeability,
		       fp* sigmax,
		       fp* sigmay,
			   fp* sigmaz,
			   fp4* ex_factor_universe,
			   fp4* ey_factor_universe,
		       fp4* ez_factor_universe,
		       fp4* hx_factor_universe,
		       fp4* hy_factor_universe,
			   fp4* hz_factor_universe)
{
	fp hf = 0;
	for (int z=0; z < USIZE; z++)
	{
    	for (int y=0; y < USIZE; y++)
    	{
    	    for (int x=0; x < USIZE; x++)
			{
				int id = x + y * USIZE + z * USIZE * USIZE;
				// Calculate the ex factors
    	    	ex_factor_universe[id].x = 1 - ((time_step * (sigmay[id] + sigmaz[id])) / vacuum_permittivity);
    	        ex_factor_universe[id].y = (sigmay[id] * sigmaz[id] * pow(time_step, 2)) / pow(vacuum_permittivity, 2);
    	        ex_factor_universe[id].z = (light_speed * time_step) / permittivity[id];
				ex_factor_universe[id].w = (light_speed * sigmax[id] * pow(time_step, 2)) / (permittivity[id] * vacuum_permittivity);

				// Calculate the ey factors
    	    	ey_factor_universe[id].x = 1 - ((time_step * (sigmax[id] + sigmaz[id])) / vacuum_permittivity);
    	        ey_factor_universe[id].y = (sigmax[id] * sigmaz[id] * pow(time_step, 2)) / pow(vacuum_permittivity, 2);
    	        ey_factor_universe[id].z = (light_speed * time_step) / permittivity[id];
				ey_factor_universe[id].w = (light_speed * sigmay[id] * pow(time_step, 2)) / (permittivity[id] * vacuum_permittivity);

				// Calculate the ez factors
    	    	ez_factor_universe[id].x = 1 - ((time_step * (sigmax[id] + sigmay[id])) / vacuum_permittivity);
    	        ez_factor_universe[id].y = (sigmax[id] * sigmay[id] * pow(time_step, 2)) / pow(vacuum_permittivity, 2);
    	        ez_factor_universe[id].z = (light_speed * time_step) / permittivity[id];
				ez_factor_universe[id].w = (light_speed * sigmaz[id] * pow(time_step, 2)) / (permittivity[id] * vacuum_permittivity);

				// Calculate the hx factors
				hf = (sigmay[id] + sigmaz[id]) / (2 * vacuum_permittivity)
					 + (sigmay[id] * sigmaz[id] * time_step) / (4 * pow(vacuum_permittivity, 2))
					 + 1 / time_step;
    	        hx_factor_universe[id].x = (- (sigmay[id] + sigmaz[id]) / (2 * vacuum_permittivity)
											    - (sigmay[id] * sigmaz[id] * time_step) / (4 * pow(vacuum_permittivity, 2))
											    + 1 / time_step) / hf;
    	        hx_factor_universe[id].y = (light_speed / permittivity[id]) /  hf;
    	        hx_factor_universe[id].z = ((light_speed * sigmax[id] * time_step) / (permeability[id] * vacuum_permittivity)) / hf;
				hx_factor_universe[id].w = ((sigmay[id] * sigmaz[id] * time_step) / pow(vacuum_permittivity, 2)) / hf;

				// Calculate the hy factors
				hf = (sigmax[id] + sigmaz[id]) / (2 * vacuum_permittivity)
					 + (sigmax[id] * sigmaz[id] * time_step) / (4 * pow(vacuum_permittivity, 2))
					 + 1 / time_step;
    	        hy_factor_universe[id].x = (- (sigmax[id] + sigmaz[id]) / (2 * vacuum_permittivity)
											    - (sigmax[id] * sigmaz[id] * time_step) / (4 * pow(vacuum_permittivity, 2))
											    + 1 / time_step) / hf;
    	        hy_factor_universe[id].y = (light_speed / permittivity[id]) /  hf;
    	        hy_factor_universe[id].z = ((light_speed * sigmay[id] * time_step) / (permeability[id] * vacuum_permittivity)) / hf;
				hy_factor_universe[id].w = ((sigmax[id] * sigmaz[id] * time_step) / pow(vacuum_permittivity, 2)) / hf;

				// Calculate the hz factors
				hf = (sigmax[id] + sigmay[id]) / (2 * vacuum_permittivity)
					 + (sigmax[id] * sigmay[id] * time_step) / (4 * pow(vacuum_permittivity, 2))
					 + 1 / time_step;
    	        hz_factor_universe[id].x = (- (sigmax[id] + sigmay[id]) / (2 * vacuum_permittivity)
											    - (sigmax[id] * sigmay[id] * time_step) / (4 * pow(vacuum_permittivity, 2))
											    + 1 / time_step) / hf;
    	        hz_factor_universe[id].y = (light_speed / permittivity[id]) /  hf;
    	        hz_factor_universe[id].z = ((light_speed * sigmaz[id] * time_step) / (permeability[id] * vacuum_permittivity)) / hf;
				hz_factor_universe[id].w = ((sigmax[id] * sigmay[id] * time_step) / pow(vacuum_permittivity, 2)) / hf;
			}
    	}
	}


}



int main(int argc, char** argv)
{
	cl_int errNum;
	cl_kernel kernel;
	cl_program program;
	cl_context context;
	cl_device_id device;
	cl_command_queue queue;
	cl_platform_id platform;

	// The simulation parameters
	fp fraction = 1;
	fp ramped_sin_length;
	fp ramped_sin_amplitude;
	fp ramped_sin_frequency;
	fp max_time;
	int samplerate_space;
	int samplerate_time;

	static struct option long_options[] =
	{
		{"usize", required_argument, 0, (int)'U'},
		{"samplerate_space", required_argument, 0, (int)'s'},
		{"samplerate_time", required_argument, 0, (int)'t'},
		{"max_time", required_argument, 0, (int)'T'},
		{"ramped_sin_length", required_argument, 0, (int)'l'},
		{"ramped_sin_amplitude", required_argument, 0, (int)'a'},
		{"ramped_sin_frequency", required_argument, 0, (int)'f'}
	};

	int c;
	int option_index = 0;

	while ((c = getopt_long(argc, argv, "U:s:t:T:l:a:f:", long_options, &option_index)) != -1)
	{
		switch (c)
		{
			case 'U':
				USIZE = atoi(optarg);
				break;
			case 's':
				samplerate_space = atoi(optarg);
				break;
			case 't':
				samplerate_time = atoi(optarg);
				break;
			case 'T':
				max_time = atof(optarg);
				break;
			case 'l':
				ramped_sin_length = atof(optarg);
				break;
			case 'a':
				ramped_sin_amplitude = atof(optarg);
				break;
			case 'f':
				ramped_sin_frequency = atof(optarg);
				break;
		}
	}

	fp grid_width_x = get_grid_width(ramped_sin_frequency, fraction, samplerate_space);
	fp grid_width_y = get_grid_width(ramped_sin_frequency, fraction, samplerate_space);
	fp grid_width_z = get_grid_width(ramped_sin_frequency, fraction, samplerate_space);

	fp time_step = get_time_resolution_3d(fraction, grid_width_x, grid_width_y, grid_width_z)
			  / (fp)samplerate_time;
	fp current_time = 0;

	// Make max_time and ramped_sin_length relativ to time_step,
	// so max_time is the number of iterations total
	max_time = max_time * time_step;
	ramped_sin_length = ramped_sin_length * time_step;

	// Print the parameters as gnuplot comments
	printf("# dimensions: %dx%d\n", USIZE, USIZE);
	printf("# space_sr: %d\n", samplerate_time);
	printf("# time_sr: %d\n", samplerate_space);
	printf("# floating point: %d\n", 8 * sizeof(fp));
	printf("# ramped_sin_frequency: %f\n", ramped_sin_frequency);
	printf("# ramped_sin_amplitude: %f\n", ramped_sin_amplitude);
	printf("# ramped_sin_length: %f\n", ramped_sin_length);
	printf("# max_time: %.32f\n", max_time);



	// The necessary arrays
	fp4* old_e_universe = (fp4*)calloc(USIZE * USIZE * USIZE, sizeof(fp4));
	fp4* old_h_universe = (fp4*)calloc(USIZE * USIZE * USIZE, sizeof(fp4));
	fp4* current_e_universe = (fp4*)calloc(USIZE * USIZE * USIZE, sizeof(fp4));
	fp4* current_h_universe = (fp4*)calloc(USIZE * USIZE * USIZE, sizeof(fp4));
	fp4* e_curl_integral_universe = (fp4*)calloc(USIZE * USIZE * USIZE, sizeof(fp4));
	fp4* e_field_integral_universe = (fp4*)calloc(USIZE * USIZE * USIZE, sizeof(fp4));
	fp4* h_curl_integral_universe = (fp4*)calloc(USIZE * USIZE * USIZE, sizeof(fp4));
	fp4* h_field_integral_universe = (fp4*)calloc(USIZE * USIZE * USIZE, sizeof(fp4));
	fp4* ex_factor_universe = (fp4*)calloc(USIZE * USIZE * USIZE, sizeof(fp4));
	fp4* ey_factor_universe = (fp4*)calloc(USIZE * USIZE * USIZE, sizeof(fp4));
	fp4* ez_factor_universe = (fp4*)calloc(USIZE * USIZE * USIZE, sizeof(fp4));
	fp4* hx_factor_universe = (fp4*)calloc(USIZE * USIZE * USIZE, sizeof(fp4));
	fp4* hy_factor_universe = (fp4*)calloc(USIZE * USIZE * USIZE, sizeof(fp4));
	fp4* hz_factor_universe = (fp4*)calloc(USIZE * USIZE * USIZE, sizeof(fp4));

	fp* sigmax = (fp*)calloc(USIZE * USIZE * USIZE, sizeof(fp));
	fp* sigmay = (fp*)calloc(USIZE * USIZE * USIZE, sizeof(fp));
	fp* sigmaz = (fp*)calloc(USIZE * USIZE * USIZE, sizeof(fp));
	fp* permittivity = (fp*)calloc(USIZE * USIZE * USIZE, sizeof(fp));
	fp* permeability = (fp*)calloc(USIZE * USIZE * USIZE, sizeof(fp));
	char* geometry = (char*)calloc(sizeof(char) * USIZE * USIZE * USIZE, sizeof(fp));

	// The permeability and permittivity are relativ
	// to the free space values. So 1 is the correct value
	// to simulate the free space.
	for (int z = 0; z < USIZE; z++)
	{
		for (int y = 0; y < USIZE; y++)
		{
			for (int x = 0; x < USIZE; x++)
			{
				int id = x + y * USIZE + z * USIZE * USIZE;
				permittivity[id] = 1;
				permeability[id] = 1;
				geometry[id] = 1;
			}
		}
	}

		
	// Set the pml geometry
	set_pml_geometry(USIZE, USIZE, USIZE, time_step, sigmax, sigmay, sigmaz);

	// Calculate factors
	calculate_factors_universe(USIZE, USIZE, time_step, permittivity, permeability, sigmax, sigmay, sigmaz,
			ex_factor_universe,
			ey_factor_universe,
			ez_factor_universe,
			hx_factor_universe,
			hy_factor_universe,
			hz_factor_universe);
	// Get the first platfrom available
	errNum = clGetPlatformIDs(1, &platform, NULL);
	verbose(errNum, "Error getting the platform");

	// Create a lambda dipol geometry one cell in +z direction from the source
	for (int x = USIZE / 2 - samplerate_space / 2; x < USIZE / 2 + samplerate_space / 2; x++)
	{
		geometry[x + (USIZE / 2) * USIZE + (USIZE / 2 + 1) * USIZE * USIZE] = 0;
	}
	geometry[(USIZE / 2) + (USIZE / 2) * USIZE + (USIZE / 2 + 1) * USIZE * USIZE] = 1;


	// Get a GPU device
	errNum = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
	verbose(errNum, "Error getting the device");

	// Create a Context
	cl_context_properties context_properties[] =
	{
			CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0
	};
	context = clCreateContext(context_properties, 1, &device, NULL, NULL, &errNum);
	verbose(errNum, "Error creating the context");

	// Create the command queue
	queue = clCreateCommandQueue(context, device, 0, &errNum);
	verbose(errNum, "Error creating the command queue");

	// Create and build the program
	FILE* code_fd = fopen("fdtd3d.cl", "r");
	char *code;
	size_t code_length;
	read_fileh(code_fd, &code, &code_length);
	program = clCreateProgramWithSource(context, 1,
			(const char**)&code, &code_length, &errNum);
	verbose(errNum, "Error creating the program");
	errNum = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
	free(code);

	if (errNum)
	{
		size_t log_size;
		char* log;
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0,
				NULL, &log_size);
		log = (char*)alloca(log_size * sizeof(char));
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size,
				log, NULL);
		printf("%s\n", log);
		return 0;
	}

	// Create the buffers
	// Old Buffers (read only)
	int n = USIZE * USIZE * USIZE;
	cl_mem old_e_universeb = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
				 n * sizeof(fp4), old_e_universe, &errNum);
	verbose(errNum, "Error creating old_e_universeb");

	cl_mem old_h_universeb = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
				 n * sizeof(fp4), old_h_universe, &errNum);
	verbose(errNum, "Error creating old_h_universeb");
	
	// Current Buffers (write only)
	cl_mem current_e_universeb = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
				 n * sizeof(fp4), current_e_universe, &errNum);
	verbose(errNum, "Error creating current_e_universeb");
	cl_mem current_h_universeb = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
				 n * sizeof(fp4), current_h_universe, &errNum);
	verbose(errNum, "Error creating current_h_universeb");

	// Curl Integral (read_write)
	cl_mem e_curl_integral_universeb = clCreateBuffer(context, CL_MEM_READ_WRITE
			| CL_MEM_USE_HOST_PTR, n * sizeof(fp4), e_curl_integral_universe,
			&errNum);
	verbose(errNum, "Error creating e_curl_integral_universeb");
	cl_mem h_curl_integral_universeb = clCreateBuffer(context, CL_MEM_READ_WRITE
			| CL_MEM_USE_HOST_PTR, n * sizeof(fp4), h_curl_integral_universe,
			&errNum);
	verbose(errNum, "Error creating h_curl_integral_universeb");

	// Field Integral (read_write)
	cl_mem e_field_integral_universeb = clCreateBuffer(context, CL_MEM_READ_WRITE
			| CL_MEM_USE_HOST_PTR, n * sizeof(fp4), e_field_integral_universe,
			&errNum);
	verbose(errNum, "Error creating e_field_integral_universeb");
	cl_mem h_field_integral_universeb = clCreateBuffer(context, CL_MEM_READ_WRITE
			| CL_MEM_USE_HOST_PTR, n * sizeof(fp4), h_field_integral_universe,
			&errNum);
	verbose(errNum, "Error creating h_field_integral_universeb");

	// E-Factors (read only)
	cl_mem ex_factor_universeb = clCreateBuffer(context, CL_MEM_READ_ONLY
			| CL_MEM_USE_HOST_PTR, n * sizeof(fp4), ex_factor_universe,
			&errNum);
	verbose(errNum, "Error creating ex_factor_universeb");
	cl_mem ey_factor_universeb = clCreateBuffer(context, CL_MEM_READ_ONLY
			| CL_MEM_USE_HOST_PTR, n * sizeof(fp4), ey_factor_universe,
			&errNum);
	verbose(errNum, "Error creating ey_factor_universeb");
	cl_mem ez_factor_universeb = clCreateBuffer(context, CL_MEM_READ_ONLY
			| CL_MEM_USE_HOST_PTR, n * sizeof(fp4), ez_factor_universe,
			&errNum);
	verbose(errNum, "Error creating ez_factor_universeb");

	// H-Factors (read only)
	cl_mem hx_factor_universeb = clCreateBuffer(context, CL_MEM_READ_ONLY
			| CL_MEM_USE_HOST_PTR, n * sizeof(fp4), hx_factor_universe,
			&errNum);
	verbose(errNum, "Error creating hx_factor_universeb");
	cl_mem hy_factor_universeb = clCreateBuffer(context, CL_MEM_READ_ONLY
			| CL_MEM_USE_HOST_PTR, n * sizeof(fp4), hy_factor_universe,
			&errNum);
	verbose(errNum, "Error creating hy_factor_universeb");
	cl_mem hz_factor_universeb = clCreateBuffer(context, CL_MEM_READ_ONLY
			| CL_MEM_USE_HOST_PTR, n * sizeof(fp4), hz_factor_universe,
			&errNum);
	verbose(errNum, "Error creating hz_factor_universeb");

	// The Geometry
	cl_mem geometryb = clCreateBuffer(context, CL_MEM_READ_ONLY
			| CL_MEM_USE_HOST_PTR, n * sizeof(char), geometry,
			&errNum);
	verbose(errNum, "Error creating geometryb");

	// Pass the arrays to the device
	errNum = clEnqueueWriteBuffer(queue, old_e_universeb, CL_TRUE, 0, USIZE * USIZE * USIZE * sizeof(fp4),
			old_e_universe, 0, NULL, NULL);
	verbose(errNum, "Error writing old_e_universeb buffer");
	errNum = clEnqueueWriteBuffer(queue, old_h_universeb, CL_TRUE, 0, USIZE * USIZE * USIZE * sizeof(fp4),
			old_h_universe, 0, NULL, NULL);
	verbose(errNum, "Error writing old_h_universeb buffer");

	errNum = clEnqueueWriteBuffer(queue, current_e_universeb, CL_TRUE, 0, USIZE * USIZE * USIZE * sizeof(fp4),
			current_e_universe, 0, NULL, NULL);
	verbose(errNum, "Error writing current_e_universeb buffer");
	errNum = clEnqueueWriteBuffer(queue, current_h_universeb, CL_TRUE, 0, USIZE * USIZE * USIZE * sizeof(fp4),
			current_h_universe, 0, NULL, NULL);
	verbose(errNum, "Error writing current_h_universeb buffer");

	errNum = clEnqueueWriteBuffer(queue, e_curl_integral_universeb, CL_TRUE, 0, USIZE * USIZE * USIZE * sizeof(fp4),
			e_curl_integral_universe, 0, NULL, NULL);
	verbose(errNum, "Error writing e_curl_integral_universeb buffer");
	errNum = clEnqueueWriteBuffer(queue, h_curl_integral_universeb, CL_TRUE, 0, USIZE * USIZE * USIZE * sizeof(fp4),
			h_curl_integral_universe, 0, NULL, NULL);
	verbose(errNum, "Error writing h_curl_integral_universeb buffer");

	errNum = clEnqueueWriteBuffer(queue, e_field_integral_universeb, CL_TRUE, 0, USIZE * USIZE * USIZE * sizeof(fp4),
			e_field_integral_universe, 0, NULL, NULL);
	verbose(errNum, "Error writing e_field_integral_universeb buffer");
	errNum = clEnqueueWriteBuffer(queue, h_field_integral_universeb, CL_TRUE, 0, USIZE * USIZE * USIZE * sizeof(fp4),
			h_field_integral_universe, 0, NULL, NULL);
	verbose(errNum, "Error writing h_field_integral_universeb buffer");
	
	errNum = clEnqueueWriteBuffer(queue, ex_factor_universeb, CL_TRUE, 0, USIZE * USIZE * USIZE * sizeof(fp4),
			ex_factor_universe, 0, NULL, NULL);
	verbose(errNum, "Error writing ex_factor_universeb");
	errNum = clEnqueueWriteBuffer(queue, ey_factor_universeb, CL_TRUE, 0, USIZE * USIZE * USIZE * sizeof(fp4),
			ey_factor_universe, 0, NULL, NULL);
	verbose(errNum, "Error writing ey_factor_universeb");
	errNum = clEnqueueWriteBuffer(queue, ez_factor_universeb, CL_TRUE, 0, USIZE * USIZE * USIZE * sizeof(fp4),
			ez_factor_universe, 0, NULL, NULL);
	verbose(errNum, "Error writing ez_factor_universeb");

	errNum = clEnqueueWriteBuffer(queue, hx_factor_universeb, CL_TRUE, 0, USIZE * USIZE * USIZE * sizeof(fp4),
			hx_factor_universe, 0, NULL, NULL);
	verbose(errNum, "Error writing hx_factor_universeb");
	errNum = clEnqueueWriteBuffer(queue, hy_factor_universeb, CL_TRUE, 0, USIZE * USIZE * USIZE * sizeof(fp4),
			hy_factor_universe, 0, NULL, NULL);
	verbose(errNum, "Error writing hy_factor_universeb");
	errNum = clEnqueueWriteBuffer(queue, hz_factor_universeb, CL_TRUE, 0, USIZE * USIZE * USIZE * sizeof(fp4),
			hz_factor_universe, 0, NULL, NULL);
	verbose(errNum, "Error writing hz_factor_universeb");
	
	errNum = clEnqueueWriteBuffer(queue, geometryb, CL_TRUE, 0, USIZE * USIZE * USIZE* sizeof(char),
			geometry, 0, NULL, NULL);
	verbose(errNum, "Error writing geometryb buffer");


	// Create the kernel object depending on fp
	if (sizeof(fp) == 4)
	{
		kernel = clCreateKernel(program, "fdtd3d_noiter_fp32", &errNum);
	}
	else
	{
		kernel = clCreateKernel(program, "fdtd3d_noiter_fp64", &errNum);
	}
	verbose(errNum, "Error creating kernel");
	

	// Set the kernel args
	errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&old_e_universeb);
	verbose(errNum, "Error setting old_e_universeb arg");
	
	errNum = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&old_h_universeb);
	verbose(errNum, "Error setting old_h_universeb arg");

	errNum = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&current_e_universeb);
	verbose(errNum, "Error setting current_e_universeb arg");

	errNum = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&current_h_universeb);
	verbose(errNum, "Error setting current_h_universeb arg");

	errNum = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&e_curl_integral_universeb);
	verbose(errNum, "Error setting e_curl_integral_universeb arg");

	errNum = clSetKernelArg(kernel, 5, sizeof(cl_mem), (void*)&e_field_integral_universeb);
	verbose(errNum, "Error setting e_field_integral_universeb arg");

	errNum = clSetKernelArg(kernel, 6, sizeof(cl_mem), (void*)&h_curl_integral_universeb);
	verbose(errNum, "Error setting h_curl_integral_universeb arg");

	errNum = clSetKernelArg(kernel, 7, sizeof(cl_mem), (void*)&h_field_integral_universeb);
	verbose(errNum, "Error setting h_field_integral_universeb arg");
	
	errNum = clSetKernelArg(kernel, 8, sizeof(cl_mem), (void*)&ex_factor_universeb);
	verbose(errNum, "Error setting ex_factor1_universeb arg");
	errNum = clSetKernelArg(kernel, 9, sizeof(cl_mem), (void*)&ey_factor_universeb);
	verbose(errNum, "Error setting ex_factor2_universeb arg");
	errNum = clSetKernelArg(kernel, 10, sizeof(cl_mem), (void*)&ez_factor_universeb);
	verbose(errNum, "Error setting ex_factor3_universeb arg");

	errNum = clSetKernelArg(kernel, 11, sizeof(cl_mem), (void*)&hx_factor_universeb);
	verbose(errNum, "Error setting hx_factor1_universeb arg");
	errNum = clSetKernelArg(kernel, 12, sizeof(cl_mem), (void*)&hy_factor_universeb);
	verbose(errNum, "Error setting hx_factor2_universeb arg");
	errNum = clSetKernelArg(kernel, 13, sizeof(cl_mem), (void*)&hz_factor_universeb);
	verbose(errNum, "Error setting hx_factor3_universeb arg");

	errNum = clSetKernelArg(kernel, 14, sizeof(cl_mem), (void*)&geometryb);
	verbose(errNum, "Error setting geometryb arg");
	errNum = clSetKernelArg(kernel, 15, sizeof(fp), (void*)&grid_width_x);
	verbose(errNum, "Error setting grid_width_x");
	errNum = clSetKernelArg(kernel, 16, sizeof(fp), (void*)&grid_width_y);
	verbose(errNum, "Error setting grid_width_y");
	errNum = clSetKernelArg(kernel, 17, sizeof(fp), (void*)&grid_width_z);
	verbose(errNum, "Error setting grid_width_z");

	int uwidth = USIZE;
	int udepth = USIZE;
	errNum = clSetKernelArg(kernel, 18, sizeof(int), (void*)&uwidth);
	verbose(errNum, "Error setting uwidth");
	errNum = clSetKernelArg(kernel, 19, sizeof(int), (void*)&udepth);
	verbose(errNum, "Error setting udepth");


	// Invoke the kernel
	size_t gws[3] = {USIZE - 2, USIZE - 2, USIZE - 2};
	while (current_time < max_time)
	{
		errNum = clEnqueueNDRangeKernel(queue, kernel, 3, NULL, gws, NULL, 0, NULL, NULL);
		verbose(errNum, "Error invoking the kernel");
		if (errNum)
		{
			printf("%d\n", errNum);
			exit(0);
		}

		// The ez field has to be copied manually to inject the source
		errNum = clEnqueueReadBuffer(queue, current_e_universeb, CL_TRUE,
				0, USIZE * USIZE * USIZE * sizeof(fp4), old_e_universe, 0, NULL, NULL);
		verbose(errNum, "Error reading the current_e_universeb");

		// Inject the source
		int sid = USIZE / 2 + (USIZE / 2) * USIZE + (USIZE / 2) * USIZE * USIZE;
		old_e_universe[sid].z = 2 * ramped_sinus(current_time, ramped_sin_length * time_step, ramped_sin_amplitude, ramped_sin_frequency);
		//old_e_universe[sid - 1].x = -ramped_sinus(current_time, ramped_sin_length * time_step, ramped_sin_amplitude, ramped_sin_frequency);
		//old_e_universe[sid + 1].x = ramped_sinus(current_time, ramped_sin_length * time_step, ramped_sin_amplitude, ramped_sin_frequency);


		// Now the field can be written to the device again
		errNum = clEnqueueWriteBuffer(queue, old_e_universeb, CL_TRUE,
				0, USIZE * USIZE * USIZE * sizeof(fp4), old_e_universe, 0, NULL, NULL);
		verbose(errNum, "Error Writing old_e_universe");

		errNum = clEnqueueCopyBuffer(queue, current_h_universeb, old_h_universeb, 0, 0,
				USIZE * USIZE * USIZE * sizeof(fp4), 0, NULL, NULL);
		verbose(errNum, "Error copying buffer current_h_universeb");
		clFinish(queue);
		current_time += time_step;
	}
	
	errNum = clEnqueueReadBuffer(queue, current_e_universeb, CL_TRUE, 0, USIZE * USIZE * USIZE * sizeof(fp4),
		(void*)(current_e_universe), 0, NULL, NULL);
	
	// Print the results in an ugly list
	for (int z = USIZE / 2; z < USIZE / 2 + 1; z++)
	{
		for (int y = 0; y < USIZE; y++)
		{
			for (int x = 0; x < USIZE; x++)
			{
				int id = x + y * USIZE + z * USIZE * USIZE;
				// Print as a gnuplot vector x z y dx dy dz
				// printf("%d\t%d\t%d\t%.32f\t%.32f\t%.32f\n", x, y, z, current_e_universe[id].x, current_e_universe[id].y, current_e_universe[id].z);
				// printf("%d\t%d\t%d\t%.32f\t%.32f\t%.32f\n", x, y, z, sigmax[id], sigmay[id], sigmaz[id]);
				printf("%d\t%d\t%.32f\n", x, y, current_e_universe[id].z);
			}

		}
	}
	return 0;
}
