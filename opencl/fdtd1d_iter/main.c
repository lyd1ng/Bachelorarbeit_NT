#include <math.h>
#include <stdio.h>
#include <alloca.h>
#include <CL/opencl.h>
#include "c_helper.h"
#include "fdtd_helper.h"

// UWIDTH has to be divisible by max_work_group_size
#define UWIDTH 1026

#ifndef fp
#define fp float
#endif


void verbose(int error, char* msg)
{
	if (error) { printf("%s\n", msg); }
}


void set_pml_geometry(fp sigmaz_universe[UWIDTH], fp time_step)
{
    int pml_width = 40;
    fp f = vacuum_permittivity / (2.0 * time_step);
    for (int x=0; x < pml_width; x++)
    {
	sigmaz_universe[pml_width - x] = f * pow((fp)x / (fp)pml_width, 3);
        sigmaz_universe[UWIDTH - pml_width + x] = f * pow((fp)x / (fp)pml_width, 3);
    }
}


void calculate_factors(fp time_step,
		       fp permittivity_universe[UWIDTH],
		       fp permeability_universe[UWIDTH],
		       fp ey_factor1_universe[UWIDTH],
		       fp ey_factor2_universe[UWIDTH],
		       fp hx_factor1_universe[UWIDTH],
		       fp hx_factor2_universe[UWIDTH],
		       fp sigmaz_universe[UWIDTH])
{
    for (int z=0; z < UWIDTH; z++)
    {
	    ey_factor1_universe[z] = 1 - ((time_step * sigmaz_universe[z]) / vacuum_permittivity);
	    ey_factor2_universe[z] = get_electric_update_coefficient_1d(time_step, permittivity_universe[z]);
	    hx_factor1_universe[z] = (-sigmaz_universe[z] / (2 * vacuum_permittivity) + 1.0 / time_step)
		    / (sigmaz_universe[z] / (2 * vacuum_permittivity) + 1.0 / time_step);
	    hx_factor2_universe[z] = (light_speed / permeability_universe[z]) /
		    (sigmaz_universe[z] / (2 * vacuum_permittivity) + 1.0 / time_step);
    }
}


int main()
{
	cl_int errNum;
	cl_kernel kernel;
	cl_program program;
	cl_context context;
	cl_device_id device;
	cl_command_queue queue;
	cl_platform_id platform;

	// The necessary arrays
	fp old_ey_universe[UWIDTH] = {0};
	fp old_hx_universe[UWIDTH] = {0};
	fp current_ey_universe[UWIDTH] = {0};
	fp current_hx_universe[UWIDTH] = {0};
	fp ey_factor1_universe[UWIDTH] = {0};
	fp ey_factor2_universe[UWIDTH] = {0};
	fp hx_factor1_universe[UWIDTH] = {0};
	fp hx_factor2_universe[UWIDTH] = {0};
	fp permittivity_universe[UWIDTH] = {0};
	fp permeability_universe[UWIDTH] = {0};
	fp sigmaz_universe[UWIDTH] = {0};
	char geometry[UWIDTH] = {0};
	char source[UWIDTH] = {0};

	// The permeability and permittivity are relativ
	// to the free space values. So 1 is the correct value
	// to simulate the free space.
	for (int x = 0; x < UWIDTH; x++)
	{
		permittivity_universe[x] = 1;
		permeability_universe[x] = 1;
		geometry[x] = 1;
	}

	// The simulation parameters
	fp ramped_sin_frequency = 2.4 * pow(10, 9);
	fp ramped_sin_amplitude = 1;
	float ramped_sin_length;
	fp fraction = 1;
	int samplerate_space = 20;
	int samplerate_time = 10;
	fp grid_width_z = get_grid_width(ramped_sin_frequency, fraction, samplerate_space);
	fp time_step = get_time_resolution_1d(fraction, grid_width_z)
			  / (fp)samplerate_time;
	fp current_time = 0;
	fp max_time = 10000 * time_step;
	ramped_sin_length = 100 * time_step;


	

	// Device characteristics
	int max_work_group_size = 256;
	int batch_size = (UWIDTH - 2) / max_work_group_size;

	// Print the parameters as gnuplot comments
	printf("# dimensions: %d\n", UWIDTH);
	printf("# space_sr: %d\n", samplerate_time);
	printf("# time_sr: %d\n", samplerate_space);
	printf("# floating point: %d\n", 8 * sizeof(fp));
	printf("# ramped_sin_frequency: %f\n", ramped_sin_frequency);
	printf("# ramped_sin_amplitude: %f\n", ramped_sin_amplitude);
	printf("# ramped_sin_length: %f\n", ramped_sin_length / time_step);
	printf("# max_time: %0.32f\n", max_time);
	
	// Set the pml geometry
	set_pml_geometry(sigmaz_universe, time_step);

	// Set the source
	source[UWIDTH / 2] = 1;

	// Calculate factors
	calculate_factors(time_step, permittivity_universe, permeability_universe, ey_factor1_universe, ey_factor2_universe,
			hx_factor1_universe, hx_factor2_universe, sigmaz_universe);

	// Get the first platfrom available
	errNum = clGetPlatformIDs(1, &platform, NULL);
	verbose(errNum, "Error getting the platform");

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
	FILE* code_fd = fopen("fdtd1d.cl", "r");
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
	// Old Buffers
	cl_mem old_ey_universeb = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
				 UWIDTH * sizeof(fp), old_ey_universe, &errNum);
	verbose(errNum, "Error creating old_ey_universeb");
	cl_mem old_hx_universeb = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
				 UWIDTH * sizeof(fp), old_hx_universe, &errNum);
	verbose(errNum, "Error creating old_hx_universeb");

	// Current Buffers
	cl_mem current_ey_universeb = clCreateBuffer(context, CL_MEM_READ_WRITE
			| CL_MEM_USE_HOST_PTR, UWIDTH * sizeof(fp), &current_ey_universe,
			&errNum);
	verbose(errNum, "Error creating current_ey_universeb");
	cl_mem current_hx_universeb = clCreateBuffer(context, CL_MEM_READ_WRITE 
			| CL_MEM_USE_HOST_PTR, UWIDTH * sizeof(fp), &current_hx_universe,
			&errNum);
	verbose(errNum, "Error creating current_hx_universeb");

	// Factors (read only)
	cl_mem ey_factor1_universeb = clCreateBuffer(context, CL_MEM_READ_ONLY
			| CL_MEM_USE_HOST_PTR, UWIDTH * sizeof(fp), ey_factor1_universe,
			&errNum);
	verbose(errNum, "Error creating ey_factor1_universe1b");
	cl_mem ey_factor2_universeb = clCreateBuffer(context, CL_MEM_READ_ONLY
			| CL_MEM_USE_HOST_PTR, UWIDTH * sizeof(fp), ey_factor2_universe,
			&errNum);
	verbose(errNum, "Error creating ey_factor2_universe1b");
	cl_mem hx_factor1_universeb = clCreateBuffer(context, CL_MEM_READ_ONLY
			| CL_MEM_USE_HOST_PTR, UWIDTH * sizeof(fp), hx_factor1_universe,
			&errNum);
	verbose(errNum, "Error creating hx_factor1_universe1b");
	cl_mem hx_factor2_universeb = clCreateBuffer(context, CL_MEM_READ_ONLY
			| CL_MEM_USE_HOST_PTR, UWIDTH * sizeof(fp), hx_factor2_universe,
			&errNum);
	verbose(errNum, "Error creating hx_factor2_universe1b");
	cl_mem geometryb = clCreateBuffer(context, CL_MEM_READ_ONLY
			| CL_MEM_USE_HOST_PTR, UWIDTH * sizeof(char), geometry,
			&errNum);
	verbose(errNum, "Error creating geometryb");
	cl_mem sourceb = clCreateBuffer(context, CL_MEM_READ_ONLY
			| CL_MEM_USE_HOST_PTR, UWIDTH * sizeof(char), source,
			&errNum);
	verbose(errNum, "Error creating sourceb");


	// Pass the arrays to the device
	errNum = clEnqueueWriteBuffer(queue, old_ey_universeb, CL_TRUE, 0, UWIDTH * sizeof(fp),
			&old_ey_universe, 0, NULL, NULL);
	verbose(errNum, "Error writing old_ey_universeb buffer");
	errNum = clEnqueueWriteBuffer(queue, old_hx_universeb, CL_TRUE, 0, UWIDTH * sizeof(fp),
			&old_hx_universe, 0, NULL, NULL);
	verbose(errNum, "Error writing old_hx_universeb buffer");
	errNum = clEnqueueWriteBuffer(queue, current_ey_universeb, CL_TRUE, 0, UWIDTH * sizeof(fp),
			&current_ey_universe, 0, NULL, NULL);
	verbose(errNum, "Error writing current_ey_universeb buffer");
	errNum = clEnqueueWriteBuffer(queue, current_hx_universeb, CL_TRUE, 0, UWIDTH * sizeof(fp),
			&current_hx_universe, 0, NULL, NULL);
	verbose(errNum, "Error writing current_hx_universeb buffer");
	errNum = clEnqueueWriteBuffer(queue, ey_factor1_universeb, CL_TRUE, 0, UWIDTH * sizeof(fp),
			&ey_factor1_universe, 0, NULL, NULL);
	errNum = clEnqueueWriteBuffer(queue, ey_factor2_universeb, CL_TRUE, 0, UWIDTH * sizeof(fp),
			&ey_factor2_universe, 0, NULL, NULL);
	verbose(errNum, "Error writing ey_factor2_universeb");
	errNum = clEnqueueWriteBuffer(queue, hx_factor1_universeb, CL_TRUE, 0, UWIDTH * sizeof(fp),
			&hx_factor1_universe, 0, NULL, NULL);
	verbose(errNum, "Error writing hx_factor1_universeb");
	errNum = clEnqueueWriteBuffer(queue, hx_factor2_universeb, CL_TRUE, 0, UWIDTH * sizeof(fp),
			&hx_factor2_universe, 0, NULL, NULL);
	verbose(errNum, "Error writing hx_factor2_universeb");
	errNum = clEnqueueWriteBuffer(queue, geometryb, CL_TRUE, 0, UWIDTH * sizeof(char),
			&geometry, 0, NULL, NULL);
	verbose(errNum, "Error writing geometryb buffer");
	errNum = clEnqueueWriteBuffer(queue, sourceb, CL_TRUE, 0, UWIDTH * sizeof(char),
			&source, 0, NULL, NULL);
	verbose(errNum, "Error writing source buffer");


	// Create the kernel object depending on fp
	if (sizeof(fp) == 4)
	{
		kernel = clCreateKernel(program, "fdtd1d_noiter_fp32", &errNum);
	}
	else
	{
		kernel = clCreateKernel(program, "fdtd1d_noiter_fp64", &errNum);
	}
	verbose(errNum, "Error creating kernel");

	// Set the kernel args
	errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&old_ey_universeb);
	verbose(errNum, "Error setting old_ey_universeb arg");
	errNum = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&old_hx_universeb);
	verbose(errNum, "Error setting old_hx_universeb arg");
	errNum = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&current_ey_universeb);
	verbose(errNum, "Error setting current_ey_universeb arg");
	errNum = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&current_hx_universeb);
	verbose(errNum, "Error setting current_hx_universeb arg");
	errNum = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&ey_factor1_universeb);
	verbose(errNum, "Error setting ey_factor1_universe1b arg");
	errNum = clSetKernelArg(kernel, 5, sizeof(cl_mem), (void*)&ey_factor2_universeb);
	verbose(errNum, "Error setting ey_factor2_universe1b arg");
	errNum = clSetKernelArg(kernel, 6, sizeof(cl_mem), (void*)&hx_factor1_universeb);
	verbose(errNum, "Error setting hx_factor1_universe1b arg");
	errNum = clSetKernelArg(kernel, 7, sizeof(cl_mem), (void*)&hx_factor2_universeb);
	verbose(errNum, "Error setting hx_factor2_universe1b arg");
	errNum = clSetKernelArg(kernel, 8, sizeof(cl_mem), (void*)&geometryb);
	verbose(errNum, "Error setting geometryb arg");
	errNum = clSetKernelArg(kernel, 9, sizeof(fp), (void*)&grid_width_z);
	verbose(errNum, "Error setting grid_width_z");
	errNum = clSetKernelArg(kernel, 10, sizeof(fp), (void*)&max_time);
	verbose(errNum, "Error setting max_time");
	errNum = clSetKernelArg(kernel, 11, sizeof(fp), (void*)&time_step);
	verbose(errNum, "Error setting time_step");
	errNum = clSetKernelArg(kernel, 12, sizeof(int), (void*)&batch_size);
	verbose(errNum, "Error setting batch_size");
	errNum = clSetKernelArg(kernel, 13, sizeof(cl_mem), (void*)&sourceb);
	verbose(errNum, "Error setting sourceb");
	errNum = clSetKernelArg(kernel, 14, sizeof(fp), (void*)&ramped_sin_amplitude);
	verbose(errNum, "Error setting ramped_sin_amplitude");
	errNum = clSetKernelArg(kernel, 15, sizeof(fp), (void*)&ramped_sin_length);
	verbose(errNum, "Error setting ramped_sin_length");
	errNum = clSetKernelArg(kernel, 16, sizeof(fp), (void*)&ramped_sin_frequency);
	verbose(errNum, "Error setting ramped_sin_frequency");


	// Invoke the kernel
	size_t gws[1] = {(UWIDTH - 2) / batch_size};
	size_t lws[1] = {1};
	errNum = clEnqueueNDRangeKernel(queue, kernel, 1, NULL,
			(size_t*)gws, (size_t*)gws, 0, 0, 0);
	verbose(errNum, "Error invoking kernel");

	// Read the result
	errNum = clEnqueueReadBuffer(queue, old_ey_universeb, CL_TRUE, 
			0, UWIDTH * sizeof(fp), current_ey_universe, 0, NULL, NULL);
	verbose(errNum, "Error reading the current_ey_universeb");

	//Print the output
	for (int z = 0; z < UWIDTH; z++)
	{
		printf("%d\t%.32f\n", z, current_ey_universe[z]);
	}
	// Add two empty lines so gnuplot will recognize this as a data block
	return 0;
}
