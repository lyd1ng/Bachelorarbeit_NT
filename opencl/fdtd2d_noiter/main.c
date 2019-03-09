#include <math.h>
#include <stdio.h>
#include <alloca.h>
#include <CL/opencl.h>
#include "c_helper.h"
#include "fdtd_helper.h"

#ifndef fp
#define fp float
#endif

#define UWIDTH 130
#define UHEIGHT 130


void verbose(int error, char* msg)
{
	if (error) { printf("%s\n", msg); }
}



void set_pml_geometry(const int uwidth, const int uheight, const fp time_step, fp sigmax[UHEIGHT][UWIDTH], fp sigmay[UHEIGHT][UWIDTH])
{
    int pml_width = 20;
    int pml_height = 20;
    fp f = vacuum_permittivity / (2.0 * time_step);
    for (int y=1; y < uheight - 1; y++)
    {
        for (int x=1; x < pml_width; x++)
	{
            sigmax[y][pml_width - x - 1] = f * pow((fp)x / (fp)pml_width, 3);
            sigmax[y][uwidth - pml_width + x] = f * pow((fp)x / (fp)pml_width, 3);
	}
    }
    for (int x=1; x < uwidth - 1; x++)
    {
        for (int y=1; y < pml_height; y++)
	{
            sigmay[pml_height - y - 1][x] = f * pow((fp)y / (fp)pml_height, 3);
            sigmay[uheight - pml_height + y][x] = f * pow((fp)y / (fp)pml_height, 3);
	}
    }
}

void calculate_factors_universe(const int uwidth,
		       const int uheight,
		       fp time_step,
		       fp permittivity[UHEIGHT][UWIDTH],
		       fp permeability[UHEIGHT][UWIDTH],
		       fp sigmax[UHEIGHT][UWIDTH],
		       fp sigmay[UHEIGHT][UWIDTH],
		       fp ez_factor1_universe[UHEIGHT][UWIDTH],
		       fp ez_factor2_universe[UHEIGHT][UWIDTH],
		       fp ez_factor3_universe[UHEIGHT][UWIDTH],
		       fp hx_factor1_universe[UHEIGHT][UWIDTH],
		       fp hx_factor2_universe[UHEIGHT][UWIDTH],
		       fp hx_factor3_universe[UHEIGHT][UWIDTH],
		       fp hy_factor1_universe[UHEIGHT][UWIDTH],
		       fp hy_factor2_universe[UHEIGHT][UWIDTH],
		       fp hy_factor3_universe[UHEIGHT][UWIDTH])
{
    for (int y=0; y < uheight; y++)
    {
        for (int x=0; x < uwidth; x++)
	{
		fp nominator = 0;
		fp denominator = 0;
            	ez_factor1_universe[y][x] = 1 - ((time_step * (sigmax[y][x] + sigmay[y][x]))
				   / vacuum_permittivity);
            	ez_factor2_universe[y][x] = (sigmax[y][x] * sigmay[y][x] * pow(time_step, 2))
				   / pow(vacuum_permittivity, 2);
            	ez_factor3_universe[y][x] = (light_speed * time_step) / permittivity[y][x];

            	hx_factor1_universe[y][x] = (1 / time_step - (sigmay[y][x]
				   / (2 * vacuum_permittivity))) / (1 / time_step
				   + (sigmay[y][x] / (2 * vacuum_permittivity)));
            	hx_factor2_universe[y][x] = (light_speed / permittivity[y][x])
				   / ((1 / time_step) + (sigmay[y][x]
				   / (2 * vacuum_permittivity)));
            	nominator = (light_speed * sigmax[y][x] * time_step)
			    / (permeability[y][x] * vacuum_permittivity);
            	denominator = 1 / time_step + (sigmay[y][x]
			      / (2 * vacuum_permittivity));
            	hx_factor3_universe[y][x] = nominator / denominator;

            	hy_factor1_universe[y][x] = (1 / time_step - (sigmax[y][x]
				   / (2 * vacuum_permittivity)))
				   / (1 / time_step + (sigmax[y][x]
				   / (2 * vacuum_permittivity)));
            	hy_factor2_universe[y][x] = (light_speed / permittivity[y][x])
				   / ((1 / time_step) + (sigmax[y][x]
				   / (2 * vacuum_permittivity)));
            	nominator = (light_speed * sigmay[y][x] * time_step)
			    / (permeability[y][x] * vacuum_permittivity);
            	denominator = 1 / time_step + (sigmax[y][x]
			      / (2 * vacuum_permittivity));
            	hy_factor3_universe[y][x] = nominator / denominator;
	}
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
	fp old_ez_universe[UHEIGHT][UWIDTH] = {0};
	fp old_hx_universe[UHEIGHT][UWIDTH] = {0};
	fp old_hy_universe[UHEIGHT][UWIDTH] = {0};
	fp current_ez_universe[UHEIGHT][UWIDTH] = {0};
	fp current_hx_universe[UHEIGHT][UWIDTH] = {0};
	fp current_hy_universe[UHEIGHT][UWIDTH] = {0};
	fp ez_integral_universe[UHEIGHT][UWIDTH] = {0};
	fp hx_integral_universe[UHEIGHT][UWIDTH] = {0};
	fp hy_integral_universe[UHEIGHT][UWIDTH] = {0};
	fp ez_factor1_universe[UHEIGHT][UWIDTH] = {0};
	fp ez_factor2_universe[UHEIGHT][UWIDTH] = {0};
	fp ez_factor3_universe[UHEIGHT][UWIDTH] = {0};
	fp hx_factor1_universe[UHEIGHT][UWIDTH] = {0};
	fp hx_factor2_universe[UHEIGHT][UWIDTH] = {0};
	fp hx_factor3_universe[UHEIGHT][UWIDTH] = {0};
	fp hy_factor1_universe[UHEIGHT][UWIDTH] = {0};
	fp hy_factor2_universe[UHEIGHT][UWIDTH] = {0};
	fp hy_factor3_universe[UHEIGHT][UWIDTH] = {0};
	fp sigmax[UHEIGHT][UWIDTH] = {0};
	fp sigmay[UHEIGHT][UWIDTH] = {0};
	fp permittivity[UHEIGHT][UWIDTH] = {0};
	fp permeability[UHEIGHT][UWIDTH] = {0};
	char geometry[UHEIGHT][UWIDTH] = {0};

	// The permeability and permittivity are relativ
	// to the free space values. So 1 is the correct value
	// to simulate the free space.
	for (int y = 0; y < UHEIGHT; y++)
	{
		for (int x = 0; x < UWIDTH; x++)
		{
			permittivity[y][x] = 1;
			permeability[y][x] = 1;
			geometry[y][x] = 1;
		}
	}

	// The simulation parameters
	fp fraction = 1;
	int ramped_sin_length = 100;
	fp ramped_sin_amplitude = 1;
	fp ramped_sin_frequency = 2.4 * pow(10, 9);
	int samplerate_space = 20;
	int samplerate_time = 10;
	fp grid_width_x = get_grid_width(ramped_sin_frequency, fraction, samplerate_space);
	fp grid_width_y = get_grid_width(ramped_sin_frequency, fraction, samplerate_space);
	fp time_step = get_time_resolution_2d(fraction, grid_width_x, grid_width_y)
			  / (fp)samplerate_time;
	fp current_time = 0;
	fp max_time = 100 * time_step;

	// Print the parameters as gnuplot comments
	printf("# dimensions: %dx%d\n", UWIDTH, UHEIGHT);
	printf("# space_sr: %d\n", samplerate_time);
	printf("# time_sr: %d\n", samplerate_space);
	printf("# floating point: %d\n", 8 * sizeof(fp));
	printf("# ramped_sin_frequency: %f\n", ramped_sin_frequency);
	printf("# ramped_sin_amplitude: %f\n", ramped_sin_amplitude);
	printf("# ramped_sin_length: %d\n", ramped_sin_length);
	printf("# max_time: %.32f\n", max_time);
	
	// Set the pml geometry
	set_pml_geometry(UWIDTH, UHEIGHT, time_step, sigmax, sigmay);

	// Calculate factors
	calculate_factors_universe(UWIDTH, UHEIGHT, time_step, permittivity, permeability, sigmax, sigmay,
			ez_factor1_universe, ez_factor2_universe, ez_factor3_universe,
			hx_factor1_universe, hx_factor2_universe, hx_factor3_universe,
			hy_factor1_universe, hy_factor2_universe, hy_factor3_universe);
	

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
	FILE* code_fd = fopen("fdtd2d.cl", "r");
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
	int n = UWIDTH * UHEIGHT;
	cl_mem old_ez_universeb = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
				 n * sizeof(fp), old_ez_universe, &errNum);
	verbose(errNum, "Error creating old_ez_universeb");
	cl_mem old_hx_universeb = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
				 n * sizeof(fp), old_hx_universe, &errNum);
	verbose(errNum, "Error creating old_hx_universeb");
	cl_mem old_hy_universeb = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
				 n * sizeof(fp), old_hy_universe, &errNum);
	verbose(errNum, "Error creating old_hy_universeb");

	// Current Buffers (write only)
	cl_mem current_ez_universeb = clCreateBuffer(context, CL_MEM_READ_WRITE
			| CL_MEM_USE_HOST_PTR, n * sizeof(fp), current_ez_universe,
			&errNum);
	verbose(errNum, "Error creating current_ez_universeb");
	cl_mem current_hx_universeb = clCreateBuffer(context, CL_MEM_READ_WRITE
			| CL_MEM_USE_HOST_PTR, n * sizeof(fp), current_hx_universe,
			&errNum);
	verbose(errNum, "Error creating current_hx_universeb");
	cl_mem current_hy_universeb = clCreateBuffer(context, CL_MEM_READ_WRITE
			| CL_MEM_USE_HOST_PTR, n * sizeof(fp), current_hy_universe,
			&errNum);
	verbose(errNum, "Error creating current_hy_universeb");

	// Integral (read_write)
	cl_mem ez_integral_universeb = clCreateBuffer(context, CL_MEM_READ_WRITE
			| CL_MEM_USE_HOST_PTR, n * sizeof(fp), ez_integral_universe,
			&errNum);
	verbose(errNum, "Error creating ez_integral_universeb");
	cl_mem hx_integral_universeb = clCreateBuffer(context, CL_MEM_READ_WRITE
			| CL_MEM_USE_HOST_PTR, n * sizeof(fp), hx_integral_universe,
			&errNum);
	verbose(errNum, "Error creating hx_integral_universeb");
	cl_mem hy_integral_universeb = clCreateBuffer(context, CL_MEM_READ_WRITE
			| CL_MEM_USE_HOST_PTR, n * sizeof(fp), hy_integral_universe,
			&errNum);
	verbose(errNum, "Error creating hy_integral_universeb");

	// Factors (read only)
	cl_mem ez_factor1_universeb = clCreateBuffer(context, CL_MEM_READ_ONLY
			| CL_MEM_USE_HOST_PTR, n * sizeof(fp), ez_factor1_universe,
			&errNum);
	verbose(errNum, "Error creating ez_factor1_universeb");
	cl_mem ez_factor2_universeb = clCreateBuffer(context, CL_MEM_READ_ONLY
			| CL_MEM_USE_HOST_PTR, n * sizeof(fp), ez_factor2_universe,
			&errNum);
	verbose(errNum, "Error creating ez_factor2_universeb");
	cl_mem ez_factor3_universeb = clCreateBuffer(context, CL_MEM_READ_ONLY
			| CL_MEM_USE_HOST_PTR, n * sizeof(fp), ez_factor3_universe,
			&errNum);
	verbose(errNum, "Error creating ez_factor3_universeb");
	cl_mem hx_factor1_universeb = clCreateBuffer(context, CL_MEM_READ_ONLY
			| CL_MEM_USE_HOST_PTR, n * sizeof(fp), hx_factor1_universe,
			&errNum);
	verbose(errNum, "Error creating hx_factor1_universeb");
	cl_mem hx_factor2_universeb = clCreateBuffer(context, CL_MEM_READ_ONLY
			| CL_MEM_USE_HOST_PTR, n * sizeof(fp), hx_factor2_universe,
			&errNum);
	verbose(errNum, "Error creating hx_factor2_universeb");
	cl_mem hx_factor3_universeb = clCreateBuffer(context, CL_MEM_READ_ONLY
			| CL_MEM_USE_HOST_PTR, n * sizeof(fp), hx_factor3_universe,
			&errNum);
	verbose(errNum, "Error creating hx_factor3_universeb");
	cl_mem hy_factor1_universeb = clCreateBuffer(context, CL_MEM_READ_ONLY
			| CL_MEM_USE_HOST_PTR, n * sizeof(fp), hy_factor1_universe,
			&errNum);
	verbose(errNum, "Error creating hy_factor1_universeb");
	cl_mem hy_factor2_universeb = clCreateBuffer(context, CL_MEM_READ_ONLY
			| CL_MEM_USE_HOST_PTR, n * sizeof(fp), hy_factor2_universe,
			&errNum);
	verbose(errNum, "Error creating hy_factor2_universeb");
	cl_mem hy_factor3_universeb = clCreateBuffer(context, CL_MEM_READ_ONLY
			| CL_MEM_USE_HOST_PTR, n * sizeof(fp), hy_factor3_universe,
			&errNum);
	verbose(errNum, "Error creating hy_factor3_universeb");
	cl_mem sigmaxb = clCreateBuffer(context, CL_MEM_READ_ONLY
			| CL_MEM_USE_HOST_PTR, n * sizeof(fp), sigmax,
			&errNum);
	verbose(errNum, "Error creating sigmaxb");
	cl_mem sigmayb = clCreateBuffer(context, CL_MEM_READ_ONLY
			| CL_MEM_USE_HOST_PTR, n * sizeof(fp), sigmay,
			&errNum);
	verbose(errNum, "Error creating sigmayb");
	cl_mem geometryb = clCreateBuffer(context, CL_MEM_READ_ONLY
			| CL_MEM_USE_HOST_PTR, n * sizeof(char), &geometry,
			&errNum);
	verbose(errNum, "Error creating geometryb");

	// Pass the arrays to the device
	errNum = clEnqueueWriteBuffer(queue, old_ez_universeb, CL_TRUE, 0, UWIDTH * UHEIGHT * sizeof(fp),
			old_ez_universe, 0, NULL, NULL);
	verbose(errNum, "Error writing old_ez_universeb buffer");
	errNum = clEnqueueWriteBuffer(queue, old_hx_universeb, CL_TRUE, 0, UWIDTH * UHEIGHT * sizeof(fp),
			old_hx_universe, 0, NULL, NULL);
	verbose(errNum, "Error writing old_hx_universeb buffer");
	errNum = clEnqueueWriteBuffer(queue, old_hy_universeb, CL_TRUE, 0, UWIDTH * UHEIGHT * sizeof(fp),
			old_hy_universe, 0, NULL, NULL);
	verbose(errNum, "Error writing old_hy_universeb buffer");
	errNum = clEnqueueWriteBuffer(queue, current_ez_universeb, CL_TRUE, 0, UWIDTH * UHEIGHT * sizeof(fp),
			current_ez_universe, 0, NULL, NULL);
	verbose(errNum, "Error writing current_ez_universeb buffer");
	errNum = clEnqueueWriteBuffer(queue, current_hx_universeb, CL_TRUE, 0, UWIDTH * UHEIGHT * sizeof(fp),
			current_hx_universe, 0, NULL, NULL);
	verbose(errNum, "Error writing current_hx_universeb buffer");
	errNum = clEnqueueWriteBuffer(queue, current_hy_universeb, CL_TRUE, 0, UWIDTH * UHEIGHT * sizeof(fp),
			current_hy_universe, 0, NULL, NULL);
	verbose(errNum, "Error writing current_hy_universeb buffer");

	errNum = clEnqueueWriteBuffer(queue, ez_integral_universeb, CL_TRUE, 0, UWIDTH * UHEIGHT * sizeof(fp),
			ez_integral_universe, 0, NULL, NULL);
	verbose(errNum, "Error writing ez_integral_universeb buffer");
	errNum = clEnqueueWriteBuffer(queue, hx_integral_universeb, CL_TRUE, 0, UWIDTH * UHEIGHT * sizeof(fp),
			hx_integral_universe, 0, NULL, NULL);
	verbose(errNum, "Error writing hx_integral_universeb buffer");
	errNum = clEnqueueWriteBuffer(queue, hy_integral_universeb, CL_TRUE, 0, UWIDTH * UHEIGHT * sizeof(fp),
			hy_integral_universe, 0, NULL, NULL);
	verbose(errNum, "Error writing hy_integral_universeb buffer");


	errNum = clEnqueueWriteBuffer(queue, ez_factor1_universeb, CL_TRUE, 0, UWIDTH * UHEIGHT * sizeof(fp),
			ez_factor1_universe, 0, NULL, NULL);
	verbose(errNum, "Error writing ez_factor1_universeb");
	errNum = clEnqueueWriteBuffer(queue, ez_factor2_universeb, CL_TRUE, 0, UWIDTH * UHEIGHT * sizeof(fp),
			ez_factor2_universe, 0, NULL, NULL);
	verbose(errNum, "Error writing ez_factor2_universeb");
	errNum = clEnqueueWriteBuffer(queue, ez_factor3_universeb, CL_TRUE, 0, UWIDTH * UHEIGHT * sizeof(fp),
			ez_factor3_universe, 0, NULL, NULL);
	verbose(errNum, "Error writing ez_factor3_universeb");
	errNum = clEnqueueWriteBuffer(queue, hx_factor1_universeb, CL_TRUE, 0, UWIDTH * UHEIGHT * sizeof(fp),
			hx_factor1_universe, 0, NULL, NULL);
	verbose(errNum, "Error writing hx_factor1_universeb");
	errNum = clEnqueueWriteBuffer(queue, hx_factor2_universeb, CL_TRUE, 0, UWIDTH * UHEIGHT * sizeof(fp),
			hx_factor2_universe, 0, NULL, NULL);
	verbose(errNum, "Error writing hx_factor2_universeb");
	errNum = clEnqueueWriteBuffer(queue, hx_factor3_universeb, CL_TRUE, 0, UWIDTH * UHEIGHT * sizeof(fp),
			hx_factor3_universe, 0, NULL, NULL);
	verbose(errNum, "Error writing hx_factor3_universeb");
	errNum = clEnqueueWriteBuffer(queue, hy_factor1_universeb, CL_TRUE, 0, UWIDTH * UHEIGHT * sizeof(fp),
			hy_factor1_universe, 0, NULL, NULL);
	verbose(errNum, "Error writing hy_factor1_universeb");
	errNum = clEnqueueWriteBuffer(queue, hy_factor2_universeb, CL_TRUE, 0, UWIDTH * UHEIGHT * sizeof(fp),
			hy_factor2_universe, 0, NULL, NULL);
	verbose(errNum, "Error writing hy_factor2_universeb");
	errNum = clEnqueueWriteBuffer(queue, hy_factor3_universeb, CL_TRUE, 0, UWIDTH * UHEIGHT * sizeof(fp),
			hy_factor3_universe, 0, NULL, NULL);
	verbose(errNum, "Error writing hy_factor3_universeb");
	errNum = clEnqueueWriteBuffer(queue, geometryb, CL_TRUE, 0, UWIDTH * UHEIGHT* sizeof(char),
			geometry, 0, NULL, NULL);
	verbose(errNum, "Error writing geometryb buffer");


	// Create the kernel object depending on fp
	if (sizeof(fp) == 4)
	{
		kernel = clCreateKernel(program, "fdtd2d_noiter_fp32", &errNum);
	}
	else
	{
		kernel = clCreateKernel(program, "fdtd2d_noiter_fp64", &errNum);
	}
	verbose(errNum, "Error creating kernel");
	

	// Set the kernel args
	errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&old_ez_universeb);
	verbose(errNum, "Error setting old_ez_universeb arg");
	errNum = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&old_hx_universeb);
	verbose(errNum, "Error setting old_hx_universeb arg");
	errNum = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&old_hy_universeb);
	verbose(errNum, "Error setting old_hy_universeb arg");
	errNum = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&current_ez_universeb);
	verbose(errNum, "Error setting current_ez_universeb arg");
	errNum = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&current_hx_universeb);
	verbose(errNum, "Error setting current_hx_universeb arg");
	errNum = clSetKernelArg(kernel, 5, sizeof(cl_mem), (void*)&current_hy_universeb);
	verbose(errNum, "Error setting current_hy_universeb arg");
	errNum = clSetKernelArg(kernel, 6, sizeof(cl_mem), (void*)&ez_integral_universeb);
	verbose(errNum, "Error setting ez_integral_universeb arg");
	errNum = clSetKernelArg(kernel, 7, sizeof(cl_mem), (void*)&hx_integral_universeb);
	verbose(errNum, "Error setting hx_integral_universeb arg");
	errNum = clSetKernelArg(kernel, 8, sizeof(cl_mem), (void*)&hy_integral_universeb);
	verbose(errNum, "Error setting hy_integral_universeb arg");
	errNum = clSetKernelArg(kernel, 9, sizeof(cl_mem), (void*)&ez_factor1_universeb);
	verbose(errNum, "Error setting ez_factor1_universeb arg");
	errNum = clSetKernelArg(kernel, 10, sizeof(cl_mem), (void*)&ez_factor2_universeb);
	verbose(errNum, "Error setting ez_factor2_universeb arg");
	errNum = clSetKernelArg(kernel, 11, sizeof(cl_mem), (void*)&ez_factor3_universeb);
	verbose(errNum, "Error setting ez_factor3_universeb arg");
	errNum = clSetKernelArg(kernel, 12, sizeof(cl_mem), (void*)&hx_factor1_universeb);
	verbose(errNum, "Error setting hx_factor1_universeb arg");
	errNum = clSetKernelArg(kernel, 13, sizeof(cl_mem), (void*)&hx_factor2_universeb);
	verbose(errNum, "Error setting hx_factor2_universeb arg");
	errNum = clSetKernelArg(kernel, 14, sizeof(cl_mem), (void*)&hx_factor3_universeb);
	verbose(errNum, "Error setting hx_factor3_universeb arg");
	errNum = clSetKernelArg(kernel, 15, sizeof(cl_mem), (void*)&hy_factor1_universeb);
	verbose(errNum, "Error setting hy_factor1_universeb arg");
	errNum = clSetKernelArg(kernel, 16, sizeof(cl_mem), (void*)&hy_factor2_universeb);
	verbose(errNum, "Error setting hy_factor2_universeb arg");
	errNum = clSetKernelArg(kernel, 17, sizeof(cl_mem), (void*)&hy_factor3_universeb);
	verbose(errNum, "Error setting hy_factor3_universeb arg");
	errNum = clSetKernelArg(kernel, 18, sizeof(cl_mem), (void*)&geometryb);
	verbose(errNum, "Error setting geometryb arg");
	errNum = clSetKernelArg(kernel, 19, sizeof(fp), (void*)&grid_width_x);
	verbose(errNum, "Error setting grid_width_x");
	errNum = clSetKernelArg(kernel, 20, sizeof(fp), (void*)&grid_width_y);
	verbose(errNum, "Error setting grid_width_y");
	int uwidth = UWIDTH;
	errNum = clSetKernelArg(kernel, 21, sizeof(int), (void*)&uwidth);
	verbose(errNum, "Error setting uwidth");

	// Invoke the kernel
	size_t gws[2] = {UWIDTH - 2, UHEIGHT - 2};
	while (current_time < max_time)
	{
		errNum = clEnqueueNDRangeKernel(queue, kernel, 2, NULL,
				gws, NULL, 0, 0, 0);
		verbose(errNum, "Error invoking the kernel");
		if (errNum)
		{
			printf("%d\n", errNum);
			exit(0);
		}

		// The ez field has to be copied manually to inject the source
		errNum = clEnqueueReadBuffer(queue, current_ez_universeb, CL_TRUE,
				0, UWIDTH * UHEIGHT * sizeof(fp), old_ez_universe, 0, NULL, NULL);
		verbose(errNum, "Error reading the current_ey_universeb");

		// Inject the source
		old_ez_universe[UWIDTH / 2][UHEIGHT / 2] = ramped_sinus(current_time, ramped_sin_length * time_step,
				ramped_sin_amplitude, ramped_sin_frequency);

		// Now the field can be written to the device again
		errNum = clEnqueueWriteBuffer(queue, old_ez_universeb, CL_TRUE,
				0, UWIDTH * UHEIGHT * sizeof(fp), old_ez_universe, 0, NULL, NULL);
		verbose(errNum, "Error Writing old_ez_universe");

		errNum = clEnqueueCopyBuffer(queue, current_hx_universeb, old_hx_universeb, 0, 0,
				UWIDTH * UHEIGHT* sizeof(fp), 0, NULL, NULL);
		verbose(errNum, "Error copying buffer current_hx_universeb");
		errNum = clEnqueueCopyBuffer(queue, current_hy_universeb, old_hy_universeb, 0, 0,
				UWIDTH * UHEIGHT* sizeof(fp), 0, NULL, NULL);
		verbose(errNum, "Error copying buffer current_hy_universeb");
		clFinish(queue);
		current_time += time_step;
	}
	
	errNum = clEnqueueReadBuffer(queue, current_ez_universeb, CL_TRUE, 0, UWIDTH * UHEIGHT* sizeof(fp),
		(void*)(current_ez_universe), 0, NULL, NULL);
	
	// Print the results in an ugly list
	for (int y = 0; y < UHEIGHT; y++)
	{
		for (int x = 0; x < UWIDTH; x++)
		{
			printf("%d\t%d\t%.32f\n", x, y, current_ez_universe[y][x]);
		}
	}
	return 0;
}
