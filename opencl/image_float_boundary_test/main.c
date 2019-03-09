#include <stdio.h>
#include <CL/opencl.h>
#include "c_helper.h"

void verbose(int error, char* msg)
{
	if (error) { printf("%s\n", msg); }
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
	FILE* code_fd = fopen("image_test.cl", "r");
	char *code;
	size_t code_length;
	read_fileh(code_fd, &code, &code_length);
	program = clCreateProgramWithSource(context, 1,
			(const char**)&code, &code_length, &errNum);
	verbose(errNum, "Error creating the program");
	errNum = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
	verbose(errNum, "Error building the program");

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



	// Creathe the kernel object
	kernel = clCreateKernel(program, "times2_test", &errNum);
	verbose(errNum, "Error creating kernel");

	// Create the arrays and buffers
	int NX = 128;
	int NY = 128;
	float in_pixel_data[NY][NX * 4];
	float out_pixel_data[NY][NX * 4];
	size_t origin[3] = {0, 0, 0};
	size_t region[3] = {NX, NY, 1};

	// Write some test data in in_pixel_data
	for (int y = 0; y < NY; y++)
	{
		for (int x = 0; x < NX * 4; x+=4)
		{
			in_pixel_data[y][x + 0] = x + 0 + y * NX;
			in_pixel_data[y][x + 1] = x + 1 + y * NX;
			in_pixel_data[y][x + 2] = x + 2 + y * NX;
			in_pixel_data[y][x + 3] = x + 3 + y * NX;
		}
	}

	cl_image_format image_format;
	image_format.image_channel_order = CL_RGBA;
	image_format.image_channel_data_type = CL_FLOAT;
	cl_mem in_imageb = clCreateImage2D(context, CL_MEM_READ_ONLY |
			CL_MEM_ALLOC_HOST_PTR, &image_format, NX, NY, 0, NULL, &errNum);
	verbose(errNum, "Error creating the in image buffer");

	cl_mem out_imageb = clCreateImage2D(context, CL_MEM_WRITE_ONLY |
			CL_MEM_ALLOC_HOST_PTR, &image_format, NX, NY, 0, NULL, &errNum);
	verbose(errNum, "Error creating the out image buffer");

	// Send the testdata to the device
	errNum = clEnqueueWriteImage(queue, in_imageb, CL_TRUE, origin, region, 0, 0,
			in_pixel_data, 0, NULL, NULL);
	verbose(errNum, "Error sending the test data");

	// Set the kernel args
	errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), &in_imageb);
	verbose(errNum, "Error setting the in imageb kernel arg");
	errNum = clSetKernelArg(kernel, 1, sizeof(cl_mem), &out_imageb);
	verbose(errNum, "Error setting the out imageb kernel arg");
	errNum = clSetKernelArg(kernel, 2, sizeof(int), &NX);
	verbose(errNum, "Error setting the width kernel arg");
	errNum = clSetKernelArg(kernel, 3, sizeof(int), &NY);
	verbose(errNum, "Error setting the height kernel arg");
	// Invoke the kernel
	size_t global_work_size[2] = {NX, NY};
	errNum = clEnqueueNDRangeKernel(queue, kernel, 2, NULL,
		(size_t*)&global_work_size, NULL, 0, 0, 0);
	verbose(errNum, "Error invoking the kernel");
	
	// Read and print the result
	errNum = clEnqueueReadImage(queue, out_imageb, CL_TRUE, origin, region, 0, 0,
			out_pixel_data, 0, NULL, NULL);
	verbose(errNum, "Error reading the image");
	for (int y = 0; y < NY; y++)
	{
		for (int x = 0; x < NX * 4; x+=4)
		{
			printf("%f\n", out_pixel_data[y][x + 0]);
			printf("%f\n", out_pixel_data[y][x + 1]);
			printf("%f\n", out_pixel_data[y][x + 2]);
			printf("%f\n", out_pixel_data[y][x + 3]);
		}
	}
	return 0;
}
