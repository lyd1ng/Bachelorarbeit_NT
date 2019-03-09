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
	FILE* code_fd = fopen("buffer_test.cl", "r");
	char *code;
	size_t code_length;
	read_fileh(code_fd, &code, &code_length);
	program = clCreateProgramWithSource(context, 1,
			(const char**)&code, &code_length, &errNum);
	verbose(errNum, "Error creating the program");
	verbose(clBuildProgram(program, 1, &device, NULL, NULL, NULL),
		"Error building the program");

	// Creathe the kernel object
	kernel = clCreateKernel(program, "buffer_test_4_2d", &errNum);
	verbose(errNum, "Error creating kernel");

	// Create the arrays and buffers
	int NX = 32;
	int NY = 32;
	int N = 1024;
	float arrayA[NX][NY];
	for (int y = 0; y < NY; y++)
	{
		for (int x = 0; x < NX; x++)
		{
			arrayA[y][x] = 1;
		}
	}
	
	cl_mem bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY
			| CL_MEM_ALLOC_HOST_PTR, N * sizeof(float), NULL, &errNum);
	verbose(errNum, "Error creating bufferA");

	errNum = clEnqueueWriteBuffer(queue, bufferA, CL_TRUE, 0, N * sizeof(float),
			arrayA, 0, 0, 0);



	// Set the kernel args
	errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&bufferA);
	verbose(errNum, "Error setting bufferA argumment");
	errNum = clSetKernelArg(kernel, 1, sizeof(int), (void*)&NX);
	verbose(errNum, "Error setting widthargumment");


	// Invoke the kernel
	size_t global_work_size[2] = {NX - 2, NY - 2};
	errNum = clEnqueueNDRangeKernel(queue, kernel, 2, NULL,
		(size_t*)&global_work_size, NULL, 0, 0, 0);
	verbose(errNum, "Error invoking the kernel");
	
	// Read the result
	errNum = clEnqueueReadBuffer(queue, bufferA, CL_TRUE, 0, N * sizeof(float),
		(void*)(arrayA), 0, NULL, NULL);
	
	for (int y = 0; y < NY; y++)
	{
		for (int x = 0; x < NX; x++)
		{
			printf("%.1f ", arrayA[y][x]);
		}
		printf("\n");
	}
	
	return 0;
}
