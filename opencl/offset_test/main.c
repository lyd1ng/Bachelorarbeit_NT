#include <stdio.h>
#include <CL/opencl.h>
#include "c_helper.h"

#define N 1024

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
	FILE* code_fd = fopen("vector_add.cl", "r");
	char *code;
	size_t code_length;
	read_fileh(code_fd, &code, &code_length);
	program = clCreateProgramWithSource(context, 1,
			(const char**)&code, &code_length, &errNum);
	verbose(errNum, "Error creating the program");
	verbose(clBuildProgram(program, 1, &device, NULL, NULL, NULL),
		"Error building the program");

	// Creathe the kernel object
	kernel = clCreateKernel(program, "vector_add", &errNum);
	verbose(errNum, "Error creating kernel");

	// Create the arrays and buffers
	float arrayA[N] = { 0 };

	cl_mem bufferA = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
			N * sizeof(float), arrayA, &errNum);
	verbose(errNum, "Error creating bufferA");


	// Set the kernel args
	errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&bufferA);
	verbose(errNum, "Error setting bufferA argumment");

	// Invoke the kernel
	size_t global_work_size = N;
	size_t offset = 10;
	errNum = clEnqueueNDRangeKernel(queue, kernel, 1, (size_t*)&offset,
		(size_t*)&global_work_size, NULL, 0, 0, 0);
	verbose(errNum, "Error invoking the kernel");
	
	// Read the result
	errNum = clEnqueueReadBuffer(queue, bufferA, CL_TRUE, 0, N * sizeof(float),
		(void*)arrayA, 0, NULL, NULL);
	
	for (int i = 0; i < N; i++)
	{
		printf("%f\n", arrayA[i]);
	}
	
	return 0;
}
