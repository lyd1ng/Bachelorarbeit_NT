#include <stdio.h>
#include <CL/opencl.h>
#include "c_helper.h"

// Es mus float4 verwendet werden auch wenn im Kernel
// float3 verwendet wird...
typedef struct __attribute__((__packed__)) float4 {
	float x;
	float y;
	float z;
	float w;
} float4;

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
	int N = 8;
	float4 vector[N];
	float scalar[N];
	for (int i=0; i < N; i++)
	{
		scalar[i] = 0;
	}

	for (int i=0; i < N; i++)
	{
		vector[i].x = i;
		vector[i].y = i * 2;
		vector[i].z = i * 3;
		vector[i].w = i * 4;
	}


	// Create the buffers
	cl_mem vectorb = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
			N * sizeof(float4), vector, &errNum);
	verbose(errNum, "Error creating vectorb");
	cl_mem scalarb = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
			N * sizeof(float), scalar, &errNum);
	verbose(errNum, "Error creating bufferB");

	// Pass the data to the gpu
	errNum = clEnqueueWriteBuffer(queue, vectorb, CL_TRUE, 0, N * sizeof(float4), vector, 0, NULL, NULL);
	verbose(errNum, "Error writing vectorb");



	// Set the kernel args
	errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&vectorb);
	verbose(errNum, "Error setting vectorb argumment");
	errNum = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&scalarb);
	verbose(errNum, "Error setting scalarb argumment");

	// Invoke the kernel
	size_t global_work_size = N;
	errNum = clEnqueueNDRangeKernel(queue, kernel, 1, NULL,
		(size_t*)&global_work_size, NULL, 0, 0, 0);
	verbose(errNum, "Error invoking the kernel");
	
	// Read the result
	errNum = clEnqueueReadBuffer(queue, scalarb, CL_TRUE, 0, N * sizeof(float), (void*)scalar, 0, NULL, NULL);
	
	for (int i = 0; i < N; i++)
	{
		printf("%f\n", scalar[i]);
	}
	
	return 0;
}
