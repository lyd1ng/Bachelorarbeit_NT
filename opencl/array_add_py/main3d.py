import numpy as np
import pyopencl as cl
from timeit import timeit

def kernel_wrapper(kernel, queue, args):
    def run_kernel():
        kernel(*args)
        queue.finish()
    return run_kernel


if __name__ == "__main__":
    platform = cl.get_platforms()[0]
    devices = platform.get_devices()
    context = cl.Context(devices=devices)
    queue = cl.CommandQueue(context)
    kernel_code = open("array_add.cl", "r").read()
    program = cl.Program(context, kernel_code).build()

    NX = 200
    NY = 200
    NZ = 200
    array_a = np.ones((NX, NY, NZ), dtype=np.float32) * 4
    array_b = np.ones((NX, NY, NZ), dtype=np.float32) * 4
    array_c = np.zeros((NX, NY, NZ), dtype=np.float32)
    buffer_size = array_a.nbytes
    print("Array initialisation passed")


    buffer_a = cl.Buffer(context, flags=cl.mem_flags.READ_ONLY, size=buffer_size)
    buffer_b = cl.Buffer(context, flags=cl.mem_flags.READ_ONLY, size=buffer_size)
    buffer_c = cl.Buffer(context, flags=cl.mem_flags.WRITE_ONLY, size=buffer_size)
    print("Buffer initialisation passed")
    cl.enqueue_copy(queue, src=array_a, dest=buffer_a)
    print("Buffer A copy passed")
    cl.enqueue_copy(queue, src=array_b, dest=buffer_b)
    print("Buffer B copy passed")
    cl.enqueue_copy(queue, src=array_c, dest=buffer_c)
    print("Buffer C copy passed")

    time = timeit(kernel_wrapper(program.array3d_add, queue, [queue, (NX, NY, NZ), None, buffer_a, buffer_b, buffer_c]), number=1)
    cl.enqueue_copy(queue, src=buffer_c, dest=array_c)
    cl.enqueue_copy(queue, src=array_c, dest=buffer_a)


    print("Kernel invokation passed")
    print(time)

    
    print("Result copy passed")
    # queue.finish()
    print(array_c)

