import numpy as np
import pyopencl as cl

if __name__ == "__main__":
    platform = cl.get_platforms()[0]
    devices = platform.get_devices()
    context = cl.Context(devices=devices)
    queue = cl.CommandQueue(context)
    kernel_code = open("array_add.cl", "r").read()
    program = cl.Program(context, kernel_code).build()

    N = 1024
    array_a = np.ones(N, dtype=np.float32) * 4
    array_b = np.ones(N, dtype=np.float32) * 4
    array_c = np.zeros(N, dtype=np.float32)
    buffer_size = N * 4


    buffer_a = cl.Buffer(context, flags=cl.mem_flags.READ_ONLY, size=buffer_size)
    buffer_b = cl.Buffer(context, flags=cl.mem_flags.READ_ONLY, size=buffer_size)
    buffer_c = cl.Buffer(context, flags=cl.mem_flags.WRITE_ONLY, size=buffer_size)
    cl.enqueue_copy(queue, src=array_a, dest=buffer_a)
    cl.enqueue_copy(queue, src=array_b, dest=buffer_b)
    cl.enqueue_copy(queue, src=array_c, dest=buffer_c)

    program.array1d_add(queue, (N,), None, buffer_a, buffer_b, buffer_c)

    cl.enqueue_copy(queue, src=buffer_c, dest=array_c)
    queue.finish()
    print(array_c)

