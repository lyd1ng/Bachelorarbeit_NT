"""
Uses ../fdtd/fdtd_2d_pml/raw_data.txt and ../fdtd/fdtd_2d_pml_c/raw_data.txt
to calculate the average performance factor.
"""

fd1 = open("../fdtd/fdtd_2d_pml/raw_data.txt", "r")
fd2 = open("../fdtd/fdtd_2d_pml_c/raw_data.txt", "r")

counter = 0
performance_gain = 0
performance_gain_sum = 0

for python_performance in fd1.readlines():
    python_performance = float(python_performance)
    c_performance = float(fd2.readline())
    performance_gain_sum += python_performance / c_performance
    counter = counter + 1

performance_gain = performance_gain_sum / counter
print(str(counter) + " samples available, the average performance gain is "\
        + str(performance_gain))
