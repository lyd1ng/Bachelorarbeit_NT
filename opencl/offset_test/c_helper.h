#include <stdio.h>
#include <stdlib.h>

// Get the size of a file
long int get_file_size(FILE* f);


// Allocates a char array on the heap 
// to store the content of fd
int read_fileh(FILE* fd, char** out, size_t* length);
