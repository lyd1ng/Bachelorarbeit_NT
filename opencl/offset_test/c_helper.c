#include <stdio.h>
#include <stdlib.h>

// Get the size of a file
long int get_file_size(FILE* f)
{
	fseek(f, 0, SEEK_END);
	long size = ftell(f);
	rewind(f);
	return size;
}


// Allocates a char array on the heap 
// to store the content of fd
int read_fileh(FILE* fd, char** out, size_t *length)
{
	*length = get_file_size(fd);
	*out = (char*)malloc(*length + 1);
	if (*out == NULL) { return -1; }
	fread(*out, *length, 1, fd);
	(*out)[*length] = 0x00;
	return 0;
}

