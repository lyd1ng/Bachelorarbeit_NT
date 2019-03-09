CC=gcc
CFLAGS=-Wno-deprecated-declarations
LDFLAGS=
LDLIBS=-lOpenCL
OBJ=main.o c_helper.o

main: $(OBJ)
	$(CC) -o main $(LDFLAGS) $(OBJ) $(LDLIBS)

%.o: %.c
	$(CC) -c $(CFLAGS) -o $@ $<


clean:
	rm -f main $(OBJ)

.PHONY:
	clean
