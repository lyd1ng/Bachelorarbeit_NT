CC=gcc
CFLAGS=-Wno-deprecated-declarations -D"fp=double"
LDFLAGS=
LDLIBS=-lm -lOpenCL
OBJ=c_helper.o fdtd_helper.o main.o  

main: $(OBJ)
	$(CC) -o main $(LDFLAGS) $(OBJ) $(LDLIBS)

%.o: %.c
	$(CC) -c $(CFLAGS) -o $@ $<


clean:
	rm -f main $(OBJ)

.PHONY:
	clean
