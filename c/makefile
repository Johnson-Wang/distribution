##Basic Makefile for spglib
.PHONY: default
default:build move clean
PYTHON=python
PYTHON_INCLUDE = ${shell ${PYTHON} -c 'from distutils import sysconfig; print( sysconfig.get_python_inc() )'}
NP_INCLUDE = ${shell ${PYTHON} -c 'import numpy; print (numpy.get_include())'}
CC = gcc 
SRC = _kappa_thin_film.c 
OBJ = $(patsubst %.c, %.o, $(SRC))
LIBS = -lm
CFLAGS= -pthread -fPIC -fopenmp
CFLAGS_SHARED= -pthread -shared -fPIC -lgomp
%.o: %.c
	$(CC) -c -o $@  $< $(CFLAGS) -I$(NP_INCLUDE) -I$(PYTHON_INCLUDE) 
.PHONY: build
build: _kthin.so
_kthin.so: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS_SHARED) $(LIBS) -I$(NP_INCLUDE)
.PHONY: move
move:
	mv _kthin.so ../kappa/
.PHONY: clean
clean:
	rm -f $(OBJ)
