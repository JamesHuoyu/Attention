CUDAC = nvcc

CUDAOPT = -O4 --use_fast_math -arch=sm_86

CUDAFLAGS = $(CUDAOPT)

CUDAPATH = $(CUDA_HOME)
LDFLAGS =
LDLIBS = $(LDFLAGS) -L$(CUDAPATH) -L$(CUDAPATH)/stubs -lcuda -lcudart -lcublas

targets =  opt
commonobj = benchmark.o
objects = $(commonobj) opt.o

.PHONY : default
default : all

.PHONY : all
all : clean $(targets)

benchmark.o : benchmark.cu
	$(CUDAC) -c $(CUDAFLAGS) $< -o $@

opt.o : opt.cu
	$(CUDAC) -c $(CUDAFLAGS) $< -o $@
opt : $(commonobj) opt.o
	$(CUDAC) -o $@ $^ $(LDLIBS)

.PHONY: clean
clean:
	rm -rf $(targets) $(objects)