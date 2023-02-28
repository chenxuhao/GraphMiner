include ../common.mk
INCLUDES+=-I./ -I./dfscode
all: fsm_omp_base fsm_gpu_base

fsm_gpu_base: gpu_base.o $(OBJS)
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(OBJS) gpu_base.o -o $@ $(LIBS)
	mv $@ $(BIN)

fsm_omp_base: omp_base.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) omp_base.o -o $@ -lgomp
	mv $@ $(BIN)

fsm_omp_lb: omp_lb.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) omp_lb.o -o $@ -lgomp
	mv $@ $(BIN)

clean:
	rm *.o
