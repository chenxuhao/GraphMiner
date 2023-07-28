include ../../common.mk
VPATH += ../../common
all: approx_profile
BIN=../../../bin
INCLUDES = -I../../../include


%.o: %.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $<

%.o: %.cu
	$(NVCC) $(NVFLAGS) $(INCLUDES) -c $<

approx_profile: $(OBJS) 
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) -o $@ -lgomp
	mv $@ $(BIN)