include ../../common.mk
OBJS := VertexSet.o graph.o
VPATH += ../../common
BIN=../../../bin
INCLUDES = -I../../../include

all: ns_path ns_clique

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $<

ns_path: $(OBJS) ns_path.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) ns_path.o -o $@ -lgomp
	mv $@ $(BIN)

ns_clique: $(OBJS) ns_clique.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) ns_clique.o -o $@ -lgomp
	mv $@ $(BIN)

ns_cycle: $(OBJS) ns_cycle.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) ns_cycle.o -o $@ -lgomp
	mv $@ $(BIN)

clean:
	rm *.o
