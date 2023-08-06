include ../../common.mk
OBJS := VertexSet.o graph.o
VPATH += ../../common
BIN=../../../bin
INCLUDES = -I../../../include

all: ns_path ns_clique ns_cycle ns_house

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

ns_house: $(OBJS) ns_house.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) ns_house.o -o $@ -lgomp
	mv $@ $(BIN)

clean:
	rm *.o
