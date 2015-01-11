EIGENROOT := $(HOME)/Main/Tools/eigen/
CXX := g++-4.9
CXXFLAGS := -std=c++0x

test: test.cpp IPCA.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@ -I$(EIGENROOT)
