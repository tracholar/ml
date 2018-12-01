CXX = g++-6

all:
	$(CXX) -g -std=c++11 -fopenmp src/*.cc -o target/glm  
clean:
	rm -rf target/*
