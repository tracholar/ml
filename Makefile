CXX = g++

all:
	$(CXX) -g -std=c++11 src/*.cc -o target/glm
clean:
	rm -rf target/*
