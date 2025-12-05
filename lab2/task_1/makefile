n=200000000

run: compile
	bin/main $(n)

compile:
	@mkdir -p bin && rm -f bin/*
	g++ -march=native -O3 -fno-tree-vectorize -fopenmp riemannsumpi.cpp -o bin/main

clean:
	@rm -rf bin
