n=1280000

run: compile
	bin/main $(n)

compile:
	@mkdir -p bin && rm -f bin/*
	g++ -march=native -O3 -fno-tree-vectorize main.cpp -o bin/main

clean:
	@rm -rf bin/

specs:
	g++ -march=native -dM -E - </dev/null | egrep "SSE|AVX|FMA"
