n=1600000
d=3

run: compile
	bin/main $(n)

compile:
	@mkdir -p bin && rm -f bin/*
	g++ -march=native -O3 -fno-tree-vectorize main.cpp -D D=$(d) -o bin/main

clean:
	@rm -rf dir/

specs:
	g++ -march=native -dM -E - </dev/null | egrep "SSE|AVX|FMA"
