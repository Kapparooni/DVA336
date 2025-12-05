h=20
t=50

run: compile
	bin/main $(h) $(t)
	
compile:
	@mkdir -p -m 700 bin && rm -rf bin/main
	g++ -O2 -Wall -fopenmp balancetree.cpp -o bin/main

edit:
	@code -n *.cpp *makefile

clean:
	@rm -rf bin
	@chmod -x *.?pp *makefile