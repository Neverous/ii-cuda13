ARCH=sm_13
normal: cudaGLIFS.cu
	nvcc -O3 -DNDEBUG -use_fast_math -arch=${ARCH} -Xcompiler "-O3 -march=native -Wall" cudaGLIFS.cu -o normal -lGL -lglut -lGLEW

curand: cudaGLCurandIFS.cu
	nvcc -O3 -DNDEBUG -use_fast_math -arch=${ARCH} -Xcompiler "-O3 -march=native -Wall" cudaGLCurandIFS.cu -o curand -lGL -lglut -lGLEW

clean:
	-rm -f curand normal

.PHONY: clean
