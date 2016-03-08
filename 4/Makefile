ARCH=sm_13
all: Levenshtein.cu
	nvcc -O3 -DNDEBUG -use_fast_math -arch=${ARCH} -Xcompiler "-O3 -march=native" Levenshtein.cu -o Levenshtein

v1: Levenshtein_v1.cu
	nvcc -O3 -DNDEBUG -use_fast_math -arch=${ARCH} -Xcompiler "-O3 -march=native" Levenshtein_v1.cu -o Levenshtein

v2: Levenshtein_v2.cu
	nvcc -O3 -DNDEBUG -use_fast_math -arch=${ARCH} -Xcompiler "-O3 -march=native" Levenshtein_v2.cu -o Levenshtein

v3: Levenshtein_v3.cu
	nvcc -O3 -DNDEBUG -use_fast_math -arch=${ARCH} -Xcompiler "-O3 -march=native" Levenshtein_v3.cu -o Levenshtein

debug: Levenshtein.cu
	nvcc -O0 -G -g -arch=${ARCH} -Xcompiler "-O0 -g -march=native" Levenshtein.cu -o Levenshtein

clean:
	-rm -f *.out Levenshtein
