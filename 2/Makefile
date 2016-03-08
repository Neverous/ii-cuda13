BUILDDIR=build

debug: clean
	cd ${BUILDDIR} && \
	cmake -DCMAKE_BUILD_TYPE=Debug .. && \
	make VERBOSE=1;

release: clean
	cd ${BUILDDIR} && \
	cmake -DCMAKE_BUILD_TYPE=Release .. && \
	make VERBOSE=1 && \
	ln -s src/fraktal ./;

clean:
	-rm -rf ${BUILDDIR};
	-rm *.ppm;
	mkdir -p ${BUILDDIR};
