BUILDDIR=build

debug: clean
	cd ${BUILDDIR} && \
	cmake -DCMAKE_BUILD_TYPE=Debug .. && \
	make VERBOSE=1;

release: clean
	cd ${BUILDDIR} && \
	cmake -DCMAKE_BUILD_TYPE=Release .. && \
	make VERBOSE=1;

clean:
	-rm -rf ${BUILDDIR};
	mkdir -p ${BUILDDIR};
