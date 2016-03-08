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

run:
	cd data && \
	../${BUILDDIR}/trafficsim

wroclaw:
	./${BUILDDIR}/trafficsim-import ./data/wroclaw.osm.pbf ./data/map.dat

poland:
	./${BUILDDIR}/trafficsim-import ./data/poland.osm.pbf ./data/map.dat

tokyo:
	./${BUILDDIR}/trafficsim-import ./data/tokyo.osm.pbf ./data/map.dat

planet:
	./${BUILDDIR}/trafficsim-import ./data/planet.osm.pbf ./data/map.dat

clean:
	-rm -rf ${BUILDDIR};
	mkdir -p ${BUILDDIR};
