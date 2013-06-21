all: sift_descriptors sift_extractor
	
clean:
	rm -rf *.o sift_descriptors sift_extractor
	
sift_descriptors: sift_descriptors.o
	g++ sift_descriptors.o -o sift_descriptors -g  `pkg-config opencv --cflags --libs`
	
sift_descriptors.o: sift_descriptors.cpp
	g++ -c sift_descriptors.cpp -g -I/opt/local/include -g

sift_extractor: sift_extractor.o
	g++ sift_extractor.o -o sift_extractor -g `pkg-config opencv --cflags --libs`
	
sift_extractor.o: sift_extractor.cpp
	g++ -c sift_extractor.cpp -g `pkg-config opencv --cflags --libs` 