all: sift_descriptors sift_extractor
	
clean:
	rm -rf *.o sift_descriptors sift_extractor
	
sift_descriptors: sift_descriptors.o
	g++ sift_descriptors.o -o sift_descriptors -O2  `pkg-config opencv --cflags --libs`
	
sift_descriptors.o: sift_descriptors.cpp
	g++ -c sift_descriptors.cpp -O2 -I/opt/local/include

sift_extractor: sift_extractor.o
	g++ sift_extractor.o -o sift_extractor -O2 `pkg-config opencv --cflags --libs`
	
sift_extractor.o: sift_extractor.cpp
	g++ -c sift_extractor.cpp -O2 `pkg-config opencv --cflags --libs` 