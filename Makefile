all: sift_descriptors sift_extractor vis_words_extractor
	
clean:
	rm -rf *.o sift_descriptors sift_extractor
	
sift_descriptors: sift_descriptors.o
	g++ sift_descriptors.o -o sift_descriptors -O2  `pkg-config opencv --cflags --libs`
	
sift_descriptors.o: sift_descriptors.cpp
	g++ -c sift_descriptors.cpp -O2 -I/opt/local/include

sift_extractor: sift_extractor.o
	g++ sift_extractor.o -o sift_extractor -O2 `pkg-config opencv --cflags --libs` -lpthread
	
sift_extractor.o: sift_extractor.cpp
	g++ -c sift_extractor.cpp -O2 `pkg-config opencv --cflags --libs` -lpthread
	
vis_words_extractor: vis_words_extractor.o
	g++ vis_words_extractor.o -o vis_words_extractor -O2 `pkg-config opencv --cflags --libs` -lpthread
	
vis_words_extractor.o: vis_words_extractor.cpp
	g++ -c vis_words_extractor.cpp -O2 `pkg-config opencv --cflags --libs` -lpthread