CC=pgcc
CFLAGS=-ta=tesla:pinned -Minfo=all
CXX=pgc++
CXXFLAGS=-ta=tesla:pinned -Minfo=all

filter.x: filter.o main.o
	$(CXX) $(CXXFLAGS) -mp $^ -lopencv_highgui -lopencv_core -lm -o $@ -Mcuda -Minfo=all

.SUFFIXES: .o .cpp .c
.cpp.o: 
	$(CXX) $(CXXFLAGS) -c -o $@ $<
.c.o: 
	$(CC) $(CFLAGS) -c -o $@ $<
.PHONY: clean
clean: 
	-rm -f *.o core filter.x
