CXX = g++
CXXFLAGS = -std=c++20
COMMONFILES = common.cpp
COMMONHEADERS = common.h
LOGGERFILES = logger.cpp

all: mouse_logger

default: mouse_logger

common.o: $(COMMONFILES) $(COMMONHEADERS)
	$(CXX) $(CXXFLAGS) -c $(COMMONFILES) -o common.o

mouse_logger: $(LOGGERFILES) common.o
	$(CXX) $(CXXFLAGS) $(LOGGERFILES) common.o -o mouse_logger

clean:
	rm -f *.o mouse_logger data.txt