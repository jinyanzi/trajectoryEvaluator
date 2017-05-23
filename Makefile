TARGETS = evaluator
CC = g++
CFLAGS = `pkg-config opencv --cflags` -g -Wall -save-temps -O3 -std=c++0x
LIBS = `pkg-config opencv --libs`

all:$(TARGETS)

trajDebugger.o: trajDebugger.h trajDebugger.cpp
	$(CC) $(CFLAGS) -c trajDebugger.cpp 

detectionDubugger.o: detectionDebugger.hpp detectionDebugger.cpp
	$(CC) $(CFLAGS) -c detectionDebugger.cpp

evaluator.o: trajDebugger.h trajDebugger.cpp  detectionDebugger.hpp detectionDebugger.cpp evaluator.h evaluator.cpp
	$(CC) $(CFLAGS) -c trajDebugger.cpp detectionDebugger.cpp evaluator.cpp

evaluator: main.cpp evaluator.o trajDebugger.o detectionDebugger.o
	$(CC) $(CFLAGS) main.cpp evaluator.o detectionDebugger.o trajDebugger.o -o evaluator $(LIBS) # -lmysqlcppconn
	rm -rf *.dSYM

clean:
	rm evaluator *.ii *.s *.o *.avi *.mp4
