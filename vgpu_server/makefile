SRC = ae.c anet.c server.c zmalloc.c ae_test.c
OBJ = ae.o anet.o server.o zmalloc.o ae_test.o
PSRC = vgpu_server.cpp
POBJ = vgpu_server.o

# test run
test_r:
	./ae_test

run:
	./vgpu_server

build: $(POBJ)
	$(CC) -g -o vgpu_server $(OBJ)

test: $(POBJ)
	$(CC) -g -o ae_test $(OBJ)

clean:
	rm -rf $(OBJ) ae_test

$(POBJ): $(OBJ)
	g++ -g -c $(patsubst %.o, %.cpp, $@) -o $@

$(OBJ):
	gcc -lpthread -g -c $(patsubst %.o, %.c, $@) -o $@




