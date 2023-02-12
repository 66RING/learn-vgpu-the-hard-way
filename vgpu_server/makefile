# ae_test.c vgpu_server.c
# ae_test.o vgpu_server.o
SRC = ae.c anet.c adlist.c server.c zmalloc.c
OBJ = ae.o anet.o adlist.o server.o zmalloc.o

# test run
test_r:
	./ae_test

run:
	./vgpu_server

build: $(OBJ)
	gcc -g -o vgpu_server $(OBJ) vgpu_server.c

build_test: $(OBJ)
	gcc -g -o ae_test $(OBJ) ae_test.c

test: $(OBJ)
	gcc -g -o ae_test $(OBJ)

clean:
	rm -rf $(OBJ) ae_test vgpu_server

$(OBJ):
	gcc -lpthread -g -c $(patsubst %.o, %.c, $@) -o $@




