# ae_test.c vgpu_server.c
# ae_test.o vgpu_server.o
SRC = ae.c anet.c adlist.c server.c zmalloc.c vgpu_server.c
OBJ = ae.o anet.o adlist.o server.o zmalloc.o vgpu_server.o

# test run
rpc_test: build_rpc_test
	./rpc_test

ae_test: build_ae_test
	./ae_test

# run:
# 	./vgpu_server

# build: $(OBJ)
# 	gcc -g -o vgpu_server $(OBJ) vgpu_server.c

build_test: build_ae_test build_rpc_test

build_ae_test: $(OBJ)
	gcc -g -o ae_test $(OBJ) ae_test.c

build_rpc_test: $(OBJ)
	gcc -g -o rpc_test $(OBJ) rpc_test.c

clean:
	rm -rf $(OBJ) ae_test vgpu_server

$(OBJ):
	gcc -lpthread -g -c $(patsubst %.o, %.c, $@) -o $@




