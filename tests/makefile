SRC=$(wildcard *.c)
TARGET=$(patsubst %.c, % ,$(SRC))

all: $(TARGET)

$(TARGET):
	gcc -o $(patsubst %.c, %c, $@) $@.c

clean:
	rm -rf $(TARGET)

.PHONY: all
