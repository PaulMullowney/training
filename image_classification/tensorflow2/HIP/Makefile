CC = hipcc
GPU_ARCH = gfx942 

#SRCS_ALL = $(shell echo *.cpp)
SRCS_ALL = roctx.cpp

CFLAGS = -O3 -Wall -fPIC --offload-arch=$(GPU_ARCH) -munsafe-fp-atomics
SRCS = $(SRCS_ALL)

INCLUDES = -I/. -I$(ROCM_PATH)/include/roctracer
LIBS = -L$(ROCM_PATH)/roctracer/lib -lroctracer64 -lroctx64


OBJS = $(SRCS:.c=.o)
TARGET = libHIPcode.so

.PHONY: clean
    
all:    $(TARGET)
	@echo  Successfully compiled ${TARGET} library.

$(TARGET): $(OBJS) 
	$(CC) $(CFLAGS) $(INCLUDES) $(LIBS) -shared -o $(TARGET) $(OBJS) 

.c.o:
	$(CC) $(CFLAGS) $(INCLUDES) -cpp $<  -o $@
clean:
	$(RM) *.o ${TARGET}
