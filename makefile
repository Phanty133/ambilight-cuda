################################################################################
# Copyright (c) 2013-2014, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
################################################################################
include ./utils.mk

UNAME = $(shell uname)
TARGET = ambilight

CC = nvcc
CFLAGS = -Iinc
ifeq ($(UNAME), Linux)
	LIBS += -ldl -lcuda
endif

SOURCES = src/main.cu src/CudaUtils.cpp src/NvFBCUtils.cpp
OBJECTS = $(call BUILD_OBJECT_LIST,$(SOURCES))
HEADERS = inc/NvFBCUtils.h inc/cuda.h inc/CudaUtils.h inc/NvFBC.h 

.PRECIOUS: $(TARGET) $(OBJECTS)
.PHONY: default all clean

default: $(TARGET)
all: default

$(foreach src,$(SOURCES),$(eval $(call DEFINE_OBJECT_RULE,$(src),$(HEADERS))))

$(TARGET): $(OBJECTS)
	$(NVCC) $(OBJECTS) $(LIBS) -o $@

clean:
	-rm -f *.o *.bmp 
	-rm -f $(TARGET)
