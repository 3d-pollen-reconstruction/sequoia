nvcc = /usr/local/cuda/bin/nvcc
cudalib = /usr/local/cuda/lib64

TF_CFLAGS := -I/usr/local/lib/python3.6/site-packages/tensorflow_core/include \
             -I/usr/local/lib/python3.6/site-packages/tensorflow_core/include/external/nsync/public

TF_LFLAGS := -L$(HOME) -ltensorflow_framework

all: tf_approxmatch_so.so tf_nndistance_so.so

tf_approxmatch_so.so: tf_approxmatch_g.cu.o tf_approxmatch.cpp
	g++ -std=c++11 tf_approxmatch.cpp tf_approxmatch_g.cu.o -o tf_approxmatch_so.so -shared -fPIC $(TF_CFLAGS) -lcudart $(TF_LFLAGS) -L$(cudalib) -O2 -D_GLIBCXX_USE_CXX11_ABI=0

tf_approxmatch_g.cu.o: tf_approxmatch_g.cu
	$(nvcc) -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11 -c -o tf_approxmatch_g.cu.o tf_approxmatch_g.cu $(TF_CFLAGS) $(TF_LFLAGS) -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2

tf_nndistance_so.so: tf_nndistance_g.cu.o tf_nndistance.cpp
	g++ -std=c++11 tf_nndistance.cpp tf_nndistance_g.cu.o -o tf_nndistance_so.so -shared -fPIC $(TF_CFLAGS) -lcudart $(TF_LFLAGS) -L$(cudalib) -O2 -D_GLIBCXX_USE_CXX11_ABI=0

tf_nndistance_g.cu.o: tf_nndistance_g.cu
	$(nvcc) -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11 -c -o tf_nndistance_g.cu.o tf_nndistance_g.cu $(TF_CFLAGS) $(TF_LFLAGS) -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2

clean:
	rm -f *.so *.o
