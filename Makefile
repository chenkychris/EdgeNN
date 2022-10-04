RM = rm
NVCC = nvcc
NVCCFLAGS = -Xcompiler -fopenmp
BINDIR = bin
SRCDIR = src
OBJECTS = ${BINDIR}/AlexNet ${BINDIR}/FCNN ${BINDIR}/LeNet ${BINDIR}/ResNet ${BINDIR}/SqueezeNet ${BINDIR}/VGG
ALEXNETMAIN = AlexNet.cu
FCNNMAIN = FCNN.cu
LENETMAIN = LeNet.cu
RESNETMAIN = ResNet.cu
SQUEEZENETMAIN = SqueezeNet.cu
VGGMAIN = VGG.cu

all: ${OBJECTS}

${BINDIR}/AlexNet: ${SRCDIR}/${ALEXNETMAIN}
	${NVCC} ${NVCCFLAGS} -lcuda -lcublas $^ -o $@

${BINDIR}/FCNN: ${SRCDIR}/${FCNNMAIN}
	${NVCC} ${NVCCFLAGS} -Wno-deprecated-gpu-targets -lcuda -lcublas -arch=compute_72 -code=sm_72 $^ -o $@

${BINDIR}/LeNet: ${SRCDIR}/${LENETMAIN}
	${NVCC} ${NVCCFLAGS} -Wno-deprecated-gpu-targets -lcuda -lcublas $^ -o $@

${BINDIR}/ResNet: ${SRCDIR}/${RESNETMAIN}
	${NVCC} ${NVCCFLAGS} -lcuda -lcublas $^ -o $@ 

${BINDIR}/SqueezeNet: ${SRCDIR}/${SQUEEZENETMAIN}
	${NVCC} ${NVCCFLAGS} -arch=compute_72 -code=sm_72 $^ -o $@

${BINDIR}/VGG: ${SRCDIR}/${VGGMAIN}
	${NVCC} ${NVCCFLAGS} $^ -o $@

clean:
	${RM} -f ${OBJECTS}