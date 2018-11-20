#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>

void query_device() {

	int deviceCount { 0 };
	cudaGetDeviceCount(&deviceCount);

	if (deviceCount == 0) {
		std::cout << "No CUDA support device found" << std::endl;
	}

	int devNo { 0 };
	cudaDeviceProp iProp;
	cudaGetDeviceProperties(&iProp, devNo);

	std::cout << "Device " << devNo << ": " << iProp.name << std::endl;
	std::cout << "  Number of multiprocessors: " << iProp.multiProcessorCount
			<< std::endl;
	std::cout << "  clock rate: " << iProp.clockRate << std::endl;
	std::cout << "  Compute capability: " << iProp.major << "." << iProp.minor
			<< std::endl;
	std::cout << "  Total amount of global memory: "
			<< iProp.totalGlobalMem / 1024 << " KB" << std::endl;
	std::cout << "  Total amount of constant memory: "
			<< iProp.totalConstMem / 1024 << " KB" << std::endl;
	std::cout << "  Total amount of shared memory per block: "
			<< iProp.sharedMemPerBlock / 1024 << " KB" << std::endl;
	std::cout << "  Total number of registers available per block: "
			<< iProp.regsPerBlock << std::endl;
	std::cout << "  Warp size: " << iProp.warpSize << std::endl;
	std::cout << "  Maximum number of threads per block: "
			<< iProp.maxThreadsPerBlock << std::endl;
	std::cout << "  Maximum Grid size: (" << iProp.maxGridSize[0] << ", "
			<< iProp.maxGridSize[1] << ", " << iProp.maxGridSize[2] << ")"
			<< std::endl;
	std::cout << "  Maximum block dimension: (" << iProp.maxThreadsDim[0]
			<< ", " << iProp.maxThreadsDim[1] << ", " << iProp.maxThreadsDim[2]
			<< ")" << std::endl;
}

int main() {
	query_device();
	return 0;
}
