#include <cuda.h>
#include <driver_types.h> /*cudaError_t*/
#include <cuda_runtime_api.h> /*cudaGetErrorString*/
#include <stdexcept>

template<class T>
class pinned_allocator {
  public:
  
    T* allocate(size_t n) {
        T* result = nullptr;
  
        cudaError_t error = cudaHostAlloc(&result, n*sizeof(T), cudaHostAllocDefault);
  
        if (error != cudaSuccess) {
            throw std::runtime_error(std::string("Unable to allocate memory: ") + cudaGetErrorString(error));
        }
  
        return result;
    }
  
    void deallocate(T* ptr, size_t) {
        cudaError_t error = cudaFree(ptr);
  
        if(error != cudaSuccess) {
            throw std::runtime_error(std::string("Unable to deallocate memory: ") + cudaGetErrorString(error));
        }
    }
};
 
