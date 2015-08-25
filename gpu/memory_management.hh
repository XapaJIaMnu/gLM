unsigned char * copyToGPUMemory(unsigned char * byte_arr, unsigned int byte_arr_size);
void freeGPUMemory(unsigned char * gpu_ptr);

unsigned int * copyToGPUMemory(unsigned int * byte_arr, unsigned int byte_arr_size);
void freeGPUMemory(unsigned int * gpu_ptr);

void allocateGPUMem(size_t size, unsigned int ** gpu_mem);
void copyToHostMemory(unsigned int * gpu_mem, unsigned int * cpu_mem, size_t size);
