unsigned char * copyToGPUMemory(unsigned char * byte_arr, unsigned int byte_arr_size);
void freeGPUMemory(unsigned char * gpu_ptr);

unsigned int * copyToGPUMemory(unsigned int * byte_arr, unsigned int byte_arr_size);
void freeGPUMemory(unsigned int * gpu_ptr);
