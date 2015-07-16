#include "gpu_search.hh" 
//#include "entry_structs.hh"
#include <cuda_runtime.h>

#define MAX_NUM_CHILDREN 256
#define ENTRIES_PER_NODE MAX_NUM_CHILDREN - 1
//Assume working with 256 thread DON'T RELY ENTIRERLY ON THOSE! Size may be smaller. need a parameter.
//Requires two more threads then num of entries per node


//We want to copy a whole BTree node to shared memory. We will know the size in advance, we need to distribute the copying between
//our threads. We might end up copying more than we need, but that is fine, as long as we avoid warp divergence.
__global__ void gpuSearchBtree(unsigned char * global_mem, unsigned int start_idx, int first_size, unsigned int key, int entry_size){
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ unsigned int offsets[MAX_NUM_CHILDREN/2 +1]; //Reads in the first child offset + the shorts
    __shared__ unsigned int entries[ENTRIES_PER_NODE];
    __shared__ unsigned int prefix_sum; //Prefix sum gives us next entry size
    __shared__ unsigned int found_idx;
    __shared__ bool booleans[4]; //booleans[0] == is_last booleans[1] = exact_match idx 3 and 4 are empty but preserve memory alignment.

    unsigned int updated_idx = start_idx; //Update the index for the while loop
    unsigned int size = first_size;

    //Split some of the shared memory onto more comfrotable places
    unsigned short * offests_incremental = (unsigned short *)&offsets[1];
    unsigned int * first_child_offset = &offsets[0];
    bool * is_last = &booleans[0];
    bool * exact_match = &booleans[1];

    //Initialize shared variable
    if (i < 2) {
        booleans[i] = false;
    }
    //@TODO. Fix this to be more efficient. Maybe move it with entries?
    if (i == 3) {
        prefix_sum = 0;
    }
    __syncthreads();

    while (!*exact_match && !*is_last) {
        //First warp divergence here. We are reading in from global memory
        if (i == 0) {
            *is_last = (bool)global_mem[updated_idx];
        }
        __syncthreads();

        if (is_last) {
            //The number of entries in the bottom most nodes may be smaller than the size
            if (i < (size - 1)/entry_size) {
                entries[i] = *(unsigned int *)(&global_mem[updated_idx + 1 + i*sizeof(unsigned int)]);
            }
        } else {
            int num_entries = (size - 1 - sizeof(unsigned int) - sizeof(unsigned short))/(entry_size + sizeof(unsigned short));
            //Load the unsigned int start offset together with the accumulated offsets to avoid warp divergence
            if (i < ((num_entries + 1)/2) + 1) {
                offsets[i] = *(unsigned int *)(&global_mem[updated_idx + 1 * i*sizeof(unsigned int)]);
            }
            __syncthreads();
            //Now load the entries
            if (i < num_entries) {
                entries[i] = *(unsigned int *)(&global_mem[updated_idx + 1 + (num_entries + 1)*sizeof(unsigned int) + i*sizeof(unsigned int)]);
            }
        }
        __syncthreads();

        //NOW search
        if (i == 0) {
            if (key <= entries[i]) {
                found_idx = i;
                if (key == entries[i]) {
                    *exact_match = true;
                }
            }
        } else if (i < ENTRIES_PER_NODE) {
            if (key >= entries[i-1] && key < entries[i]){
                found_idx = i;
                if (key == entries[i]) {
                    *exact_match = true;
                }
            }
        } else if (i == ENTRIES_PER_NODE) {
            if (key >= entries[i - 1]) {
                found_idx = i;
                if (key == entries[i - 1]) {
                    *exact_match = true;
                }
            }
        }
        __syncthreads();
        //We found either an exact match (so we can access next level) or at least an address to next btree level
        if (!*exact_match && !*is_last) {
            //Do a prefix sum on the offsets here
            //@TODO optimize later. Do a proper prefix sum rather than atomic add
            if (i < found_idx) {
               atomicAdd(&prefix_sum, (int)offests_incremental[i]); 
            }
            __syncthreads();
            //As per the cuda memory model at least one write will succeed. since they all write the same we don't care
            size = prefix_sum;
            updated_idx = *first_child_offset + prefix_sum;
            
        } else {
            //Locate the rest of the data for the entry (i.e. the payload - backoff, prob, next offset)
            break;
        }
    }

    //We didn't find anything our btree doesn't contain the key
    

}

void searchWrapper(unsigned char * global_mem, unsigned int start_idx, int first_size, unsigned int key, int grid, int block) {
    gpuSearchBtree<<<grid, block>>>(global_mem, start_idx, first_size, key, 16);
}
