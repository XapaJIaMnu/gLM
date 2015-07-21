#include "gpu_search.hh" 
//#include "entry_structs.hh"
#include <cuda_runtime.h>

#define MAX_NUM_CHILDREN 6
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
    __shared__ unsigned int size;
    __shared__ unsigned int booleans[4]; //booleans[0] == is_last booleans[1] = exact_match idx 3 and 4 are empty but preserve memory alignment.

    unsigned int updated_idx = start_idx + 4; //Update the index for the while loop

    //Split some of the shared memory onto more comfrotable places
    unsigned short * offests_incremental = (unsigned short *)&offsets[1];
    unsigned int * first_child_offset = &offsets[0];
    unsigned int * is_last = &booleans[0];
    unsigned int * exact_match = &booleans[1];
    int num_entries; //Set the number of entries per node depending on whether we are internal or leaf.

    if (i == 0) {
        size = *(unsigned int *)&global_mem[0];
    }

    //Initialize shared variable
    if (i < 2) {
        booleans[i] = false;
    }
    
    
    //__syncthreads();

    while (!*exact_match && !*is_last) {
        //First warp divergence here. We are reading in from global memory
        if (i == 0) {
            int cur_node_entries = (size - sizeof(unsigned int) - sizeof(unsigned short))/(entry_size + sizeof(unsigned short));
            //printf("Cur node entries: %d, size: %d\n", cur_node_entries, size);
            *is_last = !(ENTRIES_PER_NODE == cur_node_entries);
            //printf("Is last: %d\n", booleans[0]);
        }
        //@TODO. Fix this to be more efficient. Maybe move it with entries?
        //We need to clear prefix sum every time before we use it.
        if (i == 3) {
            prefix_sum = 0;
        }
        __syncthreads();

        if (*is_last) {
            //The number of entries in the bottom most nodes may be smaller than the size
            num_entries = size/entry_size;
            if (i < num_entries) {
                entries[i] = *(unsigned int *)(&global_mem[updated_idx + i*sizeof(unsigned int)]);
            }
            //printf("Num entries: %d size: %d\n", num_entries, size);
        } else {
            //Indexes in this case are wrong,redo them
            num_entries = ENTRIES_PER_NODE;
            //Load the unsigned int start offset together with the accumulated offsets to avoid warp divergence
            if (i < (MAX_NUM_CHILDREN/2) + 1) {
                offsets[i] = *(unsigned int *)(&global_mem[updated_idx + i*sizeof(unsigned int)]);
                //printf("NONLAST: Offset i is: %d\n", offsets[i]);
            }
            __syncthreads();
            //printf("offests_incremental: %d\n", offests_incremental[i]);
            //Now load the entries
            if (i < num_entries) {
                entries[i] = *(unsigned int *)(&global_mem[updated_idx + sizeof(unsigned int) + MAX_NUM_CHILDREN*sizeof(unsigned short) + i*sizeof(unsigned int)]);
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
        } else if (i < num_entries) {
            if (key > entries[i-1] && key <= entries[i]){
                printf("MIDDLE CASE: is_last: %d, num_entries is: %d, entries[i] is: %d found_idx is: %d\n", *is_last, num_entries, entries[i], found_idx);
                found_idx = i;
                if (key == entries[i]) {
                    *exact_match = true;
                }
            }
        } else if (i == num_entries) {
            if (key > entries[i]) {
                found_idx = i;
                printf("is_last: %d, num_entries is: %d, entries[i] is: %d found_idx is: %d\n", *is_last, num_entries, entries[i], found_idx);
                //if (key == entries[i - 1]) {
                //    *exact_match = true;
                //}
            }
        }
        __syncthreads();
        if (i == 0) {
            printf("We have found: %d at position: %d\n", entries[found_idx], found_idx);
        }
        //We found either an exact match (so we can access next level) or at least an address to next btree level
        if (!*exact_match && !*is_last) {
            //Do a prefix sum on the offsets here
            //@TODO optimize later. Do a proper prefix sum rather than atomic add
            if (i < found_idx) {
               atomicAdd(&prefix_sum, (int)offests_incremental[i]); 
            }
            __syncthreads();
            //As per the cuda memory model at least one write will succeed. since they all write the same we don't care
            size = (int)offests_incremental[found_idx];
            updated_idx = *first_child_offset + prefix_sum;
            //printf("updated_idx %d, prefix_sum: %d\n", updated_idx, prefix_sum);
            
        } else {
            //Locate the rest of the data for the entry (i.e. the payload - backoff, prob, next offset)
            if (i == 1) {
                printf("Exact match! Found_idx: %d, key: %d found: %d\n", found_idx, key, entries[found_idx]);
            }
            break;
        }
    }

    //We didn't find anything our btree doesn't contain the key
    

}

void searchWrapper(unsigned char * global_mem, unsigned int start_idx, int first_size, unsigned int key, int grid, int block) {
    gpuSearchBtree<<<grid, block>>>(global_mem, start_idx, first_size, key, 16);
}

void cudaDevSync() {
    cudaDeviceSynchronize();
}
