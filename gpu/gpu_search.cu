#include "gpu_search.hh" 
//#include "entry_structs.hh"
#include <cuda_runtime.h>

#define MAX_NUM_CHILDREN 6
#define ENTRIES_PER_NODE (MAX_NUM_CHILDREN - 1)
#define ENTRY_SIZE (sizeof(unsigned int) + sizeof(unsigned int) + 2*sizeof(float)) //Same as the getEntrySize(true)
//Assume working with 256 thread DON'T RELY ENTIRERLY ON THOSE! Size may be smaller. need a parameter.
//Requires two more threads then num of entries per node


//We want to copy a whole BTree node to shared memory. We will know the size in advance, we need to distribute the copying between
//our threads. We might end up copying more than we need, but that is fine, as long as we avoid warp divergence.
__global__ void gpuSearchBtree(unsigned char * global_mem, unsigned int start_idx, unsigned int key){
    //We don't care about blcokDim.x because we only have a single block do the searching
    int i = blockIdx.x + threadIdx.x;

    __shared__ unsigned int offsets[MAX_NUM_CHILDREN/2 +1]; //Reads in the first child offset + the shorts
    __shared__ unsigned int entries[ENTRIES_PER_NODE];
    __shared__ unsigned int prefix_sum; //Prefix sum gives us next node size
    __shared__ unsigned int found_idx;
    __shared__ unsigned int size;
    __shared__ unsigned int booleans[2]; //booleans[0] == is_last booleans[1] = exact_match
    __shared__ unsigned int payload[3]; //After we find the correct entry, load the payload here

    unsigned int updated_idx = start_idx + 4; //Update the index for the while loop

    //Split some of the shared memory onto more comfrotable places
    unsigned short * offests_incremental = (unsigned short *)&offsets[1];
    unsigned int * first_child_offset = &offsets[0];

    unsigned int * is_last = &booleans[0];
    unsigned int * exact_match = &booleans[1];

    unsigned int * next_level = &payload[0];
    float * prob = (float *)&payload[1];
    float * backoff = (float *)&payload[2];
    
    int num_entries; //Set the number of entries per node depending on whether we are internal or leaf.

    if (i == 0) {
        size = *(unsigned int *)&global_mem[0];
    }

    //Initialize shared variable
    if (i < 2) {
        booleans[i] = false;
    }


    while (!*exact_match && !*is_last) {
        //First warp divergence here. We are reading in from global memory
        if (i == 0) {
            //@TODO: Replace this with a mod check
            int cur_node_entries = (size - sizeof(unsigned int) - sizeof(unsigned short))/(ENTRY_SIZE + sizeof(unsigned short));
            *is_last = !(ENTRIES_PER_NODE == cur_node_entries);
        }
        __syncthreads();

        //@TODO. Fix this to be more efficient. Maybe move it with entries?
        //As per cuda memory model at least one write will succeed. We are clearing this value
        //So it doesn't interfere with the future values
        prefix_sum = 0;


        if (*is_last) {
            //The number of entries in the bottom most nodes may be smaller than the size
            num_entries = size/ENTRY_SIZE;
            if (i < num_entries) {
                entries[i] = *(unsigned int *)(&global_mem[updated_idx + i*sizeof(unsigned int)]);
                //printf("Entries i: %d, value %d\n", i, entries[i]);
            }
            //printf("Num entries: %d size: %d\n", num_entries, size);
        } else {
            num_entries = ENTRIES_PER_NODE;
            //Load the unsigned int start offset together with the accumulated offsets to avoid warp divergence
            if (i < (MAX_NUM_CHILDREN/2) + 1) {
                offsets[i] = *(unsigned int *)(&global_mem[updated_idx + i*sizeof(unsigned int)]);
            }
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
                found_idx = i;
                if (key == entries[i]) {
                    *exact_match = true;
                }
            }
        } else if (i == num_entries) {
            //Case where our key is greater than the last available entry. We need to do a prefix sum of i+1 elements.
            if (key > entries[i-1]) {
                found_idx = i;
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
            __syncthreads(); //This is not necessary? It is necssary because the threads that don't take the if
            //path may write to the updated idx
            //As per the cuda memory model at least one write will succeed. since they all write the same we don't care
            size = (int)offests_incremental[found_idx];
            updated_idx = *first_child_offset + prefix_sum;
            
        } else if (*is_last && !*exact_match) {
            //In this case we didn't find the key that we were looking for
            //@TODO return a invalid offset when we didn't find anything (mb 0)?
            if (i == 0) {
            printf("Key not found! Key was %d\n", key);               
            }

            break;

        } else {
            //Locate the rest of the data for the entry (i.e. the payload - backoff, prob, next offset)
            if (i < 3) {
                //What we are doing here is reading the correct memory location for our payload. The payload is found
                //After the offsets and the keys, so we skip them and then we skip to the correct payload using found_idx
                if (*is_last) {
                    payload[i] = *(unsigned int *)(&global_mem[updated_idx + num_entries*sizeof(unsigned int) //Skip the keys
                        + found_idx*(sizeof(unsigned int) + sizeof(float) + sizeof(float)) //Skip the previous keys' payload
                            + i*sizeof(unsigned int)]); //Get next_level/prob/backoff
                } else {
                    payload[i] = *(unsigned int *)(&global_mem[updated_idx + sizeof(unsigned int) + MAX_NUM_CHILDREN*sizeof(unsigned short) //Skip the offsets and first offset
                        + num_entries*sizeof(unsigned int) //Skip the keys
                            + found_idx*(sizeof(unsigned int) + sizeof(float) + sizeof(float)) //Skip the previous keys' payload
                                + i*sizeof(unsigned int)]);  //Get next_level/prob/backoff
                }
            }
            __syncthreads();
            
            if (i == 1) {
                printf("Exact match! Found_idx: %d, key: %d found: %d\n", found_idx, key, entries[found_idx]);
                printf("Next level: %d, prob %f, backoff %f\n", *next_level, *prob, *backoff);
            }
            break;
        }
    }

    //We didn't find anything our btree doesn't contain the key
    

}


void searchWrapper(unsigned char * global_mem, unsigned int start_idx, unsigned int key) {
    //Block size should always be MAX_NUM_CHILDREN for best efficiency when searching the btree
    gpuSearchBtree<<<1, MAX_NUM_CHILDREN>>>(global_mem, start_idx, key);
}

/* Can't compile easily with cmake. Maybe there's a better way
__global__ void searchInBulk(unsigned int * keys_array, unsigned char * btree_trie) {
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    gpuSearchBtree<<<1, MAX_NUM_CHILDREN>>>(btree_trie, 0, keys_array[i]);
}
*/

void cudaDevSync() {
    cudaDeviceSynchronize();
}
