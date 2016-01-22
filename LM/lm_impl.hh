#ifndef API_VERSION //Means we have included the lm.h header, so we don't need it
    #include "lm.hh"
#endif 
//Use boost to serialize the vectors
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/unordered_map.hpp>
#include <boost/filesystem.hpp>

//Compress the binary format to bzip2
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/bzip2.hpp> 

template<class StringType, class DataStructure>
void serializeDatastructure(DataStructure& byte_arr, const StringType path){
    std::ofstream os (path, std::ios::binary);

    {// The boost archive needs to be in a separate scope otherwise it doesn't flush properly
        //Create a bzip2 compressed filtered stream
        boost::iostreams::filtering_stream<boost::iostreams::output> filtered_stream;
        filtered_stream.push(boost::iostreams::bzip2_compressor());
        filtered_stream.push(os);

        boost::archive::text_oarchive oarch(filtered_stream);
        oarch << byte_arr;
    }
    os.close();
}

//The byte_arr vector should be reserved to prevent constant realocation.
template<class StringType, class DataStructure>
void readDatastructure(DataStructure& byte_arr, const StringType path){
    std::ifstream is (path, std::ios::binary);

    if (is.fail() ){
        std::cerr << "Failed to open file " << path << std::endl;
        std::exit(EXIT_FAILURE);
    }
    {// The boost archive needs to be in a separate scope otherwise it doesn't flush properly
        //Create a bzip2 compressed filtered stream for decopression
        boost::iostreams::filtering_stream<boost::iostreams::input> filtered_stream;
        filtered_stream.push(boost::iostreams::bzip2_decompressor());
        filtered_stream.push(is);

        boost::archive::text_iarchive iarch(filtered_stream);
        iarch >> byte_arr;
    }
    is.close();
}

template<class StringType>
void LM::storeConfigFile(const StringType path) {
    std::ofstream configfile(path);

    if (configfile.fail()) {
        std::cerr << "Failed to open file " << path << std::endl;
        std::exit(EXIT_FAILURE);
    }

    configfile << metadata.byteArraySize << '\n';
    configfile << metadata.intArraySize << '\n';
    configfile << metadata.max_ngram_order << '\n';
    configfile << metadata.api_version << '\n';
    configfile << metadata.btree_node_size << '\n';
    //Also store in the config file the size of the datastructures. Useful to know if we can fit our model
    //on the available GPU memory, but we don't actually need to ever read it back. It is for the user's benefit.
    configfile << "First trie level memory size: " << metadata.intArraySize/(4*1024*1024) << " MB\n";
    configfile << "BTree Trie memory size: " << metadata.byteArraySize/(1024*1024) << " MB\n";
    configfile << "Total GPU memory required: " << metadata.byteArraySize/(1024*1024) +  metadata.intArraySize/(4*1024*1024) << " MB\n";
    configfile.close();
}

template<class StringType>
void LM::readConfigFile(const StringType path) {
    std::ifstream configfile(path);

    if (configfile.fail()) {
        std::cerr << "Failed to open file " << path << std::endl;
        std::exit(EXIT_FAILURE);
    }

    std::string line;

    //Get byte array size
    getline(configfile, line);
    metadata.byteArraySize = std::stoull(line.c_str());

    //Get int array size
    getline(configfile, line);
    metadata.intArraySize = std::stoull(line.c_str());

    //Get max ngram order
    getline(configfile, line);
    metadata.max_ngram_order = atoi(line.c_str());

    //Check api version
    getline(configfile, line);
    if (atof(line.c_str()) != API_VERSION) {
        std::cerr << "The gLM API has changed, please rebinarize your language model." << std::endl;
        exit(EXIT_FAILURE);
    } else {
        metadata.api_version = atof(line.c_str());
    }

    //Get btree_node_size
    getline(configfile, line);
    metadata.btree_node_size = atoi(line.c_str());

}

template<class StringType>
void createDirIfnotPresent(const StringType path) {
    boost::filesystem::path dir(path);

    if (!(boost::filesystem::exists(dir))) {
        //Directory doesn't exist, try to create it
        if (boost::filesystem::create_directory(dir)) {
            //all is good!
        } else {
            std::cerr << "Failed to create a directory at " << path << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }
}

//Given a path and an LM binarizes the LM there
template<class StringType>
void LM::writeBinary(const StringType path) {
    createDirIfnotPresent(path);
    std::string basepath(path);
    storeConfigFile(basepath + "/config");
    serializeDatastructure(encode_map, basepath + "/encode.map");
    serializeDatastructure(decode_map, basepath + "/decode.map");

    //Use mmap for the big files
    std::ofstream os (basepath + "/lm.bin", std::ios::binary);  
    os.write(reinterpret_cast<const char *>(trieByteArray.data()), trieByteArray.size());
    os.close();

    std::ofstream os2 (basepath + "/first_lvl.bin", std::ios::binary);  
    os2.write(reinterpret_cast<const char *>(first_lvl.data()), trieByteArray.size()*sizeof(unsigned int));
    os2.close();
    
}

//Reads the model into the given (presumably empty byte_arr)
template<class StringType>
LM::LM(const StringType path) {
    diskIO = true; //Indicate that we have performed diskIO and we need to call munmap in the destructor

    std::string basepath(path);
    this->readConfigFile(basepath + "/config");
    readDatastructure(this->encode_map, basepath + "/encode.map");
    readDatastructure(this->decode_map, basepath + "/decode.map");

    //@TODO we should really make that step optional
    //We don't need the copy to vector since we're only going to copy it to GPU memory
    mmapedByteArray = (unsigned char *)readMmapTrie((basepath + "/lm.bin").c_str(), metadata.byteArraySize);
    trieByteArray.resize(metadata.byteArraySize);
    std::memcpy(trieByteArray.data(), mmapedByteArray, metadata.byteArraySize);

    if (metadata.intArraySize) { //Only readIn the int Array if it is actually used
        mmapedFirst_lvl = (unsigned int *)readMmapTrie((basepath + "/first_lvl.bin").c_str(), metadata.intArraySize*sizeof(unsigned int));
        first_lvl.resize(metadata.intArraySize);
        std::memcpy(first_lvl.data(), mmapedFirst_lvl, metadata.intArraySize*sizeof(unsigned int));
    }
}

void * readMmapTrie(const char * filename, size_t size) {
    //Initial position of the file is the end of the file, thus we know the size
    int fd;
    void * map;

    fd = open(filename, O_RDONLY);
    if (fd == -1) {
        perror("Error opening file for reading");
        exit(EXIT_FAILURE);
    }

    map = mmap(0, size, PROT_READ, MAP_SHARED, fd, 0);

    if (map == MAP_FAILED) {
        close(fd);
        perror("Error mmapping the file");
        exit(EXIT_FAILURE);
    }

    return map;
} 
