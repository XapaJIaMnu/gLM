#include <vector>
#include <utility>
#include <iostream>
#include <math.h>
#include <fstream>

class B_tree {
    public:
        void * root_node; //it should be B_tree_node. Avoid cyclical definition
        B_tree(unsigned short);
        ~B_tree();
        void draw_tree();
        void insert_entry(int value);
        void produce_graph(const char * filename);
};

class B_tree_node {
    
    public:
        unsigned short max_elem;
        std::vector<int> values;
        std::vector<B_tree_node *> children;
        B_tree_node * parent;
        B_tree * container; //Accessible only from the root node
        bool is_root;

        //Operations
        B_tree_node(unsigned short, B_tree_node *);
        B_tree_node(unsigned short, B_tree *);
        ~B_tree_node();
        //Insertion to vector location is index to be inserted (before the old one)
        void insert(B_tree_node * new_node, int location);
        void insert(int new_val, int location);
        std::pair<B_tree_node *, int> find_position(int new_value);
        void split();
};

B_tree::B_tree(unsigned short num_max_elem) {
    B_tree_node * root = new B_tree_node(num_max_elem, this); //This should be safe, I am just need "this" for the address.
    root_node = reinterpret_cast<void *>(root);
}

B_tree::~B_tree() {
    B_tree_node * actual_node = reinterpret_cast<B_tree_node *>(root_node);
    delete actual_node;
}

B_tree_node::B_tree_node(unsigned short num_max_elem, B_tree_node * parent_node) {
    max_elem = num_max_elem;
    is_root = false;
    parent = parent_node;
    container = nullptr;
}

B_tree_node::B_tree_node(unsigned short num_max_elem, B_tree * container_orig) {
    max_elem = num_max_elem;
    is_root = true;
    parent = nullptr;
    container = container_orig;
}

B_tree_node::~B_tree_node(){
    values.clear();
    for (int i = 0; i < children.size(); i++) {
        delete children[i];
    }
}

void B_tree_node::insert(B_tree_node * new_node, int location){
    std::vector<B_tree_node *>::iterator it = children.begin() + location;
    children.insert(it, new_node);
}

void B_tree_node::insert(int new_val, int location){
    std::vector<int>::iterator it = values.begin() + location;
    values.insert(it, new_val);
}

std::pair<B_tree_node *, int> B_tree_node::find_position(int new_value) {

    int candidate_position = values.size(); //Assume last position

    for (int i = 0; i < values.size(); i++){
        if (values[i] > new_value) {
            //We can never have two nodes with the same value as per specification.
            candidate_position = i;
            break;
        }
    }

    if (children.size() != 0){
        return children[candidate_position]->find_position(new_value);
    } else {
        return std::pair<B_tree_node *, int>(this, candidate_position);
    }

}

void B_tree_node::split() {
    if (values.size() <= max_elem){
        //No need to split the node. It's balanced.
        return;
    }

    int middle_idx = values.size()/2; //Integer division here, always right

    //We if we need to split we take our current node to become the left node
    //by trimming it and we will create a new node which will be the right node
    //from the elements that we previously cut out.

    //Calculate middle index for children. Different cases for odd and even
    //We can't use (children.size() + 1)/2 because children.size() can be 0
    int child_mid_idx;
    if (children.size() % 2 == 0) {
        //Even values:
        child_mid_idx = children.size()/2;
    } else {
        child_mid_idx = (children.size()/2) + 1;
    }

    //Save the middle value;
    int middle_value = values[middle_idx];


    //Create the right node
    B_tree_node * right_node = new B_tree_node(max_elem, parent);

    //populate it.
    for (std::vector<int>::iterator it = values.begin() + middle_idx + 1; it != values.end(); it++){
        right_node->values.push_back(*it);
    }
    
    for (std::vector<B_tree_node *>::iterator it = children.begin() + child_mid_idx; it != children.end(); it++){
        (*it)->parent = right_node;
        right_node->children.push_back(*it);
    }

    //Trim the left node.
    this->values.resize(middle_idx);
    this->children.resize(child_mid_idx);

    //Assign parent node and change root if necessary
    if (parent == nullptr) {
        //We are the root node, we need to make a new root.
        B_tree_node * new_root = new B_tree_node(max_elem, container);
        new_root->insert(middle_value, 0);

        this->is_root = false;
        new_root->container = this->container;
        new_root->container->root_node = reinterpret_cast<void *>(new_root);
        this->container = nullptr;
        this->parent = new_root;
        right_node->parent = new_root;

        //Now assign child pointers to the new root.
        new_root->insert(right_node, 0);
        new_root->insert(this, 0);
        //We are done, the tree is balanced and split;
    } else {
        //Find the location of the middle_value in the parent
        int new_location = parent->values.size();
        for (int i = 0; i< parent->values.size(); i++) {
            if (parent->values[i] > middle_value) {
                new_location = i;
                break;
            }
        }
        //insert the middle_value and the right node (the left was there beforehand)
        parent->insert(middle_value, new_location);
        parent->insert(right_node, new_location+1);

        //Check if parent is balanced
        parent->split();
    }


}

void B_tree::insert_entry(int value) {
    B_tree_node * root = reinterpret_cast<B_tree_node *>(root_node);
    std::pair<B_tree_node *, int> position = root->find_position(value);
    position.first->insert(value, position.second);
    position.first->split();
}

void print_node(B_tree_node * node) {
    std::cout << std::endl << "Node_ID: " << node << std::endl;
    if (node->parent) {
        std::cout << "Parent is: " << node->parent;
    }
    std::cout << " Num keys: " << node->values.size();
    std::cout << " Num children: " << node-> children.size();
    std::cout << " Values: ";
    for (unsigned short i = 0; i<node->values.size(); i++) {
        std::cout << node->values[i] << ' ';
    }
    for (unsigned short i = 0; i<node->children.size(); i++) {
        print_node(node->children[i]);
    }

}

void B_tree::draw_tree() {
    //Draws what is currently in the tree. For debugging purposes
    print_node(reinterpret_cast<B_tree_node *>(root_node));
}

void draw_node(B_tree_node * node, std::ofstream& filehandle, int num_child) {

    filehandle << "node" << node << "[label = \"<f0> ";
    for (int i = 0; i<node->values.size(); i++) {
        filehandle << '|' << node->values[i] << '|' << "<f" << (i + 1) << '>';
    }
    filehandle << "\"];\n";
    if (num_child != -1) {
        //Root has children but has no parents so it has no incoming connections
        //Otherwise draw the connection
        filehandle << "\"node" << node->parent << "\":f" <<  num_child  << " -> \"node" << node << "\"\n";
    }
    for (int i = 0; i < node->children.size(); i++) {
        draw_node(node->children[i], filehandle, i);
    }
}

void B_tree::produce_graph(const char * filename) {
    std::ofstream graphfile;
    graphfile.open(filename);
    graphfile << "digraph g {\nnode [shape = record,height=.1];\n";
    draw_node(reinterpret_cast<B_tree_node *>(root_node), graphfile, -1);
    graphfile << "}\n";
    graphfile.close();
}
