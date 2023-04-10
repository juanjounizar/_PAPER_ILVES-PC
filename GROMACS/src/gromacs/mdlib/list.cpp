//----------------------------------------------------------------------------
//-- DUMMY STATEMENT 78 CHARACTERS LONG TO ENSURE THE LINES ARE NOT TOO LONG -
//----------------------------------------------------------------------------

/*  LIST, version 1.0.0

    Author:
    Carl Christian Kjelgaard Mikkelsen,
    Department of Computing Science and HPC2N
    Umeaa University
    Sweden
    Email: spock@cs.umu.se

    Date: August 14th, 2014

    This is a library of subroutines for manipulating simply linked lists in
    the context of molecules. It would be natural to rewrite this library to
    handle arbitrary lists. The user would need to supply a function which can
    compare to elements and decide on the order.

    It is convinient to think of a singly linked list as a passenger train.
    The conductor starts at one end of the train, i.e. the root, and advances
    through the train one car at a time.

*/

#include <stdlib.h>
#include <stdio.h>
#include "list.h"

//----------------------------------------------------------------------------
int sinsert(int n, my_node_t **root) {

    /* Inserts an integer into a sorted list.

       DESCRIPTION OF INTERFACE:

       ON ENTRY:
         n       an integer which is to be inserted into the list
         root    pointer to pointer to the first node in the list

       ON EXIT:

         If the integer is not in the list, then it is inserted and RC = 1
         If the integer is in the list, then the list is unchanged and RC = 0.

     */

    //--------------------------------------------------------------------------
    // Declaration of internal variables
    //--------------------------------------------------------------------------

    // The conductor which walks though the list
    my_node_t *conductor;

    // A new node which might be used
    my_node_t *new_node;

    // Return code
    int rc = 0;

    //--------------------------------------------------------------------------
    // Start of instructions
    //--------------------------------------------------------------------------

    // Create a new node with the integer n;
    new_node = (my_node_t *)malloc(sizeof(my_node_t));
    new_node->number = n;

    // Initialize the conductor to point to the first element.
    conductor = *root;

    if (*root == NULL) { // This is a special case where the list is empty
        *root = new_node;
        new_node->next = NULL;
        // We inserted an element, so RC=1
        rc = 1;
    }
    else {
        if (n < conductor->number) {
            /* This is a special case were the new number is smaller than the
            first number in the list */
            new_node->next = *root;
            *root = new_node;
            // We inserted an element, so RC=1
            rc = 1;
        }
        else {
            // This is the general case
            while (conductor->next != NULL && conductor->next->number <= n) {
                // While there are smaller numbers ahead in the list, advance!
                conductor = conductor->next;
            }
            /* At this point there are the following possibilities
             a) conductor->next=NULL: there are no numbers ahead in the list
             b) conductor->next->number>n: the next number is strictly larger
                than n

              It is important to realize that the current number, i.e.
            conductor->number is certain to be less than or equal to n at this
            point, or we would not have advanced to this element. */

            if (conductor->number < n) {
                // The number n is not in the list, so we insert it.
                new_node->next = conductor->next;
                conductor->next = new_node;
                // We inserted an element, so RC=1.
                rc = 1;
            }
            else {
                /* The number n is already in the list, so we just free the memory
                   occupied by new. */
                free(new_node);
            }
        }
    }
    return rc;
}

//----------------------------------------------------------------------------
int sdelete(int target, my_node_t **head) {

    /* Scans a sorted linked list for a node with a given target value

       If the target is found, then the node is deleted and RC=1.
       If the target is not found, then nothing is done and RC=0.

    */

    // The head of the list
    my_node_t *current = *head;

    // The previous node
    my_node_t *previous = NULL;

    // Assume that the target is not in the list
    int rc = 0;

    while ((current != NULL) && (current->number <= target)) {
        // There is at least one element in the list ...
        if (current->number == target) {
            my_node_t *del = current;
            // The head of the list is a special case
            if (previous != NULL) {
                previous->next = current->next;
            }
            else {
                *head = current->next;
            }
            free(del);

            rc = 1;
            break;
        }
        // Move on to the next element ...
        previous = current;
        current = current->next;
    }
    return rc;
}


//----------------------------------------------------------------------------
void print_list(my_node_t *root) {

    // Prints the numbers stored in a simply linked list with a given root.
    my_node_t *conductor;

    conductor = root;
    while (conductor != NULL) {
        printf("%8d ", conductor->number);
        conductor = conductor->next;
    }
    printf("\n");
}

void insert(int n, my_node_t **root) {
    // Create a new node with the integer n;
    my_node_t *new_node = (my_node_t *) malloc(sizeof(*new_node));
    new_node->number = n;
    new_node->next = NULL;

    // Initialize the conductor to point to the first element.
    my_node_t *conductor = *root;

    if (*root == NULL) { // This is a special case where the list is empty
        *root = new_node;
    }
    else {
        while (conductor->next != NULL) {
            conductor = conductor->next;
        }
        conductor->next = new_node;
    }
}

//----------------------------------------------------------------------------
int count(my_node_t *root) {

    // Returns the length of a simply linked list with a given root.

    int k;
    my_node_t *conductor;

    k = 0;
    conductor = root;
    while (conductor != NULL) {
        conductor = conductor->next;
        k++;
    }
    return k;
}


//----------------------------------------------------------------------------
void copy_list(my_node_t *root, int *a) {

    // Copy the content of a linked list into an array a.

    int k;
    my_node_t *conductor;

    k = 0;
    conductor = root;
    while (conductor != NULL) {
        a[k] = conductor->number;
        conductor = conductor->next;
        k++;
    }
}

//----------------------------------------------------------------------------
void free_list(my_node_t *root) {

    // Releases the memory used by a simply linked list a given root.

    my_node_t *next;

    while (root != NULL) {
        next = root->next;
        free(root);
        root = next;
    }
}
