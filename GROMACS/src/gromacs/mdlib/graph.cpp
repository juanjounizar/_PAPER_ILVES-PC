//------------------------------------------------------------------------------
//-- DUMMY STATEMENT 80 CHARACTERS LONG TO ENSURE THE LINES ARE NOT TOO LONG  --
//------------------------------------------------------------------------------

/**
 * @file graph.c
 * @brief Functions related to undirected graphs
 *
 * @details This module contains functions for manipulating undirected graphs.
 * In particular, there it contains functions for computing the fill-in during
 * a partial or full LU decomposition as well as variants of the minimal degree
 * reordering algorithm.
 *
 * @author Carl Christian Kjelgaard Mikkelsen
 * @version 1.1.0
 * @date 2021-10-6
 * @warning The use of this software is at your own risk
 * @copyright GNU Public Licence
 *
 */

// Libries written primarily for the MD project
#include "graph.h"

#include "list.h"

// Standard libries
#include <stdio.h>
#include <stdlib.h>

#include <vector>
#include <list>
#include <queue>

// Third party libraries
#include <metis.h>

/**
 * Returns true if in the graph GRAPH there is a path from node a to node b
 * through strictly lower-numbered nodes. Returns false otherwise.
 *
 * @param graph A graph.
 * @param a The number of a node in GRAPH.
 * @param b The number of another node in GRAPH.
 * @return True if in the graph GRAPH there is a path from node a to node b
 * through strictly lower-numbered nodes. False otherwise.
 */
// static bool lower_path_ab(graph_t *graph, const int a, const int b) {
//     if (a == b) {
//         return true;
//     }

//     // Mark all the vertices as not visited
//     std::vector<bool> visited(graph->m, false);

//     // Create a queue for BFS
//     std::queue<int> queue;

//     // Mark the current node as visited and enqueue it
//     visited[a] = true;
//     queue.push(a);

//     while (!queue.empty()) {
//         // Dequeue a vertex from queue.
//         int v = queue.front();
//         queue.pop();

//         // Get all adjacent vertices of the dequeued vertex s
//         // If a adjacent has not been visited, then mark it visited
//         // and enqueue it
//         for (int idx = graph->xadj[v]; idx < graph->xadj[v + 1]; ++idx) {
//             int i = graph->adj[idx];
//             // If this adjacent node is the destination node, then
//             // return true
//             if (i == b) {
//                 return true;
//             }

//             // Else, continue to do BFS
//             if (!visited[i]) {
//                 visited[i] = true;

//                 // But only if the index of vertex i is < the index of vertex dst.
//                 if (i < a && i < b) {
//                     queue.push(i);
//                 }
//             }
//         }
//     }

//     // If BFS is complete without visiting d
//     return false;
// }

void graph_init(graph_t *graph) {
    /**
     * Initializes all pointers in the data structure to NULL
     *
     * @param[in] graph a datastructure representing an undirected graph
     */
    // Nullify pointer to adjacency lists
    graph->adj = NULL;
    // Nullify pointer to row indices
    graph->xadj = NULL;

}

void graph_free(graph_t *graph) {
    /**
     * Releases the heap memory used by all members of the data structure
     *
     * @param[in] graph a datastructure representing an undirected graph
     */
    // Releases the memory used to store a graph.
    free(graph->xadj);
    free(graph->adj);
}

//----------------------------------------------------------------------------
void graph_renumber_vertices(graph_t *graph, int *p) {

    /* RENUMBER VERTICES

       Renumbers the vertices of a graph using a given permutation p.
       The permutation sigma is given as in MATLAB

          p = (tau(0), tau(1), tau(2), ...., tau(m-1))

       where tau is the inverse of sigma, i.e. sigma(tau(j)) = j.

       It is necessary to compute the inverse permutation

          q = (sigma(0), sigma(1), sigma(2), ...., sigma(m-1))

       in order to faciliate the transformation.

       REMARK: The MATLAB notation can be annoying; live with it, as there are
       cases where one is preferable to the other and in this case it is simple
       to use both.

    */

    //--------------------------------------------------------------------------
    // Declaration of internal variables
    // -------------------------------------------------------------------------

    // The number of vertices
    int m;

    // Standard counter(s)
    int i, k;

    // Linked list representation of the renumbered graph
    my_node_t **root;

    // The inverse permutation
    int *q;

    // -------------------------------------------------------------------------
    // Start of instructions
    // -------------------------------------------------------------------------

    // Extract the number of vertices in the graph
    m = graph->m;

    // Initialize the linked list representation of the renumbered graph
    root = (my_node_t **)malloc(m * sizeof(graph_t *));
    for (i = 0; i < m; i++) {
        root[i] = NULL;
    }

    // Allocate space for and compute the inverse permutation
    q = (int *)malloc(m * sizeof(int));
    for (i = 0; i < m; i++) {
        q[p[i]] = i;
    }

    // Loop over the adjacency lists and apply the permutation
    for (i = 0; i < m; i++) {
        // Process the ith list
        for (k = graph->xadj[i]; k < graph->xadj[i + 1]; k++) {
            sinsert(q[graph->adj[k]], &root[q[i]]);
        }
    }

    // Release the current graph
    graph_free(graph);

    // Compress the reordered graph into the source
    graph_list_to_compact(m, root, graph);

    free(q);

    // Free the linked lists
    for (i = 0; i < m; i++) {
        free_list(root[i]);
    }
    free(root);

}

//----------------------------------------------------------------------------
void graph_print(graph_t *graph) {

    // Print the compact representation of the graph nicely

    // Standard counters
    int i, k;

    // The number of vertices
    int m;

    // Extract the number of vertices
    m = graph->m;

    // Print the number of vertices
    printf("PRINT_GRAPH:\n");
    printf("  Number of vertices : %d\n", m);
    printf("  Number of edges    : %d\n", graph->xadj[m]);
    printf("  Adjacency lists\n");
    // Loop over the adjacency lists
    for (i = 0; i < m; i++) {
        printf("    %3d -> ", i);
        // Loop over the entries of the ith adjancy list
        for (k = graph->xadj[i]; k < graph->xadj[i + 1]; k++) {
            printf("%3d, ", graph->adj[k]);
        }
        printf("\n");
    }
    printf("\n");
}

void graph_minimal_degree(graph_t *src, int *p) {

    /*

      ALGORITHM

      1: Set Gamma = {0,1,...,m-1} (linked list)
      2: FOR i=0,..,m-1 DO
      3:   Let alpha in Gamma have minimum degree, i.e.

             deg(alpha) = min{deg(gamma)| gamma in Gamma}

      4:   DELETE alpha from alpha's list

      5:   FOR beta IN alpha's list DO
      6:     DELETE alpha from beta's list
      7:   END

      8:   FORM a clique consisting of all nodes in alpha's list

      9:   DELETE alpha from Gamma

     10: END

    */

    //--------------------------------------------------------------------------
    // Declaration of internal variables
    //--------------------------------------------------------------------------

    // Standard counters
    int i;

    // The number of nodes
    int m;

    // The degree of the nodes
    int *deg;

    // Pointers to nodes
    my_node_t **root, *head, *current, *Gamma;

    // A node whose degree is minimal
    int alpha;

    //--------------------------------------------------------------------------
    // Start of instructions
    //--------------------------------------------------------------------------

    // Extract the number of nodes
    m = src->m;

    /* Initialize the list of active nodes

       REMARK: Notice that this need NOT be all available nodes !!!
       Suppose, that we only want a partial factorization and that we want
       nodes a, b, and c to be numbered last. Then we do not add a, b, and c
       to the list and write a, b, and c to the last three entries of the
       permutation.
    */

    /* Build the list of "active" nodes, i.e. nodes which are considered for
       elimination. This list should be shorter if we only want a partial
       factorization */
    Gamma = NULL;
    for (i = 0; i < m; i++) {
        sinsert(i, &Gamma);
    }

    // Allocate space to store the degree of the nodes
    deg = (int *)malloc(m * sizeof(int));

    // Compute the initial degree of the nodes
    for (i = 0; i < m; i++) {
        deg[i] = src->xadj[i + 1] - src->xadj[i];
    }

    //--------------------------------------------------------------------------
    // Build a linked list representation of the graph
    //--------------------------------------------------------------------------

    // Allocate space for the linked list representation of the graph
    root = (my_node_t **)malloc(m * sizeof(my_node_t *));

    // Initialize the individual linked list
    for (i = 0; i < m; i++) {
        root[i] = NULL;
    }

    // Convert the compact representation into linked lists
    graph_compact_to_list(src, root);

    //--------------------------------------------------------------------------
    // Main loop begins here
    //-------------------------------------------------------------------------

    // Initialize the counter.
    i = 0;

    // Loop over the list of active elements until it is empty
    while (Gamma != NULL) {
        // Determine a node alpha which has minimal degree
        alpha = Gamma->number;
        current = Gamma;
        while (current != NULL) {
            if (deg[current->number] < deg[alpha]) {
                alpha = current->number;
            }
            current = current->next;
        }

        // Update the permutation
        p[i] = alpha;

        // Delete alpha from alpha's list
        sdelete(alpha, &root[alpha]);

        // For each beta in alpha's list, delete alpha from beta's list
        current = root[alpha];
        while (current != NULL) {
            deg[current->number] -= sdelete(alpha, &root[current->number]);
            current = current->next;
        }

        // Form a clique of consisting of all members in alpha's list
        head = root[alpha];
        while (head != NULL) {
            current = root[alpha];
            while (current != NULL) {
                deg[current->number] += sinsert(head->number, &root[current->number]);
                current = current->next;
            }
            head = head->next;
        }

        // Delete alpha from the active list
        sdelete(alpha, &Gamma);

        // Increment i in preparation for the next loop.
        i++;
    }

    // Release the adjacency lists
    for (i = 0; i < m; i++) {
        free_list(root[i]);
    }

    // Release the main pointer to the linked list representation of the graph
    free(root);

    // Free the active list
    free_list(Gamma);

    // Release memory for the degrees
    free(deg);
}

//----------------------------------------------------------------------------
void graph_tril(graph_t *src, graph_t *dest) {

    /* Builds the DEST graph from the SRC graph by retaining an edge from vertex
       i to vertex j if and only if i>=j.

       This is equivalent to taking the upper triangular part of the adjacency
       matrix

       DESCRIPTION OF INTERFACE

       ON ENTRY:
        src       a pointer to a compact representation of the source
        dest      a pointer to structure which can hold the result

       ON EXIT:
         src      unchange
         dest     has been updated with the "upper triangular" graph

    */

    //--------------------------------------------------------------------------
    // Declaration of internal variables
    //--------------------------------------------------------------------------

    // Standard counters
    int j, k;

    // The number of vertices in the source/destination
    int m;

    // Pointer into the destination graph
    int d;

    //--------------------------------------------------------------------------
    // Instructions start here
    //--------------------------------------------------------------------------

    /* We pass over the SRC graph twice.
       The first time we count the edges that we want to retain. Then we
       allocate space for the DEST graph. Finally, we copy the desired edges.
    */

    // Extract the number of vertices
    m = src->m;

    // Initialize the number of edges to retain
    d = 0;

    // Loop over the vertices of the source graph
    for (j = 0; j < m; j++) {
        // Loop over the jth adjacency list
        for (k = src->xadj[j]; k < src->xadj[j + 1]; k++) {
            // Retain the current edge?
            if (j <= src->adj[k]) {
                d++;
            }
        }
    }
    // At this point we know that the DEST graph will contain exactly d edges!

    // Set the number of vertices in the DEST
    dest->m = m;

    // Allocate space for the DEST graph
    dest->xadj = (int *)malloc((m + 1) * sizeof(int));
    dest->adj = (int *)malloc(d * sizeof(int));

    // Reset the pointer into dest->adj
    d = 0;
    dest->xadj[0] = d;

    // Loop over the vertices of the source graph
    for (j = 0; j < m; j++) {
        // Loop over the jth adjacency list
        for (k = src->xadj[j]; k < src->xadj[j + 1]; k++) {
            // Retain the current edge?
            if (j >= src->adj[k]) {
                // Copy the edge
                dest->adj[d] = src->adj[k];
                // Move to the next position of dest->adj
                d++;
            }
        }
        // Record the new starting index for the (j+1)st adjacency list
        dest->xadj[j + 1] = d;
    }
    // This completes the construction of DEST.
    // Please note that dest->xadj[m] contains the correct value :)
}

// void untril(graph_t *src, graph_t *dest) {

//     /*
//         TODO

//     */

//     my_node_t **root = (my_node_t **)malloc(src->m * sizeof(my_node_t *));

//     // Convert the compact representation into linked lists
//     compact_to_list(src, root);

//     for (int row = 0; row < src->m; ++row) {
//         my_node_t *iter = root[row];

//         while (iter != NULL) {
//             auto col = iter->number;

//             sinsert(row, &root[col]);

//             iter = iter->next;
//         }
//     }

//     // Compress the linked lists into a graph
//     list_to_compact(src->m, root, dest);

//     // First the m individual lists
//     for (int i = 0; i < src->m; ++i) {
//         free_list(root[i]);
//     }

//     // The the root pointer itself
//     free(root);
// }

#if 0
//----------------------------------------------------------------------------
void graph_stril(graph_t *src, graph_t *row, graph_t *column) {

    /* STRIL = Strictly TRIangular (Lower)

       Extracts both a row and a column oriented representation of the strictly
       lower triangular portion of the graph G = (V,E).

       If (i,j) in E with i > j, then

         1) i -> j   in the row oriented representation
         2) j -> i   in the column oriented representation

       When computing a sparse Cholesky factorization of a matrix in CSC format
       you need the column oriented representation to access the columns. If
       the factorization is left looking, then you also need the row oriented
       representation.


    */

    //--------------------------------------------------------------------------
    // Declaration of internal variables
    //--------------------------------------------------------------------------

    // Pointer to the linear list representation of the rows
    my_node_t **rroot;

    // Pointer to the linear list representation of the columns
    my_node_t **croot;

    // Standard counters
    int i, j, k;

    // The number of vertices in the input graph
    int m;

    //--------------------------------------------------------------------------
    // Instructions start here
    //--------------------------------------------------------------------------

    // Extract the number of vertices in the input graph
    m = src->m;

    // Allocate space for the two different representations
    rroot = (my_node_t **)malloc(m * sizeof(my_node_t *));
    croot = (my_node_t **)malloc(m * sizeof(my_node_t *));

    // Initialize the individual lists
    for (i = 0; i < m; i++) {
        rroot[i] = NULL;
        croot[i] = NULL;
    }

    // Loop over the elements of the original graph
    for (j = 0; j < m; j++) {
        for (k = src->xadj[j]; k < src->xadj[j + 1]; k++) {
            // Extract the row index
            i = src->adj[k];
            // Are we dealing with a strictly subdiagonal entry
            if (i > j) { // Yes, this is a subdiagonal entry
                // Insert j into the ith adjacency list for the rows
                sinsert(j, &rroot[i]);
                // Insert i into the jth adjacency list for the columns
                sinsert(i, &croot[j]);
            }
        }
    }

    // This completes the construction of the linked list representations

    // Compress the linked list in to a graph structure
    graph_list_to_compact(m, rroot, row);
    graph_list_to_compact(m, croot, column);

    // Free the individual lists
    for (i = 0; i < m; i++) {
        free_list(rroot[i]);
        free_list(croot[i]);
    }

    // Free the pointers to the list of lists
    free(rroot);
    free(croot);
}
#endif

// void compute_fill(graph_t *src, graph_t *dest) {

// /* Given the adjacency graph of a nonsingular lower triangular matrix,
//    this subroutine computes the graph of the Cholesky factor.

//    The routine plays the elimination game on the source graph:

//    1) The nodes are removed in sequential order
//    2) When node i is removed, we make a clique of the neighbors.

// */

// // -------------------------------------------------------------------------
// // Declaration of internal variables
// // -------------------------------------------------------------------------

// // Pointer to the linked list representation of the source graph
// my_node_t **root;

// // Pointers to elements in the lists
// my_node_t *conductor, *apprentice;

// // The number of vertices
// int m;

// // Standard counter
// int i;

// // -------------------------------------------------------------------------
// // Start of instructions
// // -------------------------------------------------------------------------

// // Extract the number of vertices from the graph
// m = src->m;

// // Allocate space for the variable root
// root = (my_node_t **)malloc(m * sizeof(my_node_t *));

// // First expand the graph into a set of linked lists
// compact_to_list(src, root);

// // Loop over the adjacency lists
// for (i = 0; i < m; i++) {
//     // Start at the root of the ith adjancency list
//     conductor = root[i];
//     // Go over the ith adjacency list one element at a time
//     while (conductor != NULL) {
//         // We are only interest in higher indices
//         if (conductor->number > i) {
//             // The apprentice starts with the conductor
//             apprentice = conductor;
//             // Let the apprentice run to the end of the list
//             while (apprentice != NULL) {
//                 // We are only interested in higher indices
//                 if (apprentice->number > conductor->number) {
//                     // Make insertion into list number conductor->number > i
//                     sinsert(apprentice->number, &root[conductor->number]);
//                     // Please observe that we are NOT changing the ith list !!!
//                 }
//                 apprentice = apprentice->next;
//             }
//         }
//         conductor = conductor->next;
//     }
// }
// // Compress the linked lists into a graph
// list_to_compact(m, root, dest);

// //--------------------------------------------------------------------------
// // Release the dynamically allocated memory
// //--------------------------------------------------------------------------

// // First the m individual lists
// for (i = 0; i < m; i++) {
//     free_list(root[i]);
// }

// // The the root pointer itself
// free(root);
// }

void graph_fill(int n, graph_t *src, graph_t *dst, int **imap) {
    /**
     * Computes a symbolic partial LU decomposition of the form
     \f[
     \begin{bmatrix}
     A_{11} & A_{12} \\
     A_{21} & A_{22}
     \end{bmatrix} =
     \begin{bmatrix}
     L_{11} &  \\
     L_{21} & I
     \end{bmatrix}
     \begin{bmatrix}
     U_{11} & U_{12} \\
     & U_{22}
     \end{bmatrix}
     .
     \f]
     * The map is a list of integers. If map[k]=-1, then the kth entry of fill(A)
     * should be initialized as zero. If map[k]=l, then the kth entry of fill(A)
     * should be initialized ast the lth nonzero value of A.
     *
     * @param[in] n the dimension of the block \f$L_{11}\f$
     * @param[in] src the adjacency graph for the matrix A
     * @param[in] dst the adjacency graph for the matrix fill(A)
     * @param[inout] imap a map that describes how to initializes matrix fill(A)
     */

    // Compute the fill when eliminating the first n <= m variables.

    // Extract the number of vertices from the source graph
    int m = src->m;

    // ***************************************************************************
    //     Play the elimination game
    // ***************************************************************************

    // List of lists representation of the source graph
    my_node_t **root1 = (my_node_t **)malloc(m * sizeof(my_node_t *));
    my_node_t **root2 = (my_node_t **)malloc(m * sizeof(my_node_t *));

    // Expand the source graph into the list of lists
    graph_compact_to_list(src, root1);
    graph_compact_to_list(src, root2);

    for (int i = 0; i < n; i++) {
        // Eliminate the ith variable from the ith list
        sdelete(i, &root1[i]);
        // Delete the ith variable from any remaining lists
        my_node_t *current = root1[i];
        while (current != NULL) {
            sdelete(i, &root1[current->number]);
            current = current->next;
        }
        // Make a clique the of the neighbors
        current = root1[i];
        while (current != NULL) {
            my_node_t *aux = root1[i];
            while (aux != NULL) {
                sinsert(current->number, &root1[aux->number]);
                sinsert(current->number, &root2[aux->number]);
                aux = aux->next;
            }
            current = current->next;
        }
    }
    // Compress the list of lists into the destination
    graph_list_to_compact(m, root2, dst);

    // ***************************************************************************
    //   Map the origin of all entries in fill-in graph/matrix
    // ***************************************************************************

    // Isolate the number of nonzero elements in the destination
    int nnz = dst->xadj[m];

    // Allocate space for the map
    *imap = (int *)malloc(nnz * sizeof(int));

    // Auxiliary work array
    int *iwork = (int *)malloc(m * sizeof(int));
    for (int i = 0; i < m; i++) {
        iwork[i] = -1;
    }

    for (int i = 0; i < m; i++) {
        // Expand ith row of src->adj into iwork
        for (int k = src->xadj[i]; k < src->xadj[i + 1]; k++) {
            iwork[src->adj[k]] = k;
        }
        // Compress iwork into (*imap) using dst->adj
        for (int k = dst->xadj[i]; k < dst->xadj[i + 1]; k++) {
            (*imap)[k] = iwork[dst->adj[k]];
        }
        // Reset iwork
        for (int k = src->xadj[i]; k < src->xadj[i + 1]; k++) {
            iwork[src->adj[k]] = -1;
        }
    }

    // Free memory
    for (int i = 0; i < m; i++) {
        free_list(root1[i]);
        free_list(root2[i]);
    }
    free(root1);
    free(root2);
    free(iwork);
}

// ----------------------------------------------------------------------------
void graph_compute_fill(graph_t *src, graph_t *dest) {
    // Extract the number of vertices from the graph
    int m = src->m;

    // Pointer to the linked list representation of the source graph
    my_node_t **root = (my_node_t **)malloc(m * sizeof(my_node_t *));

    // First expand the graph into a set of linked lists
    graph_compact_to_list(src, root);

    // Loop over the adjacency lists
    for (int i = 0; i < m; i++) {
        // Start at the root of the ith adjancency list
        my_node_t *conductor = root[i];
        // Go over the ith adjacency list one element at a time
        while (conductor != NULL) {
            // We are only interest in higher indices
            if (conductor->number > i) {
                // The apprentice starts at the root of the ith adjancency list
                my_node_t *apprentice = root[i];
                // Let the apprentice run to the end of the list
                while (apprentice != NULL) {
                    // We are only interested in higher indices
                    if (apprentice->number > i) {
                        // Make insertion into list number conductor->number > i
                        sinsert(apprentice->number, &root[conductor->number]);
                        // Please observe that we are NOT changing the ith list !!!
                    }
                    apprentice = apprentice->next;
                }
            }
            conductor = conductor->next;
        }
    }

    // Compress the linked lists into a graph
    graph_list_to_compact(m, root, dest);

    //--------------------------------------------------------------------------
    // Release the dynamically allocated memory
    //--------------------------------------------------------------------------

    // First the m individual lists
    for (int i = 0; i < m; i++) {
        free_list(root[i]);
    }

    // The the root pointer itself
    free(root);
}

//----------------------------------------------------------------------------
void graph_compute_fill_integer(graph_t *src, graph_t *dest, int **ival) {

    /* Given the adjacency graph of a nonsingular lower triangular matrix,
       this subroutine computes the graph of the Cholesky factor. */

    // -------------------------------------------------------------------------
    // Declaration of internal variables
    // -------------------------------------------------------------------------

    // The number of vertices
    int m;

    // Standard counters
    int i, k;

    // Internal work space
    int *iwork;

    // -------------------------------------------------------------------------
    // Start of instructions
    // -------------------------------------------------------------------------

    // Compute the fill-in
    graph_compute_fill(src, dest);

    /* Construct a list of integers IVAL, such that

       (*IVAL)[k]=1 if VAL[k] is should be treated as zero from the start
       (*IVAL)[k]=0 if VAL[k] filled in during the factorization
    */

    // Extract the number of vertices from the graph
    m = src->m;

    // Allocate space for *IVAL
    (*ival) = (int *)malloc((dest->xadj[m]) * sizeof(int));

    // Allocate internal work space
    iwork = (int *)malloc(m * sizeof(int));

    // Initialize the array *IVAL to zero
    for (k = 0; k < dest->xadj[m]; k++) {
        (*ival)[k] = 0;
    }

    // Initialize the auxiliary workspace
    for (i = 0; i < m; i++) {
        iwork[i] = 0;
    }

    // Loop over the adjancency lists
    for (i = 0; i < m; i++) {
        // Process the ith adjancency list of the source graph
        for (k = src->xadj[i]; k < src->xadj[i + 1]; k++) {
            // Make a mark in IWORK
            iwork[src->adj[k]] = 1;
        }
        /* Pull the entries of IWORK into (*IVAL) which correspond to the ith
           adjacency list of the destination graph */
        for (k = dest->xadj[i]; k < dest->xadj[i + 1]; k++) {
            (*ival)[k] = iwork[dest->adj[k]];
        }
        // Clear IWORK
        for (k = src->xadj[i]; k < src->xadj[i + 1]; k++) {
            // Remove the mark in IWORK.
            iwork[src->adj[k]] = 0;
        }
    }

    // Free internal work space
    free(iwork);

}

//----------------------------------------------------------------------------
void graph_compute_fill_information(graph_t *src, graph_t *dst, int **ival,
                                    int **map) {

    /* This is a function which gathers even more information about the fill
       than compute_fill_integer. Specifically, it constructs a map of where
       the original nonzeros should be placed inside the extended array which
       is exactly large enough to handle the fill in. Specifically, if the
       k nonzero element should be store in location map[k] in the extended
       array. This information allows the generators to write a make_matrix
       routine which hardwires the location of the structural nonzeros into
       the code. */

    // -------------------------------------------------------------------------
    // Declaration of internal variables
    // -------------------------------------------------------------------------

    // Standard counters
    int i, k;

    // Number of vertices
    int m;

    // Total length of the adjacency lists for the two graphs
    int nnz1, nnz2;

    // Auxiliary arrays, deallocated when returning
    int *iwork, *iaux;

    // -------------------------------------------------------------------------
    // Start of instructions
    // -------------------------------------------------------------------------

    // Obtain all the previous information
    graph_compute_fill_integer(src, dst, ival);

    // Determine the number of vertices
    m = src->m;

    // Now determine the total length of the adjacency list
    nnz1 = src->xadj[m];
    nnz2 = dst->xadj[m];

    // Allocate space for auxiliary array used to construct the map
    iaux = (int *)malloc(nnz2 * sizeof(int));

    // Fill the array with any NEGATIVE marker
    for (i = 0; i < nnz2; i++) {
        iaux[i] = -1;
    }

    // Allocate space for the map
    *map = (int *)malloc(nnz1 * sizeof(int));

    // Initialize the map
    for (i = 0; i < nnz1; i++) {
        (*map)[i] = i;
    }

    // Allocate and initialize temporary workspace
    iwork = (int *)malloc(m * sizeof(int));
    for (i = 0; i < m; i++) {
        iwork[i] = -1;
    }

    /* Now pull the array (*map) into the auxiliary array iaux.
       This is exactly the same technique used to pull one sparse vector into
       another sparse vector with a more extensive sparsity pattern.  */

    /* Loop over the adjacency lists of the source */
    for (i = 0; i < m; i++) {
        // Process the ith adjacency list of the source
        for (k = src->xadj[i]; k < src->xadj[i + 1]; k++) {
            iwork[src->adj[k]] = k;
        }
        // Pull the entries of iwork into iaux
        for (k = dst->xadj[i]; k < dst->xadj[i + 1]; k++) {
            iaux[k] = iwork[dst->adj[k]];
        }
        // Clear the entries of iwork
        for (k = src->xadj[i]; k < src->xadj[i + 1]; k++) {
            iwork[src->adj[k]] = -1;
        }
    }

    /* At this point iaux contains the integers 0, 1, 2, ..., (nnz1-1) with
       a copies of -1 marking the location of the fill. */

    // Construct the map by doing a single pass over iaux!
    for (i = 0; i < nnz2; i++) {
        if (iaux[i] != -1) {
            // You have found the location of structural nonzero entry number iaux!
            (*map)[iaux[i]] = i;
            // Remember that iaux[i] is between 0 and nnz1-1!
        }
    }
    // Release memory
    free(iwork);
    free(iaux);
}




//----------------------------------------------------------------------------
void graph_compact_to_list(graph_t *src, my_node_t **root) {

    /* Generates a linked list representation of a graph.

       It is the responsibility of the caller to allocate space for the
       pointers to the head of the lists.

     */

    // Standard counters
    int i, j, k;

    // The number of vertices
    int m = src->m;

    // Initialize the linked lists
    for (i = 0; i < m; i++) {
        root[i] = NULL;
    }

    int count = 0;
    int real_count = 0;
    // Loop over the vertices
    for (i = 0; i < m; i++) {
        // Loop over the elements in the ith adjacency list
        for (k = src->xadj[i]; k < src->xadj[i + 1]; k++) {
            // Which vertex j are we looking at?
            j = src->adj[k];
            // Insert j into the ith list
            real_count += sinsert(j, &root[i]);
            ++count;
        }
    }
}

void graph_list_to_compact(int m, my_node_t **root, graph_t *dest) {

    /* Generates a compact representation of a graph from the linked lists

       DESCRIPTION OF INTERFACE

       ON ENTRY:

       m           the number of vertices/adjacency lists
       root[i]     pointer to the head of the ith list
       dest        pointer to destination datastructure

       ON EXIT
       dest->m     equal to the number of vertices
       dest->xadj
       dest->adj
     */

    // -------------------------------------------------------------------------
    // Declaration of internal variables
    // -------------------------------------------------------------------------

    // Standard counters
    int i;

    // Total length of the adjacency lists AND pointer into dest->adj.
    int length = 0;

    // -------------------------------------------------------------------------
    // Start of instructions
    // -------------------------------------------------------------------------

    // Loop over the lists and compute their total length
    for (i = 0; i < m; i++) {
        length += count(root[i]);
    }

    // Set the number of vertices in the dest
    dest->m = m;

    // Allocate space for dest->xadj;
    dest->xadj = (int *)malloc((m + 1) * sizeof(int));

    // Allocate space for dest->adj
    dest->adj = (int *)malloc(length * sizeof(int));

    // Reset the length/index into dest->xadj
    length = 0;
    dest->xadj[0] = 0;

    // Loop over the lists
    for (i = 0; i < m; i++) {
        // Copy the ith list into dest->adj
        copy_list(root[i], &(dest->adj[length]));
        // Update length
        length += count(root[i]);
        // Fill in the value of dest->xadj[i+1]
        dest->xadj[i + 1] = length;
    }
}

void graph_diag(graph_t *graph, int **diag) {

    /**
     * Find the indices of all diagonal entries of a sparse matrix
     *
     * @param[in] graph the adjacency graph of a square matrix of dimension m
     * @param[out] diag a pointer to an array of length m
     *
     * The function allocates an array of length m and sets the entries
     */

    // Isolate the number of vertices
    int m = graph->m;

    // Allocate space for the index pointer
    (*diag) = (int *)malloc(m * sizeof(int));

    // Loop over the rows
    for (int row = 0; row < m; row++) {
        // Assume failure
        (*diag)[row] = -1;
        // Scan for the diagonal entry
        for (int k = graph->xadj[row]; graph->xadj[row + 1]; k++) {
            // Isolate the column index
            int col = graph->adj[k];
            // Test for diagonal entry
            if (row == col) {
                // Record the location
                (*diag)[row] = k;
                // Break from the for loop
                break;
            }
        }
    }
}
