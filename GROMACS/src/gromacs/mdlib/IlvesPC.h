#pragma once

#include <inttypes.h>
#include <semaphore.h>

#include "gromacs/gmxlib/nrnb.h"
#include "gromacs/pbcutil/pbc.h"
#include "gromacs/utility/real.h"
#include "memory.h"
#include "molecule.h"

/**
 * @author Lorién López-Villellas (lorien.lopez@unizar.es)
 *
 * A parallel and peptide-chains-specific implementation of ILVES.
 *
 */

namespace gmx {

class IlvesPC {
public:
    /**
     * Initializes the ILVES solver.
     *
     * @param mol Molecule structure.
     * @param threads Number of threads to use when solve() is called.
     */
    IlvesPC(molecule_t *const mol, const int threads);

    /**
     * Destroys the dinamically allocated data.
     *
     */
    ~IlvesPC();

    /**
     * Try to solve the bond constraints of MOL in at most MAXIT iterations of
     * Newton's method with a tolerance for each atom of TOL.
     *
     * @param x Positions of the atoms before computing the external forces,
     * with the following format:
     *
     * atom 1     atom 2     atom 3
     * x, y, z,   x, y, z,   x, y, z
     *
     * @param xprime Positions of the atoms after computing the external forces,
     * with the same format as x. When returning, it will contain the final
     * position of each atom.
     * @param vprime The atoms velocities. When returning, it will contain the
     * final velocity of each atom.
     * @param numit When returning, NUMIT will contain the total number of
     * Newton's method iterations.
     * @param error When returning, ERROR will contain the final relative
     * constraint violation.
     * @param tol Maximum error allowed in each atom position.
     * @param maxit Maximum number of Newton's method iterations.
     * @param deltat The time-step in picoseconds.
     * @param constraint_virial Update the virial?
     * @param virial sum r x m delta_r (virial)
     * @param constraint_velocities Update the velocities?
     * @param pbc The PBC (container) information. Null if there is no PBC
     * information.
     * @param compute_fep Compute the free energy?
     * @param fep_lambda FEP lambda. TODO: WHAT IS THIS.
     * @param fep_force The FEP force.
     * @param perf_stats Performance statistics.
     * @return True if the constraints has been satisfied, false otherwise.
     */
    bool solve(const ArrayRef<const RVec> x,
               const ArrayRef<RVec> xprime,
               const ArrayRef<RVec> vprime,
               int &numit,
               real &error,
               const real tol,
               const int maxit,
               const real deltat,
               const bool constraint_virial,
               tensor virial,
               const bool constraint_velocities,
               const t_pbc *const pbc,
               const bool compute_fep,
               const real fep_lambda,
               real &fep_force,
               t_nrnb *const perf_stats);

private:
    molecule_t *mol; // The molecule structure.

    // Sparse independent-term matrix.
    struct g_t {
        size_t max_size;   // Number allocated entries of the matrix.
        size_t size;       // Number of populated entries of the matrix.

        real *data;           // Entries of the matrix.
        real *current_lagr;   // Current approximation of the lagrange
                              // multipliers.

        // Atoms that take part in each entry.
        int *a;
        int *b;

        // Array of sigma2.
        real *sigma2;

        // Matrix row of the first entry.
        int first_row;
    };

    // Sparse coefficient matrix.
    struct A_t {
        size_t max_size;   // Number allocated entries of the matrix.
        size_t size;       // Number of populated entries of the matrix.=

        real *data;   // Entries of the matrix.

        // Weight of A(x,y).
        real *weight;

        // Matrix row/col of each entry.
        int *row;
        int *col;
    };

    // Struct for private data (one per thread).
    struct private_data_t {
        int num_side_chains;    // Number of side-chains assigned to thread.
        int first_side_chain;   // First side-chain index assigned to thread.

        int first_sc_bond;   // First bond that belongs to a side-chain assigned
                             // to thread.
        int first_bb_bond;   // First bond that belongs to the backbone assigned
                             // to thread.
        int first_sep_bond;   // First bond that belongs to the separator
                              // assigned to thread.

        int sc_num_bonds;   // Number of side-chain bonds assigned to thread.
        int bb_num_bonds;   // Number of backbone bonds assigned to thread.
        // int sep_num_bonds = num_side_chains.
        // Total number of bonds assigned to thread.
        // = sc_num_bonds + bb_num_bonds + (2 if first thread (special))
        int num_bonds;

        A_t sc_to_sc_matrix;
        A_t sc_to_bb_matrix;
        A_t sc_to_sep_matrix;   // Only proline.
        g_t sc_g_matrix;

        A_t sc_to_sc_fillins;
        A_t sc_to_bb_fillins;   // Only proline

        A_t bb_to_sc_matrix;
        A_t bb_to_bb_matrix;
        A_t bb_to_sep_matrix;
        g_t bb_g_matrix;

        A_t bb_to_sc_fillins;   // Only proline.
        A_t bb_to_sep_fillins;

        A_t sep_to_sc_matrix;   // Only proline.
        A_t sep_to_bb_matrix;
        A_t sep_to_sep_matrix;
        g_t sep_g_matrix;

        A_t sep_to_bb_fillins;   // Only proline.
        A_t sep_to_sep_fillins;
    };

    // Struct for special atoms.
    struct special_data_t {
        A_t all_to_spl_matrix;
        A_t spl_to_all_matrix;
        g_t spl_g_matrix;

        A_t all_to_spr_matrix;
        A_t spr_to_all_matrix;
        g_t spr_g_matrix;
    };

    // Struct for shared data among threads (NTHREADS - 1 instances).
    struct shared_data_t {
        // Shared Schur row (separation row).
        A_t sep_to_sc_matrix;
        A_t sep_to_bb_matrix;
        A_t sep_to_sep_matrix;
        g_t sep_g_matrix;

        A_t sep_to_bb_fillins;
        A_t sep_to_sep_fillins;

        // Mutex to protect data.
        pthread_mutex_t lock;

        // Avoid false sharing of lock.
        // const uint8_t padding[MEM_BLOCK_SIZE];
    };

    // Pair of semaphores with false sharing prevention.
    struct safe_sem_pair_t {
        sem_t left;
        sem_t right;

        // Avoid false sharing.
        // const uint8_t padding[MEM_BLOCK_SIZE];
    };

    /*
     * Global variables that store all data needed for executing kernel().
     */

    private_data_t *private_data;

    shared_data_t *shared_data;

    special_data_t *special_data;

    // To avoid using a barrier after update_positions() (and
    // update_velocities).
    safe_sem_pair_t *boundary_sems;

    // x_ab[i][XX/YY/ZZ] contains the vector from atom a to b (using x),
    // the atoms that are part of the ith bond.
    std::vector<RVec> x_ab;
    // x_ab[i][XX/YY/ZZ] contains the vector from atom a to b (using xprime),
    // the atoms that are part of the ith bond.
    std::vector<RVec> xprime_ab;

    int nthreads;   // Number of threads to use.

    // Arrays of pointer to functions.

    static void (IlvesPC::*step_2_solvers[NUM_TYPES])(real *const,
                                                      real *const,
                                                      real *const,
                                                      real *const,
                                                      real *const,
                                                      real *const);

    static void (IlvesPC::*step_3_solvers[NUM_TYPES])(real *const,
                                                      real *const,
                                                      real *const,
                                                      real *const,
                                                      real *const,
                                                      real *const,
                                                      real *const,
                                                      real *const,
                                                      real *const,
                                                      real *const,
                                                      real *const);

    static void (IlvesPC::*step_16_solvers[NUM_TYPES])(real *const,
                                                       real *const,
                                                       real *const,
                                                       real *const);

    static void (IlvesPC::*step_17_solvers[NUM_TYPES])(real *const,
                                                       real *const,
                                                       real *const);

    // Forbidden operations:
    IlvesPC(const IlvesPC &other);
    IlvesPC(IlvesPC &&other) noexcept;
    IlvesPC &operator=(const IlvesPC &other);
    IlvesPC &operator=(IlvesPC &&other) noexcept;

    /*
     * g_t functions.
     */

    /**
     * Dynamically allocates each pointer of the Sparse matrix G.
     * The field G->SIZE must be already set in order to know how many entries
     * does G have.
     *
     * @param g Sparse independent-term matrix.
     */
    void init_g(g_t *const g);

    /**
     * Destroys G dinamically allocated data.
     *
     * @param g Sparse independent-term matrix.
     */
    void destroy_g(g_t *g);

    /*
     * A_t functions.
     */

    /**
     * Dynamically allocates each pointer of the Sparse matrix A.
     * The field A->MAX_SIZE must be already set in order to know how many
     * entries does A have.
     *
     * @param A Sparse coefficient matrix.
     * @param is_fillins If true, neither weight, row nor col will be allocated.
     */
    void init_A(A_t *const A, const bool is_fillins);

    /**
     * Destroys A dinamically allocated data.
     *
     * @param A Sparse coefficient matrix.
     */
    void destroy_A(A_t *A);

    /**
     * Adds an entry with row = ROW, col = COL and weight = WEIGHT to the end of
     * the matrix A. Increses A->size by 1.
     *
     * @param A Sparse coefficient matrix.
     * @param is_fillins If true, neither weight, row nor col will be allocated.
     */
    void A_push_back(const int row, const int col, const real weight, A_t *const A);

    /*
     * Initialization
     */

    /**
     * Populates G with MOL's data corresponding to bonds between
     * FIRST_BOND and FIRST_BOND + G->SIZE - 1. Also populates THREAD's As
     * corresponding to each bond between FIRST_BOND and FIRST_BOND + G->SIZE
     * - 1.
     *
     * @param thread Thread descriptor [0 - NTHREADS - 1].
     * @param first_bond First bond.
     * @param g Sparse coefficient matrix.
     */
    void initialize_matrices(const int thread, const int first_bond, g_t *const g);

    /**
     * Return the number of matrix entries (A + g) of a side chain of type TYPE.
     *
     * @param type Side-chain type.
     * @return the number of matrix entries (A + g) of a side chain of type
     * TYPE.
     */
    int get_num_entries_side_chain(const side_chain_type_t type);

    /**
     * Return the number of matrix entries (A + g) of the molecule MOL.
     *
     * @return the number of matrix entries (A + g) of the molecule MOL.
     */
    int get_num_entries_mol();

    /*
     * ILVES kernel.
     */

    /**
     * Computes the part of g (the right-hand side of the linear system) that
     * corresponds to the entries of G between FIRST_ELEMENT and (FIRST_ELEMENT
     * + NUM_ELEMENTS - 1). Also updates xprime_ab and x_ab (if compute_x_ab)
     * for each bond that corresponds to an entry of G between FIRST_ELEMENT and
     * (FIRST_ELEMENT + NUM_ELEMENTS - 1). Returns the largest relative (square)
     * bond length violation of the bonds that corresponds to an entry of G
     * between FIRST_ELEMENT and (FIRST_ELEMENT + NUM_ELEMENTS - 1).
     *
     * @param pbc The PBC (container) information. Null if there is no PBC
     * information.
     * @param first_element First element to compute.
     * @param num_elements Num elements to compute.
     * @param first_row Row of the of the first entry of g (g[0]). NOT the row
     * of FIRST_ELEMENT.
     * @param g Sparse independet-term matrix. When returning, G.DATA will
     * contain g(x).
     * @param x Initial atoms possitions.
     * @param xprime Current atoms positions.
     * @param compute_x_ab Compute the entries of x_ab between FIRST_ELEMENT *
     * DIM and (FIRST_ELEMENT * DIM + NUM_ELEMENTS * DIM - 1).
     * @return Largest relative error.
     */
    real make_g_loop(const t_pbc *const pbc,
                     const size_t first_element,
                     const size_t num_elements,
                     const int first_row,
                     g_t *const g,
                     const ArrayRef<const RVec> x,
                     const ArrayRef<const RVec> xprime,
                     const bool compute_x_ab);

    /**
     * Computes the part of g (the right-hand side of the linear system) that
     * corresponds to the entries of G.
     * Also updates xprime_ab and x_ab (if compute_x_ab) for each bond that
     * corresponds to an entry of G.
     * Returns the largest relative (square) bond length violation of the bonds
     * that corresponds to an entry of G.
     *
     * @param thread Thread descriptor [0 - NTHREADS - 1].
     * @param pbc The PBC (container) information. Null if there is no PBC
     * information.
     * @param g Sparse independet-term matrix. When returning, G.DATA will
     * contain g(x).
     * @param x Initial atoms possitions.
     * @param xprime Current atoms positions.
     * @param wait_start True if, before accessing to the first bond data (i ==
     * 0), the thread has to wait on its boundary-lef-semaphore. False
     * otherwise.
     * @param wait_end True if, before accessing to the last bond data (i ==
     * g->size - 1), the thread has to wait on its boundary-rigth-semaphore.
     * False otherwise.
     * @param compute_x_ab Compute the entries of x_ab assigned to THREAD.
     * @return Largest relative error.
     */
    real make_g_inm(const int thread,
                    const t_pbc *const pbc,
                    g_t *const __restrict__ g,
                    const ArrayRef<const RVec> x,
                    const ArrayRef<const RVec> xprime,
                    const bool wait_start,
                    const bool wait_end,
                    const bool compute_x_ab);

    /**
     * Constructs the part of g (the right-hand side of the linear system)
     * assigned to THREAD.
     * Also updates xprime_ab and x_ab (if compute_x_ab) for each bond assigned
     * to THREAD.
     * Returns the largest relative (square) bond length violation of the bonds
     * assigned to THREAD.
     *
     * @param thread Thread descriptor [0 - NTHREADS - 1].
     * @param pbc The PBC (container) information. Null if there is no PBC
     * information.
     * @param x Current atoms positions.
     * @param xprime Current atoms positions.
     * @param must_wait True if the function is called out of the main loop.
     * @param compute_x_ab Compute the entries of x_ab assigned to THREAD.
     * @return Largest relative error of the g(x) entries assigned to THREAD.
     */
    real make_g(const int thread,
                const t_pbc *const pbc,
                const ArrayRef<const RVec> x,
                const ArrayRef<const RVec> xprime,
                const bool must_wait,
                const bool compute_x_ab);

    /**
     * Constructs the part of A (the left-hand side of the linear system) that
     * corresponds to the entries of A.
     *
     * @param A Sparse coefficient matrix. When returning A.DATA will be
     * populated.
     */
    void make_A_loop(A_t *const __restrict__ A);

    /**
     * Constructs the part of A (the left-hand side of the linear system)
     * assigned to THREAD.
     *
     * @param thread Thread descriptor [0 - NTHREADS - 1].
     */
    void make_A(const int thread);

    /**
     * Cleans the subdiagonal entries of a spl matrix of ends type A.
     *
     * It is assumed that this function is called by thread 0.
     *
     * Initial matrix:
     *
     * spl           bb              sep      g
     * -----     -------------     -----
     * |x  | ... |x x        | ... |   | ... |x| sp    1
     * -----     -------------     -----
     *                 .
     *                 .
     *                 .
     * -----     -------------     -----
     * |x  |     |x x        |     |x  |     |x|       1
     * |x  | ... |x x x x x  | ... |x  |     |x|       2
     * -----     |  x x x x  |     |   | ... |x| bb    3
     *           |  x x x x  |     |   |     |x|       4
     *           |  x x x x x|     |  x|     |x|       5
     *           |        x x|     |  x|     |x|       6
     *           -------------     -----
     *
     * Final matrix:
     *
     * spl          bb              sep      g
     * -----     -------------     -----
     * |1  | ... |M M        | ... |   | ... |M| spl    1
     * -----     -------------     -----
     *                 .
     *                 .
     *                 .
     * -----     -------------     -----
     * |0  |     |M M        |     |x  |     |M|       1
     * |0  | ... |M M x x x  | ... |x  |     |M|       2
     * -----     |  x x x x  |     |   | ... |x| bb    3
     *           |  x x x x  |     |   |     |x|       4
     *           |  x x x x x|     |  x|     |x|       5
     *           |        x x|     |  x|     |x|       6
     *           -------------     -----
     *
     * @param bb_to_bb_matrix Pointer to the first bb-to-bb-matrix of the first
     * thread.
     * @param bb_to_bb_matrix Pointer to the first bb-g matrix of the first
     * thread.
     */
    void solve_1_spl_ends_A(real *const bb_to_bb_matrix, real *const bb_g_matrix);

    /**
     * Cleans the subdiagonal entries of a spl matrix of ends type B.
     *
     * It is assumed that this function is called by thread 0.
     *
     * Initial matrix:
     *
     *   sp          bb             sep       g
     * -----     -------------     -----
     * |x x| ... |           | ... |   | ... |x| spl   1
     * |x x|     |x x        |     |   | ... |x|       2
     * -----     -------------     -----
     *                 .
     *                 .
     *                 .
     * -----     -------------     -----
     * |  x|     |x x        |     |x  |     |x|       1
     * |  x| ... |x x x x x  | ... |x  |     |x|       2
     * -----     |  x x x x  |     |   | ... |x| bb    3 First block.
     *           |  x x x x  |     |   |     |x|       4
     *           |  x x x x x|     |  x|     |x|       5
     *           |        x x|     |  x|     |x|       6
     *           -------------     -----
     *
     * Final matrix:
     *
     *   sp          bb             sep       g
     * -----     -------------     -----
     * |1 x| ... |           | ... |   | ... |M| spl   1
     * |0 1|     |M M        |     |   |     |M|       2
     * -----     -------------     -----
     *                 .
     *                 .
     *                 .
     * -----     -------------     -----
     * |  0|     |M M        |     |x  |     |M|       1
     * |  0| ... |M M x x x  | ... |x  |     |M|       2
     * -----     |  x x x x  |     |   | ... |x| bb    3 First block.
     *           |  x x x x  |     |   |     |x|       4
     *           |  x x x x x|     |  x|     |x|       5
     *           |        x x|     |  x|     |x|       6
     *           -------------     -----
     *
     * @param bb_to_bb_matrix Pointer to the first bb-to-bb-matrix of the first
     * thread.
     * @param bb_to_bb_matrix Pointer to the first bb-g matrix of the first
     * thread.
     */
    void solve_1_spl_ends_B(real *const bb_to_bb_matrix, real *const bb_g_matrix);

    /**
     * Clean the subdiagonal entries of the spr matrix. This funcion
     * is for the special case of a proline as the last side-chain.
     *
     * It is assumed that this function is called by the last thread.
     *
     * Initial matrix (last side-chain proline):
     *
     * spr                sc                   bb            sep      g
     * -----     -------------------     -------------     -----
     * |  x|     |                x|     |        x  |     |  x|     |x| spr   2
     * -----     -------------------     -------------     -----
     *                    .
     *                    .
     *                    .
     *           -------------------     -------------     -----
     *           |x x         x    |     |      x    |     |   |     |x|      1
     *           |x x         x    |     |      x    |     |   |     |x|      2
     *           |    x x     x x  |     |           |     |   |     |x|      3
     *           |    x x     x x  |     |           |     |   |     |x|      4
     *           |        x x   x x| ... |           | ... |   | ... |x| sc   5
     *           |        x x   x x|     |           |     |   |     |x|      6
     *           |x x x x     x x  |     |      x    |     |   |     |x|      7
     * -----     |    x x x x x x x|     |           |     |   |     |x|      8
     * |  x| ... |        x x   x x|     |        x  |     |  x|     |x|      9
     * -----     -------------------     -------------     -----
     *                                          .
     *                                          .
     *                                          .
     *           -------------------     -------------     -----
     *           |                 |     |x x        |     |x  |     |x|      1
     *           |                 |     |x x x x x  | ... |x  |     |x|      2
     *           |                 | ... |  x x x x  |     |   | ... |x| bb   3
     * -----     |x x         x    |     |  x x x x  |     |   |     |x|      4
     * |  x| ... |                x|     |  x x x x  |     |  x|     |x|      5
     * -----     -------------------     -------------     -----
     *
     * -----     -------------------     -------------     -----
     * |  x|     |                x| ... |      x x  | ... |  x|     |x| sep  1
     * -----     -------------------     -------------     -----
     *
     * Final matrix:
     *
     * spr                sc                   bb            sep      g
     * -----     -------------------     -------------     -----
     * |  1|     |                M|     |        M  |     |  M|     |M| spr   2
     * -----     -------------------     -------------     -----
     *                    .
     *                    .
     *                    .
     *           -------------------     -------------     -----
     *           |x x         x    |     |      x    |     |   |     |x|      1
     *           |x x         x    |     |      x    |     |   |     |x|      2
     *           |    x x     x x  |     |           |     |   |     |x|      3
     *           |    x x     x x  |     |           |     |   |     |x|      4
     *           |        x x   x x| ... |           | ... |   | ... |x| sc   5
     *           |        x x   x x|     |           |     |   |     |x|      6
     *           |x x x x     x x  |     |      x    |     |   |     |x|      7
     * -----     |    x x x x x x x|     |           |     |   |     |x|      8
     * |  0| ... |        x x   x M|     |        M  |     |  M|     |M|      9
     * -----     -------------------     -------------     -----
     *                                          .
     *                                          .
     *                                          .
     *           -------------------     -------------     -----
     *           |                 |     |x x        |     |x  |     |x|      1
     *           |                 |     |x x x x x  | ... |x  |     |x|      2
     *           |                 | ... |  x x x x  |     |   | ... |x| bb   3
     * -----     |x x         x    |     |  x x x x  |     |   |     |x|      4
     * |  0| ... |                M|     |  x x x M  |     |  M|     |M|      5
     * -----     -------------------     -------------     -----
     *
     * -----     -------------------     -------------     -----
     * |  0|     |                M| ... |      x M  | ... |  x|     |x| sep  1
     * -----     -------------------     -------------     -----
     *
     * @param sc_to_sc_matrix Pointer to the last sc-to-sc matrix of the last
     * thread.
     * @param sc_to_bb_matrix Pointer to the last sc-to-bb matrix of the last
     * thread.
     * @param sc_to_sep_matrix Pointer to the last sc-to-sep matrix of the last
     * thread.
     * @param sc_g_matrix Pointer to the last sc-g matrix of the last thread.
     * @param bb_to_sc_matrix Pointer to the last bb-to-sc matrix of the last
     * thread.
     * @param bb_to_bb_matrix Pointer to the last bb-to-bb matrix of the last
     * thread.
     * @param bb_to_sep_matrix Pointer to the last bb-to-sep matrix of the last
     * thread.
     * @param bb_g_matrix Pointer to the last bb-g matrix of the last thread.
     * @param sep_to_sc_matrix Pointer to the last sep-to-sc matrix of the last
     * thread.
     * @param sep_to_bb_matrix Pointer to the last sep-to-bb matrix of the last
     * thread.
     * @param sep_to_sep_matrix Pointer to the last sep-to-sep matrix of the
     * last thread.
     * @param sep_g_matrix Pointer to the last sep-g matrix of the last thread.
     */
    void solve_1_spr_proline(real *const sc_to_sc_matrix,
                             real *const sc_to_bb_matrix,
                             real *const sc_to_sep_matrix,
                             real *const sc_g_matrix,
                             real *const bb_to_sc_matrix,
                             real *const bb_to_bb_matrix,
                             real *const bb_to_sep_matrix,
                             real *const bb_g_matrix,
                             real *const sep_to_sc_matrix,
                             real *const sep_to_bb_matrix,
                             real *const sep_to_sep_matrix,
                             real *const sep_g_matrix);

    /**
     * Clean the subdiagonal entries of the spr matrix. This funcion is for the
     * general case (non-proline last side-chain).
     *
     * It is assumed that this function is called by the last thread.
     *
     * Initial matrix (last side-chain not proline):
     *
     * spr           bb            sep       g
     * -----     -------------     -----
     * |  x|     |        x x|     |  x|     |x|       1
     * -----     -------------     -----
     *                 .
     *                 .
     *                 .
     *           -------------
     *           |x x        |     |x  |     |x|       1
     *           |x x x x x  | ... |x  |     |x|       2
     *           |  x x x x  |     |   | ... |x| bb    3
     * -----     |  x x x x  |     |   |     |x|       4
     * |  x|     |  x x x x x|     |  x|     |x|       5
     * |  x| ... |        x x|     |  x|     |x|       6
     * -----     -------------
     *
     * -----     -------------     -----
     * |  x|     |        x x|     |  x|     |x| sep   1
     * -----     -------------     -----
     *
     * Final matrix:
     *
     * spr            bb            sep        g
     * -----     -------------     -----
     * |  1|     |        M M|     |  M|     |M|       2
     * -----     -------------     -----
     *                 .
     *                 .
     *                 .
     *           -------------     -----
     *           |x x        |     |x  |     |x|       1
     *           |x x x x x  | ... |x  |     |x|       2
     *           |  x x x x  |     |   | ... |x| bb    3
     * -----     |  x x x x  |     |   |     |x|       4
     * |  0|     |  x x x M M|     |  M|     |M|       5
     * |  0| ... |        M M|     |  M|     |M|       6
     * -----     -------------     -----
     *
     * -----     -------------     -----
     * |  0|     |        M M|     |  M|     |M| sep   1
     * -----     -------------     -----
     *
     * @param bb_to_bb_matrix Pointer to the last bb-to-bb matrix of the last
     * thread.
     * @param bb_to_sep_matrix Pointer to the last bb-to-sep matrix of the last
     * thread.
     * @param bb_g_matrix Pointer to the last bb-g matrix of the last thread.
     * @param sep_to_bb_matrix Pointer to the last sep-to-bb matrix of the last
     * thread.
     * @param sep_to_sep_matrix Pointer to the last sep-to-sep matrix of the
     * last thread.
     * @param sep_g_matrix Pointer to the last sep-g matrix of the last thread.
     */
    void solve_1_spr_general(real *const bb_to_bb_matrix,
                             real *const bb_to_sep_matrix,
                             real *const bb_g_matrix,
                             real *const sep_to_bb_matrix,
                             real *const sep_to_sep_matrix,
                             real *const sep_g_matrix);

    /**
     * Clean every subdiagonal entry of the special submatrix.
     *
     * Initial matrix (last side-chain not proline AND ends type A):
     *
     *   sp          bb              sep      g
     * -----     -------------     -----
     * |x  | ... |x x        | ... |   | ... |x| spl   1
     * -----     -------------     -----
     * -----     -------------     -----
     * |  x| ... |        x x| ... |  x| ... |x| spr   2
     * -----     -------------     -----
     *                 .
     *                 .
     *                 .
     * -----     -------------     -----
     * |x  |     |x x        |     |x  |     |x|       1
     * |x  | ... |x x x x x  | ... |x  |     |x|       2
     * -----     |  x x x x  |     |   | ... |x| bb    3 First block.
     *           |  x x x x  |     |   |     |x|       4
     *           |  x x x x x|     |  x|     |x|       5
     *           |        x x|     |  x|     |x|       6
     *           -------------     -----
     *           -------------     -----
     *           |x x        |     |x  |     |x|       1
     *           |x x x x x  | ... |x  |     |x|       2
     *           |  x x x x  |     |   | ... |x| bb    3 Last block.
     * -----     |  x x x x  |     |   |     |x|       4
     * |  x|     |  x x x x x|     |  x|     |x|       5
     * |  x|     |        x x|     |  x|     |x|       6
     * -----     -------------     -----
     *
     * -----     -------------     -----
     * |  x|     |        x x|     |  x|     |x| sep   1 Last block.
     * -----     -------------     -----
     *
     * Initial matrix (last side-chain proline AND ends type A):
     *
     * sp                sc                   bb            sep       g
     * -----     -------------------     -------------     -----
     * |x  | ... |                 | ... |x x        | ... |   | ... |x| spl  1
     * -----     -------------------     -------------     -----
     *
     * -----     -------------------     -------------     -----
     * |  x| ... |                x| ... |        x  | ... |  x| ... |x| spr  2
     * -----     -------------------     -------------     -----
     *                    .
     *                    .
     *                    .
     *           -------------------     -------------     -----
     *           |x x         x    |     |      x    |     |   |     |x|      1
     *           |x x         x    |     |      x    |     |   |     |x|      2
     *           |    x x     x x  |     |           |     |   |     |x|      3
     *           |    x x     x x  |     |           |     |   |     |x|      4
     *           |        x x   x x| ... |           | ... |   | ... |x| sc   5
     * Last block. |        x x   x x|     |           |     |   |     |x| 6 |x
     * x x x     x x  |     |      x    |     |   |     |x|      7
     * -----     |    x x x x x x x|     |           |     |   |     |x|      8
     * |  x| ... |        x x   x x|     |        x  |     |  x|     |x|      9
     * -----     -------------------     -------------     -----
     *                                          .
     *                                          .
     *                                          .
     * -----                             -------------
     * |  x|                             |x x        |     |x  |     |x|      1
     * |  x|               ...           |x x x x x  | ... |x  |     |x|      2
     * -----                             |  x x x x  |     |   | ... |x| bb   3
     * First block. |  x x x x  |     |   |     |x|      4 |  x x x x x|     |
     * x|     |x|      5 |        x x|     |  x|     |x|      6
     *                                   -------------
     *           -------------------     -------------
     *           |                 |     |x x        |     |x  |     |x|      1
     *           |                 |     |x x x x x  | ... |x  |     |x|      2
     *           |                 | ... |  x x x x  |     |   | ... |x| bb   3
     * Last block.
     * -----     |x x         x    |     |  x x x x  |     |   |     |x|      4
     * |  x| ... |                x|     |  x x x x  |     |  x|     |x|      5
     * -----     -------------------     -------------
     *
     * -----                             -------------     -----
     * |  x|               ...           |        x x|     |  x|     |x| sep  1
     * Last block.
     * -----                             -------------     -----
     *
     * Initial matrix (ends type B):
     *
     *   sp          bb              sep      g
     * -----     -------------     -----
     * |x x| ... |x x        | ... |   | ... |x| spl   1
     * |x x|     |           |     |   |     |x|       2
     * -----     -------------     -----
     *                 .
     *                 .
     *                 .
     * -----     -------------     -----
     * |x  |     |x x        |     |x  |     |x|       1
     * |x  | ... |x x x x x  | ... |x  |     |x|       2
     * -----     |  x x x x  |     |   | ... |x| bb    3 First block.
     *           |  x x x x  |     |   |     |x|       4
     *           |  x x x x x|     |  x|     |x|       5
     *           |        x x|     |  x|     |x|       6
     *           -------------     -----
     *
     * Final matrix (last side-chain not proline AND ends type A):
     *
     *   sp          bb              sep      g
     * -----     -------------     -----
     * |1  | ... |M M        | ... |   | ... |M| spl   1
     * -----     -------------     -----
     * -----     -------------     -----
     * |  1| ... |        M M| ... |  M| ... |M| spr   2
     * -----     -------------     -----
     *                 .
     *                 .
     *                 .
     * -----     -------------     -----
     * |0  |     |M M        |     |x  |     |M|       1
     * |0  | ... |M M x x x  | ... |x  |     |M|       2
     * -----     |  x x x x  |     |   | ... |x| bb    3 First block.
     *           |  x x x x  |     |   |     |x|       4
     *           |  x x x x x|     |  x|     |x|       5
     *           |        x x|     |  x|     |x|       6
     *           -------------     -----
     *           -------------     -----
     *           |x x        |     |x  |     |x|       1
     *           |x x x x x  | ... |x  |     |x|       2
     *           |  x x x x  |     |   | ... |x| bb    3 Last block.
     * -----     |  x x x x  |     |   |     |x|       4
     * |  0|     |  x x x M M|     |  M|     |M|       5
     * |  0|     |        M M|     |  M|     |M|       6
     * -----     -------------     -----
     *
     * -----     -------------     -----
     * |  x|     |        x x|     |  x|     |x| sep   1 Last block.
     * -----     -------------     -----
     *
     * Final matrix (last side-chain proline AND ends type A):
     *
     * sp                sc                   bb            sep       g
     * -----     -------------------     -------------     -----
     * |1  | ... |                 | ... |M M        | ... |   | ... |M| spl  1
     * -----     -------------------     -------------     -----
     *
     * -----     -------------------     -------------     -----
     * |  1| ... |                M| ... |        M  | ... |  M| ... |M| spr  2
     * -----     -------------------     -------------     -----
     *                    .
     *                    .
     *                    .
     *           -------------------     -------------     -----
     *           |x x         x    |     |      x    |     |   |     |x|      1
     *           |x x         x    |     |      x    |     |   |     |x|      2
     *           |    x x     x x  |     |           |     |   |     |x|      3
     *           |    x x     x x  |     |           |     |   |     |x|      4
     *           |        x x   x x| ... |           | ... |   | ... |x| sc   5
     * Last block. |        x x   x x|     |           |     |   |     |x| 6 |x
     * x x x     x x  |     |      x    |     |   |     |x|      7
     * -----     |    x x x x x x x|     |           |     |   |     |x|      8
     * |  0| ... |        x x   x M|     |        M  |     |  M|     |M|      9
     * -----     -------------------     -------------     -----
     *                                          .
     *                                          .
     *                                          .
     * -----                             -------------
     * |  0|                             |M M        |     |x  |     |M|      1
     * |  0|               ...           |M M x x x  | ... |x  |     |M|      2
     * -----                             |  x x x x  |     |   | ... |x| bb   3
     * First block. |  x x x x  |     |   |     |x|      4 |  x x x x x|     |
     * x|     |x|      5 |        x x|     |  x|     |x|      6
     *                                   -------------
     *           -------------------     -------------
     *           |                 |     |x x        |     |x  |     |x|      1
     *           |                 |     |x x x x x  | ... |x  |     |x|      2
     *           |                 | ... |  x x x x  |     |   | ... |x| bb   3
     * Last block.
     * -----     |x x         x    |     |  x x x x  |     |   |     |x|      4
     * |  0| ... |                M|     |  x x x M  |     |  M|     |M|      5
     * -----     -------------------     -------------
     *
     * -----                             -------------     -----
     * |  0|               ...           |        M x|     |  M|     |M| sep  1
     * Last block.
     * -----                             -------------     -----
     *
     * Final matrix (ends type B):
     *
     *  sp          bb              sep       g
     * -----     -------------     -----
     * |1 M| ... |           | ... |   | ... |M| spl   1
     * |0 1|     |M M        |     |   |     |M|       2
     * -----     -------------     -----
     *                 .
     *                 .
     *                 .
     * -----     -------------     -----
     * |  0|     |M M        |     |x  |     |M|       1
     * |  0| ... |M M x x x  | ... |x  |     |M|       2
     * -----     |  x x x x  |     |   | ... |x| bb    3 First block.
     *           |  x x x x  |     |   |     |x|       4
     *           |  x x x x x|     |  x|     |x|       5
     *           |        x x|     |  x|     |x|       6
     *           -------------     -----
     *
     * @param thread Thread descriptor [0 - NTHREADS - 1].
     */
    void solve_1(const int thread);

    /**
     * Performs the Gaussian elimination of the sub-diagonal entries of the
     * first SC to SC sub-matrix pointed by SC_TO_SC_MATRIX. Also make 1s in the
     * diagonal entries.
     *
     * It is assumed that SC_TO_SC_MATRIX, SC_TO_G_MATRIX and SC_TO_BB_MATRIX
     * point to the same glycine side-chain data.
     *
     * This function has been automatically generated by
     * step_2_3_16_17_generator.c.
     *
     * Initial matrix:
     *
     * sc          bb            g
     *          1 2 3 4 5 6
     * ---     -------------
     * ---     -------------
     *
     * Final matrix:
     *
     * sc           bb            g
     *          1 2 3 4 5 6
     * ---     -------------
     * ---     -------------
     *
     * @param sc_to_sc_matrix unused.
     * @param sc_to_bb_matrix unused.
     * @param sc_to_sep_matrix unused.
     * @param sc_g_matrix unused.
     * @param sc_to_sc_fillins unused.
     * @param sc_to_bb_fillins unused.
     */
    void solve_2_glycine(__attribute__((unused)) real *const sc_to_sc_matrix,
                         __attribute__((unused)) real *const sc_to_bb_matrix,
                         __attribute__((unused)) real *const sc_to_sep_matrix,
                         __attribute__((unused)) real *const sc_g_matrix,
                         __attribute__((unused)) real *const sc_to_sc_fillins,
                         __attribute__((unused)) real *const sc_to_bb_fillins);

    /**
     * Performs the Gaussian elimination of the sub-diagonal entries of the
     * first SC to SC sub-matrix pointed by SC_TO_SC_MATRIX. Also make 1s in the
     * diagonal entries.
     *
     * It is assumed that SC_TO_SC_MATRIX, SC_TO_G_MATRIX and SC_TO_BB_MATRIX
     * point to the same proline side-chain data.
     *
     * Initial matrix:
     *
     *         sc                  bb          sep      g
     *  1 2 3 4 5 6 7 8 9       1 2 3 4 5
     * -------------------     -----------     ---
     * |x x         x    |     |      x  |     | |     |x|  1
     * |x x         x    |     |      x  |     | |     |x|  2
     * |    x x     x x  |     |         |     | |     |x|  3
     * |    x x     x x  |     |         |     | |     |x|  4
     * |        x x   x x| ... |         | ... | | ... |x|  5 sc
     * |        x x   x x|     |         |     | |     |x|  6
     * |x x x x     x x  |     |      x  |     | |     |x|  7
     * |    x x x x x x x|     |         |     | |     |x|  8
     * |        x x   x x|     |        x|     |x|     |x|  9
     * -------------------     -----------     ---
     *
     * Final matrix:
     *
     *         sc                  bb          sep      g
     *  1 2 3 4 5 6 7 8 9       1 2 3 4 5
     * -------------------     -----------     ---
     * |1 M         M    |     |      M  |     | |     |M|  1
     * |0 1         M    |     |      M  |     | |     |M|  2
     * |    1 M     M M  |     |         |     | |     |M|  3
     * |    0 1     M M  |     |         |     | |     |M|  4
     * |        1 M   M M| ... |         | ... | | .R.. |M|  5 sc
     * |        0 1   M M|     |         |     | |     |M|  6
     * |0 0 0 0     1 M  |     |      M  |     | |     |M|  7
     * |    0 0 0 0 0 1 M|     |      F  |     | |     |M|  8
     * |        0 0   0 1|     |      F M|     |M|     |M|  9
     * -------------------     -----------     ---
     *
     * @param sc_to_sc_matrix Pointer to a SC to SC proline sub-matrix.
     * @param sc_to_bb_matrix Pointer to a SC to BB proline sub-matrix.
     * @param sc_to_sep_matrix Pointer to a SC to SEP proline sub-matrix.
     * @param sc_g_matrix Pointer to a SC g proline sub-matrix.
     * @param sc_to_sc_fillins unused.
     * @param sc_to_bb_fillins Pointer to a SC to BB proline fillins sub-matrix.
     */
    void solve_2_proline(real *const sc_to_sc_matrix,
                         real *const sc_to_bb_matrix,
                         real *const sc_to_sep_matrix,
                         real *const sc_g_matrix,
                         __attribute__((unused)) real *const sc_to_sc_fillins,
                         real *const sc_to_bb_fillins);

    /**
     * Performs the Gaussian elimination of the sub-diagonal entries of the
     * first SC to SC sub-matrix pointed by SC_TO_SC_MATRIX. Also make 1s in the
     * diagonal entries.
     *
     * It is assumed that SC_TO_SC_MATRIX, SC_TO_G_MATRIX and SC_TO_BB_MATRIX
     * point to the same cysteine side-chain data.
     *
     * This function has been automatically generated by
     * step_2_3_16_17_generator.c.
     *
     * Initial matrix:
     *
     *    sc              bb            g
     *  1 2 3 4       1 2 3 4 5 6
     * ---------     -------------
     * |x x    |     |           |     |x|  1
     * |x x x x|     |      x    |     |x|  2
     * |  x x x| ... |      x    | ... |x|  3 sc
     * |  x x x|     |      x    |     |x|  4
     * ---------     -------------
     *
     * Final matrix:
     *
     *    sc              bb            g
     *  1 2 3 4       1 2 3 4 5 6
     * ---------     -------------
     * |1 M    |     |           |     |M|  1
     * |0 1 M M|     |      M    |     |M|  2
     * |  0 1 M| ... |      M    | ... |M|  3 sc
     * |  0 0 1|     |      M    |     |M|  4
     * ---------     -------------
     *
     * @param sc_to_sc_matrix Pointer to a SC to SC cysteine sub-matrix.
     * @param sc_to_bb_matrix Pointer to a SC to BB cysteine sub-matrix.
     * @param sc_to_sep_matrix unused.
     * @param sc_g_matrix Pointer to a SC g cysteine sub-matrix.
     * @param sc_to_sc_fillins unused.
     * @param sc_to_bb_fillins unused.
     */
    void solve_2_cysteine(real *const sc_to_sc_matrix,
                          real *const sc_to_bb_matrix,
                          __attribute__((unused)) real *const sc_to_sep_matrix,
                          real *const sc_g_matrix,
                          __attribute__((unused)) real *const sc_to_sc_fillins,
                          __attribute__((unused)) real *const sc_to_bb_fillins);

    /**
     * Performs the Gaussian elimination of the sub-diagonal entries of the
     * first SC to SC sub-matrix pointed by SC_TO_SC_MATRIX. Also make 1s in the
     * diagonal entries.
     *
     * It is assumed that SC_TO_SC_MATRIX, SC_TO_G_MATRIX and SC_TO_BB_MATRIX
     * point to the same methionine side-chain data.
     *
     * This function has been automatically generated by
     * step_2_3_16_17_generator.c.
     *
     * Initial matrix:
     *
     *          sc                    bb            g
     *  1 2 3 4 5 6 7 8 910       1 2 3 4 5 6
     * ---------------------     -------------
     * |x x x x            |     |           |     |x|  1
     * |x x x x            |     |           |     |x|  2
     * |x x x x            |     |           |     |x|  3
     * |x x x x x          |     |           |     |x|  4
     * |      x x x x x    |     |           |     |x|  5
     * |        x x x x    | ... |           | ... |x|  6 sc
     * |        x x x x    |     |           |     |x|  7
     * |        x x x x x x|     |      x    |     |x|  8
     * |              x x x|     |      x    |     |x|  9
     * |              x x x|     |      x    |     |x| 10
     * ---------------------     -------------
     *
     * Final matrix:
     *
     *          sc                    bb            g
     *  1 2 3 4 5 6 7 8 910       1 2 3 4 5 6
     * ---------------------     -------------
     * |1 M M M            |     |           |     |M|  1
     * |0 1 M M            |     |           |     |M|  2
     * |0 0 1 M            |     |           |     |M|  3
     * |0 0 0 1 M          |     |           |     |M|  4
     * |      0 1 M M M    |     |           |     |M|  5
     * |        0 1 M M    | ... |           | ... |M|  6 sc
     * |        0 0 1 M    |     |           |     |M|  7
     * |        0 0 0 1 M M|     |      M    |     |M|  8
     * |              0 1 M|     |      M    |     |M|  9
     * |              0 0 1|     |      M    |     |M| 10
     * ---------------------     -------------
     *
     * @param sc_to_sc_matrix Pointer to a SC to SC methionine sub-matrix.
     * @param sc_to_bb_matrix Pointer to a SC to BB methionine sub-matrix.
     * @param sc_to_sep_matrix unused.
     * @param sc_g_matrix Pointer to a SC g methionine sub-matrix.
     * @param sc_to_sc_fillins unused.
     * @param sc_to_bb_fillins unused.
     */
    void solve_2_methionine(real *const sc_to_sc_matrix,
                            real *const sc_to_bb_matrix,
                            __attribute__((unused)) real *const sc_to_sep_matrix,
                            real *const sc_g_matrix,
                            __attribute__((unused)) real *const sc_to_sc_fillins,
                            __attribute__((unused)) real *const sc_to_bb_fillins);

    /**
     * Performs the Gaussian elimination of the sub-diagonal entries of the
     * first SC to SC sub-matrix pointed by SC_TO_SC_MATRIX. Also make 1s in the
     * diagonal entries.
     *
     * It is assumed that SC_TO_SC_MATRIX, SC_TO_G_MATRIX and SC_TO_BB_MATRIX
     * point to the same alaline side-chain data.
     *
     * This function has been automatically generated by
     * step_2_3_16_17_generator.c.
     *
     * Initial matrix:
     *
     *   sc             bb            g
     *  1 2 3       1 2 3 4 5 6
     * -------     -------------
     * |x x x|     |      x    |     |x|  1
     * |x x x| ... |      x    | ... |x|  2 sc
     * |x x x|     |      x    |     |x|  3
     * -------     -------------
     *
     * Final matrix:
     *
     *   sc             bb            g
     *  1 2 3       1 2 3 4 5 6
     * -------     -------------
     * |1 M M|     |      M    |     |M|  1
     * |0 1 M| ... |      M    | ... |M|  2 sc
     * |0 0 1|     |      M    |     |M|  3
     * -------     -------------
     *
     * @param sc_to_sc_matrix Pointer to a SC to SC alaline sub-matrix.
     * @param sc_to_bb_matrix Pointer to a SC to BB alaline sub-matrix.
     * @param sc_to_sep_matrix unused.
     * @param sc_g_matrix Pointer to a SC g alaline sub-matrix.
     * @param sc_to_sc_fillins unused.
     * @param sc_to_bb_fillins unused.
     */
    void solve_2_alaline(real *const sc_to_sc_matrix,
                         real *const sc_to_bb_matrix,
                         __attribute__((unused)) real *const sc_to_sep_matrix,
                         real *const sc_g_matrix,
                         __attribute__((unused)) real *const sc_to_sc_fillins,
                         __attribute__((unused)) real *const sc_to_bb_fillins);

    /**
     * Performs the Gaussian elimination of the sub-diagonal entries of the
     * first SC to SC sub-matrix pointed by SC_TO_SC_MATRIX. Also make 1s in the
     * diagonal entries.
     *
     * It is assumed that SC_TO_SC_MATRIX, SC_TO_G_MATRIX and SC_TO_BB_MATRIX
     * point to the same valine side-chain data.
     *
     * This function has been automatically generated by
     * step_2_3_16_17_generator.c.
     *
     * Initial matrix:
     *
     *         sc                   bb            g
     *  1 2 3 4 5 6 7 8 9       1 2 3 4 5 6
     * -------------------     -------------
     * |x x x x          |     |           |     |x|  1
     * |x x x x          |     |           |     |x|  2
     * |x x x x          |     |           |     |x|  3
     * |x x x x       x x|     |      x    |     |x|  4
     * |        x x x x  | ... |           | ... |x|  5 sc
     * |        x x x x  |     |           |     |x|  6
     * |        x x x x  |     |           |     |x|  7
     * |      x x x x x x|     |      x    |     |x|  8
     * |      x       x x|     |      x    |     |x|  9
     * -------------------     -------------
     *
     * Final matrix:
     *
     *         sc                   bb            g
     *  1 2 3 4 5 6 7 8 9       1 2 3 4 5 6
     * -------------------     -------------
     * |1 M M M          |     |           |     |M|  1
     * |0 1 M M          |     |           |     |M|  2
     * |0 0 1 M          |     |           |     |M|  3
     * |0 0 0 1       M M|     |      M    |     |M|  4
     * |        1 M M M  | ... |           | ... |M|  5 sc
     * |        0 1 M M  |     |           |     |M|  6
     * |        0 0 1 M  |     |           |     |M|  7
     * |      0 0 0 0 1 M|     |      M    |     |M|  8
     * |      0       0 1|     |      M    |     |M|  9
     * -------------------     -------------
     *
     * @param sc_to_sc_matrix Pointer to a SC to SC valine sub-matrix.
     * @param sc_to_bb_matrix Pointer to a SC to BB valine sub-matrix.
     * @param sc_to_sep_matrix unused.
     * @param sc_g_matrix Pointer to a SC g valine sub-matrix.
     * @param sc_to_sc_fillins unused.
     * @param sc_to_bb_fillins unused.
     */
    void solve_2_valine(real *const sc_to_sc_matrix,
                        real *const sc_to_bb_matrix,
                        __attribute__((unused)) real *const sc_to_sep_matrix,
                        real *const sc_g_matrix,
                        __attribute__((unused)) real *const sc_to_sc_fillins,
                        __attribute__((unused)) real *const sc_to_bb_fillins);

    /**
     * Performs the Gaussian elimination of the sub-diagonal entries of the
     * first SC to SC sub-matrix pointed by SC_TO_SC_MATRIX. Also make 1s in the
     * diagonal entries.
     *
     * It is assumed that SC_TO_SC_MATRIX, SC_TO_G_MATRIX and SC_TO_BB_MATRIX
     * point to the same leucine side-chain data.
     *
     * This function has been automatically generated by
     * step_2_3_16_17_generator.c.
     *
     * Initial matrix:
     *
     *            sc                      bb            g
     *  1 2 3 4 5 6 7 8 9101112       1 2 3 4 5 6
     * -------------------------     -------------
     * |x x x x                |     |           |     |x|  1
     * |x x x x                |     |           |     |x|  2
     * |x x x x                |     |           |     |x|  3
     * |x x x x       x x x    |     |           |     |x|  4
     * |        x x x x        |     |           |     |x|  5
     * |        x x x x        |     |           |     |x|  6
     * |        x x x x        | ... |           | ... |x|  7 sc
     * |      x x x x x x x    |     |           |     |x|  8
     * |      x       x x x    |     |           |     |x|  9
     * |      x       x x x x x|     |      x    |     |x| 10
     * |                  x x x|     |      x    |     |x| 11
     * |                  x x x|     |      x    |     |x| 12
     * -------------------------     -------------
     *
     * Final matrix:
     *
     *            sc                      bb            g
     *  1 2 3 4 5 6 7 8 9101112       1 2 3 4 5 6
     * -------------------------     -------------
     * |1 M M M                |     |           |     |M|  1
     * |0 1 M M                |     |           |     |M|  2
     * |0 0 1 M                |     |           |     |M|  3
     * |0 0 0 1       M M M    |     |           |     |M|  4
     * |        1 M M M        |     |           |     |M|  5
     * |        0 1 M M        |     |           |     |M|  6
     * |        0 0 1 M        | ... |           | ... |M|  7 sc
     * |      0 0 0 0 1 M M    |     |           |     |M|  8
     * |      0       0 1 M    |     |           |     |M|  9
     * |      0       0 0 1 M M|     |      M    |     |M| 10
     * |                  0 1 M|     |      M    |     |M| 11
     * |                  0 0 1|     |      M    |     |M| 12
     * -------------------------     -------------
     *
     * @param sc_to_sc_matrix Pointer to a SC to SC leucine sub-matrix.
     * @param sc_to_bb_matrix Pointer to a SC to BB leucine sub-matrix.
     * @param sc_to_sep_matrix unused.
     * @param sc_g_matrix Pointer to a SC g leucine sub-matrix.
     * @param sc_to_sc_fillins unused.
     * @param sc_to_bb_fillins unused.
     */
    void solve_2_leucine(real *const sc_to_sc_matrix,
                         real *const sc_to_bb_matrix,
                         __attribute__((unused)) real *const sc_to_sep_matrix,
                         real *const sc_g_matrix,
                         __attribute__((unused)) real *const sc_to_sc_fillins,
                         __attribute__((unused)) real *const sc_to_bb_fillins);

    /**
     * Performs the Gaussian elimination of the sub-diagonal entries of the
     * first SC to SC sub-matrix pointed by SC_TO_SC_MATRIX. Also make 1s in the
     * diagonal entries.
     *
     * It is assumed that SC_TO_SC_MATRIX, SC_TO_G_MATRIX and SC_TO_BB_MATRIX
     * point to the same isoleucine side-chain data.
     *
     * This function has been automatically generated by
     * step_2_3_16_17_generator.c.
     *
     * Initial matrix:
     *
     *            sc                      bb            g
     *  1 2 3 4 5 6 7 8 9101112       1 2 3 4 5 6
     * -------------------------     -------------
     * |x x x x                |     |           |     |x|  1
     * |x x x x                |     |           |     |x|  2
     * |x x x x                |     |           |     |x|  3
     * |x x x x x x x          |     |           |     |x|  4
     * |      x x x x          |     |           |     |x|  5
     * |      x x x x          |     |           |     |x|  6
     * |      x x x x       x x| ... |      x    | ... |x|  7 sc
     * |              x x x x  |     |           |     |x|  8
     * |              x x x x  |     |           |     |x|  9
     * |              x x x x  |     |           |     |x| 10
     * |            x x x x x x|     |      x    |     |x| 11
     * |            x       x x|     |      x    |     |x| 12
     * -------------------------     -------------
     *
     * Final matrix:
     *
     *            sc                      bb            g
     *  1 2 3 4 5 6 7 8 9101112       1 2 3 4 5 6
     * -------------------------     -------------
     * |1 M M M                |     |           |     |M|  1
     * |0 1 M M                |     |           |     |M|  2
     * |0 0 1 M                |     |           |     |M|  3
     * |0 0 0 1 M M M          |     |           |     |M|  4
     * |      0 1 M M          |     |           |     |M|  5
     * |      0 0 1 M          |     |           |     |M|  6
     * |      0 0 0 1       M M| ... |      M    | ... |M|  7 sc
     * |              1 M M M  |     |           |     |M|  8
     * |              0 1 M M  |     |           |     |M|  9
     * |              0 0 1 M  |     |           |     |M| 10
     * |            0 0 0 0 1 M|     |      M    |     |M| 11
     * |            0       0 1|     |      M    |     |M| 12
     * -------------------------     -------------
     *
     * @param sc_to_sc_matrix Pointer to a SC to SC isoleucine sub-matrix.
     * @param sc_to_bb_matrix Pointer to a SC to BB isoleucine sub-matrix.
     * @param sc_to_sep_matrix unused.
     * @param sc_g_matrix Pointer to a SC g isoleucine sub-matrix.
     * @param sc_to_sc_fillins unused.
     * @param sc_to_bb_fillins unused.
     */
    void solve_2_isoleucine(real *const sc_to_sc_matrix,
                            real *const sc_to_bb_matrix,
                            __attribute__((unused)) real *const sc_to_sep_matrix,
                            real *const sc_g_matrix,
                            __attribute__((unused)) real *const sc_to_sc_fillins,
                            __attribute__((unused)) real *const sc_to_bb_fillins);

    /**
     * Performs the Gaussian elimination of the sub-diagonal entries of the
     * first SC to SC sub-matrix pointed by SC_TO_SC_MATRIX. Also make 1s in the
     * diagonal entries.
     *
     * It is assumed that SC_TO_SC_MATRIX, SC_TO_G_MATRIX and SC_TO_BB_MATRIX
     * point to the same aspartic_acid side-chain data.
     *
     * This function has been automatically generated by
     * step_2_3_16_17_generator.c.
     *
     * Initial matrix:
     *
     *     sc               bb            g
     *  1 2 3 4 5       1 2 3 4 5 6
     * -----------     -------------
     * |x x x    |     |           |     |x|  1
     * |x x x    |     |           |     |x|  2
     * |x x x x x| ... |      x    | ... |x|  3 sc
     * |    x x x|     |      x    |     |x|  4
     * |    x x x|     |      x    |     |x|  5
     * -----------     -------------
     *
     * Final matrix:
     *
     *     sc               bb            g
     *  1 2 3 4 5       1 2 3 4 5 6
     * -----------     -------------
     * |1 M M    |     |           |     |M|  1
     * |0 1 M    |     |           |     |M|  2
     * |0 0 1 M M| ... |      M    | ... |M|  3 sc
     * |    0 1 M|     |      M    |     |M|  4
     * |    0 0 1|     |      M    |     |M|  5
     * -----------     -------------
     *
     * @param sc_to_sc_matrix Pointer to a SC to SC aspartic_acid sub-matrix.
     * @param sc_to_bb_matrix Pointer to a SC to BB aspartic_acid sub-matrix.
     * @param sc_to_sep_matrix unused.
     * @param sc_g_matrix Pointer to a SC g aspartic_acid sub-matrix.
     * @param sc_to_sc_fillins unused.
     * @param sc_to_bb_fillins unused.
     */
    void solve_2_aspartic_acid(real *const sc_to_sc_matrix,
                               real *const sc_to_bb_matrix,
                               __attribute__((unused)) real *const sc_to_sep_matrix,
                               real *const sc_g_matrix,
                               __attribute__((unused)) real *const sc_to_sc_fillins,
                               __attribute__((unused))
                               real *const sc_to_bb_fillins);

    /**
     * Performs the Gaussian elimination of the sub-diagonal entries of the
     * first SC to SC sub-matrix pointed by SC_TO_SC_MATRIX. Also make 1s in the
     * diagonal entries.
     *
     * It is assumed that SC_TO_SC_MATRIX, SC_TO_G_MATRIX and SC_TO_BB_MATRIX
     * point to the same glutamic_acid side-chain data.
     *
     * This function has been automatically generated by
     * step_2_3_16_17_generator.c.
     *
     * Initial matrix:
     *
     *        sc                  bb            g
     *  1 2 3 4 5 6 7 8       1 2 3 4 5 6
     * -----------------     -------------
     * |x x x          |     |           |     |x|  1
     * |x x x          |     |           |     |x|  2
     * |x x x x x x    |     |           |     |x|  3
     * |    x x x x    |     |           |     |x|  4
     * |    x x x x    | ... |           | ... |x|  5 sc
     * |    x x x x x x|     |      x    |     |x|  6
     * |          x x x|     |      x    |     |x|  7
     * |          x x x|     |      x    |     |x|  8
     * -----------------     -------------
     *
     * Final matrix:
     *
     *        sc                  bb            g
     *  1 2 3 4 5 6 7 8       1 2 3 4 5 6
     * -----------------     -------------
     * |1 M M          |     |           |     |M|  1
     * |0 1 M          |     |           |     |M|  2
     * |0 0 1 M M M    |     |           |     |M|  3
     * |    0 1 M M    |     |           |     |M|  4
     * |    0 0 1 M    | ... |           | ... |M|  5 sc
     * |    0 0 0 1 M M|     |      M    |     |M|  6
     * |          0 1 M|     |      M    |     |M|  7
     * |          0 0 1|     |      M    |     |M|  8
     * -----------------     -------------
     *
     * @param sc_to_sc_matrix Pointer to a SC to SC glutamic_acid sub-matrix.
     * @param sc_to_bb_matrix Pointer to a SC to BB glutamic_acid sub-matrix.
     * @param sc_to_sep_matrix unused.
     * @param sc_g_matrix Pointer to a SC g glutamic_acid sub-matrix.
     * @param sc_to_sc_fillins unused.
     * @param sc_to_bb_fillins unused.
     */
    void solve_2_glutamic_acid(real *const sc_to_sc_matrix,
                               real *const sc_to_bb_matrix,
                               __attribute__((unused)) real *const sc_to_sep_matrix,
                               real *const sc_g_matrix,
                               __attribute__((unused)) real *const sc_to_sc_fillins,
                               __attribute__((unused))
                               real *const sc_to_bb_fillins);

    /**
     * Performs the Gaussian elimination of the sub-diagonal entries of the
     * first SC to SC sub-matrix pointed by SC_TO_SC_MATRIX. Also make 1s in the
     * diagonal entries.
     *
     * It is assumed that SC_TO_SC_MATRIX, SC_TO_G_MATRIX and SC_TO_BB_MATRIX
     * point to the same asparagine side-chain data.
     *
     * This function has been automatically generated by
     * step_2_3_16_17_generator.c.
     *
     * Initial matrix:
     *
     *       sc                 bb            g
     *  1 2 3 4 5 6 7       1 2 3 4 5 6
     * ---------------     -------------
     * |x x x        |     |           |     |x|  1
     * |x x x        |     |           |     |x|  2
     * |x x x x x    |     |           |     |x|  3
     * |    x x x    | ... |           | ... |x|  4 sc
     * |    x x x x x|     |      x    |     |x|  5
     * |        x x x|     |      x    |     |x|  6
     * |        x x x|     |      x    |     |x|  7
     * ---------------     -------------
     *
     * Final matrix:
     *
     *       sc                 bb            g
     *  1 2 3 4 5 6 7       1 2 3 4 5 6
     * ---------------     -------------
     * |1 M M        |     |           |     |M|  1
     * |0 1 M        |     |           |     |M|  2
     * |0 0 1 M M    |     |           |     |M|  3
     * |    0 1 M    | ... |           | ... |M|  4 sc
     * |    0 0 1 M M|     |      M    |     |M|  5
     * |        0 1 M|     |      M    |     |M|  6
     * |        0 0 1|     |      M    |     |M|  7
     * ---------------     -------------
     *
     * @param sc_to_sc_matrix Pointer to a SC to SC asparagine sub-matrix.
     * @param sc_to_bb_matrix Pointer to a SC to BB asparagine sub-matrix.
     * @param sc_to_sep_matrix unused.
     * @param sc_g_matrix Pointer to a SC g asparagine sub-matrix.
     * @param sc_to_sc_fillins unused.
     * @param sc_to_bb_fillins unused.
     */
    void solve_2_asparagine(real *const sc_to_sc_matrix,
                            real *const sc_to_bb_matrix,
                            __attribute__((unused)) real *const sc_to_sep_matrix,
                            real *const sc_g_matrix,
                            __attribute__((unused)) real *const sc_to_sc_fillins,
                            __attribute__((unused)) real *const sc_to_bb_fillins);

    /**
     * Performs the Gaussian elimination of the sub-diagonal entries of the
     * first SC to SC sub-matrix pointed by SC_TO_SC_MATRIX. Also make 1s in the
     * diagonal entries.
     *
     * It is assumed that SC_TO_SC_MATRIX, SC_TO_G_MATRIX and SC_TO_BB_MATRIX
     * point to the same glutamine side-chain data.
     *
     * This function has been automatically generated by
     * step_2_3_16_17_generator.c.
     *
     * Initial matrix:
     *
     *          sc                    bb            g
     *  1 2 3 4 5 6 7 8 910       1 2 3 4 5 6
     * ---------------------     -------------
     * |x x x              |     |           |     |x|  1
     * |x x x              |     |           |     |x|  2
     * |x x x x x          |     |           |     |x|  3
     * |    x x x          |     |           |     |x|  4
     * |    x x x x x x    |     |           |     |x|  5
     * |        x x x x    | ... |           | ... |x|  6 sc
     * |        x x x x    |     |           |     |x|  7
     * |        x x x x x x|     |      x    |     |x|  8
     * |              x x x|     |      x    |     |x|  9
     * |              x x x|     |      x    |     |x| 10
     * ---------------------     -------------
     *
     * Final matrix:
     *
     *          sc                    bb            g
     *  1 2 3 4 5 6 7 8 910       1 2 3 4 5 6
     * ---------------------     -------------
     * |1 M M              |     |           |     |M|  1
     * |0 1 M              |     |           |     |M|  2
     * |0 0 1 M M          |     |           |     |M|  3
     * |    0 1 M          |     |           |     |M|  4
     * |    0 0 1 M M M    |     |           |     |M|  5
     * |        0 1 M M    | ... |           | ... |M|  6 sc
     * |        0 0 1 M    |     |           |     |M|  7
     * |        0 0 0 1 M M|     |      M    |     |M|  8
     * |              0 1 M|     |      M    |     |M|  9
     * |              0 0 1|     |      M    |     |M| 10
     * ---------------------     -------------
     *
     * @param sc_to_sc_matrix Pointer to a SC to SC glutamine sub-matrix.
     * @param sc_to_bb_matrix Pointer to a SC to BB glutamine sub-matrix.
     * @param sc_to_sep_matrix unused.
     * @param sc_g_matrix Pointer to a SC g glutamine sub-matrix.
     * @param sc_to_sc_fillins unused.
     * @param sc_to_bb_fillins unused.
     */
    void solve_2_glutamine(real *const sc_to_sc_matrix,
                           real *const sc_to_bb_matrix,
                           __attribute__((unused)) real *const sc_to_sep_matrix,
                           real *const sc_g_matrix,
                           __attribute__((unused)) real *const sc_to_sc_fillins,
                           __attribute__((unused)) real *const sc_to_bb_fillins);

    /**
     * Performs the Gaussian elimination of the sub-diagonal entries of the
     * first SC to SC sub-matrix pointed by SC_TO_SC_MATRIX. Also make 1s in the
     * diagonal entries.
     *
     * It is assumed that SC_TO_SC_MATRIX, SC_TO_G_MATRIX and SC_TO_BB_MATRIX
     * point to the same histidine side-chain data.
     *
     * This function has been automatically generated by
     * step_2_3_16_17_generator.c.
     *
     * Initial matrix:
     *
     *           sc                     bb            g
     *  1 2 3 4 5 6 7 8 91011       1 2 3 4 5 6
     * -----------------------     -------------
     * |x   x   x            |     |           |     |x|  1
     * |  x x     x          |     |           |     |x|  2
     * |x x x   x x          |     |           |     |x|  3
     * |      x x     x      |     |           |     |x|  4
     * |x   x x x     x      |     |           |     |x|  5
     * |  x x     x x        | ... |           | ... |x|  6 sc
     * |          x x x x    |     |           |     |x|  7
     * |      x x   x x x    |     |           |     |x|  8
     * |            x x x x x|     |      x    |     |x|  9
     * |                x x x|     |      x    |     |x| 10
     * |                x x x|     |      x    |     |x| 11
     * -----------------------     -------------
     *
     * Final matrix:
     *
     *           sc                     bb            g
     *  1 2 3 4 5 6 7 8 91011       1 2 3 4 5 6
     * -----------------------     -------------
     * |1   M   M            |     |           |     |M|  1
     * |  1 M     M          |     |           |     |M|  2
     * |0 0 1   M M          |     |           |     |M|  3
     * |      1 M     M      |     |           |     |M|  4
     * |0   0 0 1 F   M      |     |           |     |M|  5
     * |  0 0   0 1 M F      | ... |           | ... |M|  6 sc
     * |          0 1 M M    |     |           |     |M|  7
     * |      0 0 0 0 1 M    |     |           |     |M|  8
     * |            0 0 1 M M|     |      M    |     |M|  9
     * |                0 1 M|     |      M    |     |M| 10
     * |                0 0 1|     |      M    |     |M| 11
     * -----------------------     -------------
     *
     * @param sc_to_sc_matrix Pointer to a SC to SC histidine sub-matrix.
     * @param sc_to_bb_matrix Pointer to a SC to BB histidine sub-matrix.
     * @param sc_to_sep_matrix unused.
     * @param sc_g_matrix Pointer to a SC g histidine sub-matrix.
     * @param sc_to_sc_fillins Pointer to a SC to SC histidine fillins
     * sub-matrix.
     * @param sc_to_bb_fillins unused.
     */
    void solve_2_histidine(real *const sc_to_sc_matrix,
                           real *const sc_to_bb_matrix,
                           __attribute__((unused)) real *const sc_to_sep_matrix,
                           real *const sc_g_matrix,
                           real *const sc_to_sc_fillins,
                           __attribute__((unused)) real *const sc_to_bb_fillins);

    /**
     * Performs the Gaussian elimination of the sub-diagonal entries of the
     * first SC to SC sub-matrix pointed by SC_TO_SC_MATRIX. Also make 1s in the
     * diagonal entries.
     *
     * It is assumed that SC_TO_SC_MATRIX, SC_TO_G_MATRIX and SC_TO_BB_MATRIX
     * point to the same lysine side-chain data.
     *
     * This function has been automatically generated by
     * step_2_3_16_17_generator.c.
     *
     * Initial matrix:
     *
     *               sc                         bb            g
     *  1 2 3 4 5 6 7 8 9101112131415       1 2 3 4 5 6
     * -------------------------------     -------------
     * |x x x x                      |     |           |     |x|  1
     * |x x x x                      |     |           |     |x|  2
     * |x x x x                      |     |           |     |x|  3
     * |x x x x x x x                |     |           |     |x|  4
     * |      x x x x                |     |           |     |x|  5
     * |      x x x x                |     |           |     |x|  6
     * |      x x x x x x x          |     |           |     |x|  7
     * |            x x x x          | ... |           | ... |x|  8 sc
     * |            x x x x          |     |           |     |x|  9
     * |            x x x x x x x    |     |           |     |x| 10
     * |                  x x x x    |     |           |     |x| 11
     * |                  x x x x    |     |           |     |x| 12
     * |                  x x x x x x|     |      x    |     |x| 13
     * |                        x x x|     |      x    |     |x| 14
     * |                        x x x|     |      x    |     |x| 15
     * -------------------------------     -------------
     *
     * Final matrix:
     *
     *               sc                         bb            g
     *  1 2 3 4 5 6 7 8 9101112131415       1 2 3 4 5 6
     * -------------------------------     -------------
     * |1 M M M                      |     |           |     |M|  1
     * |0 1 M M                      |     |           |     |M|  2
     * |0 0 1 M                      |     |           |     |M|  3
     * |0 0 0 1 M M M                |     |           |     |M|  4
     * |      0 1 M M                |     |           |     |M|  5
     * |      0 0 1 M                |     |           |     |M|  6
     * |      0 0 0 1 M M M          |     |           |     |M|  7
     * |            0 1 M M          | ... |           | ... |M|  8 sc
     * |            0 0 1 M          |     |           |     |M|  9
     * |            0 0 0 1 M M M    |     |           |     |M| 10
     * |                  0 1 M M    |     |           |     |M| 11
     * |                  0 0 1 M    |     |           |     |M| 12
     * |                  0 0 0 1 M M|     |      M    |     |M| 13
     * |                        0 1 M|     |      M    |     |M| 14
     * |                        0 0 1|     |      M    |     |M| 15
     * -------------------------------     -------------
     *
     * @param sc_to_sc_matrix Pointer to a SC to SC lysine sub-matrix.
     * @param sc_to_bb_matrix Pointer to a SC to BB lysine sub-matrix.
     * @param sc_to_sep_matrix unused.
     * @param sc_g_matrix Pointer to a SC g lysine sub-matrix.
     * @param sc_to_sc_fillins unused.
     * @param sc_to_bb_fillins unused.
     */
    void solve_2_lysine(real *const sc_to_sc_matrix,
                        real *const sc_to_bb_matrix,
                        __attribute__((unused)) real *const sc_to_sep_matrix,
                        real *const sc_g_matrix,
                        __attribute__((unused)) real *const sc_to_sc_fillins,
                        __attribute__((unused)) real *const sc_to_bb_fillins);

    /**
     * Performs the Gaussian elimination of the sub-diagonal entries of the
     * first SC to SC sub-matrix pointed by SC_TO_SC_MATRIX. Also make 1s in the
     * diagonal entries.
     *
     * It is assumed that SC_TO_SC_MATRIX, SC_TO_G_MATRIX and SC_TO_BB_MATRIX
     * point to the same arginine side-chain data.
     *
     * This function has been automatically generated by
     * step_2_3_16_17_generator.c.
     *
     * Initial matrix:
     *
     *                 sc                           bb            g
     *  1 2 3 4 5 6 7 8 91011121314151617       1 2 3 4 5 6
     * -----------------------------------     -------------
     * |x x x                            |     |           |     |x|  1
     * |x x x                            |     |           |     |x|  2
     * |x x x     x x                    |     |           |     |x|  3
     * |      x x x                      |     |           |     |x|  4
     * |      x x x                      |     |           |     |x|  5
     * |    x x x x x                    |     |           |     |x|  6
     * |    x     x x x x                |     |           |     |x|  7
     * |            x x x                |     |           |     |x|  8
     * |            x x x x x x          | ... |           | ... |x|  9 sc
     * |                x x x x          |     |           |     |x| 10
     * |                x x x x          |     |           |     |x| 11
     * |                x x x x x x x    |     |           |     |x| 12
     * |                      x x x x    |     |           |     |x| 13
     * |                      x x x x    |     |           |     |x| 14
     * |                      x x x x x x|     |      x    |     |x| 15
     * |                            x x x|     |      x    |     |x| 16
     * |                            x x x|     |      x    |     |x| 17
     * -----------------------------------     -------------
     *
     * Final matrix:
     *
     *                 sc                           bb            g
     *  1 2 3 4 5 6 7 8 91011121314151617       1 2 3 4 5 6
     * -----------------------------------     -------------
     * |1 M M                            |     |           |     |M|  1
     * |0 1 M                            |     |           |     |M|  2
     * |0 0 1     M M                    |     |           |     |M|  3
     * |      1 M M                      |     |           |     |M|  4
     * |      0 1 M                      |     |           |     |M|  5
     * |    0 0 0 1 M                    |     |           |     |M|  6
     * |    0     0 1 M M                |     |           |     |M|  7
     * |            0 1 M                |     |           |     |M|  8
     * |            0 0 1 M M M          | ... |           | ... |M|  9 sc
     * |                0 1 M M          |     |           |     |M| 10
     * |                0 0 1 M          |     |           |     |M| 11
     * |                0 0 0 1 M M M    |     |           |     |M| 12
     * |                      0 1 M M    |     |           |     |M| 13
     * |                      0 0 1 M    |     |           |     |M| 14
     * |                      0 0 0 1 M M|     |      M    |     |M| 15
     * |                            0 1 M|     |      M    |     |M| 16
     * |                            0 0 1|     |      M    |     |M| 17
     * -----------------------------------     -------------
     *
     * @param sc_to_sc_matrix Pointer to a SC to SC arginine sub-matrix.
     * @param sc_to_bb_matrix Pointer to a SC to BB arginine sub-matrix.
     * @param sc_to_sep_matrix unused.
     * @param sc_g_matrix Pointer to a SC g arginine sub-matrix.
     * @param sc_to_sc_fillins unused.
     * @param sc_to_bb_fillins unused.
     */
    void solve_2_arginine(real *const sc_to_sc_matrix,
                          real *const sc_to_bb_matrix,
                          __attribute__((unused)) real *const sc_to_sep_matrix,
                          real *const sc_g_matrix,
                          __attribute__((unused)) real *const sc_to_sc_fillins,
                          __attribute__((unused)) real *const sc_to_bb_fillins);

    /**
     * Performs the Gaussian elimination of the sub-diagonal entries of the
     * first SC to SC sub-matrix pointed by SC_TO_SC_MATRIX. Also make 1s in the
     * diagonal entries.
     *
     * It is assumed that SC_TO_SC_MATRIX, SC_TO_G_MATRIX and SC_TO_BB_MATRIX
     * point to the same serine side-chain data.
     *
     * This function has been automatically generated by
     * step_2_3_16_17_generator.c.
     *
     * Initial matrix:
     *
     *    sc              bb            g
     *  1 2 3 4       1 2 3 4 5 6
     * ---------     -------------
     * |x x    |     |           |     |x|  1
     * |x x x x|     |      x    |     |x|  2
     * |  x x x| ... |      x    | ... |x|  3 sc
     * |  x x x|     |      x    |     |x|  4
     * ---------     -------------
     *
     * Final matrix:
     *
     *    sc              bb            g
     *  1 2 3 4       1 2 3 4 5 6
     * ---------     -------------
     * |1 M    |     |           |     |M|  1
     * |0 1 M M|     |      M    |     |M|  2
     * |  0 1 M| ... |      M    | ... |M|  3 sc
     * |  0 0 1|     |      M    |     |M|  4
     * ---------     -------------
     *
     * @param sc_to_sc_matrix Pointer to a SC to SC serine sub-matrix.
     * @param sc_to_bb_matrix Pointer to a SC to BB serine sub-matrix.
     * @param sc_to_sep_matrix unused.
     * @param sc_g_matrix Pointer to a SC g serine sub-matrix.
     * @param sc_to_sc_fillins unused.
     * @param sc_to_bb_fillins unused.
     */
    void solve_2_serine(real *const sc_to_sc_matrix,
                        real *const sc_to_bb_matrix,
                        __attribute__((unused)) real *const sc_to_sep_matrix,
                        real *const sc_g_matrix,
                        __attribute__((unused)) real *const sc_to_sc_fillins,
                        __attribute__((unused)) real *const sc_to_bb_fillins);

    /**
     * Performs the Gaussian elimination of the sub-diagonal entries of the
     * first SC to SC sub-matrix pointed by SC_TO_SC_MATRIX. Also make 1s in the
     * diagonal entries.
     *
     * It is assumed that SC_TO_SC_MATRIX, SC_TO_G_MATRIX and SC_TO_BB_MATRIX
     * point to the same phenylaline side-chain data.
     *
     * This function has been automatically generated by
     * step_2_3_16_17_generator.c.
     *
     * Initial matrix:
     *
     *              sc                        bb            g
     *  1 2 3 4 5 6 7 8 91011121314       1 2 3 4 5 6
     * -----------------------------     -------------
     * |x   x             x        |     |           |     |x|  1
     * |  x x   x                  |     |           |     |x|  2
     * |x x x   x         x        |     |           |     |x|  3
     * |      x x   x              |     |           |     |x|  4
     * |  x x x x   x              |     |           |     |x|  5
     * |          x x   x          |     |           |     |x|  6
     * |      x x x x   x          |     |           |     |x|  7
     * |              x x   x      | ... |           | ... |x|  8 sc
     * |          x x x x   x      |     |           |     |x|  9
     * |x   x             x x x    |     |           |     |x| 10
     * |              x x x x x    |     |           |     |x| 11
     * |                  x x x x x|     |      x    |     |x| 12
     * |                      x x x|     |      x    |     |x| 13
     * |                      x x x|     |      x    |     |x| 14
     * -----------------------------     -------------
     *
     * Final matrix:
     *
     *              sc                        bb            g
     *  1 2 3 4 5 6 7 8 91011121314       1 2 3 4 5 6
     * -----------------------------     -------------
     * |1   M             M        |     |           |     |M|  1
     * |  1 M   M                  |     |           |     |M|  2
     * |0 0 1   M         M        |     |           |     |M|  3
     * |      1 M   M              |     |           |     |M|  4
     * |  0 0 0 1   M     F        |     |           |     |M|  5
     * |          1 M   M          |     |           |     |M|  6
     * |      0 0 0 1   M F        |     |           |     |M|  7
     * |              1 M   M      | ... |           | ... |M|  8 sc
     * |          0 0 0 1 F M      |     |           |     |M|  9
     * |0   0   0   0   0 1 M M    |     |           |     |M| 10
     * |              0 0 0 1 M    |     |           |     |M| 11
     * |                  0 0 1 M M|     |      M    |     |M| 12
     * |                      0 1 M|     |      M    |     |M| 13
     * |                      0 0 1|     |      M    |     |M| 14
     * -----------------------------     -------------
     *
     * @param sc_to_sc_matrix Pointer to a SC to SC phenylaline sub-matrix.
     * @param sc_to_bb_matrix Pointer to a SC to BB phenylaline sub-matrix.
     * @param sc_to_sep_matrix unused.
     * @param sc_g_matrix Pointer to a SC g phenylaline sub-matrix.
     * @param sc_to_sc_fillins Pointer to a SC to SC phenylaline fillins
     * sub-matrix.
     * @param sc_to_bb_fillins unused.
     */
    void solve_2_phenylaline(real *const sc_to_sc_matrix,
                             real *const sc_to_bb_matrix,
                             __attribute__((unused)) real *const sc_to_sep_matrix,
                             real *const sc_g_matrix,
                             real *const sc_to_sc_fillins,
                             __attribute__((unused)) real *const sc_to_bb_fillins);

    /**
     * Performs the Gaussian elimination of the sub-diagonal entries of the
     * first SC to SC sub-matrix pointed by SC_TO_SC_MATRIX. Also make 1s in the
     * diagonal entries.
     *
     * It is assumed that SC_TO_SC_MATRIX, SC_TO_G_MATRIX and SC_TO_BB_MATRIX
     * point to the same tyrosine side-chain data.
     *
     * This function has been automatically generated by
     * step_2_3_16_17_generator.c.
     *
     * Initial matrix:
     *
     *               sc                         bb            g
     *  1 2 3 4 5 6 7 8 9101112131415       1 2 3 4 5 6
     * -------------------------------     -------------
     * |x       x                    |     |           |     |x|  1
     * |  x   x             x        |     |           |     |x|  2
     * |    x x   x                  |     |           |     |x|  3
     * |  x x x   x         x        |     |           |     |x|  4
     * |x       x x   x              |     |           |     |x|  5
     * |    x x x x   x              |     |           |     |x|  6
     * |            x x   x          |     |           |     |x|  7
     * |        x x x x   x          | ... |           | ... |x|  8 sc
     * |                x x   x      |     |           |     |x|  9
     * |            x x x x   x      |     |           |     |x| 10
     * |  x   x             x x x    |     |           |     |x| 11
     * |                x x x x x    |     |           |     |x| 12
     * |                    x x x x x|     |      x    |     |x| 13
     * |                        x x x|     |      x    |     |x| 14
     * |                        x x x|     |      x    |     |x| 15
     * -------------------------------     -------------
     *
     * Final matrix:
     *
     *               sc                         bb            g
     *  1 2 3 4 5 6 7 8 9101112131415       1 2 3 4 5 6
     * -------------------------------     -------------
     * |1       M                    |     |           |     |M|  1
     * |  1   M             M        |     |           |     |M|  2
     * |    1 M   M                  |     |           |     |M|  3
     * |  0 0 1   M         M        |     |           |     |M|  4
     * |0       1 M   M              |     |           |     |M|  5
     * |    0 0 0 1   M     F        |     |           |     |M|  6
     * |            1 M   M          |     |           |     |M|  7
     * |        0 0 0 1   M F        | ... |           | ... |M|  8 sc
     * |                1 M   M      |     |           |     |M|  9
     * |            0 0 0 1 F M      |     |           |     |M| 10
     * |  0   0   0   0   0 1 M M    |     |           |     |M| 11
     * |                0 0 0 1 M    |     |           |     |M| 12
     * |                    0 0 1 M M|     |      M    |     |M| 13
     * |                        0 1 M|     |      M    |     |M| 14
     * |                        0 0 1|     |      M    |     |M| 15
     * -------------------------------     -------------
     *
     * @param sc_to_sc_matrix Pointer to a SC to SC tyrosine sub-matrix.
     * @param sc_to_bb_matrix Pointer to a SC to BB tyrosine sub-matrix.
     * @param sc_to_sep_matrix unused.
     * @param sc_g_matrix Pointer to a SC g tyrosine sub-matrix.
     * @param sc_to_sc_fillins Pointer to a SC to SC tyrosine fillins
     * sub-matrix.
     * @param sc_to_bb_fillins unused.
     */
    void solve_2_tyrosine(real *const sc_to_sc_matrix,
                          real *const sc_to_bb_matrix,
                          __attribute__((unused)) real *const sc_to_sep_matrix,
                          real *const sc_g_matrix,
                          real *const sc_to_sc_fillins,
                          __attribute__((unused)) real *const sc_to_bb_fillins);

    /**
     * Performs the Gaussian elimination of the sub-diagonal entries of the
     * first SC to SC sub-matrix pointed by SC_TO_SC_MATRIX. Also make 1s in the
     * diagonal entries.
     *
     * It is assumed that SC_TO_SC_MATRIX, SC_TO_G_MATRIX and SC_TO_BB_MATRIX
     * point to the same tryptophan side-chain data.
     *
     * This function has been automatically generated by
     * ilves_step_2_3_16_17_generator.c.
     *
     * Initial matrix:
     *
     *                   sc                             bb            g
     *  1 2 3 4 5 6 7 8 910111213141516171819       1 2 3 4 5 6
     * ---------------------------------------     -------------
     * |x   x           x                    |     |           |     |x|  1
     * |  x x   x                            |     |           |     |x|  2
     * |x x x   x       x                    |     |           |     |x|  3
     * |      x x   x                        |     |           |     |x|  4
     * |  x x x x   x                        |     |           |     |x|  5
     * |          x x x                      |     |           |     |x|  6
     * |      x x x x x                      |     |           |     |x|  7
     * |          x x x           x   x      |     |           |     |x|  8
     * |x   x           x       x x          |     |           |     |x|  9
     * |                  x   x x            | ... |           | ... |x| 10 sc
     * |                    x x     x        |     |           |     |x| 11
     * |                  x x x x   x        |     |           |     |x| 12
     * |                x x   x x x          |     |           |     |x| 13
     * |              x x       x x   x      |     |           |     |x| 14
     * |                    x x     x x x    |     |           |     |x| 15
     * |              x           x x x x    |     |           |     |x| 16
     * |                            x x x x x|     |      x    |     |x| 17
     * |                                x x x|     |      x    |     |x| 18
     * |                                x x x|     |      x    |     |x| 19
     * ---------------------------------------     -------------
     *
     * Final matrix:
     *
     *                   sc                             bb            g
     *  1 2 3 4 5 6 7 8 910111213141516171819       1 2 3 4 5 6
     * ---------------------------------------     -------------
     * |1   M           M                    |     |           |     |M|  1
     * |  1 M   M                            |     |           |     |M|  2
     * |0 0 1   M       M                    |     |           |     |M|  3
     * |      1 M   M                        |     |           |     |M|  4
     * |  0 0 0 1   M   F                    |     |           |     |M|  5
     * |          1 M M                      |     |           |     |M|  6
     * |      0 0 0 1 M F                    |     |           |     |M|  7
     * |          0 0 1 F         M   M      |     |           |     |M|  8
     * |0   0   0   0 0 1       M M   F      |     |           |     |M|  9
     * |                  1   M M            | ... |           | ... |M| 10 sc
     * |                    1 M     M        |     |           |     |M| 11
     * |                  0 0 1 M   M        |     |           |     |M| 12
     * |                0 0   0 1 M F F      |     |           |     |M| 13
     * |              0 0       0 1 F M      |     |           |     |M| 14
     * |                    0 0 0 0 1 M M    |     |           |     |M| 15
     * |              0 0       0 0 0 1 M    |     |           |     |M| 16
     * |                            0 0 1 M M|     |      M    |     |M| 17
     * |                                0 1 M|     |      M    |     |M| 18
     * |                                0 0 1|     |      M    |     |M| 19
     * ---------------------------------------     -------------
     *
     * @param sc_to_sc_matrix Pointer to a SC to SC tryptophan sub-matrix.
     * @param sc_to_bb_matrix Pointer to a SC to BB tryptophan sub-matrix.
     * @param sc_to_sep_matrix unused.
     * @param sc_g_matrix Pointer to a SC g tryptophan sub-matrix.
     * @param sc_to_sc_fillins unused.
     * @param sc_to_bb_fillins unused.
     */
    void solve_2_tryptophan(real *const sc_to_sc_matrix,
                            real *const sc_to_bb_matrix,
                            __attribute__((unused)) real *const sc_to_sep_matrix,
                            real *const sc_g_matrix,
                            real *const sc_to_sc_fillins,
                            __attribute__((unused)) real *const sc_to_bb_fillins);

    /**
     * Performs the Gaussian elimination of the sub-diagonal entries of the
     * first SC to SC sub-matrix pointed by SC_TO_SC_MATRIX. Also make 1s in the
     * diagonal entries.
     *
     * It is assumed that SC_TO_SC_MATRIX, SC_TO_G_MATRIX and SC_TO_BB_MATRIX
     * point to the same threonine side-chain data.
     *
     * This function has been automatically generated by
     * step_2_3_16_17_generator.c.
     *
     * Initial matrix:
     *
     *       sc                 bb            g
     *  1 2 3 4 5 6 7       1 2 3 4 5 6
     * ---------------     -------------
     * |x x x x      |     |           |     |x|  1
     * |x x x x      |     |           |     |x|  2
     * |x x x x      |     |           |     |x|  3
     * |x x x x   x x| ... |      x    | ... |x|  4 sc
     * |        x x  |     |           |     |x|  5
     * |      x x x x|     |      x    |     |x|  6
     * |      x   x x|     |      x    |     |x|  7
     * ---------------     -------------
     *
     * Final matrix:
     *
     *       sc                 bb            g
     *  1 2 3 4 5 6 7       1 2 3 4 5 6
     * ---------------     -------------
     * |1 M M M      |     |           |     |M|  1
     * |0 1 M M      |     |           |     |M|  2
     * |0 0 1 M      |     |           |     |M|  3
     * |0 0 0 1   M M| ... |      M    | ... |M|  4 sc
     * |        1 M  |     |           |     |M|  5
     * |      0 0 1 M|     |      M    |     |M|  6
     * |      0   0 1|     |      M    |     |M|  7
     * ---------------     -------------
     *
     * @param sc_to_sc_matrix Pointer to a SC to SC threonine sub-matrix.
     * @param sc_to_bb_matrix Pointer to a SC to BB threonine sub-matrix.
     * @param sc_to_sep_matrix unused.
     * @param sc_g_matrix Pointer to a SC g threonine sub-matrix.
     * @param sc_to_sc_fillins unused.
     * @param sc_to_bb_fillins unused.
     */
    void solve_2_threonine(real *const sc_to_sc_matrix,
                           real *const sc_to_bb_matrix,
                           __attribute__((unused)) real *const sc_to_sep_matrix,
                           real *const sc_g_matrix,
                           __attribute__((unused)) real *const sc_to_sc_fillins,
                           __attribute__((unused)) real *const sc_to_bb_fillins);

    /**
     * Performs the Gaussian elimination of the subdiagonal entries of each SC
     * to SC sub-matrix assigned to THREAD. Also make 1s in the diagonal
     * entries.
     *
     * Initial matrix:
     *
     *   sc         bb              g
     *  1 2 3       1
     * -------     ---
     * |x x  |     | |     |x|      1 Note: **
     * |x x x| ... | | ... |x| sc   2
     * |  x x|     |x|     |x|      3
     * -------     ---
     *
     * ** This SC to SC sub-matrix belongs to a linear side chain (unrealistic).
     * The side chain types can be any of the defined in molecule.h.
     *
     * Final matrix:
     *
     *   sc         bb              g
     *  1 2 3       1
     * -------     ---
     * |1 M  |     | |     |M|      1
     * |0 1 M| ... | | ... |M| sc   2
     * |  0 1|     |M|     |M|      3
     * -------     ---
     *
     * @param thread Thread descriptor [0 - NTHREADS - 1].
     */
    void solve_2(const int thread);

    /**
     * Performs the Gaussian elimination of each entry of the first BB to SC
     * sub-matrix pointed by BB_TO_SC_MATRIX.
     *
     * It is assumed that SC_TO_SC_MATRIX, SC_TO_BB_MATRIX, SC_G_MATRIX,
     * BB_TO_SC_MATRIX, BB_TO_BB_MATRIX and BB_G_MATRIX point to the same
     * glycine side-chain data.
     *
     * This function has been automatically generated by
     * step_2_3_16_17_generator.c.
     *
     * Initial matrix:
     *
     * sc           bb            g
     *          1 2 3 4 5 6
     * ---     -------------
     * ---     -------------
     *  .              .
     *  .              .
     *  .              .
     * ---     -------------
     * | |     |x x        |     |x|  1
     * | |     |x x x x x  |     |x|  2
     * | |     |  x x x x  |     |x|  3
     * | | ... |  x x x x  | ... |x|  4 bb
     * | |     |  x x x x x|     |x|  5
     * | |     |        x x|     |x|  6
     * ---     -------------
     *
     * Final matrix:
     *
     * sc           bb            g
     *          1 2 3 4 5 6
     * ---     -------------
     * ---     -------------
     *  .              .
     *  .              .
     *  .              .
     * ---     -------------
     * | |     |x x        |     |x|  1
     * | |     |x x x x x  |     |x|  2
     * | |     |  x x x x  |     |x|  3
     * | | ... |  x x x x  | ... |x|  4 bb
     * | |     |  x x x x x|     |x|  5
     * | |     |        x x|     |x|  6
     * ---     -------------
     *
     * @param sc_to_sc_matrix unused.
     * @param sc_to_bb_matrix unused.
     * @param sc_to_sep_matrix unused.
     * @param sc_g_matrix unused.
     * @param sc_to_bb_fillins unused.
     * @param bb_to_sc_matrix unused.
     * @param bb_to_bb_matrix unused.
     * @param bb_to_sep_matrix unused.
     * @param bb_g_matrix unused.
     * @param bb_to_sc_fillins unused.
     * @param bb_to_sep_fillins unused.
     */
    void solve_3_glycine(__attribute__((unused)) real *const sc_to_sc_matrix,
                         __attribute__((unused)) real *const sc_to_bb_matrix,
                         __attribute__((unused)) real *const sc_to_sep_matrix,
                         __attribute__((unused)) real *const sc_g_matrix,
                         __attribute__((unused)) real *const sc_to_bb_fillins,
                         __attribute__((unused)) real *const bb_to_sc_matrix,
                         __attribute__((unused)) real *const bb_to_bb_matrix,
                         __attribute__((unused)) real *const bb_to_sep_matrix,
                         __attribute__((unused)) real *const bb_g_matrix,
                         __attribute__((unused)) real *const bb_to_sc_fillins,
                         __attribute__((unused)) real *const bb_to_sep_fillins);

    /**
     * Performs the Gaussian elimination of each entry of the first BB to SC
     * sub-matrix pointed by BB_TO_SC_MATRIX.
     *
     * It is assumed that SC_TO_SC_MATRIX, SC_TO_BB_MATRIX, SC_G_MATRIX,
     * BB_TO_SC_MATRIX, BB_TO_BB_MATRIX and BB_G_MATRIX point to the same
     * proline side-chain data.
     *
     * Initial matrix:
     *
     *         sc                  bb           sep       g
     *  1 2 3 4 5 6 7 8 9       1 2 3 4 5
     * -------------------     -----------     -----
     * |1 x         x    |     |      x  |     |   |     |x|  1
     * |0 1         x    |     |      x  |     |   |     |x|  2
     * |    1 x     x x  |     |         |     |   |     |x|  3
     * |    0 1     x x  |     |         |     |   |     |x|  4
     * |        1 x   x x| ... |         | ... |   | ... |x|  5 sc
     * |        0 1   x x|     |         |     |   |     |x|  6
     * |0 0 0 0     1 x  |     |      x  |     |   |     |x|  7
     * |    0 0 0 0 0 1 x|     |      f  |     |   |     |x|  8
     * |        0 0   0 1|     |      f x|     |  x|     |x|  9
     * -------------------     -----------     -----
     *          .                   .            .
     *          .                   .            .
     *          .                   .            .
     * -------------------     -----------     -----
     * |                 |     |x x      |     |x  |     |x|  1
     * |                 |     |x x x x x|     |x  |     |x|  2
     * |                 | ... |  x x x x| ... |   | ... |x|  3 bb
     * |x x         x    |     |  x x x x|     |   |     |x|  4
     * |                x|     |  x x x x|     |  x|     |x|  5
     * -------------------     -----------     -----
     *
     * Final matrix:
     *
     *         sc                  bb           sep       g
     *  1 2 3 4 5 6 7 8 9       1 2 3 4 5
     * -------------------     -----------     -----
     * |1 x         x    |     |      x  |     |   |     |x|  1
     * |0 1         x    |     |      x  |     |   |     |x|  2
     * |    1 x     x x  |     |         |     |   |     |x|  3
     * |    0 1     x x  |     |         |     |   |     |x|  4
     * |        1 x   x x| ... |         | ... |   | ... |x|  5 sc
     * |        0 1   x x|     |         |     |   |     |x|  6
     * |0 0 0 0     1 x  |     |      x  |     |   |     |x|  7
     * |    0 0 0 0 0 1 x|     |      f  |     |   |     |x|  8
     * |        0 0   0 1|     |      f x|     |  x|     |x|  9
     * -------------------     -----------     -----
     *          .           .                    .
     *          .           .                    .
     *          .           .                    .
     * -------------------     -----------     -----
     * |                 |     |x x      |     |x  |     |x|  1
     * |                 |     |x x x x x|     |x  |     |x|  2
     * |                 | ... |  x x x x| ... |   | ... |x|  3 bb
     * |0 0         0 0 0|     |  x x M M|     |  F|     |M|  4
     * |                0|     |  x x M M|     |  M|     |M|  5
     * -------------------     -----------     -----
     *
     * @param sc_to_sc_matrix Pointer to a SC to SC proline sub-matrix.
     * @param sc_to_bb_matrix Pointer to a SC to BB proline sub-matrix.
     * @param sc_to_sep_matrix Pointer to a SC to SEP proline sub-matrix.
     * @param sc_g_matrix Pointer to a SC g proline sub-matrix.
     * @param sc_to_bb_fillins Pointer to a SC to BB proline fillins sub-matrix.
     * @param bb_to_sc_matrix Pointer to a BB to SC proline sub-matrix.
     * @param bb_to_bb_matrix Pointer to a BB to BB proline sub-matrix.
     * @param bb_to_sep_matrix Pointer to a BB to SEP proline sub-matrix.
     * @param bb_g_matrix Pointer to a BB g proline sub-matrix.
     * @param bb_to_sc_fillins Pointer to a BB to SC proline fillins sub-matrix.
     * @param bb_to_sep_fillins Pointer to a BB to SEP proline fillins
     * sub-matrix.
     */
    void solve_3_proline(real *const sc_to_sc_matrix,
                         real *const sc_to_bb_matrix,
                         real *const sc_to_sep_matrix,
                         real *const sc_g_matrix,
                         real *const sc_to_bb_fillins,
                         real *const bb_to_sc_matrix,
                         real *const bb_to_bb_matrix,
                         real *const bb_to_sep_matrix,
                         real *const bb_g_matrix,
                         real *const bb_to_sc_fillins,
                         real *const bb_to_sep_fillins);

    /**
     * Performs the Gaussian elimination of each entry of the first BB to SC
     * sub-matrix pointed by BB_TO_SC_MATRIX.
     *
     * It is assumed that SC_TO_SC_MATRIX, SC_TO_BB_MATRIX, SC_G_MATRIX,
     * BB_TO_SC_MATRIX, BB_TO_BB_MATRIX and BB_G_MATRIX point to the same
     * cysteine side-chain data.
     *
     * This function has been automatically generated by
     * step_2_3_16_17_generator.c.
     *
     * Initial matrix:
     *
     *    sc              bb            g
     *  1 2 3 4       1 2 3 4 5 6
     * ---------     -------------
     * |1 x    |     |           |     |x|  1
     * |0 1 x x|     |      x    |     |x|  2
     * |  0 1 x| ... |      x    | ... |x|  3 sc
     * |  0 0 1|     |      x    |     |x|  4
     * ---------     -------------
     *     .               .
     *     .               .
     *     .               .
     * ---------     -------------
     * |       |     |x x        |     |x|  1
     * |       |     |x x x x x  |     |x|  2
     * |       |     |  x x x x  |     |x|  3
     * |  x x x| ... |  x x x x  | ... |x|  4 bb
     * |       |     |  x x x x x|     |x|  5
     * |       |     |        x x|     |x|  6
     * ---------     -------------
     *
     * Final matrix:
     *
     *    sc              bb            g
     *  1 2 3 4       1 2 3 4 5 6
     * ---------     -------------
     * |1 x    |     |           |     |x|  1
     * |0 1 x x|     |      x    |     |x|  2
     * |  0 1 x| ... |      x    | ... |x|  3 sc
     * |  0 0 1|     |      x    |     |x|  4
     * ---------     -------------
     *     .               .
     *     .               .
     *     .               .
     * ---------     -------------
     * |       |     |x x        |     |x|  1
     * |       |     |x x x x x  |     |x|  2
     * |       |     |  x x x x  |     |x|  3
     * |  0 0 0| ... |  x x M x  | ... |M|  4 bb
     * |       |     |  x x x x x|     |x|  5
     * |       |     |        x x|     |x|  6
     * ---------     -------------
     *
     * @param sc_to_sc_matrix Pointer to a SC to SC cysteine sub-matrix.
     * @param sc_to_bb_matrix Pointer to a SC to BB cysteine sub-matrix.
     * @param sc_to_sep_matrix unused.
     * @param sc_g_matrix Pointer to a SC g cysteine sub-matrix.
     * @param sc_to_bb_fillins unused.
     * @param bb_to_sc_matrix Pointer to a BB to SC cysteine sub-matrix.
     * @param bb_to_bb_matrix Pointer to a BB to BB cysteine sub-matrix.
     * @param bb_to_sep_matrix unused.
     * @param bb_g_matrix Pointer to a BB g cysteine sub-matrix.
     * @param bb_to_sc_fillins unused.
     * @param bb_to_sep_fillins unused.
     */
    void solve_3_cysteine(real *const sc_to_sc_matrix,
                          real *const sc_to_bb_matrix,
                          __attribute__((unused)) real *const sc_to_sep_matrix,
                          real *const sc_g_matrix,
                          __attribute__((unused)) real *const sc_to_bb_fillins,
                          real *const bb_to_sc_matrix,
                          real *const bb_to_bb_matrix,
                          __attribute__((unused)) real *const bb_to_sep_matrix,
                          real *const bb_g_matrix,
                          __attribute__((unused)) real *const bb_to_sc_fillins,
                          __attribute__((unused)) real *const bb_to_sep_fillins);

    /**
     * Performs the Gaussian elimination of each entry of the first BB to SC
     * sub-matrix pointed by BB_TO_SC_MATRIX.
     *
     * It is assumed that SC_TO_SC_MATRIX, SC_TO_BB_MATRIX, SC_G_MATRIX,
     * BB_TO_SC_MATRIX, BB_TO_BB_MATRIX and BB_G_MATRIX point to the same
     * methionine side-chain data.
     *
     * This function has been automatically generated by
     * step_2_3_16_17_generator.c.
     *
     * Initial matrix:
     *
     *          sc                    bb            g
     *  1 2 3 4 5 6 7 8 910       1 2 3 4 5 6
     * ---------------------     -------------
     * |1 x x x            |     |           |     |x|  1
     * |0 1 x x            |     |           |     |x|  2
     * |0 0 1 x            |     |           |     |x|  3
     * |0 0 0 1 x          |     |           |     |x|  4
     * |      0 1 x x x    |     |           |     |x|  5
     * |        0 1 x x    | ... |           | ... |x|  6 sc
     * |        0 0 1 x    |     |           |     |x|  7
     * |        0 0 0 1 x x|     |      x    |     |x|  8
     * |              0 1 x|     |      x    |     |x|  9
     * |              0 0 1|     |      x    |     |x| 10
     * ---------------------     -------------
     *           .                     .
     *           .                     .
     *           .                     .
     * ---------------------     -------------
     * |                   |     |x x        |     |x|  1
     * |                   |     |x x x x x  |     |x|  2
     * |                   |     |  x x x x  |     |x|  3
     * |              x x x| ... |  x x x x  | ... |x|  4 bb
     * |                   |     |  x x x x x|     |x|  5
     * |                   |     |        x x|     |x|  6
     * ---------------------     -------------
     *
     * Final matrix:
     *
     *          sc                    bb            g
     *  1 2 3 4 5 6 7 8 910       1 2 3 4 5 6
     * ---------------------     -------------
     * |1 x x x            |     |           |     |x|  1
     * |0 1 x x            |     |           |     |x|  2
     * |0 0 1 x            |     |           |     |x|  3
     * |0 0 0 1 x          |     |           |     |x|  4
     * |      0 1 x x x    |     |           |     |x|  5
     * |        0 1 x x    | ... |           | ... |x|  6 sc
     * |        0 0 1 x    |     |           |     |x|  7
     * |        0 0 0 1 x x|     |      x    |     |x|  8
     * |              0 1 x|     |      x    |     |x|  9
     * |              0 0 1|     |      x    |     |x| 10
     * ---------------------     -------------
     *           .                     .
     *           .                     .
     *           .                     .
     * ---------------------     -------------
     * |                   |     |x x        |     |x|  1
     * |                   |     |x x x x x  |     |x|  2
     * |                   |     |  x x x x  |     |x|  3
     * |              0 0 0| ... |  x x M x  | ... |M|  4 bb
     * |                   |     |  x x x x x|     |x|  5
     * |                   |     |        x x|     |x|  6
     * ---------------------     -------------
     *
     * @param sc_to_sc_matrix Pointer to a SC to SC methionine sub-matrix.
     * @param sc_to_bb_matrix Pointer to a SC to BB methionine sub-matrix.
     * @param sc_to_sep_matrix unused.
     * @param sc_g_matrix Pointer to a SC g methionine sub-matrix.
     * @param sc_to_bb_fillins unused.
     * @param bb_to_sc_matrix Pointer to a BB to SC methionine sub-matrix.
     * @param bb_to_bb_matrix Pointer to a BB to BB methionine sub-matrix.
     * @param bb_to_sep_matrix unused.
     * @param bb_g_matrix Pointer to a BB g methionine sub-matrix.
     * @param bb_to_sc_fillins unused.
     * @param bb_to_sep_fillins unused.
     */
    void solve_3_methionine(real *const sc_to_sc_matrix,
                            real *const sc_to_bb_matrix,
                            __attribute__((unused)) real *const sc_to_sep_matrix,
                            real *const sc_g_matrix,
                            __attribute__((unused)) real *const sc_to_bb_fillins,
                            real *const bb_to_sc_matrix,
                            real *const bb_to_bb_matrix,
                            __attribute__((unused)) real *const bb_to_sep_matrix,
                            real *const bb_g_matrix,
                            __attribute__((unused)) real *const bb_to_sc_fillins,
                            __attribute__((unused)) real *const bb_to_sep_fillins);

    /**
     * Performs the Gaussian elimination of each entry of the first BB to SC
     * sub-matrix pointed by BB_TO_SC_MATRIX.
     *
     * It is assumed that SC_TO_SC_MATRIX, SC_TO_BB_MATRIX, SC_G_MATRIX,
     * BB_TO_SC_MATRIX, BB_TO_BB_MATRIX and BB_G_MATRIX point to the same
     * alaline side-chain data.
     *
     * This function has been automatically generated by
     * step_2_3_16_17_generator.c.
     *
     * Initial matrix:
     *
     *   sc             bb            g
     *  1 2 3       1 2 3 4 5 6
     * -------     -------------
     * |1 x x|     |      x    |     |x|  1
     * |0 1 x| ... |      x    | ... |x|  2 sc
     * |0 0 1|     |      x    |     |x|  3
     * -------     -------------
     *    .              .
     *    .              .
     *    .              .
     * -------     -------------
     * |     |     |x x        |     |x|  1
     * |     |     |x x x x x  |     |x|  2
     * |     |     |  x x x x  |     |x|  3
     * |x x x| ... |  x x x x  | ... |x|  4 bb
     * |     |     |  x x x x x|     |x|  5
     * |     |     |        x x|     |x|  6
     * -------     -------------
     *
     * Final matrix:
     *
     *   sc             bb            g
     *  1 2 3       1 2 3 4 5 6
     * -------     -------------
     * |1 x x|     |      x    |     |x|  1
     * |0 1 x| ... |      x    | ... |x|  2 sc
     * |0 0 1|     |      x    |     |x|  3
     * -------     -------------
     *    .              .
     *    .              .
     *    .              .
     * -------     -------------
     * |     |     |x x        |     |x|  1
     * |     |     |x x x x x  |     |x|  2
     * |     |     |  x x x x  |     |x|  3
     * |0 0 0| ... |  x x M x  | ... |M|  4 bb
     * |     |     |  x x x x x|     |x|  5
     * |     |     |        x x|     |x|  6
     * -------     -------------
     *
     * @param sc_to_sc_matrix Pointer to a SC to SC alaline sub-matrix.
     * @param sc_to_bb_matrix Pointer to a SC to BB alaline sub-matrix.
     * @param sc_to_sep_matrix unused.
     * @param sc_g_matrix Pointer to a SC g alaline sub-matrix.
     * @param sc_to_bb_fillins unused.
     * @param bb_to_sc_matrix Pointer to a BB to SC alaline sub-matrix.
     * @param bb_to_bb_matrix Pointer to a BB to BB alaline sub-matrix.
     * @param bb_to_sep_matrix unused.
     * @param bb_g_matrix Pointer to a BB g alaline sub-matrix.
     * @param bb_to_sc_fillins unused.
     * @param bb_to_sep_fillins unused.
     */
    void solve_3_alaline(real *const sc_to_sc_matrix,
                         real *const sc_to_bb_matrix,
                         __attribute__((unused)) real *const sc_to_sep_matrix,
                         real *const sc_g_matrix,
                         __attribute__((unused)) real *const sc_to_bb_fillins,
                         real *const bb_to_sc_matrix,
                         real *const bb_to_bb_matrix,
                         __attribute__((unused)) real *const bb_to_sep_matrix,
                         real *const bb_g_matrix,
                         __attribute__((unused)) real *const bb_to_sc_fillins,
                         __attribute__((unused)) real *const bb_to_sep_fillins);

    /**
     * Performs the Gaussian elimination of each entry of the first BB to SC
     * sub-matrix pointed by BB_TO_SC_MATRIX.
     *
     * It is assumed that SC_TO_SC_MATRIX, SC_TO_BB_MATRIX, SC_G_MATRIX,
     * BB_TO_SC_MATRIX, BB_TO_BB_MATRIX and BB_G_MATRIX point to the same valine
     * side-chain data.
     *
     * This function has been automatically generated by
     * step_2_3_16_17_generator.c.
     *
     * Initial matrix:
     *
     *         sc                   bb            g
     *  1 2 3 4 5 6 7 8 9       1 2 3 4 5 6
     * -------------------     -------------
     * |1 x x x          |     |           |     |x|  1
     * |0 1 x x          |     |           |     |x|  2
     * |0 0 1 x          |     |           |     |x|  3
     * |0 0 0 1       x x|     |      x    |     |x|  4
     * |        1 x x x  | ... |           | ... |x|  5 sc
     * |        0 1 x x  |     |           |     |x|  6
     * |        0 0 1 x  |     |           |     |x|  7
     * |      0 0 0 0 1 x|     |      x    |     |x|  8
     * |      0       0 1|     |      x    |     |x|  9
     * -------------------     -------------
     *          .                    .
     *          .                    .
     *          .                    .
     * -------------------     -------------
     * |                 |     |x x        |     |x|  1
     * |                 |     |x x x x x  |     |x|  2
     * |                 |     |  x x x x  |     |x|  3
     * |      x       x x| ... |  x x x x  | ... |x|  4 bb
     * |                 |     |  x x x x x|     |x|  5
     * |                 |     |        x x|     |x|  6
     * -------------------     -------------
     *
     * Final matrix:
     *
     *         sc                   bb            g
     *  1 2 3 4 5 6 7 8 9       1 2 3 4 5 6
     * -------------------     -------------
     * |1 x x x          |     |           |     |x|  1
     * |0 1 x x          |     |           |     |x|  2
     * |0 0 1 x          |     |           |     |x|  3
     * |0 0 0 1       x x|     |      x    |     |x|  4
     * |        1 x x x  | ... |           | ... |x|  5 sc
     * |        0 1 x x  |     |           |     |x|  6
     * |        0 0 1 x  |     |           |     |x|  7
     * |      0 0 0 0 1 x|     |      x    |     |x|  8
     * |      0       0 1|     |      x    |     |x|  9
     * -------------------     -------------
     *          .                    .
     *          .                    .
     *          .                    .
     * -------------------     -------------
     * |                 |     |x x        |     |x|  1
     * |                 |     |x x x x x  |     |x|  2
     * |                 |     |  x x x x  |     |x|  3
     * |      0       0 0| ... |  x x M x  | ... |M|  4 bb
     * |                 |     |  x x x x x|     |x|  5
     * |                 |     |        x x|     |x|  6
     * -------------------     -------------
     *
     * @param sc_to_sc_matrix Pointer to a SC to SC valine sub-matrix.
     * @param sc_to_bb_matrix Pointer to a SC to BB valine sub-matrix.
     * @param sc_to_sep_matrix unused.
     * @param sc_g_matrix Pointer to a SC g valine sub-matrix.
     * @param sc_to_bb_fillins unused.
     * @param bb_to_sc_matrix Pointer to a BB to SC valine sub-matrix.
     * @param bb_to_bb_matrix Pointer to a BB to BB valine sub-matrix.
     * @param bb_to_sep_matrix unused.
     * @param bb_g_matrix Pointer to a BB g valine sub-matrix.
     * @param bb_to_sc_fillins unused.
     * @param bb_to_sep_fillins unused.
     */
    void solve_3_valine(real *const sc_to_sc_matrix,
                        real *const sc_to_bb_matrix,
                        __attribute__((unused)) real *const sc_to_sep_matrix,
                        real *const sc_g_matrix,
                        __attribute__((unused)) real *const sc_to_bb_fillins,
                        real *const bb_to_sc_matrix,
                        real *const bb_to_bb_matrix,
                        __attribute__((unused)) real *const bb_to_sep_matrix,
                        real *const bb_g_matrix,
                        __attribute__((unused)) real *const bb_to_sc_fillins,
                        __attribute__((unused)) real *const bb_to_sep_fillins);

    /**
     * Performs the Gaussian elimination of each entry of the first BB to SC
     * sub-matrix pointed by BB_TO_SC_MATRIX.
     *
     * It is assumed that SC_TO_SC_MATRIX, SC_TO_BB_MATRIX, SC_G_MATRIX,
     * BB_TO_SC_MATRIX, BB_TO_BB_MATRIX and BB_G_MATRIX point to the same
     * leucine side-chain data.
     *
     * This function has been automatically generated by
     * step_2_3_16_17_generator.c.
     *
     * Initial matrix:
     *
     *            sc                      bb            g
     *  1 2 3 4 5 6 7 8 9101112       1 2 3 4 5 6
     * -------------------------     -------------
     * |1 x x x                |     |           |     |x|  1
     * |0 1 x x                |     |           |     |x|  2
     * |0 0 1 x                |     |           |     |x|  3
     * |0 0 0 1       x x x    |     |           |     |x|  4
     * |        1 x x x        |     |           |     |x|  5
     * |        0 1 x x        |     |           |     |x|  6
     * |        0 0 1 x        | ... |           | ... |x|  7 sc
     * |      0 0 0 0 1 x x    |     |           |     |x|  8
     * |      0       0 1 x    |     |           |     |x|  9
     * |      0       0 0 1 x x|     |      x    |     |x| 10
     * |                  0 1 x|     |      x    |     |x| 11
     * |                  0 0 1|     |      x    |     |x| 12
     * -------------------------     -------------
     *             .                       .
     *             .                       .
     *             .                       .
     * -------------------------     -------------
     * |                       |     |x x        |     |x|  1
     * |                       |     |x x x x x  |     |x|  2
     * |                       |     |  x x x x  |     |x|  3
     * |                  x x x| ... |  x x x x  | ... |x|  4 bb
     * |                       |     |  x x x x x|     |x|  5
     * |                       |     |        x x|     |x|  6
     * -------------------------     -------------
     *
     * Final matrix:
     *
     *            sc                      bb            g
     *  1 2 3 4 5 6 7 8 9101112       1 2 3 4 5 6
     * -------------------------     -------------
     * |1 x x x                |     |           |     |x|  1
     * |0 1 x x                |     |           |     |x|  2
     * |0 0 1 x                |     |           |     |x|  3
     * |0 0 0 1       x x x    |     |           |     |x|  4
     * |        1 x x x        |     |           |     |x|  5
     * |        0 1 x x        |     |           |     |x|  6
     * |        0 0 1 x        | ... |           | ... |x|  7 sc
     * |      0 0 0 0 1 x x    |     |           |     |x|  8
     * |      0       0 1 x    |     |           |     |x|  9
     * |      0       0 0 1 x x|     |      x    |     |x| 10
     * |                  0 1 x|     |      x    |     |x| 11
     * |                  0 0 1|     |      x    |     |x| 12
     * -------------------------     -------------
     *             .                       .
     *             .                       .
     *             .                       .
     * -------------------------     -------------
     * |                       |     |x x        |     |x|  1
     * |                       |     |x x x x x  |     |x|  2
     * |                       |     |  x x x x  |     |x|  3
     * |                  0 0 0| ... |  x x M x  | ... |M|  4 bb
     * |                       |     |  x x x x x|     |x|  5
     * |                       |     |        x x|     |x|  6
     * -------------------------     -------------
     *
     * @param sc_to_sc_matrix Pointer to a SC to SC leucine sub-matrix.
     * @param sc_to_bb_matrix Pointer to a SC to BB leucine sub-matrix.
     * @param sc_to_sep_matrix unused.
     * @param sc_g_matrix Pointer to a SC g leucine sub-matrix.
     * @param sc_to_bb_fillins unused.
     * @param bb_to_sc_matrix Pointer to a BB to SC leucine sub-matrix.
     * @param bb_to_bb_matrix Pointer to a BB to BB leucine sub-matrix.
     * @param bb_to_sep_matrix unused.
     * @param bb_g_matrix Pointer to a BB g leucine sub-matrix.
     * @param bb_to_sc_fillins unused.
     * @param bb_to_sep_fillins unused.
     */
    void solve_3_leucine(real *const sc_to_sc_matrix,
                         real *const sc_to_bb_matrix,
                         __attribute__((unused)) real *const sc_to_sep_matrix,
                         real *const sc_g_matrix,
                         __attribute__((unused)) real *const sc_to_bb_fillins,
                         real *const bb_to_sc_matrix,
                         real *const bb_to_bb_matrix,
                         __attribute__((unused)) real *const bb_to_sep_matrix,
                         real *const bb_g_matrix,
                         __attribute__((unused)) real *const bb_to_sc_fillins,
                         __attribute__((unused)) real *const bb_to_sep_fillins);

    /**
     * Performs the Gaussian elimination of each entry of the first BB to SC
     * sub-matrix pointed by BB_TO_SC_MATRIX.
     *
     * It is assumed that SC_TO_SC_MATRIX, SC_TO_BB_MATRIX, SC_G_MATRIX,
     * BB_TO_SC_MATRIX, BB_TO_BB_MATRIX and BB_G_MATRIX point to the same
     * isoleucine side-chain data.
     *
     * This function has been automatically generated by
     * step_2_3_16_17_generator.c.
     *
     * Initial matrix:
     *
     *            sc                      bb            g
     *  1 2 3 4 5 6 7 8 9101112       1 2 3 4 5 6
     * -------------------------     -------------
     * |1 x x x                |     |           |     |x|  1
     * |0 1 x x                |     |           |     |x|  2
     * |0 0 1 x                |     |           |     |x|  3
     * |0 0 0 1 x x x          |     |           |     |x|  4
     * |      0 1 x x          |     |           |     |x|  5
     * |      0 0 1 x          |     |           |     |x|  6
     * |      0 0 0 1       x x| ... |      x    | ... |x|  7 sc
     * |              1 x x x  |     |           |     |x|  8
     * |              0 1 x x  |     |           |     |x|  9
     * |              0 0 1 x  |     |           |     |x| 10
     * |            0 0 0 0 1 x|     |      x    |     |x| 11
     * |            0       0 1|     |      x    |     |x| 12
     * -------------------------     -------------
     *             .                       .
     *             .                       .
     *             .                       .
     * -------------------------     -------------
     * |                       |     |x x        |     |x|  1
     * |                       |     |x x x x x  |     |x|  2
     * |                       |     |  x x x x  |     |x|  3
     * |            x       x x| ... |  x x x x  | ... |x|  4 bb
     * |                       |     |  x x x x x|     |x|  5
     * |                       |     |        x x|     |x|  6
     * -------------------------     -------------
     *
     * Final matrix:
     *
     *            sc                      bb            g
     *  1 2 3 4 5 6 7 8 9101112       1 2 3 4 5 6
     * -------------------------     -------------
     * |1 x x x                |     |           |     |x|  1
     * |0 1 x x                |     |           |     |x|  2
     * |0 0 1 x                |     |           |     |x|  3
     * |0 0 0 1 x x x          |     |           |     |x|  4
     * |      0 1 x x          |     |           |     |x|  5
     * |      0 0 1 x          |     |           |     |x|  6
     * |      0 0 0 1       x x| ... |      x    | ... |x|  7 sc
     * |              1 x x x  |     |           |     |x|  8
     * |              0 1 x x  |     |           |     |x|  9
     * |              0 0 1 x  |     |           |     |x| 10
     * |            0 0 0 0 1 x|     |      x    |     |x| 11
     * |            0       0 1|     |      x    |     |x| 12
     * -------------------------     -------------
     *             .                       .
     *             .                       .
     *             .                       .
     * -------------------------     -------------
     * |                       |     |x x        |     |x|  1
     * |                       |     |x x x x x  |     |x|  2
     * |                       |     |  x x x x  |     |x|  3
     * |            0       0 0| ... |  x x M x  | ... |M|  4 bb
     * |                       |     |  x x x x x|     |x|  5
     * |                       |     |        x x|     |x|  6
     * -------------------------     -------------
     *
     * @param sc_to_sc_matrix Pointer to a SC to SC isoleucine sub-matrix.
     * @param sc_to_bb_matrix Pointer to a SC to BB isoleucine sub-matrix.
     * @param sc_to_sep_matrix unused.
     * @param sc_g_matrix Pointer to a SC g isoleucine sub-matrix.
     * @param sc_to_bb_fillins unused.
     * @param bb_to_sc_matrix Pointer to a BB to SC isoleucine sub-matrix.
     * @param bb_to_bb_matrix Pointer to a BB to BB isoleucine sub-matrix.
     * @param bb_to_sep_matrix unused.
     * @param bb_g_matrix Pointer to a BB g isoleucine sub-matrix.
     * @param bb_to_sc_fillins unused.
     * @param bb_to_sep_fillins unused.
     */
    void solve_3_isoleucine(real *const sc_to_sc_matrix,
                            real *const sc_to_bb_matrix,
                            __attribute__((unused)) real *const sc_to_sep_matrix,
                            real *const sc_g_matrix,
                            __attribute__((unused)) real *const sc_to_bb_fillins,
                            real *const bb_to_sc_matrix,
                            real *const bb_to_bb_matrix,
                            __attribute__((unused)) real *const bb_to_sep_matrix,
                            real *const bb_g_matrix,
                            __attribute__((unused)) real *const bb_to_sc_fillins,
                            __attribute__((unused)) real *const bb_to_sep_fillins);

    /**
     * Performs the Gaussian elimination of each entry of the first BB to SC
     * sub-matrix pointed by BB_TO_SC_MATRIX.
     *
     * It is assumed that SC_TO_SC_MATRIX, SC_TO_BB_MATRIX, SC_G_MATRIX,
     * BB_TO_SC_MATRIX, BB_TO_BB_MATRIX and BB_G_MATRIX point to the same
     * aspartic_acid side-chain data.
     *
     * This function has been automatically generated by
     * step_2_3_16_17_generator.c.
     *
     * Initial matrix:
     *
     *     sc               bb            g
     *  1 2 3 4 5       1 2 3 4 5 6
     * -----------     -------------
     * |1 x x    |     |           |     |x|  1
     * |0 1 x    |     |           |     |x|  2
     * |0 0 1 x x| ... |      x    | ... |x|  3 sc
     * |    0 1 x|     |      x    |     |x|  4
     * |    0 0 1|     |      x    |     |x|  5
     * -----------     -------------
     *      .                .
     *      .                .
     *      .                .
     * -----------     -------------
     * |         |     |x x        |     |x|  1
     * |         |     |x x x x x  |     |x|  2
     * |         |     |  x x x x  |     |x|  3
     * |    x x x| ... |  x x x x  | ... |x|  4 bb
     * |         |     |  x x x x x|     |x|  5
     * |         |     |        x x|     |x|  6
     * -----------     -------------
     *
     * Final matrix:
     *
     *     sc               bb            g
     *  1 2 3 4 5       1 2 3 4 5 6
     * -----------     -------------
     * |1 x x    |     |           |     |x|  1
     * |0 1 x    |     |           |     |x|  2
     * |0 0 1 x x| ... |      x    | ... |x|  3 sc
     * |    0 1 x|     |      x    |     |x|  4
     * |    0 0 1|     |      x    |     |x|  5
     * -----------     -------------
     *      .                .
     *      .                .
     *      .                .
     * -----------     -------------
     * |         |     |x x        |     |x|  1
     * |         |     |x x x x x  |     |x|  2
     * |         |     |  x x x x  |     |x|  3
     * |    0 0 0| ... |  x x M x  | ... |M|  4 bb
     * |         |     |  x x x x x|     |x|  5
     * |         |     |        x x|     |x|  6
     * -----------     -------------
     *
     * @param sc_to_sc_matrix Pointer to a SC to SC aspartic_acid sub-matrix.
     * @param sc_to_bb_matrix Pointer to a SC to BB aspartic_acid sub-matrix.
     * @param sc_to_sep_matrix unused.
     * @param sc_g_matrix Pointer to a SC g aspartic_acid sub-matrix.
     * @param sc_to_bb_fillins unused.
     * @param bb_to_sc_matrix Pointer to a BB to SC aspartic_acid sub-matrix.
     * @param bb_to_bb_matrix Pointer to a BB to BB aspartic_acid sub-matrix.
     * @param bb_to_sep_matrix unused.
     * @param bb_g_matrix Pointer to a BB g aspartic_acid sub-matrix.
     * @param bb_to_sc_fillins unused.
     * @param bb_to_sep_fillins unused.
     */
    void solve_3_aspartic_acid(real *const sc_to_sc_matrix,
                               real *const sc_to_bb_matrix,
                               __attribute__((unused)) real *const sc_to_sep_matrix,
                               real *const sc_g_matrix,
                               __attribute__((unused)) real *const sc_to_bb_fillins,
                               real *const bb_to_sc_matrix,
                               real *const bb_to_bb_matrix,
                               __attribute__((unused)) real *const bb_to_sep_matrix,
                               real *const bb_g_matrix,
                               __attribute__((unused)) real *const bb_to_sc_fillins,
                               __attribute__((unused))
                               real *const bb_to_sep_fillins);

    /**
     * Performs the Gaussian elimination of each entry of the first BB to SC
     * sub-matrix pointed by BB_TO_SC_MATRIX.
     *
     * It is assumed that SC_TO_SC_MATRIX, SC_TO_BB_MATRIX, SC_G_MATRIX,
     * BB_TO_SC_MATRIX, BB_TO_BB_MATRIX and BB_G_MATRIX point to the same
     * glutamic_acid side-chain data.
     *
     * This function has been automatically generated by
     * step_2_3_16_17_generator.c.
     *
     * Initial matrix:
     *
     *        sc                  bb            g
     *  1 2 3 4 5 6 7 8       1 2 3 4 5 6
     * -----------------     -------------
     * |1 x x          |     |           |     |x|  1
     * |0 1 x          |     |           |     |x|  2
     * |0 0 1 x x x    |     |           |     |x|  3
     * |    0 1 x x    |     |           |     |x|  4
     * |    0 0 1 x    | ... |           | ... |x|  5 sc
     * |    0 0 0 1 x x|     |      x    |     |x|  6
     * |          0 1 x|     |      x    |     |x|  7
     * |          0 0 1|     |      x    |     |x|  8
     * -----------------     -------------
     *         .                   .
     *         .                   .
     *         .                   .
     * -----------------     -------------
     * |               |     |x x        |     |x|  1
     * |               |     |x x x x x  |     |x|  2
     * |               |     |  x x x x  |     |x|  3
     * |          x x x| ... |  x x x x  | ... |x|  4 bb
     * |               |     |  x x x x x|     |x|  5
     * |               |     |        x x|     |x|  6
     * -----------------     -------------
     *
     * Final matrix:
     *
     *        sc                  bb            g
     *  1 2 3 4 5 6 7 8       1 2 3 4 5 6
     * -----------------     -------------
     * |1 x x          |     |           |     |x|  1
     * |0 1 x          |     |           |     |x|  2
     * |0 0 1 x x x    |     |           |     |x|  3
     * |    0 1 x x    |     |           |     |x|  4
     * |    0 0 1 x    | ... |           | ... |x|  5 sc
     * |    0 0 0 1 x x|     |      x    |     |x|  6
     * |          0 1 x|     |      x    |     |x|  7
     * |          0 0 1|     |      x    |     |x|  8
     * -----------------     -------------
     *         .                   .
     *         .                   .
     *         .                   .
     * -----------------     -------------
     * |               |     |x x        |     |x|  1
     * |               |     |x x x x x  |     |x|  2
     * |               |     |  x x x x  |     |x|  3
     * |          0 0 0| ... |  x x M x  | ... |M|  4 bb
     * |               |     |  x x x x x|     |x|  5
     * |               |     |        x x|     |x|  6
     * -----------------     -------------
     *
     * @param sc_to_sc_matrix Pointer to a SC to SC glutamic_acid sub-matrix.
     * @param sc_to_bb_matrix Pointer to a SC to BB glutamic_acid sub-matrix.
     * @param sc_to_sep_matrix unused.
     * @param sc_g_matrix Pointer to a SC g glutamic_acid sub-matrix.
     * @param sc_to_bb_fillins unused.
     * @param bb_to_sc_matrix Pointer to a BB to SC glutamic_acid sub-matrix.
     * @param bb_to_bb_matrix Pointer to a BB to BB glutamic_acid sub-matrix.
     * @param bb_to_sep_matrix unused.
     * @param bb_g_matrix Pointer to a BB g glutamic_acid sub-matrix.
     * @param bb_to_sc_fillins unused.
     * @param bb_to_sep_fillins unused.
     */
    void solve_3_glutamic_acid(real *const sc_to_sc_matrix,
                               real *const sc_to_bb_matrix,
                               __attribute__((unused)) real *const sc_to_sep_matrix,
                               real *const sc_g_matrix,
                               __attribute__((unused)) real *const sc_to_bb_fillins,
                               real *const bb_to_sc_matrix,
                               real *const bb_to_bb_matrix,
                               __attribute__((unused)) real *const bb_to_sep_matrix,
                               real *const bb_g_matrix,
                               __attribute__((unused)) real *const bb_to_sc_fillins,
                               __attribute__((unused))
                               real *const bb_to_sep_fillins);
    /**
     * Performs the Gaussian elimination of each entry of the first BB to SC
     * sub-matrix pointed by BB_TO_SC_MATRIX.
     *
     * It is assumed that SC_TO_SC_MATRIX, SC_TO_BB_MATRIX, SC_G_MATRIX,
     * BB_TO_SC_MATRIX, BB_TO_BB_MATRIX and BB_G_MATRIX point to the same
     * asparagine side-chain data.
     *
     * This function has been automatically generated by
     * step_2_3_16_17_generator.c.
     *
     * Initial matrix:
     *
     *       sc                 bb            g
     *  1 2 3 4 5 6 7       1 2 3 4 5 6
     * ---------------     -------------
     * |1 x x        |     |           |     |x|  1
     * |0 1 x        |     |           |     |x|  2
     * |0 0 1 x x    |     |           |     |x|  3
     * |    0 1 x    | ... |           | ... |x|  4 sc
     * |    0 0 1 x x|     |      x    |     |x|  5
     * |        0 1 x|     |      x    |     |x|  6
     * |        0 0 1|     |      x    |     |x|  7
     * ---------------     -------------
     *        .                  .
     *        .                  .
     *        .                  .
     * ---------------     -------------
     * |             |     |x x        |     |x|  1
     * |             |     |x x x x x  |     |x|  2
     * |             |     |  x x x x  |     |x|  3
     * |        x x x| ... |  x x x x  | ... |x|  4 bb
     * |             |     |  x x x x x|     |x|  5
     * |             |     |        x x|     |x|  6
     * ---------------     -------------
     *
     * Final matrix:
     *
     *       sc                 bb            g
     *  1 2 3 4 5 6 7       1 2 3 4 5 6
     * ---------------     -------------
     * |1 x x        |     |           |     |x|  1
     * |0 1 x        |     |           |     |x|  2
     * |0 0 1 x x    |     |           |     |x|  3
     * |    0 1 x    | ... |           | ... |x|  4 sc
     * |    0 0 1 x x|     |      x    |     |x|  5
     * |        0 1 x|     |      x    |     |x|  6
     * |        0 0 1|     |      x    |     |x|  7
     * ---------------     -------------
     *        .                  .
     *        .                  .
     *        .                  .
     * ---------------     -------------
     * |             |     |x x        |     |x|  1
     * |             |     |x x x x x  |     |x|  2
     * |             |     |  x x x x  |     |x|  3
     * |        0 0 0| ... |  x x M x  | ... |M|  4 bb
     * |             |     |  x x x x x|     |x|  5
     * |             |     |        x x|     |x|  6
     * ---------------     -------------
     *
     * @param sc_to_sc_matrix Pointer to a SC to SC asparagine sub-matrix.
     * @param sc_to_bb_matrix Pointer to a SC to BB asparagine sub-matrix.
     * @param sc_to_sep_matrix unused.
     * @param sc_g_matrix Pointer to a SC g asparagine sub-matrix.
     * @param sc_to_bb_fillins unused.
     * @param bb_to_sc_matrix Pointer to a BB to SC asparagine sub-matrix.
     * @param bb_to_bb_matrix Pointer to a BB to BB asparagine sub-matrix.
     * @param bb_to_sep_matrix unused.
     * @param bb_g_matrix Pointer to a BB g asparagine sub-matrix.
     * @param bb_to_sc_fillins unused.
     * @param bb_to_sep_fillins unused.
     */
    void solve_3_asparagine(real *const sc_to_sc_matrix,
                            real *const sc_to_bb_matrix,
                            __attribute__((unused)) real *const sc_to_sep_matrix,
                            real *const sc_g_matrix,
                            __attribute__((unused)) real *const sc_to_bb_fillins,
                            real *const bb_to_sc_matrix,
                            real *const bb_to_bb_matrix,
                            __attribute__((unused)) real *const bb_to_sep_matrix,
                            real *const bb_g_matrix,
                            __attribute__((unused)) real *const bb_to_sc_fillins,
                            __attribute__((unused)) real *const bb_to_sep_fillins);

    /**
     * Performs the Gaussian elimination of each entry of the first BB to SC
     * sub-matrix pointed by BB_TO_SC_MATRIX.
     *
     * It is assumed that SC_TO_SC_MATRIX, SC_TO_BB_MATRIX, SC_G_MATRIX,
     * BB_TO_SC_MATRIX, BB_TO_BB_MATRIX and BB_G_MATRIX point to the same
     * glutamine side-chain data.
     *
     * This function has been automatically generated by
     * step_2_3_16_17_generator.c.
     *
     * Initial matrix:
     *
     *          sc                    bb            g
     *  1 2 3 4 5 6 7 8 910       1 2 3 4 5 6
     * ---------------------     -------------
     * |1 x x              |     |           |     |x|  1
     * |0 1 x              |     |           |     |x|  2
     * |0 0 1 x x          |     |           |     |x|  3
     * |    0 1 x          |     |           |     |x|  4
     * |    0 0 1 x x x    |     |           |     |x|  5
     * |        0 1 x x    | ... |           | ... |x|  6 sc
     * |        0 0 1 x    |     |           |     |x|  7
     * |        0 0 0 1 x x|     |      x    |     |x|  8
     * |              0 1 x|     |      x    |     |x|  9
     * |              0 0 1|     |      x    |     |x| 10
     * ---------------------     -------------
     *           .                     .
     *           .                     .
     *           .                     .
     * ---------------------     -------------
     * |                   |     |x x        |     |x|  1
     * |                   |     |x x x x x  |     |x|  2
     * |                   |     |  x x x x  |     |x|  3
     * |              x x x| ... |  x x x x  | ... |x|  4 bb
     * |                   |     |  x x x x x|     |x|  5
     * |                   |     |        x x|     |x|  6
     * ---------------------     -------------
     *
     * Final matrix:
     *
     *          sc                    bb            g
     *  1 2 3 4 5 6 7 8 910       1 2 3 4 5 6
     * ---------------------     -------------
     * |1 x x              |     |           |     |x|  1
     * |0 1 x              |     |           |     |x|  2
     * |0 0 1 x x          |     |           |     |x|  3
     * |    0 1 x          |     |           |     |x|  4
     * |    0 0 1 x x x    |     |           |     |x|  5
     * |        0 1 x x    | ... |           | ... |x|  6 sc
     * |        0 0 1 x    |     |           |     |x|  7
     * |        0 0 0 1 x x|     |      x    |     |x|  8
     * |              0 1 x|     |      x    |     |x|  9
     * |              0 0 1|     |      x    |     |x| 10
     * ---------------------     -------------
     *           .                     .
     *           .                     .
     *           .                     .
     * ---------------------     -------------
     * |                   |     |x x        |     |x|  1
     * |                   |     |x x x x x  |     |x|  2
     * |                   |     |  x x x x  |     |x|  3
     * |              0 0 0| ... |  x x M x  | ... |M|  4 bb
     * |                   |     |  x x x x x|     |x|  5
     * |                   |     |        x x|     |x|  6
     * ---------------------     -------------
     *
     * @param sc_to_sc_matrix Pointer to a SC to SC glutamine sub-matrix.
     * @param sc_to_bb_matrix Pointer to a SC to BB glutamine sub-matrix.
     * @param sc_to_sep_matrix unused.
     * @param sc_g_matrix Pointer to a SC g glutamine sub-matrix.
     * @param sc_to_bb_fillins unused.
     * @param bb_to_sc_matrix Pointer to a BB to SC glutamine sub-matrix.
     * @param bb_to_bb_matrix Pointer to a BB to BB glutamine sub-matrix.
     * @param bb_to_sep_matrix unused.
     * @param bb_g_matrix Pointer to a BB g glutamine sub-matrix.
     * @param bb_to_sc_fillins unused.
     * @param bb_to_sep_fillins unused.
     */
    void solve_3_glutamine(real *const sc_to_sc_matrix,
                           real *const sc_to_bb_matrix,
                           __attribute__((unused)) real *const sc_to_sep_matrix,
                           real *const sc_g_matrix,
                           __attribute__((unused)) real *const sc_to_bb_fillins,
                           real *const bb_to_sc_matrix,
                           real *const bb_to_bb_matrix,
                           __attribute__((unused)) real *const bb_to_sep_matrix,
                           real *const bb_g_matrix,
                           __attribute__((unused)) real *const bb_to_sc_fillins,
                           __attribute__((unused)) real *const bb_to_sep_fillins);

    /**
     * Performs the Gaussian elimination of each entry of the first BB to SC
     * sub-matrix pointed by BB_TO_SC_MATRIX.
     *
     * It is assumed that SC_TO_SC_MATRIX, SC_TO_BB_MATRIX, SC_G_MATRIX,
     * BB_TO_SC_MATRIX, BB_TO_BB_MATRIX and BB_G_MATRIX point to the same
     * histidine side-chain data.
     *
     * This function has been automatically generated by
     * step_2_3_16_17_generator.c.
     *
     * Initial matrix:
     *
     *           sc                     bb            g
     *  1 2 3 4 5 6 7 8 91011       1 2 3 4 5 6
     * -----------------------     -------------
     * |1   x   x            |     |           |     |x|  1
     * |  1 x     x          |     |           |     |x|  2
     * |0 0 1   x x          |     |           |     |x|  3
     * |      1 x     x      |     |           |     |x|  4
     * |0   0 0 1 f   x      |     |           |     |x|  5
     * |  0 0   0 1 x f      | ... |           | ... |x|  6 sc
     * |          0 1 x x    |     |           |     |x|  7
     * |      0 0 0 0 1 x    |     |           |     |x|  8
     * |            0 0 1 x x|     |      x    |     |x|  9
     * |                0 1 x|     |      x    |     |x| 10
     * |                0 0 1|     |      x    |     |x| 11
     * -----------------------     -------------
     *            .                      .
     *            .                      .
     *            .                      .
     * -----------------------     -------------
     * |                     |     |x x        |     |x|  1
     * |                     |     |x x x x x  |     |x|  2
     * |                     |     |  x x x x  |     |x|  3
     * |                x x x| ... |  x x x x  | ... |x|  4 bb
     * |                     |     |  x x x x x|     |x|  5
     * |                     |     |        x x|     |x|  6
     * -----------------------     -------------
     *
     * Final matrix:
     *
     *           sc                     bb            g
     *  1 2 3 4 5 6 7 8 91011       1 2 3 4 5 6
     * -----------------------     -------------
     * |1   x   x            |     |           |     |x|  1
     * |  1 x     x          |     |           |     |x|  2
     * |0 0 1   x x          |     |           |     |x|  3
     * |      1 x     x      |     |           |     |x|  4
     * |0   0 0 1 f   x      |     |           |     |x|  5
     * |  0 0   0 1 x f      | ... |           | ... |x|  6 sc
     * |          0 1 x x    |     |           |     |x|  7
     * |      0 0 0 0 1 x    |     |           |     |x|  8
     * |            0 0 1 x x|     |      x    |     |x|  9
     * |                0 1 x|     |      x    |     |x| 10
     * |                0 0 1|     |      x    |     |x| 11
     * -----------------------     -------------
     *            .                      .
     *            .                      .
     *            .                      .
     * -----------------------     -------------
     * |                     |     |x x        |     |x|  1
     * |                     |     |x x x x x  |     |x|  2
     * |                     |     |  x x x x  |     |x|  3
     * |                0 0 0| ... |  x x M x  | ... |M|  4 bb
     * |                     |     |  x x x x x|     |x|  5
     * |                     |     |        x x|     |x|  6
     * -----------------------     -------------
     *
     * @param sc_to_sc_matrix Pointer to a SC to SC histidine sub-matrix.
     * @param sc_to_bb_matrix Pointer to a SC to BB histidine sub-matrix.
     * @param sc_to_sep_matrix unused.
     * @param sc_g_matrix Pointer to a SC g histidine sub-matrix.
     * @param sc_to_bb_fillins unused.
     * @param bb_to_sc_matrix Pointer to a BB to SC histidine sub-matrix.
     * @param bb_to_bb_matrix Pointer to a BB to BB histidine sub-matrix.
     * @param bb_to_sep_matrix unused.
     * @param bb_g_matrix Pointer to a BB g histidine sub-matrix.
     * @param bb_to_sc_fillins unused.
     * @param bb_to_sep_fillins unused.
     */
    void solve_3_histidine(real *const sc_to_sc_matrix,
                           real *const sc_to_bb_matrix,
                           __attribute__((unused)) real *const sc_to_sep_matrix,
                           real *const sc_g_matrix,
                           __attribute__((unused)) real *const sc_to_bb_fillins,
                           real *const bb_to_sc_matrix,
                           real *const bb_to_bb_matrix,
                           __attribute__((unused)) real *const bb_to_sep_matrix,
                           real *const bb_g_matrix,
                           __attribute__((unused)) real *const bb_to_sc_fillins,
                           __attribute__((unused)) real *const bb_to_sep_fillins);

    /**
     * Performs the Gaussian elimination of each entry of the first BB to SC
     * sub-matrix pointed by BB_TO_SC_MATRIX.
     *
     * It is assumed that SC_TO_SC_MATRIX, SC_TO_BB_MATRIX, SC_G_MATRIX,
     * BB_TO_SC_MATRIX, BB_TO_BB_MATRIX and BB_G_MATRIX point to the same lysine
     * side-chain data.
     *
     * This function has been automatically generated by
     * step_2_3_16_17_generator.c.
     *
     * Initial matrix:
     *
     *               sc                         bb            g
     *  1 2 3 4 5 6 7 8 9101112131415       1 2 3 4 5 6
     * -------------------------------     -------------
     * |1 x x x                      |     |           |     |x|  1
     * |0 1 x x                      |     |           |     |x|  2
     * |0 0 1 x                      |     |           |     |x|  3
     * |0 0 0 1 x x x                |     |           |     |x|  4
     * |      0 1 x x                |     |           |     |x|  5
     * |      0 0 1 x                |     |           |     |x|  6
     * |      0 0 0 1 x x x          |     |           |     |x|  7
     * |            0 1 x x          | ... |           | ... |x|  8 sc
     * |            0 0 1 x          |     |           |     |x|  9
     * |            0 0 0 1 x x x    |     |           |     |x| 10
     * |                  0 1 x x    |     |           |     |x| 11
     * |                  0 0 1 x    |     |           |     |x| 12
     * |                  0 0 0 1 x x|     |      x    |     |x| 13
     * |                        0 1 x|     |      x    |     |x| 14
     * |                        0 0 1|     |      x    |     |x| 15
     * -------------------------------     -------------
     *                .                          .
     *                .                          .
     *                .                          .
     * -------------------------------     -------------
     * |                             |     |x x        |     |x|  1
     * |                             |     |x x x x x  |     |x|  2
     * |                             |     |  x x x x  |     |x|  3
     * |                        x x x| ... |  x x x x  | ... |x|  4 bb
     * |                             |     |  x x x x x|     |x|  5
     * |                             |     |        x x|     |x|  6
     * -------------------------------     -------------
     *
     * Final matrix:
     *
     *               sc                         bb            g
     *  1 2 3 4 5 6 7 8 9101112131415       1 2 3 4 5 6
     * -------------------------------     -------------
     * |1 x x x                      |     |           |     |x|  1
     * |0 1 x x                      |     |           |     |x|  2
     * |0 0 1 x                      |     |           |     |x|  3
     * |0 0 0 1 x x x                |     |           |     |x|  4
     * |      0 1 x x                |     |           |     |x|  5
     * |      0 0 1 x                |     |           |     |x|  6
     * |      0 0 0 1 x x x          |     |           |     |x|  7
     * |            0 1 x x          | ... |           | ... |x|  8 sc
     * |            0 0 1 x          |     |           |     |x|  9
     * |            0 0 0 1 x x x    |     |           |     |x| 10
     * |                  0 1 x x    |     |           |     |x| 11
     * |                  0 0 1 x    |     |           |     |x| 12
     * |                  0 0 0 1 x x|     |      x    |     |x| 13
     * |                        0 1 x|     |      x    |     |x| 14
     * |                        0 0 1|     |      x    |     |x| 15
     * -------------------------------     -------------
     *                .                          .
     *                .                          .
     *                .                          .
     * -------------------------------     -------------
     * |                             |     |x x        |     |x|  1
     * |                             |     |x x x x x  |     |x|  2
     * |                             |     |  x x x x  |     |x|  3
     * |                        0 0 0| ... |  x x M x  | ... |M|  4 bb
     * |                             |     |  x x x x x|     |x|  5
     * |                             |     |        x x|     |x|  6
     * -------------------------------     -------------
     *
     * @param sc_to_sc_matrix Pointer to a SC to SC lysine sub-matrix.
     * @param sc_to_bb_matrix Pointer to a SC to BB lysine sub-matrix.
     * @param sc_to_sep_matrix unused.
     * @param sc_g_matrix Pointer to a SC g lysine sub-matrix.
     * @param sc_to_bb_fillins unused.
     * @param bb_to_sc_matrix Pointer to a BB to SC lysine sub-matrix.
     * @param bb_to_bb_matrix Pointer to a BB to BB lysine sub-matrix.
     * @param bb_to_sep_matrix unused.
     * @param bb_g_matrix Pointer to a BB g lysine sub-matrix.
     * @param bb_to_sc_fillins unused.
     * @param bb_to_sep_fillins unused.
     */
    void solve_3_lysine(real *const sc_to_sc_matrix,
                        real *const sc_to_bb_matrix,
                        __attribute__((unused)) real *const sc_to_sep_matrix,
                        real *const sc_g_matrix,
                        __attribute__((unused)) real *const sc_to_bb_fillins,
                        real *const bb_to_sc_matrix,
                        real *const bb_to_bb_matrix,
                        __attribute__((unused)) real *const bb_to_sep_matrix,
                        real *const bb_g_matrix,
                        __attribute__((unused)) real *const bb_to_sc_fillins,
                        __attribute__((unused)) real *const bb_to_sep_fillins);

    /**
     * Performs the Gaussian elimination of each entry of the first BB to SC
     * sub-matrix pointed by BB_TO_SC_MATRIX.
     *
     * It is assumed that SC_TO_SC_MATRIX, SC_TO_BB_MATRIX, SC_G_MATRIX,
     * BB_TO_SC_MATRIX, BB_TO_BB_MATRIX and BB_G_MATRIX point to the same
     * arginine side-chain data.
     *
     * This function has been automatically generated by
     * step_2_3_16_17_generator.c.
     *
     * Initial matrix:
     *
     *                 sc                           bb            g
     *  1 2 3 4 5 6 7 8 91011121314151617       1 2 3 4 5 6
     * -----------------------------------     -------------
     * |1 x x                            |     |           |     |x|  1
     * |0 1 x                            |     |           |     |x|  2
     * |0 0 1     x x                    |     |           |     |x|  3
     * |      1 x x                      |     |           |     |x|  4
     * |      0 1 x                      |     |           |     |x|  5
     * |    0 0 0 1 x                    |     |           |     |x|  6
     * |    0     0 1 x x                |     |           |     |x|  7
     * |            0 1 x                |     |           |     |x|  8
     * |            0 0 1 x x x          | ... |           | ... |x|  9 sc
     * |                0 1 x x          |     |           |     |x| 10
     * |                0 0 1 x          |     |           |     |x| 11
     * |                0 0 0 1 x x x    |     |           |     |x| 12
     * |                      0 1 x x    |     |           |     |x| 13
     * |                      0 0 1 x    |     |           |     |x| 14
     * |                      0 0 0 1 x x|     |      x    |     |x| 15
     * |                            0 1 x|     |      x    |     |x| 16
     * |                            0 0 1|     |      x    |     |x| 17
     * -----------------------------------     -------------
     *                  .                            .
     *                  .                            .
     *                  .                            .
     * -----------------------------------     -------------
     * |                                 |     |x x        |     |x|  1
     * |                                 |     |x x x x x  |     |x|  2
     * |                                 |     |  x x x x  |     |x|  3
     * |                            x x x| ... |  x x x x  | ... |x|  4 bb
     * |                                 |     |  x x x x x|     |x|  5
     * |                                 |     |        x x|     |x|  6
     * -----------------------------------     -------------
     *
     * Final matrix:
     *
     *                 sc                           bb            g
     *  1 2 3 4 5 6 7 8 91011121314151617       1 2 3 4 5 6
     * -----------------------------------     -------------
     * |1 x x                            |     |           |     |x|  1
     * |0 1 x                            |     |           |     |x|  2
     * |0 0 1     x x                    |     |           |     |x|  3
     * |      1 x x                      |     |           |     |x|  4
     * |      0 1 x                      |     |           |     |x|  5
     * |    0 0 0 1 x                    |     |           |     |x|  6
     * |    0     0 1 x x                |     |           |     |x|  7
     * |            0 1 x                |     |           |     |x|  8
     * |            0 0 1 x x x          | ... |           | ... |x|  9 sc
     * |                0 1 x x          |     |           |     |x| 10
     * |                0 0 1 x          |     |           |     |x| 11
     * |                0 0 0 1 x x x    |     |           |     |x| 12
     * |                      0 1 x x    |     |           |     |x| 13
     * |                      0 0 1 x    |     |           |     |x| 14
     * |                      0 0 0 1 x x|     |      x    |     |x| 15
     * |                            0 1 x|     |      x    |     |x| 16
     * |                            0 0 1|     |      x    |     |x| 17
     * -----------------------------------     -------------
     *                  .                            .
     *                  .                            .
     *                  .                            .
     * -----------------------------------     -------------
     * |                                 |     |x x        |     |x|  1
     * |                                 |     |x x x x x  |     |x|  2
     * |                                 |     |  x x x x  |     |x|  3
     * |                            0 0 0| ... |  x x M x  | ... |M|  4 bb
     * |                                 |     |  x x x x x|     |x|  5
     * |                                 |     |        x x|     |x|  6
     * -----------------------------------     -------------
     *
     * @param sc_to_sc_matrix Pointer to a SC to SC arginine sub-matrix.
     * @param sc_to_bb_matrix Pointer to a SC to BB arginine sub-matrix.
     * @param sc_to_sep_matrix unused.
     * @param sc_g_matrix Pointer to a SC g arginine sub-matrix.
     * @param sc_to_bb_fillins unused.
     * @param bb_to_sc_matrix Pointer to a BB to SC arginine sub-matrix.
     * @param bb_to_bb_matrix Pointer to a BB to BB arginine sub-matrix.
     * @param bb_to_sep_matrix unused.
     * @param bb_g_matrix Pointer to a BB g arginine sub-matrix.
     * @param bb_to_sc_fillins unused.
     * @param bb_to_sep_fillins unused.
     */
    void solve_3_arginine(real *const sc_to_sc_matrix,
                          real *const sc_to_bb_matrix,
                          __attribute__((unused)) real *const sc_to_sep_matrix,
                          real *const sc_g_matrix,
                          __attribute__((unused)) real *const sc_to_bb_fillins,
                          real *const bb_to_sc_matrix,
                          real *const bb_to_bb_matrix,
                          __attribute__((unused)) real *const bb_to_sep_matrix,
                          real *const bb_g_matrix,
                          __attribute__((unused)) real *const bb_to_sc_fillins,
                          __attribute__((unused)) real *const bb_to_sep_fillins);

    /**
     * Performs the Gaussian elimination of each entry of the first BB to SC
     * sub-matrix pointed by BB_TO_SC_MATRIX.
     *
     * It is assumed that SC_TO_SC_MATRIX, SC_TO_BB_MATRIX, SC_G_MATRIX,
     * BB_TO_SC_MATRIX, BB_TO_BB_MATRIX and BB_G_MATRIX point to the same serine
     * side-chain data.
     *
     * This function has been automatically generated by
     * step_2_3_16_17_generator.c.
     *
     * Initial matrix:
     *
     *    sc              bb            g
     *  1 2 3 4       1 2 3 4 5 6
     * ---------     -------------
     * |1 x    |     |           |     |x|  1
     * |0 1 x x|     |      x    |     |x|  2
     * |  0 1 x| ... |      x    | ... |x|  3 sc
     * |  0 0 1|     |      x    |     |x|  4
     * ---------     -------------
     *     .               .
     *     .               .
     *     .               .
     * ---------     -------------
     * |       |     |x x        |     |x|  1
     * |       |     |x x x x x  |     |x|  2
     * |       |     |  x x x x  |     |x|  3
     * |  x x x| ... |  x x x x  | ... |x|  4 bb
     * |       |     |  x x x x x|     |x|  5
     * |       |     |        x x|     |x|  6
     * ---------     -------------
     *
     * Final matrix:
     *
     *    sc              bb            g
     *  1 2 3 4       1 2 3 4 5 6
     * ---------     -------------
     * |1 x    |     |           |     |x|  1
     * |0 1 x x|     |      x    |     |x|  2
     * |  0 1 x| ... |      x    | ... |x|  3 sc
     * |  0 0 1|     |      x    |     |x|  4
     * ---------     -------------
     *     .               .
     *     .               .
     *     .               .
     * ---------     -------------
     * |       |     |x x        |     |x|  1
     * |       |     |x x x x x  |     |x|  2
     * |       |     |  x x x x  |     |x|  3
     * |  0 0 0| ... |  x x M x  | ... |M|  4 bb
     * |       |     |  x x x x x|     |x|  5
     * |       |     |        x x|     |x|  6
     * ---------     -------------
     *
     * @param sc_to_sc_matrix Pointer to a SC to SC serine sub-matrix.
     * @param sc_to_bb_matrix Pointer to a SC to BB serine sub-matrix.
     * @param sc_to_sep_matrix unused.
     * @param sc_g_matrix Pointer to a SC g serine sub-matrix.
     * @param sc_to_bb_fillins unused.
     * @param bb_to_sc_matrix Pointer to a BB to SC serine sub-matrix.
     * @param bb_to_bb_matrix Pointer to a BB to BB serine sub-matrix.
     * @param bb_to_sep_matrix unused.
     * @param bb_g_matrix Pointer to a BB g serine sub-matrix.
     * @param bb_to_sc_fillins unused.
     * @param bb_to_sep_fillins unused.
     */
    void solve_3_serine(real *const sc_to_sc_matrix,
                        real *const sc_to_bb_matrix,
                        __attribute__((unused)) real *const sc_to_sep_matrix,
                        real *const sc_g_matrix,
                        __attribute__((unused)) real *const sc_to_bb_fillins,
                        real *const bb_to_sc_matrix,
                        real *const bb_to_bb_matrix,
                        __attribute__((unused)) real *const bb_to_sep_matrix,
                        real *const bb_g_matrix,
                        __attribute__((unused)) real *const bb_to_sc_fillins,
                        __attribute__((unused)) real *const bb_to_sep_fillins);

    /**
     * Performs the Gaussian elimination of each entry of the first BB to SC
     * sub-matrix pointed by BB_TO_SC_MATRIX.
     *
     * It is assumed that SC_TO_SC_MATRIX, SC_TO_BB_MATRIX, SC_G_MATRIX,
     * BB_TO_SC_MATRIX, BB_TO_BB_MATRIX and BB_G_MATRIX point to the same
     * phenylaline side-chain data.
     *
     * This function has been automatically generated by
     * step_2_3_16_17_generator.c.
     *
     * Initial matrix:
     *
     *              sc                        bb            g
     *  1 2 3 4 5 6 7 8 91011121314       1 2 3 4 5 6
     * -----------------------------     -------------
     * |1   x             x        |     |           |     |x|  1
     * |  1 x   x                  |     |           |     |x|  2
     * |0 0 1   x         x        |     |           |     |x|  3
     * |      1 x   x              |     |           |     |x|  4
     * |  0 0 0 1   x     f        |     |           |     |x|  5
     * |          1 x   x          |     |           |     |x|  6
     * |      0 0 0 1   x f        |     |           |     |x|  7
     * |              1 x   x      | ... |           | ... |x|  8 sc
     * |          0 0 0 1 f x      |     |           |     |x|  9
     * |0   0   0   0   0 1 x x    |     |           |     |x| 10
     * |              0 0 0 1 x    |     |           |     |x| 11
     * |                  0 0 1 x x|     |      x    |     |x| 12
     * |                      0 1 x|     |      x    |     |x| 13
     * |                      0 0 1|     |      x    |     |x| 14
     * -----------------------------     -------------
     *               .                         .
     *               .                         .
     *               .                         .
     * -----------------------------     -------------
     * |                           |     |x x        |     |x|  1
     * |                           |     |x x x x x  |     |x|  2
     * |                           |     |  x x x x  |     |x|  3
     * |                      x x x| ... |  x x x x  | ... |x|  4 bb
     * |                           |     |  x x x x x|     |x|  5
     * |                           |     |        x x|     |x|  6
     * -----------------------------     -------------
     *
     * Final matrix:
     *
     *              sc                        bb            g
     *  1 2 3 4 5 6 7 8 91011121314       1 2 3 4 5 6
     * -----------------------------     -------------
     * |1   x             x        |     |           |     |x|  1
     * |  1 x   x                  |     |           |     |x|  2
     * |0 0 1   x         x        |     |           |     |x|  3
     * |      1 x   x              |     |           |     |x|  4
     * |  0 0 0 1   x     f        |     |           |     |x|  5
     * |          1 x   x          |     |           |     |x|  6
     * |      0 0 0 1   x f        |     |           |     |x|  7
     * |              1 x   x      | ... |           | ... |x|  8 sc
     * |          0 0 0 1 f x      |     |           |     |x|  9
     * |0   0   0   0   0 1 x x    |     |           |     |x| 10
     * |              0 0 0 1 x    |     |           |     |x| 11
     * |                  0 0 1 x x|     |      x    |     |x| 12
     * |                      0 1 x|     |      x    |     |x| 13
     * |                      0 0 1|     |      x    |     |x| 14
     * -----------------------------     -------------
     *               .                         .
     *               .                         .
     *               .                         .
     * -----------------------------     -------------
     * |                           |     |x x        |     |x|  1
     * |                           |     |x x x x x  |     |x|  2
     * |                           |     |  x x x x  |     |x|  3
     * |                      0 0 0| ... |  x x M x  | ... |M|  4 bb
     * |                           |     |  x x x x x|     |x|  5
     * |                           |     |        x x|     |x|  6
     * -----------------------------     -------------
     *
     * @param sc_to_sc_matrix Pointer to a SC to SC phenylaline sub-matrix.
     * @param sc_to_bb_matrix Pointer to a SC to BB phenylaline sub-matrix.
     * @param sc_to_sep_matrix unused.
     * @param sc_g_matrix Pointer to a SC g phenylaline sub-matrix.
     * @param sc_to_bb_fillins unused.
     * @param bb_to_sc_matrix Pointer to a BB to SC phenylaline sub-matrix.
     * @param bb_to_bb_matrix Pointer to a BB to BB phenylaline sub-matrix.
     * @param bb_to_sep_matrix unused.
     * @param bb_g_matrix Pointer to a BB g phenylaline sub-matrix.
     * @param bb_to_sc_fillins unused.
     * @param bb_to_sep_fillins unused.
     */
    void solve_3_phenylaline(real *const sc_to_sc_matrix,
                             real *const sc_to_bb_matrix,
                             __attribute__((unused)) real *const sc_to_sep_matrix,
                             real *const sc_g_matrix,
                             __attribute__((unused)) real *const sc_to_bb_fillins,
                             real *const bb_to_sc_matrix,
                             real *const bb_to_bb_matrix,
                             __attribute__((unused)) real *const bb_to_sep_matrix,
                             real *const bb_g_matrix,
                             __attribute__((unused)) real *const bb_to_sc_fillins,
                             __attribute__((unused))
                             real *const bb_to_sep_fillins);

    /**
     * Performs the Gaussian elimination of each entry of the first BB to SC
     * sub-matrix pointed by BB_TO_SC_MATRIX.
     *
     * It is assumed that SC_TO_SC_MATRIX, SC_TO_BB_MATRIX, SC_G_MATRIX,
     * BB_TO_SC_MATRIX, BB_TO_BB_MATRIX and BB_G_MATRIX point to the same
     * tyrosine side-chain data.
     *
     * This function has been automatically generated by
     * step_2_3_16_17_generator.c.
     *
     * Initial matrix:
     *
     *               sc                         bb            g
     *  1 2 3 4 5 6 7 8 9101112131415       1 2 3 4 5 6
     * -------------------------------     -------------
     * |1       x                    |     |           |     |x|  1
     * |  1   x             x        |     |           |     |x|  2
     * |    1 x   x                  |     |           |     |x|  3
     * |  0 0 1   x         x        |     |           |     |x|  4
     * |0       1 x   x              |     |           |     |x|  5
     * |    0 0 0 1   x     f        |     |           |     |x|  6
     * |            1 x   x          |     |           |     |x|  7
     * |        0 0 0 1   x f        | ... |           | ... |x|  8 sc
     * |                1 x   x      |     |           |     |x|  9
     * |            0 0 0 1 f x      |     |           |     |x| 10
     * |  0   0   0   0   0 1 x x    |     |           |     |x| 11
     * |                0 0 0 1 x    |     |           |     |x| 12
     * |                    0 0 1 x x|     |      x    |     |x| 13
     * |                        0 1 x|     |      x    |     |x| 14
     * |                        0 0 1|     |      x    |     |x| 15
     * -------------------------------     -------------
     *                .                          .
     *                .                          .
     *                .                          .
     * -------------------------------     -------------
     * |                             |     |x x        |     |x|  1
     * |                             |     |x x x x x  |     |x|  2
     * |                             |     |  x x x x  |     |x|  3
     * |                        x x x| ... |  x x x x  | ... |x|  4 bb
     * |                             |     |  x x x x x|     |x|  5
     * |                             |     |        x x|     |x|  6
     * -------------------------------     -------------
     *
     * Final matrix:
     *
     *               sc                         bb            g
     *  1 2 3 4 5 6 7 8 9101112131415       1 2 3 4 5 6
     * -------------------------------     -------------
     * |1       x                    |     |           |     |x|  1
     * |  1   x             x        |     |           |     |x|  2
     * |    1 x   x                  |     |           |     |x|  3
     * |  0 0 1   x         x        |     |           |     |x|  4
     * |0       1 x   x              |     |           |     |x|  5
     * |    0 0 0 1   x     f        |     |           |     |x|  6
     * |            1 x   x          |     |           |     |x|  7
     * |        0 0 0 1   x f        | ... |           | ... |x|  8 sc
     * |                1 x   x      |     |           |     |x|  9
     * |            0 0 0 1 f x      |     |           |     |x| 10
     * |  0   0   0   0   0 1 x x    |     |           |     |x| 11
     * |                0 0 0 1 x    |     |           |     |x| 12
     * |                    0 0 1 x x|     |      x    |     |x| 13
     * |                        0 1 x|     |      x    |     |x| 14
     * |                        0 0 1|     |      x    |     |x| 15
     * -------------------------------     -------------
     *                .                          .
     *                .                          .
     *                .                          .
     * -------------------------------     -------------
     * |                             |     |x x        |     |x|  1
     * |                             |     |x x x x x  |     |x|  2
     * |                             |     |  x x x x  |     |x|  3
     * |                        0 0 0| ... |  x x M x  | ... |M|  4 bb
     * |                             |     |  x x x x x|     |x|  5
     * |                             |     |        x x|     |x|  6
     * -------------------------------     -------------
     *
     * @param sc_to_sc_matrix Pointer to a SC to SC tyrosine sub-matrix.
     * @param sc_to_bb_matrix Pointer to a SC to BB tyrosine sub-matrix.
     * @param sc_to_sep_matrix unused.
     * @param sc_g_matrix Pointer to a SC g tyrosine sub-matrix.
     * @param sc_to_bb_fillins unused.
     * @param bb_to_sc_matrix Pointer to a BB to SC tyrosine sub-matrix.
     * @param bb_to_bb_matrix Pointer to a BB to BB tyrosine sub-matrix.
     * @param bb_to_sep_matrix unused.
     * @param bb_g_matrix Pointer to a BB g tyrosine sub-matrix.
     * @param bb_to_sc_fillins unused.
     * @param bb_to_sep_fillins unused.
     */
    void solve_3_tyrosine(real *const sc_to_sc_matrix,
                          real *const sc_to_bb_matrix,
                          __attribute__((unused)) real *const sc_to_sep_matrix,
                          real *const sc_g_matrix,
                          __attribute__((unused)) real *const sc_to_bb_fillins,
                          real *const bb_to_sc_matrix,
                          real *const bb_to_bb_matrix,
                          __attribute__((unused)) real *const bb_to_sep_matrix,
                          real *const bb_g_matrix,
                          __attribute__((unused)) real *const bb_to_sc_fillins,
                          __attribute__((unused)) real *const bb_to_sep_fillins);

    /**
     * Performs the Gaussian elimination of each entry of the first BB to SC
     * sub-matrix pointed by BB_TO_SC_MATRIX.
     *
     * It is assumed that SC_TO_SC_MATRIX, SC_TO_BB_MATRIX, SC_G_MATRIX,
     * BB_TO_SC_MATRIX, BB_TO_BB_MATRIX and BB_G_MATRIX point to the same
     * tryptophan side-chain data.
     *
     * This function has been automatically generated by
     * ilves_step_2_3_16_17_generator.c.
     *
     * Initial matrix:
     *
     *                   sc                             bb            g
     *  1 2 3 4 5 6 7 8 910111213141516171819       1 2 3 4 5 6
     * ---------------------------------------     -------------
     * |1   x           x                    |     |           |     |x|  1
     * |  1 x   x                            |     |           |     |x|  2
     * |0 0 1   x       x                    |     |           |     |x|  3
     * |      1 x   x                        |     |           |     |x|  4
     * |  0 0 0 1   x   f                    |     |           |     |x|  5
     * |          1 x x                      |     |           |     |x|  6
     * |      0 0 0 1 x f                    |     |           |     |x|  7
     * |          0 0 1 f         x   x      |     |           |     |x|  8
     * |0   0   0   0 0 1       x x   f      |     |           |     |x|  9
     * |                  1   x x            | ... |           | ... |x| 10 sc
     * |                    1 x     x        |     |           |     |x| 11
     * |                  0 0 1 x   x        |     |           |     |x| 12
     * |                0 0   0 1 x f f      |     |           |     |x| 13
     * |              0 0       0 1 f x      |     |           |     |x| 14
     * |                    0 0 0 0 1 x x    |     |           |     |x| 15
     * |              0 0       0 0 0 1 x    |     |           |     |x| 16
     * |                            0 0 1 x x|     |      x    |     |x| 17
     * |                                0 1 x|     |      x    |     |x| 18
     * |                                0 0 1|     |      x    |     |x| 19
     * ---------------------------------------     -------------
     *                    .                              .
     *                    .                              .
     *                    .                              .
     * ---------------------------------------     -------------
     * |                                     |     |x x        |     |x|  1
     * |                                     |     |x x x x x  |     |x|  2
     * |                                     |     |  x x x x  |     |x|  3
     * |                                x x x| ... |  x x x x  | ... |x|  4 bb
     * |                                     |     |  x x x x x|     |x|  5
     * |                                     |     |        x x|     |x|  6
     * ---------------------------------------     -------------
     *
     * Final matrix:
     *
     *                   sc                             bb            g
     *  1 2 3 4 5 6 7 8 910111213141516171819       1 2 3 4 5 6
     * ---------------------------------------     -------------
     * |1   x           x                    |     |           |     |x|  1
     * |  1 x   x                            |     |           |     |x|  2
     * |0 0 1   x       x                    |     |           |     |x|  3
     * |      1 x   x                        |     |           |     |x|  4
     * |  0 0 0 1   x   f                    |     |           |     |x|  5
     * |          1 x x                      |     |           |     |x|  6
     * |      0 0 0 1 x f                    |     |           |     |x|  7
     * |          0 0 1 f         x   x      |     |           |     |x|  8
     * |0   0   0   0 0 1       x x   f      |     |           |     |x|  9
     * |                  1   x x            | ... |           | ... |x| 10 sc
     * |                    1 x     x        |     |           |     |x| 11
     * |                  0 0 1 x   x        |     |           |     |x| 12
     * |                0 0   0 1 x f f      |     |           |     |x| 13
     * |              0 0       0 1 f x      |     |           |     |x| 14
     * |                    0 0 0 0 1 x x    |     |           |     |x| 15
     * |              0 0       0 0 0 1 x    |     |           |     |x| 16
     * |                            0 0 1 x x|     |      x    |     |x| 17
     * |                                0 1 x|     |      x    |     |x| 18
     * |                                0 0 1|     |      x    |     |x| 19
     * ---------------------------------------     -------------
     *                    .                              .
     *                    .                              .
     *                    .                              .
     * ---------------------------------------     -------------
     * |                                     |     |x x        |     |x|  1
     * |                                     |     |x x x x x  |     |x|  2
     * |                                     |     |  x x x x  |     |x|  3
     * |                                0 0 0| ... |  x x M x  | ... |M|  4 bb
     * |                                     |     |  x x x x x|     |x|  5
     * |                                     |     |        x x|     |x|  6
     * ---------------------------------------     -------------
     *
     * @param sc_to_sc_matrix Pointer to a SC to SC tryptophan sub-matrix.
     * @param sc_to_bb_matrix Pointer to a SC to BB tryptophan sub-matrix.
     * @param sc_to_sep_matrix unused.
     * @param sc_g_matrix Pointer to a SC g tryptophan sub-matrix.
     * @param sc_to_bb_fillins unused.
     * @param bb_to_sc_matrix Pointer to a BB to SC tryptophan sub-matrix.
     * @param bb_to_bb_matrix Pointer to a BB to BB tryptophan sub-matrix.
     * @param bb_to_sep_matrix unused.
     * @param bb_g_matrix Pointer to a BB g tryptophan sub-matrix.
     * @param bb_to_sc_fillins unused.
     * @param bb_to_sep_fillins unused.
     */
    void solve_3_tryptophan(real *const sc_to_sc_matrix,
                            real *const sc_to_bb_matrix,
                            __attribute__((unused)) real *const sc_to_sep_matrix,
                            real *const sc_g_matrix,
                            __attribute__((unused)) real *const sc_to_bb_fillins,
                            real *const bb_to_sc_matrix,
                            real *const bb_to_bb_matrix,
                            __attribute__((unused)) real *const bb_to_sep_matrix,
                            real *const bb_g_matrix,
                            __attribute__((unused)) real *const bb_to_sc_fillins,
                            __attribute__((unused)) real *const bb_to_sep_fillins);

    /**
     * Performs the Gaussian elimination of each entry of the first BB to SC
     * sub-matrix pointed by BB_TO_SC_MATRIX.
     *
     * It is assumed that SC_TO_SC_MATRIX, SC_TO_BB_MATRIX, SC_G_MATRIX,
     * BB_TO_SC_MATRIX, BB_TO_BB_MATRIX and BB_G_MATRIX point to the same
     * threonine side-chain data.
     *
     * This function has been automatically generated by
     * step_2_3_16_17_generator.c.
     *
     * Initial matrix:
     *
     *       sc                 bb            g
     *  1 2 3 4 5 6 7       1 2 3 4 5 6
     * ---------------     -------------
     * |1 x x x      |     |           |     |x|  1
     * |0 1 x x      |     |           |     |x|  2
     * |0 0 1 x      |     |           |     |x|  3
     * |0 0 0 1   x x| ... |      x    | ... |x|  4 sc
     * |        1 x  |     |           |     |x|  5
     * |      0 0 1 x|     |      x    |     |x|  6
     * |      0   0 1|     |      x    |     |x|  7
     * ---------------     -------------
     *        .                  .
     *        .                  .
     *        .                  .
     * ---------------     -------------
     * |             |     |x x        |     |x|  1
     * |             |     |x x x x x  |     |x|  2
     * |             |     |  x x x x  |     |x|  3
     * |      x   x x| ... |  x x x x  | ... |x|  4 bb
     * |             |     |  x x x x x|     |x|  5
     * |             |     |        x x|     |x|  6
     * ---------------     -------------
     *
     * Final matrix:
     *
     *       sc                 bb            g
     *  1 2 3 4 5 6 7       1 2 3 4 5 6
     * ---------------     -------------
     * |1 x x x      |     |           |     |x|  1
     * |0 1 x x      |     |           |     |x|  2
     * |0 0 1 x      |     |           |     |x|  3
     * |0 0 0 1   x x| ... |      x    | ... |x|  4 sc
     * |        1 x  |     |           |     |x|  5
     * |      0 0 1 x|     |      x    |     |x|  6
     * |      0   0 1|     |      x    |     |x|  7
     * ---------------     -------------
     *        .                  .
     *        .                  .
     *        .                  .
     * ---------------     -------------
     * |             |     |x x        |     |x|  1
     * |             |     |x x x x x  |     |x|  2
     * |             |     |  x x x x  |     |x|  3
     * |      0   0 0| ... |  x x M x  | ... |M|  4 bb
     * |             |     |  x x x x x|     |x|  5
     * |             |     |        x x|     |x|  6
     * ---------------     -------------
     *
     * @param sc_to_sc_matrix Pointer to a SC to SC threonine sub-matrix.
     * @param sc_to_bb_matrix Pointer to a SC to BB threonine sub-matrix.
     * @param sc_to_sep_matrix unused.
     * @param sc_g_matrix Pointer to a SC g threonine sub-matrix.
     * @param sc_to_bb_fillins unused.
     * @param bb_to_sc_matrix Pointer to a BB to SC threonine sub-matrix.
     * @param bb_to_bb_matrix Pointer to a BB to BB threonine sub-matrix.
     * @param bb_to_sep_matrix unused.
     * @param bb_g_matrix Pointer to a BB g threonine sub-matrix.
     * @param bb_to_sc_fillins unused.
     * @param bb_to_sep_fillins unused.
     */
    void solve_3_threonine(real *const sc_to_sc_matrix,
                           real *const sc_to_bb_matrix,
                           __attribute__((unused)) real *const sc_to_sep_matrix,
                           real *const sc_g_matrix,
                           __attribute__((unused)) real *const sc_to_bb_fillins,
                           real *const bb_to_sc_matrix,
                           real *const bb_to_bb_matrix,
                           __attribute__((unused)) real *const bb_to_sep_matrix,
                           real *const bb_g_matrix,
                           __attribute__((unused)) real *const bb_to_sc_fillins,
                           __attribute__((unused)) real *const bb_to_sep_fillins);

    /**
     * Performs the Gaussian elimination of each BB to SC sub-matrix entry
     * assigned to THREAD.
     *
     * Initial matrix:
     *
     *   sc          bb               g
     *  1 2 3       1 2 3 4 5 6
     * -------     -------------
     * |1 x  |     |           |     |x|      1 Note: **
     * |0 1 x| ... |           | ... |x| sc   2
     * |  0 1|     |      x    |     |x|      3
     * -------     -------------
     *    .              .
     *    .              .
     *    .              .
     * -------     -------------
     * |     |     |x x        |     |x|      1 Note: **
     * |     |     |x x x x x  |     |x|      2
     * |     |     |  x x x x  |     |x| bb   3
     * |    x| ... |  x x x x  | ... |x|      4
     * |     |     |  x x x x x|     |x|      5
     * |     |     |        x x|     |x|      6
     * -------     -------------
     *
     * ** This SC/BB to SC sub-matrix belongs to a linear side chain
     * (unrealistic). The side chain types can be any of the defined in
     * molecule.h.
     *
     * Final matrix:
     *
     *   sc          bb               g
     *  1 2 3       1 2 3 4 5 6
     * -------     -------------
     * |1 x  |     |           |     |x|      1 Note: **
     * |0 1 x| ... |           | ... |x| sc   2
     * |  0 1|     |      x    |     |x|      3
     * -------     -------------
     *    .              .
     *    .              .
     *    .              .
     * -------     -------------
     * |     |     |x x        |     |x|      1 Note: **
     * |     |     |x x x x x  |     |x|      2
     * |     |     |  x x x x  |     |x| bb   3
     * |    0| ... |  x x M x  | ... |M|      4
     * |     |     |  x x x x x|     |x|      5
     * |     |     |        x x|     |x|      6
     * -------     -------------
     *
     * @param thread Thread descriptor [0 - NTHREADS - 1].
     */
    void solve_3(const int thread);

    /**
     * Performs the Gaussian elimination of the subdiagonal entries of the first
     * BB to BB sub-matrix pointed by BB_TO_BB_MATRIX. Also make 1s in the
     * diagonal entries.
     *
     * It is assumed that BB_TO_BB_MATRIX, BB_TO_SEP_MATRIX, BB_G_MATRIX and
     * BB_TO_SEP_FILLINS point to the same proline backbone data.
     *
     * Initial matrix:
     *
     *      bb          sep       g
     *  1 2 3 4 5       1 2
     * -----------     -----
     * |x x      |     |x  |     |x|       1
     * |x x x x x|     |x  |     |x|       2
     * |  x x x x|     |   |     |x| bb    3
     * |  x x x x| ... |  f| ... |x|       4
     * |  x x x x|     |  x|     |x|       5
     * -----------     -----
     *
     * Final matrix:
     *
     *      bb          sep       g
     *  1 2 3 4 5       1 2
     * -----------     -----
     * |1 M      |     |M  |     |M|       1
     * |0 1 M M M|     |M  |     |M|       2
     * |  0 1 M M|     |F  |     |M| bb    3
     * |  0 0 1 M| ... |F M| ... |M|       4
     * |  0 0 0 1|     |F M|     |M|       5
     * -----------     -----
     *
     * @param bb_to_bb_matrix Pointer to a BB to BB proline sub-matrix.
     * @param bb_to_sep_matrix Pointer to a SEP to SEP proline sub-matrix.
     * @param bb_g_matrix Pointer to a BB g proline sub-matrix.
     * @param bb_to_sep_fillins Pointer to a BB to SEP proline sub-matrix.
     */
    void solve_4_proline(real *const bb_to_bb_matrix,
                         real *const bb_to_sep_matrix,
                         real *const bb_g_matrix,
                         real *const bb_to_sep_fillins);

    /**
     * Performs the Gaussian elimination of the subdiagonal entries of the first
     * BB to BB sub-matrix pointed by BB_TO_BB_MATRIX. Also make 1s in the
     * diagonal entries.
     *
     * It is assumed that BB_TO_BB_MATRIX, BB_TO_SEP_MATRIX, BB_G_MATRIX and
     * BB_TO_SEP_FILLINS point to the same non-proline backbone data.
     *
     * Initial matrix:
     *
     *      bb            sep       g
     *  1 2 3 4 5 6       1 2
     * -------------     -----
     * |x x        |     |x  |     |x|       1
     * |x x x x x  |     |x  |     |x|       2
     * |  x x x x  |     |   |     |x| bb    3
     * |  x x x x  | ... |   | ... |x|       4
     * |  x x x x x|     |  x|     |x|       5
     * |        x x|     |  x|     |x|       6
     * -------------     -----
     *
     * Final matrix:
     *
     *      bb            sep       g
     *  1 2 3 4 5 6       1 2
     * -------------     -----
     * |1 M        |     |M  |     |M|       1
     * |0 1 M M M  |     |M  |     |M|       2
     * |  0 1 M M  |     |F  |     |M| bb    3
     * |  0 0 1 M  | ... |F  | ... |M|       4
     * |  0 0 0 1 M|     |F M|     |M|       5
     * |        0 1|     |F M|     |M|       6
     * -------------     -----
     *
     * @param bb_to_bb_matrix Pointer to a BB to BB non-proline sub-matrix.
     * @param bb_to_sep_matrix Pointer to a SEP to SEP non-proline sub-matrix.
     * @param bb_g_matrix Pointer to a BB g non-proline sub-matrix.
     * @param bb_to_sep_fillins Pointer to a BB to SEP non-proline sub-matrix.
     */
    void solve_4_general(real *const bb_to_bb_matrix,
                         real *const bb_to_sep_matrix,
                         real *const bb_g_matrix,
                         real *const bb_to_sep_fillins);

    /**
     * Performs the Gaussian elimination of the subdiagonal entries of each BB
     * to BB sub-matrix assigned to THREAD. Also make 1s in the diagonal
     * entries.
     *
     * Initial matrix (general case):
     *
     *      bb            sep       g
     *  1 2 3 4 5 6       1 2
     * -------------     -----
     * |x x        |     |x  |     |x|       1
     * |x x x x x  |     |x  |     |x|       2
     * |  x x x x  |     |   |     |x| bb    3
     * |  x x x x  | ... |   | ... |x|       4
     * |  x x x x x|     |  x|     |x|       5
     * |        x x|     |  x|     |x|       6
     * -------------     -----
     *
     * Initial matrix (proline):
     *
     *      bb          sep       g
     *  1 2 3 4 5       1 2
     * -----------     -----
     * |x x      |     |x  |     |x|       1
     * |x x x x x|     |x  |     |x|       2
     * |  x x x x|     |   |     |x| bb    3
     * |  x x x x| ... |  f| ... |x|       4
     * |  x x x x|     |  x|     |x|       5
     * -----------     -----
     *
     * Final matrix (general case):
     *
     *      bb            sep       g
     *  1 2 3 4 5 6       1 2
     * -------------     -----
     * |1 M        |     |M  |     |M|       1
     * |0 1 M M M  |     |M  |     |M|       2
     * |  0 1 M M  |     |F  |     |M| bb    3
     * |  0 0 1 M  | ... |F  | ... |M|       4
     * |  0 0 0 1 M|     |F M|     |M|       5
     * |        0 1|     |F M|     |M|       6
     * -------------     -----
     *
     * Final matrix (proline):
     *
     *      bb          sep       g
     *  1 2 3 4 5       1 2
     * -----------     -----
     * |1 M      |     |M  |     |M|       1
     * |0 1 M M M|     |M  |     |M|       2
     * |  0 1 M M|     |F  |     |M| bb    3
     * |  0 0 1 M| ... |F M| ... |M|       4
     * |  0 0 0 1|     |F M|     |M|       5
     * -----------     -----
     *
     * @param thread Thread descriptor [0 - NTHREADS - 1].
     */
    void solve_4(const int thread);

    /**
     * Performs the Gaussian elimination of each SEP to SC sub-matrix entry
     * assigned to THREAD. A SEP to BB sub-matrix is populated only on proline
     * side-chains.
     *
     * Initial matrix:
     *
     *         sc                  bb            sep        g
     *  1 2 3 4 5 6 7 8 9       1 2 3 4 5
     * -------------------     -----------     -------
     * |1 x         x    |     |      x  |     |     |     |x|  1
     * |0 1         x    |     |      x  |     |     |     |x|  2
     * |    1 x     x x  |     |         |     |     |     |x|  3
     * |    0 1     x x  |     |         |     |     |     |x|  4
     * |        1 x   x x| ... |         | ... |     | ... |x|  5 sc
     * |        0 1   x x|     |         |     |     |     |x|  6
     * |0 0 0 0     1 x  |     |      x  |     |     |     |x|  7
     * |    0 0 0 0 0 1 x|     |      f  |     |     |     |x|  8
     * |        0 0   0 1|     |      f x|     |  x  |     |x|  9
     * -------------------     -----------     -------
     *         .                    .             .
     *         .                    .             .
     *         .                    .             .
     * -------------------     -----------     -------
     * |                x| ... |        x| ... |? x ?| ... |x|  1  sep **
     * -------------------     -----------     -------
     *
     * ** ? can be 0 or a fillin.
     *
     * Final matrix:
     *
     *         sc                  bb            sep        g
     *  1 2 3 4 5 6 7 8 9       1 2 3 4 5
     * -------------------     -----------     -------
     * |1 x         x    |     |      x  |     |     |     |x|  1
     * |0 1         x    |     |      x  |     |     |     |x|  2
     * |    1 x     x x  |     |         |     |     |     |x|  3
     * |    0 1     x x  |     |         |     |     |     |x|  4
     * |        1 x   x x| ... |         | ... |     | ... |x|  5 sc
     * |        0 1   x x|     |         |     |     |     |x|  6
     * |0 0 0 0     1 x  |     |      x  |     |     |     |x|  7
     * |    0 0 0 0 0 1 x|     |      f  |     |     |     |x|  8
     * |        0 0   0 1|     |      f x|     |  x  |     |x|  9
     * -------------------     -----------     -------
     *         .                    .             .
     *         .                    .             .
     *         .                    .             .
     * -------------------     -----------     -------
     * |                0| ... |      F M| ... |? M ?| ... |M|  1  sep
     * -------------------     -----------     -------
     *
     * @param thread Thread descriptor [0 - NTHREADS - 1].
     */
    void solve_5(const int thread);

    /**
     * Performs the Gaussian elimination of the first two SEP to BB sub-matrix
     * entries pointed by SEP_TO_BB. It is assumed that SEP_TO_BB points to the
     * separation row corresponding to the backbone block BB_TO_BB_MATRIX.
     *
     * This function works only for proline side-chains.
     *
     * Initial matrix:
     *
     *      bb           sep        g
     *  1 2 3 4 5       1 2 3
     * -----------     -------
     * |1 x      |     |x    |     |x|       1
     * |0 1 x x x|     |x    |     |x|       2
     * |  0 1 x x|     |f    |     |x| bb    3
     * |  0 0 1 x| ... |f f  | ... |x|       4
     * |  0 0 0 1|     |f x  |     |x|       5
     * -----------     -------
     *       .            .
     *       .            .
     *       .            .
     * -----------     -------
     * |      f x|     |  x ?|     |x| sep   1 ? Can be 0 or fillin.
     * -----------     -------
     *
     * Final matrix:
     *
     *      bb           sep        g
     *  1 2 3 4 5       1 2 3
     * -----------     -------
     * |1 x      |     |x    |     |x|       1
     * |0 1 x x x|     |x    |     |x|       2
     * |  0 1 x x|     |f    |     |x| bb    3
     * |  0 0 1 x| ... |f f  | ... |x|       4
     * |  0 0 0 1|     |f x  |     |x|       5
     * -----------     -------
     *      .             .
     *      .             .
     *      .             .
     * -----------     -------
     * |      0 0|     |F x ?|     |x| sep   1 ? Can be 0 or fillin.
     * -----------     -------
     *
     * @param bb_to_bb_matrix Pointer to a BB to BB proline sub-matrix.
     * @param bb_to_sep_matrix Pointer to a BB to SEP proline sub-matrix.
     * @param bb_g_matrix Pointer to a BB g proline sub-matrix.
     * @param bb_to_sep_fillins Pointer to a BB to SEP proline fillins
     * sub-matrix.
     * @param sep_to_bb_matrix Pointer to a SEP to BB proline sub-matrix.
     * @param sep_to_sep_matrix Pointer to a SEP to SEP proline sub-matrix.
     * @param sep_g_matrix Pointer to a SEP g proline sub-matrix.
     * @param sep_to_bb_fillins Pointer to a SEP to BB proline fillins
     * sub-matrix.
     * @param sep_to_sep_fillins Pointer to a SEP to SEP proline fillins
     * sub-matrix.
     */
    void solve_6_proline_current_row(real *const bb_to_bb_matrix,
                                     real *const bb_to_sep_matrix,
                                     real *const bb_g_matrix,
                                     real *const bb_to_sep_fillins,
                                     real *const sep_to_bb_matrix,
                                     real *const sep_to_sep_matrix,
                                     real *const sep_g_matrix,
                                     real *const sep_to_bb_fillins,
                                     real *const sep_to_sep_fillins);

    /**
     * Performs the Gaussian elimination of the first two SEP to BB sub-matrix
     * entries pointed by SEP_TO_BB. It is assumed that SEP_TO_BB points to the
     * separation row corresponding to the backbone block BB_TO_BB_MATRIX.
     *
     * This function works only for non-proline side-chains.
     *
     * Initial matrix (general case):
     *
     *      bb             sep        g
     *  1 2 3 4 5 6       1 2 3
     * -------------     -------
     * |1 x        |     |x    |     |x|       1
     * |0 1 x x x  |     |x    |     |x|       2
     * |  0 1 x x  |     |f    |     |x| bb    3
     * |  0 0 1 x  | ... |f    | ... |x|       4
     * |  0 0 0 1 x|     |f x  |     |x|       5
     * |        0 1|     |f x  |     |x|       6
     * -------------     -------
     *       .              .
     *       .              .
     *       .              .
     * -------------     -------
     * |        x x|     |  x ?|     |x| sep   1 ? Can be 0 or fillin.
     * -------------     -------
     *
     * Final matrix (general case):
     *
     *      bb             sep        g
     *  1 2 3 4 5 6       1 2 3
     * -------------     -------
     * |1 x        |     |x    |     |x|       1
     * |0 1 x x x  |     |x    |     |x|       2
     * |  0 1 x x  |     |f    |     |x| bb    3
     * |  0 0 1 x  | ... |f    | ... |x|       4
     * |  0 0 0 1 x|     |f x  |     |x|       5
     * |        0 1|     |f x  |     |x|       6
     * -------------     -------
     *       .              .
     *       .              .
     *       .              .
     * -------------     -------
     * |        0 0|     |F x ?|     |x| sep   1 ? Can be 0 or fillin.
     * -------------     -------
     *
     * @param bb_to_bb_matrix Pointer to a BB to BB non-proline sub-matrix.
     * @param bb_to_sep_matrix Pointer to a BB to SEP non-proline sub-matrix.
     * @param bb_g_matrix Pointer to a BB g non-proline sub-matrix.
     * @param bb_to_sep_fillins Pointer to a BB to SEP non-proline fillins
     * sub-matrix.
     * @param sep_to_bb_matrix Pointer to a SEP to BB non-proline sub-matrix.
     * @param sep_to_sep_matrix Pointer to a SEP to SEP non-proline sub-matrix.
     * @param sep_g_matrix Pointer to a SEP g non-proline sub-matrix.
     * @param sep_to_sep_fillins Pointer to a SEP to SEP non-proline fillins
     * sub-matrix.
     */
    void solve_6_general_current_row(real *const bb_to_bb_matrix,
                                     real *const bb_to_sep_matrix,
                                     real *const bb_g_matrix,
                                     real *const bb_to_sep_fillins,
                                     real *const sep_to_bb_matrix,
                                     real *const sep_to_sep_matrix,
                                     real *const sep_g_matrix,
                                     real *const sep_to_sep_fillins);

    /**
     * Performs the Gaussian elimination of the first two SEP to BB sub-matrix
     * entries pointed by SEP_TO_BB. It is assumed that SEP_TO_BB points to the
     * separation row corresponding to the non-proline backbone block preceding
     * BB_TO_BB_MATRIX.
     *
     * This function works only for proline backbone-blocks and non-proline
     * separation rows.
     *
     * Initial matrix:
     *
     *      bb           sep        g
     *  1 2 3 4 5       1 2 3
     * -----------     -------
     * |1 x      |     |  x  |     |x|       1
     * |0 1 x x x|     |  x  |     |x|       2
     * |  0 1 x x|     |  f  |     |x| bb    3
     * |  0 0 1 x| ... |  f f| ... |x|       4
     * |  0 0 0 1|     |  f x|     |x|       5
     * -----------     -------
     *       .            .
     *       .            .
     *       .            .
     * -----------     -------
     * |x x      |     |? x  |     |x| sep   1 ? Can be 0 or a fillin.
     * -----------     -------
     *
     * Final matrix:
     *
     *      bb           sep        g
     *  1 2 3 4 5       1 2 3
     * -----------     -------
     * |1 x      |     |  x  |     |x|       1
     * |0 1 x x x|     |  x  |     |x|       2
     * |  0 1 x x|     |  f  |     |x| bb    3
     * |  0 0 1 x| ... |  f f| ... |x|       4
     * |  0 0 0 1|     |  f x|     |x|       5
     * -----------     -------
     *       .            .
     *       .            .
     *       .            .
     * -----------     -------
     * |0 0 0 0 0|     |? M F|     |M| sep   1 ? Can be 0 or a fillin.
     * -----------     -------
     *
     * @param bb_to_bb_matrix Pointer to a BB to BB proline sub-matrix.
     * @param bb_to_sep_matrix Pointer to a BB to SEP proline sub-matrix.
     * @param bb_g_matrix Pointer to a BB g proline sub-matrix.
     * @param bb_to_sep_fillins Pointer to a BB to SEP proline fillins
     * sub-matrix.
     * @param sep_to_bb_matrix Pointer to a SEP to BB proline sub-matrix.
     * @param sep_to_sep_matrix Pointer to a SEP to SEP proline sub-matrix.
     * @param sep_g_matrix Pointer to a SEP g proline sub-matrix.
     * @param sep_to_bb_fillins Pointer to a SEP to BB proline fillins
     * sub-matrix.
     * @param sep_to_sep_fillins Pointer to a SEP to SEP proline fillins
     * sub-matrix.
     */
    void solve_6_previous_row_proline(real *const bb_to_bb_matrix,
                                      real *const bb_to_sep_matrix,
                                      real *const bb_g_matrix,
                                      real *const bb_to_sep_fillins,
                                      real *const sep_to_bb_matrix,
                                      real *const sep_to_sep_matrix,
                                      real *const sep_g_matrix,
                                      real *const sep_to_bb_fillins,
                                      real *const sep_to_sep_fillins);

    /**
     * Performs the Gaussian elimination of the first two SEP to BB sub-matrix
     * entries pointed by SEP_TO_BB. It is assumed that SEP_TO_BB points to the
     * separation row corresponding to the non-proline backbone block preceding
     * BB_TO_BB_MATRIX.
     *
     * This function works only for non-proline backbone-blocks and non-proline
     * separation rows.
     *
     * Initial matrix:
     *
     *      bb             sep        g
     *  1 2 3 4 5 6       1 2 3
     * -------------     -------
     * |1 x        |     |  x  |     |x|       1
     * |0 1 x x x  |     |  x  |     |x|       2
     * |  0 1 x x  |     |  f  |     |x| bb    3
     * |  0 0 1 x  | ... |  f  | ... |x|       4
     * |  0 0 0 1 x|     |  f x|     |x|       5
     * |        0 1|     |  f x|     |x|       6
     * -------------     -------
     *       .              .
     *       .              .
     *       .              .
     * -------------     -------
     * |x x        |     |? x  |     |x| sep   1 ? Can be 0 or a fillin.
     * -------------     -------
     *
     * Final matrix:
     *
     *      bb             sep        g
     *  1 2 3 4 5 6       1 2 3
     * -------------     -------
     * |1 x        |     |  x  |     |x|       1
     * |0 1 x x x  |     |  x  |     |x|       2
     * |  0 1 x x  |     |  f  |     |x| bb    3
     * |  0 0 1 x  | ... |  f  | ... |x|       4
     * |  0 0 0 1 x|     |  f x|     |x|       5
     * |        0 1|     |  f x|     |x|       6
     * -------------     -------
     *       .              .
     *       .              .
     *       .              .
     * -------------     -------
     * |0 0 0 0 0 0|     |? M F|     |M| sep   1 ? Can be 0 or a fillin.
     * -------------     -------
     *
     * @param bb_to_bb_matrix Pointer to a BB to BB non-proline sub-matrix.
     * @param bb_to_sep_matrix Pointer to a BB to SEP non-proline sub-matrix.
     * @param bb_g_matrix Pointer to a BB g non-proline sub-matrix.
     * @param bb_to_sep_fillins Pointer to a BB to SEP non-proline fillins
     * sub-matrix.
     * @param sep_to_bb_matrix Pointer to a SEP to BB non-proline sub-matrix.
     * @param sep_to_sep_matrix Pointer to a SEP to SEP non-proline sub-matrix.
     * @param sep_g_matrix Pointer to a SEP g non-proline sub-matrix.
     * @param sep_to_bb_fillins Pointer to a SEP to BB non-proline fillins
     * sub-matrix.
     * @param sep_to_sep_fillins Pointer to a SEP to SEP non-proline fillins
     * sub-matrix.
     */
    void solve_6_previous_row_general(real *const bb_to_bb_matrix,
                                      real *const bb_to_sep_matrix,
                                      real *const bb_g_matrix,
                                      real *const bb_to_sep_fillins,
                                      real *const sep_to_bb_matrix,
                                      real *const sep_to_sep_matrix,
                                      real *const sep_g_matrix,
                                      real *const sep_to_bb_fillins,
                                      real *const sep_to_sep_fillins);

    /**
     * Performs the Gaussian elimination of each SEP to BB sub-matrix entry
     * assigned to THREAD.
     *
     * Initial matrix (general case):
     *
     *      bb              sep         g
     *  1 2 3 4 5 6       1 2 3 4
     * -------------     ---------
     * |1 x        |     |  x    |     |x|       1
     * |0 1 x x x  |     |  x    |     |x|       2
     * |  0 1 x x  |     |  f    |     |x| bb    3
     * |  0 0 1 x  | ... |  f    | ... |x|       4
     * |  0 0 0 1 x|     |  f x  |     |x|       5
     * |        0 1|     |  f x  |     |x|       6
     * -------------     ---------
     *       .              .
     *       .              .
     *       .              .
     * -------------     ---------
     * |x x        |     |  x    |     |x| sep   1 (previous row)
     * |        x x|     |    x  |     |x| sep   2 (current row)
     * -------------     ---------
     *
     * Initial matrix (proline):
     *
     *      bb              sep       g
     *  1 2 3 4 5       1 2 3 4
     * -----------     ---------
     * |1 x      |     |  x    |     |x|       1
     * |0 1 x x x|     |  x    |     |x|       2
     * |  0 1 x x|     |  f    |     |x| bb    3
     * |  0 0 1 x| ... |  f f  | ... |x|       4
     * |  0 0 0 1|     |  f x  |     |x|       5
     * -----------     ---------
     *       .              .
     *       .              .
     *       .              .
     * -----------     ---------
     * |x x      |     |  x    |     |x| sep   1 (previous row)
     * |      f x|     |    x  |     |x| sep   2 (current row)
     * -----------     ---------
     *
     * Final matrix (general case):
     *
     *      bb              sep         g
     *  1 2 3 4 5 6       1 2 3 4
     * -------------     ---------
     * |1 x        |     |  x    |     |x|       1
     * |0 1 x x x  |     |  x    |     |x|       2
     * |  0 1 x x  |     |  f    |     |x| bb    3
     * |  0 0 1 x  | ... |  f    | ... |x|       4
     * |  0 0 0 1 x|     |  f x  |     |x|       5
     * |        0 1|     |  f x  |     |x|       6
     * -------------     ---------
     *       .              .
     *       .              .
     *       .              .
     * -------------     ---------
     * |0 0 0 0 0 0|     |f x f  |     |x| sep   1 (previous row)
     * |        0 0|     |  f x f|     |x| sep   2 (current row)
     * -------------     ---------
     *
     * Final matrix (proline):
     *
     *      bb              sep       g
     *  1 2 3 4 5       1 2 3 4
     * -----------     ---------
     * |1 x      |     |  x    |     |x|       1
     * |0 1 x x x|     |  x    |     |x|       2
     * |  0 1 x x|     |  f    |     |x| bb    3
     * |  0 0 1 x| ... |  f f  | ... |x|       4
     * |  0 0 0 1|     |  f x  |     |x|       5
     * -----------     ---------
     *       .              .
     *       .              .
     *       .              .
     * -----------     ---------
     * |0 0 0 0 0|     |f x f  |     |x| sep   1 (previous row)
     * |        0|     |  f x f|     |x| sep   2 (current row)
     * -----------     ---------
     *
     * @param thread Thread descriptor [0 - NTHREADS - 1].
     */
    void solve_6(const int thread);

    /**
     * Performs the Gaussian elimination of each private SEP to SEP left-fillin
     * assigned to thread. Also cleans the left-fillin of the THREADth shared
     * SEP to SEP sub-matrix (if there is one).
     *
     * On return there is one new column of sub-diagonal left-fillins starting
     * on the diagonal entry of the THREAD - 1th shared row (first private row
     * for thread 0).
     *
     * Initial matrix (4 threads, 4 side-chains per thread, 3 side-chains last
     * thread):
     *
     *               sep                    g
     * -------------------------------
     * |x f                          |     |x|      1
     * |f x f                        |     |x|      2
     * |  f x f                      |     |x|      3
     * |    f x f                    |     |x|      Shared row 1
     * |      f x f                  |     |x|      1
     * |        f x f                |     |x|      2
     * |          f x f              |     |x| sep  3
     * |            f x f            |     |x|      Shared row 2
     * |              f x f          |     |x|      1
     * |                f x f        |     |x|      2
     * |                  f x f      |     |x|      3
     * |                    f x f    |     |x|      Shared row 3
     * |                      f x f  |     |x|      1
     * |                        f x f|     |x|      2
     * |                          f x|     |x|      3
     * -------------------------------
     *
     * Final matrix:
     *
     *             sep                      g
     * -------------------------------
     * |1 M                          |     |M|      1
     * |0 1 M                        |     |M|      2
     * |  0 1 M                      |     |M|      3
     * |    0 M f                    |     |M|      Shared row 1
     * |      M 1 M                  |     |M|      1
     * |      F 0 1 M                |     |M|      2
     * |      F   0 1 M              |     |M| sep  3
     * |      F     0 M f            |     |M|      Shared row 2
     * |              M 1 M          |     |M|      1
     * |              F 0 1 M        |     |M|      2
     * |              F   0 1 M      |     |M|      3
     * |              F     0 M f    |     |M|      Shared row 3
     * |                      M 1 M  |     |M|      1
     * |                      F 0 1 M|     |M|      2
     * |                      F   0 1|     |M|      3
     * -------------------------------
     *
     * @param thread Thread descriptor [0 - NTHREADS - 1].
     */
    void solve_7(const int thread);

    /**
     * Performs the Gaussian elimination of each private SEP to SEP right-fillin
     * assigned to thread. Also cleans the right-fillin of the THREAD - 1th
     * shared SEP to SEP sub-matrix (if there is one).
     *
     * On return there is one new column of super-diagonal right-fillins
     * starting on the diagonal entry of the THREAD - 1th shared row (first
     * private row for thread 0).
     *
     * Initial matrix (4 threads, 4 side-chains per thread, 3 side-chains last
     * thread):
     *
     *               sep                    g
     * -------------------------------
     * |1 f                          |     |x|      1
     * |0 1 f                        |     |x|      2
     * |  0 1 f                      |     |x|      3
     * |    0 x f                    |     |x|      Shared row 1
     * |      f 1 f                  |     |x|      1
     * |      f 0 1 f                |     |x|      2
     * |      f   0 1 f              |     |x| sep  3
     * |      f     0 x f            |     |x|      Shared row 2
     * |              f 1 f          |     |x|      1
     * |              f 0 1 f        |     |x|      2
     * |              f   0 1 f      |     |x|      3
     * |              f     0 x f    |     |x|      Shared row 3
     * |                      f 1 f  |     |x|      1
     * |                      f 0 1 f|     |x|      2
     * |                      f   0 1|     |x|      3
     * -------------------------------
     *
     * Final matrix:
     *
     *               sep                    g
     * -------------------------------
     * |1 0   F                      |     |M|      1
     * |0 1 0 F                      |     |M|      2
     * |  0 1 f                      |     |x|      3
     * |    0 M 0     F              |     |M|      Shared row 1
     * |      M 1 0   F              |     |M|      1
     * |      M 0 1 0 F              |     |M|      2
     * |      f   0 1 f              |     |x| sep  3
     * |      f     0 M 0     F      |     |M|      Shared row 2
     * |              M 1 0   F      |     |M|      1
     * |              M 0 1 0 F      |     |M|      2
     * |              f   0 1 f      |     |x|      3
     * |              f     0 M 0    |     |M|      Shared row 3
     * |                      M 1 0  |     |M|      1
     * |                      M 0 1 0|     |M|      2
     * |                      f   0 1|     |x|      3
     * -------------------------------
     *
     * @param thread Thread descriptor [0 - NTHREADS - 1].
     */
    void solve_8(const int thread);

    /**
     * Performs the Gaussian elimination of the subdiagonal entries of the Schur
     * matrix. Also make 1s in the diagonal entries.
     *
     * Example for NTHREADS = 4 =>
     *
     * Initial Schur matrix:
     *
     *   bb         g
     *  1 2 3
     * -------
     * |x x  |     |x|      1 Schur row 1
     * |x x x| ..  |x| bb   2 Schur row 2
     * |  x x|     |x|      3 Schur row 3
     * -------
     *
     * Final Schur matrix:
     *
     *   bb         g
     *  1 2 3
     * -------
     * |1 M  |     |M|      1 Schur row 1
     * |0 1 M| ..  |M| bb   2 Schur row 2
     * |  0 1|     |M|      3 Schur row 3
     * -------
     *
     */
    void solve_9();

    /**
     * Performs the Gaussian elimination of the superdiagonal entries of the
     * Schur matrix.
     *
     * Example for NTHREADS = 4 =>
     *
     * Initial Schur matrix:
     *
     *   bb         g
     *  2 3 4
     * -------
     * |1 x  |     |x|      1 Schur row 1
     * |0 1 x| ..  |x| bb   2 Schur row 2
     * |  0 1|     |x|      3 Schur row 3
     * -------
     *
     * Final Schur matrix:
     *
     *   bb         g
     *  2 3 4
     * -------
     * |1 0  |     |M|      1 Schur row 1
     * |0 1 0| ..  |M| bb   2 Schur row 2
     * |  0 1|     |x|      3 Schur row 3
     * -------
     *
     */
    void solve_10();

    /**
     * Cleans every subdiagonal fillin of each private SEP to SEP sub-matrix
     * assigned to THREAD using the Schur rows (shared rows).
     *
     * Initial matrix (4 threads, 4 side-chains per thread, 3 side-chains last
     * thread):
     *
     *                 sep                  g
     * -------------------------------
     * |1 0   f                      |     |x|      1
     * |0 1 0 f                      |     |x|      2
     * |  0 1 f                      |     |x|      3
     * |    0 1 0     0              |     |x|      Shared row 1
     * |      f 1 0   f              |     |x|      1
     * |      f 0 1 0 f              |     |x|      2
     * |      f   0 1 f              |     |x| sep  3
     * |      0     0 1 0     0      |     |x|      Shared row 2
     * |              f 1 0   f      |     |x|      1
     * |              f 0 1 0 f      |     |x|      2
     * |              f   0 1 f      |     |x|      3
     * |              0     0 1 0    |     |x|      Shared row 3
     * |                      f 1 0  |     |x|      1
     * |                      f 0 1 0|     |x|      2
     * |                      f   0 1|     |x|      3
     * -------------------------------
     *
     * Final matrix:
     *
     *                 sep                  g
     * -------------------------------
     * |1 0   f                      |     |x|      1
     * |0 1 0 f                      |     |x|      2
     * |  0 1 f                      |     |x|      3
     * |    0 1 0     0              |     |x|      Shared row 1
     * |      0 1 0   f              |     |M|      1
     * |      0 0 1 0 f              |     |M|      2
     * |      0   0 1 f              |     |M| sep  3
     * |      0     0 1 0     0      |     |x|      Shared row 2
     * |              0 1 0   f      |     |M|      1
     * |              0 0 1 0 f      |     |M|      2
     * |              0   0 1 f      |     |M|      3
     * |              0     0 1 0    |     |x|      Shared row 3
     * |                      0 1 0  |     |M|      1
     * |                      0 0 1 0|     |M|      2
     * |                      0   0 1|     |M|      3
     * -------------------------------
     *
     * @param thread Thread descriptor [0 - NTHREADS - 1].
     */
    void solve_11(const int thread);

    /**
     * Cleans every superdiagonal fillin of each private SEP to SEP sub-matrix
     * assigned to THREAD using the Schur rows (shared rows).
     *
     * Initial matrix (4 threads, 4 side-chains per thread, 3 side-chains last
     * thread):
     *
     *                 sep                  g
     * -------------------------------
     * |1 0   f                      |     |x|      1
     * |0 1 0 f                      |     |x|      2
     * |  0 1 f                      |     |x|      3
     * |    0 1 0     0              |     |x|      Shared row 1
     * |      0 1 0   f              |     |x|      1
     * |      0 0 1 0 f              |     |x|      2
     * |      0   0 1 f              |     |x| sep  3
     * |      0     0 1 0     0      |     |x|      Shared row 2
     * |              0 1 0   f      |     |x|      1
     * |              0 0 1 0 f      |     |x|      2
     * |              0   0 1 f      |     |x|      3
     * |              0     0 1 0    |     |x|      Shared row 3
     * |                      0 1 0  |     |x|      1
     * |                      0 0 1 0|     |x|      2
     * |                      0   0 1|     |x|      3
     * -------------------------------
     *
     * Final matrix:
     *
     *                 sep                  g
     * -------------------------------
     * |1 0   0                      |     |M|      1
     * |0 1 0 0                      |     |M|      2
     * |  0 1 0                      |     |M|      3
     * |    0 1 0     0              |     |x|      Shared row 1
     * |      0 1 0   0              |     |M|      1
     * |      0 0 1 0 0              |     |M|      2
     * |      0   0 1 0              |     |M| sep  3
     * |      0     0 1 0     0      |     |x|      Shared row 2
     * |              0 1 0   0      |     |M|      1
     * |              0 0 1 0 0      |     |M|      2
     * |              0   0 1 0      |     |M|      3
     * |              0     0 1 0    |     |x|      Shared row 3
     * |                      0 1 0  |     |x|      1
     * |                      0 0 1 0|     |x|      2
     * |                      0   0 1|     |x|      3
     * -------------------------------
     *
     * @param thread Thread descriptor [0 - NTHREADS - 1].
     */
    void solve_12(const int thread);

    /**
     * Performs the Gaussian elimination of each BB to SEP sub-matrix entry
     * pointed by BB_TO_SEP_MATRIX and BB_TO_SEP_FILLINS.
     *
     * This function works only for proline side-chains.
     *
     * Initial matrix:
     *
     *      bb            sep       g
     *  1 2 3 4 5       1 2
     * -----------     -----
     * |1 x      |     |x  |     |x|       1
     * |0 1 x x x|     |x  |     |x|       2
     * |  0 1 x x|     |f  |     |x| bb    3 Nth Backbone Block
     * |  0 0 1 x| ... |f f| ... |x|       4
     * |  0 0 0 1|     |f x|     |x|       5
     * -------------   -----
     *                   .
     *                   .
     *                 -----
     *                 |1  |     |x| sep   1 N - 1th separation row
     *                 |  1|     |x|       2 Nth separation row
     *                 -----
     *
     * Final matrix:
     *
     *      bb          sep       g
     *  1 2 3 4 5       1 2
     * -----------     -----
     * |1 x      |     |0  |     |M|       1
     * |0 1 x x x|     |0  |     |M|       2
     * |  0 1 x x|     |0  |     |M| bb    3 Nth Backbone Block
     * |  0 0 1 x| ... |0 0| ... |M|       4
     * |  0 0 0 1|     |0 0|     |M|       5
     * -------------   -----
     *                   .
     *                   .
     *                 -----
     *                 |1  |     |x| sep   1 N - 1th separation row
     *                 |  1|     |x|       2 Nth separation row
     *                 -----
     *
     * @param bb_to_sep_matrix Pointer to a BB to SEP proline sub-matrix.
     * @param bb_g_matrix Pointer to a BB g proline sub-matrix.
     * @param bb_to_sep_fillins Pointer to a BB to SEP proline fillins
     * sub-matrix.
     * @param sep_g_entry value of g of the Nth separation row.
     * @param prev_sep_g_entry value of g of the Nth - 1 separation row (0 if
     * there is not one).
     */
    void solve_13_proline(real *bb_to_sep_matrix,
                          real *bb_g_matrix,
                          real *bb_to_sep_fillins,
                          real sep_g_entry,
                          real prev_sep_g_entry);

    /**
     * Performs the Gaussian elimination of each BB to SEP sub-matrix entry
     * pointed by BB_TO_SEP_MATRIX and BB_TO_SEP_FILLINS.
     *
     * This function works only for non-proline side-chains.
     *
     * Initial matrix:
     *
     *      bb            sep       g
     *  1 2 3 4 5 6       1 2
     * -------------     -----
     * |1 x        |     |x  |     |x|       1
     * |0 1 x x x  |     |x  |     |x|       2
     * |  0 1 x x  |     |f  |     |x| bb    3 Nth Backbone Block
     * |  0 0 1 x  | ... |f  | ... |x|       4
     * |  0 0 0 1 x|     |f x|     |x|       5
     * |        0 1|     |f x|     |x|       6
     * -------------     -----
     *                     .
     *                     .
     *                   -----
     *                   |1  |     |x| sep   1 N - 1th separation row
     *                   |  1|     |x|       2 Nth separation row
     *                   -----
     *
     * Final matrix:
     *
     *      bb            sep       g
     *  1 2 3 4 5 6       1 2
     * -------------     -----
     * |1 x        |     |0  |     |M|       1
     * |0 1 x x x  |     |0  |     |M|       2
     * |  0 1 x x  |     |0  |     |M| bb    3 Nth Backbone Block
     * |  0 0 1 x  | ... |0  | ... |M|       4
     * |  0 0 0 1 x|     |0 0|     |M|       5
     * |        0 1|     |0 0|     |M|       6
     * -------------     -----
     *                     .
     *                     .
     *                   -----
     *                   |1  |     |x| sep   1 N - 1th separation row
     *                   |  1|     |x|       2 Nth separation row
     *                   -----
     *
     * @param bb_to_sep_matrix Pointer to a BB to SEP non-proline sub-matrix.
     * @param bb_g_matrix Pointer to a BB g non-proline sub-matrix.
     * @param bb_to_sep_fillins Pointer to a BB to SEP non-proline fillins
     * sub-matrix.
     * @param sep_g_entry value of g of the Nth separation row.
     * @param prev_sep_g_entry value of g of the Nth - 1 separation row (0 if
     * there is not one).
     */
    void solve_13_general(real *bb_to_sep_matrix,
                          real *bb_g_matrix,
                          real *bb_to_sep_fillins,
                          real sep_g_entry,
                          real prev_sep_g_entry);

    /**
     * Performs the Gaussian elimination of each BB to SEP assigned to THREAD.
     *
     * Initial matrix (general case):
     *
     *      bb            sep       g
     *  1 2 3 4 5 6       1 2
     * -------------     -----
     * |1 x        |     |x  |     |x|       1
     * |0 1 x x x  |     |x  |     |x|       2
     * |  0 1 x x  |     |f  |     |x| bb    3 Nth Backbone Block
     * |  0 0 1 x  | ... |f  | ... |x|       4
     * |  0 0 0 1 x|     |f x|     |x|       5
     * |        0 1|     |f x|     |x|       6
     * -------------     -----
     *                     .
     *                     .
     *                   -----
     *                   |1  |     |x| sep   1 N - 1th separation row
     *                   |  1|     |x|       2 Nth separation row
     *                   -----
     *
     * Initial matrix (proline):
     *
     *      bb          sep       g
     *  1 2 3 4 5       1 2
     * -----------     -----
     * |1 x      |     |x  |     |x|       1
     * |0 1 x x x|     |x  |     |x|       2
     * |  0 1 x x|     |f  |     |x| bb    3 Nth Backbone Block
     * |  0 0 1 x| ... |f f| ... |x|       4
     * |  0 0 0 1|     |f x|     |x|       5
     * -----------     -----
     *                   .
     *                   .
     *                 -----
     *                 |1  |     |x| sep   1 N - 1th separation row
     *                 |  1|     |x|       2 Nth separation row
     *                 -----
     *
     * Final matrix (general case):
     *
     *      bb            sep       g
     *  1 2 3 4 5 6       1 2
     * -------------     -----
     * |1 x        |     |0  |     |M|       1
     * |0 1 x x x  |     |0  |     |M|       2
     * |  0 1 x x  |     |0  |     |M| bb    3 Nth Backbone Block
     * |  0 0 1 x  | ... |0  | ... |M|       4
     * |  0 0 0 1 x|     |0 0|     |M|       5
     * |        0 1|     |0 0|     |M|       6
     * -------------     -----
     *                     .
     *                     .
     *                   -----
     *                   |1  |     |x| sep   1 N - 1th separation row
     *                   |  1|     |x|       2 Nth separation row
     *                   -----
     *
     * Final matrix (proline):
     *
     *      bb          sep       g
     *  1 2 3 4 5       1 2
     * -----------     -----
     * |1 x      |     |0  |     |M|       1
     * |0 1 x x x|     |0  |     |M|       2
     * |  0 1 x x|     |0  |     |M| bb    3 Nth Backbone Block
     * |  0 0 1 x| ... |0 0| ... |M|       4
     * |  0 0 0 1|     |0 0|     |M|       5
     * -----------     -----
     *                   .
     *                   .
     *                 -----
     *                 |1  |     |x| sep   1 N - 1th separation row
     *                 |  1|     |x|       2 Nth separation row
     *                 -----
     *
     * @param thread Thread descriptor [0 - NTHREADS - 1].
     */
    void solve_13(const int thread);

    /**
     * Performs the Gaussian elimination of each of the SC to SEP
     * proline-only-entries assigned to THREAD.
     *
     * Initial matrix:
     *
     * sep      g
     *
     * ---
     * | |     |x|  1
     * | |     |x|  2
     * | |     |x|  3
     * | |     |x|  4
     * | | ... |x|  5 sc
     * | |     |x|  6
     * | |     |x|  7
     * | |     |x|  8
     * |x|     |x|  9
     * ---
     *  .
     *  .
     * ---
     * |1| ... |x| sep   1 Nth separation row
     * ---
     *
     * Final matrix:
     *
     * sep      g
     *
     * ---
     * | |     |x|  1
     * | |     |x|  2
     * | |     |x|  3
     * | |     |x|  4
     * | | ... |x|  5 sc
     * | |     |x|  6
     * | |     |x|  7
     * | |     |x|  8
     * |0|     |M|  9
     * ---
     *  .
     *  .
     * ---
     * |1| ... |x| sep   1 Nth separation row
     * ---
     *
     * @param thread Thread descriptor [0 - NTHREADS - 1].
     */
    void solve_14(const int thread);

    /**
     * Performs the Gaussian elimination of the superdiagonal entries of the
     * first BB to BB sub-matrix pointed by BB_TO_BB_MATRIX.
     *
     * It is assumed that BB_TO_BB_MATRIX and BB_G_MATRIX point to the same
     * proline backbone data.
     *
     * Initial matrix:
     *
     *      bb          sep       g
     *  1 2 3 4 5       1 2
     * -----------     -----
     * |1 x      |     |0  |     |x|       1
     * |0 1 x x x|     |0  |     |x|       2
     * |  0 1 x x|     |0  |     |x| bb    3
     * |  0 0 1 x| ... |0 0| ... |x|       4
     * |  0 0 0 1|     |0 0|     |x|       5
     * -----------     -----
     *
     * Final matrix:
     *
     *      bb          sep       g
     *  1 2 3 4 5       1 2
     * -----------     -----
     * |1 0      |     |0  |     |M|       1
     * |0 1 0 0 0|     |0  |     |M|       2
     * |  0 1 0 0|     |0  |     |M| bb    3
     * |  0 0 1 0| ... |0 0| ... |M|       4
     * |  0 0 0 1|     |0 0|     |x|       5
     * -----------     -----
     *
     * @param bb_to_bb_matrix Pointer to a BB to BB proline sub-matrix.
     * @param bb_g_matrix Pointer to a BB g proline sub-matrix.
     */
    void solve_15_proline(real *const bb_to_bb_matrix, real *const bb_g_matrix);

    /**
     * Performs the Gaussian elimination of the superdiagonal entries of the
     * first BB to BB sub-matrix pointed by BB_TO_BB_MATRIX.
     *
     * It is assumed that BB_TO_BB_MATRIX and BB_G_MATRIX point to the same
     * non-proline backbone data.
     *
     * Initial matrix:
     *
     *      bb            sep       g
     *  1 2 3 4 5 6       1 2
     * -------------     -----
     * |1 x        |     |0  |     |x|       1
     * |0 1 x x x  |     |0  |     |x|       2
     * |  0 1 x x  |     |0  |     |x| bb    3
     * |  0 0 1 x  | ... |0  | ... |x|       4
     * |  0 0 0 1 x|     |0 0|     |x|       5
     * |        0 1|     |0 0|     |x|       6
     * -------------     -----
     *
     * Final matrix:
     *
     *      bb            sep       g
     *  1 2 3 4 5 6       1 2
     * -------------     -----
     * |1 0        |     |0  |     |M|       1
     * |0 1 0 0 0  |     |0  |     |M|       2
     * |  0 1 0 0  |     |0  |     |M| bb    3
     * |  0 0 1 0  | ... |0  | ... |M|       4
     * |  0 0 0 1 0|     |0 0|     |M|       5
     * |        0 1|     |0 0|     |x|       6
     * -------------     -----
     *
     * @param bb_to_bb_matrix Pointer to a BB to BB non-proline sub-matrix.
     * @param bb_g_matrix Pointer to a BB g non-proline sub-matrix.
     */
    void solve_15_general(real *const bb_to_bb_matrix, real *const bb_g_matrix);

    /**
     * Performs the Gaussian elimination of the superdiagonal entries of each BB
     * to BB sub-matrix assigned to THREAD.
     *
     * Initial matrix (general case):
     *
     *      bb            sep       g
     *  1 2 3 4 5 6       1 2
     * -------------     -----
     * |1 x        |     |0  |     |x|       1
     * |0 1 x x x  |     |0  |     |x|       2
     * |  0 1 x x  |     |0  |     |x| bb    3
     * |  0 0 1 x  | ... |0  | ... |x|       4
     * |  0 0 0 1 x|     |0 0|     |x|       5
     * |        0 1|     |0 0|     |x|       6
     * -------------     -----
     *
     * Initial matrix (proline):
     *
     *      bb          sep       g
     *  1 2 3 4 5       1 2
     * -----------     -----
     * |1 x      |     |0  |     |x|       1
     * |0 1 x x x|     |0  |     |x|       2
     * |  0 1 x x|     |0  |     |x| bb    3
     * |  0 0 1 x| ... |0 0| ... |x|       4
     * |  0 0 0 1|     |0 0|     |x|       5
     * -----------     -----
     *
     * Final matrix (general case):
     *
     *      bb            sep       g
     *  1 2 3 4 5 6       1 2
     * -------------     -----
     * |1 0        |     |0  |     |M|       1
     * |0 1 0 0 0  |     |0  |     |M|       2
     * |  0 1 0 0  |     |0  |     |M| bb    3
     * |  0 0 1 0  | ... |0  | ... |M|       4
     * |  0 0 0 1 0|     |0 0|     |M|       5
     * |        0 1|     |0 0|     |x|       6
     * -------------     -----
     *
     * Final matrix (proline):
     *
     *      bb          sep       g
     *  1 2 3 4 5       1 2
     * -----------     -----
     * |1 0      |     |0  |     |M|       1
     * |0 1 0 0 0|     |0  |     |M|       2
     * |  0 1 0 0|     |0  |     |M| bb    3
     * |  0 0 1 0| ... |0 0| ... |M|       4
     * |  0 0 0 1|     |0 0|     |x|       5
     * -----------     -----
     *
     * @param thread Thread descriptor [0 - NTHREADS - 1].
     */
    void solve_15(const int thread);

    /**
     * Performs the Gaussian elimination of each entry of the first SC to BB
     * sub-matrix pointed by SC_TO_BB_MATRIX.
     *
     * It is assumed that SC_TO_SC_MATRIX, SC_G_MATRIX and BB_G_MATRIX
     * point to the same glycine side-chain data.
     *
     * This function has been automatically generated by
     * step_2_3_16_17_generator.c.
     *
     * Initial matrix:
     *
     * sc           bb            g
     *          1 2 3 4 5 6
     * ---     -------------
     * ---     -------------
     *  .             .
     *  .             .
     *  .             .
     * ---     -------------
     * | |     |1 0        |     |x|  1
     * | |     |0 1 0 0 0  |     |x|  2
     * | |     |  0 1 0 0  |     |x|  3
     * | | ... |  0 0 1 0  | ... |x|  4 bb
     * | |     |  0 0 0 1 0|     |x|  5
     * | |     |        0 1|     |x|  6
     * ---     -------------
     *
     * Final matrix:
     *
     * sc           bb            g
     *          1 2 3 4 5 6
     * ---     -------------
     * ---     -------------
     *  .             .
     *  .             .
     *  .             .
     * ---     -------------
     * | |     |1 0        |     |x|  1
     * | |     |0 1 0 0 0  |     |x|  2
     * | |     |  0 1 0 0  |     |x|  3
     * | | ... |  0 0 1 0  | ... |x|  4 bb
     * | |     |  0 0 0 1 0|     |x|  5
     * | |     |        0 1|     |x|  6
     * ---     -------------
     *
     * @param sc_to_bb_matrix unused.
     * @param sc_g_matrix unused.
     * @param sc_to_bb_fillins unused.
     * @param bb_g_matrix unused.
     */
    void solve_16_glycine(__attribute__((unused)) real *const sc_to_bb_matrix,
                          __attribute__((unused)) real *const sc_g_matrix,
                          __attribute__((unused)) real *const sc_to_bb_fillins,
                          __attribute__((unused)) real *const bb_g_matrix);

    /**
     * Performs the Gaussian elimination of each entry of the first SC to BB
     * sub-matrix pointed by SC_TO_BB_MATRIX.
     *
     * It is assumed that SC_TO_SC_MATRIX, SC_G_MATRIX and BB_G_MATRIX
     * point to the same proline side-chain data.
     *
     * Initial matrix:
     *
     *         sc                  bb           g
     *  1 2 3 4 5 6 7 8 9       1 2 3 4 5
     * -------------------     -----------
     * |1 x         x    |     |      x  |     |x|  1
     * |0 1         x    |     |      x  |     |x|  2
     * |    1 x     x x  |     |         |     |x|  3
     * |    0 1     x x  |     |         |     |x|  4
     * |        1 x   x x| ... |         | ... |x|  5 sc
     * |        0 1   x x|     |         |     |x|  6
     * |0 0 0 0     1 x  |     |      x  |     |x|  7
     * |    0 0 0 0 0 1 x|     |      f  |     |x|  8
     * |        0 0   0 1|     |      f x|     |x|  9
     * -------------------     -----------
     *          .                   .
     *          .                   .
     *          .                   .
     * -------------------     -----------
     * |                 |     |1 0      |     |x|  1
     * |                 |     |0 1 0 0 0|     |x|  2
     * |                 | ... |  0 1 0 0| ... |x|  3 bb
     * |0 0         0 0 0|     |  0 0 1 0|     |x|  4
     * |                0|     |  0 0 0 1|     |x|  5
     * -------------------     -----------
     *
     * Final matrix:
     *
     *         sc                  bb           g
     *  1 2 3 4 5 6 7 8 9       1 2 3 4 5
     * -------------------     -----------
     * |1 x         x    |     |      0  |     |M|  1
     * |0 1         x    |     |      0  |     |M|  2
     * |    1 x     x x  |     |         |     |x|  3
     * |    0 1     x x  |     |         |     |x|  4
     * |        1 x   x x| ... |         | ... |x|  5 sc
     * |        0 1   x x|     |         |     |x|  6
     * |0 0 0 0     1 x  |     |      0  |     |M|  7
     * |    0 0 0 0 0 1 x|     |      0  |     |M|  8
     * |        0 0   0 1|     |      0 0|     |M|  9
     * -------------------     -----------
     *          .                   .
     *          .                   .
     *          .                   .
     * -------------------     -----------
     * |                 |     |1 0      |     |x|  1
     * |                 |     |0 1 0 0 0|     |x|  2
     * |                 | ... |  0 1 0 0| ... |x|  3 bb
     * |0 0         0 0 0|     |  0 0 1 0|     |x|  4
     * |                0|     |  0 0 0 1|     |x|  5
     * -------------------     -----------
     *
     * @param sc_to_bb_matrix Pointer to a SC to BB proline sub-matrix.
     * @param sc_g_matrix Pointer to a SC g proline sub-matrix.
     * @param sc_to_bb_fillins Pointer to a SC to BB proline fillins sub-matrix.
     * @param bb_g_matrix Pointer to a BB g proline sub-matrix.
     */
    void solve_16_proline(real *const sc_to_bb_matrix,
                          real *const sc_g_matrix,
                          real *const sc_to_bb_fillins,
                          real *const bb_g_matrix);

    /**
     * Performs the Gaussian elimination of each entry of the first SC to BB
     * sub-matrix pointed by SC_TO_BB_MATRIX.
     *
     * It is assumed that SC_TO_SC_MATRIX, SC_G_MATRIX and BB_G_MATRIX
     * point to the same cysteine side-chain data.
     *
     * This function has been automatically generated by
     * step_2_3_16_17_generator.c.
     *
     * Initial matrix:
     *
     *    sc              bb            g
     *  1 2 3 4       1 2 3 4 5 6
     * ---------     -------------
     * |1 x    |     |           |     |x|  1
     * |0 1 x x|     |      x    |     |x|  2
     * |  0 1 x| ... |      x    | ... |x|  3 sc
     * |  0 0 1|     |      x    |     |x|  4
     * ---------     -------------
     *     .               .
     *     .               .
     *     .               .
     * ---------     -------------
     * |       |     |1 0        |     |x|  1
     * |       |     |0 1 0 0 0  |     |x|  2
     * |       |     |  0 1 0 0  |     |x|  3
     * |  0 0 0| ... |  0 0 1 0  | ... |x|  4 bb
     * |       |     |  0 0 0 1 0|     |x|  5
     * |       |     |        0 1|     |x|  6
     * ---------     -------------
     *
     * Final matrix:
     *
     *    sc              bb            g
     *  1 2 3 4       1 2 3 4 5 6
     * ---------     -------------
     * |1 x    |     |           |     |x|  1
     * |0 1 x x|     |      0    |     |M|  2
     * |  0 1 x| ... |      0    | ... |M|  3 sc
     * |  0 0 1|     |      0    |     |M|  4
     * ---------     -------------
     *     .               .
     *     .               .
     *     .               .
     * ---------     -------------
     * |       |     |1 0        |     |x|  1
     * |       |     |0 1 0 0 0  |     |x|  2
     * |       |     |  0 1 0 0  |     |x|  3
     * |  0 0 0| ... |  0 0 1 0  | ... |x|  4 bb
     * |       |     |  0 0 0 1 0|     |x|  5
     * |       |     |        0 1|     |x|  6
     * ---------     -------------
     *
     * @param sc_to_bb_matrix Pointer to a SC to BB cysteine sub-matrix.
     * @param sc_g_matrix Pointer to a SC g cysteine sub-matrix.
     * @param bb_g_matrix Pointer to a BB g cysteine sub-matrix.
     */
    void solve_16_cysteine(real *const sc_to_bb_matrix,
                           real *const sc_g_matrix,
                           __attribute__((unused)) real *const sc_to_bb_fillins,
                           real *const bb_g_matrix);

    /**
     * Performs the Gaussian elimination of each entry of the first SC to BB
     * sub-matrix pointed by SC_TO_BB_MATRIX.
     *
     * It is assumed that SC_TO_SC_MATRIX, SC_G_MATRIX and BB_G_MATRIX
     * point to the same methionine side-chain data.
     *
     * This function has been automatically generated by
     * step_2_3_16_17_generator.c.
     *
     * Initial matrix:
     *
     *          sc                    bb            g
     *  1 2 3 4 5 6 7 8 910       1 2 3 4 5 6
     * ---------------------     -------------
     * |1 x x x            |     |           |     |x|  1
     * |0 1 x x            |     |           |     |x|  2
     * |0 0 1 x            |     |           |     |x|  3
     * |0 0 0 1 x          |     |           |     |x|  4
     * |      0 1 x x x    |     |           |     |x|  5
     * |        0 1 x x    | ... |           | ... |x|  6 sc
     * |        0 0 1 x    |     |           |     |x|  7
     * |        0 0 0 1 x x|     |      x    |     |x|  8
     * |              0 1 x|     |      x    |     |x|  9
     * |              0 0 1|     |      x    |     |x| 10
     * ---------------------     -------------
     *           .                     .
     *           .                     .
     *           .                     .
     * ---------------------     -------------
     * |                   |     |1 0        |     |x|  1
     * |                   |     |0 1 0 0 0  |     |x|  2
     * |                   |     |  0 1 0 0  |     |x|  3
     * |              0 0 0| ... |  0 0 1 0  | ... |x|  4 bb
     * |                   |     |  0 0 0 1 0|     |x|  5
     * |                   |     |        0 1|     |x|  6
     * ---------------------     -------------
     *
     * Final matrix:
     *
     *          sc                    bb            g
     *  1 2 3 4 5 6 7 8 910       1 2 3 4 5 6
     * ---------------------     -------------
     * |1 x x x            |     |           |     |x|  1
     * |0 1 x x            |     |           |     |x|  2
     * |0 0 1 x            |     |           |     |x|  3
     * |0 0 0 1 x          |     |           |     |x|  4
     * |      0 1 x x x    |     |           |     |x|  5
     * |        0 1 x x    | ... |           | ... |x|  6 sc
     * |        0 0 1 x    |     |           |     |x|  7
     * |        0 0 0 1 x x|     |      0    |     |M|  8
     * |              0 1 x|     |      0    |     |M|  9
     * |              0 0 1|     |      0    |     |M| 10
     * ---------------------     -------------
     *           .                     .
     *           .                     .
     *           .                     .
     * ---------------------     -------------
     * |                   |     |1 0        |     |x|  1
     * |                   |     |0 1 0 0 0  |     |x|  2
     * |                   |     |  0 1 0 0  |     |x|  3
     * |              0 0 0| ... |  0 0 1 0  | ... |x|  4 bb
     * |                   |     |  0 0 0 1 0|     |x|  5
     * |                   |     |        0 1|     |x|  6
     * ---------------------     -------------
     *
     * @param sc_to_bb_matrix Pointer to a SC to BB methionine sub-matrix.
     * @param sc_g_matrix Pointer to a SC g methionine sub-matrix.
     * @param bb_g_matrix Pointer to a BB g methionine sub-matrix.
     */
    void solve_16_methionine(real *const sc_to_bb_matrix,
                             real *const sc_g_matrix,
                             __attribute__((unused)) real *const sc_to_bb_fillins,
                             real *const bb_g_matrix);

    /**
     * Performs the Gaussian elimination of each entry of the first SC to BB
     * sub-matrix pointed by SC_TO_BB_MATRIX.
     *
     * It is assumed that SC_TO_SC_MATRIX, SC_G_MATRIX and BB_G_MATRIX
     * point to the same alaline side-chain data.
     *
     * This function has been automatically generated by
     * step_2_3_16_17_generator.c.
     *
     * Initial matrix:
     *
     *   sc             bb            g
     *  1 2 3       1 2 3 4 5 6
     * -------     -------------
     * |1 x x|     |      x    |     |x|  1
     * |0 1 x| ... |      x    | ... |x|  2 sc
     * |0 0 1|     |      x    |     |x|  3
     * -------     -------------
     *    .              .
     *    .              .
     *    .              .
     * -------     -------------
     * |     |     |1 0        |     |x|  1
     * |     |     |0 1 0 0 0  |     |x|  2
     * |     |     |  0 1 0 0  |     |x|  3
     * |0 0 0| ... |  0 0 1 0  | ... |x|  4 bb
     * |     |     |  0 0 0 1 0|     |x|  5
     * |     |     |        0 1|     |x|  6
     * -------     -------------
     *
     * Final matrix:
     *
     *   sc             bb            g
     *  1 2 3       1 2 3 4 5 6
     * -------     -------------
     * |1 x x|     |      0    |     |M|  1
     * |0 1 x| ... |      0    | ... |M|  2 sc
     * |0 0 1|     |      0    |     |M|  3
     * -------     -------------
     *    .              .
     *    .              .
     *    .              .
     * -------     -------------
     * |     |     |1 0        |     |x|  1
     * |     |     |0 1 0 0 0  |     |x|  2
     * |     |     |  0 1 0 0  |     |x|  3
     * |0 0 0| ... |  0 0 1 0  | ... |x|  4 bb
     * |     |     |  0 0 0 1 0|     |x|  5
     * |     |     |        0 1|     |x|  6
     * -------     -------------
     *
     * @param sc_to_bb_matrix Pointer to a SC to BB alaline sub-matrix.
     * @param sc_g_matrix Pointer to a SC g alaline sub-matrix.
     * @param sc_to_bb_fillins unused.
     * @param bb_g_matrix Pointer to a BB g alaline sub-matrix.
     */
    void solve_16_alaline(real *const sc_to_bb_matrix,
                          real *const sc_g_matrix,
                          __attribute__((unused)) real *const sc_to_bb_fillins,
                          real *const bb_g_matrix);

    /**
     * Performs the Gaussian elimination of each entry of the first SC to BB
     * sub-matrix pointed by SC_TO_BB_MATRIX.
     *
     * It is assumed that SC_TO_SC_MATRIX, SC_G_MATRIX and BB_G_MATRIX
     * point to the same valine side-chain data.
     *
     * This function has been automatically generated by
     * step_2_3_16_17_generator.c.
     *
     * Initial matrix:
     *
     *         sc                   bb            g
     *  1 2 3 4 5 6 7 8 9       1 2 3 4 5 6
     * -------------------     -------------
     * |1 x x x          |     |           |     |x|  1
     * |0 1 x x          |     |           |     |x|  2
     * |0 0 1 x          |     |           |     |x|  3
     * |0 0 0 1       x x|     |      x    |     |x|  4
     * |        1 x x x  | ... |           | ... |x|  5 sc
     * |        0 1 x x  |     |           |     |x|  6
     * |        0 0 1 x  |     |           |     |x|  7
     * |      0 0 0 0 1 x|     |      x    |     |x|  8
     * |      0       0 1|     |      x    |     |x|  9
     * -------------------     -------------
     *          .                    .
     *          .                    .
     *          .                    .
     * -------------------     -------------
     * |                 |     |1 0        |     |x|  1
     * |                 |     |0 1 0 0 0  |     |x|  2
     * |                 |     |  0 1 0 0  |     |x|  3
     * |      0       0 0| ... |  0 0 1 0  | ... |x|  4 bb
     * |                 |     |  0 0 0 1 0|     |x|  5
     * |                 |     |        0 1|     |x|  6
     * -------------------     -------------
     *
     * Final matrix:
     *
     *         sc                   bb            g
     *  1 2 3 4 5 6 7 8 9       1 2 3 4 5 6
     * -------------------     -------------
     * |1 x x x          |     |           |     |x|  1
     * |0 1 x x          |     |           |     |x|  2
     * |0 0 1 x          |     |           |     |x|  3
     * |0 0 0 1       x x|     |      0    |     |M|  4
     * |        1 x x x  | ... |           | ... |x|  5 sc
     * |        0 1 x x  |     |           |     |x|  6
     * |        0 0 1 x  |     |           |     |x|  7
     * |      0 0 0 0 1 x|     |      0    |     |M|  8
     * |      0       0 1|     |      0    |     |M|  9
     * -------------------     -------------
     *          .                    .
     *          .                    .
     *          .                    .
     * -------------------     -------------
     * |                 |     |1 0        |     |x|  1
     * |                 |     |0 1 0 0 0  |     |x|  2
     * |                 |     |  0 1 0 0  |     |x|  3
     * |      0       0 0| ... |  0 0 1 0  | ... |x|  4 bb
     * |                 |     |  0 0 0 1 0|     |x|  5
     * |                 |     |        0 1|     |x|  6
     * -------------------     -------------
     *
     * @param sc_to_bb_matrix Pointer to a SC to BB valine sub-matrix.
     * @param sc_g_matrix Pointer to a SC g valine sub-matrix.
     * @param bb_g_matrix Pointer to a BB g valine sub-matrix.
     */
    void solve_16_valine(real *const sc_to_bb_matrix,
                         real *const sc_g_matrix,
                         __attribute__((unused)) real *const sc_to_bb_fillins,
                         real *const bb_g_matrix);

    /**
     * Performs the Gaussian elimination of each entry of the first SC to BB
     * sub-matrix pointed by SC_TO_BB_MATRIX.
     *
     * It is assumed that SC_TO_SC_MATRIX, SC_G_MATRIX and BB_G_MATRIX
     * point to the same leucine side-chain data.
     *
     * This function has been automatically generated by
     * step_2_3_16_17_generator.c.
     *
     * Initial matrix:
     *
     *            sc                      bb            g
     *  1 2 3 4 5 6 7 8 9101112       1 2 3 4 5 6
     * -------------------------     -------------
     * |1 x x x                |     |           |     |x|  1
     * |0 1 x x                |     |           |     |x|  2
     * |0 0 1 x                |     |           |     |x|  3
     * |0 0 0 1       x x x    |     |           |     |x|  4
     * |        1 x x x        |     |           |     |x|  5
     * |        0 1 x x        |     |           |     |x|  6
     * |        0 0 1 x        | ... |           | ... |x|  7 sc
     * |      0 0 0 0 1 x x    |     |           |     |x|  8
     * |      0       0 1 x    |     |           |     |x|  9
     * |      0       0 0 1 x x|     |      x    |     |x| 10
     * |                  0 1 x|     |      x    |     |x| 11
     * |                  0 0 1|     |      x    |     |x| 12
     * -------------------------     -------------
     *             .                       .
     *             .                       .
     *             .                       .
     * -------------------------     -------------
     * |                       |     |1 0        |     |x|  1
     * |                       |     |0 1 0 0 0  |     |x|  2
     * |                       |     |  0 1 0 0  |     |x|  3
     * |                  0 0 0| ... |  0 0 1 0  | ... |x|  4 bb
     * |                       |     |  0 0 0 1 0|     |x|  5
     * |                       |     |        0 1|     |x|  6
     * -------------------------     -------------
     *
     * Final matrix:
     *
     *            sc                      bb            g
     *  1 2 3 4 5 6 7 8 9101112       1 2 3 4 5 6
     * -------------------------     -------------
     * |1 x x x                |     |           |     |x|  1
     * |0 1 x x                |     |           |     |x|  2
     * |0 0 1 x                |     |           |     |x|  3
     * |0 0 0 1       x x x    |     |           |     |x|  4
     * |        1 x x x        |     |           |     |x|  5
     * |        0 1 x x        |     |           |     |x|  6
     * |        0 0 1 x        | ... |           | ... |x|  7 sc
     * |      0 0 0 0 1 x x    |     |           |     |x|  8
     * |      0       0 1 x    |     |           |     |x|  9
     * |      0       0 0 1 x x|     |      0    |     |M| 10
     * |                  0 1 x|     |      0    |     |M| 11
     * |                  0 0 1|     |      0    |     |M| 12
     * -------------------------     -------------
     *             .                       .
     *             .                       .
     *             .                       .
     * -------------------------     -------------
     * |                       |     |1 0        |     |x|  1
     * |                       |     |0 1 0 0 0  |     |x|  2
     * |                       |     |  0 1 0 0  |     |x|  3
     * |                  0 0 0| ... |  0 0 1 0  | ... |x|  4 bb
     * |                       |     |  0 0 0 1 0|     |x|  5
     * |                       |     |        0 1|     |x|  6
     * -------------------------     -------------
     *
     * @param sc_to_bb_matrix Pointer to a SC to BB leucine sub-matrix.
     * @param sc_g_matrix Pointer to a SC g leucine sub-matrix.
     * @param bb_g_matrix Pointer to a BB g leucine sub-matrix.
     */
    void solve_16_leucine(real *const sc_to_bb_matrix,
                          real *const sc_g_matrix,
                          __attribute__((unused)) real *const sc_to_bb_fillins,
                          real *const bb_g_matrix);

    /**
     * Performs the Gaussian elimination of each entry of the first SC to BB
     * sub-matrix pointed by SC_TO_BB_MATRIX.
     *
     * It is assumed that SC_TO_SC_MATRIX, SC_G_MATRIX and BB_G_MATRIX
     * point to the same isoleucine side-chain data.
     *
     * This function has been automatically generated by
     * step_2_3_16_17_generator.c.
     *
     * Initial matrix:
     *
     *            sc                      bb            g
     *  1 2 3 4 5 6 7 8 9101112       1 2 3 4 5 6
     * -------------------------     -------------
     * |1 x x x                |     |           |     |x|  1
     * |0 1 x x                |     |           |     |x|  2
     * |0 0 1 x                |     |           |     |x|  3
     * |0 0 0 1 x x x          |     |           |     |x|  4
     * |      0 1 x x          |     |           |     |x|  5
     * |      0 0 1 x          |     |           |     |x|  6
     * |      0 0 0 1       x x| ... |      x    | ... |x|  7 sc
     * |              1 x x x  |     |           |     |x|  8
     * |              0 1 x x  |     |           |     |x|  9
     * |              0 0 1 x  |     |           |     |x| 10
     * |            0 0 0 0 1 x|     |      x    |     |x| 11
     * |            0       0 1|     |      x    |     |x| 12
     * -------------------------     -------------
     *             .                       .
     *             .                       .
     *             .                       .
     * -------------------------     -------------
     * |                       |     |1 0        |     |x|  1
     * |                       |     |0 1 0 0 0  |     |x|  2
     * |                       |     |  0 1 0 0  |     |x|  3
     * |            0       0 0| ... |  0 0 1 0  | ... |x|  4 bb
     * |                       |     |  0 0 0 1 0|     |x|  5
     * |                       |     |        0 1|     |x|  6
     * -------------------------     -------------
     *
     * Final matrix:
     *
     *            sc                      bb            g
     *  1 2 3 4 5 6 7 8 9101112       1 2 3 4 5 6
     * -------------------------     -------------
     * |1 x x x                |     |           |     |x|  1
     * |0 1 x x                |     |           |     |x|  2
     * |0 0 1 x                |     |           |     |x|  3
     * |0 0 0 1 x x x          |     |           |     |x|  4
     * |      0 1 x x          |     |           |     |x|  5
     * |      0 0 1 x          |     |           |     |x|  6
     * |      0 0 0 1       x x| ... |      0    | ... |M|  7 sc
     * |              1 x x x  |     |           |     |x|  8
     * |              0 1 x x  |     |           |     |x|  9
     * |              0 0 1 x  |     |           |     |x| 10
     * |            0 0 0 0 1 x|     |      0    |     |M| 11
     * |            0       0 1|     |      0    |     |M| 12
     * -------------------------     -------------
     *             .                       .
     *             .                       .
     *             .                       .
     * -------------------------     -------------
     * |                       |     |1 0        |     |x|  1
     * |                       |     |0 1 0 0 0  |     |x|  2
     * |                       |     |  0 1 0 0  |     |x|  3
     * |            0       0 0| ... |  0 0 1 0  | ... |x|  4 bb
     * |                       |     |  0 0 0 1 0|     |x|  5
     * |                       |     |        0 1|     |x|  6
     * -------------------------     -------------
     *
     * @param sc_to_bb_matrix Pointer to a SC to BB isoleucine sub-matrix.
     * @param sc_g_matrix Pointer to a SC g isoleucine sub-matrix.
     * @param bb_g_matrix Pointer to a BB g isoleucine sub-matrix.
     */
    void solve_16_isoleucine(real *const sc_to_bb_matrix,
                             real *const sc_g_matrix,
                             __attribute__((unused)) real *const sc_to_bb_fillins,
                             real *const bb_g_matrix);

    /**
     * Performs the Gaussian elimination of each entry of the first SC to BB
     * sub-matrix pointed by SC_TO_BB_MATRIX.
     *
     * It is assumed that SC_TO_SC_MATRIX, SC_G_MATRIX and BB_G_MATRIX
     * point to the same aspartic_acid side-chain data.
     *
     * This function has been automatically generated by
     * step_2_3_16_17_generator.c.
     *
     * Initial matrix:
     *
     *     sc               bb            g
     *  1 2 3 4 5       1 2 3 4 5 6
     * -----------     -------------
     * |1 x x    |     |           |     |x|  1
     * |0 1 x    |     |           |     |x|  2
     * |0 0 1 x x| ... |      x    | ... |x|  3 sc
     * |    0 1 x|     |      x    |     |x|  4
     * |    0 0 1|     |      x    |     |x|  5
     * -----------     -------------
     *      .                .
     *      .                .
     *      .                .
     * -----------     -------------
     * |         |     |1 0        |     |x|  1
     * |         |     |0 1 0 0 0  |     |x|  2
     * |         |     |  0 1 0 0  |     |x|  3
     * |    0 0 0| ... |  0 0 1 0  | ... |x|  4 bb
     * |         |     |  0 0 0 1 0|     |x|  5
     * |         |     |        0 1|     |x|  6
     * -----------     -------------
     *
     * Final matrix:
     *
     *     sc               bb            g
     *  1 2 3 4 5       1 2 3 4 5 6
     * -----------     -------------
     * |1 x x    |     |           |     |x|  1
     * |0 1 x    |     |           |     |x|  2
     * |0 0 1 x x| ... |      0    | ... |M|  3 sc
     * |    0 1 x|     |      0    |     |M|  4
     * |    0 0 1|     |      0    |     |M|  5
     * -----------     -------------
     *      .                .
     *      .                .
     *      .                .
     * -----------     -------------
     * |         |     |1 0        |     |x|  1
     * |         |     |0 1 0 0 0  |     |x|  2
     * |         |     |  0 1 0 0  |     |x|  3
     * |    0 0 0| ... |  0 0 1 0  | ... |x|  4 bb
     * |         |     |  0 0 0 1 0|     |x|  5
     * |         |     |        0 1|     |x|  6
     * -----------     -------------
     *
     * @param sc_to_bb_matrix Pointer to a SC to BB aspartic_acid sub-matrix.
     * @param sc_g_matrix Pointer to a SC g aspartic_acid sub-matrix.
     * @param bb_g_matrix Pointer to a BB g aspartic_acid sub-matrix.
     */
    void solve_16_aspartic_acid(real *const sc_to_bb_matrix,
                                real *const sc_g_matrix,
                                __attribute__((unused))
                                real *const sc_to_bb_fillins,
                                real *const bb_g_matrix);

    /**
     * Performs the Gaussian elimination of each entry of the first SC to BB
     * sub-matrix pointed by SC_TO_BB_MATRIX.
     *
     * It is assumed that SC_TO_SC_MATRIX, SC_G_MATRIX and BB_G_MATRIX
     * point to the same glutamic_acid side-chain data.
     *
     * This function has been automatically generated by
     * step_2_3_16_17_generator.c.
     *
     * Initial matrix:
     *
     *        sc                  bb            g
     *  1 2 3 4 5 6 7 8       1 2 3 4 5 6
     * -----------------     -------------
     * |1 x x          |     |           |     |x|  1
     * |0 1 x          |     |           |     |x|  2
     * |0 0 1 x x x    |     |           |     |x|  3
     * |    0 1 x x    |     |           |     |x|  4
     * |    0 0 1 x    | ... |           | ... |x|  5 sc
     * |    0 0 0 1 x x|     |      x    |     |x|  6
     * |          0 1 x|     |      x    |     |x|  7
     * |          0 0 1|     |      x    |     |x|  8
     * -----------------     -------------
     *         .                   .
     *         .                   .
     *         .                   .
     * -----------------     -------------
     * |               |     |1 0        |     |x|  1
     * |               |     |0 1 0 0 0  |     |x|  2
     * |               |     |  0 1 0 0  |     |x|  3
     * |          0 0 0| ... |  0 0 1 0  | ... |x|  4 bb
     * |               |     |  0 0 0 1 0|     |x|  5
     * |               |     |        0 1|     |x|  6
     * -----------------     -------------
     *
     * Final matrix:
     *
     *        sc                  bb            g
     *  1 2 3 4 5 6 7 8       1 2 3 4 5 6
     * -----------------     -------------
     * |1 x x          |     |           |     |x|  1
     * |0 1 x          |     |           |     |x|  2
     * |0 0 1 x x x    |     |           |     |x|  3
     * |    0 1 x x    |     |           |     |x|  4
     * |    0 0 1 x    | ... |           | ... |x|  5 sc
     * |    0 0 0 1 x x|     |      0    |     |M|  6
     * |          0 1 x|     |      0    |     |M|  7
     * |          0 0 1|     |      0    |     |M|  8
     * -----------------     -------------
     *         .                   .
     *         .                   .
     *         .                   .
     * -----------------     -------------
     * |               |     |1 0        |     |x|  1
     * |               |     |0 1 0 0 0  |     |x|  2
     * |               |     |  0 1 0 0  |     |x|  3
     * |          0 0 0| ... |  0 0 1 0  | ... |x|  4 bb
     * |               |     |  0 0 0 1 0|     |x|  5
     * |               |     |        0 1|     |x|  6
     * -----------------     -------------
     *
     * @param sc_to_bb_matrix Pointer to a SC to BB glutamic_acid sub-matrix.
     * @param sc_g_matrix Pointer to a SC g glutamic_acid sub-matrix.
     * @param bb_g_matrix Pointer to a BB g glutamic_acid sub-matrix.
     */
    void solve_16_glutamic_acid(real *const sc_to_bb_matrix,
                                real *const sc_g_matrix,
                                __attribute__((unused))
                                real *const sc_to_bb_fillins,
                                real *const bb_g_matrix);

    /**
     * Performs the Gaussian elimination of each entry of the first SC to BB
     * sub-matrix pointed by SC_TO_BB_MATRIX.
     *
     * It is assumed that SC_TO_SC_MATRIX, SC_G_MATRIX and BB_G_MATRIX
     * point to the same asparagine side-chain data.
     *
     * This function has been automatically generated by
     * step_2_3_16_17_generator.c.
     *
     * Initial matrix:
     *
     *       sc                 bb            g
     *  1 2 3 4 5 6 7       1 2 3 4 5 6
     * ---------------     -------------
     * |1 x x        |     |           |     |x|  1
     * |0 1 x        |     |           |     |x|  2
     * |0 0 1 x x    |     |           |     |x|  3
     * |    0 1 x    | ... |           | ... |x|  4 sc
     * |    0 0 1 x x|     |      x    |     |x|  5
     * |        0 1 x|     |      x    |     |x|  6
     * |        0 0 1|     |      x    |     |x|  7
     * ---------------     -------------
     *        .                  .
     *        .                  .
     *        .                  .
     * ---------------     -------------
     * |             |     |1 0        |     |x|  1
     * |             |     |0 1 0 0 0  |     |x|  2
     * |             |     |  0 1 0 0  |     |x|  3
     * |        0 0 0| ... |  0 0 1 0  | ... |x|  4 bb
     * |             |     |  0 0 0 1 0|     |x|  5
     * |             |     |        0 1|     |x|  6
     * ---------------     -------------
     *
     * Final matrix:
     *
     *       sc                 bb            g
     *  1 2 3 4 5 6 7       1 2 3 4 5 6
     * ---------------     -------------
     * |1 x x        |     |           |     |x|  1
     * |0 1 x        |     |           |     |x|  2
     * |0 0 1 x x    |     |           |     |x|  3
     * |    0 1 x    | ... |           | ... |x|  4 sc
     * |    0 0 1 x x|     |      0    |     |M|  5
     * |        0 1 x|     |      0    |     |M|  6
     * |        0 0 1|     |      0    |     |M|  7
     * ---------------     -------------
     *        .                  .
     *        .                  .
     *        .                  .
     * ---------------     -------------
     * |             |     |1 0        |     |x|  1
     * |             |     |0 1 0 0 0  |     |x|  2
     * |             |     |  0 1 0 0  |     |x|  3
     * |        0 0 0| ... |  0 0 1 0  | ... |x|  4 bb
     * |             |     |  0 0 0 1 0|     |x|  5
     * |             |     |        0 1|     |x|  6
     * ---------------     -------------
     *
     * @param sc_to_bb_matrix Pointer to a SC to BB asparagine sub-matrix.
     * @param sc_g_matrix Pointer to a SC g asparagine sub-matrix.
     * @param bb_g_matrix Pointer to a BB g asparagine sub-matrix.
     */
    void solve_16_asparagine(real *const sc_to_bb_matrix,
                             real *const sc_g_matrix,
                             __attribute__((unused)) real *const sc_to_bb_fillins,
                             real *const bb_g_matrix);

    /**
     * Performs the Gaussian elimination of each entry of the first SC to BB
     * sub-matrix pointed by SC_TO_BB_MATRIX.
     *
     * It is assumed that SC_TO_SC_MATRIX, SC_G_MATRIX and BB_G_MATRIX
     * point to the same glutamine side-chain data.
     *
     * This function has been automatically generated by
     * step_2_3_16_17_generator.c.
     *
     * Initial matrix:
     *
     *          sc                    bb            g
     *  1 2 3 4 5 6 7 8 910       1 2 3 4 5 6
     * ---------------------     -------------
     * |1 x x              |     |           |     |x|  1
     * |0 1 x              |     |           |     |x|  2
     * |0 0 1 x x          |     |           |     |x|  3
     * |    0 1 x          |     |           |     |x|  4
     * |    0 0 1 x x x    |     |           |     |x|  5
     * |        0 1 x x    | ... |           | ... |x|  6 sc
     * |        0 0 1 x    |     |           |     |x|  7
     * |        0 0 0 1 x x|     |      x    |     |x|  8
     * |              0 1 x|     |      x    |     |x|  9
     * |              0 0 1|     |      x    |     |x| 10
     * ---------------------     -------------
     *           .                     .
     *           .                     .
     *           .                     .
     * ---------------------     -------------
     * |                   |     |1 0        |     |x|  1
     * |                   |     |0 1 0 0 0  |     |x|  2
     * |                   |     |  0 1 0 0  |     |x|  3
     * |              0 0 0| ... |  0 0 1 0  | ... |x|  4 bb
     * |                   |     |  0 0 0 1 0|     |x|  5
     * |                   |     |        0 1|     |x|  6
     * ---------------------     -------------
     *
     * Final matrix:
     *
     *          sc                    bb            g
     *  1 2 3 4 5 6 7 8 910       1 2 3 4 5 6
     * ---------------------     -------------
     * |1 x x              |     |           |     |x|  1
     * |0 1 x              |     |           |     |x|  2
     * |0 0 1 x x          |     |           |     |x|  3
     * |    0 1 x          |     |           |     |x|  4
     * |    0 0 1 x x x    |     |           |     |x|  5
     * |        0 1 x x    | ... |           | ... |x|  6 sc
     * |        0 0 1 x    |     |           |     |x|  7
     * |        0 0 0 1 x x|     |      0    |     |M|  8
     * |              0 1 x|     |      0    |     |M|  9
     * |              0 0 1|     |      0    |     |M| 10
     * ---------------------     -------------
     *           .                     .
     *           .                     .
     *           .                     .
     * ---------------------     -------------
     * |                   |     |1 0        |     |x|  1
     * |                   |     |0 1 0 0 0  |     |x|  2
     * |                   |     |  0 1 0 0  |     |x|  3
     * |              0 0 0| ... |  0 0 1 0  | ... |x|  4 bb
     * |                   |     |  0 0 0 1 0|     |x|  5
     * |                   |     |        0 1|     |x|  6
     * ---------------------     -------------
     *
     * @param sc_to_bb_matrix Pointer to a SC to BB glutamine sub-matrix.
     * @param sc_g_matrix Pointer to a SC g glutamine sub-matrix.
     * @param bb_g_matrix Pointer to a BB g glutamine sub-matrix.
     */
    void solve_16_glutamine(real *const sc_to_bb_matrix,
                            real *const sc_g_matrix,
                            __attribute__((unused)) real *const sc_to_bb_fillins,
                            real *const bb_g_matrix);

    /**
     * Performs the Gaussian elimination of each entry of the first SC to BB
     * sub-matrix pointed by SC_TO_BB_MATRIX.
     *
     * It is assumed that SC_TO_SC_MATRIX, SC_G_MATRIX and BB_G_MATRIX
     * point to the same histidine side-chain data.
     *
     * This function has been automatically generated by
     * step_2_3_16_17_generator.c.
     *
     * Initial matrix:
     *
     *           sc                     bb            g
     *  1 2 3 4 5 6 7 8 91011       1 2 3 4 5 6
     * -----------------------     -------------
     * |1   x   x            |     |           |     |x|  1
     * |  1 x     x          |     |           |     |x|  2
     * |0 0 1   x x          |     |           |     |x|  3
     * |      1 x     x      |     |           |     |x|  4
     * |0   0 0 1 f   x      |     |           |     |x|  5
     * |  0 0   0 1 x f      | ... |           | ... |x|  6 sc
     * |          0 1 x x    |     |           |     |x|  7
     * |      0 0 0 0 1 x    |     |           |     |x|  8
     * |            0 0 1 x x|     |      x    |     |x|  9
     * |                0 1 x|     |      x    |     |x| 10
     * |                0 0 1|     |      x    |     |x| 11
     * -----------------------     -------------
     *            .                      .
     *            .                      .
     *            .                      .
     * -----------------------     -------------
     * |                     |     |1 0        |     |x|  1
     * |                     |     |0 1 0 0 0  |     |x|  2
     * |                     |     |  0 1 0 0  |     |x|  3
     * |                0 0 0| ... |  0 0 1 0  | ... |x|  4 bb
     * |                     |     |  0 0 0 1 0|     |x|  5
     * |                     |     |        0 1|     |x|  6
     * -----------------------     -------------
     *
     * Final matrix:
     *
     *           sc                     bb            g
     *  1 2 3 4 5 6 7 8 91011       1 2 3 4 5 6
     * -----------------------     -------------
     * |1   x   x            |     |           |     |x|  1
     * |  1 x     x          |     |           |     |x|  2
     * |0 0 1   x x          |     |           |     |x|  3
     * |      1 x     x      |     |           |     |x|  4
     * |0   0 0 1 f   x      |     |           |     |x|  5
     * |  0 0   0 1 x f      | ... |           | ... |x|  6 sc
     * |          0 1 x x    |     |           |     |x|  7
     * |      0 0 0 0 1 x    |     |           |     |x|  8
     * |            0 0 1 x x|     |      0    |     |M|  9
     * |                0 1 x|     |      0    |     |M| 10
     * |                0 0 1|     |      0    |     |M| 11
     * -----------------------     -------------
     *            .                      .
     *            .                      .
     *            .                      .
     * -----------------------     -------------
     * |                     |     |1 0        |     |x|  1
     * |                     |     |0 1 0 0 0  |     |x|  2
     * |                     |     |  0 1 0 0  |     |x|  3
     * |                0 0 0| ... |  0 0 1 0  | ... |x|  4 bb
     * |                     |     |  0 0 0 1 0|     |x|  5
     * |                     |     |        0 1|     |x|  6
     * -----------------------     -------------
     *
     * @param sc_to_bb_matrix Pointer to a SC to BB histidine sub-matrix.
     * @param sc_g_matrix Pointer to a SC g histidine sub-matrix.
     * @param bb_g_matrix Pointer to a BB g histidine sub-matrix.
     */
    void solve_16_histidine(real *const sc_to_bb_matrix,
                            real *const sc_g_matrix,
                            __attribute__((unused)) real *const sc_to_bb_fillins,
                            real *const bb_g_matrix);

    /**
     * Performs the Gaussian elimination of each entry of the first SC to BB
     * sub-matrix pointed by SC_TO_BB_MATRIX.
     *
     * It is assumed that SC_TO_SC_MATRIX, SC_G_MATRIX and BB_G_MATRIX
     * point to the same lysine side-chain data.
     *
     * This function has been automatically generated by
     * step_2_3_16_17_generator.c.
     *
     * Initial matrix:
     *
     *               sc                         bb            g
     *  1 2 3 4 5 6 7 8 9101112131415       1 2 3 4 5 6
     * -------------------------------     -------------
     * |1 x x x                      |     |           |     |x|  1
     * |0 1 x x                      |     |           |     |x|  2
     * |0 0 1 x                      |     |           |     |x|  3
     * |0 0 0 1 x x x                |     |           |     |x|  4
     * |      0 1 x x                |     |           |     |x|  5
     * |      0 0 1 x                |     |           |     |x|  6
     * |      0 0 0 1 x x x          |     |           |     |x|  7
     * |            0 1 x x          | ... |           | ... |x|  8 sc
     * |            0 0 1 x          |     |           |     |x|  9
     * |            0 0 0 1 x x x    |     |           |     |x| 10
     * |                  0 1 x x    |     |           |     |x| 11
     * |                  0 0 1 x    |     |           |     |x| 12
     * |                  0 0 0 1 x x|     |      x    |     |x| 13
     * |                        0 1 x|     |      x    |     |x| 14
     * |                        0 0 1|     |      x    |     |x| 15
     * -------------------------------     -------------
     *                .                          .
     *                .                          .
     *                .                          .
     * -------------------------------     -------------
     * |                             |     |1 0        |     |x|  1
     * |                             |     |0 1 0 0 0  |     |x|  2
     * |                             |     |  0 1 0 0  |     |x|  3
     * |                        0 0 0| ... |  0 0 1 0  | ... |x|  4 bb
     * |                             |     |  0 0 0 1 0|     |x|  5
     * |                             |     |        0 1|     |x|  6
     * -------------------------------     -------------
     *
     * Final matrix:
     *
     *               sc                         bb            g
     *  1 2 3 4 5 6 7 8 9101112131415       1 2 3 4 5 6
     * -------------------------------     -------------
     * |1 x x x                      |     |           |     |x|  1
     * |0 1 x x                      |     |           |     |x|  2
     * |0 0 1 x                      |     |           |     |x|  3
     * |0 0 0 1 x x x                |     |           |     |x|  4
     * |      0 1 x x                |     |           |     |x|  5
     * |      0 0 1 x                |     |           |     |x|  6
     * |      0 0 0 1 x x x          |     |           |     |x|  7
     * |            0 1 x x          | ... |           | ... |x|  8 sc
     * |            0 0 1 x          |     |           |     |x|  9
     * |            0 0 0 1 x x x    |     |           |     |x| 10
     * |                  0 1 x x    |     |           |     |x| 11
     * |                  0 0 1 x    |     |           |     |x| 12
     * |                  0 0 0 1 x x|     |      0    |     |M| 13
     * |                        0 1 x|     |      0    |     |M| 14
     * |                        0 0 1|     |      0    |     |M| 15
     * -------------------------------     -------------
     *                .                          .
     *                .                          .
     *                .                          .
     * -------------------------------     -------------
     * |                             |     |1 0        |     |x|  1
     * |                             |     |0 1 0 0 0  |     |x|  2
     * |                             |     |  0 1 0 0  |     |x|  3
     * |                        0 0 0| ... |  0 0 1 0  | ... |x|  4 bb
     * |                             |     |  0 0 0 1 0|     |x|  5
     * |                             |     |        0 1|     |x|  6
     * -------------------------------     -------------
     *
     * @param sc_to_bb_matrix Pointer to a SC to BB lysine sub-matrix.
     * @param sc_g_matrix Pointer to a SC g lysine sub-matrix.
     * @param bb_g_matrix Pointer to a BB g lysine sub-matrix.
     */
    void solve_16_lysine(real *const sc_to_bb_matrix,
                         real *const sc_g_matrix,
                         __attribute__((unused)) real *const sc_to_bb_fillins,
                         real *const bb_g_matrix);

    /**
     * Performs the Gaussian elimination of each entry of the first SC to BB
     * sub-matrix pointed by SC_TO_BB_MATRIX.
     *
     * It is assumed that SC_TO_SC_MATRIX, SC_G_MATRIX and BB_G_MATRIX
     * point to the same arginine side-chain data.
     *
     * This function has been automatically generated by
     * step_2_3_16_17_generator.c.
     *
     * Initial matrix:
     *
     *                sc                          bb            g
     *  1 2 3 4 5 6 7 8 910111213141516       1 2 3 4 5 6
     * ---------------------------------     -------------
     * |1 x x                          |     |           |     |x|  1
     * |0 1 x                          |     |           |     |x|  2
     * |0 0 1     x x                  |     |           |     |x|  3
     * |      1 x x                    |     |           |     |x|  4
     * |      0 1 x                    |     |           |     |x|  5
     * |    0 0 0 1 x                  |     |           |     |x|  6
     * |    0     0 1 x                |     |           |     |x|  7
     * |            0 1 x x x          |     |           |     |x|  8
     * |              0 1 x x          | ... |           | ... |x|  9 sc
     * |              0 0 1 x          |     |           |     |x| 10
     * |              0 0 0 1 x x x    |     |           |     |x| 11
     * |                    0 1 x x    |     |           |     |x| 12
     * |                    0 0 1 x    |     |           |     |x| 13
     * |                    0 0 0 1 x x|     |      x    |     |x| 14
     * |                          0 1 x|     |      x    |     |x| 15
     * |                          0 0 1|     |      x    |     |x| 16
     * ---------------------------------     -------------
     *                 .                           .
     *                 .                           .
     *                 .                           .
     * ---------------------------------     -------------
     * |                               |     |1 0        |     |x|  1
     * |                               |     |0 1 0 0 0  |     |x|  2
     * |                               |     |  0 1 0 0  |     |x|  3
     * |                          0 0 0| ... |  0 0 1 0  | ... |x|  4 bb
     * |                               |     |  0 0 0 1 0|     |x|  5
     * |                               |     |        0 1|     |x|  6
     * ---------------------------------     -------------
     *
     * Final matrix:
     *
     *                 sc                           bb            g
     *  1 2 3 4 5 6 7 8 91011121314151617       1 2 3 4 5 6
     * -----------------------------------     -------------
     * |1 x x                            |     |           |     |x|  1
     * |0 1 x                            |     |           |     |x|  2
     * |0 0 1     x x                    |     |           |     |x|  3
     * |      1 x x                      |     |           |     |x|  4
     * |      0 1 x                      |     |           |     |x|  5
     * |    0 0 0 1 x                    |     |           |     |x|  6
     * |    0     0 1 x x                |     |           |     |x|  7
     * |            0 1 x                |     |           |     |x|  8
     * |            0 0 1 x x x          | ... |           | ... |x|  9 sc
     * |                0 1 x x          |     |           |     |x| 10
     * |                0 0 1 x          |     |           |     |x| 11
     * |                0 0 0 1 x x x    |     |           |     |x| 12
     * |                      0 1 x x    |     |           |     |x| 13
     * |                      0 0 1 x    |     |           |     |x| 14
     * |                      0 0 0 1 x x|     |      x    |     |x| 15
     * |                            0 1 x|     |      x    |     |x| 16
     * |                            0 0 1|     |      x    |     |x| 17
     * -----------------------------------     -------------
     *                  .                            .
     *                  .                            .
     *                  .                            .
     * -----------------------------------     -------------
     * |                                 |     |1 0        |     |x|  1
     * |                                 |     |0 1 0 0 0  |     |x|  2
     * |                                 |     |  0 1 0 0  |     |x|  3
     * |                            0 0 0| ... |  0 0 1 0  | ... |x|  4 bb
     * |                                 |     |  0 0 0 1 0|     |x|  5
     * |                                 |     |        0 1|     |x|  6
     * -----------------------------------     -------------
     *
     * Final matrix:
     *
     *                 sc                           bb            g
     *  1 2 3 4 5 6 7 8 91011121314151617       1 2 3 4 5 6
     * -----------------------------------     -------------
     * |1 x x                            |     |           |     |x|  1
     * |0 1 x                            |     |           |     |x|  2
     * |0 0 1     x x                    |     |           |     |x|  3
     * |      1 x x                      |     |           |     |x|  4
     * |      0 1 x                      |     |           |     |x|  5
     * |    0 0 0 1 x                    |     |           |     |x|  6
     * |    0     0 1 x x                |     |           |     |x|  7
     * |            0 1 x                |     |           |     |x|  8
     * |            0 0 1 x x x          | ... |           | ... |x|  9 sc
     * |                0 1 x x          |     |           |     |x| 10
     * |                0 0 1 x          |     |           |     |x| 11
     * |                0 0 0 1 x x x    |     |           |     |x| 12
     * |                      0 1 x x    |     |           |     |x| 13
     * |                      0 0 1 x    |     |           |     |x| 14
     * |                      0 0 0 1 x x|     |      0    |     |M| 15
     * |                            0 1 x|     |      0    |     |M| 16
     * |                            0 0 1|     |      0    |     |M| 17
     * -----------------------------------     -------------
     *
     * @param sc_to_bb_matrix Pointer to a SC to BB arginine sub-matrix.
     * @param sc_g_matrix Pointer to a SC g arginine sub-matrix.
     * @param bb_g_matrix Pointer to a BB g arginine sub-matrix.
     */
    void solve_16_arginine(real *const sc_to_bb_matrix,
                           real *const sc_g_matrix,
                           __attribute__((unused)) real *const sc_to_bb_fillins,
                           real *const bb_g_matrix);

    /**
     * Performs the Gaussian elimination of each entry of the first SC to BB
     * sub-matrix pointed by SC_TO_BB_MATRIX.
     *
     * It is assumed that SC_TO_SC_MATRIX, SC_G_MATRIX and BB_G_MATRIX
     * point to the same serine side-chain data.
     *
     * This function has been automatically generated by
     * step_2_3_16_17_generator.c.
     *
     * Initial matrix:
     *
     *    sc              bb            g
     *  1 2 3 4       1 2 3 4 5 6
     * ---------     -------------
     * |1 x    |     |           |     |x|  1
     * |0 1 x x|     |      x    |     |x|  2
     * |  0 1 x| ... |      x    | ... |x|  3 sc
     * |  0 0 1|     |      x    |     |x|  4
     * ---------     -------------
     *     .               .
     *     .               .
     *     .               .
     * ---------     -------------
     * |       |     |1 0        |     |x|  1
     * |       |     |0 1 0 0 0  |     |x|  2
     * |       |     |  0 1 0 0  |     |x|  3
     * |  0 0 0| ... |  0 0 1 0  | ... |x|  4 bb
     * |       |     |  0 0 0 1 0|     |x|  5
     * |       |     |        0 1|     |x|  6
     * ---------     -------------
     *
     * Final matrix:
     *
     *    sc              bb            g
     *  1 2 3 4       1 2 3 4 5 6
     * ---------     -------------
     * |1 x    |     |           |     |x|  1
     * |0 1 x x|     |      0    |     |M|  2
     * |  0 1 x| ... |      0    | ... |M|  3 sc
     * |  0 0 1|     |      0    |     |M|  4
     * ---------     -------------
     *     .               .
     *     .               .
     *     .               .
     * ---------     -------------
     * |       |     |1 0        |     |x|  1
     * |       |     |0 1 0 0 0  |     |x|  2
     * |       |     |  0 1 0 0  |     |x|  3
     * |  0 0 0| ... |  0 0 1 0  | ... |x|  4 bb
     * |       |     |  0 0 0 1 0|     |x|  5
     * |       |     |        0 1|     |x|  6
     * ---------     -------------
     *
     * @param sc_to_bb_matrix Pointer to a SC to BB serine sub-matrix.
     * @param sc_g_matrix Pointer to a SC g serine sub-matrix.
     * @param bb_g_matrix Pointer to a BB g serine sub-matrix.
     */
    void solve_16_serine(real *const sc_to_bb_matrix,
                         real *const sc_g_matrix,
                         __attribute__((unused)) real *const sc_to_bb_fillins,
                         real *const bb_g_matrix);

    /**
     * Performs the Gaussian elimination of each entry of the first SC to BB
     * sub-matrix pointed by SC_TO_BB_MATRIX.
     *
     * It is assumed that SC_TO_SC_MATRIX, SC_G_MATRIX and BB_G_MATRIX
     * point to the same phenylaline side-chain data.
     *
     * This function has been automatically generated by
     * step_2_3_16_17_generator.c.
     *
     * Initial matrix:
     *
     *              sc                        bb            g
     *  1 2 3 4 5 6 7 8 91011121314       1 2 3 4 5 6
     * -----------------------------     -------------
     * |1   x             x        |     |           |     |x|  1
     * |  1 x   x                  |     |           |     |x|  2
     * |0 0 1   x         x        |     |           |     |x|  3
     * |      1 x   x              |     |           |     |x|  4
     * |  0 0 0 1   x     f        |     |           |     |x|  5
     * |          1 x   x          |     |           |     |x|  6
     * |      0 0 0 1   x f        |     |           |     |x|  7
     * |              1 x   x      | ... |           | ... |x|  8 sc
     * |          0 0 0 1 f x      |     |           |     |x|  9
     * |0   0   0   0   0 1 x x    |     |           |     |x| 10
     * |              0 0 0 1 x    |     |           |     |x| 11
     * |                  0 0 1 x x|     |      x    |     |x| 12
     * |                      0 1 x|     |      x    |     |x| 13
     * |                      0 0 1|     |      x    |     |x| 14
     * -----------------------------     -------------
     *               .                         .
     *               .                         .
     *               .                         .
     * -----------------------------     -------------
     * |                           |     |1 0        |     |x|  1
     * |                           |     |0 1 0 0 0  |     |x|  2
     * |                           |     |  0 1 0 0  |     |x|  3
     * |                      0 0 0| ... |  0 0 1 0  | ... |x|  4 bb
     * |                           |     |  0 0 0 1 0|     |x|  5
     * |                           |     |        0 1|     |x|  6
     * -----------------------------     -------------
     *
     * Final matrix:
     *
     *              sc                        bb            g
     *  1 2 3 4 5 6 7 8 91011121314       1 2 3 4 5 6
     * -----------------------------     -------------
     * |1   x             x        |     |           |     |x|  1
     * |  1 x   x                  |     |           |     |x|  2
     * |0 0 1   x         x        |     |           |     |x|  3
     * |      1 x   x              |     |           |     |x|  4
     * |  0 0 0 1   x     f        |     |           |     |x|  5
     * |          1 x   x          |     |           |     |x|  6
     * |      0 0 0 1   x f        |     |           |     |x|  7
     * |              1 x   x      | ... |           | ... |x|  8 sc
     * |          0 0 0 1 f x      |     |           |     |x|  9
     * |0   0   0   0   0 1 x x    |     |           |     |x| 10
     * |              0 0 0 1 x    |     |           |     |x| 11
     * |                  0 0 1 x x|     |      0    |     |M| 12
     * |                      0 1 x|     |      0    |     |M| 13
     * |                      0 0 1|     |      0    |     |M| 14
     * -----------------------------     -------------
     *               .                         .
     *               .                         .
     *               .                         .
     * -----------------------------     -------------
     * |                           |     |1 0        |     |x|  1
     * |                           |     |0 1 0 0 0  |     |x|  2
     * |                           |     |  0 1 0 0  |     |x|  3
     * |                      0 0 0| ... |  0 0 1 0  | ... |x|  4 bb
     * |                           |     |  0 0 0 1 0|     |x|  5
     * |                           |     |        0 1|     |x|  6
     * -----------------------------     -------------
     *
     * @param sc_to_bb_matrix Pointer to a SC to BB phenylaline sub-matrix.
     * @param sc_g_matrix Pointer to a SC g phenylaline sub-matrix.
     * @param bb_g_matrix Pointer to a BB g phenylaline sub-matrix.
     */
    void solve_16_phenylaline(real *const sc_to_bb_matrix,
                              real *const sc_g_matrix,
                              __attribute__((unused)) real *const sc_to_bb_fillins,
                              real *const bb_g_matrix);

    /**
     * Performs the Gaussian elimination of each entry of the first SC to BB
     * sub-matrix pointed by SC_TO_BB_MATRIX.
     *
     * It is assumed that SC_TO_SC_MATRIX, SC_G_MATRIX and BB_G_MATRIX
     * point to the same tyrosine side-chain data.
     *
     * This function has been automatically generated by
     * step_2_3_16_17_generator.c.
     *
     * Initial matrix:
     *
     *               sc                         bb            g
     *  1 2 3 4 5 6 7 8 9101112131415       1 2 3 4 5 6
     * -------------------------------     -------------
     * |1       x                    |     |           |     |x|  1
     * |  1   x             x        |     |           |     |x|  2
     * |    1 x   x                  |     |           |     |x|  3
     * |  0 0 1   x         x        |     |           |     |x|  4
     * |0       1 x   x              |     |           |     |x|  5
     * |    0 0 0 1   x     f        |     |           |     |x|  6
     * |            1 x   x          |     |           |     |x|  7
     * |        0 0 0 1   x f        | ... |           | ... |x|  8 sc
     * |                1 x   x      |     |           |     |x|  9
     * |            0 0 0 1 f x      |     |           |     |x| 10
     * |  0   0   0   0   0 1 x x    |     |           |     |x| 11
     * |                0 0 0 1 x    |     |           |     |x| 12
     * |                    0 0 1 x x|     |      x    |     |x| 13
     * |                        0 1 x|     |      x    |     |x| 14
     * |                        0 0 1|     |      x    |     |x| 15
     * -------------------------------     -------------
     *                .                          .
     *                .                          .
     *                .                          .
     * -------------------------------     -------------
     * |                             |     |1 0        |     |x|  1
     * |                             |     |0 1 0 0 0  |     |x|  2
     * |                             |     |  0 1 0 0  |     |x|  3
     * |                        0 0 0| ... |  0 0 1 0  | ... |x|  4 bb
     * |                             |     |  0 0 0 1 0|     |x|  5
     * |                             |     |        0 1|     |x|  6
     * -------------------------------     -------------
     *
     * Final matrix:
     *
     *               sc                         bb            g
     *  1 2 3 4 5 6 7 8 9101112131415       1 2 3 4 5 6
     * -------------------------------     -------------
     * |1       x                    |     |           |     |x|  1
     * |  1   x             x        |     |           |     |x|  2
     * |    1 x   x                  |     |           |     |x|  3
     * |  0 0 1   x         x        |     |           |     |x|  4
     * |0       1 x   x              |     |           |     |x|  5
     * |    0 0 0 1   x     f        |     |           |     |x|  6
     * |            1 x   x          |     |           |     |x|  7
     * |        0 0 0 1   x f        | ... |           | ... |x|  8 sc
     * |                1 x   x      |     |           |     |x|  9
     * |            0 0 0 1 f x      |     |           |     |x| 10
     * |  0   0   0   0   0 1 x x    |     |           |     |x| 11
     * |                0 0 0 1 x    |     |           |     |x| 12
     * |                    0 0 1 x x|     |      0    |     |M| 13
     * |                        0 1 x|     |      0    |     |M| 14
     * |                        0 0 1|     |      0    |     |M| 15
     * -------------------------------     -------------
     *                .                          .
     *                .                          .
     *                .                          .
     * -------------------------------     -------------
     * |                             |     |1 0        |     |x|  1
     * |                             |     |0 1 0 0 0  |     |x|  2
     * |                             |     |  0 1 0 0  |     |x|  3
     * |                        0 0 0| ... |  0 0 1 0  | ... |x|  4 bb
     * |                             |     |  0 0 0 1 0|     |x|  5
     * |                             |     |        0 1|     |x|  6
     * -------------------------------     -------------
     *
     * @param sc_to_bb_matrix Pointer to a SC to BB tyrosine sub-matrix.
     * @param sc_g_matrix Pointer to a SC g tyrosine sub-matrix.
     * @param bb_g_matrix Pointer to a BB g tyrosine sub-matrix.
     */
    void solve_16_tyrosine(real *const sc_to_bb_matrix,
                           real *const sc_g_matrix,
                           __attribute__((unused)) real *const sc_to_bb_fillins,
                           real *const bb_g_matrix);

    /**
     * Performs the Gaussian elimination of each entry of the first SC to BB
     * sub-matrix pointed by SC_TO_BB_MATRIX.
     *
     * It is assumed that SC_TO_SC_MATRIX, SC_G_MATRIX and BB_G_MATRIX
     * point to the same tryptophan side-chain data.
     *
     * This function has been automatically generated by
     * ilves_step_2_3_16_17_generator.c.
     *
     * Initial matrix:
     *
     *                   sc                             bb            g
     *  1 2 3 4 5 6 7 8 910111213141516171819       1 2 3 4 5 6
     * ---------------------------------------     -------------
     * |1   x           x                    |     |           |     |x|  1
     * |  1 x   x                            |     |           |     |x|  2
     * |0 0 1   x       x                    |     |           |     |x|  3
     * |      1 x   x                        |     |           |     |x|  4
     * |  0 0 0 1   x   f                    |     |           |     |x|  5
     * |          1 x x                      |     |           |     |x|  6
     * |      0 0 0 1 x f                    |     |           |     |x|  7
     * |          0 0 1 f         x   x      |     |           |     |x|  8
     * |0   0   0   0 0 1       x x   f      |     |           |     |x|  9
     * |                  1   x x            | ... |           | ... |x| 10 sc
     * |                    1 x     x        |     |           |     |x| 11
     * |                  0 0 1 x   x        |     |           |     |x| 12
     * |                0 0   0 1 x f f      |     |           |     |x| 13
     * |              0 0       0 1 f x      |     |           |     |x| 14
     * |                    0 0 0 0 1 x x    |     |           |     |x| 15
     * |              0 0       0 0 0 1 x    |     |           |     |x| 16
     * |                            0 0 1 x x|     |      x    |     |x| 17
     * |                                0 1 x|     |      x    |     |x| 18
     * |                                0 0 1|     |      x    |     |x| 19
     * ---------------------------------------     -------------
     *                    .                              .
     *                    .                              .
     *                    .                              .
     * ---------------------------------------     -------------
     * |                                     |     |1 0        |     |x|  1
     * |                                     |     |0 1 0 0 0  |     |x|  2
     * |                                     |     |  0 1 0 0  |     |x|  3
     * |                                0 0 0| ... |  0 0 1 0  | ... |x|  4 bb
     * |                                     |     |  0 0 0 1 0|     |x|  5
     * |                                     |     |        0 1|     |x|  6
     * ---------------------------------------     -------------
     *
     * Final matrix:
     *
     *                   sc                             bb            g
     *  1 2 3 4 5 6 7 8 910111213141516171819       1 2 3 4 5 6
     * ---------------------------------------     -------------
     * |1   x           x                    |     |           |     |x|  1
     * |  1 x   x                            |     |           |     |x|  2
     * |0 0 1   x       x                    |     |           |     |x|  3
     * |      1 x   x                        |     |           |     |x|  4
     * |  0 0 0 1   x   f                    |     |           |     |x|  5
     * |          1 x x                      |     |           |     |x|  6
     * |      0 0 0 1 x f                    |     |           |     |x|  7
     * |          0 0 1 f         x   x      |     |           |     |x|  8
     * |0   0   0   0 0 1       x x   f      |     |           |     |x|  9
     * |                  1   x x            | ... |           | ... |x| 10 sc
     * |                    1 x     x        |     |           |     |x| 11
     * |                  0 0 1 x   x        |     |           |     |x| 12
     * |                0 0   0 1 x f f      |     |           |     |x| 13
     * |              0 0       0 1 f x      |     |           |     |x| 14
     * |                    0 0 0 0 1 x x    |     |           |     |x| 15
     * |              0 0       0 0 0 1 x    |     |           |     |x| 16
     * |                            0 0 1 x x|     |      0    |     |M| 17
     * |                                0 1 x|     |      0    |     |M| 18
     * |                                0 0 1|     |      0    |     |M| 19
     * ---------------------------------------     -------------
     *                    .                              .
     *                    .                              .
     *                    .                              .
     * ---------------------------------------     -------------
     * |                                     |     |1 0        |     |x|  1
     * |                                     |     |0 1 0 0 0  |     |x|  2
     * |                                     |     |  0 1 0 0  |     |x|  3
     * |                                0 0 0| ... |  0 0 1 0  | ... |x|  4 bb
     * |                                     |     |  0 0 0 1 0|     |x|  5
     * |                                     |     |        0 1|     |x|  6
     * ---------------------------------------     -------------
     *
     * @param sc_to_bb_matrix Pointer to a SC to BB tryptophan sub-matrix.
     * @param sc_g_matrix Pointer to a SC g tryptophan sub-matrix.
     * @param bb_g_matrix Pointer to a BB g tryptophan sub-matrix.
     */
    void solve_16_tryptophan(real *const sc_to_bb_matrix,
                             real *const sc_g_matrix,
                             __attribute__((unused)) real *const sc_to_bb_fillins,
                             real *const bb_g_matrix);

    /**
     * Performs the Gaussian elimination of each entry of the first SC to BB
     * sub-matrix pointed by SC_TO_BB_MATRIX.
     *
     * It is assumed that SC_TO_SC_MATRIX, SC_G_MATRIX and BB_G_MATRIX
     * point to the same threonine side-chain data.
     *
     * This function has been automatically generated by
     * step_2_3_16_17_generator.c.
     *
     * Initial matrix:
     *
     *       sc                 bb            g
     *  1 2 3 4 5 6 7       1 2 3 4 5 6
     * ---------------     -------------
     * |1 x x x      |     |           |     |x|  1
     * |0 1 x x      |     |           |     |x|  2
     * |0 0 1 x      |     |           |     |x|  3
     * |0 0 0 1   x x| ... |      x    | ... |x|  4 sc
     * |        1 x  |     |           |     |x|  5
     * |      0 0 1 x|     |      x    |     |x|  6
     * |      0   0 1|     |      x    |     |x|  7
     * ---------------     -------------
     *        .                  .
     *        .                  .
     *        .                  .
     * ---------------     -------------
     * |             |     |1 0        |     |x|  1
     * |             |     |0 1 0 0 0  |     |x|  2
     * |             |     |  0 1 0 0  |     |x|  3
     * |      0   0 0| ... |  0 0 1 0  | ... |x|  4 bb
     * |             |     |  0 0 0 1 0|     |x|  5
     * |             |     |        0 1|     |x|  6
     * ---------------     -------------
     *
     * Final matrix:
     *
     *       sc                 bb            g
     *  1 2 3 4 5 6 7       1 2 3 4 5 6
     * ---------------     -------------
     * |1 x x x      |     |           |     |x|  1
     * |0 1 x x      |     |           |     |x|  2
     * |0 0 1 x      |     |           |     |x|  3
     * |0 0 0 1   x x| ... |      0    | ... |M|  4 sc
     * |        1 x  |     |           |     |x|  5
     * |      0 0 1 x|     |      0    |     |M|  6
     * |      0   0 1|     |      0    |     |M|  7
     * ---------------     -------------
     *        .                  .
     *        .                  .
     *        .                  .
     * ---------------     -------------
     * |             |     |1 0        |     |x|  1
     * |             |     |0 1 0 0 0  |     |x|  2
     * |             |     |  0 1 0 0  |     |x|  3
     * |      0   0 0| ... |  0 0 1 0  | ... |x|  4 bb
     * |             |     |  0 0 0 1 0|     |x|  5
     * |             |     |        0 1|     |x|  6
     * ---------------     -------------
     *
     * @param sc_to_bb_matrix Pointer to a SC to BB threonine sub-matrix.
     * @param sc_g_matrix Pointer to a SC g threonine sub-matrix.
     * @param bb_g_matrix Pointer to a BB g threonine sub-matrix.
     */
    void solve_16_threonine(real *const sc_to_bb_matrix,
                            real *const sc_g_matrix,
                            __attribute__((unused)) real *const sc_to_bb_fillins,
                            real *const bb_g_matrix);

    /**
     * Cleans each SC to BB sub-matrix assigned to THREAD.
     *
     * Initial matrix:
     *
     *   sc         bb              g
     *  1 2 3       1
     * -------     ---
     * |1 x  |     | |     |x|      1 Note: **
     * |0 1 x| ... | | ... |x| sc   2
     * |  0 1|     |x|     |x|      3
     * -------     ---
     *
     * ** This SC to SC sub-matrix belongs to a linear side chain (unrealistic).
     * The side chain types can be any of the defined in molecule.h.
     *
     * Final matrix:
     *
     *   sc         bb              g
     *  1 2 3       1
     * -------     ---
     * |1 x  |     | |     |x|      1
     * |0 1 x| ... | | ... |x| sc   2
     * |  0 1|     |0|     |M|      3
     * -------     ---
     *
     * @param thread Thread descriptor [0 - NTHREADS - 1].
     */
    void solve_16(const int thread);

    /**
     * Performs the Gaussian elimination of the super-diagonal entries
     * of the first SC to SC sub-matrix pointed by SC_TO_SC_MATRIX.
     *
     * It is assumed that SC_TO_SC_MATRIX and SC_G_MATRIX point to the
     * same glycine side-chain data.
     *
     * This function has been automatically generated by
     * step_2_3_16_17_generator.c.
     *
     * Initial matrix:
     *
     *  sc          bb            g
     *          1 2 3 4 5 6
     * ---     -------------
     * ---     -------------
     *
     * Final matrix:
     *
     * sc           bb            g
     *          1 2 3 4 5 6
     * ---     -------------
     * ---     -------------
     *
     * @param sc_to_sc_matrix unused.
     * @param sc_g_matrix unused.
     * @param sc_to_sc_fillins unused.
     */
    void solve_17_glycine(__attribute__((unused)) real *const sc_to_sc_matrix,
                          __attribute__((unused)) real *const sc_g_matrix,
                          __attribute__((unused)) real *const sc_to_sc_fillins);

    /**
     * Performs the Gaussian elimination of the super-diagonal entries
     * of the first SC to SC sub-matrix pointed by SC_TO_SC_MATRIX.
     *
     * It is assumed that SC_TO_SC_MATRIX and SC_G_MATRIX point to the
     * same proline side-chain data.
     *
     * This function has been automatically generated by
     * step_2_3_16_17_generator.c.
     *
     * Initial matrix:
     *
     *         sc                  bb           g
     *  1 2 3 4 5 6 7 8 9       1 2 3 4 5
     * -------------------     -----------
     * |1 x         x    |     |      0  |     |x|  1
     * |0 1         x    |     |      0  |     |x|  2
     * |    1 x     x x  |     |         |     |x|  3
     * |    0 1     x x  |     |         |     |x|  4
     * |        1 x   x x| ... |         | ... |x|  5 sc
     * |        0 1   x x|     |         |     |x|  6
     * |0 0 0 0     1 x  |     |      0  |     |x|  7
     * |    0 0 0 0 0 1 x|     |      0  |     |x|  8
     * |        0 0   0 1|     |      0 0|     |x|  9
     * -------------------     -----------
     *
     * Final matrix:
     *
     *         sc                  bb           g
     *  1 2 3 4 5 6 7 8 9       1 2 3 4 5
     * -------------------     -----------
     * |1 0         0    |     |      0  |     |M|  1
     * |0 1         0    |     |      0  |     |M|  2
     * |    1 0     0 0  |     |         |     |M|  3
     * |    0 1     0 0  |     |         |     |M|  4
     * |        1 0   0 0| ... |         | ... |M|  5 sc
     * |        0 1   0 0|     |         |     |M|  6
     * |0 0 0 0     1 0  |     |      0  |     |M|  7
     * |    0 0 0 0 0 1 0|     |      0  |     |M|  8
     * |        0 0   0 1|     |      0 0|     |x|  9
     * -------------------     -----------
     *
     * @param sc_to_sc_matrix Pointer to a SC to SC proline sub-matrix.
     * @param sc_g_matrix Pointer to a SC g proline sub-matrix.
     * @param sc_to_sc_fillins unused.
     */
    void solve_17_proline(real *const sc_to_sc_matrix,
                          real *const sc_g_matrix,
                          __attribute__((unused)) real *const sc_to_sc_fillins);

    /**
     * Performs the Gaussian elimination of the super-diagonal entries
     * of the first SC to SC sub-matrix pointed by SC_TO_SC_MATRIX.
     *
     * It is assumed that SC_TO_SC_MATRIX and SC_G_MATRIX point to the
     * same cysteine side-chain data.
     *
     * This function has been automatically generated by
     * step_2_3_16_17_generator.c.
     *
     * Initial matrix:
     *
     *    sc              bb            g
     *  1 2 3 4       1 2 3 4 5 6
     * ---------     -------------
     * |1 x    |     |           |     |x|  1
     * |0 1 x x|     |      0    |     |x|  2
     * |  0 1 x| ... |      0    | ... |x|  3 sc
     * |  0 0 1|     |      0    |     |x|  4
     * ---------     -------------
     *
     * Final matrix:
     *
     *    sc              bb            g
     *  1 2 3 4       1 2 3 4 5 6
     * ---------     -------------
     * |1 0    |     |           |     |M|  1
     * |0 1 0 0|     |      0    |     |M|  2
     * |  0 1 0| ... |      0    | ... |M|  3 sc
     * |  0 0 1|     |      0    |     |x|  4
     * ---------     -------------
     *
     * @param sc_to_sc_matrix Pointer to a SC to SC cysteine sub-matrix.
     * @param sc_g_matrix Pointer to a SC g cysteine sub-matrix.
     * @param sc_to_sc_fillins unused.
     */
    void solve_17_cysteine(real *const sc_to_sc_matrix,
                           real *const sc_g_matrix,
                           __attribute__((unused)) real *const sc_to_sc_fillins);

    /**
     * Performs the Gaussian elimination of the super-diagonal entries
     * of the first SC to SC sub-matrix pointed by SC_TO_SC_MATRIX.
     *
     * It is assumed that SC_TO_SC_MATRIX and SC_G_MATRIX point to the
     * same methionine side-chain data.
     *
     * This function has been automatically generated by
     * step_2_3_16_17_generator.c.
     *
     * Initial matrix:
     *
     *          sc                    bb            g
     *  1 2 3 4 5 6 7 8 910       1 2 3 4 5 6
     * ---------------------     -------------
     * |1 x x x            |     |           |     |x|  1
     * |0 1 x x            |     |           |     |x|  2
     * |0 0 1 x            |     |           |     |x|  3
     * |0 0 0 1 x          |     |           |     |x|  4
     * |      0 1 x x x    |     |           |     |x|  5
     * |        0 1 x x    | ... |           | ... |x|  6 sc
     * |        0 0 1 x    |     |           |     |x|  7
     * |        0 0 0 1 x x|     |      0    |     |x|  8
     * |              0 1 x|     |      0    |     |x|  9
     * |              0 0 1|     |      0    |     |x| 10
     * ---------------------     -------------
     *
     * Final matrix:
     *
     *          sc                    bb            g
     *  1 2 3 4 5 6 7 8 910       1 2 3 4 5 6
     * ---------------------     -------------
     * |1 0 0 0            |     |           |     |M|  1
     * |0 1 0 0            |     |           |     |M|  2
     * |0 0 1 0            |     |           |     |M|  3
     * |0 0 0 1 0          |     |           |     |M|  4
     * |      0 1 0 0 0    |     |           |     |M|  5
     * |        0 1 0 0    | ... |           | ... |M|  6 sc
     * |        0 0 1 0    |     |           |     |M|  7
     * |        0 0 0 1 0 0|     |      0    |     |M|  8
     * |              0 1 0|     |      0    |     |M|  9
     * |              0 0 1|     |      0    |     |x| 10
     * ---------------------     -------------
     *
     * @param sc_to_sc_matrix Pointer to a SC to SC methionine sub-matrix.
     * @param sc_g_matrix Pointer to a SC g methionine sub-matrix.
     * @param sc_to_sc_fillins unused.
     */
    void solve_17_methionine(real *const sc_to_sc_matrix,
                             real *const sc_g_matrix,
                             __attribute__((unused)) real *const sc_to_sc_fillins);

    /**
     * Performs the Gaussian elimination of the super-diagonal entries
     * of the first SC to SC sub-matrix pointed by SC_TO_SC_MATRIX.
     *
     * It is assumed that SC_TO_SC_MATRIX and SC_G_MATRIX point to the
     * same alaline side-chain data.
     *
     * This function has been automatically generated by
     * step_2_3_16_17_generator.c.
     *
     * Initial matrix:
     *
     *   sc             bb            g
     *  1 2 3       1 2 3 4 5 6
     * -------     -------------
     * |1 x x|     |      0    |     |x|  1
     * |0 1 x| ... |      0    | ... |x|  2 sc
     * |0 0 1|     |      0    |     |x|  3
     * -------     -------------
     *
     * Final matrix:
     *
     *   sc             bb            g
     *  1 2 3       1 2 3 4 5 6
     * -------     -------------
     * |1 0 0|     |      0    |     |M|  1
     * |0 1 0| ... |      0    | ... |M|  2 sc
     * |0 0 1|     |      0    |     |x|  3
     * -------     -------------
     *
     * @param sc_to_sc_matrix Pointer to a SC to SC alaline sub-matrix.
     * @param sc_g_matrix Pointer to a SC g alaline sub-matrix.
     * @param sc_to_sc_fillins unused.
     */
    void solve_17_alaline(real *const sc_to_sc_matrix,
                          real *const sc_g_matrix,
                          __attribute__((unused)) real *const sc_to_sc_fillins);

    /**
     * Performs the Gaussian elimination of the super-diagonal entries
     * of the first SC to SC sub-matrix pointed by SC_TO_SC_MATRIX.
     *
     * It is assumed that SC_TO_SC_MATRIX and SC_G_MATRIX point to the
     * same valine side-chain data.
     *
     * This function has been automatically generated by
     * step_2_3_16_17_generator.c.
     *
     * Initial matrix:
     *
     *         sc                   bb            g
     *  1 2 3 4 5 6 7 8 9       1 2 3 4 5 6
     * -------------------     -------------
     * |1 x x x          |     |           |     |x|  1
     * |0 1 x x          |     |           |     |x|  2
     * |0 0 1 x          |     |           |     |x|  3
     * |0 0 0 1       x x|     |      0    |     |x|  4
     * |        1 x x x  | ... |           | ... |x|  5 sc
     * |        0 1 x x  |     |           |     |x|  6
     * |        0 0 1 x  |     |           |     |x|  7
     * |      0 0 0 0 1 x|     |      0    |     |x|  8
     * |      0       0 1|     |      0    |     |x|  9
     * -------------------     -------------
     *
     * Final matrix:
     *
     *         sc                   bb            g
     *  1 2 3 4 5 6 7 8 9       1 2 3 4 5 6
     * -------------------     -------------
     * |1 0 0 0          |     |           |     |M|  1
     * |0 1 0 0          |     |           |     |M|  2
     * |0 0 1 0          |     |           |     |M|  3
     * |0 0 0 1       0 0|     |      0    |     |M|  4
     * |        1 0 0 0  | ... |           | ... |M|  5 sc
     * |        0 1 0 0  |     |           |     |M|  6
     * |        0 0 1 0  |     |           |     |M|  7
     * |      0 0 0 0 1 0|     |      0    |     |M|  8
     * |      0       0 1|     |      0    |     |x|  9
     * -------------------     -------------
     *
     * @param sc_to_sc_matrix Pointer to a SC to SC valine sub-matrix.
     * @param sc_g_matrix Pointer to a SC g valine sub-matrix.
     * @param sc_to_sc_fillins unused.
     */
    void solve_17_valine(real *const sc_to_sc_matrix,
                         real *const sc_g_matrix,
                         __attribute__((unused)) real *const sc_to_sc_fillins);

    /**
     * Performs the Gaussian elimination of the super-diagonal entries
     * of the first SC to SC sub-matrix pointed by SC_TO_SC_MATRIX.
     *
     * It is assumed that SC_TO_SC_MATRIX and SC_G_MATRIX point to the
     * same leucine side-chain data.
     *
     * This function has been automatically generated by
     * step_2_3_16_17_generator.c.
     *
     * Initial matrix:
     *
     *            sc                      bb            g
     *  1 2 3 4 5 6 7 8 9101112       1 2 3 4 5 6
     * -------------------------     -------------
     * |1 x x x                |     |           |     |x|  1
     * |0 1 x x                |     |           |     |x|  2
     * |0 0 1 x                |     |           |     |x|  3
     * |0 0 0 1       x x x    |     |           |     |x|  4
     * |        1 x x x        |     |           |     |x|  5
     * |        0 1 x x        |     |           |     |x|  6
     * |        0 0 1 x        | ... |           | ... |x|  7 sc
     * |      0 0 0 0 1 x x    |     |           |     |x|  8
     * |      0       0 1 x    |     |           |     |x|  9
     * |      0       0 0 1 x x|     |      0    |     |x| 10
     * |                  0 1 x|     |      0    |     |x| 11
     * |                  0 0 1|     |      0    |     |x| 12
     * -------------------------     -------------
     *
     * Final matrix:
     *
     *            sc                      bb            g
     *  1 2 3 4 5 6 7 8 9101112       1 2 3 4 5 6
     * -------------------------     -------------
     * |1 0 0 0                |     |           |     |M|  1
     * |0 1 0 0                |     |           |     |M|  2
     * |0 0 1 0                |     |           |     |M|  3
     * |0 0 0 1       0 0 0    |     |           |     |M|  4
     * |        1 0 0 0        |     |           |     |M|  5
     * |        0 1 0 0        |     |           |     |M|  6
     * |        0 0 1 0        | ... |           | ... |M|  7 sc
     * |      0 0 0 0 1 0 0    |     |           |     |M|  8
     * |      0       0 1 0    |     |           |     |M|  9
     * |      0       0 0 1 0 0|     |      0    |     |M| 10
     * |                  0 1 0|     |      0    |     |M| 11
     * |                  0 0 1|     |      0    |     |x| 12
     * -------------------------     -------------
     *
     * @param sc_to_sc_matrix Pointer to a SC to SC leucine sub-matrix.
     * @param sc_g_matrix Pointer to a SC g leucine sub-matrix.
     * @param sc_to_sc_fillins unused.
     */
    void solve_17_leucine(real *const sc_to_sc_matrix,
                          real *const sc_g_matrix,
                          __attribute__((unused)) real *const sc_to_sc_fillins);

    /**
     * Performs the Gaussian elimination of the super-diagonal entries
     * of the first SC to SC sub-matrix pointed by SC_TO_SC_MATRIX.
     *
     * It is assumed that SC_TO_SC_MATRIX and SC_G_MATRIX point to the
     * same isoleucine side-chain data.
     *
     * This function has been automatically generated by
     * step_2_3_16_17_generator.c.
     *
     * Initial matrix:
     *
     *            sc                      bb            g
     *  1 2 3 4 5 6 7 8 9101112       1 2 3 4 5 6
     * -------------------------     -------------
     * |1 x x x                |     |           |     |x|  1
     * |0 1 x x                |     |           |     |x|  2
     * |0 0 1 x                |     |           |     |x|  3
     * |0 0 0 1 x x x          |     |           |     |x|  4
     * |      0 1 x x          |     |           |     |x|  5
     * |      0 0 1 x          |     |           |     |x|  6
     * |      0 0 0 1       x x| ... |      0    | ... |x|  7 sc
     * |              1 x x x  |     |           |     |x|  8
     * |              0 1 x x  |     |           |     |x|  9
     * |              0 0 1 x  |     |           |     |x| 10
     * |            0 0 0 0 1 x|     |      0    |     |x| 11
     * |            0       0 1|     |      0    |     |x| 12
     * -------------------------     -------------
     *
     * Final matrix:
     *
     *            sc                      bb            g
     *  1 2 3 4 5 6 7 8 9101112       1 2 3 4 5 6
     * -------------------------     -------------
     * |1 0 0 0                |     |           |     |M|  1
     * |0 1 0 0                |     |           |     |M|  2
     * |0 0 1 0                |     |           |     |M|  3
     * |0 0 0 1 0 0 0          |     |           |     |M|  4
     * |      0 1 0 0          |     |           |     |M|  5
     * |      0 0 1 0          |     |           |     |M|  6
     * |      0 0 0 1       0 0| ... |      0    | ... |M|  7 sc
     * |              1 0 0 0  |     |           |     |M|  8
     * |              0 1 0 0  |     |           |     |M|  9
     * |              0 0 1 0  |     |           |     |M| 10
     * |            0 0 0 0 1 0|     |      0    |     |M| 11
     * |            0       0 1|     |      0    |     |x| 12
     * -------------------------     -------------
     *
     * @param sc_to_sc_matrix Pointer to a SC to SC isoleucine sub-matrix.
     * @param sc_g_matrix Pointer to a SC g isoleucine sub-matrix.
     * @param sc_to_sc_fillins unused.
     */
    void solve_17_isoleucine(real *const sc_to_sc_matrix,
                             real *const sc_g_matrix,
                             __attribute__((unused)) real *const sc_to_sc_fillins);

    /**
     * Performs the Gaussian elimination of the super-diagonal entries
     * of the first SC to SC sub-matrix pointed by SC_TO_SC_MATRIX.
     *
     * It is assumed that SC_TO_SC_MATRIX and SC_G_MATRIX point to the
     * same aspartic_acid side-chain data.
     *
     * This function has been automatically generated by
     * step_2_3_16_17_generator.c.
     *
     * Initial matrix:
     *
     *     sc               bb            g
     *  1 2 3 4 5       1 2 3 4 5 6
     * -----------     -------------
     * |1 x x    |     |           |     |x|  1
     * |0 1 x    |     |           |     |x|  2
     * |0 0 1 x x| ... |      0    | ... |x|  3 sc
     * |    0 1 x|     |      0    |     |x|  4
     * |    0 0 1|     |      0    |     |x|  5
     * -----------     -------------
     *
     * Final matrix:
     *
     *     sc               bb            g
     *  1 2 3 4 5       1 2 3 4 5 6
     * -----------     -------------
     * |1 0 0    |     |           |     |M|  1
     * |0 1 0    |     |           |     |M|  2
     * |0 0 1 0 0| ... |      0    | ... |M|  3 sc
     * |    0 1 0|     |      0    |     |M|  4
     * |    0 0 1|     |      0    |     |x|  5
     * -----------     -------------
     *
     * @param sc_to_sc_matrix Pointer to a SC to SC aspartic_acid sub-matrix.
     * @param sc_g_matrix Pointer to a SC g aspartic_acid sub-matrix.
     * @param sc_to_sc_fillins unused.
     */
    void solve_17_aspartic_acid(real *const sc_to_sc_matrix,
                                real *const sc_g_matrix,
                                __attribute__((unused))
                                real *const sc_to_sc_fillins);

    /**
     * Performs the Gaussian elimination of the super-diagonal entries
     * of the first SC to SC sub-matrix pointed by SC_TO_SC_MATRIX.
     *
     * It is assumed that SC_TO_SC_MATRIX and SC_G_MATRIX point to the
     * same glutamic_acid side-chain data.
     *
     * This function has been automatically generated by
     * step_2_3_16_17_generator.c.
     *
     * Initial matrix:
     *
     *        sc                  bb            g
     *  1 2 3 4 5 6 7 8       1 2 3 4 5 6
     * -----------------     -------------
     * |1 x x          |     |           |     |x|  1
     * |0 1 x          |     |           |     |x|  2
     * |0 0 1 x x x    |     |           |     |x|  3
     * |    0 1 x x    |     |           |     |x|  4
     * |    0 0 1 x    | ... |           | ... |x|  5 sc
     * |    0 0 0 1 x x|     |      0    |     |x|  6
     * |          0 1 x|     |      0    |     |x|  7
     * |          0 0 1|     |      0    |     |x|  8
     * -----------------     -------------
     *
     * Final matrix:
     *
     *        sc                  bb            g
     *  1 2 3 4 5 6 7 8       1 2 3 4 5 6
     * -----------------     -------------
     * |1 0 0          |     |           |     |M|  1
     * |0 1 0          |     |           |     |M|  2
     * |0 0 1 0 0 0    |     |           |     |M|  3
     * |    0 1 0 0    |     |           |     |M|  4
     * |    0 0 1 0    | ... |           | ... |M|  5 sc
     * |    0 0 0 1 0 0|     |      0    |     |M|  6
     * |          0 1 0|     |      0    |     |M|  7
     * |          0 0 1|     |      0    |     |x|  8
     * -----------------     -------------
     *
     * @param sc_to_sc_matrix Pointer to a SC to SC glutamic_acid sub-matrix.
     * @param sc_g_matrix Pointer to a SC g glutamic_acid sub-matrix.
     * @param sc_to_sc_fillins unused.
     */
    void solve_17_glutamic_acid(real *const sc_to_sc_matrix,
                                real *const sc_g_matrix,
                                __attribute__((unused))
                                real *const sc_to_sc_fillins);

    /**
     * Performs the Gaussian elimination of the super-diagonal entries
     * of the first SC to SC sub-matrix pointed by SC_TO_SC_MATRIX.
     *
     * It is assumed that SC_TO_SC_MATRIX and SC_G_MATRIX point to the
     * same asparagine side-chain data.
     *
     * This function has been automatically generated by
     * step_2_3_16_17_generator.c.
     *
     * Initial matrix:
     *
     *       sc                 bb            g
     *  1 2 3 4 5 6 7       1 2 3 4 5 6
     * ---------------     -------------
     * |1 x x        |     |           |     |x|  1
     * |0 1 x        |     |           |     |x|  2
     * |0 0 1 x x    |     |           |     |x|  3
     * |    0 1 x    | ... |           | ... |x|  4 sc
     * |    0 0 1 x x|     |      0    |     |x|  5
     * |        0 1 x|     |      0    |     |x|  6
     * |        0 0 1|     |      0    |     |x|  7
     * ---------------     -------------
     *
     * Final matrix:
     *
     *       sc                 bb            g
     *  1 2 3 4 5 6 7       1 2 3 4 5 6
     * ---------------     -------------
     * |1 0 0        |     |           |     |M|  1
     * |0 1 0        |     |           |     |M|  2
     * |0 0 1 0 0    |     |           |     |M|  3
     * |    0 1 0    | ... |           | ... |M|  4 sc
     * |    0 0 1 0 0|     |      0    |     |M|  5
     * |        0 1 0|     |      0    |     |M|  6
     * |        0 0 1|     |      0    |     |x|  7
     * ---------------     -------------
     *
     * @param sc_to_sc_matrix Pointer to a SC to SC asparagine sub-matrix.
     * @param sc_g_matrix Pointer to a SC g asparagine sub-matrix.
     * @param sc_to_sc_fillins unused.
     */
    void solve_17_asparagine(real *const sc_to_sc_matrix,
                             real *const sc_g_matrix,
                             __attribute__((unused)) real *const sc_to_sc_fillins);

    /**
     * Performs the Gaussian elimination of the super-diagonal entries
     * of the first SC to SC sub-matrix pointed by SC_TO_SC_MATRIX.
     *
     * It is assumed that SC_TO_SC_MATRIX and SC_G_MATRIX point to the
     * same glutamine side-chain data.
     *
     * This function has been automatically generated by
     * step_2_3_16_17_generator.c.
     *
     * Initial matrix:
     *
     *          sc                    bb            g
     *  1 2 3 4 5 6 7 8 910       1 2 3 4 5 6
     * ---------------------     -------------
     * |1 x x              |     |           |     |x|  1
     * |0 1 x              |     |           |     |x|  2
     * |0 0 1 x x          |     |           |     |x|  3
     * |    0 1 x          |     |           |     |x|  4
     * |    0 0 1 x x x    |     |           |     |x|  5
     * |        0 1 x x    | ... |           | ... |x|  6 sc
     * |        0 0 1 x    |     |           |     |x|  7
     * |        0 0 0 1 x x|     |      0    |     |x|  8
     * |              0 1 x|     |      0    |     |x|  9
     * |              0 0 1|     |      0    |     |x| 10
     * ---------------------     -------------
     *
     * Final matrix:
     *
     *          sc                    bb            g
     *  1 2 3 4 5 6 7 8 910       1 2 3 4 5 6
     * ---------------------     -------------
     * |1 0 0              |     |           |     |M|  1
     * |0 1 0              |     |           |     |M|  2
     * |0 0 1 0 0          |     |           |     |M|  3
     * |    0 1 0          |     |           |     |M|  4
     * |    0 0 1 0 0 0    |     |           |     |M|  5
     * |        0 1 0 0    | ... |           | ... |M|  6 sc
     * |        0 0 1 0    |     |           |     |M|  7
     * |        0 0 0 1 0 0|     |      0    |     |M|  8
     * |              0 1 0|     |      0    |     |M|  9
     * |              0 0 1|     |      0    |     |x| 10
     * ---------------------     -------------
     *
     * @param sc_to_sc_matrix Pointer to a SC to SC glutamine sub-matrix.
     * @param sc_g_matrix Pointer to a SC g glutamine sub-matrix.
     * @param sc_to_sc_fillins unused.
     */
    void solve_17_glutamine(real *const sc_to_sc_matrix,
                            real *const sc_g_matrix,
                            __attribute__((unused)) real *const sc_to_sc_fillins);

    /**
     * Performs the Gaussian elimination of the super-diagonal entries
     * of the first SC to SC sub-matrix pointed by SC_TO_SC_MATRIX.
     *
     * It is assumed that SC_TO_SC_MATRIX and SC_G_MATRIX point to the
     * same histidine side-chain data.
     *
     * This function has been automatically generated by
     * step_2_3_16_17_generator.c.
     *
     * Initial matrix:
     *
     *           sc                     bb            g
     *  1 2 3 4 5 6 7 8 91011       1 2 3 4 5 6
     * -----------------------     -------------
     * |1   x   x            |     |           |     |x|  1
     * |  1 x     x          |     |           |     |x|  2
     * |0 0 1   x x          |     |           |     |x|  3
     * |      1 x     x      |     |           |     |x|  4
     * |0   0 0 1 f   x      |     |           |     |x|  5
     * |  0 0   0 1 x f      | ... |           | ... |x|  6 sc
     * |          0 1 x x    |     |           |     |x|  7
     * |      0 0 0 0 1 x    |     |           |     |x|  8
     * |            0 0 1 x x|     |      0    |     |x|  9
     * |                0 1 x|     |      0    |     |x| 10
     * |                0 0 1|     |      0    |     |x| 11
     * -----------------------     -------------
     *
     * Final matrix:
     *
     *           sc                     bb            g
     *  1 2 3 4 5 6 7 8 91011       1 2 3 4 5 6
     * -----------------------     -------------
     * |1   0   0            |     |           |     |M|  1
     * |  1 0     0          |     |           |     |M|  2
     * |0 0 1   0 0          |     |           |     |M|  3
     * |      1 0     0      |     |           |     |M|  4
     * |0   0 0 1 0   0      |     |           |     |M|  5
     * |  0 0   0 1 0 0      | ... |           | ... |M|  6 sc
     * |          0 1 0 0    |     |           |     |M|  7
     * |      0 0 0 0 1 0    |     |           |     |M|  8
     * |            0 0 1 0 0|     |      0    |     |M|  9
     * |                0 1 0|     |      0    |     |M| 10
     * |                0 0 1|     |      0    |     |x| 11
     * -----------------------     -------------
     *
     * @param sc_to_sc_matrix Pointer to a SC to SC histidine sub-matrix.
     * @param sc_g_matrix Pointer to a SC g histidine sub-matrix.
     * @param sc_to_sc_fillins Pointer to a SC to SC histidine fillins
     * sub-matrix.
     */
    void solve_17_histidine(real *const sc_to_sc_matrix,
                            real *const sc_g_matrix,
                            real *const sc_to_sc_fillins);

    /**
     * Performs the Gaussian elimination of the super-diagonal entries
     * of the first SC to SC sub-matrix pointed by SC_TO_SC_MATRIX.
     *
     * It is assumed that SC_TO_SC_MATRIX and SC_G_MATRIX point to the
     * same lysine side-chain data.
     *
     * This function has been automatically generated by
     * step_2_3_16_17_generator.c.
     *
     * Initial matrix:
     *
     *               sc                         bb            g
     *  1 2 3 4 5 6 7 8 9101112131415       1 2 3 4 5 6
     * -------------------------------     -------------
     * |1 x x x                      |     |           |     |x|  1
     * |0 1 x x                      |     |           |     |x|  2
     * |0 0 1 x                      |     |           |     |x|  3
     * |0 0 0 1 x x x                |     |           |     |x|  4
     * |      0 1 x x                |     |           |     |x|  5
     * |      0 0 1 x                |     |           |     |x|  6
     * |      0 0 0 1 x x x          |     |           |     |x|  7
     * |            0 1 x x          | ... |           | ... |x|  8 sc
     * |            0 0 1 x          |     |           |     |x|  9
     * |            0 0 0 1 x x x    |     |           |     |x| 10
     * |                  0 1 x x    |     |           |     |x| 11
     * |                  0 0 1 x    |     |           |     |x| 12
     * |                  0 0 0 1 x x|     |      0    |     |x| 13
     * |                        0 1 x|     |      0    |     |x| 14
     * |                        0 0 1|     |      0    |     |x| 15
     * -------------------------------     -------------
     *
     * Final matrix:
     *
     *               sc                         bb            g
     *  1 2 3 4 5 6 7 8 9101112131415       1 2 3 4 5 6
     * -------------------------------     -------------
     * |1 0 0 0                      |     |           |     |M|  1
     * |0 1 0 0                      |     |           |     |M|  2
     * |0 0 1 0                      |     |           |     |M|  3
     * |0 0 0 1 0 0 0                |     |           |     |M|  4
     * |      0 1 0 0                |     |           |     |M|  5
     * |      0 0 1 0                |     |           |     |M|  6
     * |      0 0 0 1 0 0 0          |     |           |     |M|  7
     * |            0 1 0 0          | ... |           | ... |M|  8 sc
     * |            0 0 1 0          |     |           |     |M|  9
     * |            0 0 0 1 0 0 0    |     |           |     |M| 10
     * |                  0 1 0 0    |     |           |     |M| 11
     * |                  0 0 1 0    |     |           |     |M| 12
     * |                  0 0 0 1 0 0|     |      0    |     |M| 13
     * |                        0 1 0|     |      0    |     |M| 14
     * |                        0 0 1|     |      0    |     |x| 15
     * -------------------------------     -------------
     *
     * @param sc_to_sc_matrix Pointer to a SC to SC lysine sub-matrix.
     * @param sc_g_matrix Pointer to a SC g lysine sub-matrix.
     * @param sc_to_sc_fillins unused.
     */
    void solve_17_lysine(real *const sc_to_sc_matrix,
                         real *const sc_g_matrix,
                         __attribute__((unused)) real *const sc_to_sc_fillins);

    /**
     * Performs the Gaussian elimination of the super-diagonal entries
     * of the first SC to SC sub-matrix pointed by SC_TO_SC_MATRIX.
     *
     * It is assumed that SC_TO_SC_MATRIX and SC_G_MATRIX point to the
     * same arginine side-chain data.
     *
     * This function has been automatically generated by
     * step_2_3_16_17_generator.c.
     *
     * Initial matrix:
     *
     *                 sc                           bb            g
     *  1 2 3 4 5 6 7 8 91011121314151617       1 2 3 4 5 6
     * -----------------------------------     -------------
     * |1 x x                            |     |           |     |x|  1
     * |0 1 x                            |     |           |     |x|  2
     * |0 0 1     x x                    |     |           |     |x|  3
     * |      1 x x                      |     |           |     |x|  4
     * |      0 1 x                      |     |           |     |x|  5
     * |    0 0 0 1 x                    |     |           |     |x|  6
     * |    0     0 1 x x                |     |           |     |x|  7
     * |            0 1 x                |     |           |     |x|  8
     * |            0 0 1 x x x          | ... |           | ... |x|  9 sc
     * |                0 1 x x          |     |           |     |x| 10
     * |                0 0 1 x          |     |           |     |x| 11
     * |                0 0 0 1 x x x    |     |           |     |x| 12
     * |                      0 1 x x    |     |           |     |x| 13
     * |                      0 0 1 x    |     |           |     |x| 14
     * |                      0 0 0 1 x x|     |      0    |     |x| 15
     * |                            0 1 x|     |      0    |     |x| 16
     * |                            0 0 1|     |      0    |     |x| 17
     * -----------------------------------     -------------
     *
     * Final matrix:
     *
     *                 sc                           bb            g
     *  1 2 3 4 5 6 7 8 91011121314151617       1 2 3 4 5 6
     * -----------------------------------     -------------
     * |1 0 0                            |     |           |     |M|  1
     * |0 1 0                            |     |           |     |M|  2
     * |0 0 1     0 0                    |     |           |     |M|  3
     * |      1 0 0                      |     |           |     |M|  4
     * |      0 1 0                      |     |           |     |M|  5
     * |    0 0 0 1 0                    |     |           |     |M|  6
     * |    0     0 1 0 0                |     |           |     |M|  7
     * |            0 1 0                |     |           |     |M|  8
     * |            0 0 1 0 0 0          | ... |           | ... |M|  9 sc
     * |                0 1 0 0          |     |           |     |M| 10
     * |                0 0 1 0          |     |           |     |M| 11
     * |                0 0 0 1 0 0 0    |     |           |     |M| 12
     * |                      0 1 0 0    |     |           |     |M| 13
     * |                      0 0 1 0    |     |           |     |M| 14
     * |                      0 0 0 1 0 0|     |      0    |     |M| 15
     * |                            0 1 0|     |      0    |     |M| 16
     * |                            0 0 1|     |      0    |     |x| 17
     * -----------------------------------     -------------
     *
     * @param sc_to_sc_matrix Pointer to a SC to SC arginine sub-matrix.
     * @param sc_g_matrix Pointer to a SC g arginine sub-matrix.
     * @param sc_to_sc_fillins unused.
     */
    void solve_17_arginine(real *const sc_to_sc_matrix,
                           real *const sc_g_matrix,
                           __attribute__((unused)) real *const sc_to_sc_fillins);

    /**
     * Performs the Gaussian elimination of the super-diagonal entries
     * of the first SC to SC sub-matrix pointed by SC_TO_SC_MATRIX.
     *
     * It is assumed that SC_TO_SC_MATRIX and SC_G_MATRIX point to the
     * same serine side-chain data.
     *
     * This function has been automatically generated by
     * step_2_3_16_17_generator.c.
     *
     * Initial matrix:
     *
     *    sc              bb            g
     *  1 2 3 4       1 2 3 4 5 6
     * ---------     -------------
     * |1 x    |     |           |     |x|  1
     * |0 1 x x|     |      0    |     |x|  2
     * |  0 1 x| ... |      0    | ... |x|  3 sc
     * |  0 0 1|     |      0    |     |x|  4
     * ---------     -------------
     *
     * Final matrix:
     *
     *    sc              bb            g
     *  1 2 3 4       1 2 3 4 5 6
     * ---------     -------------
     * |1 0    |     |           |     |M|  1
     * |0 1 0 0|     |      0    |     |M|  2
     * |  0 1 0| ... |      0    | ... |M|  3 sc
     * |  0 0 1|     |      0    |     |x|  4
     * ---------     -------------
     *
     * @param sc_to_sc_matrix Pointer to a SC to SC serine sub-matrix.
     * @param sc_g_matrix Pointer to a SC g serine sub-matrix.
     * @param sc_to_sc_fillins unused.
     */
    void solve_17_serine(real *const sc_to_sc_matrix,
                         real *const sc_g_matrix,
                         __attribute__((unused)) real *const sc_to_sc_fillins);

    /**
     * Performs the Gaussian elimination of the super-diagonal entries
     * of the first SC to SC sub-matrix pointed by SC_TO_SC_MATRIX.
     *
     * It is assumed that SC_TO_SC_MATRIX and SC_G_MATRIX point to the
     * same phenylaline side-chain data.
     *
     * This function has been automatically generated by
     * step_2_3_16_17_generator.c.
     *
     * Initial matrix:
     *
     *              sc                        bb            g
     *  1 2 3 4 5 6 7 8 91011121314       1 2 3 4 5 6
     * -----------------------------     -------------
     * |1   x             x        |     |           |     |x|  1
     * |  1 x   x                  |     |           |     |x|  2
     * |0 0 1   x         x        |     |           |     |x|  3
     * |      1 x   x              |     |           |     |x|  4
     * |  0 0 0 1   x     f        |     |           |     |x|  5
     * |          1 x   x          |     |           |     |x|  6
     * |      0 0 0 1   x f        |     |           |     |x|  7
     * |              1 x   x      | ... |           | ... |x|  8 sc
     * |          0 0 0 1 f x      |     |           |     |x|  9
     * |0   0   0   0   0 1 x x    |     |           |     |x| 10
     * |              0 0 0 1 x    |     |           |     |x| 11
     * |                  0 0 1 x x|     |      0    |     |x| 12
     * |                      0 1 x|     |      0    |     |x| 13
     * |                      0 0 1|     |      0    |     |x| 14
     * -----------------------------     -------------
     *
     * Final matrix:
     *
     *              sc                        bb            g
     *  1 2 3 4 5 6 7 8 91011121314       1 2 3 4 5 6
     * -----------------------------     -------------
     * |1   0             0        |     |           |     |M|  1
     * |  1 0   0                  |     |           |     |M|  2
     * |0 0 1   0         0        |     |           |     |M|  3
     * |      1 0   0              |     |           |     |M|  4
     * |  0 0 0 1   0     0        |     |           |     |M|  5
     * |          1 0   0          |     |           |     |M|  6
     * |      0 0 0 1   0 0        |     |           |     |M|  7
     * |              1 0   0      | ... |           | ... |M|  8 sc
     * |          0 0 0 1 0 0      |     |           |     |M|  9
     * |0   0   0   0   0 1 0 0    |     |           |     |M| 10
     * |              0 0 0 1 0    |     |           |     |M| 11
     * |                  0 0 1 0 0|     |      0    |     |M| 12
     * |                      0 1 0|     |      0    |     |M| 13
     * |                      0 0 1|     |      0    |     |x| 14
     * -----------------------------     -------------
     *
     * @param sc_to_sc_matrix Pointer to a SC to SC phenylaline sub-matrix.
     * @param sc_g_matrix Pointer to a SC g phenylaline sub-matrix.
     * @param sc_to_sc_fillins Pointer to a SC to SC phenylaline fillins
     * sub-matrix.
     */
    void solve_17_phenylaline(real *const sc_to_sc_matrix,
                              real *const sc_g_matrix,
                              real *const sc_to_sc_fillins);

    /**
     * Performs the Gaussian elimination of the super-diagonal entries
     * of the first SC to SC sub-matrix pointed by SC_TO_SC_MATRIX.
     *
     * It is assumed that SC_TO_SC_MATRIX and SC_G_MATRIX point to the
     * same tyrosine side-chain data.
     *
     * This function has been automatically generated by
     * step_2_3_16_17_generator.c.
     *
     * Initial matrix:
     *
     *               sc                         bb            g
     *  1 2 3 4 5 6 7 8 9101112131415       1 2 3 4 5 6
     * -------------------------------     -------------
     * |1       x                    |     |           |     |x|  1
     * |  1   x             x        |     |           |     |x|  2
     * |    1 x   x                  |     |           |     |x|  3
     * |  0 0 1   x         x        |     |           |     |x|  4
     * |0       1 x   x              |     |           |     |x|  5
     * |    0 0 0 1   x     f        |     |           |     |x|  6
     * |            1 x   x          |     |           |     |x|  7
     * |        0 0 0 1   x f        | ... |           | ... |x|  8 sc
     * |                1 x   x      |     |           |     |x|  9
     * |            0 0 0 1 f x      |     |           |     |x| 10
     * |  0   0   0   0   0 1 x x    |     |           |     |x| 11
     * |                0 0 0 1 x    |     |           |     |x| 12
     * |                    0 0 1 x x|     |      0    |     |x| 13
     * |                        0 1 x|     |      0    |     |x| 14
     * |                        0 0 1|     |      0    |     |x| 15
     * -------------------------------     -------------
     *
     * Final matrix:
     *
     *               sc                         bb            g
     *  1 2 3 4 5 6 7 8 9101112131415       1 2 3 4 5 6
     * -------------------------------     -------------
     * |1       0                    |     |           |     |M|  1
     * |  1   0             0        |     |           |     |M|  2
     * |    1 0   0                  |     |           |     |M|  3
     * |  0 0 1   0         0        |     |           |     |M|  4
     * |0       1 0   0              |     |           |     |M|  5
     * |    0 0 0 1   0     0        |     |           |     |M|  6
     * |            1 0   0          |     |           |     |M|  7
     * |        0 0 0 1   0 0        | ... |           | ... |M|  8 sc
     * |                1 0   0      |     |           |     |M|  9
     * |            0 0 0 1 0 0      |     |           |     |M| 10
     * |  0   0   0   0   0 1 0 0    |     |           |     |M| 11
     * |                0 0 0 1 0    |     |           |     |M| 12
     * |                    0 0 1 0 0|     |      0    |     |M| 13
     * |                        0 1 0|     |      0    |     |M| 14
     * |                        0 0 1|     |      0    |     |x| 15
     * -------------------------------     -------------
     *
     * @param sc_to_sc_matrix Pointer to a SC to SC tyrosine sub-matrix.
     * @param sc_g_matrix Pointer to a SC g tyrosine sub-matrix.
     * @param sc_to_sc_fillins Pointer to a SC to SC tyrosine fillins
     * sub-matrix.
     */
    void solve_17_tyrosine(real *const sc_to_sc_matrix,
                           real *const sc_g_matrix,
                           real *const sc_to_sc_fillins);

    /**
     * Performs the Gaussian elimination of the super-diagonal entries
     * of the first SC to SC sub-matrix pointed by SC_TO_SC_MATRIX.
     *
     * It is assumed that SC_TO_SC_MATRIX and SC_G_MATRIX point to the
     * same tryptophan side-chain data.
     *
     * This function has been automatically generated by
     * ilves_step_2_3_16_17_generator.c.
     *
     * Initial matrix:
     *
     *                   sc                             bb            g
     *  1 2 3 4 5 6 7 8 910111213141516171819       1 2 3 4 5 6
     * ---------------------------------------     -------------
     * |1   x           x                    |     |           |     |x|  1
     * |  1 x   x                            |     |           |     |x|  2
     * |0 0 1   x       x                    |     |           |     |x|  3
     * |      1 x   x                        |     |           |     |x|  4
     * |  0 0 0 1   x   f                    |     |           |     |x|  5
     * |          1 x x                      |     |           |     |x|  6
     * |      0 0 0 1 x f                    |     |           |     |x|  7
     * |          0 0 1 f         x   x      |     |           |     |x|  8
     * |0   0   0   0 0 1       x x   f      |     |           |     |x|  9
     * |                  1   x x            | ... |           | ... |x| 10 sc
     * |                    1 x     x        |     |           |     |x| 11
     * |                  0 0 1 x   x        |     |           |     |x| 12
     * |                0 0   0 1 x f f      |     |           |     |x| 13
     * |              0 0       0 1 f x      |     |           |     |x| 14
     * |                    0 0 0 0 1 x x    |     |           |     |x| 15
     * |              0 0       0 0 0 1 x    |     |           |     |x| 16
     * |                            0 0 1 x x|     |      0    |     |x| 17
     * |                                0 1 x|     |      0    |     |x| 18
     * |                                0 0 1|     |      0    |     |x| 19
     * ---------------------------------------     -------------
     *
     * Final matrix:
     *
     *                   sc                             bb            g
     *  1 2 3 4 5 6 7 8 910111213141516171819       1 2 3 4 5 6
     * ---------------------------------------     -------------
     * |1   0           0                    |     |           |     |M|  1
     * |  1 0   0                            |     |           |     |M|  2
     * |0 0 1   0       0                    |     |           |     |M|  3
     * |      1 0   0                        |     |           |     |M|  4
     * |  0 0 0 1   0   0                    |     |           |     |M|  5
     * |          1 0 0                      |     |           |     |M|  6
     * |      0 0 0 1 0 0                    |     |           |     |M|  7
     * |          0 0 1 0         0   0      |     |           |     |M|  8
     * |0   0   0   0 0 1       0 0   0      |     |           |     |M|  9
     * |                  1   0 0            | ... |           | ... |M| 10 sc
     * |                    1 0     0        |     |           |     |M| 11
     * |                  0 0 1 0   0        |     |           |     |M| 12
     * |                0 0   0 1 0 0 0      |     |           |     |M| 13
     * |              0 0       0 1 0 0      |     |           |     |M| 14
     * |                    0 0 0 0 1 0 0    |     |           |     |M| 15
     * |              0 0       0 0 0 1 0    |     |           |     |M| 16
     * |                            0 0 1 0 0|     |      0    |     |M| 17
     * |                                0 1 0|     |      0    |     |M| 18
     * |                                0 0 1|     |      0    |     |x| 19
     * ---------------------------------------     -------------
     *
     * @param sc_to_sc_matrix Pointer to a SC to SC tryptophan sub-matrix.
     * @param sc_g_matrix Pointer to a SC g tryptophan sub-matrix.
     */
    void solve_17_tryptophan(real *const sc_to_sc_matrix,
                             real *const sc_g_matrix,
                             real *const sc_to_sc_fillins);

    /**
     * Performs the Gaussian elimination of the super-diagonal entries
     * of the first SC to SC sub-matrix pointed by SC_TO_SC_MATRIX.
     *
     * It is assumed that SC_TO_SC_MATRIX and SC_G_MATRIX point to the
     * same threonine side-chain data.
     *
     * This function has been automatically generated by
     * step_2_3_16_17_generator.c.
     *
     * Initial matrix:
     *
     *       sc                 bb            g
     *  1 2 3 4 5 6 7       1 2 3 4 5 6
     * ---------------     -------------
     * |1 x x x      |     |           |     |x|  1
     * |0 1 x x      |     |           |     |x|  2
     * |0 0 1 x      |     |           |     |x|  3
     * |0 0 0 1   x x| ... |      0    | ... |x|  4 sc
     * |        1 x  |     |           |     |x|  5
     * |      0 0 1 x|     |      0    |     |x|  6
     * |      0   0 1|     |      0    |     |x|  7
     * ---------------     -------------
     *
     * Final matrix:
     *
     *       sc                 bb            g
     *  1 2 3 4 5 6 7       1 2 3 4 5 6
     * ---------------     -------------
     * |1 0 0 0      |     |           |     |M|  1
     * |0 1 0 0      |     |           |     |M|  2
     * |0 0 1 0      |     |           |     |M|  3
     * |0 0 0 1   0 0| ... |      0    | ... |M|  4 sc
     * |        1 0  |     |           |     |M|  5
     * |      0 0 1 0|     |      0    |     |M|  6
     * |      0   0 1|     |      0    |     |x|  7
     * ---------------     -------------
     *
     * @param sc_to_sc_matrix Pointer to a SC to SC threonine sub-matrix.
     * @param sc_g_matrix Pointer to a SC g threonine sub-matrix.
     * @param sc_to_sc_fillins unused.
     */
    void solve_17_threonine(real *const sc_to_sc_matrix,
                            real *const sc_g_matrix,
                            __attribute__((unused)) real *const sc_to_sc_fillins);

    /**
     * Performs the Gaussian elimination of the superdiagonal entries of each SC
     * to SC sub-matrix assigned to THREAD.
     *
     * Initial matrix:
     *
     *   sc         bb              g
     *  1 2 3       1
     * -------     ---
     * |1 x  |     | |     |x|      1 Note: **
     * |0 1 x| ... | | ... |x| sc   2
     * |  0 1|     |0|     |x|      3
     * -------     ---
     *
     * ** This SC to SC sub-matrix belongs to a linear side chain (unrealistic).
     * The side chain types can be any of the defined in molecule.h.
     *
     * Final matrix:
     *
     *   sc         bb              g
     *  1 2 3       1
     * -------     ---
     * |1 0  |     | |     |M|      1
     * |0 1 0| ... | | ... |M| sc   2
     * |  0 1|     |0|     |x|      3
     * -------     ---
     *
     * @param thread Thread descriptor [0 - NTHREADS - 1].
     */
    void solve_17(const int thread);

    /**
     * Cleans superdiagonal entries of a spl submatrix with ends type A.
     *
     * It is assumed that this function is called by thread 0.
     *
     * Initial matrix:
     *
     * spl           bb             sep       g
     * -----     -------------     -----
     * |1  | ... |x x        | ... |   | ... |x| sp    1
     * -----     -------------     -----
     *                 .
     *                 .
     *                 .
     * -----     -------------     -----
     * |0  |     |1 0        |     |0  |     |x|       1
     * |0  | ... |0 1 0 0 0  | ... |0  |     |x|       2
     * -----     |  0 1 0 0  |     |   | ... |x| bb    3
     *           |  0 0 1 0  |     |   |     |x|       4
     *           |  0 0 0 1 0|     |  0|     |x|       5
     *           |        0 1|     |  0|     |x|       6
     *           -------------     -----
     *
     * Final matrix:
     *
     * spl           bb             sep       g
     * -----     -------------     -----
     * |1  | ... |0 0        | ... |   | ... |M| spl    1
     * -----     -------------     -----
     *                 .
     *                 .
     *                 .
     * -----     -------------     -----
     * |0  |     |1 0        |     |0  |     |x|       1
     * |0  | ... |0 1 0 0 0  | ... |0  |     |x|       2
     * -----     |  0 1 0 0  |     |   | ... |x| bb    3
     *           |  0 0 1 0  |     |   |     |x|       4
     *           |  0 0 0 1 0|     |  0|     |x|       5
     *           |        0 1|     |  0|     |x|       6
     *           -------------     -----
     *
     * @param bb_g_matrix Pointer to the first bb-g matrix of the first thread.
     */
    void solve_18_spl_ends_A(real *const bb_g_matrix);

    /**
     * Cleans superdiagonal entries of a spl submatrix with ends type B.
     *
     * It is assumed that this function is called by thread 0.
     *
     * Initial matrix:
     *
     *  sp          bb              sep       g
     * -----     -------------     -----
     * |1 x| ... |           | ... |   | ... |x| spl   1
     * |0 1|     |x x        |     |   |     |x|       2
     * -----     -------------     -----
     *                 .
     *                 .
     *                 .
     * -----     -------------     -----
     * |  0|     |1 0        |     |x  |     |x|       1
     * |  0| ... |0 1 0 0 0  | ... |x  |     |x|       2
     * -----     |  0 1 0 0  |     |   | ... |x| bb    3 First block.
     *           |  0 0 1 0  |     |   |     |x|       4
     *           |  0 0 0 1 0|     |  x|     |x|       5
     *           |        0 1|     |  x|     |x|       6
     *           -------------     -----
     *
     * Final matrix:
     *
     *  sp          bb              sep       g
     * -----     -------------     -----
     * |1 0| ... |           | ... |   | ... |M| spl   1
     * |0 1|     |0 0        |     |   |     |M|       2
     * -----     -------------     -----
     *                 .
     *                 .
     *                 .
     * -----     -------------     -----
     * |  0|     |1 0        |     |x  |     |x|       1
     * |  0| ... |0 1 0 0 0  | ... |x  |     |x|       2
     * -----     |  0 1 0 0  |     |   | ... |x| bb    3 First block.
     *           |  0 0 1 0  |     |   |     |x|       4
     *           |  0 0 0 1 0|     |  x|     |x|       5
     *           |        0 1|     |  x|     |x|       6
     *           -------------     -----
     *
     * @param bb_g_matrix Pointer to the first bb-g matrix of the first thread.
     */
    void solve_18_spl_ends_B(real *const bb_g_matrix);

    /**
     * Clean the superdiagonal entries of the spr matrix. This funcion
     * is for the special case of a proline as the last side-chain.
     *
     * It is assumed that this function is called by the last thread.
     *
     * Initial matrix (last side-chain proline):
     *
     * sp                sc                   bb            sep       g
     * -----     -------------------     -------------     -----
     * |  1| ... |                x| ... |        x  | ... |  x| ... |x| spr  1
     * -----     -------------------     -------------     -----
     *                    .
     *                    .
     *                    .
     *           -------------------     -------------     -----
     *           |1 0         0    |     |      0    |     |   |     |x|      1
     *           |0 1         0    |     |      0    |     |   |     |x|      2
     *           |    1 0     0 0  |     |           |     |   |     |x|      3
     *           |    0 1     0 0  |     |           |     |   |     |x|      4
     *           |        1 0   0 0| ... |           | ... |   | ... |x| sc   5
     *           |        0 1   0 0|     |           |     |   |     |x|      6
     *           |0 0 0 0     1 0  |     |      0    |     |   |     |x|      7
     * -----     |    0 0 0 0 0 1 0|     |           |     |   |     |x|      8
     * |  0| ... |        0 0   0 1|     |        0  |     |  0|     |x|      9
     * -----     -------------------     -------------     -----
     *                                          .
     *                                          .
     *                                          .
     *           -------------------     -------------
     *           |                 |     |1 0        |     |0  |     |x|      1
     *           |                 |     |0 1 0 0 0  | ... |0  |     |x|      2
     *           |                 | ... |  0 1 0 0  |     |   | ... |x| bb   3
     * -----     |0 0         0    |     |  0 0 1 0  |     |  0|     |x|      4
     * |  0| ... |                0|     |  0 0 0 1  |     |  0|     |x|      5
     * -----     -------------------     -------------
     *
     * -----                             -------------     -----
     * |  0|               ...           |        0 0|     |  1|     |x| sep  1
     * -----                             -------------     -----
     *
     * Final matrix:
     *
     * sp                sc                   bb            sep       g
     * -----     -------------------     -------------     -----
     * |  1| ... |                0| ... |        0  | ... |  0| ... |M| spr  1
     * -----     -------------------     -------------     -----
     *                    .
     *                    .
     *                    .
     *           -------------------     -------------     -----
     *           |1 0         0    |     |      0    |     |   |     |x|      1
     *           |0 1         0    |     |      0    |     |   |     |x|      2
     *           |    1 0     0 0  |     |           |     |   |     |x|      3
     *           |    0 1     0 0  |     |           |     |   |     |x|      4
     *           |        1 0   0 0| ... |           | ... |   | ... |x| sc   5
     *           |        0 1   0 0|     |           |     |   |     |x|      6
     *           |0 0 0 0     1 0  |     |      0    |     |   |     |x|      7
     * -----     |    0 0 0 0 0 1 0|     |           |     |   |     |x|      8
     * |  0| ... |        0 0   0 1|     |        0  |     |  0|     |x|      9
     * -----     -------------------     -------------     -----
     *                                          .
     *                                          .
     *                                          .
     *           -------------------     -------------
     *           |                 |     |1 0        |     |0  |     |x|      1
     *           |                 |     |0 1 0 0 0  | ... |0  |     |x|      2
     *           |                 | ... |  0 1 0 0  |     |   | ... |x| bb   3
     * -----     |0 0         0    |     |  0 0 1 0  |     |  0|     |x|      4
     * |  0| ... |                0|     |  0 0 0 1  |     |  0|     |x|      5
     * -----     -------------------     -------------
     *
     * -----                             -------------     -----
     * |  0|               ...           |        0 0|     |  1|     |x| sep  1
     * -----                             -------------     -----
     *
     * @param sc_g_matrix Pointer to the last sc-g matrix of the last thread.
     * @param bb_g_matrix Pointer to the last bb-g matrix of the last thread.
     * @param sep_g_matrix Pointer to the last sep-g matrix of the last thread.
     */
    void solve_18_spr_proline(real *const sc_g_matrix,
                              real *const bb_g_matrix,
                              real *const sep_g_matrix);

    /**
     * Clean the subdiagonal entries of the spr matrix. This funcion is for the
     * general case (non-proline last side-chain).
     *
     * It is assumed that this function is called by the last thread.
     *
     * Initial matrix (last side-chain not proline):
     *
     *   sp          bb              sep      g
     * -----     -------------     -----
     * |  1| ... |        x x| ... |  x| ... |x| spr   1
     * -----     -------------     -----
     *                 .
     *                 .
     *                 .
     *           -------------     -----
     *           |1 0        |     |0  |     |x|       1
     *           |0 1 0 0 0  | ... |0  |     |x|       2
     *           |  0 1 0 0  |     |   | ... |x| bb    3
     * -----     |  0 0 1 0  |     |   |     |x|       4
     * |  0|     |  0 0 0 1 0|     |  0|     |x|       5
     * |  0|     |        0 1|     |  0|     |x|       6
     * -----     -------------     -----
     *
     * -----     -------------     -----
     * |  0|     |        0 0|     |  1|     |x| sep   1
     * -----     -------------     -----
     *
     * Final matrix (last side-chain not proline):
     *
     * sp            bb             sep       g
     * -----     -------------     -----
     * |  1| ... |        0 0| ... |  0| ... |M| spr   1
     * -----     -------------     -----
     *                 .
     *                 .
     *                 .
     *           -------------     -----
     *           |1 0        |     |0  |     |x|       1
     *           |0 1 0 0 0  | ... |0  |     |x|       2
     *           |  0 1 0 0  |     |   | ... |x| bb    3
     * -----     |  0 0 1 0  |     |   |     |x|       4
     * |  0|     |  0 0 0 1 0|     |  0|     |x|       5
     * |  0|     |        0 1|     |  0|     |x|       6
     * -----     -------------     -----
     *
     * -----     -------------     -----
     * |  0|     |        0 0|     |  1|     |x| sep   1
     * -----     -------------     -----
     *
     * @param bb_g_matrix Pointer to the last bb-g matrix of the last thread.
     * @param sep_g_matrix Pointer to the last sep-g matrix of the last thread.
     */
    void solve_18_spr_general(real *const bb_g_matrix, real *const sep_g_matrix);

    /**
     * Clean every superdiagonal entry of the special submatrix.
     *
     * Initial matrix (last side-chain not proline AND ends type A):
     *
     * sp          bb              sep      g
     * -----     -------------     -----
     * |1  | ... |x x        | ... |   | ... |x| spl   1
     * -----     -------------     -----
     * -----     -------------     -----
     * |  1| ... |        x x| ... |  x| ... |x| spr   2
     * -----     -------------     -----
     *                 .
     *                 .
     *                 .
     * -----     -------------     -----
     * |0  |     |1 0        |     |0  |     |x|       1
     * |0  | ... |0 1 0 0 0  | ... |0  |     |x|       2
     * -----     |  0 1 0 0  |     |   | ... |x| bb    3 First block.
     *           |  0 0 1 0  |     |   |     |x|       4
     *           |  0 0 0 1 0|     |  0|     |x|       5
     *           |        0 1|     |  0|     |x|       6
     *           -------------     -----
     *           -------------     -----
     *           |1 0        |     |0  |     |x|       1
     *           |0 1 0 0 0  | ... |0  |     |x|       2
     *           |  0 1 0 0  |     |   | ... |x| bb    3 Last block.
     * -----     |  0 0 1 0  |     |   |     |x|       4
     * |  0|     |  0 0 0 1 0|     |  0|     |x|       5
     * |  0|     |        0 1|     |  0|     |x|       6
     * -----     -------------     -----
     *
     * -----     -------------     -----
     * |  0|     |        0 0|     |  1|     |x| sep   1 Last block.
     * -----     -------------     -----
     *
     * Initial matrix (last side-chain proline AND ends type A):
     *
     * sp                sc                   bb            sep       g
     * -----     -------------------     -------------     -----
     * |1  | ... |                 | ... |x x        | ... |   | ... |x| spl  1
     * -----     -------------------     -------------     -----
     *
     * -----     -------------------     -------------     -----
     * |  1| ... |                x| ... |        x  | ... |  x| ... |x| spr  2
     * -----     -------------------     -------------     -----
     *                    .
     *                    .
     *                    .
     *           -------------------     -------------     -----
     *           |1 0         0    |     |      0    |     |   |     |x|      1
     *           |0 1         0    |     |      0    |     |   |     |x|      2
     *           |    1 0     0 0  |     |           |     |   |     |x|      3
     *           |    0 1     0 0  |     |           |     |   |     |x|      4
     *           |        1 0   0 0| ... |           | ... |   | ... |x| sc   5
     * Last block. |        0 1   0 0|     |           |     |   |     |x| 6 |0
     * 0 0 0     1 0  |     |      0    |     |   |     |x|      7
     * -----     |    0 0 0 0 0 1 0|     |           |     |   |     |x|      8
     * |  0| ... |        0 0   0 1|     |        0  |     |  0|     |x|      9
     * -----     -------------------     -------------     -----
     *                                          .
     *                                          .
     *                                          .
     * -----                             -------------
     * |  0|                             |1 0        |     |0  |     |x|      1
     * |  0|               ...           |0 1 0 0 0  | ... |0  |     |x|      2
     * -----                             |  0 1 0 0  |     |   | ... |x| bb   3
     * First block. |  0 0 1 0  |     |   |     |x|      4 |  0 0 0 1 0|     |
     * 0|     |x|      5 |        0 1|     |  0|     |x|      6
     *                                   -------------
     *           -------------------     -------------
     *           |                 |     |1 0        |     |0  |     |x|      1
     *           |                 |     |0 1 0 0 0  | ... |0  |     |x|      2
     *           |                 | ... |  0 1 0 0  |     |   | ... |x| bb   3
     * Last block.
     * -----     |0 0         0    |     |  0 0 1 0  |     |  0|     |x|      4
     * |  0| ... |                0|     |  0 0 0 1  |     |  0|     |x|      5
     * -----     -------------------     -------------
     *
     * -----                             -------------     -----
     * |  0|               ...           |        0 0|     |  1|     |x| sep  1
     * Last block.
     * -----                             -------------     -----
     *
     * Initial matrix (ends type B):
     *
     *  sp          bb              sep       g
     * -----     -------------     -----
     * |1 x| ... |           | ... |   | ... |x| spl   1
     * |0 1|     |x x        |     |   |     |x|       2
     * -----     -------------     -----
     *                 .
     *                 .
     *                 .
     * -----     -------------     -----
     * |  0|     |1 0        |     |x  |     |x|       1
     * |  0| ... |0 1 0 0 0  | ... |x  |     |x|       2
     * -----     |  0 1 0 0  |     |   | ... |x| bb    3 First block.
     *           |  0 0 1 0  |     |   |     |x|       4
     *           |  0 0 0 1 0|     |  x|     |x|       5
     *           |        0 1|     |  x|     |x|       6
     *           -------------     -----
     *
     * Final matrix (last side-chain not proline AND ends type A):
     *
     * sp            bb             sep       g
     * -----     -------------     -----
     * |1  | ... |0 0        | ... |   | ... |M| spl   1
     * -----     -------------     -----
     * -----     -------------     -----
     * |  1| ... |        0 0| ... |  0| ... |M| spr   2
     * -----     -------------     -----
     *                 .
     *                 .
     *                 .
     * -----     -------------     -----
     * |0  |     |1 0        |     |x  |     |x|       1
     * |0  | ... |0 1 0 0 0  | ... |0  |     |x|       2
     * -----     |  0 1 0 0  |     |   | ... |x| bb    3 First block.
     *           |  0 0 1 0  |     |   |     |x|       4
     *           |  0 0 0 1 0|     |  0|     |x|       5
     *           |        0 1|     |  0|     |x|       6
     *           -------------     -----
     *           -------------     -----
     *           |1 0        |     |0  |     |x|       1
     *           |0 1 0 0 0  | ... |0  |     |x|       2
     *           |  0 1 0 0  |     |   | ... |x| bb    3 Last block.
     * -----     |  0 0 1 0  |     |   |     |x|       4
     * |  0|     |  0 0 0 1 0|     |  0|     |x|       5
     * |  0|     |        0 1|     |  0|     |x|       6
     * -----     -------------     -----
     *
     * -----     -------------     -----
     * |  0|     |        0 0|     |  1|     |x| sep   1 Last block.
     * -----     -------------     -----
     *
     * Final matrix (last side-chain proline AND ends type A):
     *
     * sp                sc                   bb            sep       g
     * -----     -------------------     -------------     -----
     * |1  | ... |                 | ... |0 0        | ... |   | ... |M| spl  1
     * -----     -------------------     -------------     -----
     *
     * -----     -------------------     -------------     -----
     * |  1| ... |                0| ... |        0  | ... |  0| ... |M| spr  2
     * -----     -------------------     -------------     -----
     *                    .
     *                    .
     *                    .
     *           -------------------     -------------     -----
     *           |1 0         0    |     |      0    |     |   |     |x|      1
     *           |0 1         0    |     |      0    |     |   |     |x|      2
     *           |    1 0     0 0  |     |           |     |   |     |x|      3
     *           |    0 1     0 0  |     |           |     |   |     |x|      4
     *           |        1 0   0 0| ... |           | ... |   | ... |x| sc   5
     * Last block. |        0 1   0 0|     |           |     |   |     |x| 6 |0
     * 0 0 0     1 0  |     |      0    |     |   |     |x|      7
     * -----     |    0 0 0 0 0 1 0|     |           |     |   |     |x|      8
     * |  0| ... |        0 0   0 1|     |        0  |     |  0|     |x|      9
     * -----     -------------------     -------------     -----
     *                                          .
     *                                          .
     *                                          .
     * -----                             -------------
     * |  0|                             |1 0        |     |0  |     |x|      1
     * |  0|               ...           |0 1 0 0 0  | ... |0  |     |x|      2
     * -----                             |  0 1 0 0  |     |   | ... |x| bb   3
     * First block. |  0 0 1 0  |     |   |     |x|      4 |  0 0 0 1 0|     |
     * 0|     |x|      5 |        0 1|     |  0|     |x|      6
     *                                   -------------
     *           -------------------     -------------
     *           |                 |     |1 0        |     |0  |     |x|      1
     *           |                 |     |0 1 0 0 0  | ... |0  |     |x|      2
     *           |                 | ... |  0 1 0 0  |     |   | ... |x| bb   3
     * Last block.
     * -----     |0 0         0    |     |  0 0 1 0  |     |  0|     |x|      4
     * |  0| ... |                0|     |  0 0 0 1  |     |  0|     |x|      5
     * -----     -------------------     -------------
     *
     * -----                             -------------     -----
     * |  0|               ...           |        0 0|     |  1|     |x| sep  1
     * Last block.
     * -----                             -------------     -----
     *
     * Final matrix (ends type B):
     *
     *  sp          bb              sep       g
     * -----     -------------     -----
     * |1 0| ... |           | ... |   | ... |M| spl   1
     * |0 1|     |0 0        |     |   |     |M|       2
     * -----     -------------     -----
     *                 .
     *                 .
     *                 .
     * -----     -------------     -----
     * |  0|     |1 0        |     |0  |     |x|       1
     * |  0| ... |0 1 0 0 0  | ... |0  |     |x|       2
     * -----     |  0 1 0 0  |     |   | ... |x| bb    3 First block.
     *           |  0 0 1 0  |     |   |     |x|       4
     *           |  0 0 0 1 0|     |  0|     |x|       5
     *           |        0 1|     |  0|     |x|       6
     *           -------------     -----
     *
     * @param thread Thread descriptor [0 - NTHREADS - 1].
     */
    void solve_18(const int thread);

    /**
     * For each entry of G between FIRST_ELEMENT and (FIRST_ELEMENT +
     * NUM_ELEMENTS - 1) that corresponds to a bond, that joins atoms a and b:
     *     XPRIME[a] += G[b] * (X>[b] - X>[a]) * (1 / mass[a]);
     *     XPRIME[b] -= G[b] * (X>[b] - X>[a]) * (1 / mass[a]);
     *
     * @param first_element First element to compute.
     * @param num_elements Num elements to compute.
     * @param first_row Row of the of FIRST_ELEMENT.
     * @param g Sparse independent-term matrix.
     * @param xprime Current atoms positions.
     * @param reset_current_lagr Reset current approximation of the lagrange
     * multipliers? (true if this is the first time that the function is
     * called in this time-step).
     */
    void update_positions_loop(const size_t first_element,
                               const size_t num_elements,
                               const int first_row,
                               const g_t *const __restrict__ g,
                               const ArrayRef<RVec> xprime,
                               const bool reset_current_lagr);

    /**
     * For each entry of G that corresponds to a bond, that joins atoms a and b:
     *     XPRIME[a] += G[b] * (X>[b] - X>[a]) * (1 / mass[a]);
     *     XPRIME[b] -= G[b] * (X>[b] - X>[a]) * (1 / mass[a]);
     *
     * @param thread Thread descriptor [0 - NTHREADS - 1].
     * @param g Sparse independent-term matrix.
     * @param xprime Current atoms positions.
     * @param current_lagr Reset the current approximation of the lagrange
     * multipliers? (true if this is the first time that the function is called
     * in this time-step).
     * @param lock_start True if, before accessing to the first bond data (i ==
     * 0), the thread has to acquire the (THREAD - 1)th lock before updating the
     * positions. False otherwise. The lock will be released when i == 1.
     * @param lock_end True if, before accessing to the last bond data (i ==
     * g->size - 1), the thread has to acquire the (THREAD)th lock before
     * updating the positions. False otherwise. The lock will be released at the
     * end of the same iteration.
     */
    void update_positions_inm(const int thread,
                              const g_t *const __restrict__ g,
                              const ArrayRef<RVec> xprime,
                              const bool reset_current_lagr,
                              const bool lock_start,
                              const bool lock_end);

    /**
     * For each bond b assigned to THREAD, that joins atoms a and b:
     *     XPRIME[a] += G[b] * (X>[b] - X>[a]) * (1 / mass[a]);
     *     XPRIME[b] -= G[b] * (X>[b] - X>[a]) * (1 / mass[a]);
     *
     * @param thread Thread descriptor [0 - NTHREADS - 1].
     * @param xprime Current atoms positions.
     * @param reset_current_lagr Reset the current approximation of the
     * lagrange multiplers? (true if this is the first time that the function
     * is called in this time-step).
     */
    void update_positions(const int thread,
                          const ArrayRef<RVec> xprime,
                          const bool reset_current_lagr);

    /**
     * For each entry of G that corresponds to a bond, that joins atoms a and b:
     *
     * for d1 in DIM
     *     for d2 in DIM
     *         VIRIAL[d1][d2] -= -g[b] * (X>[b] - X>[a])[d1] * (X>[b] -
     * X>[a])[d2]
     *
     * @param g
     * @param virial
     */
    void update_virial_loop(const g_t *const __restrict__ g, tensor virial);

    /**
     * For each bond b asssigned to THREAD, that joins atoms a and b:
     *
     * for d1 in DIM
     *     for d2 in DIM
     *         VIRIAL[d1][d2] -= -g[b] * (X>[b] - X>[a])[d1] * (X>[b] -
     * X>[a])[d2]
     *
     * @param thread Thread descriptor [0 - NTHREADS - 1].
     * @param virial The virial (a DIMxDIM matrix).
     */
    void update_virial(const int thread, tensor virial);

    /**
     * For each entry of G between FIRST_ELEMENT and (FIRST_ELEMENT +
     * NUM_ELEMENTS - 1) that corresponds to a bond, that joins atoms a and b:
     *     XPRIME[a] += G[b] * (X>[b] - X>[a]) * (1 / mass[a]) * invdt;
     *     XPRIME[b] -= G[b] * (X>[b] - X>[a]) * (1 / mass[a]) * invdt;
     *
     * @param first_element First element to compute.
     * @param num_elements Num elements to compute.
     * @param first_row Row of the of FIRST_ELEMENT.
     * @param g Sparse independent-term matrix.
     * @param vprime Pointer to the current atoms velocities.
     * @param invdt Inverse of the time-step in picoseconds.
     */
    void update_velocities_loop(const size_t first_element,
                                const size_t num_elements,
                                const int first_row,
                                const g_t *const __restrict__ g,
                                const ArrayRef<RVec> vprime,
                                const real invdt);

    /**
     * For each entry of G that corresponds to a bond, that joins atoms a and b:
     *     XPRIME[a] += G[b] * (X>[b] - X>[a]) * (1 / mass[a]) * invdt;
     *     XPRIME[b] -= G[b] * (X>[b] - X>[a]) * (1 / mass[a]) * invdt;
     *
     * @param thread Thread descriptor [0 - NTHREADS - 1].
     * @param g Sparse independent-term matrix.
     * @param vprime Pointer to the current atoms velocities
     * @param invdt Inverse of the time-step in picoseconds.
     * @param lock_start 1 if, before accessing to the first bond data (i == 0),
     * the thread has to acquire the (THREAD - 1)th lock before updating the
     * positions. 0 otherwise. The lock will be released when i == 1.
     * @param lock_end True if, before accessing to the last bond data (i ==
     * g->size - 1), the thread has to acquire the (THREAD)th lock before
     * updating the positions. False otherwise. The lock will be released at the
     * end of the same iteration.
     */
    void update_velocities_inm(const int thread,
                               const g_t *const __restrict__ g,
                               const ArrayRef<RVec> vprime,
                               const real invdt,
                               const bool lock_start,
                               const bool lock_end);

    /**
     * For each bond b assigned to THREAD, that joins atoms a and b:
     *     XPRIME[a] += G[b] * (X>[b] - X>[a]) * (1 / mass[a]) * invdt;
     *     XPRIME[b] -= G[b] * (X>[b] - X>[a]) * (1 / mass[a]) * invdt;
     *
     * @param thread Thread descriptor [0 - NTHREADS - 1].
     * @param vprime Current atoms velocities.
     * @param invdt Inverse of the time-step in picoseconds.
     */
    void update_velocities(const int thread,
                           const ArrayRef<RVec> vprime,
                           const real invdt);

    /*
     * Debug functions.
     */

    /**
     * Print an array of SIZE reals.
     *
     * @param array Array of reals.
     * @param size Number of reals in ARRAY.
     */
    void print_array(const real *const array, const size_t size);

    /**
     * Print every sub-matrix (shared and private) of every thread.
     *
     */
    void print_data();

    /**
     * Copy the sparse matrix A_SRC into the dense matrix A_DST.
     *
     * @param A_src Sparse matrix.
     * @param n Size of the square matrix A_DST.
     * @param A_dst Dense matrix.
     */
    void sparse_to_dense_A(const A_t *const A_src, const int n, real *A_dst);

    /**
     * Copy the sparse matrix G_SRC into the dense matrix G_DST.
     *
     * @param g_src Sparse matrix.
     * @param g_dst Dense matrix.
     */
    void sparse_to_dense_g(const g_t *const g_src, real g_dst[]);

    /**
     * Print every sub-matrix (shared and private) of every thread in a
     * prettified manner.
     *
     */
    void print_data_prettified();
};

}   // namespace gmx