#pragma once

#include "gromacs/utility/real.h"

#include "graph.h"

/* MOLECULE

	 Molecules structures and routines.

	 Main author:
	 Carl Christian Kjelgaard Mikkelsen,
	 Department of Computing Science and HPC2N
	 Umeaa University
	 Sweden
	 Email: spock@cs.umu.se

	 Additional programming by:
	 Jesús Alastruey Benedé
	 Departamento de Informática e Ingeniería de Sistemas
	 Universidad de Zaragoza
	 España
	 Email: jalastru@unizar.es

	 Additional programming by:
	 Lorién López-Villellas
	 Departamento de Informática e Ingeniería de Sistemas
	 Universidad de Zaragoza
	 España
	 Email: lorien.lopez@unizar.es

*/

// Side-chain types.
typedef enum side_chain_type {
    // Special
    GLYCINE,
    PROLINE,

    // Sulfur-containing
    CYSTEINE,
    METHIONINE,

    // Aliphatic
    ALALINE,
    VALINE,
    LEUCINE,
    ISOLEUCINE,

    // Acid
    ASPARTIC_ACID,
    GLUTAMIC_ACID,

    // Amides
    ASPARAGINE,
    GLUTAMINE,

    // Basic
    HISTIDINE,
    LYSINE,
    ARGININE,

    // Alcohols
    SERINE,

    // Aromatic
    PHENYLALANINE,
    TYROSINE,
    TRYPTOPHAN,
    THREONINE,

    NUM_TYPES
} side_chain_type_t;

typedef enum element {
    HYDROGEN = 1,
    HELIUM = 2,
    LITHIUM = 3,
    BERYLLIUM = 4,
    BORON = 5,
    CARBON = 6,
    NITROGEN = 7,
    OXYGEN = 8,
    FLUORINE = 9,
    NEON = 10,
    SODIUM = 11,
    MAGNESIUM = 12,
    ALUMINUM = 13,
    SILICON = 14,
    PHOSPHORUS = 15,
    SULFUR = 16,
    CHLORINE = 17,
    ARGON = 18,
    POTASSIUM = 19,
    CALCIUM = 20,
    SCANDIUM = 21,
    TITANIUM = 22,
    VANADIUM = 23,
    CHROMIUM = 24,
    MANGANESE = 25,
    IRON = 26,
    COBALT = 27,
    NICKEL = 28,
    COPPER = 29,
    ZINC = 30,
    GALLIUM = 31,
    GERMANIUM = 32,
    ARSENIC = 33,
    SELENIUM = 34,
    BROMINE = 35,
    KRYPTON = 36,
    RUBIDIUM = 37,
    STRONTIUM = 38,
    YTTRIUM = 39,
    ZIRCONIUM = 40,
    NIOBIUM = 41,
    MOLYBDENUM = 42,
    TECHNETIUM = 43,
    RUTHENIUM = 44,
    RHODIUM = 45,
    PALLADIUM = 46,
    SILVER = 47,
    CADMIUM = 48,
    INDIUM = 49,
    TIN = 50,
    ANTIMONY = 51,
    TELLURIUM = 52,
    IODINE = 53,
    XENON = 54,
    CESIUM = 55,
    BARIUM = 56,
    LANTHANUM = 57,
    CERIUM = 58,
    PRASEODYMIUM = 59,
    NEODYMIUM = 60,
    PROMETHIUM = 61,
    SAMARIUM = 62,
    EUROPIUM = 63,
    GADOLINIUM = 64,
    TERBIUM = 65,
    DYSPROSIUM = 66,
    HOLMIUM = 67,
    ERBIUM = 68,
    THULIUM = 69,
    YTTERBIUM = 70,
    LUTETIUM = 71,
    HAFNIUM = 72,
    TANTALUM = 73,
    TUNGSTEN = 74,
    RHENIUM = 75,
    OSMIUM = 76,
    IRIDIUM = 77,
    PLATINUM = 78,
    GOLD = 79,
    MERCURY = 80,
    THALLIUM = 81,
    LEAD = 82,
    BISMUTH = 83,
    POLONIUM = 84,
    ASTATINE = 85,
    RADON = 86,
    FRANCIUM = 87,
    RADIUM = 88,
    ACTINIUM = 89,
    THORIUM = 90,
    PROTACTINIUM = 91,
    URANIUM = 92,
    NEPTUNIUM = 93,
    PLUTONIUM = 94,
    AMERICIUM = 95,
    CURIUM = 96,
    BERKELIUM = 97,
    CALIFORNIUM = 98,
    EINSTEINIUM = 99,
    FERMIUM = 100,
    MENDELEVIUM = 101,
    NOBELIUM = 102,
    LAWRENCIUM = 103,
    RUTHERFORDIUM = 104,
    DUBNIUM = 105,
    SEABORGIUM = 106,
    BOHRIUM = 107,
    HASSIUM = 108,
    MEITNERIUM = 109,
    DARMSTADTIUM = 110,
    ROENTGENIUM = 111,
    UNUNBIIUM = 112,

    NUM_ELEMENTS = 112
} element_t;

typedef enum ends_type {
    TYPE_A,  // two oxygens left and three hydrogens right
    TYPE_B   // two oxygens one hydrogens left two hydrogens right
} ends_type_t;

// Name of each element.
extern const char *elements[NUM_ELEMENTS];

// Name of each side chain type.
extern const char *side_chain_types[NUM_TYPES];

// Number of atoms of each side chain type.
extern const int side_chain_atoms[NUM_TYPES];

// Number of bonds of each side chain type.
extern const int side_chain_bonds[NUM_TYPES];

/* This structure contains all the information which can be derived from the
	 bond list and the list of masses. As soon as x, and y are given, this is
	 the information you need to compute the matrix A = A(x,y) rapidly with
	 monotone access of the main memory.

	 You will only need one structure for each type of molecule present in your
	 simulation.
*/
typedef struct molecule {
    char *name;             // the name of the molecule
    char *abb;              // abbreviation of the name
    int m;                  // The number of atoms
    int n;                  // number of bonds
    element_t *atoms;       // the element of each atom
    real *invmass;          // invmass[i] is 1/mass of the ith atom
    int *bonds;             // sequential bond list similar to GROMACS
    // TODO define this better
    real *sigmaA;            // sequential list of bond lengths A
    real *sigmaB;            // sequential list of bond lengths B
    real *sigma2;           // sequential list of the square of bond lengths A
    //
    graph_t *atomic_graph;  // the atomic graph of the molecule.
    graph_t *bond_graph;    // the line graph of the atomic graph
    graph_t *chol_graph;    // the adjacency graph of the Cholesky factor
    real *weights;          // weights used to compute the entries of A(x,y)

    // Only peptide-chains

    int num_side_chains;    // number of side chains
    int n_sc;               // number of bonds that belong to side-chains
    side_chain_type_t *side_chain_types; // type of each side chain

    ends_type_t ends_type;  // Structure of the ends of the molecule.

    /**
     * A constructor that sets every pointer to null, to make this struct
     * easier to manage. TODO: This struct requieres a very heavy refactor
     * to transform it to OOO C++.
     */
    molecule();

    /**
     * Free dynamically allocated data.
     *
     */
    ~molecule();
} molecule_t;

// Reads the description of a molecule from a text file
void read_molecule_file(molecule_t *mol, const char *filename);

/**
 * Renumber the bonds of MOL to match the numbering used by IlvesPeptides.
 *
 * @param mol Molecule structure.
 */
void ilves_peptides_reorder(molecule_t *const mol);

//----------------------------------------------------------------------------
/* Renumber the bonds (list/graph) using a given permutation.

    There is a important point to consider here:

    Do we or do we not allow subroutines to change pointer values when we
    are simply updating the information which they target?

    Here it would have been easy to save a 2*n memory operations by simply
    NOT copying the bond list into the auxiliary list and simply changing
    the value of the pointer mol->bond_graph!

    However, CCKM finds that it is far safer not to change this pointer.
    Suppose the user had created another pointer to the bond graph before
    calling renumber_bonds. This pointer **might** be invalid on the return
    from renumber_bonds.

    Tentative guiding principles:

        1) If the target size is unchanged, then preserve the pointer
        2) If the target size might change, then do not preserve the pointer.

    WARNING: There is only limited check of the sanity of the input.
*/
void renumber_bonds(molecule_t *mol, int *p);

// Builds the bond graph from the bond list
void make_bond_graph(molecule_t *mol);

// Computes auxiliary weights needed for the construction of A
void make_weights(molecule_t *mol, graph_t *graph);
