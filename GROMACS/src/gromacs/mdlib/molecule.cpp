#include "molecule.h"

#include "list.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LINE_LEN 4096

// Name of each side-chain type.
const char *side_chain_types[] = {
    "Glycine", "Proline", "Cysteine", "Methionine", "Alaline", "Valine",
    "Leucine", "Isoleucine", "Aspartic acid", "Glutamic acid", "Asparagine",
    "Glutamine", "Histidine", "Lysine", "Arginine", "Serine", "Phenylalanine",
    "Tyrosine", "Tryptophan", "Threonine",
};

// Name of each side-chain type.
const char *elements[NUM_ELEMENTS] = {
    "Hydrogen", "Helium", "Lithium", "Beryllium", "Boron", "Carbon",
    "Nitrogen", "Oxygen", "Fluorine", "Neon", "Sodium", "Magnesium",
    "Aluminum", "Silicon", "PhosphoruS", "Sulfur", "Chlorine", "Argon",
    "Potassium", "Calcium", "Scandium", "Titanium", "Vanadium", "Chromium",
    "Manganese", "Iron", "Cobalt", "Nickel", "Copper", "Zinc", "Gallium",
    "Germanium", "Arsenic", "Selenium", "Bromine", "Krypton", "Rubidium",
    "Strontium", "Yttrium", "Zirconium", "Niobium", "MolybdenuM", "TechnetiuM",
    "Ruthenium", "Rhodium", "Palladium", "Silver", "Cadmium", "Indium", "Tin",
    "Antimony", "Tellurium", "Iodine", "Xenon", "Cesium", "Barium", "Lanthanum",
    "Cerium", "Praseodymium", "Neodymium", "PromethiuM", "Samarium", "Europium",
    "GadoliniuM", "Terbium", "DysprosiuM", "Holmium", "Erbium", "Thulium",
    "Ytterbium", "Lutetium", "Hafnium", "Tantalum", "Tungsten", "Rhenium",
    "Osmium", "Iridium", "Platinum", "Gold", "Mercury", "Thallium", "Lead",
    "Bismuth", "Polonium", "Astatine", "Radon", "Francium", "Radium", "Actinium",
    "Thorium", "Protactinium", "Uranium", "Neptunium", "Plutonium", "Americium",
    "Curium", "Berkelium", "Californium", "Einsteinium", "Fermium", "Mendelevium",
    "Nobelium", "LawrenciuM", "RutherforDium", "Dubnium", "SeaborgiuM", "Bohrium",
    "Hassium", "MeitneriuM", "Darmstadtium", "Roentgenium", "Ununbiium",
};

// Number of atoms of each side-chain type.
const int side_chain_atoms[] = {
    // Special
    1, // GLYCINE
    9, // PROLINE

    // Sulfur-containing
    5, // CYSTEINE
    11, // METHIONINE

    // Aliphatic
    4, // ALALINE
    10, // VALINE
    13, // LEUCINE
    13, // ISOLEUCINE

    // Acid
    6, // ASPARTIC_ACID
    9, // GLUTAMIC_ACID

    // Amides
    8, // ASPARAGINE
    11, // GLUTAMINE

    // Basic
    11, // HISTIDINE
    16, // LYSINE
    18, // ARGININE

    // Alcohols
    5, // SERINE

    // Aromatic
    14, // PHENYLALANINE
    15, // TYROSINE
    19, // TRYPTOPHAN
    8, // THREONINE
};

// Number of bonds of each side-chain type.
const int side_chain_bonds[] = {
    // Special
    0, // GLYCINE
    9, // PROLINE

    // Sulfur-containing
    4, // CYSTEINE
    10, // METHIONINE

    // Aliphatic
    3, // ALALINE
    9, // VALINE
    12, // LEUCINE
    12, // ISOLEUCINE

    // Acid
    5, // ASPARTIC_ACID
    8, // GLUTAMIC_ACID

    // Amides
    7, // ASPARAGINE
    10, // GLUTAMINE

    // Basic
    11, // HISTIDINE
    15, // LYSINE
    17, // ARGININE

    // Alcohols
    4, // SERINE

    // Aromatic
    14, // PHENYLALANINE
    15, // TYROSINE
    19, // TRYPTOPHAN
    7 // THREONINE
};

// typedef enum bond_type {
//     HYDROGEN_HYDROGEN,
//     HYDROGEN_CARBON,
//     HYDROGEN_NITROGEN,
//     HYDROGEN_OXYGEN,
//     HYDROGEN_SULFUR,
//     CARBON_CARBON,
//     CARBON_NITROGEN,
//     CARBON_OXYGEN,
//     CARBON_SULFUR,
//     NITROGEN
//     OXYGEN
//     SULFUR
// }

// const

molecule::molecule() {
    name = nullptr;
    abb = nullptr;
    atoms = nullptr;
    invmass = nullptr;
    bonds = nullptr;
    sigmaA = nullptr;
    sigmaB = nullptr;
    sigma2 = nullptr;
    atomic_graph = nullptr;
    bond_graph = nullptr;
    chol_graph = nullptr;
    weights = nullptr;
    side_chain_types = nullptr;
}

molecule::~molecule() {
    free(name);
    free(abb);
    free(atoms);
    free(invmass);
    free(bonds);
    free(sigmaA);
    free(sigmaB);
    free(sigma2);
    if (atomic_graph) {
        graph_free(atomic_graph);
        free(atomic_graph);
    }
    if (bond_graph) {
        graph_free(bond_graph);
        free(bond_graph);
    }
    if (chol_graph) {
        graph_free(chol_graph);
        free(chol_graph);
    }
    free(weights);
    free(this->side_chain_types);
}

/**
 * Get the neighbour bonds of BOND.
 *
 * @param mol Molecule structure.
 * @param bond Bond index.
 * @return A linked list that contains the neighbour bonds of BOND
 * (NULL if there is none).
 */
static my_node_t *get_neighbours(const molecule_t *const mol, const int bond) {
    my_node_t *root = NULL;

    for (int j = mol->bond_graph->xadj[bond];
            j < mol->bond_graph->xadj[bond + 1]; ++j) {

        // Determine bond of the jth entry.
        int m = mol->bond_graph->adj[j];

        if (m == bond) {
            continue;
        }

        sinsert(m, &root);
    }

    return root;
}

/**
 * Get the neighbour bonds of NEW_BOND that are not neighbours of OLD_BOND.
 *
 * @param mol Molecule structure.
 * @param new_bond New bond index.
 * @param old_bond Old bond index.
 * @return A linked list that contains the neighbour bonds of NEW_BOND that
 * are not neighbours of OLD_BOND (NULL if there is none).
 */
static my_node_t *get_new_neighbours(const molecule_t *const mol, const int new_bond,
                                     const int old_bond) {

    my_node_t *neighbours_of_new = get_neighbours(mol, new_bond);
    my_node_t *neighbours_of_old = get_neighbours(mol, old_bond);

    sdelete(old_bond, &neighbours_of_new);

    my_node_t *new_neighbours = neighbours_of_new;
    my_node_t *old_neighbours = neighbours_of_old;

    // The linked list is ordered.
    while (new_neighbours != NULL && old_neighbours != NULL) {
        if (new_neighbours->number == old_neighbours->number) {
            sdelete(new_neighbours->number, &neighbours_of_new);

            new_neighbours = new_neighbours->next;
            old_neighbours = old_neighbours->next;
        }
        else if (new_neighbours->number < old_neighbours->number) {
            new_neighbours = new_neighbours->next;
        }
        else {
            old_neighbours = old_neighbours->next;
        }
    }

    free_list(neighbours_of_old);

    return neighbours_of_new;
}

/**
 * Get the neighbour bonds of BOND that connect one atom of element ELEMENT_A
 * with another atom of ELEMENT_B.
 *
 * @param mol Molecule structure.
 * @param bond Bond index.
 * @param element_a Element of one atom of the neighbour bonds.
 * @param element_b Element of one atom of the neighbour bonds.
 * @return A linked list that contains the neighbour bonds of BONDthat connect
 * one atom of element ELEMENT_A with another atom of ELEMENT_B (NULL if there
 * is none).
 */
static my_node_t *get_neighbours_given_elements(const molecule_t *const mol,
        const int bond,
        const element_t element_a,
        const element_t element_b) {

    my_node_t *root = NULL;

    for (int j = mol->bond_graph->xadj[bond];
            j < mol->bond_graph->xadj[bond + 1]; ++j) {

        // Determine bond of the jth entry.
        int m = mol->bond_graph->adj[j];

        if (m == bond) {
            continue;
        }

        int c = mol->bonds[2 * m + 0]; // Atom c of the bond.
        int d = mol->bonds[2 * m + 1]; // Atom d of the bond.

        element_t element_c = mol->atoms[c];
        element_t element_d = mol->atoms[d];

        if ((element_c == element_a && element_d == element_b) ||
                (element_d == element_a && element_c == element_b)) {
            sinsert(m, &root);
        }
    }

    return root;
}

/**
 * Get the neighbour bonds of NEW_BOND that are not neighbours of OLD_BOND and
 * connect one atom of element ELEMENT_A with another atom of ELEMENT_B.
 *
 * @param mol Molecule structure.
 * @param new_bond New bond index.
 * @param old_bond Old bond index.
 * @param element_a Element of one atom of the neighbour bonds.
 * @param element_b Element of one atom of the neighbour bonds.
 * @return A linked list that contains the neighbour bonds of BONDthat connect
 * one atom of element ELEMENT_A with another atom of ELEMENT_B (NULL if there
 * is none).
 * Get the neighbour bonds of  .
 */
static my_node_t *get_new_neighbours_given_elements(const molecule_t *const mol,
        const int new_bond,
        const int old_bond,
        const element_t element_a,
        const element_t element_b) {

    my_node_t *neighbours_of_new = get_neighbours_given_elements(mol, new_bond, element_a, element_b);
    my_node_t *neighbours_of_old = get_neighbours_given_elements(mol, old_bond, element_a, element_b);

    sdelete(old_bond, &neighbours_of_new);

    my_node_t *new_neighbours = neighbours_of_new;
    my_node_t *old_neighbours = neighbours_of_old;

    // The linked list is ordered.
    while (new_neighbours != NULL && old_neighbours != NULL) {
        if (new_neighbours->number == old_neighbours->number) {
            int number = new_neighbours->number;
            new_neighbours = new_neighbours->next;

            sdelete(number, &neighbours_of_new);

            old_neighbours = old_neighbours->next;
        }
        else if (new_neighbours->number < old_neighbours->number) {
            new_neighbours = new_neighbours->next;
        }
        else {
            old_neighbours = old_neighbours->next;
        }
    }

    free_list(neighbours_of_old);

    return neighbours_of_new;
}

/**
 * Identifies the side-chain type of the first side-chain connected with
 * BB_CC_BOND.
 *
 * @param mol Molecule structure.
 * @param bb_CC_bond The bond of the backbone that connects one Carbon atom with
 * another (each chunk of the backbone has one C-C bond).
 * @return the side-chain type of the first side-chain connected with BB_CC_BOND to match the
 * numbering used by ILVES.
 */
static side_chain_type_t identify_side_chain(const molecule_t *const mol, const int bb_CC_bond) {
    // Find the connection with the side-chain.
    int pivot = bb_CC_bond;
    my_node_t *CC_neigh = get_neighbours_given_elements(mol, bb_CC_bond, CARBON, CARBON);

    // There is no C-C bond -> GLYCINE side-chain.
    if (CC_neigh == NULL) {
        free_list(CC_neigh);

        return GLYCINE;
    }

    int prev_pivot = pivot;
    pivot = CC_neigh->number;

    free_list(CC_neigh);

    // Find C-S neighbours.
    my_node_t *CS_neigh = get_new_neighbours_given_elements(mol, pivot, prev_pivot, CARBON, SULFUR);
    int CS_neigh_count = count(CS_neigh);
    free_list(CS_neigh);

    // Connection with bond C-S -> CYSTEINE
    if (CS_neigh_count == 1) {
        return CYSTEINE;
    }

    // Find C-N neighbours
    my_node_t *CO_neigh = get_new_neighbours_given_elements(mol, pivot, prev_pivot, CARBON, OXYGEN);
    int CO_neigh_count = count(CO_neigh);
    free_list(CO_neigh);

    // Find C-C neighbours
    CC_neigh = get_new_neighbours_given_elements(mol, pivot, prev_pivot, CARBON, CARBON);
    int CC_neigh_count = count(CC_neigh);

    // 1 C-O neighbour, 0 C-C neighbours -> SERINE.
    if (CO_neigh_count == 1 && CC_neigh_count == 0) {
        free_list(CC_neigh);
        return SERINE;
    }
    // 1 C-N neighbour, 1 C-C neighbours -> THERIONINE.
    else if (CO_neigh_count == 1 && CC_neigh_count == 1) {
        free_list(CC_neigh);
        return THREONINE;
    }
    // No C-C neighbour -> ALALINE
    else if (CC_neigh == NULL) {
        free_list(CC_neigh);
        return ALALINE;
    }
    else if (CC_neigh_count == 2) { // Valine or Isoleucine.

        int CC_bond_right = CC_neigh->number;
        int CC_bond_left = CC_neigh->next->number;

        free_list(CC_neigh);

        my_node_t *CC_neigh_right = get_new_neighbours_given_elements(mol, CC_bond_right, pivot, CARBON, CARBON);
        my_node_t *CC_neigh_left = get_new_neighbours_given_elements(mol, CC_bond_left, pivot, CARBON, CARBON);

        int CC_neigh_right_count = count(CC_neigh_right);
        int CC_neigh_left_count = count(CC_neigh_left);

        free_list(CC_neigh_right);
        free_list(CC_neigh_left);

        // Connection with other C-C (left of right) -> ISOLEUCINE
        if (CC_neigh_right_count == 1 || CC_neigh_left_count == 1) {
            return ISOLEUCINE;
        }
        else {
            return VALINE;
        }
    }

    prev_pivot = pivot;
    pivot = CC_neigh->number;
    free_list(CC_neigh);

    CO_neigh = get_new_neighbours_given_elements(mol, pivot, prev_pivot, CARBON, OXYGEN);
    CO_neigh_count = count(CO_neigh);
    free_list(CO_neigh);

    my_node_t *CN_neigh = get_new_neighbours_given_elements(mol, pivot, prev_pivot, CARBON, NITROGEN);
    int CN_neigh_count = count(CN_neigh);
    free_list(CN_neigh);

    CS_neigh = get_new_neighbours_given_elements(mol, pivot, prev_pivot, CARBON, SULFUR);
    CS_neigh_count = count(CS_neigh);
    free_list(CS_neigh);

    // Connection with 2 C-O -> ASPARTIC ACID.
    if (CO_neigh_count == 2) {
        return ASPARTIC_ACID;
    }
    // Connection with 1 C-O -> ASPARAGINE.
    else if (CO_neigh_count == 1) {
        return ASPARAGINE;
    }
    // Connection with 1 C-N -> HISTIDINE.
    else if (CN_neigh_count == 1) {
        return HISTIDINE;
    }
    // Connection with 1 C-S -> METHIONINE/
    else if (CS_neigh_count == 1) {
        return METHIONINE;
    }

    CC_neigh = get_new_neighbours_given_elements(mol, pivot, prev_pivot, CARBON, CARBON);
    CC_neigh_count = count(CC_neigh);

    if (CC_neigh_count == 1) { // Proline, Glutamic acid, Glutamine, Lysine, Arginine
        prev_pivot = pivot;
        pivot = CC_neigh->number;

        free_list(CC_neigh);

        CO_neigh = get_new_neighbours_given_elements(mol, pivot, prev_pivot, CARBON, OXYGEN);
        int CO_neigh_count = count(CO_neigh);
        free_list(CO_neigh);

        CN_neigh = get_new_neighbours_given_elements(mol, pivot, prev_pivot, CARBON, NITROGEN);
        int CN_neigh_count = count(CN_neigh);

        if (CN_neigh != NULL) {
            prev_pivot = pivot;
            pivot = CN_neigh->number;

            free_list(CN_neigh);
        }

        // Connection with 2 C-O -> GLUTAMIC ACID.
        if (CO_neigh_count == 2) {
            return GLUTAMIC_ACID;
        }
        // Connection with 1 C-O -> GLUTAMINE.
        else if (CO_neigh_count == 1) {
            return GLUTAMINE;
        }
        // Connection with 0 C-N -> LYSINE.
        else if (CN_neigh_count == 0) {
            return LYSINE;
        }

        // Else -> ARGININE or PROLINE

        CN_neigh = get_new_neighbours_given_elements(mol, pivot, prev_pivot, CARBON, NITROGEN);
        prev_pivot = pivot;
        pivot = CN_neigh->number;
        free_list(CN_neigh);

        CN_neigh = get_new_neighbours_given_elements(mol, pivot, prev_pivot, CARBON, NITROGEN);
        CN_neigh_count = count(CN_neigh);
        free_list(CN_neigh);

        // Connection with 1 C-N -> ARGININE.
        if (CN_neigh_count == 2) {
            return ARGININE;
        }
        // Else -> PROLINE.
        else {
            return PROLINE;
        }
    }
    else { // Leucine, Phenylaline, Tyrosie, Tryptophan
        int CC_bond_right = CC_neigh->number;
        int CC_bond_left = CC_neigh->next->number;

        free_list(CC_neigh);

        my_node_t *CN_neigh_right = get_new_neighbours_given_elements(mol,
                                    CC_bond_right, pivot, CARBON, NITROGEN);
        my_node_t *CN_neigh_left = get_new_neighbours_given_elements(mol,
                                   CC_bond_left, pivot, CARBON, NITROGEN);

        int CN_neigh_right_count = count(CN_neigh_right);
        int CN_neigh_left_count = count(CN_neigh_left);

        free_list(CN_neigh_right);
        free_list(CN_neigh_left);

        // Connection with 1 C-N -> TRYPTOPHAN.
        if (CN_neigh_left_count == 1 || CN_neigh_right_count == 1) {
            return TRYPTOPHAN;
        }

        prev_pivot = pivot;
        pivot = CC_bond_right;

        my_node_t *CH_neigh = get_new_neighbours_given_elements(mol, pivot, prev_pivot, CARBON, HYDROGEN);
        int CH_neigh_count = count(CH_neigh);
        free_list(CH_neigh);

        // Connection with 3 C-H -> LUCINE.
        if (CH_neigh_count == 3) {
            return LEUCINE;
        }

        CC_neigh = get_new_neighbours_given_elements(mol, pivot, prev_pivot, CARBON, CARBON);

        if (CC_neigh == NULL) {
            fprintf(stderr, "ERROR while reordering the bonds\n");
            exit(1);
        }

        prev_pivot = pivot;
        pivot = CC_neigh->number;
        free_list(CC_neigh);

        CC_neigh = get_new_neighbours_given_elements(mol, pivot, prev_pivot, CARBON, CARBON);

        if (CC_neigh == NULL) {
            fprintf(stderr, "ERROR while reordering the bonds\n");
            exit(1);
        }

        prev_pivot = pivot;
        pivot = CC_neigh->number;
        free_list(CC_neigh);

        CO_neigh = get_new_neighbours_given_elements(mol, pivot, prev_pivot, CARBON, OXYGEN);
        int CO_neigh_count = count(CO_neigh);
        free_list(CO_neigh);

        if (CO_neigh_count == 1) {
            return TYROSINE;
        }
        else {
            return PHENYLALANINE;
        }
    }
}

/**
 * Renumbers the Glycine side-chain connected with BB_CC_BOND to match the
 * numbering used by ILVES.
 *
 * @param mol Molecule structure.
 * @param bb_CC_bond The bond of the backbone that connects one Carbon atom with
 * another (each chunk of the backbone has one C-C bond).
 * @param bond_counter First bond index of the new side-chain numbering. When
 * returning it will be its initial value plus the number of bonds of the side-chain.
 * @param permutation Permutation to apply to the numbering of the bonds of
 * the molecule. If you want to change the index of the current bond 0 to
 * 5 -> permutation[5] = 0.
 *
 */
static void ilves_peptides_reorder_glycine_side_chain(__attribute__((unused)) const molecule_t *const mol,
        __attribute__((unused)) const int bb_CC_bond,
        __attribute__((unused)) int *const bond_counter,
        __attribute__((unused)) int *const permutation) {
}

/**
 * Renumbers the Proline side-chain connected with BB_CC_BOND to match the
 * numbering used by ILVES.
 *
 * @param mol Molecule structure.
 * @param bb_CC_bond The bond of the backbone that connects one Carbon atom with
 * another (each chunk of the backbone has one C-C bond).
 * @param bond_counter First bond index of the new side-chain numbering. When
 * returning it will be its initial value plus the number of bonds of the side-chain.
 * @param permutation Permutation to apply to the numbering of the bonds of
 * the molecule. If you want to change the index of the current bond 0 to
 * 5 -> permutation[5] = 0.
 *
 */
static void ilves_peptides_reorder_proline_side_chain(const molecule_t *const mol,
        const int bb_CC_bond,
        int *const bond_counter,
        int *const permutation) {
    /*
    proline side-chain (bonds):
          bb_sc_bond
             |
     H - 0 - C - 1 - H
             |
             6
             |
     H - 2 - C - 3 - H
             |
             7
             |
     H - 4 - C - 5 - H
             |
             8
             |
    proline_backbone_atom
    */

    // Find the connection with the backbone.
    my_node_t *CC_neigh = get_neighbours_given_elements(mol, bb_CC_bond, CARBON, CARBON);
    int prev_pivot = bb_CC_bond;
    int pivot = CC_neigh->number;
    free_list(CC_neigh);

    int CH_counter = 0;
    int CC_counter = 6;

    for (int i = 0; i < 3; ++i, CH_counter += 2, CC_counter += 1) {
        my_node_t *CH_neigh = get_new_neighbours_given_elements(mol, pivot,
                              prev_pivot, CARBON, HYDROGEN);

        my_node_t *it = CH_neigh;
        while (it != NULL) {
            it = it -> next;
        }

        // 0
        permutation[*bond_counter + CH_counter] = CH_neigh->number;
        // 1
        permutation[*bond_counter + CH_counter + 1] = CH_neigh->next->number;

        free_list(CH_neigh);

        my_node_t *CCN_neigh = get_new_neighbours_given_elements(mol, pivot,
                               prev_pivot, CARBON, i == 2 ? NITROGEN : CARBON);

        // 6
        permutation[*bond_counter + CC_counter] = CCN_neigh->number;

        prev_pivot = pivot;
        pivot = CCN_neigh->number;

        free_list(CCN_neigh);
    }
    *bond_counter += 9;
}

/**
 * Renumbers the Cysteine side-chain connected with BB_CC_BOND to match the
 * numbering used by ILVES.
 *
 * @param mol Molecule structure.
 * @param bb_CC_bond The bond of the backbone that connects one Carbon atom with
 * another (each chunk of the backbone has one C-C bond).
 * @param bond_counter First bond index of the new side-chain numbering. When
 * returning it will be its initial value plus the number of bonds of the side-chain.
 * @param permutation Permutation to apply to the numbering of the bonds of
 * the molecule. If you want to change the index of the current bond 0 to
 * 5 -> permutation[5] = 0.
 *
 */
static void ilves_peptides_reorder_cysteine_side_chain(const molecule_t *const mol,
        const int bb_CC_bond,
        int *const bond_counter,
        int *const permutation) {
    /*
    Cysteine side-chain (bonds):
          Backbone
             |
     H - 2 - C - 3 - H
             |
             1
             |
             S
             |
             0
             |
             H
    */

    // Find the connection with the backbone.
    my_node_t *CC_neigh = get_neighbours_given_elements(mol, bb_CC_bond, CARBON, CARBON);
    int prev_pivot = bb_CC_bond;
    int pivot = CC_neigh->number;
    free_list(CC_neigh);

    my_node_t *CH_neigh = get_new_neighbours_given_elements(mol, pivot,
                          prev_pivot, CARBON, HYDROGEN);

    // 2
    permutation[*bond_counter + 2] = CH_neigh->number;
    // 3
    permutation[*bond_counter + 3] = CH_neigh->next->number;

    free_list(CH_neigh);

    my_node_t *CS_neigh = get_new_neighbours_given_elements(mol, pivot,
                          prev_pivot, SULFUR, CARBON);

    // 1
    permutation[*bond_counter + 1] = CS_neigh->number;

    prev_pivot = pivot;
    pivot = CS_neigh->number;

    free_list(CS_neigh);

    my_node_t *SH_neigh = get_new_neighbours_given_elements(mol, pivot,
                          prev_pivot, SULFUR, HYDROGEN);

    // 0
    permutation[*bond_counter] = SH_neigh->number;

    free_list(SH_neigh);

    *bond_counter += 4;
}

/**
 * Renumbers the Methionine side-chain connected with BB_CC_BOND to match the
 * numbering used by ILVES.
 *
 * @param mol Molecule structure.
 * @param bb_CC_bond The bond of the backbone that connects one Carbon atom with
 * another (each chunk of the backbone has one C-C bond).
 * @param bond_counter First bond index of the new side-chain numbering. When
 * returning it will be its initial value plus the number of bonds of the side-chain.
 * @param permutation Permutation to apply to the numbering of the bonds of
 * the molecule. If you want to change the index of the current bond 0 to
 * 5 -> permutation[5] = 0.
 *
 */
static void ilves_peptides_reorder_methionine_side_chain(const molecule_t *const mol,
        const int bb_CC_bond,
        int *const bond_counter,
        int *const permutation) {

    /*
    Methiodine side-chain (bonds):
          Backbone
             |
     H - 8 - C - 9 - H
             |
             7
             |
     H - 5 - C - 6 - H
             |
             4
             |
             S
             |
             3
             |
     H - 1 - C - 2 - H
             |
             0
             |
             H
    */

    // Find the connection with the backbone.
    my_node_t *CC_neigh = get_neighbours_given_elements(mol, bb_CC_bond, CARBON, CARBON);
    int prev_pivot = bb_CC_bond;
    int pivot = CC_neigh->number;
    free_list(CC_neigh);

    my_node_t *CH_neigh = get_new_neighbours_given_elements(mol, pivot,
                          prev_pivot, CARBON, HYDROGEN);

    // 8
    permutation[*bond_counter + 8] = CH_neigh->number;
    // 9
    permutation[*bond_counter + 9] = CH_neigh->next->number;

    free_list(CH_neigh);

    CC_neigh = get_new_neighbours_given_elements(mol, pivot,
               prev_pivot, CARBON, CARBON);

    // 7
    permutation[*bond_counter + 7] = CC_neigh->number;

    prev_pivot = pivot;
    pivot = CC_neigh->number;

    free_list(CC_neigh);

    CH_neigh = get_new_neighbours_given_elements(mol, pivot,
               prev_pivot, CARBON, HYDROGEN);

    // 6
    permutation[*bond_counter + 6] = CH_neigh->number;
    // 5
    permutation[*bond_counter + 5] = CH_neigh->next->number;

    free_list(CH_neigh);

    my_node_t *CS_neigh = get_new_neighbours_given_elements(mol, pivot,
                          prev_pivot, CARBON, SULFUR);

    // 4
    permutation[*bond_counter + 4] = CS_neigh->number;

    prev_pivot = pivot;
    pivot = CS_neigh->number;

    free_list(CS_neigh);

    CS_neigh = get_new_neighbours_given_elements(mol, pivot,
               prev_pivot, CARBON, SULFUR);

    // 3
    permutation[*bond_counter + 3] = CS_neigh->number;

    prev_pivot = pivot;
    pivot = CS_neigh->number;

    free_list(CS_neigh);

    CH_neigh = get_new_neighbours_given_elements(mol, pivot,
               prev_pivot, CARBON, HYDROGEN);

    // 2
    permutation[*bond_counter + 2] = CH_neigh->number;
    // 1
    permutation[*bond_counter + 1] = CH_neigh->next->number;
    // 0
    permutation[*bond_counter + 0] = CH_neigh->next->next->number;

    free_list(CH_neigh);

    *bond_counter += 10;
}

/**
 * Renumbers the Alaline side-chain connected with BB_CC_BOND to match the
 * numbering used by ILVES.
 *
 * @param mol Molecule structure.
 * @param bb_CC_bond The bond of the backbone that connects one Carbon atom with
 * another (each chunk of the backbone has one C-C bond).
 * @param bond_counter First bond index of the new side-chain numbering. When
 * returning it will be its initial value plus the number of bonds of the side-chain.
 * @param permutation Permutation to apply to the numbering of the bonds of
 * the molecule. If you want to change the index of the current bond 0 to
 * 5 -> permutation[5] = 0.
 *
 */
static void ilves_peptides_reorder_alaline_side_chain(const molecule_t *const mol,
        const int bb_CC_bond,
        int *const bond_counter,
        int *const permutation) {
    /*
    Alaline side-chain (bonds):
          Backbone
             |
     H - 1 - C - 2 - H
             |
             0
             |
             H
    */

    // Find the connection with the backbone.
    my_node_t *CC_neigh = get_neighbours_given_elements(mol, bb_CC_bond, CARBON, CARBON);
    int prev_pivot = bb_CC_bond;
    int pivot = CC_neigh->number;
    free_list(CC_neigh);

    my_node_t *CH_neigh = get_new_neighbours_given_elements(mol, pivot,
                          prev_pivot, CARBON, HYDROGEN);

    // 2
    permutation[*bond_counter + 2] = CH_neigh->number;
    // 1
    permutation[*bond_counter + 1] = CH_neigh->next->number;
    // 0
    permutation[*bond_counter + 0] = CH_neigh->next->next->number;

    free_list(CH_neigh);

    *bond_counter += 3;
}

/**
 * Renumbers the Valine side-chain connected with BB_CC_BOND to match the
 * numbering used by ILVES.
 *
 * @param mol Molecule structure.
 * @param bb_CC_bond The bond of the backbone that connects one Carbon atom with
 * another (each chunk of the backbone has one C-C bond).
 * @param bond_counter First bond index of the new side-chain numbering. When
 * returning it will be its initial value plus the number of bonds of the side-chain.
 * @param permutation Permutation to apply to the numbering of the bonds of
 * the molecule. If you want to change the index of the current bond 0 to
 * 5 -> permutation[5] = 0.
 *
 */
static void ilves_peptides_reorder_valine_side_chain(const molecule_t *const mol,
        const int bb_CC_bond,
        int *const bond_counter,
        int *const permutation) {
    /*
    Valine side-chain (bonds):
                Backbone
            H       |       H
            |       |       |
            2       |       6
            |       |       |
    H - 1 - C - 3 - C - 7 - C - 5 - H
            |       |       |
            0       8       4
            |       |       |
            H       H       H
    */

    // Find the connection with the backbone.
    my_node_t *CC_neigh = get_neighbours_given_elements(mol, bb_CC_bond, CARBON, CARBON);
    int prev_pivot = bb_CC_bond;
    int pivot = CC_neigh->number;
    free_list(CC_neigh);

    my_node_t *CH_neigh = get_new_neighbours_given_elements(mol, pivot,
                          prev_pivot, CARBON, HYDROGEN);

    // 8
    permutation[*bond_counter + 8] = CH_neigh->number;

    free_list(CH_neigh);

    CC_neigh = get_new_neighbours_given_elements(mol, pivot,
               prev_pivot, CARBON, CARBON);

    my_node_t *it = CC_neigh;
    int counter = 0;
    for (int i = 0; i < 2; ++i, it = it->next, counter += 4) {

        my_node_t *CH_neigh = get_new_neighbours_given_elements(mol, it->number,
                              pivot, CARBON, HYDROGEN);

        // 0
        permutation[*bond_counter + counter + 0] = CH_neigh->number;
        // 1
        permutation[*bond_counter + counter + 1] = CH_neigh->next->number;
        // 2
        permutation[*bond_counter + counter + 2] = CH_neigh->next->next->number;

        free_list(CH_neigh);

        // 3
        permutation[*bond_counter + counter + 3] = it->number;
    }

    free_list(CC_neigh);

    *bond_counter += 9;
}

/**
 * Renumbers the Leucine side-chain connected with BB_CC_BOND to match the
 * numbering used by ILVES.
 *
 * @param mol Molecule structure.
 * @param bb_CC_bond The bond of the backbone that connects one Carbon atom with
 * another (each chunk of the backbone has one C-C bond).
 * @param bond_counter First bond index of the new side-chain numbering. When
 * returning it will be its initial value plus the number of bonds of the side-chain.
 * @param permutation Permutation to apply to the numbering of the bonds of
 * the molecule. If you want to change the index of the current bond 0 to
 * 5 -> permutation[5] = 0.
 *
 */
static void ilves_peptides_reorder_leucine_side_chain(const molecule_t *const mol,
        const int bb_CC_bond,
        int *const bond_counter,
        int *const permutation) {
    /*
    Leucine side-chain (bonds):
                Backbone
                    |
            H -10 - C - 11- H
                    |
                    9
            H       |       H
            |       |       |
            2       |       6
            |       |       |
    H - 1 - C - 3 - C - 7 - C - 5 - H
            |       |       |
            0       8       4
            |       |       |
            H       H       H
    */

    // Find the connection with the backbone.
    my_node_t *CC_neigh = get_neighbours_given_elements(mol, bb_CC_bond, CARBON, CARBON);
    int prev_pivot = bb_CC_bond;
    int pivot = CC_neigh->number;
    free_list(CC_neigh);

    my_node_t *CH_neigh = get_new_neighbours_given_elements(mol, pivot,
                          prev_pivot, CARBON, HYDROGEN);

    // 11
    permutation[*bond_counter + 11] = CH_neigh->number;
    // 10
    permutation[*bond_counter + 10] = CH_neigh->next->number;

    free_list(CH_neigh);

    CC_neigh = get_new_neighbours_given_elements(mol, pivot, prev_pivot, CARBON, CARBON);

    prev_pivot = pivot;
    pivot = CC_neigh->number;

    // 9
    permutation[*bond_counter + 9] = CC_neigh->number;

    free_list(CC_neigh);

    CH_neigh = get_new_neighbours_given_elements(mol, pivot, prev_pivot,
               CARBON, HYDROGEN);

    // 8
    permutation[*bond_counter + 8] = CH_neigh->number;

    free_list(CH_neigh);

    CC_neigh = get_new_neighbours_given_elements(mol, pivot, prev_pivot,
               CARBON, CARBON);

    my_node_t *it = CC_neigh;
    int counter = 0;
    for (int i = 0; i < 2; ++i, it = it->next, counter += 4) {

        my_node_t *CH_neigh = get_new_neighbours_given_elements(mol, it->number,
                              pivot, CARBON, HYDROGEN);

        // 0
        permutation[*bond_counter + counter + 0] = CH_neigh->number;
        // 1
        permutation[*bond_counter + counter + 1] = CH_neigh->next->number;
        // 2
        permutation[*bond_counter + counter + 2] = CH_neigh->next->next->number;

        free_list(CH_neigh);

        // 3
        permutation[*bond_counter + counter + 3] = it->number;
    }

    free_list(CC_neigh);

    *bond_counter += 12;
}

/**
 * Renumbers the Isoleucine side-chain connected with BB_CC_BOND to match the
 * numbering used by ILVES.
 *
 * @param mol Molecule structure.
 * @param bb_CC_bond The bond of the backbone that connects one Carbon atom with
 * another (each chunk of the backbone has one C-C bond).
 * @param bond_counter First bond index of the new side-chain numbering. When
 * returning it will be its initial value plus the number of bonds of the side-chain.
 * @param permutation Permutation to apply to the numbering of the bonds of
 * the molecule. If you want to change the index of the current bond 0 to
 * 5 -> permutation[5] = 0.
 *
 */
static void ilves_peptides_reorder_isoleucine_side_chain(const molecule_t *const mol,
        const int bb_CC_bond,
        int *const bond_counter,
        int *const permutation) {
    /*
    Isoleucine side-chain (bonds):
                Backbone
                    |
            H       |       H       H
            |       |       |       |
            9       |       5       2
            |       |       |       |
    H - 8 - C - 10- C - 6 - C - 3 - C - 1 - H
            |       |       |       |
            7      11       4       0
            |       |       |       |
            H       H       H       H
    */

    // Find the connection with the backbone.
    my_node_t *CC_neigh = get_neighbours_given_elements(mol, bb_CC_bond, CARBON, CARBON);
    int prev_pivot = bb_CC_bond;
    int pivot = CC_neigh->number;
    free_list(CC_neigh);

    my_node_t *CH_neigh = get_new_neighbours_given_elements(mol, pivot, prev_pivot,
                          CARBON, HYDROGEN);

    // 11
    permutation[*bond_counter + 11] = CH_neigh->number;

    free_list(CH_neigh);

    CC_neigh = get_new_neighbours_given_elements(mol, pivot, prev_pivot,
               CARBON, CARBON);

    my_node_t *it = CC_neigh;
    for (int i = 0; i < 2; ++i, it = it->next) {
        my_node_t *CH_neigh = get_new_neighbours_given_elements(mol, it->number,
                              pivot, CARBON, HYDROGEN);
        int CH_count = count(CH_neigh);

        if (CH_count == 3) { // Left
            // 10
            permutation[*bond_counter + 10] = it->number;
            // 7
            permutation[*bond_counter + 7] = CH_neigh->number;
            // 8
            permutation[*bond_counter + 8] = CH_neigh->next->number;
            // 9
            permutation[*bond_counter + 9] = CH_neigh->next->next->number;
        }
        else { // Right
            // 6
            permutation[ *bond_counter + 6] = it->number;

            // 4
            permutation[*bond_counter + 4] = CH_neigh->number;
            // 5
            permutation[*bond_counter + 5] = CH_neigh->next->number;

            my_node_t *CC_neigh2 = get_new_neighbours_given_elements(mol, it->number,
                                   pivot, CARBON, CARBON);

            // 3
            permutation[*bond_counter + 3] = CC_neigh2->number;

            my_node_t *CH_neigh2 = get_new_neighbours_given_elements(mol, CC_neigh2->number,
                                   it->number, CARBON, HYDROGEN);

            // 0
            permutation[*bond_counter + 0] = CH_neigh2->number;
            // 1
            permutation[*bond_counter + 1] = CH_neigh2->next->number;
            // 2
            permutation[*bond_counter + 2] = CH_neigh2->next->next->number;

            free_list(CH_neigh2);

            free_list(CC_neigh2);
        }

        free_list(CH_neigh);
    }
    free_list(CC_neigh);

    *bond_counter += 12;
}

/**
 * Renumbers the Aspartic acid side-chain connected with BB_CC_BOND to match the
 * numbering used by ILVES.
 *
 * @param mol Molecule structure.
 * @param bb_CC_bond The bond of the backbone that connects one Carbon atom with
 * another (each chunk of the backbone has one C-C bond).
 * @param bond_counter First bond index of the new side-chain numbering. When
 * returning it will be its initial value plus the number of bonds of the side-chain.
 * @param permutation Permutation to apply to the numbering of the bonds of
 * the molecule. If you want to change the index of the current bond 0 to
 * 5 -> permutation[5] = 0.
 *
 */
static void ilves_peptides_reorder_aspartic_acid_side_chain(const molecule_t *const mol,
        const int bb_CC_bond,
        int *const bond_counter,
        int *const permutation) {
    /*
    Aspartic acid side-chain (bonds):
         Backbone
             |
     H - 3 - C - 4 - H
             |
             2
             |
     O - 0 - C - 1 - O
    */

    // Find the connection with the backbone.
    my_node_t *CC_neigh = get_neighbours_given_elements(mol, bb_CC_bond, CARBON, CARBON);
    int prev_pivot = bb_CC_bond;
    int pivot = CC_neigh->number;
    free_list(CC_neigh);

    my_node_t *CH_neigh = get_new_neighbours_given_elements(mol, pivot, prev_pivot,
                          CARBON, HYDROGEN);

    // 3
    permutation[*bond_counter + 3] = CH_neigh->number;
    // 4
    permutation[*bond_counter + 4] = CH_neigh->next->number;

    free_list(CH_neigh);

    CC_neigh = get_new_neighbours_given_elements(mol, pivot, prev_pivot,
               CARBON, CARBON);

    prev_pivot = pivot;
    pivot = CC_neigh->number;

    permutation[*bond_counter + 2] = CC_neigh->number;

    free_list(CC_neigh);

    my_node_t *CO_neigh = get_new_neighbours_given_elements(mol, pivot, prev_pivot,
                          CARBON, OXYGEN);

    // 0
    permutation[*bond_counter + 0] = CO_neigh->number;
    // 1
    permutation[*bond_counter + 1] = CO_neigh->next->number;

    free_list(CO_neigh);

    *bond_counter += 5;
}

/**
 * Renumbers the Glutamic acid side-chain connected with BB_CC_BOND to match the
 * numbering used by ILVES.
 *
 * @param mol Molecule structure.
 * @param bb_CC_bond The bond of the backbone that connects one Carbon atom with
 * another (each chunk of the backbone has one C-C bond).
 * @param bond_counter First bond index of the new side-chain numbering. When
 * returning it will be its initial value plus the number of bonds of the side-chain.
 * @param permutation Permutation to apply to the numbering of the bonds of
 * the molecule. If you want to change the index of the current bond 0 to
 * 5 -> permutation[5] = 0.
 *
 */
static void ilves_peptides_reorder_glutamic_acid_side_chain(const molecule_t *const mol,
        const int bb_CC_bond,
        int *const bond_counter,
        int *const permutation) {
    /*
    Glutamic acid side-chain (bonds):
         Backbone
             |
     H - 6 - C - 7 - H
             |
             5
             |
     H - 3 - C - 4 - H
             |
             2
             |
     O - 0 - C - 1 - O
    */

    // Find the connection with the backbone.
    my_node_t *CC_neigh = get_neighbours_given_elements(mol, bb_CC_bond, CARBON, CARBON);
    int prev_pivot = bb_CC_bond;
    int pivot = CC_neigh->number;
    free_list(CC_neigh);

    int counter = 7;
    for (int i = 0; i < 2; ++i, counter -= 3) {
        my_node_t *CH_neigh = get_new_neighbours_given_elements(mol, pivot, prev_pivot,
                              CARBON, HYDROGEN);

        // 6-3
        permutation[*bond_counter + counter - 0] = CH_neigh->number;
        // 7-4
        permutation[*bond_counter + counter - 1] = CH_neigh->next->number;

        free_list(CH_neigh);

        CC_neigh = get_new_neighbours_given_elements(mol, pivot, prev_pivot,
                   CARBON, CARBON);

        prev_pivot = pivot;
        pivot = CC_neigh->number;

        // 5-2
        permutation[*bond_counter + counter - 2] = CC_neigh->number;

        free_list(CC_neigh);
    }

    my_node_t *CO_neigh = get_new_neighbours_given_elements(mol, pivot, prev_pivot,
                          CARBON, OXYGEN);

    // 0
    permutation[*bond_counter + 0] = CO_neigh->number;
    // 1
    permutation[*bond_counter + 1] = CO_neigh->next->number;

    free_list(CO_neigh);

    *bond_counter += 8;
}

/**
 * Renumbers the Asparagine side-chain connected with BB_CC_BOND to match the
 * numbering used by ILVES.
 *
 * @param mol Molecule structure.
 * @param bb_CC_bond The bond of the backbone that connects one Carbon atom with
 * another (each chunk of the backbone has one C-C bond).
 * @param bond_counter First bond index of the new side-chain numbering. When
 * returning it will be its initial value plus the number of bonds of the side-chain.
 * @param permutation Permutation to apply to the numbering of the bonds of
 * the molecule. If you want to change the index of the current bond 0 to
 * 5 -> permutation[5] = 0.
 *
 */
static void ilves_peptides_reorder_asparagine_side_chain(const molecule_t *const mol,
        const int bb_CC_bond,
        int *const bond_counter,
        int *const permutation) {
    /*
    Asparagine side-chain (bonds):
         Backbone
             |
     H - 5 - C - 6 - H
             |
             4
             |
     O - 3 - C
             |
             2
             |
     H - 0 - N - 1 - H
    */

    // Find the connection with the backbone.
    my_node_t *CC_neigh = get_neighbours_given_elements(mol, bb_CC_bond, CARBON, CARBON);
    int prev_pivot = bb_CC_bond;
    int pivot = CC_neigh->number;
    free_list(CC_neigh);

    my_node_t *CH_neigh = get_new_neighbours_given_elements(mol, pivot, prev_pivot,
                          CARBON, HYDROGEN);

    // 5
    permutation[*bond_counter + 5] = CH_neigh->number;
    // 6
    permutation[*bond_counter + 6] = CH_neigh->next->number;

    free_list(CH_neigh);

    CC_neigh = get_new_neighbours_given_elements(mol, pivot, prev_pivot,
               CARBON, CARBON);

    prev_pivot = pivot;
    pivot = CC_neigh->number;

    // 4
    permutation[*bond_counter + 4] = CC_neigh->number;

    free_list(CC_neigh);

    my_node_t *CO_neigh = get_new_neighbours_given_elements(mol, pivot, prev_pivot,
                          CARBON, OXYGEN);

    // 3
    permutation[*bond_counter + 3] = CO_neigh->number;

    free_list(CO_neigh);

    my_node_t *CN_neigh = get_new_neighbours_given_elements(mol, pivot, prev_pivot,
                          CARBON, NITROGEN);

    prev_pivot = pivot;
    pivot = CN_neigh->number;

    permutation[*bond_counter + 2] = CN_neigh->number;

    free_list(CN_neigh);

    my_node_t *NH_neigh = get_new_neighbours_given_elements(mol, pivot, prev_pivot,
                          NITROGEN, HYDROGEN);

    // 0
    permutation[*bond_counter + 0] = NH_neigh->number;
    // 1
    permutation[*bond_counter + 1] = NH_neigh->next->number;

    free_list(NH_neigh);

    *bond_counter += 7;
}

/**
 * Renumbers the Glutamine side-chain connected with BB_CC_BOND to match the
 * numbering used by ILVES.
 *
 * @param mol Molecule structure.
 * @param bb_CC_bond The bond of the backbone that connects one Carbon atom with
 * another (each chunk of the backbone has one C-C bond).
 * @param bond_counter First bond index of the new side-chain numbering. When
 * returning it will be its initial value plus the number of bonds of the side-chain.
 * @param permutation Permutation to apply to the numbering of the bonds of
 * the molecule. If you want to change the index of the current bond 0 to
 * 5 -> permutation[5] = 0.
 *
 */
static void ilves_peptides_reorder_glutamine_side_chain(const molecule_t *const mol,
        const int bb_CC_bond,
        int *const bond_counter,
        int *const permutation) {
    /*
    Glutamine side-chain (bonds):
                 Backbone
                     |
             H - 8 - C - 9 - H
                     |
                     7
                     |
             H - 5 - C - 6 - H
                     |
                     4
                     |
     H - 1 - N - 2 - C - 3 - O
             |
             0
             |
             H
    */

    // Find the connection with the backbone.
    my_node_t *CC_neigh = get_neighbours_given_elements(mol, bb_CC_bond, CARBON, CARBON);
    int prev_pivot = bb_CC_bond;
    int pivot = CC_neigh->number;
    free_list(CC_neigh);

    int counter = 9;
    for (int i = 0; i < 2; ++i, counter -= 3) {
        my_node_t *CH_neigh = get_new_neighbours_given_elements(mol, pivot, prev_pivot,
                              CARBON, HYDROGEN);

        // 9-6
        permutation[*bond_counter + counter] = CH_neigh->number;
        // 8-5
        permutation[*bond_counter + counter - 1] = CH_neigh->next->number;

        free_list(CH_neigh);

        CC_neigh = get_new_neighbours_given_elements(mol, pivot, prev_pivot,
                   CARBON, CARBON);

        prev_pivot = pivot;
        pivot = CC_neigh->number;

        // 7-4
        permutation[*bond_counter + counter - 2] = CC_neigh->number;

        free_list(CC_neigh);
    }

    my_node_t *CO_neigh = get_new_neighbours_given_elements(mol, pivot, prev_pivot,
                          CARBON, OXYGEN);

    // 3
    permutation[*bond_counter + 3] = CO_neigh->number;

    free_list(CO_neigh);

    my_node_t *CN_neigh = get_new_neighbours_given_elements(mol, pivot, prev_pivot,
                          CARBON, NITROGEN);

    prev_pivot = pivot;
    pivot = CN_neigh->number;

    // 2
    permutation[*bond_counter + 2] = CN_neigh->number;

    free_list(CN_neigh);

    my_node_t *NH_neigh = get_new_neighbours_given_elements(mol, pivot, prev_pivot,
                          NITROGEN, HYDROGEN);

    // 0
    permutation[*bond_counter + 0] = NH_neigh->number;
    // 1
    permutation[*bond_counter + 1] = NH_neigh->next->number;

    free_list(NH_neigh);

    *bond_counter += 10;
}

/**
 * Renumbers the Histidine side-chain connected with BB_CC_BOND to match the
 * numbering used by ILVES.
 *
 * @param mol Molecule structure.
 * @param bb_CC_bond The bond of the backbone that connects one Carbon atom with
 * another (each chunk of the backbone has one C-C bond).
 * @param bond_counter First bond index of the new side-chain numbering. When
 * returning it will be its initial value plus the number of bonds of the side-chain.
 * @param permutation Permutation to apply to the numbering of the bonds of
 * the molecule. If you want to change the index of the current bond 0 to
 * 5 -> permutation[5] = 0.
 *
 */
static void ilves_peptides_reorder_histidine_side_chain(const molecule_t *const mol,
        const int bb_CC_bond,
        int *const bond_counter,
        int *const permutation) {
    /*
    Histidine side-chain (bonds):
         Backbone
             |
     H - 9 - C - 10 - H
             |
             8
             |
     N - 6 - C ------7
     |               |
     5               |
     |               |
     C - 2 - N - 4 - C
     |       |       |
     1       0       3
     |       |       |
     H       H       H
    */

    // Find the connection with the backbone.
    my_node_t *CC_neigh = get_neighbours_given_elements(mol, bb_CC_bond, CARBON, CARBON);
    int prev_pivot = bb_CC_bond;
    int pivot = CC_neigh->number;
    free_list(CC_neigh);

    my_node_t *CH_neigh = get_new_neighbours_given_elements(mol, pivot, prev_pivot,
                          CARBON, HYDROGEN);

    // 10
    permutation[*bond_counter + 10] = CH_neigh->number;
    // 9
    permutation[*bond_counter + 9] = CH_neigh->next->number;

    free_list(CH_neigh);

    CC_neigh = get_new_neighbours_given_elements(mol, pivot, prev_pivot,
               CARBON, CARBON);

    prev_pivot = pivot;
    pivot = CC_neigh->number;

    // 8
    permutation[*bond_counter + 8] = CC_neigh->number;

    free_list(CC_neigh);

    my_node_t *CN_neigh = get_new_neighbours_given_elements(mol, pivot, prev_pivot,
                          CARBON, NITROGEN);

    prev_pivot = pivot;
    pivot = CN_neigh->number;

    // 6
    permutation[*bond_counter + 6] = CN_neigh->number;

    free_list(CN_neigh);

    CN_neigh = get_new_neighbours_given_elements(mol, pivot, prev_pivot,
               CARBON, NITROGEN);

    prev_pivot = pivot;
    pivot = CN_neigh->number;

    // 5
    permutation[*bond_counter + 5] = CN_neigh->number;

    free_list(CN_neigh);

    CH_neigh = get_new_neighbours_given_elements(mol, pivot, prev_pivot,
               CARBON, HYDROGEN);

    // 1
    permutation[*bond_counter + 1] = CH_neigh->number;

    free_list(CH_neigh);

    CN_neigh = get_new_neighbours_given_elements(mol, pivot, prev_pivot,
               CARBON, NITROGEN);

    prev_pivot = pivot;
    pivot = CN_neigh->number;

    // 2
    permutation[*bond_counter + 2] = CN_neigh->number;

    free_list(CN_neigh);

    my_node_t *NH_neigh = get_new_neighbours_given_elements(mol, pivot, prev_pivot,
                          NITROGEN, HYDROGEN);

    // 0
    permutation[*bond_counter + 0] = NH_neigh->number;

    free_list(NH_neigh);

    CN_neigh = get_new_neighbours_given_elements(mol, pivot, prev_pivot,
               CARBON, NITROGEN);

    prev_pivot = pivot;
    pivot = CN_neigh->number;

    // 4
    permutation[*bond_counter + 4] = CN_neigh->number;

    free_list(CN_neigh);

    CH_neigh = get_new_neighbours_given_elements(mol, pivot, prev_pivot,
               CARBON, HYDROGEN);

    // 3
    permutation[*bond_counter + 3] = CH_neigh->number;

    free_list(CH_neigh);

    CC_neigh = get_new_neighbours_given_elements(mol, pivot, prev_pivot,
               CARBON, CARBON);

    // 3
    permutation[*bond_counter + 7] = CC_neigh->number;

    free_list(CC_neigh);

    *bond_counter += 11;
}

/**
 * Renumbers the Lysine side-chain connected with BB_CC_BOND to match the
 * numbering used by ILVES.
 *
 * @param mol Molecule structure.
 * @param bb_CC_bond The bond of the backbone that connects one Carbon atom with
 * another (each chunk of the backbone has one C-C bond).
 * @param bond_counter First bond index of the new side-chain numbering. When
 * returning it will be its initial value plus the number of bonds of the side-chain.
 * @param permutation Permutation to apply to the numbering of the bonds of
 * the molecule. If you want to change the index of the current bond 0 to
 * 5 -> permutation[5] = 0.
 *
 */
static void ilves_peptides_reorder_lysine_side_chain(const molecule_t *const mol,
        const int bb_CC_bond,
        int *const bond_counter,
        int *const permutation) {
    /*
    Lysine side-chain (bonds):
         Backbone
             |
     H -13 - C - 14- H
             |
             12
             |
     H -10 - C - 11- H
             |
             9
             |
     H - 7 - C - 8 - H
             |
             6
             |
     H - 4 - C - 5 - H
             |
             3
             |
     H - 1 - N - 2 - H
             |
             0
             |
             H
    */

    // Find the connection with the backbone.
    my_node_t *CC_neigh = get_neighbours_given_elements(mol, bb_CC_bond, CARBON, CARBON);
    int prev_pivot = bb_CC_bond;
    int pivot = CC_neigh->number;
    free_list(CC_neigh);

    int counter = 14;
    for (int i = 0; i < 5; i++) {
        my_node_t *NCH_neigh = get_new_neighbours_given_elements(mol, pivot, prev_pivot,
                               i == 4 ? NITROGEN : CARBON, HYDROGEN);

        my_node_t *it = NCH_neigh;
        while (it != NULL) {
            // HYDROGEN bond.
            permutation[*bond_counter + counter] = it->number;

            --counter;
            it = it->next;
        }
        free_list(NCH_neigh);

        if (i <= 3) {
            my_node_t *CCN_neigh = get_new_neighbours_given_elements(mol, pivot, prev_pivot,
                                   CARBON, i == 3 ? NITROGEN : CARBON);

            prev_pivot = pivot;
            pivot = CCN_neigh->number;

            permutation[*bond_counter + counter] = CCN_neigh->number;

            free_list(CCN_neigh);

            --counter;
        }
    }

    *bond_counter += 15;
}

/**
 * Renumbers the Arginine side-chain connected with BB_CC_BOND to match the
 * numbering used by ILVES.
 *
 * @param mol Molecule structure.
 * @param bb_CC_bond The bond of the backbone that connects one Carbon atom with
 * another (each chunk of the backbone has one C-C bond).
 * @param bond_counter First bond index of the new side-chain numbering. When
 * returning it will be its initial value plus the number of bonds of the side-chain.
 * @param permutation Permutation to apply to the numbering of the bonds of
 * the molecule. If you want to change the index of the current bond 0 to
 * 5 -> permutation[5] = 0.
 *
 */
static void ilves_peptides_reorder_arginine_side_chain(const molecule_t *const mol,
        const int bb_CC_bond,
        int *const bond_counter,
        int *const permutation) {
    /*
    Arginine side-chain (bonds):
         Backbone
             |
     H -15 - C - 16- H
             |
             14
             |
     H -12 - C - 13- H
             |
             11
             |
     H - 9 - C - 10 - H
             |
             8
             |
             N - 7 - H
             |
     H       6       H
     |       |       |
     1       |       4
     |       |       |
     N - 2 - C - 5 - N
     |               |
     0               3
     |               |
     H               H
    */

    // Find the connection with the backbone.
    my_node_t *CC_neigh = get_neighbours_given_elements(mol, bb_CC_bond, CARBON, CARBON);
    int prev_pivot = bb_CC_bond;
    int pivot = CC_neigh->number;
    free_list(CC_neigh);

    int counter = 16;
    for (int i = 0; i < 3; i++) {
        my_node_t *CH_neigh = get_new_neighbours_given_elements(mol, pivot, prev_pivot,
                              CARBON, HYDROGEN);

        my_node_t *it = CH_neigh;
        while (it != NULL) {
            // HYDROGEN bond.
            permutation[*bond_counter + counter] = it->number;

            --counter;
            it = it->next;
        }
        free_list(CH_neigh);

        my_node_t *CCN_neigh = get_new_neighbours_given_elements(mol, pivot, prev_pivot,
                               CARBON, i == 2 ? NITROGEN : CARBON);

        prev_pivot = pivot;
        pivot = CCN_neigh->number;

        // 14 - 11 - 8
        permutation[*bond_counter + counter] = CCN_neigh->number;
        --counter;

        free_list(CCN_neigh);
    }

    my_node_t *NH_neigh = get_new_neighbours_given_elements(mol, pivot, prev_pivot,
                          NITROGEN, HYDROGEN);

    // 7
    permutation[*bond_counter + 7] = NH_neigh->number;

    free_list(NH_neigh);

    my_node_t *CN_neigh = get_new_neighbours_given_elements(mol, pivot, prev_pivot,
                          CARBON, NITROGEN);

    prev_pivot = pivot;
    pivot = CN_neigh->number;

    // 6
    permutation[*bond_counter + 6] = CN_neigh->number;

    free_list(CN_neigh);

    CN_neigh = get_new_neighbours_given_elements(mol, pivot, prev_pivot,
               CARBON, NITROGEN);

    counter = 5;
    my_node_t *it = CN_neigh;

    while (it != NULL) {
        // 5 - 2
        permutation[*bond_counter + counter - 0] = it->number;

        my_node_t *NH_neigh = get_new_neighbours_given_elements(mol, it->number,
                              pivot, NITROGEN, HYDROGEN);

        permutation[*bond_counter + counter - 1] = NH_neigh->number;
        permutation[*bond_counter + counter - 2] = NH_neigh->next->number;

        free_list(NH_neigh);

        counter -= 3;

        it = it->next;
    }

    free_list(CN_neigh);

    *bond_counter += 17;
}

/**
 * Renumbers the Serine side-chain connected with BB_CC_BOND to match the
 * numbering used by ILVES.
 *
 * @param mol Molecule structure.
 * @param bb_CC_bond The bond of the backbone that connects one Carbon atom with
 * another (each chunk of the backbone has one C-C bond).
 * @param bond_counter First bond index of the new side-chain numbering. When
 * returning it will be its initial value plus the number of bonds of the side-chain.
 * @param permutation Permutation to apply to the numbering of the bonds of
 * the molecule. If you want to change the index of the current bond 0 to
 * 5 -> permutation[5] = 0.
 *
 */
static void ilves_peptides_reorder_serine_side_chain(const molecule_t *const mol,
        const int bb_CC_bond,
        int *const bond_counter,
        int *const permutation) {
    /*
    Serine side-chain (bonds):
         Backbone
             |
     H - 2 - C - 3 - H
             |
             1
             |
             O
             |
             0
             |
             H
    */

    // Find the connection with the backbone.
    my_node_t *CC_neigh = get_neighbours_given_elements(mol, bb_CC_bond, CARBON, CARBON);
    int prev_pivot = bb_CC_bond;
    int pivot = CC_neigh->number;
    free_list(CC_neigh);

    my_node_t *CH_neigh = get_new_neighbours_given_elements(mol, pivot, prev_pivot,
                          CARBON, HYDROGEN);

    // 2
    permutation[*bond_counter + 2] = CH_neigh->number;
    // 3
    permutation[*bond_counter + 3] = CH_neigh->next->number;

    free_list(CH_neigh);

    my_node_t *CO_neigh = get_new_neighbours_given_elements(mol, pivot, prev_pivot,
                          CARBON, OXYGEN);
    prev_pivot = pivot;
    pivot = CO_neigh->number;

    // 1
    permutation[*bond_counter + 1] = CO_neigh->number;

    free_list(CO_neigh);

    my_node_t *OH_neigh = get_new_neighbours_given_elements(mol, pivot, prev_pivot,
                          OXYGEN, HYDROGEN);

    // 0
    permutation[*bond_counter + 0] = OH_neigh->number;

    free_list(OH_neigh);

    *bond_counter += 4;
}

/**
 * Renumbers the Phenylaline side-chain connected with BB_CC_BOND to match the
 * numbering used by ILVES.
 *
 * @param mol Molecule structure.
 * @param bb_CC_bond The bond of the backbone that connects one Carbon atom with
 * another (each chunk of the backbone has one C-C bond).
 * @param bond_counter First bond index of the new side-chain numbering. When
 * returning it will be its initial value plus the number of bonds of the side-chain.
 * @param permutation Permutation to apply to the numbering of the bonds of
 * the molecule. If you want to change the index of the current bond 0 to
 * 5 -> permutation[5] = 0.
 *
 */
static void ilves_peptides_reorder_phenylaline_side_chain(const molecule_t *const mol,
        const int bb_CC_bond,
        int *const bond_counter,
        int *const permutation) {
    /*
    Phenylaline side-chain (bonds):
                 Backbone
                     |
             H - 12- C -13 - H
                     |
                     11
                     |
     9 ------------- C ------------ 10
     |                               |
     C - 2 - C - 4 - C - 6 - C - 8 - C
     |       |       |       |       |
     0       1       3       5       7
     |       |       |       |       |
     H       H       H       H       H
    */

    // Find the connection with the backbone.
    my_node_t *CC_neigh = get_neighbours_given_elements(mol, bb_CC_bond, CARBON, CARBON);
    int prev_pivot = bb_CC_bond;
    int pivot = CC_neigh->number;
    free_list(CC_neigh);

    my_node_t *CH_neigh = get_new_neighbours_given_elements(mol, pivot, prev_pivot,
                          CARBON, HYDROGEN);

    // 12
    permutation[*bond_counter + 12] = CH_neigh->number;
    // 13
    permutation[*bond_counter + 13] = CH_neigh->next->number;

    free_list(CH_neigh);

    CC_neigh = get_new_neighbours_given_elements(mol, pivot, prev_pivot, CARBON, CARBON);

    prev_pivot = pivot;
    pivot = CC_neigh->number;

    // 11
    permutation[*bond_counter + 11] = CC_neigh->number;

    free_list(CC_neigh);

    CC_neigh = get_new_neighbours_given_elements(mol, pivot, prev_pivot, CARBON, CARBON);

    prev_pivot = pivot;
    pivot = CC_neigh->number;

    // 9
    permutation[*bond_counter + 9] = CC_neigh->number;
    // 10
    permutation[*bond_counter + 10] = CC_neigh->next->number;

    free_list(CC_neigh);

    CH_neigh = get_new_neighbours_given_elements(mol, pivot, prev_pivot,
               CARBON, HYDROGEN);

    // 0
    permutation[*bond_counter + 0] = CH_neigh->number;

    free_list(CH_neigh);

    int counter = 1;
    for (int i = 0; i < 4; ++i, counter += 2) {
        CC_neigh = get_new_neighbours_given_elements(mol, pivot, prev_pivot,
                   CARBON, CARBON);

        prev_pivot = pivot;
        pivot = CC_neigh->number;

        CH_neigh = get_new_neighbours_given_elements(mol, pivot, prev_pivot,
                   CARBON, HYDROGEN);

        // 1 - 3 - 5 - 7
        permutation[*bond_counter + counter + 0] = CH_neigh->number;

        free_list(CH_neigh);

        // 2 - 4 - 6 - 8
        permutation[*bond_counter + counter + 1] = CC_neigh->number;

        free_list(CC_neigh);
    }

    *bond_counter += 14;
}

/**
 * Renumbers the Tyrosine side-chain connected with BB_CC_BOND to match the
 * numbering used by ILVES.
 *
 * @param mol Molecule structure.
 * @param bb_CC_bond The bond of the backbone that connects one Carbon atom with
 * another (each chunk of the backbone has one C-C bond).
 * @param bond_counter First bond index of the new side-chain numbering. When
 * returning it will be its initial value plus the number of bonds of the side-chain.
 * @param permutation Permutation to apply to the numbering of the bonds of
 * the molecule. If you want to change the index of the current bond 0 to
 * 5 -> permutation[5] = 0.
 *
 */
static void ilves_peptides_reorder_tyrosine_side_chain(const molecule_t *const mol,
        const int bb_CC_bond,
        int *const bond_counter,
        int *const permutation) {
    /*
    Phenylaline side-chain (bonds):
                 Backbone
                     |
             H - 13- C -14 - H
                     |
                     12
                     |
     10 ------------ C ------------ 11
     |                               |
     C - 3 - C - 5 - C - 7 - C - 9 - C
     |       |       |       |       |
     1       2       4       6       8
     |       |       |       |       |
     H       H       O       H       H
                     |
                     0
                     |
                     H
    */

    // Find the connection with the backbone.
    my_node_t *CC_neigh = get_neighbours_given_elements(mol, bb_CC_bond, CARBON, CARBON);
    int prev_pivot = bb_CC_bond;
    int pivot = CC_neigh->number;
    free_list(CC_neigh);

    my_node_t *CH_neigh = get_new_neighbours_given_elements(mol, pivot, prev_pivot,
                          CARBON, HYDROGEN);

    // 13
    permutation[*bond_counter + 13] = CH_neigh->number;
    // 14
    permutation[*bond_counter + 14] = CH_neigh->next->number;

    free_list(CH_neigh);

    CC_neigh = get_new_neighbours_given_elements(mol, pivot, prev_pivot, CARBON, CARBON);

    prev_pivot = pivot;
    pivot = CC_neigh->number;

    // 12
    permutation[*bond_counter + 12] = CC_neigh->number;

    free_list(CC_neigh);

    CC_neigh = get_new_neighbours_given_elements(mol, pivot, prev_pivot, CARBON, CARBON);

    prev_pivot = pivot;
    pivot = CC_neigh->number;

    // 10
    permutation[*bond_counter + 10] = CC_neigh->number;
    // 11
    permutation[*bond_counter + 11] = CC_neigh->next->number;

    free_list(CC_neigh);

    CH_neigh = get_new_neighbours_given_elements(mol, pivot,
               prev_pivot, CARBON, HYDROGEN);

    // 1
    permutation[*bond_counter + 1] = CH_neigh->number;

    free_list(CH_neigh);

    int counter = 2;
    for (int i = 0; i < 4; ++i, counter += 2) {
        my_node_t *CC_neigh = get_new_neighbours_given_elements(mol, pivot,
                              prev_pivot, CARBON, CARBON);

        prev_pivot = pivot;
        pivot = CC_neigh->number;

        my_node_t *CHO_neigh = get_new_neighbours_given_elements(mol, pivot, prev_pivot,
                               CARBON, i == 1 ? OXYGEN : HYDROGEN);

        // 2 - 4 - 6 - 8
        permutation[*bond_counter + counter + 0] = CHO_neigh->number;

        // C-O-H (0)
        if (i == 1) {
            my_node_t *OH_neigh = get_new_neighbours_given_elements(mol, CHO_neigh->number,
                                  pivot, OXYGEN, HYDROGEN);

            permutation[*bond_counter + 0] = OH_neigh->number;

            free_list(OH_neigh);
        }

        free_list(CHO_neigh);

        // 3 - 5 - 7 - 9
        permutation[*bond_counter + counter + 1] = CC_neigh->number;

        free_list(CC_neigh);
    }

    *bond_counter += 15;
}

/**
 * Renumbers the Tryptophan side-chain connected with BB_CC_BOND to match the
 * numbering used by ILVES.
 *
 * @param mol Molecule structure.
 * @param bb_CC_bond The bond of the backbone that connects one Carbon atom with
 * another (each chunk of the backbone has one C-C bond).
 * @param bond_counter First bond index of the new side-chain numbering. When
 * returning it will be its initial value plus the number of bonds of the side-chain.
 * @param permutation Permutation to apply to the numbering of the bonds of
 * the molecule. If you want to change the index of the current bond 0 to
 * 5 -> permutation[5] = 0.
 *
 */
static void ilves_peptides_reorder_tryptophan_side_chain(const molecule_t *const mol,
        const int bb_CC_bond,
        int *const bond_counter,
        int *const permutation) {
    /*
    Tryptophan side-chain (bonds):
                             Backbone
                                 |
                        H - 17 - C - 18 - H
                                 |
                                 16
                                 |
               H - 10 - C - 14 - C ----- 15
                        |                |
                       11                |
                        |                |
               H -  9 - N - 12 - C -13 - C
                                 |       |
                                 8       7
                                 |       |
                         H - 0 - C       C - 5 - H
                                 |       |
                                 2       6
                                 |       |
                         H - 1 - C - 4 - C - 3 - H
    */

    // Find the connection with the backbone.
    my_node_t *CC_neigh = get_neighbours_given_elements(mol, bb_CC_bond, CARBON, CARBON);
    int prev_pivot = bb_CC_bond;
    int pivot = CC_neigh->number;
    free_list(CC_neigh);

    my_node_t *CH_neigh = get_new_neighbours_given_elements(mol, pivot, prev_pivot,
                          CARBON, HYDROGEN);

    // 17
    permutation[*bond_counter + 17] = CH_neigh->number;
    // 18
    permutation[*bond_counter + 18] = CH_neigh->next->number;

    free_list(CH_neigh);

    CC_neigh = get_new_neighbours_given_elements(mol, pivot,
               prev_pivot, CARBON, CARBON);

    prev_pivot = pivot;
    pivot = CC_neigh->number;

    // 16
    permutation[*bond_counter + 16] = CC_neigh->number;

    free_list(CC_neigh);

    CC_neigh = get_new_neighbours_given_elements(mol, pivot,
               prev_pivot, CARBON, CARBON);

    my_node_t *CN_neigh = get_new_neighbours_given_elements(mol, CC_neigh->number,
                          pivot, CARBON, NITROGEN);

    int CN_count = count(CN_neigh);
    free_list(CN_neigh);

    prev_pivot = pivot;
    if (CN_count == 0) { // Right -> go to left.
        pivot = CC_neigh->next->number;

        // 15
        permutation[*bond_counter + 15] = CC_neigh->number;
    }
    else { // Left, OK.
        pivot = CC_neigh->number;

        // 15
        permutation[*bond_counter + 15] = CC_neigh->next->number;
    }

    // 14
    permutation[*bond_counter + 14] = pivot;

    free_list(CC_neigh);

    CH_neigh = get_new_neighbours_given_elements(mol, pivot,
               prev_pivot, CARBON, HYDROGEN);

    // 10
    permutation[*bond_counter + 10] = CH_neigh->number;

    free_list(CH_neigh);

    CN_neigh = get_new_neighbours_given_elements(mol, pivot,
               prev_pivot, CARBON, NITROGEN);

    prev_pivot = pivot;
    pivot = CN_neigh->number;

    // 11
    permutation[*bond_counter + 11] = CN_neigh->number;

    free_list(CN_neigh);

    my_node_t *NH_neigh = get_new_neighbours_given_elements(mol, pivot,
                          prev_pivot, NITROGEN, HYDROGEN);

    // 9
    permutation[*bond_counter + 9] = NH_neigh->number;

    free_list(NH_neigh);

    CN_neigh = get_new_neighbours_given_elements(mol, pivot,
               prev_pivot, CARBON, NITROGEN);

    prev_pivot = pivot;
    pivot = CN_neigh->number;

    // 12
    permutation[*bond_counter + 12] = CN_neigh->number;

    free_list(CN_neigh);

    CC_neigh = get_new_neighbours_given_elements(mol, pivot,
               prev_pivot, CARBON, CARBON);

    CH_neigh = get_new_neighbours_given_elements(mol, CC_neigh->number,
               pivot, CARBON, HYDROGEN);

    int CH_count = count(CH_neigh);
    free_list(CH_neigh);

    prev_pivot = pivot;
    if (CH_count == 0) { // Right -> go down.
        pivot = CC_neigh->next->number;

        // 13
        permutation[*bond_counter + 13] = CC_neigh->number;
    }
    else { // Down, OK.
        pivot = CC_neigh->number;

        // 13
        permutation[*bond_counter + 13] = CC_neigh->next->number;
    }

    free_list(CC_neigh);

    // 8
    permutation[*bond_counter + 8] = pivot;

    CH_neigh = get_new_neighbours_given_elements(mol, pivot,
               prev_pivot, CARBON, HYDROGEN);

    // 0
    permutation[*bond_counter + 0] = CH_neigh->number;
    free_list(CH_neigh);

    CC_neigh = get_new_neighbours_given_elements(mol, pivot, prev_pivot, CARBON, CARBON);

    prev_pivot = pivot;
    pivot = CC_neigh->number;

    // 2
    permutation[*bond_counter + 2] = CC_neigh->number;

    free_list(CC_neigh);

    CH_neigh = get_new_neighbours_given_elements(mol, pivot, prev_pivot, CARBON, HYDROGEN);

    // 1
    permutation[*bond_counter + 1] = CH_neigh->number;
    free_list(CH_neigh);

    CC_neigh = get_new_neighbours_given_elements(mol, pivot, prev_pivot, CARBON, CARBON);

    prev_pivot = pivot;
    pivot = CC_neigh->number;

    // 4
    permutation[*bond_counter + 4] = CC_neigh->number;
    free_list(CC_neigh);

    CH_neigh = get_new_neighbours_given_elements(mol, pivot, prev_pivot, CARBON, HYDROGEN);

    // 3
    permutation[*bond_counter + 3] = CH_neigh->number;
    free_list(CH_neigh);

    CC_neigh = get_new_neighbours_given_elements(mol, pivot, prev_pivot, CARBON, CARBON);

    prev_pivot = pivot;
    pivot = CC_neigh->number;

    // 6
    permutation[*bond_counter + 6] = CC_neigh->number;
    free_list(CC_neigh);

    CH_neigh = get_new_neighbours_given_elements(mol, pivot, prev_pivot, CARBON, HYDROGEN);

    // 5
    permutation[*bond_counter + 5] = CH_neigh->number;
    free_list(CH_neigh);

    CC_neigh = get_new_neighbours_given_elements(mol, pivot, prev_pivot, CARBON, CARBON);

    // 7
    permutation[*bond_counter + 7] = CC_neigh->number;
    free_list(CC_neigh);

    *bond_counter += 19;
}

/**
 * Renumbers the Theronine side-chain connected with BB_CC_BOND to match the
 * numbering used by ILVES.
 *
 * @param mol Molecule structure.
 * @param bb_CC_bond The bond of the backbone that connects one Carbon atom with
 * another (each chunk of the backbone has one C-C bond).
 * @param bond_counter First bond index of the new side-chain numbering. When
 * returning it will be its initial value plus the number of bonds of the side-chain.
 * @param permutation Permutation to apply to the numbering of the bonds of
 * the molecule. If you want to change the index of the current bond 0 to
 * 5 -> permutation[5] = 0.
 *
 */
static void ilves_peptides_reorder_theronine_side_chain(const molecule_t *const mol,
        const int bb_CC_bond,
        int *const bond_counter,
        int *const permutation) {
    /*
    Theronine side-chain (bonds):
                 Backbone
                     |
             H       |
             |       |
             2       |
             |       |
     H - 1 - C - 3 - C - 5 - O
             |       |       |
             0       6       4
             |       |       |
             H       H       H
    */

    // Find the connection with the backbone.
    my_node_t *CC_neigh = get_neighbours_given_elements(mol, bb_CC_bond, CARBON, CARBON);
    int prev_pivot = bb_CC_bond;
    int pivot = CC_neigh->number;
    free_list(CC_neigh);

    my_node_t *CH_neigh = get_new_neighbours_given_elements(mol, pivot, prev_pivot,
                          CARBON, HYDROGEN);

    // 6
    permutation[*bond_counter + 6] = CH_neigh->number;

    free_list(CH_neigh);

    my_node_t *CO_neigh = get_new_neighbours_given_elements(mol, pivot, prev_pivot,
                          CARBON, OXYGEN);

    // 5
    permutation[*bond_counter + 5] = CO_neigh->number;

    my_node_t *OH_neigh = get_new_neighbours_given_elements(mol, CO_neigh->number,
                          pivot, OXYGEN, HYDROGEN);

    // 4
    permutation[*bond_counter + 4] = OH_neigh->number;

    free_list(CO_neigh);
    free_list(OH_neigh);

    CC_neigh = get_new_neighbours_given_elements(mol, pivot, prev_pivot,
               CARBON, CARBON);

    // 3
    permutation[*bond_counter + 3] = CC_neigh->number;

    CH_neigh = get_new_neighbours_given_elements(mol, CC_neigh->number,
               pivot, CARBON, HYDROGEN);

    // 0
    permutation[*bond_counter + 0] = CH_neigh->number;
    // 1
    permutation[*bond_counter + 1] = CH_neigh->next->number;
    // 2
    permutation[*bond_counter + 2] = CH_neigh->next->next->number;

    free_list(CC_neigh);
    free_list(CH_neigh);

    *bond_counter += 7;
}

// Array of pointers to side-chain renumbering functions.
static void (*side_chain_renumerators[NUM_TYPES])(const molecule_t *const mol,
        const int bb_CC_bond,
        int *const bond_counter,
        int *const permutation) = {
    ilves_peptides_reorder_glycine_side_chain,
    ilves_peptides_reorder_proline_side_chain,
    ilves_peptides_reorder_cysteine_side_chain,
    ilves_peptides_reorder_methionine_side_chain,
    ilves_peptides_reorder_alaline_side_chain,
    ilves_peptides_reorder_valine_side_chain,
    ilves_peptides_reorder_leucine_side_chain,
    ilves_peptides_reorder_isoleucine_side_chain,
    ilves_peptides_reorder_aspartic_acid_side_chain,
    ilves_peptides_reorder_glutamic_acid_side_chain,
    ilves_peptides_reorder_asparagine_side_chain,
    ilves_peptides_reorder_glutamine_side_chain,
    ilves_peptides_reorder_histidine_side_chain,
    ilves_peptides_reorder_lysine_side_chain,
    ilves_peptides_reorder_arginine_side_chain,
    ilves_peptides_reorder_serine_side_chain,
    ilves_peptides_reorder_phenylaline_side_chain,
    ilves_peptides_reorder_tyrosine_side_chain,
    ilves_peptides_reorder_tryptophan_side_chain,
    ilves_peptides_reorder_theronine_side_chain
};

/**
 * Renumbers the backbone of MOL to match the numbering used by ILVES.
 *
 * @param mol Molecule structure.
 * @param bb_CC_bond The first bond of the backbone that connects one Carbon atom
 * with another (each chunk of the backbone has one C-C bond).
 * @param bond_counter First bond index of the new backbone numbering. When
 * returning it will be its initial value plus the number of bonds of the backbone.
 * @param permutation Permutation to apply to the numbering of the bonds of
 * the molecule. If you want to change the index of the current bond 0 to
 * 5 -> permutation[5] = 0.
 *
 */
static void ilves_peptides_reorder_backbone(const molecule_t *const mol,
        const int first_bb_CC_bond,
        int *const bond_counter,
        int *const permutation) {

    int sep_bond = mol->n - mol->num_side_chains;

    int prev_pivot = -1;
    int pivot = first_bb_CC_bond;

    for (int i = 0; i < mol->num_side_chains; ++i) {

        // Renumber C-O bond.
        my_node_t *CO_neigh = get_neighbours_given_elements(mol, pivot, CARBON, OXYGEN);

        // First side-chain
        if (i == 0) {
            if (mol->ends_type == TYPE_B) {
                my_node_t *it = CO_neigh;
                while (it != NULL) {
                    my_node_t *OH_neigh = get_neighbours_given_elements(mol,
                                          it->number, HYDROGEN, OXYGEN);
                    if (OH_neigh != NULL) { // spl
                        permutation[0] = OH_neigh->number;
                        permutation[1] = it->number;
                    }
                    else {
                        permutation[*bond_counter] = it->number;
                        ++*bond_counter;
                    }
                    free_list(OH_neigh);

                    it = it->next;
                }
            }
            else {
                permutation[0] = CO_neigh->number;
                permutation[*bond_counter] = CO_neigh->next->number;
                ++*bond_counter;
            }
        }
        else {
            permutation[*bond_counter] = CO_neigh->number;
            ++*bond_counter;
        }
        free_list(CO_neigh);


        // Renumber C-C bond.
        permutation[*bond_counter] = pivot;
        ++*bond_counter;

        // Renumber C-H bond(s) (a Glycine aminoacid has two C-H backbone bonds).
        my_node_t *CH_neigh = get_neighbours_given_elements(mol, pivot, CARBON, HYDROGEN);

        my_node_t *it = CH_neigh;
        while (it != NULL) {
            permutation[*bond_counter] = it->number;

            ++*bond_counter;

            it = it->next;
        }
        free_list(CH_neigh);

        // Renumber bb-C-sc-C bond (a Glycine aminoacid has no bb-C-sc-C backbone bonds).
        my_node_t *CC_neigh = get_neighbours_given_elements(mol, pivot, CARBON, CARBON);

        it = CC_neigh;
        while (it != NULL) {
            permutation[*bond_counter] = it->number;

            it = it->next;

            ++*bond_counter;
        }
        free_list(CC_neigh);

        // Renumber C-N bond.
        my_node_t *CN_neigh;

        if (prev_pivot == -1) {
            CN_neigh = get_neighbours_given_elements(mol, pivot,
                       CARBON, NITROGEN);
        }
        else {
            CN_neigh = get_new_neighbours_given_elements(mol, pivot, prev_pivot,
                       CARBON, NITROGEN);
        }

        prev_pivot = pivot;
        pivot = CN_neigh->number;

        permutation[*bond_counter] = CN_neigh->number;

        ++*bond_counter;

        free_list(CN_neigh);

        // Renumber N-H bond(s).
        my_node_t *NH_neigh = get_new_neighbours_given_elements(mol, pivot,
                              prev_pivot, NITROGEN, HYDROGEN);

        int NH_count = count(NH_neigh);

        // Last side-chain
        if (i == mol->num_side_chains - 1) {
            // Type A molecule.
            if (mol->ends_type == TYPE_A) {
                permutation[1] = NH_neigh->number; // spr
                permutation[sep_bond] = NH_neigh->next->number;

                if (NH_count == 3) {
                    permutation[*bond_counter] = NH_neigh->next->next->number;
                }
            }
            // Type B molecule.
            else {
                permutation[sep_bond] = NH_neigh->number;

                if (NH_count == 2) {
                    permutation[*bond_counter] = NH_neigh->next->number;
                }
            }
        }
        // Inner side-chain
        else {
            // Proline has no N-H bond.
            if (NH_count == 1) {
                permutation[*bond_counter] = NH_neigh->number;
                ++*bond_counter;
            }

            my_node_t *NC_neigh = get_new_neighbours_given_elements(mol, pivot,
                                  prev_pivot, NITROGEN, CARBON);

            prev_pivot = pivot;

            if (mol->side_chain_types[i] == PROLINE) { // PROLINE, we must chose the correct path.
                my_node_t *CO_neigh = get_new_neighbours_given_elements(mol,
                                      NC_neigh->number, pivot, CARBON, OXYGEN);

                int CO_neigh_count = count(CO_neigh);
                free_list(CO_neigh);

                if (CO_neigh_count == 0) {
                    pivot = NC_neigh->next->number;
                }
                else {
                    pivot = NC_neigh->number;
                }
            }
            else {
                pivot = NC_neigh->number;
            }

            free_list(NC_neigh);

            permutation[sep_bond] = pivot;
            ++sep_bond;

            CC_neigh = get_neighbours_given_elements(mol, pivot, CARBON, CARBON);

            prev_pivot = pivot;
            pivot = CC_neigh->number;

            free_list(CC_neigh);


        }
        free_list(NH_neigh);
    }
}

void renumber_bonds(molecule_t *mol, int *p) {

    // -------------------------------------------------------------------------
    // Declaration of internal variables
    // -------------------------------------------------------------------------

    // Number of bonds
    int n;

    // Auxiliary bond list
    int *aux_bonds;
    real *aux_sigmaA;
    real *aux_sigmaB;
    real *aux_sigma2;

    // Standard counter
    int i;

    // -------------------------------------------------------------------------
    // Start of instructions
    // -------------------------------------------------------------------------

    if (mol->bonds != NULL) {
        // Extract the number of bonds
        n = mol->n;

        // Allocate space for the bond auxiliary list
        aux_bonds = (int *)malloc(2 * n * sizeof(int));
        aux_sigmaA = (real *)malloc(n * sizeof(*aux_sigmaA));
        aux_sigmaB = (real *)malloc(n * sizeof(*aux_sigmaB));
        aux_sigma2 = (real *)malloc(n * sizeof(*aux_sigma2));

        // Copy the original bond list into the auxiliary list
        for (i = 0; i < 2 * n; i++) {
            aux_bonds[i] = mol->bonds[i];
        }
        for (i = 0; i < n; i++) {
            aux_sigmaA[i] = mol->sigmaA[i];
            aux_sigmaB[i] = mol->sigmaB[i];
            aux_sigma2[i] = mol->sigma2[i];
        }

        // Apply the permutation to the bond list
        for (i = 0; i < n; i++) {
            mol->bonds[2 * i + 0] = aux_bonds[2 * p[i] + 0];
            mol->bonds[2 * i + 1] = aux_bonds[2 * p[i] + 1];

            mol->sigmaA[i] = aux_sigmaA[p[i]];
            mol->sigmaB[i] = aux_sigmaB[p[i]];
            mol->sigma2[i] = aux_sigma2[p[i]];
        }

        // Free the auxiliary bond list
        free(aux_bonds);
        free(aux_sigmaA);
        free(aux_sigmaB);
        free(aux_sigma2);
    }
    if (mol->bond_graph != NULL) {
        // Apply the permutation to the bond graph
        graph_renumber_vertices(mol->bond_graph, p);
    }
}

/**
 * Renumber the bonds of MOL to match the numbering used by ILVES.
 *
 * @param mol Molecule structure.
 */
void ilves_peptides_reorder(molecule_t *const mol) {
    // Number of bonds.
    int n = mol->n;

    // Permutation to apply to the bond list.
    int *permutation = (int *) malloc(n * sizeof(*permutation));

#if 0
    for (int i = 0; i < n; i++) {
        permutation[i] = -1;
    }
#endif

    // Find the beginning of the backbone.
    int first_bb_CC_bond = -1;
    for (int i = 0; i < n; ++i)  {
        my_node_t *neigh = get_neighbours_given_elements(mol, i, CARBON, OXYGEN);
        int neigh_CO_count = count(neigh);
        free_list(neigh);

        if (neigh_CO_count < 2) {
            continue;
        }

        neigh = get_neighbours_given_elements(mol, i, CARBON, NITROGEN);
        int neigh_CN_count = count(neigh);
        free_list(neigh);

        if (neigh_CN_count < 1) {
            continue;
        }

        // Bond found.
        first_bb_CC_bond = i;
        break;
    }

    if (first_bb_CC_bond < 0) {
        fprintf(stderr, "ERROR while reordering the bonds\n");
        exit(1);
    }

    my_node_t *side_chain_types_root = NULL;

    int bond_counter = 2; // The first two bonds are the special bonds.

    int num_side_chains = 0;
    int num_side_chain_bonds = 0;

    char molecule_end = 0;

    int prev_pivot = -1;
    int pivot = first_bb_CC_bond;

    // Renumber the side-chains.
    while (!molecule_end) {
        side_chain_type_t type = identify_side_chain(mol, pivot);

        (*side_chain_renumerators[type])(mol, pivot, &bond_counter, permutation);

        num_side_chain_bonds += side_chain_bonds[type];
        num_side_chains += 1;
        insert(type, &side_chain_types_root);

        my_node_t *CN_neigh;

        if (prev_pivot == -1) { // First side-chain
            CN_neigh = get_neighbours_given_elements(mol, pivot,
                       CARBON, NITROGEN);
        }
        else {
            CN_neigh = get_new_neighbours_given_elements(mol, pivot, prev_pivot,
                       CARBON, NITROGEN);
        }

        prev_pivot = pivot;
        pivot = CN_neigh->number;

        free_list(CN_neigh);

        my_node_t *NC_neigh = get_new_neighbours_given_elements(mol, pivot, prev_pivot,
                              NITROGEN, CARBON);
        int NC_neigh_count = count(NC_neigh);

        if (NC_neigh == NULL || (type == PROLINE && NC_neigh_count == 1)) {
            molecule_end = 1;

            my_node_t *NH_neigh = get_new_neighbours_given_elements(mol, pivot,
                                  prev_pivot, NITROGEN, HYDROGEN);

            int NH_count = count(NH_neigh);

            free_list(NH_neigh);

            // Type A molecule.
            if ((type == PROLINE && NH_count == 2) ||
                    (type != PROLINE && NH_count == 3)) {

                mol->ends_type = TYPE_A;
            }
            else {
                mol->ends_type = TYPE_B;
            }
        }
        else {
            prev_pivot = pivot;

            if (type == PROLINE) { // PROLINE, we must chose the correct path.
                my_node_t *CO_neigh = get_new_neighbours_given_elements(mol,
                                      NC_neigh->number, pivot, CARBON, OXYGEN);
                int CO_neigh_count = count(CO_neigh);
                free_list(CO_neigh);

                if (CO_neigh_count == 0) {
                    pivot = NC_neigh->next->number;
                }
                else {
                    pivot = NC_neigh->number;
                }
            }
            else {
                pivot = NC_neigh->number;
            }

            my_node_t *CC_neigh = get_neighbours_given_elements(mol, pivot,
                                  CARBON, CARBON);

            pivot = CC_neigh->number;

            free_list(CC_neigh);
        }
        free_list(NC_neigh);
    }

    // Copy the side-chains into the molecule.
    mol->num_side_chains = num_side_chains;
    mol->side_chain_types = (side_chain_type_t *) malloc(mol->num_side_chains * sizeof(*mol->side_chain_types));
    mol->n_sc = num_side_chain_bonds;

    copy_list(side_chain_types_root, (int *)mol->side_chain_types);
    free_list(side_chain_types_root);

    ilves_peptides_reorder_backbone(mol, first_bb_CC_bond, &bond_counter, permutation);

#if 0
    for (int i = 0; i < n; i++) {
        bool bond_in_permutation = false;
        for (int j = 0; j < n; j++) {
            if (permutation[j] == i) {
                bond_in_permutation = true;
            }
        }

        if (!bond_in_permutation) {
            fprintf(stderr, "ERROR: bond %d not found in the permutation\n", i);
        }
        else if (permutation[i] == -1) {
            fprintf(stderr, "ERROR: When renumbering the bonds, permutation[%d] is -1\n", i);
            exit(1);
        }
    }
#endif

    renumber_bonds(mol, permutation);

    free(permutation);
}

//----------------------------------------------------------------------------
/* read lines starting with the % character */
static void skip_comments(char *line, int length, FILE *fp) {

    while (fgets(line, length, fp) != NULL) {

        size_t len = strlen(line);

        if (len == 0) {
            break;
        }

        char aux[MAX_LINE_LEN];
        aux[0] = line[len - 1];

        len = 1;

        // Read until new line.
        while (aux[len - 1] != '\n') {
            if (fgets(aux, MAX_LINE_LEN, fp) == NULL) {
                break;
            }

            len = strlen(aux);
        }

        if (strncmp("%", line, 1) != 0) {
            break;
        }
    }
}

//----------------------------------------------------------------------------
/* read the next line */
static void advance_line(char *line, int length, FILE *fp) {

    if (fgets(line, length, fp) == NULL) {
        printf("ERROR: Unexpected End Of File\n\n");
        exit(-1);
    }
}

//----------------------------------------------------------------------------
void read_molecule_file(molecule_t *mol, const char *filename) {

    /* Reads the description of a molecule from a formatted text file */

    // Counter
    int i;

    // Atoms, bonds, side-chain bonds
    int m, n, n_sc;

    // Number of side-chains
    int num_side_chains;

    // A file pointer
    FILE *fp;

    // Define a text line of MAX_LINE_LEN characters.
    char line[MAX_LINE_LEN];

    // Length of strings.
    size_t l;

    // Open the file for reading
    fp = fopen(filename, "r");
    if (fp == NULL) {
        printf("ERROR: cannot open %s file\n\n", filename);
        exit(EXIT_FAILURE);
    }

    // Skip initial comments
    skip_comments(line, MAX_LINE_LEN, fp);
    l = strlen(line);
    mol->name = (char *) calloc((l + 1), sizeof(char));
    memcpy(mol->name, line, l + 1);
    skip_comments(line, MAX_LINE_LEN, fp);
    l = strlen(line);
    mol->abb = (char *) calloc((l + 1), sizeof(char));
    memcpy(mol->abb, line, l + 1);
    skip_comments(line, MAX_LINE_LEN, fp);

    // Read the ends type
    sscanf(line, "%d", (int *)&mol->ends_type);
    skip_comments(line, MAX_LINE_LEN, fp);

    // Read the number of atoms
    sscanf(line, "%d", &m);
    mol->m = m;
    skip_comments(line, MAX_LINE_LEN, fp);

    // Read the number of bonds
    sscanf(line, "%d", &n);
    mol->n = n;
    skip_comments(line, MAX_LINE_LEN, fp);

    // Read the number of side-chains
    sscanf(line, "%d", &num_side_chains);
    // mol->num_side_chains = num_side_chains;
    skip_comments(line, MAX_LINE_LEN, fp);

    if (num_side_chains > 0) {
        // Read the number of side-chain bonds
        sscanf(line, "%d", &n_sc);
        mol->n_sc = n_sc;
        skip_comments(line, MAX_LINE_LEN, fp);

        // Read the list of side-chain types
        // mol->side_chain_types = (side_chain_type_t *) calloc((num_side_chains), sizeof(side_chain_type_t));
        for (i = 0; i < num_side_chains; i++) {
            int trash;
            sscanf(line, "%d", &trash);
            // scanf(line, "%d", (int *) & (mol->side_chain_types[i]));
            advance_line(line, MAX_LINE_LEN, fp);

            // fprintf(stderr, "%d ", mol->side_chain_types[i]);
        }
        skip_comments(line, MAX_LINE_LEN, fp);
    }
    // Read the list of atoms
    mol->atoms = (element_t *) malloc(m * sizeof(mol->atoms[0]));
    for (i = 0; i < m; i++) {
        sscanf(line, "%d", (int *)&mol->atoms[i]);
        advance_line(line, MAX_LINE_LEN, fp);
    }
    skip_comments(line, MAX_LINE_LEN, fp);

    // Allocate space for the list of bonds
    mol->bonds = (int *) malloc(2 * n * sizeof(int));

    // Allocate space for the bond lengths
    //mol->sigma2 = (real *) malloc(n*sizeof(real));
    mol->sigma2 = (real *) malloc(n * sizeof(real));
    mol->sigmaA = (real *) malloc(n * sizeof(real));

    // Read the list of bonds and the bond lengths
    for (i = 0; i < n; i++) {
#if PRECISION==0
        // Read the bond length as
        sscanf(line, "%d %d %f", &(mol->bonds[2 * i]), &(mol->bonds[2 * i + 1]),
               &(mol->sigma2[i]));
#else
        sscanf(line, "%d %d %lf", &(mol->bonds[2 * i]), &(mol->bonds[2 * i + 1]),
               &(mol->sigma2[i]));
#endif
        mol->sigmaA[i] = std::sqrt(mol->sigma2[i]);
        advance_line(line, MAX_LINE_LEN, fp);
    }

    // Close the input file
    fclose(fp);

    // We now initialize the invmass.
    mol->invmass = (real *) malloc(m * sizeof(real));
    for (i = 0; i < m; i++) {
        mol->invmass[i] = 1;
    }

    // Generate the bond graph from the list of bonds.
    make_bond_graph(mol);

    // Reorder the bonds and the bond graph.
    // renumber_bonds(mol);

    // free_mol(mol);
    // free_graph(mol->bond_graph);
    // free(mol->bond_graph);

    // exit(1);
}

//----------------------------------------------------------------------------
void make_bond_graph(molecule_t *mol) {
    /* Builds the bond graph from the bond list

       Description of interface

       ON ENTRY:
           mol->m       the number of atoms
           mol->n       the number of bonds
           mol->bonds   sequential list of n bonds

       ON EXIT:
           mol->graph   a pointer to a compact representation of the bond graph

       No other variables are modified!

       In the bond graph there is an edge between vertices i and j if and only
       if bonds i and j have one atom in common. The bond graph is stored using
       two arrays called XADJ and ADJ. Each adjacency list is stored in strictly
       increasing order. The n lists are stored in strictly increasing order in
       the array ADJ. The number XADJ[j] is the index inside ADJ of the first
       entry of the jth adjacency list. XADJ[n] points just beyond the end of
       ADJ and is the length of ADJ.

       IDEA:

       Every atom participates in at most K bonds. Suppose that we have m
       auxiliary lists, such that the ath list gives the bonds which atom a
       partakes in. Then for each bond i, involving atoms a(i) and b(i), we can
       immediately construct the adjacency list for bond i, simply by merging
       all the auxiliary list for atoms a(i) and b(i). Therefore our first step
       is the construction of these auxiliary lists.

       ALGORITHM

       STEP 1: Construction of the auxiliary lists

       for each bond i do
         for each atom a in bond i do
           insert i into the ath auxiliary list
         end
       end

       COST: There are n bonds which involve 2 atoms each. Each auxiliary list
       will never be longer than K, and inserting an element into a sorted list
       of length L requires at most L comparisons, so the cost is less than 2nK
       comparisons.

       STEP 2: Construction of the adjacency lists

       for each bond i do
         for each atom a in bond i do
           for each bond j in the ath auxiliary list do
             insert j into the ith adjacency lists
           end
         end
       end

       COST: There are again n bonds which involve at 2 atoms each. Each
       auxiliary list has a length which is at most K. Each adjacency list will
       have a length which is at most 2K+1. Inserting M elements into a list of
       length at most L requires less than LM comparisons, so in total we do
       O(nK^2) comparisons.

       STEPS 3: Compress the adjacency lists into the array ADJ and create XADJ.
       There are n adjacency lists of length at most 2K+1, so the cost is O(nK).

    */


    // -------------------------------------------------------------------------
    // Declaration of internal variables
    // -------------------------------------------------------------------------

    // The number of atoms
    int m;

    // The number of bonds
    int n;

    // The list of bonds
    int *bonds;

    // The bond graph
    graph_t *graph;

    // Standard counters
    int i, j;

    // The current bond is between atoms a and b.
    int a, b;

    // An auxilary variable used to generate XADJ
    int temp;

    // Variables needed to manipulate linked lists
    struct my_node *conductor;
    struct my_node **aux, **root;

    // -------------------------------------------------------------------------
    // Start of instructions
    // -------------------------------------------------------------------------

    // Extract the number of atoms (m) and the number of bonds
    m = mol->m;
    n = mol->n;

    // Establish shortcut to the list of bonds
    bonds = mol->bonds;

    // Allocate space for the bond graph
    graph = (graph_t *)malloc(sizeof(graph_t));

    // allocate space for the auxilary lists
    aux = (struct my_node **)malloc(m * sizeof(struct my_node *));

    /* Initialize the auxiliary lists. The ith auxiliary list will record the
       bonds which atom i partakes in. */

    for (i = 0; i < m; i++) {
        aux[i] = NULL;
    }

    // Loop over the sequential list of bonds
    for (j = 0; j < n; j++) {
        // Isolate the numbers of the atoms which partake in bond j.
        a = bonds[2 * j];
        b = bonds[2 * j + 1];
        // Insert bond j into the ath auxiliary list
        sinsert(j, &aux[a]);
        // Insert bond j into the bth auxiliary list
        sinsert(j, &aux[b]);
    }

    /* This completes the construction of the auxiliary list. For each atom we
       now have a list of the bonds it partakes in! */

    // Allocate space for n adjacency lists
    root = (struct my_node **)malloc(n * sizeof(struct my_node *));

    // Initialize the adjacency lists
    for (j = 0; j < n; j++) {
        root[j] = NULL;
    }

    // Loop over the sequential list of bonds
    for (j = 0; j < n; j++) {
        // Isolate the numbers of the atoms which partake in bond j.
        a = bonds[2 * j];
        b = bonds[2 * j + 1];
        /* Insert every element of the a'th auxiliary list into the adjacency
           list for bond j. */
        conductor = aux[a];
        while (conductor != NULL) {
            sinsert(conductor->number, &root[j]);
            conductor = conductor->next;
        }
        /* Insert every element of the b'th auxiliary list into the adjacency
           list for bond j */
        conductor = aux[b];
        while (conductor != NULL) {
            sinsert(conductor->number, &root[j]);
            conductor = conductor->next;
        }
    }

    // Free the memory used by the auxilliary lists
    for (i = 0; i < m; i++) {
        free_list(aux[i]);
    }
    // Do not forget to release aux itself.
    free(aux);

    /* This complete the construction of the adjacency lists as simply linked
       list. It remains to compress the information into two arrays XADJ and
       ADJ. */

    // Allocate space for the array of indices into the combined adjacency list
    graph->xadj = (int *)malloc((n + 1) * sizeof(int));

    // Initialize the counters
    temp = 0;
    graph->xadj[0] = temp;

    // Find the length of the adjacency lists and compute the entries of XADJ
    for (j = 0; j < n; j++) {
        temp = temp + count(root[j]);
        graph->xadj[j + 1] = temp;
    }

    // Allocate space for the graph
    graph->adj = (int *)malloc(temp * sizeof(int));

    // Copy the lists into graph->adj and free the memory used.
    for (j = 0; j < n; j++) {
        copy_list(root[j], &graph->adj[graph->xadj[j]]);
        free_list(root[j]);
    }
    // Do not forget to release root itself
    free(root);

    // Set the number of vertices in the bond graph
    graph->m = n;

    // Do NOT forget to save the result of your work into MOL!
    mol->bond_graph = graph;
}

//----------------------------------------------------------------------------
void make_weights(molecule_t *mol, graph_t *graph) {

    /* Precomputes the weights needed to generate the matrix A(x,y).

       Description of interface:

       ON ENTRY:

         mol->m        the number of atoms
         mol->invmass  pointer to the inverse of the atomic masses
         mol->n        the number of bonds
         mol->bonds    a pointer to a sequential list of n bonds
         graph         a pointer to a compact representation of the graph
                       to use, typically the bond graph or the lower triangular
                       portion of it

      ON EXIT:
         mol->weights  a pointer to list of weights compatible with the graph

      No other variables have been modified.

      REMARK(S):

       1) At this point we do not take fill into account. Fill is identified
          by playing the elimination game on the bond graph. Right now, I am
          simply not sure where this code should go.


       Description of the construction of the weights:

       The following example explains the origins of the weights and how to
       compute them.

       EXAMPLE:

       A molecule with m = 6 atoms and n = 5 bonds. The atoms are numbered 0
       through 5, and the bonds are numbered 0 through 4. It is irrelevant if
       this molecule exists in the real world or not :)


        0                 4           The bonds are given by the array
         \               /
         (0)           (3)            bonds = {0,2; 1,2; 2,3; 3,4; 3,5}
           \           /
            2 ------- 3               Each bond involves 2 atoms. Bond 2 is
           /    (2)    \              between atoms 2 and 3.
         (1)           (4)
         /               \
       1                  5

       The adjacency graph for the bonds is the graph


          0   3              matrix A is 5 by 5
         / \ / \
        |   2   |            |xxx  |
         \ / \ /             |xxx  |
          1   4              |xxxxx|
                             |  xxx|
                             |  xxx|

       This is how the graph is encoded

        adj  = {0,1,2; 0,1,2; 0,1,2,3,4; 2,3,4; 2,3,4}     (17 entries)
       xadj  = {0, 3, 6, 11, 14, 17}                       ( 6 entries)

       Please note that there are n+1 entries in XADJ, and the last entry point
       just beyond the end of adj. Therefore XADJ[n+1] is the number of nonzero
       entries in the matrix.

       NOTATION:

         1) We write rij or r(i,j) for the vector from atom i to j.
         2) We write <x,y> for the scalar product between the vectors x and y.

       In the above example bond number 2 involves atoms 2 and 3 and is a
       (mathematical) constraint of the type

                  0.5*(||r23||^2 - (bond length)^2) = 0

       The factor 0.5 is only included to give the Jacobian of the constraint
       function a form which I (CCKM) consider aesthetically pleasing.

       Now, our matrix of the form

                  A =  Dg(r)*inv(M)*Dg(s)'

       where r and s are vectors describing two different configurations of
       the m atoms, so r and s each have 3m components each.

       Below is a table of the nonzero entries of the matrix A for our current
       example:

                   bond = {0,2,1,2,2,3,3,4,3,5}

       i    j      entry A(i,j)
       ---------------------------------------------------------------
       0    0     (invmass(0)+invmass(2))*<r02,s03>
       1    0     +invmass(2)*<r12,s02>
       2    0     -invmass(2)*<r23,s02>

       0    1     +invmass(2)*<r02,s12>
       1    1     (invmass(1)+invmass(2))*<r12,s12>
       2    1     -invmass(2)*<r23,s12>

       0    2     -invmass(2)*<r02,s23>
       1    2     -invmass(2)*<r12,s23>
       2    2     (invmass(2)+invmass(3))*<r23,s23>
       3    2     -invmass(3)*<r34,s23>
       4    2     -invmass(3)*<r35,s23>

       2    3     -invmass(3)*<r23,s34>
       3    3     (invmass(3)+invmass(4))*<r34,s34>
       4    3     +invmass(3)*<r35,s34>

       2    4     -invmass(3)*<r23,s35>
       3    4     +invmass(3)*<r34,s35>
       4    4     (+invmass(3)+invmass(5))*<r35,s35>

       This table is very carefully constructed! Please note the following:

       a) Reading the table from the top to the bottom takes us through the
       nonzero entries of A in column major format, exactly as matrices are
       stored in LAPACK. Moreover, we are writing to main memory in an order
       which is strictly increasing.

       b) For each column of the matrix A we deal with a FIXED vector s(a,b),
       rather than both s(a,b) and s(b,a) = -s(a,b).

       c) The order of the indices as in r(a,b) for a bond k, is EXACTLY the
       order in which the atoms which partake in bond k are given in the bond
       table.

       EXAMPLE: As noted above bond 2 is a bond between atoms 2 and 3. The bond
       table lists atom 2 BEFORE atom 3. Therefore, we use the vector r23,
       rather than the (equvalent) vector r32 = -r23

       d) The diagonal entries are different from the off diagonal entries
       because the weights are different.

       In order to efficiently generate the matrix A we precompute the following
       weights

       i    j     weight
       ----------------------------------------------
       0    0     +invmass(0)+invmass(2)
       1    0     +invmass(2)
       2    0     -invmass(2)

       0    1     +invmass(2)
       1    1     +invmass(1)+invmass(2)
       2    1     -invmass(2)

       0    2     -invmass(2)
       1    2     -invmass(2)
       2    2     +invmass(2)+invmass(3)
       3    2     -invmass(3)
       4    2     -invmass(3)

       2    3     -invmass(3)
       3    3     +invmass(3)+invmass(4)
       4    3     +invmass(3)

       2    4     -invmass(3)
       3    4     +invmass(3)
       4    4     +invmass(3)+invmass(5)

       In short, the weights includes all the relevant information about the
       signs as well as the masses of the atoms.

       Given vectors r and s with a 3m components as well as the bond graph, we
       can now generate the matrix A one entry at a time, moving through RAM
       memory in a strictly increasing order.

    */

    // -------------------------------------------------------------------------
    // Declaration of internal variables
    // -------------------------------------------------------------------------

    // The number of bonds;
    int n;

    // Shortcut to the list of inverse masses
    real *invmass;

    // Shortcut to the list of bonds
    int *bonds;

    // Shortcut to the list of weights
    real *weights;

    // The number of nonzeros
    int nnz;

    // Standard counters
    int i, j, k;

    // Atomic labels
    int a, b, c, d, e;

    // -------------------------------------------------------------------------
    // Start of instructions
    // -------------------------------------------------------------------------

    // Extract the number of bonds
    n = mol->n;

    // Establish shortcut to the list of inverse masses.
    invmass = mol->invmass;

    // Establish shortcut to the list of bonds.
    bonds = mol->bonds;

    // Extract the number of nonzero entries
    nnz = graph->xadj[n];

    // Allocate enough space for nnz weights
    weights = (real *)malloc(nnz * sizeof(weights));

    // Initialize the pointer to the nonzero entries
    k = 0;
    // Loop over the bonds or equivalently the ROWS of the matrix
    for (i = 0; i < n; i++) {
        // Isolate the atoms which partake in the ith bond
        a = bonds[2 * i];
        b = bonds[2 * i + 1];
        // Loop over the entries of the adjacency list of the ith bond.
        for (j = graph->xadj[i]; j < graph->xadj[i + 1]; j++) {
            if (i != graph->adj[j]) {
                // This is an off diagonal diagonal entry.
                // We begin by isolating the atoms which partake in the ith bond
                c = bonds[2 * graph->adj[j]];
                d = bonds[2 * graph->adj[j] + 1];

                // Is a fillin.
                if (a != c && a != d && b != c && b != d) {
                    weights[k] = 0;
                }
                else {
                    // Determine the atom which is common to bond j and bond i
                    if ((a == c) || (a == d)) {
                        // The common atom is atom a
                        e = a;
                    }
                    else {
                        // The common atom must necessarily be atom b.
                        e = b;
                    }
                    weights[k] = invmass[e];
                    // Determine the appropriate sign
                    if ((a == d) || (b == c)) {
                        /* You should reverse the order the atoms for one of the two bonds,
                        but this impractical, so we just reverse the sign of the weight
                        */
                        weights[k] = -weights[k];
                    }
                }
            }
            else {
                /* This is a diagonal entry.
                   The weight is special, but the order of the atoms in the bond list
                   is irrelevant. Yes, you could flip the order of one pair of atoms,
                   but then you would be compelled to flip the order of the second
                   pair, and so you would change sign twice.
                */
                weights[k] = invmass[a] + invmass[b];
            }

            // Move on to the next nonzero entry of the matrix
            k++;
        }
    }
    // Remember to save the results of your work!
    mol->weights = weights;
}

//----------------------------------------------------------------------------
void free_mol(molecule_t *mol) {

    // Name and abbreviation.
    free(mol->name);
    free(mol->abb);

    // The inverse mass.
    free(mol->invmass);

    // atoms, bonds, sigma2, sigma.
    free(mol->atoms);
    free(mol->bonds);
    free(mol->sigma2);
    free(mol->sigmaA);
    free(mol->sigmaB);

    // side-chain-types.
    free(mol->side_chain_types);
}
