import csv
import itertools
import sys

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data

def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def inherit(parent, one_gene, two_genes):
    # probability that child inherits gene??from mom or dad
    if parent in two_genes:
        parent_prob = 1-PROBS["mutation"]
        return parent_prob
    elif parent in one_gene:
        parent_prob = 0.50
        return parent_prob
    else:
        parent_prob = PROBS["mutation"]
        return parent_prob


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """

    jp = 1
    for person in people:
        # If no parent listed pull from PROBS
        if people[person]["mother"] == None and people[person]["father"] == None:
            if person in have_trait:
                if person in one_gene:
                    calc = PROBS["gene"][1]*PROBS["trait"][1][True]
                    jp = jp*calc
                elif person in two_genes:
                    calc = PROBS["gene"][2]*PROBS["trait"][2][True]
                    jp = jp*calc
                else:
                    calc = PROBS["gene"][0]*PROBS["trait"][0][True]
                    jp = jp*calc
            else:
                if person in one_gene:
                    calc = PROBS["gene"][1]*PROBS["trait"][1][False]
                    jp = jp*calc
                elif person in two_genes:
                    calc = PROBS["gene"][2]*PROBS["trait"][2][False]
                    jp = jp*calc
                else:
                    calc = PROBS["gene"][0]*PROBS["trait"][0][False]
                    jp = jp*calc
        # else calculate probability of having gene w/given parent
        else:
            prob_mom = inherit(people[person]["mother"], one_gene, two_genes)
            prob_dad = inherit(people[person]["father"], one_gene, two_genes)
            if person in have_trait:
                if person in one_gene:
                    calc = (prob_mom*(1-prob_dad) + prob_dad*(1-prob_mom))*PROBS["trait"][1][True]
                    jp = jp*calc
                elif person in two_genes:
                    calc = (prob_mom*prob_dad)*PROBS["trait"][2][True]
                    jp = jp*calc
                else:
                    calc = (1-prob_mom)*(1-prob_dad)*PROBS["trait"][0][True]
                    jp = jp*calc
            else:
                if person in one_gene:
                    calc = (prob_mom*(1-prob_dad) + prob_dad*(1-prob_mom))*PROBS["trait"][1][False]
                    jp = jp*calc
                elif person in two_genes:
                    calc = prob_mom*prob_dad*PROBS["trait"][2][False]
                    jp = jp*calc
                else:
                    calc = (1-prob_mom)*(1-prob_dad)*PROBS["trait"][0][False]
                    jp = jp*calc
    return jp

def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    for person in probabilities:
        #update gene
        if person in one_gene:
            probabilities[person]["gene"][1] += p
        elif person in two_genes:
            probabilities[person]["gene"][2] += p
        else:
            probabilities[person]["gene"][0] += p

        #update trait
        if person in have_trait:
            probabilities[person]["trait"][True] += p
        else:
            probabilities[person]["trait"][False] += p

def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    #Check if the values sum to 1
    for person in probabilities:
        sum_gene = 0
        sum_trait = 0
        for i in probabilities[person]["gene"].values():
            sum_gene = sum_gene + i
        for j in probabilities[person]["trait"].values():
            sum_trait = sum_trait + j
        if sum_gene != 1:
            #normalize
            norm_gene = [probabilities[person]["gene"][k]/sum_gene for k in probabilities[person]["gene"]]
            #[2, 1, 0]
            g=2
            for n in range(len(norm_gene)):
                probabilities[person]["gene"][g]=norm_gene[n]
                g-=1

        if sum_trait != 1:
            t = [True, False]
            norm_trait = [probabilities[person]["trait"][l]/sum_trait for l in probabilities[person]["trait"]]
            for m in range(len(norm_trait)):
                probabilities[person]["trait"][t[m]]=norm_trait[m]

if __name__ == "__main__":
    main()
