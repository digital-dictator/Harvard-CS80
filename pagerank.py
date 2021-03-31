import numpy as np
import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    if corpus[page]:
        probs = {
            page: (1 / len(corpus)) * (1 - damping_factor)
            for page in corpus
        }
        for link in corpus[page]:
            probs[link] += (1 / len(corpus[page])) * damping_factor
    else:
        probs = {page: 1 / len(corpus) for page in corpus}
    return probs


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    counts = {page: 0 for page in corpus}
    current = random.choice(list(corpus))
    counts[current] = 1

    for i in range(n - 1):
        transitions = transition_model(corpus, current, damping_factor)
        states = list(transitions)
        probs = [transitions[state] for state in states]
        current = np.random.choice(states, p=probs)
        counts[current] += 1

    return {page: counts[page] / n for page in corpus}


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    ranks = {page: 1 / len(corpus) for page in corpus}
    while True:
        new_ranks = dict()
        for page in ranks:
            incoming_pages = [
                other for other in corpus
                if page in corpus[other] or len(corpus[other]) == 0
            ]
            new_ranks[page] = (
                ((1 - damping_factor) / len(corpus))
                + damping_factor * sum(
                    ranks[other] / (len(corpus[other]) or len(corpus))
                    for other in incoming_pages
                )
            )

        max_delta = max(abs(new_ranks[page] - ranks[page]) for page in corpus)
        if max_delta < 0.001:
            break
        ranks = new_ranks
    return ranks


if __name__ == "__main__":
    main()
