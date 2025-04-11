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
    # return a dictionary where key = page : value = probability it will used
    # 
    probability_dict = {}
    total_number_of_pages = len(corpus)
    # fill dict with keys for pages and set prob to 0 allowing addition later on
    for key in corpus:
        probability_dict[key] = 0
    #  get the links from the working page
    # if page doesn't exists take all pages for equal probability
    if corpus[page]:
        links = corpus[page]
    else:
        links = corpus.keys()
    # Cycle each page and work out prob it will be returned
    for web_page in probability_dict:
        probability_dict[web_page] = (1- damping_factor) / total_number_of_pages
        if web_page in links:
            probability_dict[web_page] += damping_factor / len(links)
    return probability_dict


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # create a dict to store the counters
    page_ranks = {}

    for web_page in corpus:
        page_ranks[web_page] = 0
    # randomly select a page to work from
    working_page = random.choice(list(corpus.keys()))
    
    # start looping through the pages
    for i in range(n):
        page_ranks[working_page] += 1
        prob_dict = transition_model(corpus, working_page, damping_factor)
        # split the pages and probs into an ordered list
        pages = list(prob_dict.keys())
        probabilities = list(prob_dict.values())
        # get next working page used weighted decision
        working_page = random.choices(pages, probabilities)[0]
    # convert to %
    for page in page_ranks:
        page_ranks[page] /= n
    return page_ranks


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    number_of_pages = len(corpus)

    page_ranks = {}
    for page in corpus:
        page_ranks[page] = 1 / number_of_pages
    
    threshold = 0.001
    new_page_rank = page_ranks.copy()

    while True:
        for page in corpus:
            total = (1 - damping_factor) / number_of_pages
            for working_page in corpus:
                # check if the page we are working on is found linked in other pages
                if page in corpus[working_page]:
                    total += damping_factor * (
                        page_ranks[working_page] / len(corpus[working_page])
                        )
                    # treat no links a link to all
                if not corpus[working_page]:
                    total += damping_factor * (
                        page_ranks[working_page] / number_of_pages
                        )
            new_page_rank[page] = total
        converged = True
        for page in page_ranks:
            difference = abs(new_page_rank[page] - page_ranks[page])
            if difference >= threshold:
                converged = False
                break
        if converged == True:
            break
        else:
            page_ranks = new_page_rank.copy()
    return page_ranks


if __name__ == "__main__":
    main()
