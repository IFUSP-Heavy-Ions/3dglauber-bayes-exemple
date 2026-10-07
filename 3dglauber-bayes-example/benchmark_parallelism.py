import os
import time
import multiprocessing as mp
import numpy as np

_BENCHMARK_IMPORT = True

with open("bayes_mcmc.py") as f:
    exec(compile(f.read(), "bayes_mcmc.py", "exec"), globals())


N = 240
NPROCESSES = int(os.environ.get("NPROCESSES", "12"))

worker_chain = None


def init_worker():
    global worker_chain
    worker_chain = Chain()


def evaluate(theta):
    return worker_chain.log_posterior(theta)[0]


if __name__ == "__main__":

    mode = os.environ["BENCH_MODE"]

    # Chain usada pelo processo principal.
    main_chain = Chain()

    # Exatamente os mesmos theta em todos os experimentos.
    np.random.seed(12345)
    X = main_chain.random_pos(N)

    if mode == "serial":

        worker_chain = main_chain

        # Mede somente as avaliações das likelihoods.
        start = time.perf_counter()
        results = [evaluate(theta) for theta in X]
        elapsed = time.perf_counter() - start

    elif mode == "pool":

        # Inicializa processos e Chains ANTES do cronômetro.
        with mp.Pool(
            processes=NPROCESSES,
            initializer=init_worker,
        ) as pool:

            start = time.perf_counter()
            results = pool.map(evaluate, X)
            elapsed = time.perf_counter() - start

    else:
        raise ValueError(f"Unknown BENCH_MODE={mode}")

    results = np.asarray(results)

    print()
    print("========== RESULT ==========")
    print(f"mode       : {mode}")
    print(f"N          : {N}")
    if mode == "pool":
        print(f"processes  : {NPROCESSES}")
    print(f"wall       : {elapsed:.3f} s")
    print(f"throughput : {N / elapsed:.3f} likelihoods/s")
    print(f"checksum   : {results.sum():.12e}")
    finite = np.isfinite(results)
    print(f"finite     : {finite.sum()} / {N}")
    print(f"-inf       : {np.isneginf(results).sum()}")
    print(f"checksum   : {results[finite].sum():.12e}")