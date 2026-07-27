#ifndef COSMA_HELPER_HPP
#define COSMA_HELPER_HPP

#include <mpi.h>

#ifdef COSMA
extern "C" {
    void cosma_pdgemm_(const char *transa, const char *transb,
        const int *m, const int *n, const int *k,
        const double *alpha,
        const double *a, const int *ia, const int *ja, const int *desca,
        const double *b, const int *ib, const int *jb, const int *descb,
        const double *beta,
        double *c, const int *ic, const int *jc, const int *descc);
}
#endif

extern "C" {
    void pdgemm_(const char *transa, const char *transb,
        const int *m, const int *n, const int *k,
        const double *alpha,
        const double *a, const int *ia, const int *ja, const int *desca,
        const double *b, const int *ib, const int *jb, const int *descb,
        const double *beta,
        double *c, const int *ic, const int *jc, const int *descc);
}

inline void my_pdgemm(const char *transa, const char *transb,
    const int *m, const int *n, const int *k,
    const double *alpha,
    const double *a, const int *ia, const int *ja, const int *desca,
    const double *b, const int *ib, const int *jb, const int *descb,
    const double *beta,
    double *c, const int *ic, const int *jc, const int *descc)
{
#ifdef COSMA
    cosma_pdgemm_(transa, transb, m, n, k, alpha,
        a, ia, ja, desca, b, ib, jb, descb,
        beta, c, ic, jc, descc);
#else
    pdgemm_(transa, transb, m, n, k, alpha,
        a, ia, ja, desca, b, ib, jb, descb,
        beta, c, ic, jc, descc);
#endif
}

#endif
