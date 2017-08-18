/* Create macros so that the matrices are stored in column-major order */

#define A(i,j) a[ (j)*lda + (i) ]
#define B(i,j) b[ (j)*ldb + (i) ]
#define C(i,j) c[ (j)*ldc + (i) ]

/* Routine for computing C = A * B + C */

void AddDot(int, double*, int, double*, double*);
void AddDot4x4(int k, double* a, int lda, double *b, int ldb,
    double* c, int ldc);

void MY_MMult( int m, int n, int k, double *a, int lda, 
                                    double *b, int ldb,
                                    double *c, int ldc )
{
  int i, j;

  for ( j=0; j<n; j+=4 ){        /* Loop over the columns of C */
    for ( i=0; i<m; i+=4 ){        /* Loop over the rows of C */
      /* Update C( i,j ) with the inner
         product of the ith row of A and
         the jth column of B */
      AddDot4x4(k, &A(i,0), lda, &B(0,j), ldb, &C(i,j), ldc);
    }
  }
}

#include <mmintrin.h>
#include <xmmintrin.h>
#include <pmmintrin.h>
#include <emmintrin.h>

typedef union {
  __m128d v;
  double d[2];
} v2df_t;

void AddDot4x4(int k, double* a, int lda, double *b, int ldb,
    double* c, int ldc) {
  /* So, this routine computes a 4x4 block of matrix A

           C( 0, 0 ), C( 0, 1 ), C( 0, 2 ), C( 0, 3 ).  
           C( 1, 0 ), C( 1, 1 ), C( 1, 2 ), C( 1, 3 ).  
           C( 2, 0 ), C( 2, 1 ), C( 2, 2 ), C( 2, 3 ).  
           C( 3, 0 ), C( 3, 1 ), C( 3, 2 ), C( 3, 3 ).  

     Notice that this routine is called with c = C( i, j ) in the
     previous routine, so these are actually the elements 

           C( i  , j ), C( i  , j+1 ), C( i  , j+2 ), C( i  , j+3 ) 
           C( i+1, j ), C( i+1, j+1 ), C( i+1, j+2 ), C( i+1, j+3 ) 
           C( i+2, j ), C( i+2, j+1 ), C( i+2, j+2 ), C( i+2, j+3 ) 
           C( i+3, j ), C( i+3, j+1 ), C( i+3, j+2 ), C( i+3, j+3 ) 
	  
     in the original matrix C */ 
  int p;
  v2df_t c00_c10_vreg, c01_c11_vreg, 
         c02_c12_vreg, c03_c13_vreg,
         c20_c30_vreg, c21_c31_vreg,
         c22_c32_vreg, c23_c33_vreg,
         a0p_a1p_vreg, a2p_a3p_vreg,
         b_p0_vreg, b_p1_vreg, b_p2_vreg, b_p3_vreg;

  double *b_p0_ptr, *b_p1_ptr, *b_p2_ptr, *b_p3_ptr;

  c00_c10_vreg.v = _mm_setzero_pd();
  c01_c11_vreg.v = _mm_setzero_pd();
  c02_c12_vreg.v = _mm_setzero_pd();
  c03_c13_vreg.v = _mm_setzero_pd();
  c20_c30_vreg.v = _mm_setzero_pd();
  c21_c31_vreg.v = _mm_setzero_pd();
  c22_c32_vreg.v = _mm_setzero_pd();
  c23_c33_vreg.v = _mm_setzero_pd();

  b_p0_ptr = &B(0,0);
  b_p1_ptr = &B(0,1);
  b_p2_ptr = &B(0,2);
  b_p3_ptr = &B(0,3);

  for(p=0; p<k; p++) { 
    a0p_a1p_vreg.v = _mm_load_pd((double*) &A(0,p));
    a2p_a3p_vreg.v = _mm_load_pd((double*) &A(2,p));
    //load and duplicate
    b_p0_vreg.v = _mm_loaddup_pd((double*) b_p0_ptr++);
    b_p1_vreg.v = _mm_loaddup_pd((double*) b_p1_ptr++);
    b_p2_vreg.v = _mm_loaddup_pd((double*) b_p2_ptr++);
    b_p3_vreg.v = _mm_loaddup_pd((double*) b_p3_ptr++);

    //first and second rows
    c00_c10_vreg.v += a0p_a1p_vreg.v * b_p0_vreg.v;
    c01_c11_vreg.v += a0p_a1p_vreg.v * b_p1_vreg.v;
    c02_c12_vreg.v += a0p_a1p_vreg.v * b_p2_vreg.v;
    c03_c13_vreg.v += a0p_a1p_vreg.v * b_p3_vreg.v;

    //third and fourth rows
    c20_c30_vreg.v += a2p_a3p_vreg.v * b_p0_vreg.v;
    c21_c31_vreg.v += a2p_a3p_vreg.v * b_p1_vreg.v;
    c22_c32_vreg.v += a2p_a3p_vreg.v * b_p2_vreg.v;
    c23_c33_vreg.v += a2p_a3p_vreg.v * b_p3_vreg.v;
  }
  C(0,0) += c00_c10_vreg.d[0];
  C(0,1) += c01_c11_vreg.d[0];
  C(0,2) += c02_c12_vreg.d[0];
  C(0,3) += c03_c13_vreg.d[0];
  C(1,0) += c00_c10_vreg.d[1];
  C(1,1) += c01_c11_vreg.d[1];
  C(1,2) += c02_c12_vreg.d[1];
  C(1,3) += c03_c13_vreg.d[1];

  C(2,0) += c20_c30_vreg.d[0];
  C(2,1) += c21_c31_vreg.d[0];
  C(2,2) += c22_c32_vreg.d[0];
  C(2,3) += c23_c33_vreg.d[0];
  C(3,0) += c20_c30_vreg.d[1];
  C(3,1) += c21_c31_vreg.d[1];
  C(3,2) += c22_c32_vreg.d[1];
  C(3,3) += c23_c33_vreg.d[1];
}
