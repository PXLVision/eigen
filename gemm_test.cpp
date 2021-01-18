#include <iostream>
#include <Eigen/Dense>

#ifndef __FOR_DEBUG__
#include <chrono>
#endif

using namespace std;
using namespace Eigen;

using Scalar = float;
using MatrixType = Matrix<Scalar, Dynamic, Dynamic>;

int main(int argc, char* argv[])
{
#ifdef __FOR_DEBUG__
    long size = 9;
#else
    long size = 512;
    auto start = chrono::high_resolution_clock::now();
    auto end = chrono::high_resolution_clock::now();
#endif
    auto m = size, k = size, n = size;
    MatrixType A = MatrixType::Random(m,k),
               B = MatrixType::Random(k,n),
               C = MatrixType::Zero(m,n),
               D = MatrixType::Zero(m,n);

#ifdef __FOR_DEBUG__
    for(auto i = 0; i < m; i++)
    {
        for(auto j = 0; j < k; j++)
        {
            A(i,j) = 100 + i*10 + j;
        }
    }
    for(auto i = 0; i < k; i++)
    {
        for(auto j = 0; j < n; j++)
        {
            B(i,j) = 200+ i*10 + j;
        }
    }
    for(auto i = 0; i < m; i++)
    {
        for(auto j = 0; j < n; j++)
        {
            for(auto d = 0; d < k; d++)
            {
                C(i,j) += A(i,d)*B(d,j);
            }
        }
    }
/*
    for(auto j = 0; j < m; j++)
    {
        for(auto i = 0; i < n; i++)
        {
            Acc1 = 0;
            Acc2 = 0;
            Acc3 = 0;
            Acc4 = 0;
            for(auto d = 0; d < k; d++)
            {
                Acc1 += A(i+0,d)*B(d,j);
                Acc2 += A(i+1,d)*B(d,j);
                Acc3 += A(i+2,d)*B(d,j);
                Acc4 += A(i+3,d)*B(d,j);
                acc = Av(i,d)*B(d,j);
            }
            C(i,j) = Acc;
        }
    }
*/
/*
    for(auto j = 0; j < m; j++)
    {
        for(auto d = 0; d < n; d++)
        {
            b = B(d,j);
            for(auto i = 0; i < k; i++)
            {
                C(i,j) += A(i,d)*b;
            }
        }
    }
*/
    cout << "For:" << endl << C << endl;
    cout << "Eigen: " << endl << A*B << endl;
#else
    long runs = 50;
    start = chrono::high_resolution_clock::now();
    for(auto i = 0; i < runs; i++)
        D = A*B;
    end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;
    cout << elapsed.count() << endl;
#endif
    return 0;
}