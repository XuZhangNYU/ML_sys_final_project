#include <getopt.h>
#include <assert.h>
#include "utils/tensor.cuh"
#include "ops/op_mm.cuh"
#include "ops/op_elemwise.cuh"
#include "ops/op_reduction.cuh"
#include "ops/op_cross_entropy.cuh"

unsigned long long randgen_seed = 0;

void test_matmul(int m, int n, int k, bool on_gpu)
{
    Tensor<float> A{m, k, on_gpu};
    op_uniform_fill(A);

    Tensor<float> B{k, n, on_gpu};
    op_uniform_fill(B);

    Tensor<float> C{m, n, on_gpu};
    op_mm(A, B, C);

    Tensor<float> C2{n, m, on_gpu};
    op_mm(B.transpose(), A.transpose(), C2);
    assert(C.allclose(C2.transpose())); // test transpose

    std::cout << "matmul passed..." << std::endl;
}

void test_elemwise(int m, int n, bool on_gpu)
{
    Tensor<float> Zref_two{m, n, false};
    op_const_fill(Zref_two, 2.0f);

    Tensor<float> X{m, n, on_gpu};
    op_const_fill(X, 2.0f);
    assert(X.allclose(Zref_two));

    Tensor<float> Y{m, n, on_gpu};
    op_const_fill(Y, 3.0f);

    Tensor<float> Z{m, n, on_gpu};
    op_add(X, Y, Z);

    Tensor<float> Zref_five{m, n, false};
    op_const_fill(Zref_five, 5.0f);
    assert(Z.allclose(Zref_five));

    Tensor<float> Y2{1, n, on_gpu};
    op_const_fill(Y2, 3.0f);
    op_add(X, Y2, Z); //test broadcasting
    assert(Z.allclose(Zref_five));

    op_add<float>(X, 3.0f, Z);
    assert(Z.allclose(Zref_five));

    std::cout << "op_add passed..." << std::endl;

    op_sub<float>(Z, Y, Z);
    assert(Z.allclose(Zref_two));
    std::cout << "op_sub passed..." << std::endl;

    op_multiply(X, Y, Z);

    Tensor<float> Zref_six{m, n, false};
    op_const_fill(Zref_six, 6.0f);
    assert(Z.allclose(Zref_six));

    op_multiply(X, Y2, Z);
    assert(Z.allclose(Zref_six));

    op_multiply<float>(X, 3.0f, Z);
    assert(Z.allclose(Zref_six));

    std::cout << "op_multiply passed..." << std::endl;

    Tensor<int> X_int{m, n, on_gpu};
    op_const_fill(X_int, 2);

    Tensor<int> Y_int{m, n, false};
    op_const_fill(Y_int, 2);
    Index(Y_int, m/2, n/2) = 5;
    if (on_gpu) {
        Y_int = Y_int.toDevice();
    }

    Tensor<int> Z_int{m, n, on_gpu};
    op_equal(X_int, Y_int, Z_int);
    Z_int = Z_int.toHost();
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (i == m/2 && j == n/2) {
                assert(Index(Z_int, i, j) == 0);
            } else {
                assert(Index(Z_int, i, j) == 1);
            }
        }
    }
    std::cout << "op_equal passed..." << std::endl;

    auto X_host = X.toHost();
    Index(X_host, m/2, n/2) = -10.0;
    if (on_gpu) {
        X = X_host.toDevice();
    } else {
        X = X_host;
    }
    op_relu(X, Z);
    auto Z_host = Z.toHost();
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (i == m/2 && j == n/2) {
                assert(Index(Z_host, i, j) == 0.0);
            } else {
                assert(Index(Z_host, i, j) == 2.0);
            }
        }
    }
    std::cout << "op_relu passed..." << std::endl;

    op_relu_back(X, Y, Z);
    Z_host = Z.toHost();
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (i == m/2 && j == n/2) {
                assert(Index(Z_host, i, j) == 0.0);
            } else {
                assert(Index(Z_host, i, j) == 3.0);
            }
        }
    }
    std::cout << "op_relu_back passed..." << std::endl;
}

bool is_close_enough(float a, float b) {
    if (std::abs(a - b) > 0.0001) {
        return false;
    } else {
        return true;
    }
}
void assert_all_close_enough(Tensor<float> t, std::vector<float> v)
{
    for (int i = 0; i < t.h; i++) {
        for (int j = 0; j < t.w; j++) {
            assert(is_close_enough(Index(t, i, j), v[i*t.w+j]));
        }
    }
}

void
test_op_cross_entropy_loss(bool on_gpu)
{
    Tensor<float> logits_host{2, 3};
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            Index(logits_host, i, j) = i*3+j;
        }
    }
    Tensor<char> targets{2,1};
    for (int i = 0; i < 2; i++) {        
        Index(targets, i, 0) = i;
    }
    Tensor<float> logits = logits_host;
    if (on_gpu) {
        logits = logits.toDevice();
        targets = targets.toDevice();
    }
    Tensor<float> d_logits{2, 3, on_gpu};
    float loss = op_cross_entropy_loss(logits, targets, d_logits);
    //std::cout << "loss=" << loss << std::endl;
    assert(is_close_enough(loss, 1.9076));

    Tensor<float> d_logits_host = d_logits;
    if (on_gpu) {
        d_logits_host = d_logits.toHost();
    }
    std::vector<float> d_logits_ref{-0.4550, 0.1224, 0.3326, 0.0450, -0.3776, 0.3326};
    assert_all_close_enough(d_logits_host, d_logits_ref);

    //test if the safe version is implemented
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            Index(logits_host, i, j) += 100000;
        }
    }
    if (on_gpu) {
        logits = logits_host.toDevice();
    }
    loss = op_cross_entropy_loss(logits, targets, d_logits);
    if (on_gpu) {
        d_logits_host = d_logits.toHost();
    }
    assert_all_close_enough(d_logits_host, d_logits_ref);

    std::cout << "op_cross_entropy_loss passed..." << std::endl;

}

void 
test_reduction(int m, int n, bool on_gpu)
{
    Tensor<int> X_host{m, n};
    op_const_fill(X_host, 0);

    int reduce_sum = m>n?n:m;
    for (int i = 0; i < X_host.h; i++) 
    {
        if (i >= X_host.w) {
            break;
        }
        Index(X_host, i, i) = 1;
    }

    Tensor<int> X;
    if (on_gpu) {
        X = X_host.toDevice();
    } else {
        X = X_host;
    } 
    Tensor<int> Y{1,n, on_gpu};
    op_sum(X,Y);

    Tensor<int> Yref{1,n};
    op_const_fill(Yref,0);
    for (int j=0; j< reduce_sum; j++) {
        Index(Yref, 0, j) = 1;
    }
    Tensor<int> Y_host = Y.toHost();
    assert(Y.allclose(Yref));

    Tensor<int> Y1{1,1, on_gpu};
    op_sum(Y, Y1);
    Tensor<int> Y1_host = Y1.toHost();
    assert(Index(Y1_host,0,0) == reduce_sum);

    op_const_fill(X, 1);
    op_sum(X, Y);
    op_const_fill(Yref, X.h);
    assert(Y.allclose(Yref));
    
    Tensor<int> YY{m, 1, on_gpu};
    op_sum(X, YY);
    Tensor<int> YYref{m, 1};
    op_const_fill(YYref, n);
    op_sum(YY, Y1);
    Y1_host = Y1.toHost();
    assert(Index(Y1_host, 0, 0) == m*n);

    std::cout << "op_sum passed..." << std::endl;

    //try to create an A matrix whose last column has the biggest value
    Tensor<float> A{m, n, on_gpu};
    op_uniform_fill<float>(A, 0.0, 1.0);
    auto AA = A.slice(0, A.h, A.w-1, A.w);
    op_add<float>(AA, 10.0, AA);

    Tensor<int> ind{m, 1, on_gpu};
    op_argmax(A, ind);
    Tensor<int> indref{m, 1};
    op_const_fill(indref, n-1);
    assert(ind.allclose(indref));
    std::cout << "op_argmax passed..." << std::endl;
}

void 
test_views()
{
    Tensor<float> A{5, 5};
    for (int i = 0; i < A.h; i++) {
        for (int j = 0; j < A.w; j++) {
            Index(A, i, j) = i*A.w+j;
        }
    }
    auto B = A.slice(1,3,1,4);
    assert(Index(B, 0, 0) == 6);
    assert(Index(B, 0, 1) == 7);
    assert(Index(B, 0, 2) == 8);
    assert(Index(B, 1, 0) == 11);
    assert(Index(B, 1, 1) == 12);
    assert(Index(B, 1, 2) == 13);

    auto C = B.transpose();
    assert(Index(C, 0, 0) == 6);
    assert(Index(C, 0, 1) == 11);
    assert(Index(C, 1, 0) == 7);
    assert(Index(C, 1, 1) == 12);
    assert(Index(C, 2, 0) == 8);
    assert(Index(C, 2, 1) == 13);

    auto D = C.slice(1,2,1,2);
    assert(Index(D, 0, 0) == 12);
    std::cout << "slice passed..." << std::endl;
}

int main(int argc, char *argv[])
{
    bool test_gpu = true;
    int test_m = 335, test_n = 587, test_k= 699;

    for (;;)
    {
        switch (getopt(argc, argv, "s:ch:l:b:e:"))
        {
        case 's':
            randgen_seed = atoll(optarg);
            continue;
        case 'c': //cpu testing only
            test_gpu = false;
            continue;
        case 'm':
            test_m = atoi(optarg);
            continue;
        case 'n':
            test_n = atoi(optarg);
            continue;
        case 'k':
            test_k = atoi(optarg);
            continue;
        case -1:
            break;
        }
        break;
    }
    test_views();
    test_elemwise(test_m, test_n, test_gpu);
    test_matmul(test_m, test_n, test_k, test_gpu);
    test_reduction(test_m, test_n, test_gpu);
    test_op_cross_entropy_loss(test_gpu);
    std::cout << "All tests completed successfully!" << std::endl;
    return 0;
}
