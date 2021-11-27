#include "gpu_func.h"
#include <iostream>
#include <fstream>
extern char matrixName[];

void matrix_to_three_vector(int &M, int &N, int &nnz, mX_matrix_utils::distributed_sparse_matrix* A, int_vector_device& csrRowPtr, int_vector_device& csrColIdx, double_vector_device& csrVal)
{
    M = A->end_row - A->start_row + 1;
    nnz = A->local_nnz;

    int_vector_host csrRowPtr_host, csrColIdx_host;
    double_vector_host csrVal_host;
    csrRowPtr_host.push_back(0);
    int sum_nnz = 0;
    for (int j=A->start_row, cnt=0; j<=A->end_row; ++j, ++cnt)
    {
        mX_matrix_utils::distributed_sparse_matrix_entry* curr = A->row_headers[cnt];
        while (curr)
        {            
            csrColIdx_host.push_back(curr->column);
            csrVal_host.push_back(curr->value);
            sum_nnz++;
            curr = curr->next_in_row;
        }
        csrRowPtr_host.push_back(sum_nnz);
    }
    csrRowPtr = csrRowPtr_host;
    csrColIdx = csrColIdx_host;
    csrVal = csrVal_host;

    // write data to mtx file
    std::ofstream out;
    strcat(matrixName, ".mtx");
    out.open(matrixName, std::ios::out);
    out << "%%MatrixMarket matrix coordinate real general" << std::endl;
    out << M << " " << M << " " << nnz << std::endl;

    for(int i = 0; i < M; i++)
        for (int j = csrRowPtr_host[i]; j < csrRowPtr_host[i+1]; j++)
        {
            int colIdx = csrColIdx_host[j];
            out << i+1 << " " << colIdx+1 << " " << csrVal_host[j] << std::endl;
        }
    out.close();
}


void my_gmres(int M, int N, int nnz, int_vector_device &csrRowPtr, int_vector_device &csrColIdx, 
                            double_vector_device &csrVal, double_vector_host &b, double_vector_host &x0, 
                            double &tol, double &err, int k, double_vector_host &x, int &iters, int &restarts)
{
	// here's the star of the show, the guest of honor, none other than Mr.GMRES
	
	// first Mr.GMRES will compute the error in the initial guess
		// if it's already smaller than tol, he calls it a day
		// otherwise he settles down to work in mysterious ways his wonders to perform

	int start_row = 0;
	int end_row = M-1;

	x = x0;

	std::vector<double> temp1(M);
    double_vector_device temp1_dev(M);
    double_vector_device x_dev = x;
	my_SpMV(M, N, nnz, csrRowPtr,csrColIdx, csrVal, x_dev, temp1_dev);
    thrust::copy(temp1_dev.begin(), temp1_dev.end(), temp1.begin());

	for (int i = 0; i < temp1.size(); i++)
	{
		temp1[i] -= b[i];
	}

	err = mX_matrix_utils::norm(temp1);
  	restarts = -1;
	iters = 0;

	while (err > tol)
	{
		// at the start of every re-start
			// the initial guess is already stored in x

		restarts++;
		
		std::vector<double> temp1(M);
		std::vector< std::vector<double> > V;
		// sparse_matrix_vector_product(A,x,temp1);
        double_vector_device temp1_dev(M);
        double_vector_device x_dev(x.begin(), x.end());
        my_SpMV(M, N, nnz, csrRowPtr,csrColIdx, csrVal, x_dev, temp1_dev);
        thrust::copy(temp1_dev.begin(), temp1_dev.end(), temp1.begin());

		for (int i = start_row; i <= end_row; i++)
		{
			temp1[i-start_row] -= b[i-start_row];
			temp1[i-start_row] *= (double)(-1);

			std::vector<double> temp2;
			temp2.push_back(temp1[i-start_row]);
			V.push_back(temp2);
		}

		double beta = mX_matrix_utils::norm(temp1);

		for (int i = start_row; i <= end_row; i++)
		{
			V[i-start_row][0] /= beta;
		}

		err = beta;
		iters = 0;

		std::vector<double> cosines;
		std::vector<double> sines;
		std::vector<double> g;
		std::vector< std::vector<double> > R;
		
		g.push_back(beta);

		// ok, Mr.GMRES has determined the initial values for
			// V,R,g,sines,cosines,err and iters
			// he updates these at every iteration until
				// either err becomes less than tol
				// or a new restart is required

		// note that Mr.GMRES does not think it necessary to store the upper Hessenberg matrix at each iteration
			// he computes R at each iteration from the new Hessenberg matrix column
			// and he's clever enough to determine err without having to solve for x at each iteration

		while ((err > tol) && (iters < k))
		{
			iters++;

			// Mr.GMRES is now going to update the V matrix
				// for which he will require a matrix vector multiplication

			std::vector<double> temp1;
			std::vector<double> temp2(M);

			for (int i = start_row; i <= end_row; i++)
			{
				temp1.push_back(V[i-start_row][iters-1]);	
			}
            double_vector_device temp1_dev(temp1.begin(), temp1.end());
            double_vector_device temp2_dev(M);
            my_SpMV(M, N, nnz, csrRowPtr,csrColIdx, csrVal, temp1_dev, temp2_dev);
            thrust::copy(temp2_dev.begin(), temp2_dev.end(), temp2.begin());


			// Right, Mr.GMRES now has the matrix vector product
				// now he will orthogonalize this vector with the previous ones 
					// with some help from Messrs Gram and Schmidt
			
			std::vector<double> new_col_H;

			for (int i = 0; i < iters; i++)
			{
				double local_dot = 0.0;
				double global_dot;

				for (int j = start_row; j <= end_row; j++)
				{
					local_dot += temp2[j-start_row]*V[j-start_row][i];
				}
#ifdef HAVE_MPI
				MPI_Allreduce(&local_dot,&global_dot,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
#else
                                global_dot = local_dot;
#endif				
				for (int j = start_row; j <= end_row; j++)
				{
					temp2[j-start_row] -= global_dot*V[j-start_row][i];
				}

				new_col_H.push_back(global_dot);
			}

			new_col_H.push_back(mX_matrix_utils::norm(temp2));

			for (int i = start_row; i <= end_row; i++)
			{
				temp2[i-start_row] /= new_col_H.back();
				V[i-start_row].push_back(temp2[i-start_row]);
			}

			// Right, Mr.GMRES has successfully updated V
				// on the side, he has also been computing the new column of the Hessenberg matrix
			// now he's going to get the new column of R using the current sines and cosines
				// and he will also add a new sine and a new cosine for future use

			for (int i = 0; i < iters-1; i++)
			{
				double old_i = new_col_H[i];
				double old_i_plus_one = new_col_H[i+1];

				new_col_H[i] = cosines[i]*old_i + sines[i]*old_i_plus_one;
				new_col_H[i+1] = -sines[i]*old_i + cosines[i]*old_i_plus_one;
			}

			double r = std::sqrt(new_col_H[iters-1]*new_col_H[iters-1] + new_col_H[iters]*new_col_H[iters]);
			cosines.push_back(new_col_H[iters-1]/r);
			sines.push_back(new_col_H[iters]/r);
			
			double old_i = new_col_H[iters-1];
			double old_i_plus_one = new_col_H[iters];

			new_col_H[iters-1] = cosines.back()*old_i + sines.back()*old_i_plus_one;
			new_col_H.pop_back();

			R.push_back(new_col_H);

			// Right, the new column of R is ready
			// the only thing left to do is to update g
				// which will also tell Mr.GMRES what the new error is

			double old_g = g[iters-1];
			g[iters-1] = old_g*cosines.back();
			g.push_back(-old_g*sines.back());

			err = std::abs(g.back());
		}

		// ok, so either Mr.GMRES has a solution
			// or he's being forced to restart
		// either way, he needs to compute x
			// now he needs to solve Ry = g
			// after which he will say x += (V without its last column)*y

		std::vector<double> y;

		for (int i = iters-1; i >= 0; i--)
		{
			double sum = (double)(0);

			for (int j = iters-1; j > i; j--)
			{
				sum += R[j][i]*y[iters-1-j];
			}

			y.push_back((g[i] - sum)/R[i][i]);
		}

		// ok, so y is ready (although it's stored upside down)

		for (int i = start_row; i <= end_row; i++)
		{
			double sum = (double)(0);

			for (int j = iters-1; j >= 0; j--)
			{
				sum += y[iters-1-j]*V[i-start_row][j];
			}

			x[i-start_row] += sum;
		}

		// the new x is also ready
			// either return it or use it as an initial guess for the next restart
	}
	
	// if Mr.GMRES ever reaches here, it means he's solved the problem

	if (restarts < 0)
	{
		restarts = 0;
	}
}


void my_SpMV(int M, int N, int nnz, int_vector_device& csrRowPtr, int_vector_device& csrColIdx, double_vector_device& csrVal, double_vector_device& x, double_vector_device& y)
{
    // int TPV = nnz/M;
    int TPV = 32;
	CSR_coop_spmv_prepare0(M, TPV, csrRowPtr, csrColIdx, csrVal, x, y);
}

template <  unsigned int THREADS_PER_VECTOR,
            unsigned int VECTORS_PER_BLOCK>
__global__ void CSR_coop_spmv_kernel(
    const int rows,
    const int* Ap,
    const int* Aj,
    const double* Ax,
    const double* x,
    double* y)
{
    __shared__ volatile double sdata[VECTORS_PER_BLOCK * THREADS_PER_VECTOR + THREADS_PER_VECTOR/2];  // padded to avoid reduction conditionals
    __shared__ volatile int ptrs[VECTORS_PER_BLOCK][2];

    const int thread_id   = blockDim.x * blockIdx.x + threadIdx.x;    // global thread index
    const int thread_lane = threadIdx.x & (THREADS_PER_VECTOR - 1);          // thread index within the vector
    const int vector_id   = thread_id   /  THREADS_PER_VECTOR;               // global vector index
    const int vector_lane = threadIdx.x /  THREADS_PER_VECTOR;               // vector index within the block
    const int num_vectors = VECTORS_PER_BLOCK * gridDim.x;                   // total number of active vectors
 
    for(int row = vector_id; row < rows; row += num_vectors)
    {
        // use two threads to fetch Ap[row] and Ap[row+1]
        // this is considerably faster than the straightforward version
            if(thread_lane < 2)
                ptrs[vector_lane][thread_lane] = Ap[row + thread_lane];
        
		__syncwarp();
        const int row_start    = ptrs[vector_lane][0];                   //same as: row_start = Ap[row];
        const int row_end      = ptrs[vector_lane][1];                   //same as: row_end   = Ap[row+1];

        double sum = 0.0;

        if (THREADS_PER_VECTOR == 32 && row_end - row_start > 32)
        {
           // ensure aligned memory access to Aj and Ax
           int jj = row_start - (row_start & (THREADS_PER_VECTOR - 1)) + thread_lane;

           // accumulate local sums
            if(jj >= row_start && jj < row_end)
                sum += Ax[jj] * x[Aj[jj]];

            // accumulate local sums
            for(jj += THREADS_PER_VECTOR; jj < row_end; jj += THREADS_PER_VECTOR) 
                sum += Ax[jj] * x[Aj[jj]];
        }
        else
        {
            // accumulate local sums
            for(int jj = row_start + thread_lane; jj < row_end; jj += THREADS_PER_VECTOR) 
                sum += Ax[jj] * x[Aj[jj]];
        }

        sdata[threadIdx.x] = sum;
        double temp;

        for (int stride = THREADS_PER_VECTOR/2; stride>0; stride/=2)
        {
			__syncwarp();
            temp = sdata[threadIdx.x + stride];
            sdata[threadIdx.x] = sum = sum + temp;
        }
		__syncwarp();
        if (thread_lane == 0)  
        {
            y[row] = sdata[threadIdx.x];
        }
    }
}

template <unsigned int THREADS_PER_VECTOR>
void CSR_coop_spmv_prepare1(int M, int_vector_device RowPtr, int_vector_device ColIdx, double_vector_device& Val, double_vector_device& x, double_vector_device& y)
{
    const int THREADS_PER_BLOCK = 256;
    const int VECTORS_PER_BLOCK = THREADS_PER_BLOCK / THREADS_PER_VECTOR;
	const int MAX_BLOCKS = cusp::system::cuda::detail::max_active_blocks(CSR_coop_spmv_kernel<THREADS_PER_VECTOR, VECTORS_PER_BLOCK>, THREADS_PER_BLOCK, 0);
	// const int MAX_BLOCKS = cusp::system::cuda::detail::max_active_blocks(spmv_vector_sync<THREADS_PER_VECTOR, VECTORS_PER_BLOCK>, THREADS_PER_BLOCK, 0);
	const int REQUIRED_BLOCKS = (M + VECTORS_PER_BLOCK -1)/VECTORS_PER_BLOCK;
    // const int MAX_BLOCKS = REQUIRED_BLOCKS;
	const int NUM_BLOCKS = std::min<int>(MAX_BLOCKS, REQUIRED_BLOCKS);
    // CSR_coop_spmv_kernel<THREADS_PER_VECTOR, VECTORS_PER_BLOCK><<<NUM_BLOCKS, THREADS_PER_BLOCK, 0, 0>>>(
    //         M, thrust::raw_pointer_cast(&RowPtr[0]), thrust::raw_pointer_cast(&ColIdx[0]), thrust::raw_pointer_cast(&Val[0]), 
    //         thrust::raw_pointer_cast(&x[0]), thrust::raw_pointer_cast(&y[0]));
    CSR_coop_spmv_kernel<THREADS_PER_VECTOR, VECTORS_PER_BLOCK><<<NUM_BLOCKS, THREADS_PER_BLOCK, 0, 0>>>(
        M, thrust::raw_pointer_cast(RowPtr.data()), thrust::raw_pointer_cast(ColIdx.data()), thrust::raw_pointer_cast(Val.data()), 
        thrust::raw_pointer_cast(x.data()), thrust::raw_pointer_cast(y.data()));

    // spmv_vector_sync<THREADS_PER_VECTOR, VECTORS_PER_BLOCK><<<NUM_BLOCKS, THREADS_PER_BLOCK, 0, 0>>>(
    //         M, RowPtr, ColIdx, Val, x, y);
}

void CSR_coop_spmv_prepare0(int M, int TPV, int_vector_device &RowPtr, int_vector_device &ColIdx, double_vector_device& Val, double_vector_device& x,double_vector_device& y)
{
    if (TPV <= 2)
    {
        CSR_coop_spmv_prepare1<2>(M, RowPtr, ColIdx, Val, x, y);
        return;
    }
    else if (TPV <= 4)
    {
        CSR_coop_spmv_prepare1<4>(M, RowPtr, ColIdx, Val, x, y);
        return;
    }
    else if (TPV <= 8)
    {
        CSR_coop_spmv_prepare1<8>(M, RowPtr, ColIdx, Val, x, y);
        return;
    }
    else if (TPV <= 16)
    {
        CSR_coop_spmv_prepare1<16>(M, RowPtr, ColIdx, Val, x, y);
        return;
    }

    CSR_coop_spmv_prepare1<32>(M, RowPtr, ColIdx, Val, x, y);
}