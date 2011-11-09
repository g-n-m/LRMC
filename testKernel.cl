// MMT is calculating C = A*A',
// where A is a width x height type matrix
// NOTE: localisation

// __kernel 
// void MMTold(__global float* a, 
// 	 __global float* c,
// 	 const unsigned int width,
// 	 const unsigned int height
// 	)
// {
//     uint i = get_global_id(0);
//     uint j = get_global_id(1);
// 
//     float sum = 0;
// 
//     for(int k=0; k<width; k++){
// 
//       sum+=a[j*width+k]*a[i*width+k];
//     }
// 
//     c[j*width+i]=sum;
// 
// //     CODE for M.Transpose
// //     c[i*height+j]=a[j*width+i];
// }

// MMT is calculating C = A*A',
// where A is a " width x height " type matrix
// Blocked version of MMT
// TODO: refresh the role of for cycles
__kernel 
void MMT(__global float* a, 
	 __global float* c,
	 const unsigned int width,
	 const unsigned int height,
	 __local  float* sdata,
	 __local  float* sdatb
	)
{
    uint li = get_local_id(0);
    uint lj = get_local_id(1);

    __private uint lsi = get_local_size(0);
    __private uint lsj = get_local_size(1);

    uint col = get_group_id(0) * lsi + li;
    uint row = get_group_id(1) * lsj + lj;

    float sum = 0;

    for(int m=0; m<width/lsi; m++){
	sdata[lj*lsi+li] = a[row * width + (m * lsi + li)];
// 	sdatb[lj*lsi+li] = a[(m * lsj + lj) * width + col]; //without transpose it's just a normal mult.
	sdatb[lj*lsi+li] = a[col * width + (m * lsj + lj)]; //copy of transposed elements
	barrier(CLK_LOCAL_MEM_FENCE);
	
	for(int k=0; k<lsi; k++){
	    sum += sdatb[k * lsi + li] * sdata[lj * lsi + k];
// 	    barrier(CLK_LOCAL_MEM_FENCE); //needed ??
	}
    }
    c[row*width+col]=sum;
    //     c[row*width+col]=sdata[lj*lsi+li];
}

// PNorm2 is calculating c = ||a||,
// where a is a vector
// NOTE: size-padding needed!
__kernel 
void PNorm2G(__global float* a, 
	    __global float* c,
	    __local  float* sdata
	   )
{
    uint li = get_local_id(0);
    uint bi = get_group_id(0);
    __private uint ls = get_local_size(0);
    uint j  = bi*ls*2+li;

    sdata[li]=a[j]*a[j]+a[j+ls]*a[j+ls];
    barrier(CLK_LOCAL_MEM_FENCE);

    if (ls >= 512) {if (li < 256) { sdata[li] += sdata[li + 256];} barrier(CLK_LOCAL_MEM_FENCE); }
    if (ls >= 256) {if (li < 128) { sdata[li] += sdata[li + 128];} barrier(CLK_LOCAL_MEM_FENCE); }
    if (ls >= 128) {if (li <  64) { sdata[li] += sdata[li +  64];} barrier(CLK_LOCAL_MEM_FENCE); }

    if (li < 32) {
      if (ls >= 64) { sdata[li] += sdata[li +  32]; barrier(CLK_LOCAL_MEM_FENCE); }
      if (ls >= 32) { sdata[li] += sdata[li +  16]; barrier(CLK_LOCAL_MEM_FENCE); }
      if (ls >= 16) { sdata[li] += sdata[li +   8]; barrier(CLK_LOCAL_MEM_FENCE); }
      if (ls >=  8) { sdata[li] += sdata[li +   4]; barrier(CLK_LOCAL_MEM_FENCE); }
      if (ls >=  4) { sdata[li] += sdata[li +   2]; barrier(CLK_LOCAL_MEM_FENCE); }
      if (ls >=  2) { sdata[li] += sdata[li +   1]; barrier(CLK_LOCAL_MEM_FENCE); }
    }

    if ( li==0 ) {c[0] = sqrt(sdata[0]);}
}

// PNorm2 with local output
// __kernel 
void PNorm2L(__global float* a, 
	    __local  float* c,
	    __local  float* sdata
	   )
{
    uint li = get_local_id(0);
    uint bi = get_group_id(0);
    uint ls = get_local_size(0);
    uint j  = bi*ls*2+li;

    sdata[li]=a[j]*a[j]+a[j+ls]*a[j+ls];
    barrier(CLK_LOCAL_MEM_FENCE);

    if (ls >= 512) {if (li < 256) { sdata[li] += sdata[li + 256];} }
    if (ls >= 256) {if (li < 128) { sdata[li] += sdata[li + 128];} }
    if (ls >= 128) {if (li <  64) { sdata[li] += sdata[li +  64];} }

    if (li < 32) {
      if (ls >= 64) { sdata[li] += sdata[li +  32]; }
      if (ls >= 32) { sdata[li] += sdata[li +  16]; }
      if (ls >= 16) { sdata[li] += sdata[li +   8]; }
      if (ls >=  8) { sdata[li] += sdata[li +   4]; }
      if (ls >=  4) { sdata[li] += sdata[li +   2]; }
      if (ls >=  2) { sdata[li] += sdata[li +   1]; }
    }

    if ( li==0 ) {c[0] = sqrt(sdata[0]);}
}

// PNorm2v2 is calculating c = a/||a||,
// where a is a vector
// NOTE: size-padding

__kernel 
void PNorm2v2(__global float* a, 
	      __global float* c,
// 	      const unsigned int ei,
// 	      const float sigma,
	      __local  float* sdata,
	      __local  float* i
	     )
{
    PNorm2L(a, i ,sdata);
    barrier(CLK_LOCAL_MEM_FENCE); //-!
    
    uint gi = get_global_id(0);
    uint li = get_local_id(0);
    uint bi = get_group_id(0);
    uint ls = get_local_size(0);
    uint j  = bi*ls*2+li;

    c[gi]   = a[gi]/i[0];
    c[gi+ls]= a[gi+ls]/i[0];
}

// SMV is calculating <result> = s*M*v,
// where s is scalar, M is a matrix (different sizes?) and v is a vector (size?)
__kernel
void SMV(__global float* s,
	 __global float* m,
	 __global float* v,
	 __global float* c,
	 const unsigned int width, // this should be for both the vector parts and the matrix parts??
	 __local float* sdata,
	 __local float* sdatb
	 )
{
    uint li = get_local_id(0);
    uint lj = get_local_id(1);

    __private uint lsi = get_local_size(0);  // is private really necessery?
    __private uint lsj = get_local_size(1);  // NOTE: care accentuated letters!

    uint col = get_group_id(0) * lsi + li;
    uint row = get_group_id(1) * lsj + lj;

    float sum = 0;

    for(int i=0; i < width / lsi; i++) {
      sdata[lj] = v[i * lsi + li];
      sdatb[lj * lsi + li] = m[(i * lsj + lj) * width + col];
      
      barrier(CLK_LOCAL_MEM_FENCE);

      for(int k=0; k < lsi; k++) {
	sum += sdatb[k * lsi + li] * sdata[k]; // k was assumed
      }
    }
    c[col]=sum;
}


// __kernel 
// void matrixTranspose(__global float * output,
//                      __global float * input,
//                      __local  float * block,
//                      const    uint    width,
//                      const    uint    height,
//                      const    uint blockSize
//                        )

//     uint localIdx = get_local_id(0);
//     uint localIdy = get_local_id(1);
//         
//     /* copy from input to local memory */
//     block[localIdy*blockSize + localIdx] = input[globalIdy*width + globalIdx];
// 
//     /* wait until the whole block is filled */
//     barrier(CLK_LOCAL_MEM_FENCE);
// 
//     uint groupIdx = get_group_id(0);
//     uint groupIdy = get_group_id(1);
// 
//     /* calculate the corresponding target location for transpose  by inverting x and y values*/
//     uint targetGlobalIdx = groupIdy*blockSize + localIdy;
//     uint targetGlobalIdy = groupIdx*blockSize + localIdx;
// 
//     /* calculate the corresponding raster indices of source and target */
//     uint targetIndex  = targetGlobalIdy*height     + targetGlobalIdx;
//     uint sourceIndex  = localIdy       * blockSize + localIdx;
//         
//      output[targetIndex] = block[sourceIndex];
